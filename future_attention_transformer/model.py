import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.transformer_modules import (BaseModel, FeedForward, LayerNorm,
                                       MultiAttentionHead, SubModuleStats)


class FutureXLossType(str, Enum):
    MSE = "MSE"
    COSINE_SIM = "COSINE_SIM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return FutureXLossType.MSE
        elif num == 2:
            return FutureXLossType.COSINE_SIM
        else:
            raise ValueError("Invalid future x loss number")


@dataclass
class ModelConfig(BaseModelConfig):
    start_layer: int  # layer at which to start using future attention
    future_dim: int  # number of future tokens to attend to
    future_x_loss_type: Union[FutureXLossType, int]
    use_future_x_loss: bool = True
    end_layer: Optional[int] = None
    future_x_loss_coeff: Optional[float] = None

    def __post_init__(self):
        if type(self.future_x_loss_type) == int:
            self.future_x_loss_type = FutureXLossType.get_type_from_int(
                self.future_x_loss_type
            )

        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        if self.start_layer > self.n_layer or self.start_layer < 1:
            raise ValueError("start_layer must be <= n_layer and >= 1")
        if self.end_layer > self.n_layer or self.end_layer < 1:
            raise ValueError("end_layer must be <= n_layer and >= 1")

        assert 1 <= self.future_dim <= (self.context_size - 1)

        if not self.use_future_x_loss and self.future_x_loss_coeff is not None:
            raise ValueError(
                "future_x_loss_coeff must be None if use_future_x_loss is False"
            )

        if self.future_x_loss_coeff is not None:
            assert self.future_x_loss_coeff > 0
        elif self.use_future_x_loss:
            self.future_x_loss_coeff = 1.0


class DynamicLinear(nn.Module):
    def __init__(self, head_dim, in_dim, out_dim, use_bias):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.randn(head_dim, in_dim, out_dim))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = (
            nn.Parameter(torch.zeros(head_dim, 1, out_dim)) if use_bias else None
        )

    def forward(self, x, in_size=None, out_size=None):
        in_size = in_size or self.in_dim
        out_size = out_size or self.out_dim

        weight = self.weight[:, :in_size, :out_size]
        bias = self.bias[:, :, :out_size] if self.bias is not None else None

        x = x @ weight
        if bias is not None:
            x = x + bias
        return x


class FutureMultiAttentionHead(SubModuleStats):
    extra_stats = ["mask_loss"]

    def __init__(
        self,
        dim_in,
        n_head,
        use_bias,
        context_size,
        dropout_rate,
        future_dim,
        future_x_loss_type,
    ):
        super().__init__()
        assert dim_in % n_head == 0
        self.dim_in = dim_in
        self.context_size = context_size
        self.n_head = n_head
        self.head_size = dim_in // n_head
        self.future_dim = future_dim
        self.future_x_loss_type = future_x_loss_type

        self.batch_attn_weights = nn.Linear(dim_in, dim_in * 3, bias=use_bias)
        self.future_k_weights = DynamicLinear(
            n_head, self.head_size, context_size - 1, use_bias
        )
        self.future_v_weights = DynamicLinear(
            n_head, context_size - 1, self.head_size, use_bias
        )
        self.residual_proj = nn.Linear(dim_in, dim_in, bias=use_bias)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.register_buffer(
            "causal_tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                )
            ),
        )
        self.register_buffer(
            "future_tril",
            (
                torch.tril(torch.ones(context_size, context_size - 1), diagonal=-1)
                + torch.triu(
                    torch.ones(context_size, context_size - 1),
                    diagonal=future_dim,
                )
            ).view(1, 1, context_size, context_size - 1),
        )
        self.register_buffer(
            "full_tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                ),
                diagonal=future_dim,
            ),
        )
        self.register_buffer("mask_loss", torch.tensor(0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        T_w_future = min(T + self.future_dim, self.context_size)
        q, k, v = self.batch_attn_weights(x).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        if self.training:
            with torch.no_grad():
                true_attn = attn.masked_fill(
                    self.full_tril[:, :, :T, :T_w_future] == 0,
                    float("-inf"),
                )
                true_attn = F.softmax(true_attn, dim=-1)
                true_future_attn = true_attn[:, :, :T, 1:]
                true_future_attn = true_future_attn.masked_fill(
                    self.future_tril[
                        :,
                        :,
                        :T,
                        : T_w_future - 1,
                    ]
                    != 0,
                    0.0,
                )
                true_future_x = true_future_attn @ v[:, :, 1:T_w_future, :]

        causal_attn = attn.masked_fill(self.causal_tril[:, :, :T, :T] == 0, 0.0)
        pad_size = min(self.future_dim, self.context_size - T)
        if pad_size > 0:
            padded_causal_attn = F.pad(causal_attn, (0, pad_size), "constant", 0)
        else:
            padded_causal_attn = causal_attn

        future_attn = self.future_k_weights(q, out_size=T_w_future - 1) * (
            self.head_size**-0.5
        )
        future_attn = future_attn.masked_fill(
            self.future_tril[:, :, :T, : T_w_future - 1] != 0,
            0.0,
        )
        padded_future_attn = F.pad(future_attn, (1, 0), "constant", 0)

        full_attn = padded_causal_attn + padded_future_attn
        full_attn = full_attn.masked_fill(
            self.full_tril[:, :, :T, :T_w_future] == 0,
            float("-inf"),
        )
        softmax_full_attn = F.softmax(full_attn, dim=-1)
        softmax_full_attn = self.dropout_1(softmax_full_attn)

        softmax_causal_attn = softmax_full_attn[:, :, :T, :T]
        softmax_causal_attn = softmax_causal_attn.masked_fill(
            self.causal_tril[:, :, :T, :T] == 0, 0.0
        )
        softmax_future_attn = softmax_full_attn[:, :, :T, 1:T_w_future]
        softmax_future_attn = softmax_future_attn.masked_fill(
            self.future_tril[:, :, :T, : T_w_future - 1] != 0,
            0.0,
        )

        causal_x = softmax_causal_attn @ v
        future_x = self.future_v_weights(softmax_future_attn, in_size=T_w_future - 1)
        new_x = causal_x + future_x
        new_x = new_x.transpose(1, 2).contiguous().view(B, T, C)

        new_x = self.residual_proj(new_x)
        new_x = self.dropout_2(new_x)

        if self.training:
            if self.future_x_loss_type == FutureXLossType.MSE:
                self.mask_loss = F.mse_loss(future_x, true_future_x)
            elif self.future_x_loss_type == FutureXLossType.COSINE_SIM:
                self.mask_loss = (
                    F.cosine_similarity(future_x, true_future_x).mean() ** 2
                )
        return new_x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        use_future_attn=False,
    ):
        super().__init__()
        if use_future_attn:
            self.multi_attn_head = FutureMultiAttentionHead(
                config.n_embed,
                config.n_head,
                config.use_bias,
                config.context_size,
                config.dropout_rate,
                config.future_dim,
                config.future_x_loss_type,
            )
        else:
            self.multi_attn_head = MultiAttentionHead(
                config.n_embed,
                config.n_head,
                config.use_bias,
                config.context_size,
                config.dropout_rate,
                config.use_flash,
            )
        self.feed_forward = FeedForward(
            config.n_embed, config.use_bias, config.dropout_rate
        )
        self.ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class FutureAttentionTransformer(BaseModel):
    model_config_cls = ModelConfig

    def _init_model(self, config: ModelConfig):
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config,
                    (i + 1) >= config.start_layer and (i + 1) <= config.end_layer,
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.use_bias)
        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)

        self.token_embedding.weight = self.output_layer.weight  # weight tying
        self.apply(self._init_weights)

        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("residual_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def forward(self, x, targets=None):
        device = x.device
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(
            torch.arange(x.shape[1], dtype=torch.long, device=device)
        )
        embed = token_embed + pos_embed
        embed = self.dropout(embed)
        out = self.transformer_blocks(embed)
        out = self.ln(out)

        if targets is None:
            loss = None
            logits = self.output_layer(out[:, [-1], :])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            if self.training:
                n_learned_dropout = 1
                mask_losses = torch.empty(n_learned_dropout, device=device)
                curr_idx = 0
                for module in self.modules():
                    if isinstance(module, FutureMultiAttentionHead):
                        mask_losses[curr_idx] = module.mask_loss
                        curr_idx += 1

                self.mask_loss = mask_losses.mean() * self.config.future_x_loss_coeff

            loss = F.cross_entropy(logits, targets.view(-1))
            if self.config.use_future_x_loss:
                loss += self.mask_loss

        self._update_running_stats()
        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None
