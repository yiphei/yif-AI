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
    start_layer: int = 1  # layer at which to start using future attention
    future_dim: int = None  # number of future tokens to attend to
    future_x_loss_type: Union[FutureXLossType, int] = FutureXLossType.COSINE_SIM
    use_future_x_loss: bool = True
    detach_future_x: Optional[bool] = False
    end_layer: Optional[int] = None
    future_x_loss_coeff: Optional[float] = 1.0

    def __post_init__(self):
        if type(self.future_x_loss_type) == int:
            self.future_x_loss_type = FutureXLossType.get_type_from_int(
                self.future_x_loss_type
            )

        if self.future_dim is None:
            self.future_dim = self.context_size - 1

        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        if self.start_layer > self.n_layer or self.start_layer < 1:
            raise ValueError("start_layer must be <= n_layer and >= 1")
        if self.end_layer > self.n_layer or self.end_layer < 1:
            raise ValueError("end_layer must be <= n_layer and >= 1")

        assert 1 <= self.future_dim <= (self.context_size - 1)

        assert self.future_x_loss_coeff > 0

        if self.detach_future_x is None:
            assert not self.use_future_x_loss
        else:
            assert self.use_future_x_loss


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

    def forward(self, x, max_in_size=None, max_out_size=None):
        max_in_size = max_in_size or self.in_dim
        max_out_size = max_out_size or self.out_dim

        weight = self.weight[:, :max_in_size, :max_out_size]
        bias = self.bias[:, :, :max_out_size] if self.bias is not None else None

        x = x @ weight
        if bias is not None:
            x = x + bias
        return x


class FutureMultiAttentionHead(SubModuleStats):
    extra_stats = ["future_loss"]

    def __init__(
        self,
        dim_in,
        n_head,
        use_bias,
        context_size,
        dropout_rate,
        future_dim,
        future_x_loss_type,
        detach_future_x,
    ):
        super().__init__()
        assert dim_in % n_head == 0
        self.dim_in = dim_in
        self.context_size = context_size
        self.n_head = n_head
        self.head_size = dim_in // n_head
        self.future_dim = future_dim
        self.future_x_loss_type = future_x_loss_type
        self.detach_future_x = detach_future_x or False

        self.batch_attn_weights = nn.Linear(dim_in, dim_in * 4, bias=use_bias)
        self.up_future_conv = nn.ConvTranspose1d(
            in_channels=self.head_size,
            out_channels=self.head_size,
            kernel_size=self.future_dim,
            stride=self.future_dim,
            bias=use_bias,
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

    @property
    def k_weights(self):
        return self.batch_attn_weights.weight[self.dim_in : self.dim_in * 2, :].T

    @property
    def v_weights(self):
        return self.batch_attn_weights.weight[self.dim_in * 2 : self.dim_in * 3, :].T

    def forward(self, x):
        B, T, C = x.shape
        T_w_future = min(T + self.future_dim, self.context_size)

        q, k, v, f = self.batch_attn_weights(x).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        f = f.view(B, T, self.n_head, self.head_size).transpose(1, 2)

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
                if self.detach_future_x:
                    true_future_x = true_future_x.detach()

        causal_attn = attn.masked_fill(self.causal_tril[:, :, :T, :T] == 0, 0.0)
        pad_size = self.future_dim
        if pad_size > 0:
            padded_causal_attn = F.pad(causal_attn, (0, pad_size), "constant", 0)
        else:
            padded_causal_attn = causal_attn

        adapted_f = f.transpose(2, 3).reshape(B * self.n_head, self.head_size, T)
        up_future = self.up_future_conv(adapted_f)
        up_future = up_future.reshape(
            B, self.n_head, self.head_size, T, self.future_dim
        )
        up_future = up_future.permute(0, 3, 4, 1, 2)
        up_future = up_future.reshape(
            B, T, self.future_dim, self.n_head * self.head_size
        )
        k_future = up_future @ self.k_weights
        k_future = k_future.view(B, T, self.future_dim, self.n_head, self.head_size)
        k_future = k_future.permute(0, 3, 1, 2, 4)

        future_attention = torch.einsum("bhts,bhtfs->bhtf", q, k_future)
        padding = torch.zeros(
            (B, self.n_head, T, self.future_dim + T), dtype=x.dtype, device=x.device
        )
        indices = torch.arange(self.future_dim, device=x.device).unsqueeze(0) + torch.arange(
            1, T + 1, device=x.device
        ).unsqueeze(1)
        indices = indices.unsqueeze(0).unsqueeze(0)
        padded_future_attn = padding.scatter_(1, indices, future_attention)

        full_attn = padded_causal_attn + padded_future_attn
        full_attn = full_attn.masked_fill(
            self.full_tril[:, :, :T, :] == 0,
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
        unpadded_future_attn = softmax_future_attn[:, indices]
        v_future = up_future @ self.v_weights
        v_future = v_future.view(B, T, self.future_dim, self.n_head, self.head_size)
        v_future = v_future.permute(0, 3, 1, 2, 4)
        future_x = torch.einsum("bhtf,bhtfs->bhts", unpadded_future_attn, v_future)

        causal_x = softmax_causal_attn @ v

        new_x = causal_x + future_x
        new_x = new_x.transpose(1, 2).contiguous().view(B, T, C)

        new_x = self.residual_proj(new_x)
        new_x = self.dropout_2(new_x)

        if self.training:
            if self.future_x_loss_type == FutureXLossType.MSE:
                self.future_loss = F.mse_loss(future_x, true_future_x)
            elif self.future_x_loss_type == FutureXLossType.COSINE_SIM:
                cosine_sim = F.cosine_similarity(future_x, true_future_x, dim=-1)
                self.future_loss = (1 - (1 + cosine_sim) / 2).mean()

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
                config.detach_future_x,
            )
        else:
            self.multi_attn_head = MultiAttentionHead(
                config.n_embed,
                config.n_head,
                config.use_bias,
                config.context_size,
                config.dropout_rate,
                True,
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
    extra_stats = ["scaled_future_loss"]

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
                self.aggregate_sub_module_stats()
                self.scaled_future_loss = (
                    self.config.future_x_loss_coeff * self.future_loss
                )

            loss = F.cross_entropy(logits, targets.view(-1))
            if self.training and self.config.use_future_x_loss:
                loss += self.scaled_future_loss

        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None
