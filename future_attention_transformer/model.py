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

        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        if self.start_layer > self.n_layer or self.start_layer < 1:
            raise ValueError("start_layer must be <= n_layer and >= 1")
        if self.end_layer > self.n_layer or self.end_layer < 1:
            raise ValueError("end_layer must be <= n_layer and >= 1")

        assert self.future_x_loss_coeff > 0

        if self.detach_future_x is None:
            assert not self.use_future_x_loss
        else:
            assert self.use_future_x_loss


class FutureMultiAttentionHead(SubModuleStats):
    extra_stats = ["future_loss"]

    def __init__(
        self,
        dim_in,
        n_head,
        use_bias,
        context_size,
        dropout_rate,
        future_x_loss_type,
        detach_future_x,
    ):
        super().__init__()
        assert dim_in % n_head == 0
        self.dim_in = dim_in
        self.context_size = context_size
        self.n_head = n_head
        self.head_size = dim_in // n_head
        self.future_x_loss_type = future_x_loss_type
        self.detach_future_x = detach_future_x or False

        self.batch_attn_weights = nn.Linear(dim_in, dim_in * 2, bias=use_bias)
        self.k_weights = nn.Linear(dim_in, dim_in, bias=use_bias)
        self.v_weights = nn.Linear(dim_in, dim_in, bias=use_bias)

        self.f_ln = LayerNorm(dim_in, use_bias)
        self.residual_proj = nn.Linear(dim_in, dim_in, bias=use_bias)

        self.dropout_rate = dropout_rate
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.register_buffer(
            "causal_tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                )
            ),
        )

    def forward(self, x):
        B, T, C = x.shape

        q, f = self.batch_attn_weights(x).split(self.dim_in, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        f = self.f_ln(f)

        k_pres = self.k_weights(x)
        k_future = self.k_weights(f)
        k_pres = k_pres.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k_future = k_future.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        v_pres = self.v_weights(x)
        v_future = self.v_weights(f)
        v_pres = v_pres.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v_future = v_future.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if self.training:
            attn = (q @ k_pres.transpose(-2, -1)) * (self.head_size**-0.5)
            true_attn = attn.masked_fill(self.causal_tril != 0, float("-inf"))
            true_attn = F.softmax(true_attn, dim=-1)
            # TODO: add dropout here for consistency, but its best not to use it
            true_future_attn = true_attn[:, :, :-1, :]
            true_future_x = true_future_attn @ v_pres
            if self.detach_future_x:
                true_future_x = true_future_x.detach()

        out_pres = F.scaled_dot_product_attention(
            q,
            k_pres,
            v_pres,
            attn_mask=None,
            dropout_p=self.dropout_rate,
            is_causal=True,
        )
        out_future = F.scaled_dot_product_attention(
            q,
            k_future,
            v_future,
            attn_mask=None,
            dropout_p=self.dropout_rate,
            is_causal=True,
        )
        out = out_pres + out_future
        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        out = self.residual_proj(out)
        out = self.dropout_2(out)

        if self.training:
            adapted_out_future = out_future[:, :, :-1, :]
            if self.future_x_loss_type == FutureXLossType.MSE:
                self.future_loss = F.mse_loss(adapted_out_future, true_future_x)
            elif self.future_x_loss_type == FutureXLossType.COSINE_SIM:
                cosine_sim = F.cosine_similarity(
                    adapted_out_future, true_future_x, dim=-1
                )
                self.future_loss = (1 - (1 + cosine_sim) / 2).mean()

        return out


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
