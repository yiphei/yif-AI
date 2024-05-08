import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.transformer_modules import (BaseModel, FeedForward, LayerNorm,
                                       MultiAttentionHead)


class MaskLossType(str, Enum):
    MSE = "MSE"
    COSINE_SIM = "COSINE_SIM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return MaskLossType.MSE
        elif num == 2:
            return MaskLossType.COSINE_SIM
        else:
            raise ValueError("Invalid mask loss number")


@dataclass
class LearnedDropoutConfig:
    use_bias: bool
    start_layer: int
    future_dim: int
    mask_loss_type: Union[MaskLossType, int]
    use_mask_loss: bool = True
    end_layer: Optional[int] = None
    mask_loss_coeff: Optional[float] = None
    n_heads: int = 1
    profile_dropout_mask: bool = False

    def __post_init__(self):
        if self.mask_loss_coeff is not None:
            assert self.mask_loss_coeff > 0

        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        if type(self.mask_loss_type) == int:
            self.mask_loss_type = MaskLossType.get_type_from_int(self.mask_loss_type)

        assert self.n_heads >= 1

        if not self.use_mask_loss and self.mask_loss_coeff is not None:
            raise ValueError("mask_loss_coeff must be None if use_mask_loss is False")


@dataclass
class ModelConfig:
    context_size: int
    n_embed: int
    n_layer: int
    n_head: int
    use_learned_dropout: bool
    learned_dropout_config: LearnedDropoutConfig = None
    dropout_rate: Optional[float] = field(default=None)
    alphabet_size: Optional[int] = field(default=None)
    bias: bool = False
    profile_layer_x: int = None
    profile_attn: bool = False

    def __post_init__(self):
        if not (self.use_learned_dropout == (self.learned_dropout_config is not None)):
            raise ValueError(
                "use_learned_dropout and learned_dropout_config are mutually inclusive"
            )

        elif not self.use_learned_dropout and self.dropout_rate is None:
            raise ValueError("dropout_rate must be set if not use_learned_dropout")

        if (
            self.learned_dropout_config is not None
            and type(self.learned_dropout_config) == dict
        ):
            self.learned_dropout_config = LearnedDropoutConfig(
                **self.learned_dropout_config
            )

        if self.learned_dropout_config:
            if (
                self.learned_dropout_config.start_layer > self.n_layer
                or self.learned_dropout_config.start_layer < 1
            ):
                raise ValueError("start_layer <= n_layer and >= 1")
            if (
                self.learned_dropout_config.end_layer > self.n_layer
                or self.learned_dropout_config.end_layer < 1
            ):
                raise ValueError("end_layer <= n_layer and >= 1")

        if (
            self.use_learned_dropout
            and self.learned_dropout_config.profile_dropout_mask
            and self.profile_layer_x is not None
        ):
            raise ValueError(
                "profile_layer_x cannot be set if profile_dropout_mask is True"
            )

        if self.use_learned_dropout:
            assert (
                1 <= self.learned_dropout_config.future_dim <= (self.context_size - 1)
            )

class LearnedDropout(nn.Module):
    def __init__(self, embed_dim, context_size, config, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size

        self.config = config
        self.module_name = None  # used for logging

        self.head_size = embed_dim // config.n_heads
        self.batch_attn_weights = nn.Linear(
            embed_dim, embed_dim * 3, bias=config.use_bias
        )
        self.future_k_weights = nn.Parameter(
            torch.randn(config.n_heads, self.head_size, context_size - 1)
        )
        self.future_v_weights = nn.Parameter(
            torch.randn(config.n_heads, context_size - 1, self.head_size)
        )
        torch.nn.init.normal_(self.future_k_weights, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.future_v_weights, mean=0.0, std=0.02)
        self.residual_proj = nn.Linear(embed_dim, embed_dim, bias=config.use_bias)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.register_buffer(
            "tril",
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
                    diagonal=self.config.future_dim,
                )
            ).view(1, 1, context_size, context_size - 1),
        )
        self.register_buffer(
            "full_tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                ),
                diagonal=self.config.future_dim,
            ),
        )
        self.register_buffer("mask_loss", torch.tensor(0), persistent=False)

    def forward(self, x):
        import wandb

        # dropout_x = x.detach() if self.config.use_detached_x_in_dropout_mask else x
        dropout_x = x

        B, T, C = dropout_x.shape
        q, k, v = self.batch_attn_weights(dropout_x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        if self.training:
            with torch.no_grad():
                true_attn = attn.masked_fill(
                    self.full_tril[
                        :, :, :T, : min(T + self.config.future_dim, self.context_size)
                    ]
                    == 0,
                    float("-inf"),
                )
                true_attn = F.softmax(true_attn, dim=-1)
                true_future_attn = true_attn[:, :, :T, 1:]
                true_future_attn = true_future_attn.masked_fill(
                    self.future_tril[
                        :,
                        :,
                        :T,
                        : min(T + self.config.future_dim - 1, self.context_size - 1),
                    ]
                    != 0,
                    0.0,
                )
                true_future_mask = (
                    true_future_attn
                    @ v[:, :, 1 : min(T + self.config.future_dim, self.context_size), :]
                )

        causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, 0.0)
        pad_size = min(self.config.future_dim, self.context_size - T)
        if pad_size > 0:
            padded_causal_attn = F.pad(causal_attn, (0, pad_size), "constant", 0)
        else:
            padded_causal_attn = causal_attn

        future_attn = (
            q
            @ self.future_k_weights[
                :, :, : min(T + self.config.future_dim - 1, self.context_size - 1)
            ]
        ) * (self.head_size**-0.5)
        future_attn = future_attn.masked_fill(
            self.future_tril[
                :, :, :T, : min(T + self.config.future_dim - 1, self.context_size - 1)
            ]
            != 0,
            0.0,
        )
        padded_future_attn = F.pad(future_attn, (1, 0), "constant", 0)

        full_attn = padded_causal_attn + padded_future_attn
        full_attn = full_attn.masked_fill(
            self.full_tril[
                :, :, :T, : min(T + self.config.future_dim, self.context_size)
            ]
            == 0,
            float("-inf"),
        )
        softmax_full_attn = F.softmax(full_attn, dim=-1)
        softmax_full_attn = self.dropout_1(softmax_full_attn)

        softmax_causal_attn = softmax_full_attn[:, :, :T, :T]
        softmax_causal_attn = softmax_causal_attn.masked_fill(
            self.tril[:, :, :T, :T] == 0, 0.0
        )
        softmax_future_attn = softmax_full_attn[
            :, :, :T, 1 : min(T + self.config.future_dim, self.context_size)
        ]
        softmax_future_attn = softmax_future_attn.masked_fill(
            self.future_tril[
                :, :, :T, : min(T + self.config.future_dim - 1, self.context_size - 1)
            ]
            != 0,
            0.0,
        )

        causal_mask = softmax_causal_attn @ v
        future_mask = (
            softmax_future_attn
            @ self.future_v_weights[
                :, : min(T + self.config.future_dim - 1, self.context_size - 1), :
            ]
        )
        full_mask = causal_mask + future_mask
        dropout_mask = full_mask.transpose(1, 2).contiguous().view(B, T, C)

        new_x = self.residual_proj(dropout_mask)
        new_x = self.dropout_2(new_x)

        if self.training:
            if self.config.mask_loss_type == MaskLossType.MSE:
                self.mask_loss = F.mse_loss(future_mask, true_future_mask)
            elif self.config.mask_loss_type == MaskLossType.COSINE_SIM:
                self.mask_loss = (
                    F.cosine_similarity(future_mask, true_future_mask).mean() ** 2
                )

        # if (
        #     self.training
        #     and self.config.profile_dropout_mask
        #     and self.is_last_minibatch
        # ):
        #     # NB: because of gradient accumulation, this will only log the last batch

        #     with torch.no_grad():
        #         if self.config.softmax_dim == 2:
        #             causal_attn_dim_1_mean = causal_attn.mean(dim=-1)
        #             causal_attn_dim_1_mean[:, :, : T // 2] *= -1
        #             causal_attn_dim_1_mean_head_mean = causal_attn_dim_1_mean.mean(
        #                 dim=-2
        #             )
        #             causal_attn_dim_1_mean_head_std = causal_attn_dim_1_mean.std(dim=-2)
        #             causal_attn_dim_1_mean_head_std[:, : T // 2] *= -1
        #         elif self.config.softmax_dim == 1:
        #             causal_attn_dim_2_mean = causal_attn.mean(dim=-2)
        #             causal_attn_dim_2_mean[:, :, : T // 2] *= -1
        #             causal_attn_dim_2_mean_head_mean = causal_attn_dim_2_mean.mean(
        #                 dim=-2
        #             )
        #             causal_attn_dim_2_mean_head_std = causal_attn_dim_2_mean.std(dim=-2)
        #             causal_attn_dim_2_mean_head_std[:, : T // 2] *= -1

        #     log_x = x.detach()
        #     log_new_x = new_x.detach()
        #     log_dropout_mask = dropout_mask.detach()
        #     log_causal_attn = causal_attn.detach()
        #     log_attn = attn.detach()
        #     log_proj_mask = proj_mask.detach() if proj_mask is not None else None
        #     log_pre_new_x = pre_new_x.detach() if pre_new_x is not None else None

        #     if (
        #         dropout_mask.dtype == torch.bfloat16
        #         or causal_attn.dtype == torch.bfloat16
        #         or dropout_mask.dtype == torch.bfloat16
        #     ):
        #         log_x = log_x.half()
        #         log_new_x = log_new_x.half()
        #         log_dropout_mask = log_dropout_mask.half()
        #         log_causal_attn = log_causal_attn.half()
        #         log_attn = log_attn.half()
        #         log_proj_mask = (
        #             log_proj_mask.half() if log_proj_mask is not None else None
        #         )
        #         log_pre_new_x = (
        #             log_pre_new_x.half() if log_pre_new_x is not None else None
        #         )
        #         if self.config.softmax_dim == 1:
        #             causal_attn_dim_2_mean = causal_attn_dim_2_mean.detach().half()
        #             causal_attn_dim_2_mean_head_mean = (
        #                 causal_attn_dim_2_mean_head_mean.detach().half()
        #             )
        #             causal_attn_dim_2_mean_head_std = (
        #                 causal_attn_dim_2_mean_head_std.detach().half()
        #             )
        #         elif self.config.softmax_dim == 2:
        #             causal_attn_dim_1_mean = causal_attn_dim_1_mean.detach().half()
        #             causal_attn_dim_1_mean_head_mean = (
        #                 causal_attn_dim_1_mean_head_mean.detach().half()
        #             )
        #             causal_attn_dim_1_mean_head_std = (
        #                 causal_attn_dim_1_mean_head_std.detach().half()
        #             )

        #     metrics = {
        #         self.module_name + ".a__input_x": log_x,
        #         self.module_name + ".l__new_x": log_new_x,
        #         self.module_name + ".g__mask": log_dropout_mask,
        #         self.module_name + ".c__causal_attn": log_causal_attn,
        #         self.module_name + ".b__attn": log_attn,
        #         self.module_name + ".h__mask_dim_2_std": log_dropout_mask.std(dim=-2),
        #     }

        #     if log_proj_mask is not None:
        #         metrics[self.module_name + ".i__proj_mask"] = log_proj_mask
        #     if log_pre_new_x is not None:
        #         metrics[self.module_name + ".k__pre_new_x"] = log_pre_new_x

        #     if self.config.softmax_dim == 2:
        #         metrics = {
        #             **metrics,
        #             self.module_name
        #             + ".d__causal_attn_dim_1_mean": causal_attn_dim_1_mean,
        #             self.module_name
        #             + ".e__causal_attn_dim_1_mean_head_mean": causal_attn_dim_1_mean_head_mean,
        #             self.module_name
        #             + ".f__causal_attn_dim_1_mean_head_std": causal_attn_dim_1_mean_head_std,
        #         }
        #     elif self.config.softmax_dim == 1:
        #         metrics = {
        #             **metrics,
        #             self.module_name
        #             + ".d__causal_attn_dim_2_mean": causal_attn_dim_2_mean,
        #             self.module_name
        #             + ".e__causal_attn_dim_2_mean_head_mean": causal_attn_dim_2_mean_head_mean,
        #             self.module_name
        #             + ".f__causal_attn_dim_2_mean_head_std": causal_attn_dim_2_mean_head_std,
        #         }
        #     wandb.log(
        #         metrics,
        #         commit=False,
        #     )
        return new_x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        use_learned_dropout=False,
        should_profile_layer_x=False,
    ):
        super().__init__()
        self.use_learned_dropout = use_learned_dropout
        self.learned_dropout_config = config.learned_dropout_config
        if use_learned_dropout:
            self.multi_attn_head = LearnedDropout(
                config.n_embed,
                config.context_size,
                config.learned_dropout_config,
                config.dropout_rate,
            )
        else:
            self.multi_attn_head = OptimizedMultiAttentionHead(config)
        self.feed_forward = FeedForward(
            config, use_learned_dropout, should_profile_layer_x
        )
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.ln2 = LayerNorm(config.n_embed, config.bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class DropoutTransformer(nn.Module):
    model_config_cls = ModelConfig

    def __init__(self, config: ModelConfig, gradient_accumulation_steps):
        super().__init__()
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.training_step = (
            None  # this is provided by the context manager in the training script
        )
        self.is_last_minibatch = False

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_embed)
        if config.use_learned_dropout and False:
            self.dropout = LearnedDropout(config.n_embed, config.learned_dropout_config)
        else:
            self.dropout = nn.Dropout(config.dropout_rate)

        learned_config_start_layer = (
            config.learned_dropout_config.start_layer
            if config.use_learned_dropout
            else config.n_layer + 1
        )
        learned_config_end_layer = (
            config.learned_dropout_config.end_layer if config.use_learned_dropout else 0
        )

        profile_layer_x = config.profile_layer_x or 0

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config,
                    (i + 1) >= (learned_config_start_layer)
                    and (i + 1) <= (learned_config_end_layer),
                    i + 1 == profile_layer_x,
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.bias)
        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)

        self.token_embedding.weight = self.output_layer.weight  # weight tying
        self.apply(self._init_weights)

        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("residual_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # maybe there is a better way
        n_learned_dropout = 0
        param_to_param_name = {p: n for n, p in self.named_parameters()}
        for module in self.modules():
            if isinstance(module, LearnedDropout):
                module.module_name = ".".join(
                    param_to_param_name[module.batch_attn_weights.weight].split(".")[
                        :-2
                    ]
                )
                n_learned_dropout += 1
            elif isinstance(module, FeedForward):
                module.module_name = ".".join(
                    param_to_param_name[module.linear.weight].split(".")[:-2]
                )
            elif isinstance(module, OptimizedMultiAttentionHead):
                module.module_name = ".".join(
                    param_to_param_name[module.batch_attn_weights.weight].split(".")[
                        :-2
                    ]
                )

            module.is_last_minibatch = False
        self.n_learned_dropout = n_learned_dropout

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

            additional_loss = torch.tensor(0.0, device=device)
            mean_mask_losses = torch.tensor(0.0, device=device)
            if self.training and self.config.use_learned_dropout:
                mask_losses = torch.empty(self.n_learned_dropout, device=device)
                curr_idx = 0
                for module in self.modules():
                    if isinstance(module, LearnedDropout):
                        mask_losses[curr_idx] = module.mask_loss
                        curr_idx += 1

                coeff = self.config.learned_dropout_config.mask_loss_coeff or 1.0
                mean_mask_losses = mask_losses.mean() * coeff
                if self.config.learned_dropout_config.use_mask_loss:
                    additional_loss = mean_mask_losses

            loss = F.cross_entropy(logits, targets.view(-1)) + additional_loss
        return (logits, loss, mean_mask_losses)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None