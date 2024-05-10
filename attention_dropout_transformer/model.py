import math
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.transformer_modules import (BaseModel, LayerNorm,
                                       MultiAttentionHead, SubModuleStats)


@dataclass
class RegularizingLambdaConfig:
    min_lambda: float = None
    max_lambda: float = 1.0
    coefficient: float = None

    def __post_init__(self):
        assert self.max_lambda > 0

        if self.min_lambda is not None:
            self.min_lambda >= 0
            assert self.min_lambda < self.max_lambda
            assert self.coefficient is not None

        if self.coefficient is not None:
            assert self.coefficient < 1
            slope_1_step = np.log(1 / self.coefficient) * (1 / self.coefficient)
            print(f"STEP at which slope is 1: {slope_1_step}")


class RoundingType(str, Enum):
    SIGMOID = "SIGMOID"
    NOISE_AND_LINEAR = "NOISE_AND_LINEAR"
    LINEAR = "LINEAR"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return RoundingType.SIGMOID
        elif num == 2:
            return RoundingType.NOISE_AND_LINEAR
        elif num == 3:
            return RoundingType.LINEAR
        else:
            raise ValueError("Invalid rounding type number")


@dataclass
class AttentionDropoutConfig:
    use_bias: bool
    softmax_dim: int = 1
    rounding_type: Optional[Union[RoundingType, int]] = None
    sigmoid_slope: Optional[float] = None
    shift_init: float = torch.pi / 2
    n_heads: int = 1
    use_canonical_entropy: bool = False
    use_detached_x_in_dropout_mask: bool = False

    def __post_init__(self):
        assert 0 <= self.shift_init <= torch.pi
        assert self.softmax_dim in [0, 1]

        if type(self.rounding_type) == int:
            assert self.rounding_type in [1, 2, 3]
            self.rounding_type = RoundingType.get_type_from_int(self.rounding_type)

        if self.rounding_type != RoundingType.SIGMOID and self.sigmoid_slope:
            raise ValueError(
                "sigmoid_slope can only be set if rounding_type is SIGMOID"
            )

        if self.rounding_type == RoundingType.SIGMOID and not self.sigmoid_slope:
            self.sigmoid_slope = 60


@dataclass
class ModelConfig(BaseModelConfig):
    use_dropout_entropy_in_loss: bool
    use_dropout_l1_norm_in_loss: bool
    start_layer: int
    attention_dropout_config: AttentionDropoutConfig
    end_layer: Optional[int] = None
    dropout_entropy_lambda: Optional[RegularizingLambdaConfig] = None
    dropout_l1_norm_lambda: Optional[RegularizingLambdaConfig] = None

    def __post_init__(self):
        if type(self.attention_dropout_config) == dict:
            self.attention_dropout_config = AttentionDropoutConfig(
                **self.attention_dropout_config
            )
        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        if self.start_layer > self.n_layer or self.start_layer < 1:
            raise ValueError("start_layer <= n_layer and >= 1")
        if self.end_layer > self.n_layer or self.end_layer < 1:
            raise ValueError("end_layer <= n_layer and >= 1")

        if (
            self.use_dropout_entropy_in_loss
            and self.attention_dropout_config.rounding_type
            and self.attention_dropout_config.rounding_type in [2, 3]
        ):
            raise ValueError(
                "rounding_type cannot be 2 or 3 if use_dropout_entropy_in_loss"
            )

        if (
            not self.use_dropout_entropy_in_loss
            and self.dropout_entropy_lambda is not None
        ):
            raise ValueError(
                "dropout_entropy_lambda is set but use_dropout_entropy_in_loss is False"
            )

        if (
            not self.use_dropout_l1_norm_in_loss
            and self.dropout_l1_norm_lambda is not None
        ):
            raise ValueError(
                "dropout_l1_norm_lambda is set but use_dropout_l1_norm_in_loss is False"
            )

        for attr_name, flag_attr_name in [
            ("dropout_entropy_lambda", "use_dropout_entropy_in_loss"),
            ("dropout_l1_norm_lambda", "use_dropout_l1_norm_in_loss"),
        ]:
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                if type(attr_value) not in [dict, RegularizingLambdaConfig]:
                    raise ValueError(
                        f"{attr_name} must be a dict or RegularizingLambdaConfig"
                    )

                if type(attr_value) == dict:
                    setattr(self, attr_name, RegularizingLambdaConfig(**attr_value))
            else:
                if getattr(self, flag_attr_name):
                    setattr(self, attr_name, RegularizingLambdaConfig(max_lambda=1))


class AttentionDropout(SubModuleStats):
    extra_stats = [
        "dropout_entropy",
        "dropout_l1_norm",
        "dropout_near_one_percent",
        "dropout_near_zero_percent",
        "dropout_change_rate_from_prev",
        "active_dropout_percent",
    ]

    def __init__(
        self,
        embed_dim,
        context_size,
        config,
        use_dropout_entropy_in_loss,
        use_dropout_l1_norm_in_loss,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size

        self.config = config
        self.module_name = None  # used for logging

        self.head_size = embed_dim // config.n_heads
        self.batch_attn_weights = nn.Linear(
            embed_dim, embed_dim * 3, bias=config.use_bias
        )
        self.shift = nn.Parameter(
            torch.full((embed_dim,), config.shift_init, dtype=torch.float32)
        )
        self.uniform = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                )
            ),
        )
        self.register_buffer("prev_dropout_mask", torch.empty(0), persistent=False)
        self.dropout_entropy_context = (
            nullcontext() if use_dropout_entropy_in_loss else torch.no_grad()
        )
        self.dropout_l1_norm_context = (
            nullcontext() if use_dropout_l1_norm_in_loss else torch.no_grad()
        )
        self.entropy_fn = (
            self.canonical_entropy
            if config.use_canonical_entropy
            else self.alternate_entropy
        )

    def update_stats(self, dropout_mask):
        with self.dropout_entropy_context:
            self.dropout_entropy = self.entropy_fn(dropout_mask)
        with self.dropout_l1_norm_context:
            # TODO: change this to a simple sum
            self.dropout_l1_norm = torch.norm(dropout_mask, p=1) / dropout_mask.numel()

        with torch.no_grad():
            self.dropout_near_one_percent = (
                dropout_mask > 0.9
            ).sum() / dropout_mask.numel()
            self.dropout_near_zero_percent = (
                dropout_mask < 0.1
            ).sum() / dropout_mask.numel()
            self.active_dropout_percent = (
                dropout_mask > 0.05
            ).sum() / dropout_mask.numel()

            if self.prev_dropout_mask.nelement() != 0:
                matching_1s = (dropout_mask >= 0.5) & (self.prev_dropout_mask >= 0.5)
                matching_0s = (dropout_mask < 0.5) & (self.prev_dropout_mask < 0.5)
                self.dropout_change_rate_from_prev = (
                    1 - (matching_0s.sum() + matching_1s.sum()) / dropout_mask.numel()
                )
            self.prev_dropout_mask = dropout_mask.clone()

    def canonical_entropy(self, dropout_mask):
        # the small constant is for numerical stability
        return (dropout_mask * -torch.log2(dropout_mask + 1e-9)).mean()

    def alternate_entropy(self, dropout_mask):
        # the alternate entropy has the peak above 0.5, while the canonical one has
        # it below 0.5. In theory, this should be better for achieving both low entropy
        # and low l1 norm because there is more curvature towards 0.
        return ((dropout_mask - 1) * torch.log2((-dropout_mask + 1) + 1e-9)).mean()

    def forward(self, x):
        dropout_x = x.detach() if self.config.use_detached_x_in_dropout_mask else x

        B, T, C = dropout_x.shape
        q, k, v = self.batch_attn_weights(dropout_x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        if self.config.softmax_dim != 0:
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            causal_attn = F.softmax(causal_attn, dim=-self.config.softmax_dim)
        else:
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, 0)
        dropout_logits = causal_attn @ v
        dropout_logits = dropout_logits.transpose(1, 2).contiguous().view(B, T, C)
        dropout_mask = 0.5 * torch.cos(dropout_logits + self.shift) + 0.5

        if self.config.rounding_type:
            if self.config.rounding_type == RoundingType.SIGMOID:
                dropout_mask = torch.sigmoid(
                    self.config.sigmoid_slope * (dropout_mask - 0.5)
                )
            elif self.config.rounding_type == RoundingType.NOISE_AND_LINEAR:
                complement_mask = 1 - dropout_mask.detach()
                noise = self.uniform.sample(dropout_mask.shape).to(dropout_mask.device)
                scaling = torch.where(
                    noise >= complement_mask, complement_mask, complement_mask - 1
                )
                dropout_mask = dropout_mask.to(dtype=torch.float16) + scaling.to(
                    dtype=torch.float16
                )

            elif self.config.rounding_type == RoundingType.LINEAR:
                complement_mask = 1 - dropout_mask.detach()
                scaling = torch.where(
                    dropout_mask >= 0.5, complement_mask, complement_mask - 1
                )
                dropout_mask = dropout_mask.to(dtype=torch.float16) + scaling.to(
                    dtype=torch.float16
                )

        if self.training:
            self.update_stats(dropout_mask)

        new_x = x * dropout_mask
        return new_x


class FeedForward(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        use_attention_dropout=False,
    ):
        super().__init__()
        self.module_name = None
        self.config = config
        self.linear = nn.Linear(
            config.n_embed, config.n_embed * 4, bias=config.use_bias
        )
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(
            config.n_embed * 4, config.n_embed, bias=config.use_bias
        )
        if use_attention_dropout:
            self.dropout = AttentionDropout(
                config.n_embed,
                config.context_size,
                config.attention_dropout_config,
                config.use_dropout_entropy_in_loss,
                config.use_dropout_l1_norm_in_loss,
            )
        else:
            self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.residual_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        use_learned_dropout=False,
    ):
        super().__init__()
        self.multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            True,
        )
        self.feed_forward = FeedForward(config, use_learned_dropout)
        self.ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class AttentionDropoutTransformer(BaseModel):
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
                    (i + 1) >= (config.start_layer) and (i + 1) <= (config.end_layer),
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

        self.register_buffer(
            "dropout_entropy_coefficient", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "dropout_l1_norm_coefficient", torch.empty(0), persistent=False
        )
        self.need_new_coefficients = True

    def get_annealed_dropout_coefficient(self, lambda_config):
        if lambda_config is None:
            return torch.empty(0)

        if lambda_config.coefficient is None:
            return torch.tensor(lambda_config.max_lambda)

        assert self.training_step is not None
        intersect = (
            lambda_config.min_lambda - 1 if lambda_config.min_lambda is not None else -1
        )
        return torch.tensor(
            min(
                np.exp(lambda_config.coefficient * self.training_step) + intersect,
                lambda_config.max_lambda,
            )
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

            additional_loss = torch.tensor(0.0, device=device)
            if self.training:
                self.aggregate_sub_module_stats()
                if self.need_new_coefficients:
                    self.dropout_entropy_coefficient = (
                        self.get_annealed_dropout_coefficient(
                            self.config.dropout_entropy_lambda
                        )
                    )
                    self.dropout_l1_norm_coefficient = (
                        self.get_annealed_dropout_coefficient(
                            self.config.dropout_l1_norm_lambda
                        )
                    )
                    self.need_new_coefficients = True

                if self.config.use_dropout_entropy_in_loss:
                    additional_loss += (
                        self.dropout_entropy * self.dropout_entropy_coefficient
                    )
                if self.config.use_dropout_l1_norm_in_loss:
                    additional_loss += (
                        self.dropout_l1_norm * self.dropout_l1_norm_coefficient
                    )

            loss = F.cross_entropy(logits, targets.view(-1)) + additional_loss
        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params(True)
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embed // self.config.n_head,
            self.config.context_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_token += (
            (self.running_active_dropout_percent)
            * 12
            * self.config.n_embed
            * self.config.context_size
            * (self.config.end_layer - self.config.start_layer + 1)
        )

        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        return flops_achieved
