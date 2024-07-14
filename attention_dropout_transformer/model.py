import math
from dataclasses import field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.common import IntMappedEnum, custom_dataclass
from utils.transformer_modules import BaseModel, LayerNorm, SubModuleStats, MultiAttentionHead


@custom_dataclass
class RegularizingLambdaConfig:
    min_lambda: float = None
    max_lambda: float = 1.0
    exp_coefficient: float = None

    def __post_init__(self):
        assert self.max_lambda > 0

        if self.min_lambda is not None:
            assert self.min_lambda >= 0
            assert self.min_lambda < self.max_lambda
            assert self.exp_coefficient is not None

        if self.exp_coefficient is not None:
            assert self.exp_coefficient < 1


class RoundingType(IntMappedEnum):
    SIGMOID = "SIGMOID"
    SIGMOID_DETACH = "SIGMOID_DETACH"
    NOISE_AND_LINEAR = "NOISE_AND_LINEAR"


class MaskInputType(IntMappedEnum):
    HIDDEN_STATE = "HIDDEN_STATE"
    EMBED = "EMBED"

class L1NormLossType(IntMappedEnum):
    LINEAR = "LINEAR"
    SQUARED = "SQUARED"


@custom_dataclass
class AttentionDropoutConfig:
    use_all_dropout: bool = False
    use_bias: Optional[bool] = None
    n_head: Optional[int] = None
    softmax_dim: int = 1
    rounding_type: Optional[RoundingType] = RoundingType.NOISE_AND_LINEAR
    sigmoid_scale: Optional[float] = None
    shift_init: float = 0
    use_canonical_entropy: bool = False
    use_detached_x_in_dropout_mask: bool = False
    mask_input_type: MaskInputType = MaskInputType.HIDDEN_STATE

    def __post_init__(self):
        assert 0 <= self.shift_init <= torch.pi
        assert self.softmax_dim in [0, 1]

        if (
            self.rounding_type
            not in [RoundingType.SIGMOID, RoundingType.SIGMOID_DETACH]
            and self.sigmoid_scale is not None
        ):
            raise ValueError(
                "sigmoid_slope can only be set if rounding_type is SIGMOID"
            )

        if (
            self.rounding_type in [RoundingType.SIGMOID, RoundingType.SIGMOID_DETACH]
            and self.sigmoid_scale is None
        ):
            self.sigmoid_scale = 60


@custom_dataclass
class ModelConfig(BaseModelConfig):
    attention_dropout_config: AttentionDropoutConfig = field(
        default_factory=AttentionDropoutConfig
    )
    use_dropout_entropy_in_loss: bool = False
    use_dropout_l1_norm_in_loss: bool = True
    l1_norm_loss_type: Optional[L1NormLossType] = L1NormLossType.LINEAR
    start_layer: Optional[int] = None
    end_layer: Optional[int] = None
    dropout_entropy_lambda: Optional[RegularizingLambdaConfig] = None
    dropout_l1_norm_lambda: Optional[RegularizingLambdaConfig] = None

    def __post_init__(self):
        if self.attention_dropout_config.n_head is None:
            self.attention_dropout_config.n_head = self.n_head
        if self.attention_dropout_config.use_bias is None:
            self.attention_dropout_config.use_bias = self.use_bias

        if self.start_layer is None:
            self.start_layer = 1
        if self.end_layer is None:
            self.end_layer = self.n_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")
        if self.start_layer > self.n_layer or self.start_layer < 1:
            raise ValueError("start_layer must be <= n_layer and >= 1")
        if self.end_layer > self.n_layer or self.end_layer < 1:
            raise ValueError("end_layer must be <= n_layer and >= 1")

        if not self.use_dropout_l1_norm_in_loss and self.l1_norm_loss_type is not None:
            raise ValueError(
                "l1_loss_type is set but use_dropout_l1_norm_in_loss is False"
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
            if attr_value is None and getattr(self, flag_attr_name):
                setattr(self, attr_name, RegularizingLambdaConfig(max_lambda=1))


class AttentionDropout(SubModuleStats):
    extra_stats = [
        "dropout_entropy",
        "dropout_l1_norm",
        "dropout_near_one_percent",
        "dropout_near_zero_percent",
        "dropout_change_rate_from_prev",
        "rounded_dropout_l1_norm",
    ]

    def __init__(
        self,
        embed_dim,
        context_size,
        config,
        use_dropout_entropy_in_loss,
        use_dropout_l1_norm_in_loss,
        l1_norm_loss_type,
    ):
        super().__init__()
        assert embed_dim % config.n_head == 0
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.config = config
        self.head_size = embed_dim // config.n_head
        self.use_dropout_entropy_in_loss = use_dropout_entropy_in_loss
        self.use_dropout_l1_norm_in_loss = use_dropout_l1_norm_in_loss

        self.batch_attn_weights = nn.Linear(
            embed_dim, embed_dim * 3, bias=config.use_bias
        )
        self.shift = nn.Parameter(
            torch.full((embed_dim,), config.shift_init, dtype=torch.float32)
        )
        if self.config.mask_input_type == MaskInputType.EMBED:
            self.embed_ln = LayerNorm(embed_dim, config.use_bias)

        self.entropy_fn = (
            self.canonical_entropy
            if config.use_canonical_entropy
            else self.alternate_entropy
        )
        self.l1_norm_fn = self.get_l1_loss_fn(l1_norm_loss_type)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                )
            ),
        )

        self.register_buffer("prev_dropout_mask", torch.empty(0), persistent=False)

    def get_l1_loss_fn(self, l1_loss_type):
        if l1_loss_type == L1NormLossType.LINEAR:
            return lambda x: torch.norm(x, p=1)
        elif l1_loss_type == L1NormLossType.SQUARED:
            return lambda x: torch.norm((x**2) / 2, p=1)
        else:
            raise ValueError(f"Unknown l1_loss_type: {l1_loss_type}")

    def update_stats(self, dropout_mask):
        self.dropout_entropy = self.entropy_fn(dropout_mask)
        self.dropout_l1_norm = self.l1_norm_fn(dropout_mask) / dropout_mask.numel()

        if not self.use_dropout_entropy_in_loss:
            self.dropout_entropy = self.dropout_entropy.detach()
        if not self.use_dropout_l1_norm_in_loss:
            self.dropout_l1_norm = self.dropout_l1_norm.detach()

        with torch.no_grad():
            self.dropout_near_one_percent = (
                dropout_mask > 0.9
            ).sum() / dropout_mask.numel()
            self.dropout_near_zero_percent = (
                dropout_mask < 0.1
            ).sum() / dropout_mask.numel()

            if self.prev_dropout_mask.nelement() != 0:
                matching_1s = (dropout_mask >= 0.5) & (self.prev_dropout_mask >= 0.5)
                matching_0s = (dropout_mask < 0.5) & (self.prev_dropout_mask < 0.5)
                self.dropout_change_rate_from_prev = (
                    1 - (matching_0s.sum() + matching_1s.sum()) / dropout_mask.numel()
                )
            self.prev_dropout_mask = dropout_mask.clone()

    def update_rounded_stats(self, rounded_dropout_mask):
        with torch.no_grad():
            self.rounded_dropout_l1_norm = (
                torch.norm(rounded_dropout_mask, p=1) / rounded_dropout_mask.numel()
            )

    def canonical_entropy(self, dropout_mask):
        # the small constant is for numerical stability
        return (dropout_mask * -torch.log2(dropout_mask + 1e-9)).mean()

    def alternate_entropy(self, dropout_mask):
        # the alternate entropy has the peak at > 0.5, while the canonical one has
        # it < 0.5. In theory, this should be better for achieving both low entropy
        # and low l1 norm because there is more curvature towards 0.
        return ((dropout_mask - 1) * torch.log2((-dropout_mask + 1) + 1e-9)).mean()

    def forward(self, x, embed):
        if self.config.mask_input_type == MaskInputType.HIDDEN_STATE:
            dropout_input = x
        elif self.config.mask_input_type == MaskInputType.EMBED:
            dropout_input = embed
            dropout_input = self.embed_ln(dropout_input)

        dropout_input = (
            dropout_input.detach()
            if self.config.use_detached_x_in_dropout_mask
            else dropout_input
        )

        B, T, C = dropout_input.shape
        q, k, v = self.batch_attn_weights(dropout_input).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)

        if self.config.softmax_dim == 1:
            dropout_values = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, 0)
            dropout_values = causal_attn @ v

        dropout_values = dropout_values.transpose(1, 2).contiguous().view(B, T, C)
        dropout_mask = 0.5 * torch.cos(dropout_values + self.shift) + 0.5

        if self.training:
            self.update_stats(dropout_mask)

        if self.config.rounding_type:
            if self.config.rounding_type == RoundingType.SIGMOID:
                dropout_mask = torch.sigmoid(
                    self.config.sigmoid_scale * (dropout_mask - 0.5)
                )
            elif self.config.rounding_type == RoundingType.SIGMOID_DETACH:
                complement_dropout_mask = (
                    torch.sigmoid(
                        self.config.sigmoid_scale * (dropout_mask.detach() - 0.5)
                    )
                    - dropout_mask.detach()
                )
                dropout_mask = dropout_mask + complement_dropout_mask
            elif self.config.rounding_type == RoundingType.NOISE_AND_LINEAR:
                complement_mask = 1 - dropout_mask.detach()
                if self.training:
                    noise = torch.rand(dropout_mask.shape, device=dropout_mask.device)
                    scaling = torch.where(
                        noise >= complement_mask, complement_mask, complement_mask - 1
                    )
                else:
                    scaling = torch.where(
                        dropout_mask >= 0.5, complement_mask, complement_mask - 1
                    )

                # scaling + dropout_mask should produce either 0s or 1s, but because of
                # precision, it may not. Reducing the precision helps.
                dropout_mask = dropout_mask.to(dtype=torch.float16) + scaling.to(
                    dtype=torch.float16
                )
                dropout_mask = dropout_mask.to(dtype=torch.float32)

        if self.training:
            self.update_rounded_stats(dropout_mask)

        new_x = x * dropout_mask
        return new_x


class FeedForward(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.linear = nn.Linear(
            config.n_embed, config.n_embed * 4, bias=config.use_bias
        )
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(
            config.n_embed * 4, config.n_embed, bias=config.use_bias
        )
        self.dropout = AttentionDropout(
            config.n_embed,
            config.context_size,
            config.attention_dropout_config,
            config.use_dropout_entropy_in_loss,
            config.use_dropout_l1_norm_in_loss,
            config.l1_norm_loss_type,
        )

    def forward(self, x, embed):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.residual_proj(x)
        x = self.dropout(x, embed)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            True,
            config,
        )
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, x, embed):
        x = x + self.multi_attn_head(self.ln1(x), embed)
        x = x + self.feed_forward(self.ln2(x), embed)
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
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config                )
                for _ in range(config.n_layer)
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

        self.register_buffer("dropout_entropy_lambda", torch.empty(0), persistent=False)
        self.register_buffer("dropout_l1_norm_lambda", torch.empty(0), persistent=False)

    @torch.compiler.disable()  # without this, torch.compile keeps recompiling this function. It's caused by self.training_step changing too frequently
    def get_dropout_lambda(self, lambda_config, device):
        if lambda_config is None:
            return torch.empty(0, device=device)

        if lambda_config.exp_coefficient is None:
            return torch.tensor(lambda_config.max_lambda, device=device)

        assert self.training_step is not None
        intersect = (
            lambda_config.min_lambda - 1 if lambda_config.min_lambda is not None else -1
        )
        return torch.tensor(
            min(
                np.exp(lambda_config.exp_coefficient * self.training_step) + intersect,
                lambda_config.max_lambda,
            ),
            device=device,
            dtype=torch.float32,  # need to set explicitly otherwise MPS will complain that it's float64
        )

    def forward(self, x, targets=None):
        device = x.device
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(
            torch.arange(x.shape[1], dtype=torch.long, device=device)
        )
        embed = token_embed + pos_embed
        embed = self.dropout(embed)
        x = embed
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, embed)
        out = self.ln(x)

        if targets is None:
            loss = None
            logits = self.output_layer(out[:, [-1], :])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            additional_loss = None
            if self.training:
                self.aggregate_sub_module_stats()
                if self.is_first_minibatch:
                    self.dropout_entropy_lambda = self.get_dropout_lambda(
                        self.config.dropout_entropy_lambda, device
                    )
                    self.dropout_l1_norm_lambda = self.get_dropout_lambda(
                        self.config.dropout_l1_norm_lambda, device
                    )

                if self.config.use_dropout_entropy_in_loss:
                    additional_loss = self.dropout_entropy * self.dropout_entropy_lambda
                if self.config.use_dropout_l1_norm_in_loss:
                    if additional_loss is None:
                        additional_loss = (
                            self.dropout_l1_norm * self.dropout_l1_norm_lambda
                        )
                    else:
                        additional_loss += (
                            self.dropout_l1_norm * self.dropout_l1_norm_lambda
                        )

            loss = F.cross_entropy(logits, targets.view(-1))
            if additional_loss is not None:
                loss += additional_loss
        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # N = self.get_num_params(True)
        # L, H, Q, T = (
        #     self.config.n_layer,
        #     self.config.n_head,
        #     self.config.n_embed // self.config.n_head,
        #     self.config.context_size,
        # )
        # flops_per_token = 6 * N + 12 * L * H * Q * T

        # # this is contributed by the attention dropout
        # flops_per_token += (
        #     (self.running_active_dropout_percent)
        #     * 12
        #     * self.config.n_embed
        #     * self.config.context_size
        #     * (self.config.end_layer - self.config.start_layer + 1)
        # )

        # flops_per_fwdbwd = flops_per_token * T
        # flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # return flops_achieved
        return None

    def dump_extra_stats(self):
        extra_stats = super().dump_extra_stats()
        if self.dropout_entropy_lambda.nelement() != 0:
            extra_stats["dropout_entropy_lambda"] = self.dropout_entropy_lambda.item()
        if self.dropout_l1_norm_lambda.nelement() != 0:
            extra_stats["dropout_l1_norm_lambda"] = self.dropout_l1_norm_lambda.item()
        return extra_stats
