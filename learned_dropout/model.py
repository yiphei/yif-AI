import math
from dataclasses import field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.common import IntMappedEnum, custom_dataclass
from utils.transformer_modules import (BaseModel, LayerNorm,
                                       MultiAttentionHead, SubModuleStats)


@custom_dataclass
class PenaltyCoeffConfig:
    min_coeff: float = None
    max_coeff: float = 1.0
    exp_rate: float = None

    def __post_init__(self):
        assert self.max_coeff > 0

        if self.min_coeff is not None:
            assert self.min_coeff >= 0
            assert self.min_coeff < self.max_coeff
            assert self.exp_rate is not None

        if self.exp_rate is not None:
            assert 0 < self.exp_rate < 1


class MaskRoundingType(IntMappedEnum):
    SIGMOID = "SIGMOID"
    SIGMOID_DETACH = "SIGMOID_DETACH"
    NOISE_AND_LINEAR = "NOISE_AND_LINEAR"


class DropoutInputType(IntMappedEnum):
    HIDDEN_STATE = "HIDDEN_STATE"
    EMBED = "EMBED"
    EMBED_WITH_LN = "EMBED_WITH_LN"
    EMBED_WITH_TRANSFORMATION = "EMBED_WITH_TRANSFORMATION"
    EMBED_WITH_TRANSFORMATION_AND_LN = "EMBED_WITH_TRANSFORMATION_AND_LN"
    EMBED_WITH_TRANSFORMATION_AND_RES = "EMBED_WITH_TRANSFORMATION_AND_RES"
    EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES = (
        "EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES"
    )
    EMBED_WITH_TRANSFORMATION_WITH_INIT_LN = "EMBED_WITH_TRANSFORMATION_WITH_INIT_LN"
    EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN = (
        "EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN"
    )


class L1NormPenaltyType(IntMappedEnum):
    LINEAR = "LINEAR"
    SQUARED = "SQUARED"


@custom_dataclass
class LearnedDropoutConfig:
    """The default field values are the suggested ones for the best performance.

    NB: there are more hyperparameters here than described in the README. This is because
        either they were found to be detrimental or were trivial additions.

    Args:
        use_bias: whether to use bias. Best to be consistent with
            the rest of the model
        n_head: number of attention heads. Best to be consistent with
            the rest of the model
        mask_rounding_type: the type of rounding applied to the dropout mask.
            MaskRoundingType.NOISE_AND_LINEAR performed better.
        sigmoid_scale: the scaling factor for the sigmoid rounding function.
            This is only used if mask_rounding_type is MaskRoundingType.SIGMOID or
            MaskRoundingType.SIGMOID_DETACH.
        shift_init: the initialization value for the shift bias parameter.
        use_detached_input: whether to detach the dropout input first.
        dropout_input_type: the type of input used for dropout.
            DropoutInputType.HIDDEN_STATE performed better.
    """

    use_bias: Optional[bool] = None
    n_head: Optional[int] = None
    mask_rounding_type: Optional[MaskRoundingType] = MaskRoundingType.NOISE_AND_LINEAR
    sigmoid_scale: Optional[float] = None
    shift_init: float = 0
    use_detached_input: bool = False
    dropout_input_type: DropoutInputType = DropoutInputType.HIDDEN_STATE

    def __post_init__(self):
        assert 0 <= self.shift_init <= torch.pi

        if (
            self.mask_rounding_type
            not in [MaskRoundingType.SIGMOID, MaskRoundingType.SIGMOID_DETACH]
            and self.sigmoid_scale is not None
        ):
            raise ValueError(
                "sigmoid_scale can only be set if mask_rounding_type is SIGMOID"
            )

        if (
            self.mask_rounding_type
            in [MaskRoundingType.SIGMOID, MaskRoundingType.SIGMOID_DETACH]
            and self.sigmoid_scale is None
        ):
            self.sigmoid_scale = 60


@custom_dataclass
class ModelConfig(BaseModelConfig):
    """The default field values are the suggested ones for the best performance.
    Fine-tuning dropout_l1_norm_coeff_config may improve performance.

    NB: there are more hyperparameters here than described in the README. This is because
        either they were found to be detrimental or were trivial additions.

    Args:
        learned_dropout_config: config for the LearnedDropout module.
        use_dropout_entropy_penalty: whether to apply the (Shannon's information theory)
            entropy of dropout as penalty. This proved detrimental because it conflicts with
            l1 norm penalty.
        use_dropout_l1_norm_penalty: whether to apply the L1 norm penalty.
        l1_norm_penalty_type: the type of L1 norm penalty applied.
            L1NormPenaltyType.SQUARED performed better.
        dropout_entropy_coeff_config: config for the dropout entropy penalty coefficient.
        dropout_l1_norm_coeff_config: config for the dropout L1 norm penalty coefficient.
            This may be fine-tuned for best performance.
    """

    learned_dropout_config: LearnedDropoutConfig = field(
        default_factory=LearnedDropoutConfig
    )
    use_dropout_entropy_penalty: bool = False
    use_dropout_l1_norm_penalty: bool = True
    l1_norm_penalty_type: Optional[L1NormPenaltyType] = L1NormPenaltyType.SQUARED
    dropout_entropy_coeff_config: Optional[PenaltyCoeffConfig] = None
    dropout_l1_norm_coeff_config: Optional[PenaltyCoeffConfig] = None

    def __post_init__(self):
        if self.learned_dropout_config.n_head is None:
            self.learned_dropout_config.n_head = self.n_head
        if self.learned_dropout_config.use_bias is None:
            self.learned_dropout_config.use_bias = self.use_bias

        if (
            not self.use_dropout_l1_norm_penalty
            and self.l1_norm_penalty_type is not None
        ):
            raise ValueError(
                "l1_norm_penalty_type is set but use_dropout_l1_norm_penalty is False"
            )

        for attr_name, flag_attr_name in [
            ("dropout_entropy_coeff_config", "use_dropout_entropy_penalty"),
            ("dropout_l1_norm_coeff_config", "use_dropout_l1_norm_penalty"),
        ]:
            attr_value = getattr(self, attr_name)
            if attr_value is None and getattr(self, flag_attr_name):
                setattr(self, attr_name, PenaltyCoeffConfig(max_coeff=1))
            elif not getattr(self, flag_attr_name) and attr_value is not None:
                raise ValueError(f"{attr_name} is set but {flag_attr_name} is False")


class LearnedDropout(SubModuleStats):
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
        use_dropout_entropy_penalty,
        use_dropout_l1_norm_penalty,
        l1_norm_penalty_type,
    ):
        super().__init__()
        assert embed_dim % config.n_head == 0
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.config = config
        self.head_size = embed_dim // config.n_head
        self.use_dropout_entropy_penalty = use_dropout_entropy_penalty
        self.use_dropout_l1_norm_penalty = use_dropout_l1_norm_penalty

        self.batch_attn_weights = nn.Linear(
            embed_dim, embed_dim * 3, bias=config.use_bias
        )
        self.shift = nn.Parameter(
            torch.full((embed_dim,), config.shift_init, dtype=torch.float32)
        )
        if self.config.dropout_input_type in [
            DropoutInputType.EMBED_WITH_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES,
        ]:
            self.embed_ln = LayerNorm(embed_dim, config.use_bias)

        self.l1_norm_fn = self.get_l1_norm_penalty_fn(l1_norm_penalty_type)

        self.register_buffer("prev_dropout_mask", torch.empty(0), persistent=False)

    def get_l1_norm_penalty_fn(self, l1_norm_penalty_type):
        if l1_norm_penalty_type == L1NormPenaltyType.LINEAR:
            return lambda x: torch.norm(x, p=1)
        elif l1_norm_penalty_type == L1NormPenaltyType.SQUARED:
            return lambda x: torch.norm((x**2) / 2, p=1)
        elif l1_norm_penalty_type is None:
            return lambda x: torch.norm(x, p=1)
        else:
            raise ValueError(f"Unknown l1_norm_penalty_type: {l1_norm_penalty_type}")

    def update_stats(self, dropout_mask):
        self.dropout_entropy = (dropout_mask * -torch.log2(dropout_mask + 1e-9)).mean()
        self.dropout_l1_norm = self.l1_norm_fn(dropout_mask) / dropout_mask.numel()

        if not self.use_dropout_entropy_penalty:
            self.dropout_entropy = self.dropout_entropy.detach()
        if not self.use_dropout_l1_norm_penalty:
            self.dropout_l1_norm = self.dropout_l1_norm.detach()

        with torch.no_grad():
            self.dropout_near_one_percent = (
                dropout_mask > 0.9
            ).sum() / dropout_mask.numel()
            self.dropout_near_zero_percent = (
                dropout_mask < 0.1
            ).sum() / dropout_mask.numel()

            if self.prev_dropout_mask.nelement() != 0:
                # this does not work well with torch.compile. You won't get any errors
                # but dropout_change_rate_from_prev will probably have NaNs
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

    def forward(self, x, embed):
        if self.config.dropout_input_type == DropoutInputType.HIDDEN_STATE:
            dropout_input = x
        elif self.config.dropout_input_type in [
            DropoutInputType.EMBED,
            DropoutInputType.EMBED_WITH_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_WITH_INIT_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN,
        ]:
            dropout_input = embed
            if self.config.dropout_input_type in [
                DropoutInputType.EMBED_WITH_LN,
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN,
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES,
            ]:
                dropout_input = self.embed_ln(dropout_input)

        dropout_input = (
            dropout_input.detach() if self.config.use_detached_input else dropout_input
        )

        B, T, C = dropout_input.shape
        q, k, v = self.batch_attn_weights(dropout_input).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, self.head_size).transpose(1, 2)

        dropout_values = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )
        dropout_values = dropout_values.transpose(1, 2).contiguous().view(B, T, C)
        dropout_mask = 0.5 * torch.cos(dropout_values + self.shift) + 0.5

        if self.training:
            self.update_stats(dropout_mask)

        if self.config.mask_rounding_type:
            if self.config.mask_rounding_type == MaskRoundingType.SIGMOID:
                dropout_mask = torch.sigmoid(
                    self.config.sigmoid_scale * (dropout_mask - 0.5)
                )
            elif self.config.mask_rounding_type == MaskRoundingType.SIGMOID_DETACH:
                dropout_mask_scaling = (
                    torch.sigmoid(
                        self.config.sigmoid_scale * (dropout_mask.detach() - 0.5)
                    )
                    - dropout_mask.detach()
                )
                dropout_mask = dropout_mask + dropout_mask_scaling
            elif self.config.mask_rounding_type == MaskRoundingType.NOISE_AND_LINEAR:
                complement_mask = 1 - dropout_mask.detach()
                if self.training:
                    noise = torch.rand(dropout_mask.shape, device=dropout_mask.device)
                    scaling = torch.where(
                        noise <= dropout_mask, complement_mask, -dropout_mask.detach()
                    )
                else:
                    scaling = torch.where(
                        dropout_mask >= 0.5, complement_mask, -dropout_mask.detach()
                    )

                # scaling + dropout_mask should produce either 0s or 1s, but because of
                # precision, it may not. Reducing the precision helps.
                dropout_mask = dropout_mask.to(dtype=torch.float16) + scaling.to(
                    dtype=torch.float16
                )
                dropout_mask = dropout_mask.to(dtype=torch.float32)

        if self.training:
            self.update_rounded_stats(dropout_mask)

        return x * dropout_mask


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
        self.dropout = LearnedDropout(
            config.n_embed,
            config.context_size,
            config.learned_dropout_config,
            config.use_dropout_entropy_penalty,
            config.use_dropout_l1_norm_penalty,
            config.l1_norm_penalty_type,
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
        )
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, x, embed):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x), embed)
        return x


class EmbedAttentionHead(nn.Module):
    def __init__(
        self, dim_in, n_head, use_bias, context_size, dropout_rate=0, use_flash=True
    ):
        super().__init__()
        assert dim_in % n_head == 0
        self.dim_in = dim_in
        self.head_size = dim_in // n_head
        self.n_head = n_head
        self.dropout_rate = dropout_rate

        self.batch_attn_weights = nn.Linear(self.dim_in, self.dim_in * 3, bias=use_bias)
        self.linear = nn.Linear(self.dim_in, self.dim_in, bias=use_bias)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.using_flash = False
        if not hasattr(F, "scaled_dot_product_attention") or not use_flash:
            self.register_buffer(
                "tril",
                torch.tril(
                    torch.ones(context_size, context_size).view(
                        1, 1, context_size, context_size
                    )
                ),
            )
        else:
            print("Using flash attention.")
            self.using_flash = True

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.batch_attn_weights(x).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if self.using_flash:
            new_x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (
                self.head_size**-0.5
            )  # B,H,T,S @ B,H,S,T ->B, H, T, T
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            causal_attn_probs = F.softmax(causal_attn, dim=-1)
            causal_attn_probs = self.dropout_1(causal_attn_probs)
            new_x = causal_attn_probs @ v  # B,H,T,T @ B,H,T,S -> B,H,T,S

        new_x = (
            new_x.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        new_x = self.linear(new_x)
        new_x = self.dropout_2(new_x)
        return new_x


class LearnedDropoutTransformer(BaseModel):
    model_config_cls = ModelConfig

    def _init_model(self, config: ModelConfig):
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.learned_dropout_config.dropout_input_type in [
            DropoutInputType.EMBED_WITH_TRANSFORMATION,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_WITH_INIT_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN,
        ]:
            self.embed_transform = EmbedAttentionHead(
                config.n_embed,
                config.n_head,
                config.use_bias,
                config.context_size,
                config.dropout_rate,
                True,
            )
            if config.learned_dropout_config.dropout_input_type in [
                DropoutInputType.EMBED_WITH_TRANSFORMATION_WITH_INIT_LN,
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN,
            ]:
                self.embed_transform_ln = LayerNorm(config.n_embed, config.use_bias)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
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

        self.register_buffer("dropout_entropy_coeff", torch.empty(0), persistent=False)
        self.register_buffer("dropout_l1_norm_coeff", torch.empty(0), persistent=False)

    @torch.compiler.disable()  # without this, torch.compile keeps recompiling this function. It's caused by self.training_step changing too frequently
    def get_penalty_coeff(self, coeff_config, device):
        if coeff_config is None:
            return torch.empty(0, device=device)

        if coeff_config.exp_rate is None:
            return torch.tensor(coeff_config.max_coeff, device=device)

        assert self.training_step is not None
        intersect = (
            coeff_config.min_coeff - 1 if coeff_config.min_coeff is not None else -1
        )
        return torch.tensor(
            min(
                np.exp(coeff_config.exp_rate * self.training_step) + intersect,
                coeff_config.max_coeff,
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

        if self.config.learned_dropout_config.dropout_input_type in [
            DropoutInputType.EMBED_WITH_TRANSFORMATION,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_WITH_INIT_LN,
            DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN,
        ]:
            if self.config.learned_dropout_config.dropout_input_type in [
                DropoutInputType.EMBED_WITH_TRANSFORMATION_WITH_INIT_LN,
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN,
            ]:
                embed = self.embed_transform_ln(embed)

            transformed = self.embed_transform(embed)
            if self.config.learned_dropout_config.dropout_input_type in [
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES,
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_LN_AND_RES,
                DropoutInputType.EMBED_WITH_TRANSFORMATION_AND_RES_WITH_INIT_LN,
            ]:
                embed = embed + transformed
            else:
                embed = transformed

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
                    self.dropout_entropy_coeff = self.get_penalty_coeff(
                        self.config.dropout_entropy_coeff_config, device
                    )
                    self.dropout_l1_norm_coeff = self.get_penalty_coeff(
                        self.config.dropout_l1_norm_coeff_config, device
                    )

                if self.config.use_dropout_entropy_penalty:
                    additional_loss = self.dropout_entropy * self.dropout_entropy_coeff
                if self.config.use_dropout_l1_norm_penalty:
                    if additional_loss is None:
                        additional_loss = (
                            self.dropout_l1_norm * self.dropout_l1_norm_coeff
                        )
                    else:
                        additional_loss += (
                            self.dropout_l1_norm * self.dropout_l1_norm_coeff
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
        if self.dropout_entropy_coeff.nelement() != 0:
            extra_stats["dropout_entropy_coeff"] = self.dropout_entropy_coeff.item()
        if self.dropout_l1_norm_coeff.nelement() != 0:
            extra_stats["dropout_l1_norm_coeff"] = self.dropout_l1_norm_coeff.item()
        return extra_stats
