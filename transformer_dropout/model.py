import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


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


class OrderType(str, Enum):
    ORIGINAL = "ORIGINAL"
    ALT = "ALT"
    ALT2 = "ALT2"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return OrderType.ORIGINAL
        elif num == 2:
            return OrderType.ALT
        elif num == 3:
            return OrderType.ALT2
        else:
            raise ValueError("Invalid rounding type number")


class SubPosEmbedType(str, Enum):
    NO = "NO"
    NO_LN = "NO_LN"
    YES_LN = "YES_LN"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return SubPosEmbedType.NO
        elif num == 2:
            return SubPosEmbedType.NO_LN
        elif num == 3:
            return SubPosEmbedType.YES_LN
        else:
            raise ValueError("Invalid sub pos embed type number")


class TokenLossType(str, Enum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE_SIM_NORM = "CONSINE_SIM_NORM"
    COSINE_SIM_LOG = "CONSINE_SIM_LOG"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return TokenLossType.NONE
        elif num == 2:
            return TokenLossType.MSE
        elif num == 3:
            return TokenLossType.COSINE_SIM_NORM
        elif num == 4:
            return TokenLossType.COSINE_SIM_LOG
        else:
            raise ValueError("Invalid token loss type number")


class TokenLossDetachType(str, Enum):
    NONE = "NONE"
    ORIGINAL = "ORIGINAL"
    FINAL = "FINAL"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return TokenLossDetachType.NONE
        elif num == 2:
            return TokenLossDetachType.ORIGINAL
        elif num == 3:
            return TokenLossDetachType.FINAL
        else:
            raise ValueError("Invalid token loss detatch type number")


class TokenEmbedLayerNormType(str, Enum):
    NONE = "NONE"
    X_ORIGINAL = "X_ORIGINAL"
    AVG_CUM_SUM = "AVG_CUM_SUM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return TokenEmbedLayerNormType.NONE
        elif num == 2:
            return TokenEmbedLayerNormType.X_ORIGINAL
        elif num == 3:
            return TokenEmbedLayerNormType.AVG_CUM_SUM
        else:
            raise ValueError("Invalid token embed layer norm type number")


@dataclass
class LearnedDropoutConfig:
    use_bias: bool
    start_layer: int
    order_type: Union[OrderType, int]
    add_pos_embed: bool
    sub_pos_embed: Union[SubPosEmbedType, int]
    token_loss_type: Union[TokenLossType, int] = TokenLossType.NONE
    token_loss_detach_type: Optional[Union[TokenLossDetachType, int]] = None
    token_loss_coeff: Optional[float] = None
    use_ln_on_final_x_state: Optional[bool] = None
    token_embed_layer_norm_type: Optional[Union[TokenEmbedLayerNormType, int]] = None
    end_layer: Optional[int] = None
    n_heads: int = 1
    profile_dropout_mask: bool = False

    def __post_init__(self):
        if type(self.token_loss_type) == int:
            self.token_loss_type = TokenLossType.get_type_from_int(self.token_loss_type)

        if type(self.token_loss_detach_type) == int:
            self.token_loss_detach_type = TokenLossDetachType.get_type_from_int(
                self.token_loss_detach_type
            )

        if type(self.token_embed_layer_norm_type) == int:
            self.token_embed_layer_norm_type = (
                TokenEmbedLayerNormType.get_type_from_int(
                    self.token_embed_layer_norm_type
                )
            )

        if self.token_loss_type != TokenLossType.NONE:
            if self.token_loss_coeff is None:
                self.token_loss_coeff = 1.0
            else:
                assert self.token_loss_coeff > 0
            assert self.use_ln_on_final_x_state is not None
            assert self.token_embed_layer_norm_type is not None
            assert self.token_loss_detach_type is not None
        else:
            assert self.token_loss_coeff is None
            assert self.use_ln_on_final_x_state is None
            assert self.token_embed_layer_norm_type is None
            assert self.token_loss_detach_type is None

        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        if type(self.order_type) == int:
            self.order_type = OrderType.get_type_from_int(self.order_type)

        if type(self.sub_pos_embed) == int:
            self.sub_pos_embed = SubPosEmbedType.get_type_from_int(self.sub_pos_embed)

        assert self.n_heads >= 1

        if self.token_loss_detach_type is not None:
            assert self.token_loss_type != TokenLossType.NONE
        else:
            assert self.token_loss_type == TokenLossType.NONE


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


class LayerNorm(nn.Module):
    """From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, dim_in, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_in)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class OptimizedMultiAttentionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.n_embed % config.n_head == 0
        self.dim_in = config.n_embed
        self.head_size = config.n_embed // config.n_head
        self.n_heads = config.n_head
        self.dropout_rate = config.dropout_rate
        self.profile_attn = config.profile_attn

        self.batch_attn_weights = nn.Linear(
            self.dim_in, self.dim_in * 3, bias=config.bias
        )
        self.residual_proj = nn.Linear(self.dim_in, self.dim_in, bias=config.bias)

        if config.use_learned_dropout and False:
            self.dropout_1 = LearnedDropout(
                config.context_size, config.learned_dropout_config
            )
            self.dropout_2 = LearnedDropout(self.dim_in, config.learned_dropout_config)
        else:
            self.dropout_1 = nn.Dropout(config.dropout_rate)
            self.dropout_2 = nn.Dropout(config.dropout_rate)

        self.use_flash = False
        if (
            not hasattr(F, "scaled_dot_product_attention")
            or isinstance(self.dropout_1, LearnedDropout)
            or config.profile_attn
        ):
            self.register_buffer(
                "tril",
                torch.tril(
                    torch.ones(config.context_size, config.context_size).view(
                        1, 1, config.context_size, config.context_size
                    )
                ),
            )
        else:
            print("Using flash attention.")
            self.use_flash = True

    def forward(self, x):
        import wandb

        B, T, C = x.shape

        q, k, v = self.batch_attn_weights(x).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True
            )
            # TODO: add custom dropout here. Otherwise, avoid using flash attention for now
            # if dropout_1 is LearnedDropout
        else:
            attn = (q @ k.transpose(-2, -1)) * (
                self.head_size**-0.5
            )  # B,H,T,S @ B,H,S,T ->B, H, T, T
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            causal_attn = F.softmax(causal_attn, dim=-1)
            causal_attn = self.dropout_1(causal_attn)
            out = causal_attn @ v  # B,H,T,T @ B,H,T,S -> B,H,T,S

        mask = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        new_x = self.residual_proj(mask)
        new_x = self.dropout_2(new_x)

        if self.training and self.is_last_minibatch and self.profile_attn:
            log_x = x.detach()
            log_new_x = new_x.detach()
            log_dropout_mask = mask.detach()
            log_causal_attn = causal_attn.detach()
            log_attn = attn.detach()
            causal_attn_dim_2_mean = causal_attn.mean(dim=-2)
            causal_attn_dim_2_mean[:, :, : T // 2] *= -1
            causal_attn_dim_2_mean_head_mean = causal_attn_dim_2_mean.mean(dim=-2)
            causal_attn_dim_2_mean_head_std = causal_attn_dim_2_mean.std(dim=-2)
            causal_attn_dim_2_mean_head_std[:, : T // 2] *= -1
            if mask.dtype == torch.bfloat16 or causal_attn.dtype == torch.bfloat16:
                log_x = log_x.half()
                log_new_x = log_new_x.half()
                log_dropout_mask = log_dropout_mask.half()
                log_causal_attn = log_causal_attn.half()
                log_attn = log_attn.half()
                causal_attn_dim_2_mean = causal_attn_dim_2_mean.detach().half()
                causal_attn_dim_2_mean_head_mean = (
                    causal_attn_dim_2_mean_head_mean.detach().half()
                )
                causal_attn_dim_2_mean_head_std = (
                    causal_attn_dim_2_mean_head_std.detach().half()
                )
            wandb.log(
                {
                    self.module_name + ".a__input_x": log_x,
                    self.module_name + ".l__new_x": log_new_x,
                    self.module_name + ".g__mask": log_dropout_mask,
                    self.module_name + ".c__causal_attn": log_causal_attn,
                    self.module_name + ".b__attn": log_attn,
                    self.module_name
                    + ".h__mask_dim_2_std": log_dropout_mask.std(dim=-2),
                    self.module_name
                    + ".d__causal_attn_dim_2_mean": causal_attn_dim_2_mean,
                    self.module_name
                    + ".e__causal_attn_dim_2_mean_head_mean": causal_attn_dim_2_mean_head_mean,
                    self.module_name
                    + ".f__causal_attn_dim_2_mean_head_std": causal_attn_dim_2_mean_head_std,
                },
                commit=False,
            )

        return new_x


class LearnedDropout(nn.Module):
    def __init__(self, embed_dim, context_size, config, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size

        self.config = config
        self.module_name = None  # used for logging

        self.head_size = embed_dim // config.n_heads
        self.k_v_weights = nn.Linear(embed_dim, embed_dim * 2, bias=config.use_bias)
        self.q_weights = nn.Linear(embed_dim, embed_dim, bias=config.use_bias)
        self.residual_proj = nn.Linear(embed_dim, embed_dim, bias=config.use_bias)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.dropout_rate = dropout_rate

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                )
            ),
        )

    def forward(self, x_state, x_pred):
        import wandb

        B, T, C = x_state.shape
        k, v = self.k_v_weights(x_state).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)

        q = self.q_weights(x_pred)
        q = q.view(B, T, self.config.n_heads, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True
        )

        mask = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        new_x_pred = self.residual_proj(mask)
        new_x_pred = self.dropout_2(new_x_pred)

        return new_x_pred


class FeedForward(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        use_learned_dropout=False,
        should_profile_layer_x=False,
    ):
        super().__init__()
        self.module_name = None
        self.should_profile_layer_x = should_profile_layer_x
        self.use_learned_dropout = use_learned_dropout
        self.config = config
        self.linear = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(
            config.n_embed * 4, config.n_embed, bias=config.bias
        )
        if config.use_learned_dropout and use_learned_dropout and False:
            self.dropout = LearnedDropout(
                config.n_embed, config.context_size, config.learned_dropout_config
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
        should_profile_layer_x=False,
        is_last=False,
    ):
        super().__init__()
        self.use_learned_dropout = config.use_learned_dropout
        if self.use_learned_dropout:
            self.update_state = not (
                is_last and config.learned_dropout_config.order_type == OrderType.ALT2
            )
            self.order_type = config.learned_dropout_config.order_type

            if self.update_state:
                self.multi_attn_head_state = OptimizedMultiAttentionHead(config)
            self.multi_attn_head_pred = OptimizedMultiAttentionHead(config)
            self.multi_attn_head_merge = LearnedDropout(
                config.n_embed,
                config.context_size,
                config.learned_dropout_config,
                config.dropout_rate,
            )
            if self.update_state:
                self.feed_forward_state = FeedForward(
                    config, use_learned_dropout, should_profile_layer_x
                )
            self.feed_forward_pred = FeedForward(
                config, use_learned_dropout, should_profile_layer_x
            )

            self.merge_ln_state = LayerNorm(config.n_embed, config.bias)
            if self.update_state:
                self.ln1_state = LayerNorm(config.n_embed, config.bias)
                self.ln2_state = LayerNorm(config.n_embed, config.bias)

            self.ln1_pred = LayerNorm(config.n_embed, config.bias)
            self.ln2_pred = LayerNorm(config.n_embed, config.bias)
            self.merge_ln_pred = LayerNorm(config.n_embed, config.bias)
        else:
            self.multi_attn_head = OptimizedMultiAttentionHead(config)
            self.feed_forward = FeedForward(
                config, use_learned_dropout, should_profile_layer_x
            )
            self.ln1 = LayerNorm(config.n_embed, config.bias)
            self.ln2 = LayerNorm(config.n_embed, config.bias)

    def forward(self, x_state, x_pred):
        if self.use_learned_dropout:
            if self.order_type == OrderType.ORIGINAL:
                x_state = x_state + self.multi_attn_head_state(self.ln1_state(x_state))
                x_state = x_state + self.feed_forward_state(self.ln2_state(x_state))

                x_pred = x_pred + self.multi_attn_head_pred(self.ln1_pred(x_pred))
                x_pred = x_pred + self.multi_attn_head_merge(
                    self.merge_ln_state(x_state), self.merge_ln_pred(x_pred)
                )
                x_pred = x_pred + self.feed_forward_pred(self.ln2_pred(x_pred))
            elif self.order_type == OrderType.ALT:
                x_state = x_state + self.multi_attn_head_state(self.ln1_state(x_state))
                x_state = x_state + self.feed_forward_state(self.ln2_state(x_state))

                x_pred = x_pred + self.multi_attn_head_merge(
                    self.merge_ln_state(x_state), self.merge_ln_pred(x_pred)
                )
                x_pred = x_pred + self.multi_attn_head_pred(self.ln1_pred(x_pred))
                x_pred = x_pred + self.feed_forward_pred(self.ln2_pred(x_pred))
            elif self.order_type == OrderType.ALT2:
                x_pred = x_pred + self.multi_attn_head_merge(
                    self.merge_ln_state(x_state), self.merge_ln_pred(x_pred)
                )
                if self.update_state:
                    x_state = x_state + self.multi_attn_head_state(
                        self.ln1_state(x_state)
                    )
                x_pred = x_pred + self.multi_attn_head_pred(self.ln1_pred(x_pred))

                if self.update_state:
                    x_state = x_state + self.feed_forward_state(self.ln2_state(x_state))
                x_pred = x_pred + self.feed_forward_pred(self.ln2_pred(x_pred))
            else:
                raise ValueError("Invalid order type")
        else:
            x_state = x_state + self.multi_attn_head(self.ln1(x_state))
            x_state = x_state + self.feed_forward(self.ln2(x_state))
        return x_state, x_pred


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
        if config.use_learned_dropout and (
            config.learned_dropout_config.add_pos_embed
            or self.config.learned_dropout_config.sub_pos_embed != SubPosEmbedType.NO
        ):
            self.positional_embedding = nn.Embedding(
                config.context_size + 1, config.n_embed
            )
        else:
            self.positional_embedding = nn.Embedding(
                config.context_size, config.n_embed
            )
        if config.use_learned_dropout:
            self.pred_feed_forward = nn.Linear(
                config.n_embed, config.n_embed, bias=config.bias
            )
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

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    (i + 1) >= (learned_config_start_layer)
                    and (i + 1) <= (learned_config_end_layer),
                    i + 1 == profile_layer_x,
                    i == (config.n_layer - 1),
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.bias)
        if (
            config.use_learned_dropout
            and config.learned_dropout_config.sub_pos_embed == SubPosEmbedType.YES_LN
        ):
            self.positional_embedding_ln = LayerNorm(config.n_embed, config.bias)

        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)

        if (
            self.config.use_learned_dropout
            and self.config.learned_dropout_config.token_loss_type != TokenLossType.NONE
            and self.config.learned_dropout_config.use_ln_on_final_x_state
        ):
            self.final_x_state_ln = LayerNorm(config.n_embed, True)
        if (
            self.config.use_learned_dropout
            and self.config.learned_dropout_config.token_loss_type != TokenLossType.NONE
            and self.config.learned_dropout_config.token_embed_layer_norm_type
            != TokenEmbedLayerNormType.NONE
        ):
            self.token_embed_layer_norm = LayerNorm(config.n_embed, True)

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
                    param_to_param_name[module.k_v_weights.weight].split(".")[:-2]
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def update_is_last_minibatch(self, val):
        if val != self.is_last_minibatch:
            self.is_last_minibatch = val
            for module in self.modules():
                module.is_last_minibatch = val

    @classmethod
    def init_from_checkpoint(cls, checkpoint_dict, gradient_accumulation_steps=None):
        model_config = ModelConfig(**checkpoint_dict["model_config"])
        model = cls(model_config, gradient_accumulation_steps)
        state_dict = checkpoint_dict["model"]

        # This is caused by compiling the model. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        return model

    def forward(self, x, targets=None):
        device = x.device
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(
            torch.arange(x.shape[1], dtype=torch.long, device=device)
        )
        embed = token_embed + pos_embed
        x_state = self.dropout(embed)
        x_original = x_state
        x_pred = None
        if self.config.use_learned_dropout:
            if (
                self.config.use_learned_dropout
                and self.config.learned_dropout_config.add_pos_embed
            ):
                x_pred = self.pred_feed_forward(x_state) + self.positional_embedding(
                    torch.arange(
                        start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                    )
                )
            else:
                x_pred = self.pred_feed_forward(x_state)

        for transformer_block in self.transformer_blocks:
            x_state, x_pred = transformer_block(x_state, x_pred)

        if self.config.use_learned_dropout:
            out = self.ln(x_pred)
        else:
            out = self.ln(x_state)

        additional_loss = torch.tensor(0.0, device=device)
        raw_loss = torch.tensor(0.0, device=device)
        if (
            self.training
            and self.config.use_learned_dropout
            and self.config.learned_dropout_config.token_loss_type != TokenLossType.NONE
        ):
            if (
                self.config.learned_dropout_config.token_loss_detach_type
                == TokenLossDetachType.ORIGINAL
            ):
                x_original = x_original.detach()
            elif (
                self.config.learned_dropout_config.token_loss_detach_type
                == TokenLossDetachType.FINAL
            ):
                x_state = x_state.detach()

            if self.config.learned_dropout_config.use_ln_on_final_x_state:
                x_state = self.final_x_state_ln(x_state)

            if (
                self.config.learned_dropout_config.token_embed_layer_norm_type
                == TokenEmbedLayerNormType.X_ORIGINAL
            ):
                x_original = self.token_embed_layer_norm(x_original)

            cum_sum = torch.cumsum(x_original, dim=-2)
            avg_sum = cum_sum / torch.arange(
                1, x.shape[1] + 1, dtype=torch.long, device=device
            ).unsqueeze(0).unsqueeze(-1)

            if (
                self.config.learned_dropout_config.token_embed_layer_norm_type
                == TokenEmbedLayerNormType.AVG_CUM_SUM
            ):
                avg_sum = self.token_embed_layer_norm(avg_sum)

            if self.config.learned_dropout_config.token_loss_type == TokenLossType.MSE:
                raw_loss = F.mse_loss(avg_sum, x_state, reduction="mean")
                additional_loss = (
                    raw_loss * self.config.learned_dropout_config.token_loss_coeff
                )
            elif (
                self.config.learned_dropout_config.token_loss_type
                == TokenLossType.COSINE_SIM_NORM
            ):
                cosine_sim = F.cosine_similarity(avg_sum, x_state, dim=-1)
                raw_loss = (1 - (cosine_sim + 1) / 2).mean()
                additional_loss = (
                    raw_loss * self.config.learned_dropout_config.token_loss_coeff
                )
            elif (
                self.config.learned_dropout_config.token_loss_type
                == TokenLossType.COSINE_SIM_LOG
            ):
                cosine_sim = F.cosine_similarity(avg_sum, x_state, dim=-1)
                raw_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                additional_loss = (
                    raw_loss * self.config.learned_dropout_config.token_loss_coeff
                )
            else:
                raise ValueError("Invalid token loss type")

        if (
            self.config.use_learned_dropout
            and self.config.learned_dropout_config.sub_pos_embed != SubPosEmbedType.NO
        ):
            out = out - self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )
            if (
                self.config.learned_dropout_config.sub_pos_embed
                == SubPosEmbedType.YES_LN
            ):
                out = self.positional_embedding_ln(out)

        if targets is None:
            loss = None
            logits = self.output_layer(out[:, [-1], :])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1)) + additional_loss
        return (logits, loss, raw_loss, additional_loss)

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        adamw_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        sgd_groups = []

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        adam_optimizer = torch.optim.AdamW(
            adamw_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        sgd_optimizer = (
            torch.optim.SGD(sgd_groups, lr=learning_rate)
            if len(sgd_groups) > 0
            else None
        )
        return OptimizerWrapper(adam_optimizer, sgd_optimizer)

    def get_num_params(self, exclude_positional_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_positional_embedding:
            n_params -= self.positional_embedding.weight.numel()
        return n_params

    def get_accuracy(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        return (probs.max(dim=-1).indices.view(-1) != targets.view(-1)).float().mean()

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        From https://github.com/karpathy/nanoGPT/blob/master/model.py#L289
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params(True)
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embed // self.config.n_head,
            self.config.context_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        if self.config.use_learned_dropout and False:
            flops_per_token += (
                (self.running_active_dropout_percent)
                * 12
                * self.config.n_embed
                * self.config.context_size
                * self.n_learned_dropout
            )

        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        return flops_achieved

    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(x[:, -self.config.context_size :], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x


class OptimizerWrapper:
    def __init__(self, adamw_optimizer, sgd_optimizer):
        self.adamw_optimizer = adamw_optimizer
        self.sgd_optimizer = sgd_optimizer

    def state_dict(self):
        return {
            "adam_optimizer": self.adamw_optimizer.state_dict(),
            **(
                {"sgd_optimizer": self.sgd_optimizer.state_dict()}
                if self.sgd_optimizer is not None
                else {}
            ),
        }

    def load_state_dict(self, state_dict):
        self.adamw_optimizer.load_state_dict(state_dict["adam_optimizer"])
        if state_dict.get("sgd_optimizer", None) is not None:
            self.sgd_optimizer.load_state_dict(state_dict["sgd_optimizer"])

    def change_lr(self, lr):
        for optimizer in [self.adamw_optimizer, self.sgd_optimizer]:
            if optimizer is None:
                continue

            for param_group in optimizer.param_groups:
                if param_group.get("is_lr_fixed", False):
                    continue
                param_group["lr"] = lr

    def step(self, *args, **kwargs):
        self.adamw_optimizer.step(*args, **kwargs)
        if self.sgd_optimizer is not None:
            self.sgd_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.adamw_optimizer.zero_grad(*args, **kwargs)
        if self.sgd_optimizer is not None:
            self.sgd_optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.adamw_optimizer.param_groups + (
            self.sgd_optimizer.param_groups if self.sgd_optimizer is not None else []
        )
