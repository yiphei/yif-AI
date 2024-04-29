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


class ReturnType(str, Enum):
    NO_RES_PROJ_MASK = "NO_RES_PROJ_MASK"
    NO_RES_PROJ_NEW_X = "NO_RES_PROJ_NEW_X"
    RES_PROJ_MASK_THEN_MASK = "RES_PROJ_MASK_THEN_MASK"
    RES_PROJ_MASK_THEN_NEW_X = "RES_PROJ_MASK_THEN_NEW_X"
    RES_PROJ_NEW_X_THEN_NEW_X = "RES_PROJ_NEW_X_THEN_NEW_X"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return ReturnType.NO_RES_PROJ_MASK
        elif num == 2:
            return ReturnType.NO_RES_PROJ_NEW_X
        elif num == 3:
            return ReturnType.RES_PROJ_MASK_THEN_MASK
        elif num == 4:
            return ReturnType.RES_PROJ_MASK_THEN_NEW_X
        elif num == 5:
            return ReturnType.RES_PROJ_NEW_X_THEN_NEW_X
        else:
            raise ValueError("Invalid return type number")


@dataclass
class LearnedDropoutConfig:
    # use_dropout_entropy_in_loss: bool
    # use_dropout_l1_norm_in_loss: bool
    use_bias: bool
    start_layer: int
    normalize_by_context_size: bool
    return_type: Union[ReturnType, int]
    use_res_add: bool
    end_layer: Optional[int] = None
    dropout_rate: float = 0.0
    softmax_dim: int = 2
    # rounding_type: Optional[Union[RoundingType, int]] = None
    # sigmoid_slope: Optional[float] = None
    # shift_init: float = torch.pi / 2
    n_heads: int = 1
    # use_canonical_entropy: bool = False
    # use_detached_x_in_dropout_mask: bool = False
    # dropout_entropy_lambda: Optional[RegularizingLambdaConfig] = field(default=None)
    # dropout_l1_norm_lambda: Optional[RegularizingLambdaConfig] = field(default=None)
    profile_dropout_mask: bool = False

    def __post_init__(self):
        # assert 0 <= self.shift_init <= torch.pi
        assert self.softmax_dim in [0, 1, 2, 3]
        assert 0.0 <= self.dropout_rate < 1.0
        if self.end_layer is None:
            self.end_layer = self.start_layer

        if self.start_layer > self.end_layer:
            raise ValueError("start_layer must be <= end_layer")

        assert self.n_heads >= 1

        if type(self.return_type) == int:
            assert self.return_type in [1, 2, 3, 4, 5]
            self.return_type = ReturnType.get_type_from_int(self.return_type)

        # if (
        #     self.use_dropout_entropy_in_loss
        #     and self.rounding_type
        #     and self.rounding_type in [2, 3]
        # ):
        #     raise ValueError(
        #         "rounding_type cannot be 2 or 3 if use_dropout_entropy_in_loss"
        #     )

        # if type(self.rounding_type) == int:
        #     assert self.rounding_type in [1, 2, 3]
        #     self.rounding_type = RoundingType.get_type_from_int(self.rounding_type)

        # if self.rounding_type != RoundingType.SIGMOID and self.sigmoid_slope:
        #     raise ValueError(
        #         "sigmoid_slope can only be set if rounding_type is SIGMOID"
        #     )

        # if self.rounding_type == RoundingType.SIGMOID and not self.sigmoid_slope:
        #     self.sigmoid_slope = 60

        # if (
        #     not self.use_dropout_entropy_in_loss
        #     and self.dropout_entropy_lambda is not None
        # ):
        #     raise ValueError(
        #         "dropout_entropy_lambda is set but use_dropout_entropy_in_loss is False"
        #     )

        # if (
        #     not self.use_dropout_l1_norm_in_loss
        #     and self.dropout_l1_norm_lambda is not None
        # ):
        #     raise ValueError(
        #         "dropout_l1_norm_lambda is set but use_dropout_l1_norm_in_loss is False"
        #     )

        # for attr_name, flag_attr_name in [
        #     ("dropout_entropy_lambda", "use_dropout_entropy_in_loss"),
        #     ("dropout_l1_norm_lambda", "use_dropout_l1_norm_in_loss"),
        # ]:
        #     attr_value = getattr(self, attr_name)
        #     if attr_value is not None:
        #         if type(attr_value) not in [dict, RegularizingLambdaConfig]:
        #             raise ValueError(
        #                 f"{attr_name} must be a dict or RegularizingLambdaConfig"
        #             )

        #         if type(attr_value) == dict:
        #             setattr(self, attr_name, RegularizingLambdaConfig(**attr_value))
        #     else:
        #         if getattr(self, flag_attr_name):
        #             setattr(self, attr_name, RegularizingLambdaConfig(max_lambda=1))


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
        if not hasattr(F, "scaled_dot_product_attention") or isinstance(
            self.dropout_1, LearnedDropout
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

        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        out = self.residual_proj(out)
        out = self.dropout_2(out)
        return out


class LearnedDropout(nn.Module):
    def __init__(self, embed_dim, context_size, config):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size

        self.config = config
        self.module_name = None  # used for logging

        self.head_size = embed_dim // config.n_heads
        self.batch_attn_weights = nn.Linear(
            embed_dim, embed_dim * 3, bias=config.use_bias
        )
        if self.config.return_type in [
            ReturnType.RES_PROJ_MASK_THEN_MASK,
            ReturnType.RES_PROJ_MASK_THEN_NEW_X,
            ReturnType.RES_PROJ_NEW_X_THEN_NEW_X,
        ]:
            self.residual_proj = nn.Linear(embed_dim, embed_dim, bias=config.use_bias)
        else:
            self.residual_proj = None

        if config.dropout_rate > 0:
            self.dropout_1 = nn.Dropout(config.dropout_rate)
            self.dropout_2 = nn.Dropout(config.dropout_rate)
        else:
            self.dropout_1 = None
            self.dropout_2 = None

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(context_size, context_size).view(
                    1, 1, context_size, context_size
                )
            ),
        )

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
        if self.config.softmax_dim != 0:
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            if self.config.softmax_dim != 3:
                causal_attn = F.softmax(causal_attn, dim=-self.config.softmax_dim)
            else:
                assert self.config.softmax_dim == 3
                adjusted_causal_attn = F.softmax(
                    causal_attn.view(causal_attn.size(0), -1), dim=-1
                )
                causal_attn = adjusted_causal_attn.view_as(causal_attn)
        else:
            causal_attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, 0)

        if self.config.dropout_rate > 0:
            causal_attn = self.dropout_1(causal_attn)

        dropout_mask = causal_attn @ v
        if self.config.normalize_by_context_size:
            dropout_mask = dropout_mask * (T**-0.5)
        dropout_mask = dropout_mask.transpose(1, 2).contiguous().view(B, T, C)

        proj_mask = None
        pre_new_x = None
        if self.config.return_type == ReturnType.NO_RES_PROJ_MASK:
            new_x = dropout_mask
        elif self.config.return_type == ReturnType.NO_RES_PROJ_NEW_X:
            new_x = x * dropout_mask
        elif self.config.return_type == ReturnType.RES_PROJ_MASK_THEN_MASK:
            proj_mask = self.residual_proj(dropout_mask)
            new_x = proj_mask
        elif self.config.return_type == ReturnType.RES_PROJ_MASK_THEN_NEW_X:
            proj_mask = self.residual_proj(dropout_mask)
            new_x = x * proj_mask
        elif self.config.return_type == ReturnType.RES_PROJ_NEW_X_THEN_NEW_X:
            pre_new_x = x * dropout_mask
            new_x = self.residual_proj(pre_new_x)
        else:
            raise ValueError("Invalid return type")

        if self.config.dropout_rate > 0:
            new_x = self.dropout_2(new_x)

        if (
            self.training
            and self.config.profile_dropout_mask
            and self.is_last_minibatch
        ):
            # NB: because of gradient accumulation, this will only log the last batch

            with torch.no_grad():
                if self.config.softmax_dim == 2:
                    causal_attn_dim_1_mean = causal_attn.mean(dim=-1)
                    causal_attn_dim_1_mean[:, :, : T // 2] *= -1
                    causal_attn_dim_1_mean_head_mean = causal_attn_dim_1_mean.mean(
                        dim=-2
                    )
                    causal_attn_dim_1_mean_head_std = causal_attn_dim_1_mean.std(dim=-2)
                    causal_attn_dim_1_mean_head_std[:, : T // 2] *= -1
                elif self.config.softmax_dim == 1:
                    causal_attn_dim_2_mean = causal_attn.mean(dim=-2)
                    causal_attn_dim_2_mean[:, :, : T // 2] *= -1
                    causal_attn_dim_2_mean_head_mean = causal_attn_dim_2_mean.mean(
                        dim=-2
                    )
                    causal_attn_dim_2_mean_head_std = causal_attn_dim_2_mean.std(dim=-2)
                    causal_attn_dim_2_mean_head_std[:, : T // 2] *= -1

            log_x = x.detach()
            log_new_x = new_x.detach()
            log_dropout_mask = dropout_mask.detach()
            log_causal_attn = causal_attn.detach()
            log_attn = attn.detach()
            log_proj_mask = proj_mask.detach() if proj_mask is not None else None
            log_pre_new_x = pre_new_x.detach() if pre_new_x is not None else None

            if (
                dropout_mask.dtype == torch.bfloat16
                or causal_attn.dtype == torch.bfloat16
                or dropout_mask.dtype == torch.bfloat16
            ):
                log_x = log_x.half()
                log_new_x = log_new_x.half()
                log_dropout_mask = log_dropout_mask.half()
                log_causal_attn = log_causal_attn.half()
                log_attn = log_attn.half()
                log_proj_mask = (
                    log_proj_mask.half() if log_proj_mask is not None else None
                )
                log_pre_new_x = (
                    log_pre_new_x.half() if log_pre_new_x is not None else None
                )
                if self.config.softmax_dim == 1:
                    causal_attn_dim_2_mean = causal_attn_dim_2_mean.detach().half()
                    causal_attn_dim_2_mean_head_mean = (
                        causal_attn_dim_2_mean_head_mean.detach().half()
                    )
                    causal_attn_dim_2_mean_head_std = (
                        causal_attn_dim_2_mean_head_std.detach().half()
                    )
                elif self.config.softmax_dim == 2:
                    causal_attn_dim_1_mean = causal_attn_dim_1_mean.detach().half()
                    causal_attn_dim_1_mean_head_mean = (
                        causal_attn_dim_1_mean_head_mean.detach().half()
                    )
                    causal_attn_dim_1_mean_head_std = (
                        causal_attn_dim_1_mean_head_std.detach().half()
                    )

            metrics = {
                self.module_name + ".a__input_x": log_x,
                self.module_name + ".l__new_x": log_new_x,
                self.module_name + ".g__mask": log_dropout_mask,
                self.module_name + ".c__causal_attn": log_causal_attn,
                self.module_name + ".b__attn": log_attn,
                self.module_name + ".h__mask_dim_2_std": log_dropout_mask.std(dim=-2),
            }

            if log_proj_mask is not None:
                metrics[self.module_name + ".i__proj_mask"] = log_proj_mask
            if log_pre_new_x is not None:
                metrics[self.module_name + ".k__pre_new_x"] = log_pre_new_x

            if self.config.softmax_dim == 2:
                metrics = {
                    **metrics,
                    self.module_name
                    + ".d__causal_attn_dim_1_mean": causal_attn_dim_1_mean,
                    self.module_name
                    + ".e__causal_attn_dim_1_mean_head_mean": causal_attn_dim_1_mean_head_mean,
                    self.module_name
                    + ".f__causal_attn_dim_1_mean_head_std": causal_attn_dim_1_mean_head_std,
                }
            elif self.config.softmax_dim == 1:
                metrics = {
                    **metrics,
                    self.module_name
                    + ".d__causal_attn_dim_2_mean": causal_attn_dim_2_mean,
                    self.module_name
                    + ".e__causal_attn_dim_2_mean_head_mean": causal_attn_dim_2_mean_head_mean,
                    self.module_name
                    + ".f__causal_attn_dim_2_mean_head_std": causal_attn_dim_2_mean_head_std,
                }
            wandb.log(
                metrics,
                commit=False,
            )
        return new_x


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
    ):
        super().__init__()
        self.use_learned_dropout = use_learned_dropout
        self.learned_dropout_config = config.learned_dropout_config
        if use_learned_dropout:
            self.multi_attn_head = LearnedDropout(
                config.n_embed, config.context_size, config.learned_dropout_config
            )
        else:
            self.multi_attn_head = OptimizedMultiAttentionHead(config)
        self.feed_forward = FeedForward(
            config, use_learned_dropout, should_profile_layer_x
        )
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.ln2 = LayerNorm(config.n_embed, config.bias)

    def forward(self, x):
        if (
            self.use_learned_dropout and self.learned_dropout_config.use_res_add
        ) or not self.use_learned_dropout:
            x = x + self.multi_attn_head(self.ln1(x))
        else:
            x = self.multi_attn_head(self.ln1(x))
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
    def init_from_checkpoint(cls, checkpoint_dict, gradient_accumulation_steps):
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

            additional_loss = 0
            if self.training and self.config.use_learned_dropout and False:
                self.update_stats()
                if self.config.learned_dropout_config.use_dropout_entropy_in_loss:
                    additional_loss += (
                        self.dropout_entropy * self.dropout_entropy_coefficient
                    )
                if self.config.learned_dropout_config.use_dropout_l1_norm_in_loss:
                    additional_loss += (
                        self.dropout_l1_norm * self.dropout_l1_norm_coefficient
                    )

            loss = F.cross_entropy(logits, targets.view(-1)) + additional_loss
        return (logits, loss)

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
            logits, _, _ = self(x[:, -self.config.context_size :], None)
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
