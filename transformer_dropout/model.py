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


@dataclass
class LearnedDropoutConfig:
    use_dropout_entropy_in_loss: bool
    use_dropout_l1_norm_in_loss: bool
    use_bias: bool
    softmax_dim: int = 2
    rounding_type: Optional[Union[RoundingType, int]] = None
    sigmoid_slope: Optional[float] = None
    shift_init: float = torch.pi / 2
    n_heads: int = 1
    use_canonical_entropy: bool = False
    use_detached_x_in_dropout_mask: bool = False
    dropout_entropy_lambda: Optional[RegularizingLambdaConfig] = field(default=None)
    dropout_l1_norm_lambda: Optional[RegularizingLambdaConfig] = field(default=None)
    profile_dropout_mask: bool = False

    def __post_init__(self):
        assert 0 <= self.shift_init <= torch.pi
        assert self.softmax_dim in [0, 1, 2]

        if (
            self.use_dropout_entropy_in_loss
            and self.rounding_type
            and self.rounding_type in [2, 3]
        ):
            raise ValueError(
                "rounding_type cannot be 2 or 3 if use_dropout_entropy_in_loss"
            )

        if type(self.rounding_type) == int:
            assert self.rounding_type in [1, 2, 3]
            self.rounding_type = RoundingType.get_type_from_int(self.rounding_type)

        if self.rounding_type != RoundingType.SIGMOID and self.sigmoid_slope:
            raise ValueError(
                "sigmoid_slope can only be set if rounding_type is SIGMOID"
            )

        if self.rounding_type == RoundingType.SIGMOID and not self.sigmoid_slope:
            self.sigmoid_slope = 60

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


@dataclass
class ModelConfig:
    context_size: int
    n_embed: int
    n_layer: int
    n_head: int
    use_learned_dropout: bool
    learned_dropout_layers: int = None
    learned_dropout_config: LearnedDropoutConfig = None
    dropout_rate: Optional[float] = field(default=None)
    alphabet_size: Optional[int] = field(default=None)
    bias: bool = False

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

        if self.use_learned_dropout and not self.learned_dropout_layers:
            raise ValueError(
                "learned_dropout_layers must be set if use_learned_dropout"
            )

        if self.use_learned_dropout and self.learned_dropout_layers:
            assert (
                self.learned_dropout_layers >= 1
                and self.learned_dropout_layers <= self.n_layer
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
    
class BaseDropoutStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dropout_entropy", torch.tensor(float('nan')), persistent=False)
        self.register_buffer("dropout_l1_norm", torch.tensor(float('nan')), persistent=False)
        self.register_buffer("dropout_near_one_percent", torch.tensor(float('nan')), persistent=False)
        self.register_buffer("dropout_near_zero_percent", torch.tensor(float('nan')), persistent=False)
        self.register_buffer("dropout_mask_change_rate_from_prev", torch.tensor(float('nan')), persistent=False)
        self.register_buffer("prev_dropout_mask", None, persistent=False)
        self.register_buffer("active_dropout_mask_percent", torch.tensor(float('nan')), persistent=False)
        self.blacklist = ["prev_dropout_mask"]

class RunningDropoutStats(BaseDropoutStats):
    def __init__(self):
        super().__init__()
        to_add = []
        for name, buffer in self._buffers.items():
            if name not in self.blacklist:
                to_add.append((f"running_{name}", buffer.clone()))

        for name, buffer in to_add:
            self.register_buffer(name, torch.tensor(float('nan')), persistent=False)

    def reset_running(self):
        for name, buffer in self._buffers.items():
            if name.startswith("running_") and name not in self.blacklist:
                buffer.fill_(torch.nan)

    def dump_running(self):
        return {
            "dropout_entropy": self.running_dropout_entropy,
            "dropout_l1_norm": self.running_dropout_l1_norm,
            "active_dropout_mask_percent": self.running_active_dropout_mask_percent,
            "dropout_mask_change_rate_from_prev": self.running_dropout_mask_change_rate_from_prev,
        }

    def update_running(self):
        local_buffers = []

        for name, _ in self._buffers.items():
            if not name.startswith("running_") and name not in self.blacklist:
                local_buffers.append(name)

        for name in local_buffers:
            local_value = getattr(self, name)
            if name in ["dropout_entropy", "dropout_entropy"]:
                local_value = local_value.detach()

            running_update = local_value / self.gradient_accumulation_steps
            curr_running_value = getattr(self, f"running_{name}")
            if curr_running_value.isnan().any():
                setattr(self, f"running_{name}", running_update)
            else:
                setattr(self, f"running_{name}", curr_running_value + running_update)
    
    def update_stats(self):
        values_dict = {}
        for name, buffer in self._buffers.items():
            if not name.startswith("running_") and name not in self.blacklist:
                values_dict[name] = torch.empty(1, device=buffer.device)

        module_idx = 0
        for module in self.modules():
            if isinstance(module, LearnedDropout):
                for name, buffer in module.named_buffers():
                    if name in values_dict:
                        values_dict[name][module_idx] = buffer
                module_idx += 1

        for name, values in values_dict.items():
            setattr(self, name, values.mean())

        self.update_running()

class LearnedDropoutStats(BaseDropoutStats):
    def __init__(self, config):
        super().__init__()
        self.dropout_entropy_context = (
            nullcontext() if config.use_dropout_entropy_in_loss else torch.no_grad()
        )
        self.dropout_l1_norm_context = (
            nullcontext() if config.use_dropout_l1_norm_in_loss else torch.no_grad()
        )
        self.entropy_fn = (
            self.canonical_entropy
            if config.use_canonical_entropy
            else self.alternate_entropy
        )

    def canonical_entropy(self, dropout_mask):
        # the small constant is for numerical stability
        return (dropout_mask * -torch.log2(dropout_mask + 1e-9)).mean()

    def alternate_entropy(self, dropout_mask):
        # the alternate entropy has the peak above 0.5, while the canonical one has
        # it below 0.5. In theory, this should be better for achieving both low entropy
        # and low l1 norm because there is more curvature towards 0.
        return ((dropout_mask - 1) * torch.log2((-dropout_mask + 1) + 1e-9)).mean()

    def update_stats(self, dropout_mask, B, T, C):
        with self.dropout_entropy_context:
            self.dropout_entropy = self.entropy_fn(dropout_mask)
        with self.dropout_l1_norm_context:
            # TODO: change this to a simple sum
            self.dropout_l1_norm = torch.norm(dropout_mask, p=1) / (B * T * C)

        with torch.no_grad():
            self.dropout_near_one_percent = (
                dropout_mask > 0.9
            ).sum() / dropout_mask.numel()
            self.dropout_near_zero_percent = (
                dropout_mask < 0.1
            ).sum() / dropout_mask.numel()
            self.active_dropout_mask_percent = (
                dropout_mask > 0.05
            ).sum() / dropout_mask.numel()

            if self.prev_dropout_mask is not None:
                matching_1s = (dropout_mask >= 0.5) & (self.prev_dropout_mask >= 0.5)
                matching_0s = (dropout_mask < 0.5) & (self.prev_dropout_mask < 0.5)
                self.dropout_mask_change_rate_from_prev = (
                    1
                    - (matching_0s.sum() + matching_1s.sum())
                    / dropout_mask.numel()
                )
            self.prev_dropout_mask = dropout_mask.clone()

class LearnedDropout(LearnedDropoutStats):
    def __init__(self, embed_dim, context_size, config):
        super().__init__(config)
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

    def forward(self, x):
        import wandb

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
            if self.training and self.config.profile_dropout_mask:
                wandb.log(
                    {self.module_name + ".pre-rounding_mask": dropout_mask},
                    commit=False,
                )

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
            self.update_stats(dropout_mask, B, T, C)

        if self.training and self.config.profile_dropout_mask:
            # NB: because of gradient accumulation, this will only log the last batch
            wandb.log({self.module_name + ".mask": dropout_mask}, commit=False)
        return x * dropout_mask


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, use_learned_dropout=False):
        super().__init__()
        self.linear = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(
            config.n_embed * 4, config.n_embed, bias=config.bias
        )
        if config.use_learned_dropout and use_learned_dropout:
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
    def __init__(self, config: ModelConfig, use_learned_dropout=False):
        super().__init__()
        self.multi_attn_head = OptimizedMultiAttentionHead(config)
        self.feed_forward = FeedForward(config, use_learned_dropout)
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.ln2 = LayerNorm(config.n_embed, config.bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class DropoutTransformer(RunningDropoutStats):
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

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_embed)
        if config.use_learned_dropout and False:
            self.dropout = LearnedDropout(config.n_embed, config.learned_dropout_config)
        else:
            self.dropout = nn.Dropout(config.dropout_rate)

        check = config.learned_dropout_layers or 0
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config, (i) >= (config.n_layer - check))
                for i in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.bias)
        self.output_layer = nn.Linear(
            config.n_embed, config.alphabet_size, bias=config.bias
        )

        self.token_embedding.weight = self.output_layer.weight  # weight tying
        self.apply(self._init_weights)

        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("residual_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # maybe there is a better way
        param_to_param_name = {p: n for n, p in self.named_parameters()}
        for module in self.modules():
            if isinstance(module, LearnedDropout):
                module.module_name = ".".join(
                    param_to_param_name[module.batch_attn_weights.weight].split(".")[
                        :-2
                    ]
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def get_annealed_dropout_coefficient(self, lambda_config):
        if (
            not self.config.use_learned_dropout
            or not self.training
            or lambda_config is None
        ):
            return None

        if lambda_config.coefficient is None:
            return lambda_config.max_lambda

        assert self.training_step is not None
        intersect = (
            lambda_config.min_lambda - 1 if lambda_config.min_lambda is not None else -1
        )
        return min(
            np.exp(lambda_config.coefficient * self.training_step) + intersect,
            lambda_config.max_lambda,
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

        (
            mean_dropout_entropy_coefficient,
            mean_dropout_l1_norm_coefficient,
        ) = [None] * 2
        if targets is None:
            loss = None
            logits = self.output_layer(out[:, [-1], :])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            additional_loss = 0
            if self.training and self.config.use_learned_dropout:
                self.update_stats()

                mean_dropout_entropy = self.dropout_entropy
                mean_dropout_l1_norm = self.dropout_l1_norm
                mean_dropout_entropy_coefficient = (
                    self.get_annealed_dropout_coefficient(
                        self.config.learned_dropout_config.dropout_entropy_lambda
                    )
                )
                mean_dropout_l1_norm_coefficient = (
                    self.get_annealed_dropout_coefficient(
                        self.config.learned_dropout_config.dropout_l1_norm_lambda
                    )
                )

                if self.config.learned_dropout_config.use_dropout_entropy_in_loss:
                    additional_loss += (
                        mean_dropout_entropy * mean_dropout_entropy_coefficient
                    )
                if self.config.learned_dropout_config.use_dropout_l1_norm_in_loss:
                    additional_loss += (
                        mean_dropout_l1_norm * mean_dropout_l1_norm_coefficient
                    )

            loss = F.cross_entropy(logits, targets.view(-1)) + additional_loss
        return (
            logits,
            loss,
            (
                mean_dropout_entropy_coefficient,
                mean_dropout_l1_norm_coefficient,
            ),
        )

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

    def estimate_mfu(self, fwdbwd_per_iter, dt, active_dropout_mask_percent):
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
        flops_per_token += (
            (active_dropout_mask_percent)
            * 12
            * self.config.n_embed
            * self.config.context_size
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
