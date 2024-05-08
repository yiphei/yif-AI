import inspect
from typing import List, Type

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, dim_in, use_bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_in)) if use_bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiAttentionHead(nn.Module):
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
        self.residual_proj = nn.Linear(self.dim_in, self.dim_in, bias=use_bias)

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
        new_x = self.residual_proj(new_x)
        new_x = self.dropout_2(new_x)
        return new_x


class FeedForward(nn.Module):
    def __init__(self, dim_in, use_bias, dropout_rate=0):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in * 4, bias=use_bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(dim_in * 4, dim_in, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.residual_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        n_head,
        use_bias,
        context_size,
        dropout_rate=0,
        use_flash=True,
    ):
        super().__init__()
        self.multi_attn_head = MultiAttentionHead(
            dim_in, n_head, use_bias, context_size, dropout_rate, use_flash
        )
        self.feed_forward = FeedForward(dim_in, use_bias, dropout_rate)
        self.ln1 = LayerNorm(dim_in, use_bias)
        self.ln2 = LayerNorm(dim_in, use_bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


RUNNING_STAT_PREFIX = "running_"


class BaseModel(nn.Module):
    model_config_cls: Type
    extra_stats: List[str] = []

    def __init__(self, gradient_accumulation_steps=None, is_master_process=True):
        super().__init__()
        # these variables enable profiling
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # these two variables are set by the context manager in the training script
        self.training_step = None
        self.is_last_minibatch = False
        self.is_master_process = is_master_process

        # It's faster to use keep stats on the same device, hence the buffer registration
        for stat in self.extra_stats:
            self.register_buffer(stat, torch.empty(0), persistent=False)
            self.register_buffer(
                RUNNING_STAT_PREFIX + stat, torch.empty(0), persistent=False
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _update_running_stats(self):
        if self.training and self.is_master_process:
            for stat in self.extra_stats:
                current_running_stat = getattr(self, RUNNING_STAT_PREFIX + stat)
                current_stat = getattr(self, stat)
                if current_stat.numel() == 0:
                    continue
                if current_running_stat.numel() == 0:
                    current_running_stat = (
                        current_stat.clone() / self.gradient_accumulation_steps
                    )
                else:
                    current_running_stat += current_stat / self.gradient_accumulation_steps

                setattr(self, RUNNING_STAT_PREFIX + stat, current_running_stat)

    def reset_running_stats(self):
        if self.is_master_process:
            for stat in self.extra_stats:
                setattr(self, RUNNING_STAT_PREFIX + stat, torch.empty(0))

    def dump_extra_stats(self):
        return {
            stat: getattr(self, RUNNING_STAT_PREFIX + stat).item()
            for stat in self.extra_stats
            if getattr(self, RUNNING_STAT_PREFIX + stat).numel() != 0
        }

    def update_is_last_minibatch(self, new_val):
        # this is called by the context manager in the training script
        if self.training and self.is_master_process and new_val != self.is_last_minibatch:
            self.is_last_minibatch = new_val
            for module in self.modules():
                module.is_last_minibatch = new_val

    @classmethod
    def init_from_checkpoint(cls, checkpoint_dict, gradient_accumulation_steps=None, is_master_process=True):
        model_config = cls.model_config_cls(**checkpoint_dict["model_config"])
        model = cls(model_config, gradient_accumulation_steps, is_master_process)
        state_dict = checkpoint_dict["model"]

        # This is caused by compiling the model. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        return model

    def forward(self):
        raise NotImplementedError

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

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        adam_optimizer = torch.optim.AdamW(
            adamw_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return OptimizerWrapper(adam_optimizer, None)

    def get_num_params(self, exclude_positional_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_positional_embedding:
            n_params -= self.positional_embedding.weight.numel()
        return n_params

    def get_accuracy_loss(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        return (probs.max(dim=-1).indices.view(-1) != targets.view(-1)).float().mean()

    def estimate_mfu(self):
        raise NotImplementedError

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
