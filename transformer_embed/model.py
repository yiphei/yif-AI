import inspect
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class NewOutputLayerConfig:
    subtract_out_pos_embed: bool
    use_cross_entropy_loss: bool


@dataclass
class ModelConfig:
    context_size: int
    n_embed: int
    n_layer: int
    n_head: int
    use_new_output_layer: bool
    use_final_ln_layer: bool
    dropout_rate: float
    new_output_layer_config: Optional[NewOutputLayerConfig] = None
    alphabet_size: Optional[int] = field(default=None)
    bias: bool = False

    def __post_init__(self):
        if self.use_new_output_layer and self.new_output_layer_config is None:
            raise ValueError(
                "new_output_layer_config must be provided when using new output layer"
            )

        if (
            self.new_output_layer_config is not None
            and type(self.new_output_layer_config) == dict
        ):
            self.new_output_layer_config = NewOutputLayerConfig(
                **self.new_output_layer_config
            )


class LayerNorm(nn.Module):
    """From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, dim_in, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_in)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class OutputLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

    def forward(self, x):
        new_mse = (torch.cdist(x, self.weight, p=2) ** 2) / self.weight.shape[1]
        return new_mse


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

        self.dropout_1 = nn.Dropout(config.dropout_rate)
        self.dropout_2 = nn.Dropout(config.dropout_rate)

        self.use_flash = False
        if not hasattr(F, "scaled_dot_product_attention"):
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


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(
            config.n_embed * 4, config.n_embed, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.residual_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.multi_attn_head = OptimizedMultiAttentionHead(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.ln2 = LayerNorm(config.n_embed, config.bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class EmbedTransformer(nn.Module):

    model_config_cls = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(
            config.context_size + 1, config.n_embed
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln = (
            LayerNorm(config.n_embed, config.bias)
            if self.config.use_final_ln_layer
            else None
        )
        if config.use_new_output_layer:
            self.output_layer = OutputLayer(config.alphabet_size, config.n_embed)
        else:
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def init_from_checkpoint(cls, checkpoint_dict):
        model_config = ModelConfig(**checkpoint_dict["model_config"])
        # create the model
        model = cls(model_config)
        state_dict = checkpoint_dict["model"]

        # This is caused by compiling the model. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        return model

    def get_accuracy(self, logits, targets):
        if (
            self.config.use_new_output_layer
            and not self.config.new_output_layer_config.use_cross_entropy_loss
        ):
            return (
                (logits.min(dim=-1).indices.view(-1) != targets.view(-1)).float().mean()
            )
        else:
            probs = F.softmax(logits, dim=-1)
            return (
                (probs.max(dim=-1).indices.view(-1) != targets.view(-1)).float().mean()
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
        if self.config.use_final_ln_layer:
            out = self.ln(out)

        if targets is None:
            loss = None
            # TODO: handle new output layer
            logits = self.output_layer(out[:, [-1], :])
        else:
            if (
                self.config.use_new_output_layer
                and self.config.new_output_layer_config.subtract_out_pos_embed
            ):
                final_pos_embed = self.positional_embedding(
                    torch.arange(1, x.shape[1] + 1, dtype=torch.long, device=device)
                )
                out = out - final_pos_embed
            logits = self.output_layer(out)
            if (
                self.config.use_new_output_layer
                and self.config.new_output_layer_config.use_cross_entropy_loss
            ) or not self.config.use_new_output_layer:
                if self.config.use_new_output_layer:
                    logits = -logits
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                loss = F.cross_entropy(logits, targets.view(-1))
            else:
                targets_exp = targets.unsqueeze(
                    2
                )  # Adds a third dimension for compatibility with gather
                loss = torch.gather(logits, 2, targets_exp).squeeze(2).view(-1).mean()
        return (
            logits,
            loss,
            None,
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
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        adam_optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        def change_lr(self, lr):
            for param_group in self.param_groups:
                param_group["lr"] = lr

        adam_optimizer.change_lr = change_lr.__get__(adam_optimizer)
        return adam_optimizer

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.positional_embedding.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, x, max_tokens):
        raise NotImplementedError
