import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.transformer_modules import LayerNorm, TransformerBlock, BaseModel
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    context_size: int
    n_embed: int
    n_layer: int
    n_head: int
    dropout_rate: float = 0.0
    alphabet_size: Optional[int] = None
    use_bias: bool = False
    use_flash: bool = True

class TransformerModel(BaseModel):
    modeL_config = ModelConfig

    def __init__(self, config: ModelConfig, gradient_accumulation_steps):
        super().__init__(gradient_accumulation_steps)
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(
            config.context_size, config.n_embed
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                            config.n_embed,
                            config.n_head,
                            config.use_bias,
                            config.context_size,
                            config.dropout_rate,
                            config.use_flash,
                )
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

    def forward(self, x, targets=None):
        device = x.device
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(
            torch.arange(x.shape[1], dtype=torch.long, device=device)
        )
        x = token_embed + pos_embed
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        out = self.ln(x)

        if targets is None:
            loss = None
            logits = self.output_layer(out[:, [-1], :])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss

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
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        return flops_achieved