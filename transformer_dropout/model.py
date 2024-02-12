import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from dataclasses import dataclass

@dataclass
class ModelConfig:
    context_size: int
    n_embed: int
    alphabet_size: int
    n_layer: int
    n_head: int
    bias: bool = False
    use_flash: bool = False

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

        self.batch_attn_weights = nn.Linear(self.dim_in, self.dim_in*3, bias = config.bias)
        self.dropout_1 = LearnedDropout(config.context_size)
        self.residual_proj = nn.Linear(self.dim_in, self.dim_in, bias = config.bias)
        self.dropout_2 = LearnedDropout(self.dim_in)

        self.flash = False
        if not hasattr(F, 'scaled_dot_product_attention') or not config.use_flash:
            self.register_buffer("tril", torch.tril(torch.ones(config.context_size, config.context_size).view(1,1,config.context_size, config.context_size)))
        else:
            print("Using flas attention.")
            self.flash = True

    def forward(self, x):
        B, T, C = x.shape  # B, T, C

        q, k, v = self.batch_attn_weights(x).split(self.dim_in, dim=2)
        k = k.view(B,T, self.n_heads, self.head_size).transpose(1,2)
        q = q.view(B,T, self.n_heads, self.head_size).transpose(1,2)
        v = v.view(B,T, self.n_heads, self.head_size).transpose(1,2)

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
            # TODO: add custom dropout here
        else:
            attn = (
            (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        ) # B,H,T,S @ B,H,S,T ->B, H, T, T
            causal_attn = attn.masked_fill(self.tril[:,:,:T,:T] == 0, float("-inf"))
            causal_attn = F.softmax(causal_attn, dim=-1)
            causal_attn = self.dropout_1(causal_attn)
            out = causal_attn @ v  # B,H,T,T @ B,H,T,S -> B,H,T,S

        out = out.transpose(1, 2).contiguous().view(B, T, C) # B,H,T,S -> B,T,H,S -> B,T,C
        out = self.residual_proj(out)
        out = self.dropout_2(out)
        return out


class LearnedDropout(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(dim_in))
        self.B = nn.Parameter(torch.zeros(dim_in))
        self.entropy = None # somehow, this has to be initialized to zero, otherwise it becomes None in get_mean_entropy
        self.l1_norm = None # currently unused

    def forward(self, x):
        dropout_mask = (0.5 * torch.cos(self.A * x + self.B) + 0.5)
        self.entropy = (dropout_mask * -torch.log(dropout_mask + 1e-9)).sum(dim=-1)
        self.l1_norm = dropout_mask.sum()
        return x * dropout_mask

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(config.n_embed * 4, config.n_embed, bias=config.bias)
        self.dropout = LearnedDropout(config.n_embed)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.residual_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.multi_attn_head = OptimizedMultiAttentionHead(
            config
        )
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.ln2 = LayerNorm(config.n_embed, config.bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class DropoutTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_embed)
        self.dropout = LearnedDropout(config.n_embed)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.bias)
        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias = config.bias)

        self.token_embedding.weight = self.output_layer.weight # weight tying

        self.apply(self._init_weights)

        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('residual_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_mean_entropy(self):
        entropy_list = []
        for module in self.modules():
            if isinstance(module, LearnedDropout):
                entropy_list.append(module.entropy.flatten())
        return torch.cat(entropy_list, dim=0).mean()
    
    def print_dropout_params(self):
        for module in self.modules():
            if isinstance(module, LearnedDropout):
                print(module.A, module.B)
                print("----------------")


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
            mean_entropy = None
            logits = self.output_layer(out[:,[-1],:])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            mean_entropy = self.get_mean_entropy()
            loss = F.cross_entropy(logits, targets.view(-1)) + mean_entropy
        return logits, loss, mean_entropy
    
    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        """From https://github.com/karpathy/nanoGPT/blob/master/model.py"""
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _, _ = self(x[:, -self.context_size :], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x
