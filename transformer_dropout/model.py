import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LayerNorm(nn.Module):
    """From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class OptimizedMultiAttentionHead(nn.Module):
    def __init__(self, dim_in, n_heads, context_size, bias=False):
        super().__init__()

        assert dim_in % n_heads == 0
        self.dim_in = dim_in
        self.head_size = dim_in // n_heads
        self.n_heads = n_heads

        self.batch_attn_weights = nn.Linear(dim_in, dim_in*3, bias = bias)
        self.dropout_1 = LearnedDropout(context_size)
        self.residual_proj = nn.Linear(dim_in, dim_in, bias = bias)
        self.dropout_2 = LearnedDropout(dim_in)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size).view(1,1,context_size, context_size)))       

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
        self.entropy = None
        self.l1_norm = None # currently unused

    def forward(self, x):
        dropout_mask = (0.5 * torch.cos(self.A * x + self.B) + 0.5)
        self.entropy = -(dropout_mask * torch.log(dropout_mask + 1e-9)).sum()
        self.l1_norm = dropout_mask.sum()
        return x * dropout_mask

class FeedForward(nn.Module):
    def __init__(self, dim_in, bias=False):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in * 4, bias=bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(dim_in * 4, dim_in, bias=bias)
        self.dropout = LearnedDropout(dim_in)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.residual_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, context_size):
        super().__init__()
        self.multi_attn_head = OptimizedMultiAttentionHead(
            n_embed, n_heads, context_size
        )
        self.feed_forward = FeedForward(n_embed)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class DropoutTransformer(nn.Module):
    def __init__(
        self,
        token_size,
        n_embed,
        context_size,
        n_head,
        transform_blocks,
        device,
    ):
        super().__init__()
        self.context_size = context_size
        self.device = device

        self.token_embedding = nn.Embedding(token_size, n_embed)
        self.positional_embedding = nn.Embedding(context_size, n_embed)
        self.dropout = LearnedDropout(n_embed)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(n_embed, n_head, context_size)
                for _ in range(transform_blocks)
            ]
        )
        self.ln = LayerNorm(n_embed)
        self.output_layer = nn.Linear(n_embed, token_size, bias = False)

        self.token_embedding.weight = self.output_layer.weight # weight tying

        self.apply(self._init_weights)

        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('residual_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * transform_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_mean_entropy(self):
        total_entropy = 0
        cts = 0
        for module in self.modules():
            if isinstance(module, LearnedDropout):
                cts +=1
                total_entropy += module.entropy
        return total_entropy / cts

    def forward(self, x, targets=None):
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(
            torch.arange(x.shape[1], dtype=torch.long, device=self.device)
        )
        embed = token_embed + pos_embed
        embed = self.dropout(embed)
        out = self.transformer_blocks(embed)
        out = self.ln(out)
        if targets is None:
            loss = None
            logits = self.output_layer(out[:,[-1],:])
        else:
            logits = self.output_layer(out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1)) + self.get_mean_entropy()
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(x[:, -self.context_size :], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x
