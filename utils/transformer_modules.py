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
    def __init__(self, dim_in, n_head, use_bias, context_size, dropout_rate = 0, use_flash = True):
        super().__init__()
        assert dim_in % n_head == 0
        self.dim_in = dim_in
        self.head_size = dim_in // n_head
        self.n_head = n_head
        self.dropout_rate = dropout_rate

        self.batch_attn_weights = nn.Linear(
            self.dim_in, self.dim_in * 3, bias=use_bias
        )
        self.residual_proj = nn.Linear(self.dim_in, self.dim_in, bias=use_bias)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.using_flash = False
        if (
            not hasattr(F, "scaled_dot_product_attention")
            or not use_flash
        ):
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
    def __init__(
        self,
        dim_in,
        use_bias,
        dropout_rate = 0
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in * 4, bias=use_bias)
        self.gelu = nn.GELU()
        self.residual_proj = nn.Linear(
            dim_in * 4, dim_in, bias=use_bias
        )
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
        dropout_rate = 0,
        use_flash = True,
    ):
        super().__init__()
        self.multi_attn_head = MultiAttentionHead(dim_in, n_head, use_bias, context_size, dropout_rate, use_flash)
        self.feed_forward = FeedForward(
                    dim_in,
                use_bias,
                dropout_rate
        )
        self.ln1 = LayerNorm(dim_in, use_bias)
        self.ln2 = LayerNorm(dim_in, use_bias)

    def forward(self, x):
        x = x + self.multi_attn_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x