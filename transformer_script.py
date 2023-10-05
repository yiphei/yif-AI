import torch
import torch.nn as nn
from torch.nn import functional as F

# Data preparation
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}

decoder = lambda x: "".join([itoc[i] for i in x])
encoder = lambda x: [ctoi[c] for c in x]

data = torch.tensor(encoder(text)).long()
data.shape, data.dtype

training_split = int(data.shape[0] * 0.9)
train_data = data[:training_split]
val_data = data[training_split:]

torch.manual_seed(1337)

# HYPERPARAMETERS
batch_size = 64
block_size = 256
n_embed = 384
training_steps = 5000
estimation_interval = 500
estimation_iter = 200
vocab_size = len(chars)
transform_block = 6
lr = 3e-4
dropout = 0.2
n_head = 6

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(0, data.shape[0] - block_size - 1, (batch_size,))
    x = torch.stack([data[idx : idx + block_size] for idx in idxs])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in idxs])
    return x, y

@torch.no_grad()
def estimate_loss(model):
    mean_losses = []
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(estimation_iter)
        for i in range(estimation_iter):
            xb,yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()

        mean_losses.append(losses.mean(dim=0).item())
    model.train()
    return mean_losses
    


class AttentionHead(nn.Module):
    def __init__(self, dim_in, head_size):
        super().__init__()
        self.head_size = head_size
        self.q_layer = nn.Linear(dim_in, head_size, bias = False)
        self.k_layer = nn.Linear(dim_in, head_size, bias=False)
        self.v_layer = nn.Linear(dim_in, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out


class MultiAttentionHead(nn.Module):

    def __init__(self, dim_in, n_heads, head_size):
        super().__init__()
        self.sa_heads = nn.ModuleList([AttentionHead(dim_in, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.sa_heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_in * 4),
            nn.ReLU(),
            nn.Linear(dim_in * 4, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):

    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_multi_head = MultiAttentionHead(n_embed, n_heads, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_multi_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.transformer_blocks = nn.Sequential( *[TransformerBlock(n_embed, n_head) for _ in range(transform_block)])
        self.ln = nn.LayerNorm(n_embed)
        self.output_layer = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(torch.arange(x.shape[1]))
        final_embed = token_embed + pos_embed
        out = self.transformer_blocks(final_embed)
        out = self.ln(out)
        logits = self.output_layer(out)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(x[:,-block_size:], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x


m = BigramLanguageModel()
optimizer = torch.optim.Adam(m.parameters(), lr=lr)

for steps in range(training_steps):

    if steps % estimation_interval == 0:
        train_loss, val_loss = estimate_loss(m)
        print(f"Train loss: {train_loss}, Val loss: {val_loss}")

    xb,yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

generation = m.generate(torch.tensor([[0]]), 400)
print(decoder(generation[0].tolist()))