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
batch_size = 4
block_size = 8
n_embed = 64
training_steps = 20000
estimation_interval = 1000
estimation_iter = 200
lr = 1e-3

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
    


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(torch.arange(x.shape[1]))
        final_embed = token_embed + pos_embed
        logits = self.lm_head(final_embed)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(x[:,max(x.shape[1]-block_size, 0):], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x


m = BigramLanguageModel(len(chars))
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