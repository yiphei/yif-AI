import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import logging
import os
import sys

class OptimizedMultiAttentionHead(nn.Module):

    def __init__(self, dim_in, n_heads, head_size, dropout, block_size):
        super().__init__()
        self.head_size = head_size
        self.q_weight = nn.Parameter(torch.randn(n_heads, dim_in, head_size)  * 0.02)
        self.k_weight = nn.Parameter(torch.randn(n_heads, dim_in, head_size) * 0.02)
        self.v_weight = nn.Parameter(torch.randn(n_heads, dim_in, head_size)* 0.02)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape # B, T, C
        x_expanded = x.unsqueeze(1) # B, 1, T, C
        q = x_expanded @ self.q_weight # B,H,T,C @ H,C,S ->B, H, T, S
        k = x_expanded @ self.k_weight # B,H,T,C @ H,C,S ->B, H, T, S
        v = x_expanded @ self.v_weight # B,H,T,C @ H,C,S ->B, H, T, S

        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5 # B,H,T,S @ B,H,S,T ->B, H, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # B,H,T,T @ B,H,T,S -> B,H,T,S
        B,H,T,S = out.shape
        out = out.permute(0, 2, 1, 3).reshape(B, T, S*H) # B,H,T,S -> B,T,H,S -> B,T,H*S
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, dim_in, dropout):
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

    def __init__(self, n_embed, n_heads, dropout, block_size):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_multi_head = OptimizedMultiAttentionHead(n_embed, n_heads, head_size, dropout, block_size)
        self.feed_forward = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_multi_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, token_size, n_embed, block_size, n_head, transform_blocks, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.token_embedding = nn.Embedding(token_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.transformer_blocks = nn.Sequential( *[TransformerBlock(n_embed, n_head, dropout, block_size) for _ in range(transform_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        self.output_layer = nn.Linear(n_embed, token_size)     
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(torch.arange(x.shape[1], device=self.device))
        embed = token_embed + pos_embed
        out = self.transformer_blocks(embed)
        out = self.ln(out)
        logits = self.output_layer(out)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(x[:,-self.block_size:], None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_t), dim=1)
        return x



def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for the custom model.")
        
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train_file', type=str)

    # Add any other arguments you'd like to customize based on your script's parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--block_size', type=int, default=256, help='Block size for sequences.')
    parser.add_argument('--n_embed', type=int, default=384, help='Embedding size.')
    parser.add_argument('--training_steps', type=int, default=5000, help='Training steps.')
    parser.add_argument('--est_interval', type=int, default=500, help='Estimation interval.')
    parser.add_argument('--est_steps', type=int, default=200, help='Estimation steps.')
    parser.add_argument('--transform_blocks', type=int, default=6, help='Transform blocks.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout.')
    parser.add_argument('--n_head', type=int, default=6, help='Number of heads.')
    parser.add_argument('--is_local', type=bool, default=False, help='Number of heads.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
    logger = logging.getLogger()
    logger.info("Starting training script.")

    args = parse_arguments()

    train_file_path = os.path.join(args.train, args.train_file)

    # Data preparation
    with open(train_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for i, c in enumerate(chars)}

    encoder = lambda x: [ctoi[c] for c in x]

    data = torch.tensor(encoder(text)).long()
    data.shape, data.dtype

    training_split = int(data.shape[0] * 0.9)
    train_data = data[:training_split]
    val_data = data[training_split:]

    torch.manual_seed(1337)

    # HYPERPARAMETERS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = args.batch_size
    BLOCK_SIZE = args.block_size
    N_EMBED = args.n_embed
    TRAINING_STEPS = args.training_steps
    EST_INTERVAL = args.est_interval
    EST_STEPS = args.est_steps
    TOKEN_SIZE = len(chars)
    TRANSFORM_BLOCKS = args.transform_blocks
    LR = args.lr
    DROPOUT = args.dropout
    N_HEAD = args.n_head

    def get_batch(split="train"):
        data = train_data if split == "train" else val_data
        idxs = torch.randint(0, data.shape[0] - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([data[idx : idx + BLOCK_SIZE] for idx in idxs])
        y = torch.stack([data[idx + 1 : idx + BLOCK_SIZE + 1] for idx in idxs])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(model):
        mean_losses = []
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(EST_STEPS, device = device)
            for i in range(EST_STEPS):
                xb,yb = get_batch(split)
                _, loss = model(xb, yb)
                if device == "cuda" and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                losses[i] = loss

            mean_losses.append(losses.mean().item())
        model.train()
        return mean_losses
        
    model = Transformer(TOKEN_SIZE, N_EMBED, BLOCK_SIZE, N_HEAD, TRANSFORM_BLOCKS, DROPOUT, device).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        logger.info("YIFEI YANNNNNN YAYYYY")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for steps in range(TRAINING_STEPS):
        if steps % EST_INTERVAL == 0 and steps != (TRAINING_STEPS - 1):
            train_loss, val_loss = estimate_loss(model)
            logger.info(f"Train loss: {train_loss}, Val loss: {val_loss}")

        xb,yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda" and torch.cuda.device_count() > 1:
            loss = loss.mean()
        loss.backward()
        optimizer.step()

    if device == "cuda" and torch.cuda.device_count() > 1:
        loss = loss.mean()
    logger.info(loss.item())
    torch.save({"state_dict": model.state_dict(),
                "hyperparameters":
                {
                    "block_size": args.block_size,
                    "n_embed": args.n_embed,
                    "token_size": len(chars),
                    "transform_blocks": args.transform_blocks,
                    "dropout": args.dropout,
                    "n_head": args.n_head,
                },
                "itoc": itoc,
                }, 'model.pth' if args.is_local else os.path.join(os.environ['SM_MODEL_DIR'], 'model.pth'))