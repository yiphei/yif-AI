import argparse
import logging
import os
import sys
import wandb

import torch
from model import DropoutTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train_file", type=str)
    parser.add_argument(
        "--batch_size", type=int, default=70, help="Training batch size."
    )
    parser.add_argument(
        "--block_size", type=int, default=256, help="Block size for sequences."
    )
    parser.add_argument("--n_embed", type=int, default=384, help="Embedding size.")
    parser.add_argument(
        "--training_steps", type=int, default=5000, help="Training steps."
    )
    parser.add_argument(
        "--est_interval", type=int, default=500, help="Estimation interval."
    )
    parser.add_argument("--est_steps", type=int, default=200, help="Estimation steps.")
    parser.add_argument(
        "--transform_blocks", type=int, default=6, help="Transform blocks."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--n_head", type=int, default=6, help="Number of heads.")
    parser.add_argument("--is_local", type=bool, default=False, help="Number of heads.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    logger = logging.getLogger()
    logger.info("Starting training script.")

    args = parse_arguments()

    wandb.init(
        # set the wandb project where this run will be logged
        project="transformer_dropout",
    )    

    # Load and prepare training data
    training_data_file_path = os.path.join(args.train, args.train_file)

    with open(training_data_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for i, c in enumerate(chars)}

    encoder = lambda x: [ctoi[c] for c in x]

    data = torch.tensor(encoder(text)).long()

    training_split = int(data.shape[0] * 0.9)
    train_data = data[:training_split]
    val_data = data[training_split:]

    torch.manual_seed(1337)

    # HYPERPARAMETERS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = args.batch_size
    BLOCK_SIZE = args.block_size
    N_EMBED = args.n_embed
    TRAINING_STEPS = args.training_steps
    EST_INTERVAL = args.est_interval
    EST_STEPS = args.est_steps
    TOKEN_SIZE = len(chars)
    TRANSFORM_BLOCKS = args.transform_blocks
    LR = args.lr
    N_HEAD = args.n_head

    def get_data_batch(split="train"):
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
            losses = torch.zeros(EST_STEPS, device=device)
            for i in range(EST_STEPS):
                xb, yb = get_data_batch(split)
                _, loss = model(xb, yb)
                if device == "cuda" and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                losses[i] = loss

            mean_losses.append(losses.mean().item())
        model.train()
        return mean_losses

    model = DropoutTransformer(
        TOKEN_SIZE, N_EMBED, BLOCK_SIZE, N_HEAD, TRANSFORM_BLOCKS, device
    ).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for steps in range(TRAINING_STEPS):
        if steps % EST_INTERVAL == 0 and steps != (TRAINING_STEPS - 1):
            train_loss, val_loss = estimate_loss(model)
            logger.info(f"Train loss: {train_loss}, Val loss: {val_loss}")

        xb, yb = get_data_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda" and torch.cuda.device_count() > 1:
            loss = loss.mean()
        
        wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()

    if device == "cuda" and torch.cuda.device_count() > 1:
        loss = loss.mean()
    logger.info(loss.item())
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": {
                "block_size": args.block_size,
                "n_embed": args.n_embed,
                "token_size": len(chars),
                "transform_blocks": args.transform_blocks,
                "n_head": args.n_head,
            },
            "itoc": itoc,
        },
        (
            "model.pth"
            if args.is_local
            else os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
        ),
    )
