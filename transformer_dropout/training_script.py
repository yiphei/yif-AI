import argparse
import logging
import os
import sys
import wandb
from dataclasses import asdict
from datetime import datetime

import torch
from model import DropoutTransformer, ModelConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--is_local", type=bool, default=True)

    # Model config
    parser.add_argument(
        "--n_layer", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--bias", type=int)
    parser.add_argument(
        "--context_size", type=int
    )
    parser.add_argument("--n_embed", type=int)
    
    # Train config
    parser.add_argument(
        "--batch_size", type=int
    )
    parser.add_argument(
        "--training_steps", type=int
    )
    parser.add_argument("--lr", type=float)


    # Estimation config
    parser.add_argument(
        "--est_interval", type=int
    )
    parser.add_argument("--est_steps", type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    if args.config_file is not None:
        assert all([v is None for k,v in args_dict.items() if k not in ["train", "train_file", "config_file", "is_local"]])
    else:
        assert all([v is not None for k,v in args_dict.items() if k not in ["train", "train_file", "config_file", "is_local"]])

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    logger = logging.getLogger()
    logger.info("Starting training script.")

    args = parse_arguments()

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

    config_dict = {}
    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('__')}
    else:
        config_dict = {k:v for k,v in vars(args).items() if k not in ["train", "train_file", "config_file", "is_local"]}
    config_dict['alphabet_size'] = len(chars)

    # HYPERPARAMETERS
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_CONFIG = ModelConfig(**{k:v for k,v in config_dict.items() if k in ModelConfig.__annotations__}) 
    BATCH_SIZE = config_dict["batch_size"]
    TRAINING_STEPS = config_dict["training_steps"]
    LR = config_dict["lr"]
    EST_INTERVAL = config_dict["est_interval"]
    EST_STEPS = config_dict["est_steps"]
    WEIGHT_DECAY = 1e-1
    BETA1 = 0.9
    BETA2 = 0.95

    def get_data_batch(split="train"):
        data = train_data if split == "train" else val_data
        idxs = torch.randint(0, data.shape[0] - MODEL_CONFIG.context_size - 1, (BATCH_SIZE,))
        x = torch.stack([data[idx : idx + MODEL_CONFIG.context_size] for idx in idxs])
        y = torch.stack([data[idx + 1 : idx + MODEL_CONFIG.context_size + 1] for idx in idxs])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y

    @torch.no_grad()
    def estimate_loss(model):
        mean_losses = []
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(EST_STEPS, device=DEVICE)
            for i in range(EST_STEPS):
                xb, yb = get_data_batch(split)
                _, loss = model(xb, yb)
                if DEVICE == "cuda" and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                losses[i] = loss

            mean_losses.append(losses.mean().item())
        model.train()
        return mean_losses

    wandb.init(
        # set the wandb project where this run will be logged
        project="transformer_dropout",
        config=config_dict,
        mode="online",
    )    

    model = DropoutTransformer(
        MODEL_CONFIG
    ).to(DEVICE)
    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = model.configure_optimizer(WEIGHT_DECAY, LR, (BETA1, BETA2), DEVICE)

    model.train()
    for steps in range(TRAINING_STEPS):
        if steps % EST_INTERVAL == 0 and steps != (TRAINING_STEPS - 1):
            train_loss, val_loss = estimate_loss(model)
            logger.info(f"Train loss: {train_loss}, Val loss: {val_loss}")

        xb, yb = get_data_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        if DEVICE == "cuda" and torch.cuda.device_count() > 1:
            loss = loss.mean()
        
        wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()

    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        loss = loss.mean()
    logger.info(loss.item())
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": asdict(MODEL_CONFIG),
            "itoc": itoc,
        },
        (
            f"model_weights/model_{datetime.now().strftime('%H-%M-%S-%d-%m-%y')}.pth"
            if args.is_local
            else os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
        ),
    )
