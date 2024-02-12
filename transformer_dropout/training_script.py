import argparse
import logging
import os
import sys
import wandb
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from contextlib import nullcontext
import math
import torch
from model import DropoutTransformer, ModelConfig


def require_prop_exception():
    raise ValueError("Missing required property")

@dataclass
class TrainConfig:
    DEVICE: str = field(default_factory= lambda: "cuda" if torch.cuda.is_available() else "cpu")
    MODEL_CONFIG: ModelConfig = field(default_factory= require_prop_exception)
    # Training
    BATCH_SIZE: int = field(default_factory= require_prop_exception) # this will be scaled by GRADIENT_ACCUMULATION_STEPS
    TRAIN_STEPS: int = field(default_factory= require_prop_exception)
    GRADIENT_ACCUMULATION_STEPS: int = field(default_factory= require_prop_exception) # used to simulate large batches
    # Optimizer
    LR: float = field(default= 6e-4) # max learning rate
    WEIGHT_DECAY: float = field(default= 1e-1)
    BETA1: float = field(default= 0.9)
    BETA2: float = field(default= 0.95)
    DECAY_LR: bool = True
    WARMUP_ITERS: int = field(default_factory= require_prop_exception)
    LR_DECAY_ITERS: int = field(default_factory= require_prop_exception)
    MIN_LR: float = field(default= 6e-5)
    # Estimation
    EST_INTERVAL: int = field(default_factory= require_prop_exception)
    EST_STEPS: int = field(default_factory= require_prop_exception)
    # Other
    DTYPE: str = field(default_factory= lambda:'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
    COMPILE: bool = False

    @classmethod
    def create_from_config_file(cls, config_file: str, alphabet_size: int):
        config_dict = {}
        with open(config_file, 'r') as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('__')}

        model_config_props = [f.name.upper() for f in fields(ModelConfig)]
        model_config_dict = {k.lower():v for k,v in config_dict.items() if k in model_config_props}
        model_config_dict['alphabet_size'] = alphabet_size
        model_config = ModelConfig(**model_config_dict)

        config_dict = {k:v for k,v in config_dict.items() if k not in model_config_props}
        config_dict['MODEL_CONFIG'] = model_config
        return cls(**config_dict)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--is_local", type=bool, default=True)
    args = parser.parse_args()
    return args


def get_data_batch(device, context_size, batch_size, split="train"):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(0, data.shape[0] - context_size - 1, (batch_size,))
    x = torch.stack([data[idx : idx + context_size] for idx in idxs])
    y = torch.stack([data[idx + 1 : idx + context_size + 1] for idx in idxs])

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # from https://github.com/karpathy/nanoGPT/blob/master/train.py
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, est_steps, context_size, batch_size, device, ctx):
    mean_losses = []
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(est_steps, device=device)
        for i in range(est_steps):
            xb, yb = get_data_batch(device, context_size, batch_size, split)
            with ctx:
                _, loss, _, _ = model(xb, yb)
            if device == "cuda" and torch.cuda.device_count() > 1:
                loss = loss.mean()
            losses[i] = loss

        mean_losses.append(losses.mean().item())
    model.train()
    return mean_losses

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

    TRAIN_CONFIG = TrainConfig.create_from_config_file(args.config_file, len(chars))

    # From https://github.com/karpathy/nanoGPT/blob/master/train.py
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[TRAIN_CONFIG.DTYPE]
    ctx = nullcontext() if TRAIN_CONFIG.DEVICE == 'cpu' else torch.amp.autocast(device_type=TRAIN_CONFIG.DEVICE, dtype=ptdtype)

    model = DropoutTransformer(
        TRAIN_CONFIG.MODEL_CONFIG
    ).to(TRAIN_CONFIG.DEVICE)

    scaler = torch.cuda.amp.GradScaler(enabled=(TRAIN_CONFIG.DTYPE == 'float16'))

    # if COMPILE:
    #     print("compiling the model... (takes a ~minute)")
    #     unoptimized_model = model
    #     model = torch.compile(model) # requires PyTorch 2.0
    
    if TRAIN_CONFIG.DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = model.configure_optimizer(TRAIN_CONFIG.WEIGHT_DECAY, TRAIN_CONFIG.LR, (TRAIN_CONFIG.BETA1, TRAIN_CONFIG.BETA2), TRAIN_CONFIG.DEVICE)

    # learning rate decay scheduler (cosine with warmup). From https://github.com/karpathy/nanoGPT/blob/master/train.py
    def get_lr(training_step):
        # 1) linear warmup for warmup_iters steps
        if training_step < TRAIN_CONFIG.WARMUP_ITERS:
            return TRAIN_CONFIG.LR * training_step / TRAIN_CONFIG.WARMUP_ITERS
        # 2) if it > lr_decay_iters, return min learning rate
        if training_step > TRAIN_CONFIG.LR_DECAY_ITERS:
            return TRAIN_CONFIG.MIN_LR
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (training_step - TRAIN_CONFIG.WARMUP_ITERS) / (TRAIN_CONFIG.LR_DECAY_ITERS - TRAIN_CONFIG.WARMUP_ITERS)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return TRAIN_CONFIG.MIN_LR + coeff * (TRAIN_CONFIG.LR - TRAIN_CONFIG.MIN_LR)

    wandb.init(
        # set the wandb project where this run will be logged
        project="transformer_dropout",
        config=asdict(TRAIN_CONFIG),
        mode="online",
    )    
    model.train()
    X, Y = get_data_batch(TRAIN_CONFIG.DEVICE, TRAIN_CONFIG.MODEL_CONFIG.context_size, TRAIN_CONFIG.BATCH_SIZE, 'train') # fetch the very first batch
    for step in range(TRAIN_CONFIG.TRAIN_STEPS):

        # determine and set the learning rate for this iteration. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        lr = get_lr(step) if TRAIN_CONFIG.DECAY_LR else TRAIN_CONFIG.LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if step % TRAIN_CONFIG.EST_INTERVAL == 0 and step != (TRAIN_CONFIG.TRAIN_STEPS - 1) and step != 0:
            train_loss, val_loss = estimate_loss(model, TRAIN_CONFIG.EST_STEPS, TRAIN_CONFIG.MODEL_CONFIG.context_size, TRAIN_CONFIG.BATCH_SIZE, TRAIN_CONFIG.DEVICE, ctx)
            wandb.log({
                "est_iter": step,
                "est_train_loss": train_loss,
                "est_val_loss": val_loss,
                "est_lr": lr,
            })

        running_loss = 0
        running_entropy = 0
        running_l1_norm = 0
        for micro_step in range(TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS):
            with ctx:
                logits, loss, entropy, dropout_l1_norm = model(X, Y)
                if TRAIN_CONFIG.DEVICE == "cuda" and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss = loss / TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS # scale the loss to account for gradient accumulation
                running_loss += loss.item()
                running_entropy += entropy.item() / TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS
                running_l1_norm += dropout_l1_norm.item() / TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_data_batch(TRAIN_CONFIG.DEVICE, TRAIN_CONFIG.MODEL_CONFIG.context_size, TRAIN_CONFIG.BATCH_SIZE, 'train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        wandb.log({
            "iter": step,
            "loss": running_loss,
            "dropout_entropy": running_entropy,
            "dropout_l1_norm": running_l1_norm,
        })

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": asdict(TRAIN_CONFIG.MODEL_CONFIG),
            "itoc": itoc,
        },
        (
            f"transformer_dropout/model_weights/model_{datetime.now().strftime('%H-%M-%S-%d-%m-%y')}.pth"
            if args.is_local
            else os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
        ),
    )
