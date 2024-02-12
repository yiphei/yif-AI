import argparse
import logging
import os
import sys
import wandb
from dataclasses import asdict
from datetime import datetime
from contextlib import nullcontext
import math
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
    BATCH_SIZE = config_dict["batch_size"] # this will be scaled by GRADIENT_ACCUMULATION_STEPS
    TRAINING_STEPS = config_dict["training_steps"]
    LR = config_dict["lr"]
    EST_INTERVAL = config_dict["est_interval"]
    EST_STEPS = config_dict["est_steps"]
    WEIGHT_DECAY = 1e-1
    BETA1 = 0.9
    BETA2 = 0.95
    DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    COMPILE = True
    DECAY_LR = True
    WARMUP_ITERS = config_dict['warmup_iters']
    LR_DECAY_ITERS = 600000
    MIN_LR = config_dict['min_lr']
    GRADIENT_ACCUMULATION_STEPS = config_dict['gradient_accumulation_steps']

    # From https://github.com/karpathy/nanoGPT/blob/master/train.py
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
    ctx = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == 'float16'))

    wandb.init(
        # set the wandb project where this run will be logged
        project="transformer_dropout",
        config=config_dict,
        mode="online",
    )    

    model = DropoutTransformer(
        MODEL_CONFIG
    ).to(DEVICE)

    # if COMPILE:
    #     print("compiling the model... (takes a ~minute)")
    #     unoptimized_model = model
    #     model = torch.compile(model) # requires PyTorch 2.0
    
    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = model.configure_optimizer(WEIGHT_DECAY, LR, (BETA1, BETA2), DEVICE)

    # learning rate decay scheduler (cosine with warmup). From https://github.com/karpathy/nanoGPT/blob/master/train.py
    def get_lr(training_step):
        # 1) linear warmup for warmup_iters steps
        if training_step < WARMUP_ITERS:
            return LR * training_step / WARMUP_ITERS
        # 2) if it > lr_decay_iters, return min learning rate
        if training_step > LR_DECAY_ITERS:
            return MIN_LR
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (training_step - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return MIN_LR + coeff * (LR - MIN_LR)

    model.train()
    X, Y = get_data_batch(DEVICE, MODEL_CONFIG.context_size, BATCH_SIZE, 'train') # fetch the very first batch
    for step in range(TRAINING_STEPS):

        # determine and set the learning rate for this iteration. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        lr = get_lr(step) if DECAY_LR else LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if step % EST_INTERVAL == 0 and step != (TRAINING_STEPS - 1) and step != 0:
            train_loss, val_loss = estimate_loss(model, EST_STEPS, MODEL_CONFIG.context_size, BATCH_SIZE, DEVICE, ctx)
            wandb.log({
                "est_iter": step,
                "est_train_loss": train_loss,
                "est_val_loss": val_loss,
                "est_lr": lr,
            })

        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            with ctx:
                logits, loss, entropy, dropout_l1_norm = model(X, Y)
                if DEVICE == "cuda" and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss = loss / GRADIENT_ACCUMULATION_STEPS # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_data_batch(DEVICE, MODEL_CONFIG.context_size, BATCH_SIZE, 'train')
            # backward pass, with gradient scaling if training in fp16
            wandb.log({
                "dropout_entropy": entropy,
                "dropout_l1_norm": dropout_l1_norm,
            })
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        lossf = loss.item() * GRADIENT_ACCUMULATION_STEPS
        wandb.log({
            "loss": lossf,
        })

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": asdict(MODEL_CONFIG),
            "itoc": itoc,
        },
        (
            f"transformer_dropout/model_weights/model_{datetime.now().strftime('%H-%M-%S-%d-%m-%y')}.pth"
            if args.is_local
            else os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
        ),
    )
