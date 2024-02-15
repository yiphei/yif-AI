import argparse
import logging
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from distutils.util import strtobool
from enum import Enum

import numpy as np
import torch

# ugly workound to make Sagemaker happy
try:
    # Attempt relative import when running as part of a package
    from .model import DropoutTransformer, ModelConfig
except ImportError:
    # Fallback to absolute import when running as a standalone script (e.g., in SageMaker)
    from model import DropoutTransformer, ModelConfig

from torch.distributed import destroy_process_group, init_process_group

import wandb


# This is a hack to circumvent the dataclass requirement that fields with non-default values must precede those with them
def require_field_exception():
    raise ValueError("Missing required property")


class InitializationType(Enum):
    SCRATCH = "scratch"
    RESUME = "resume"

@dataclass
class TrainConfig:
    DEVICE: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    MODEL_CONFIG: ModelConfig = field(default_factory=require_field_exception)
    # Training
    BATCH_SIZE: int = field(
        default_factory=require_field_exception
    )  # this will be scaled by GRADIENT_ACCUMULATION_STEPS
    TRAIN_STEPS: int = field(default_factory=require_field_exception)
    GRADIENT_ACCUMULATION_STEPS: int = field(
        default_factory=require_field_exception
    )  # used to simulate large batches. Must be a multiple of world_size (i.e. # of GPUs) if using DDP
    # Optimizer
    LR: float = field(default=6e-4)  # max learning rate
    WEIGHT_DECAY: float = field(default=1e-1)
    BETA1: float = field(default=0.9)
    BETA2: float = field(default=0.95)
    DECAY_LR: bool = True
    WARMUP_ITERS: int = field(default_factory=require_field_exception)
    LR_DECAY_ITERS: int = field(default_factory=require_field_exception)
    MIN_LR: float = field(default=6e-5)
    # Estimation
    EST_INTERVAL: int = field(default_factory=require_field_exception)
    EST_STEPS: int = field(default_factory=require_field_exception)
    # Other
    DTYPE: str = field(
        default_factory=lambda: (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    )
    COMPILE: bool = True
    USE_DP: bool = False  # DataParallel
    USE_DDP: bool = True  # DistributedDataParallel

    def __post_init__(self):
        if self.USE_DDP and self.USE_DP:
            raise ValueError("cannot have both USE_DDP and USE_DP set to True")
        if self.TRAIN_STEPS <= self.EST_INTERVAL:
            raise ValueError("EST_INTERVAL must be less than TRAIN_STEPS")
        if self.MIN_LR >= self.LR:
            raise ValueError("MIN_LR must be less than LR")
        if self.WARMUP_ITERS >= self.TRAIN_STEPS:
            raise ValueError("WARMUP_ITERS must be less than TRAIN_STEPS")
        if self.EST_STEPS >= self.TRAIN_STEPS:
            raise ValueError("EST_STEPS must be less than TRAIN_STEPS")
        if self.LR_DECAY_ITERS > self.TRAIN_STEPS:
            raise ValueError("LR_DECAY_ITERS must be less than TRAIN_STEPS")
        if self.WARMUP_ITERS > self.LR_DECAY_ITERS:
            raise ValueError("WARMUP_ITERS must be less than LR_DECAY_ITERS")

    @classmethod
    def create_from_config_file(cls, config_file: str):
        config_dict = {}
        with open(config_file, "r") as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}

        model_config_props = [f.name.upper() for f in fields(ModelConfig)]
        model_config_dict = {
            k.lower(): v for k, v in config_dict.items() if k in model_config_props
        }
        model_config = ModelConfig(**model_config_dict)

        config_dict = {
            k: v for k, v in config_dict.items() if k not in model_config_props
        }
        config_dict["MODEL_CONFIG"] = model_config
        return cls(**config_dict)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--val_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--initialization_type", type=lambda x : InitializationType(x), default=InitializationType.SCRATCH)
    parser.add_argument("--is_local", type=lambda v: bool(strtobool(v)))
    args = parser.parse_args()
    return args


def get_data_batch(device, device_type, context_size, batch_size, split="train"):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(0, len(data) - context_size - 1, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[idx : idx + context_size]).astype(np.int64))
            for idx in idxs
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[idx + 1 : idx + context_size + 1]).astype(np.int64))
            for idx in idxs
        ]
    )

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # from https://github.com/karpathy/nanoGPT/blob/master/train.py
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model, est_steps, context_size, batch_size, device, ctx, using_DP, device_type
):
    mean_losses = []
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(est_steps, device=device)
        for i in range(est_steps):
            xb, yb = get_data_batch(
                device, device_type, context_size, batch_size, split
            )
            with ctx:
                _, loss, _, _ = model(xb, yb)
            if using_DP:
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
    TRAIN_CONFIG = TrainConfig.create_from_config_file(args.config_file)

    using_DDP = (
        TRAIN_CONFIG.USE_DDP
        and TRAIN_CONFIG.DEVICE == "cuda"
        and torch.cuda.device_count() > 1
    )
    using_DP = (
        TRAIN_CONFIG.DEVICE == "cuda"
        and torch.cuda.device_count() > 1
        and TRAIN_CONFIG.USE_DP
    )
    if using_DDP:
        init_process_group(backend="nccl")
        ddp_rank = torch.distributed.get_rank()
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = torch.distributed.get_world_size()
        TRAIN_CONFIG.DEVICE = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(TRAIN_CONFIG.DEVICE)
        is_master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS % ddp_world_size == 0
        TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS //= ddp_world_size
    else:
        is_master_process = True
        seed_offset = 0

    torch.manual_seed(1337 + seed_offset)  # this allows for distributed training data
    # From https://github.com/karpathy/nanoGPT/blob/master/train.py
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in TRAIN_CONFIG.DEVICE else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[TRAIN_CONFIG.DTYPE]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Load and prepare training data
    training_data_file_path = os.path.join(args.train, args.train_file)
    val_date_file_path = os.path.join(args.train, args.val_file)
    train_data = np.memmap(training_data_file_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_date_file_path, dtype=np.uint16, mode="r")

    meta_path = os.path.join(args.train, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    TRAIN_CONFIG.MODEL_CONFIG.alphabet_size = meta["alphabet_size"]

    iter_num = 0
    wandb_run_id = None
    if args.initialization_type == InitializationType.SCRATCH:
        model = DropoutTransformer(TRAIN_CONFIG.MODEL_CONFIG)
    else:
        args.initialization_type == InitializationType.RESUME
        ckpt_path = os.path.join(args.checkpoint_dir or os.environ["SM_MODEL_DIR"], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=TRAIN_CONFIG.DEVICE)
        TRAIN_CONFIG.MODEL_CONFIG = ModelConfig(**checkpoint['model_config'])
        # create the model
        model = DropoutTransformer(TRAIN_CONFIG.MODEL_CONFIG)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        wandb_run_id = checkpoint['wandb_run_id']

    model.to(TRAIN_CONFIG.DEVICE)

    MODEL_PARAMS = model.get_num_params()
    scaler = torch.cuda.amp.GradScaler(enabled=(TRAIN_CONFIG.DTYPE == "float16"))
    optimizer = model.configure_optimizer(
        TRAIN_CONFIG.WEIGHT_DECAY,
        TRAIN_CONFIG.LR,
        (TRAIN_CONFIG.BETA1, TRAIN_CONFIG.BETA2),
        device_type,
    )
    if args.initialization_type == InitializationType.RESUME:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    # when using DDP and LearnedDropout, compiling the model causes a bug in sagemaker
    if TRAIN_CONFIG.COMPILE and using_DDP and False:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    if using_DP:
        model = torch.nn.DataParallel(model)
    elif using_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank]
        )

    # learning rate decay scheduler (cosine with warmup). From https://github.com/karpathy/nanoGPT/blob/master/train.py
    def get_lr(training_step):
        # 1) linear warmup for warmup_iters steps
        if training_step < TRAIN_CONFIG.WARMUP_ITERS:
            return TRAIN_CONFIG.LR * training_step / TRAIN_CONFIG.WARMUP_ITERS
        # 2) if it > lr_decay_iters, return min learning rate
        if training_step > TRAIN_CONFIG.LR_DECAY_ITERS:
            return TRAIN_CONFIG.MIN_LR
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (training_step - TRAIN_CONFIG.WARMUP_ITERS) / (
            TRAIN_CONFIG.LR_DECAY_ITERS - TRAIN_CONFIG.WARMUP_ITERS
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return TRAIN_CONFIG.MIN_LR + coeff * (TRAIN_CONFIG.LR - TRAIN_CONFIG.MIN_LR)

    if is_master_process:
        if wandb_run_id is None:
            wandb.init(
                # set the wandb project where this run will be logged
                project="transformer_dropout",
                config={
                    **asdict(TRAIN_CONFIG),
                    "params": MODEL_PARAMS,
                    "using_DP": using_DP,
                    "using_DDP": using_DDP,
                    "world_size": ddp_world_size if using_DDP else None,
                },
                mode="online",
                resume=True
            )
            # wandb_run_id = wandb.util.generate_id()
            wandb_run_id = None
        else:
            wandb.init(resume="must", id=wandb_run_id)

    raw_model = model.module if using_DP or using_DDP else model
    model.train()
    X, Y = get_data_batch(
        TRAIN_CONFIG.DEVICE,
        device_type,
        TRAIN_CONFIG.MODEL_CONFIG.context_size,
        TRAIN_CONFIG.BATCH_SIZE,
        "train",
    )  # fetch the very first batch
    t0 = time.time()
    while iter_num < TRAIN_CONFIG.TRAIN_STEPS:
        # determine and set the learning rate for this iteration. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        lr = get_lr(iter_num) if TRAIN_CONFIG.DECAY_LR else TRAIN_CONFIG.LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if (
            iter_num % TRAIN_CONFIG.EST_INTERVAL == 0
            and iter_num != (TRAIN_CONFIG.TRAIN_STEPS - 1)
            and iter_num != 0
        ) and is_master_process:
            train_loss, val_loss = estimate_loss(
                model,
                TRAIN_CONFIG.EST_STEPS,
                TRAIN_CONFIG.MODEL_CONFIG.context_size,
                TRAIN_CONFIG.BATCH_SIZE,
                TRAIN_CONFIG.DEVICE,
                ctx,
                using_DP,
                device_type,
            )
            wandb.log(
                {
                    "est_iter": iter_num,
                    "est_train_loss": train_loss,
                    "est_val_loss": val_loss,
                    "est_lr": lr,
                }
            )
            checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': asdict(TRAIN_CONFIG.MODEL_CONFIG),
                    'iter_num': iter_num,
                    'config': asdict(TRAIN_CONFIG),
                    "wandb_run_id": wandb_run_id,
                }
            torch.save(
                checkpoint,
                (
                    f"transformer_dropout/model_checkpoints/ckpt.pt"
                    if args.is_local
                    else os.path.join(os.environ["SM_MODEL_DIR"], "ckpt.pt")
                ),
            )

        running_loss = 0
        running_entropy = 0 if TRAIN_CONFIG.MODEL_CONFIG.use_learned_dropout else None
        running_l1_norm = 0 if TRAIN_CONFIG.MODEL_CONFIG.use_learned_dropout else None
        for micro_step in range(TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS):
            if using_DDP:
                model.require_backward_grad_sync = (
                    micro_step == TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS - 1
                )
            with ctx:
                logits, loss, entropy, dropout_l1_norm = model(X, Y)
                if using_DP:
                    loss = loss.mean()
                    if TRAIN_CONFIG.MODEL_CONFIG.use_learned_dropout:
                        entropy = entropy.mean()
                        dropout_l1_norm = dropout_l1_norm.mean()

                loss = (
                    loss / TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS
                )  # scale the loss to account for gradient accumulation
                running_loss += loss.item()
                if TRAIN_CONFIG.MODEL_CONFIG.use_learned_dropout:
                    running_entropy += (
                        entropy.item() / TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS
                    )
                    running_l1_norm += (
                        dropout_l1_norm.item()
                        / TRAIN_CONFIG.GRADIENT_ACCUMULATION_STEPS
                    )
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_data_batch(
                TRAIN_CONFIG.DEVICE,
                device_type,
                TRAIN_CONFIG.MODEL_CONFIG.context_size,
                TRAIN_CONFIG.BATCH_SIZE,
                "train",
            )
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if is_master_process:
            wandb.log(
                {
                    "iter": iter_num,
                    "loss": running_loss,
                    "dropout_entropy": running_entropy,
                    "dropout_l1_norm": running_l1_norm,
                    "time": float(f"{dt*1000:.2f}"),
                }
            )
        iter_num += 1
    if is_master_process:
        torch.save(
            {
                "state_dict": raw_model.state_dict(),
                "hyperparameters": asdict(TRAIN_CONFIG.MODEL_CONFIG),
                "itoc": None,  # TODO: add decoder
            },
            (
                f"transformer_dropout/model_weights/model_{datetime.now().strftime('%H-%M-%S-%d-%m-%y')}.pth"
                if args.is_local
                else os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
            ),
        )

    if using_DDP:
        destroy_process_group()
