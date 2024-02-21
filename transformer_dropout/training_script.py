import argparse
import logging
import math
import os
import pickle
import random
import sys
import tarfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from distutils.util import strtobool
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.datapipes.iter import S3FileLoader, Shuffler

# ugly workound to make both Sagemaker, python, and me happy
try:
    # I like to run the script from the project root as a module, so it needs to be relative import
    from .data_loading import (DistributedIterableLocalDataset,
                               IterableLocalDataset, MapLocalDataset)
    from .model import DropoutTransformer, ModelConfig
except ImportError:
    # Sagemaker prob runs the script as a standalone file, so it needs to be an absolute import
    from model import DropoutTransformer, ModelConfig
    from data_loading import (
        MapLocalDataset,
        IterableLocalDataset,
        DistributedIterableLocalDataset,
    )

import wandb
from torch.distributed import destroy_process_group, init_process_group


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
    RANDOM_SEED: int = field(default=1337)
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

        model_config_fields = [f.name.upper() for f in fields(ModelConfig)]
        model_config_dict = {
            k.lower(): v for k, v in config_dict.items() if k in model_config_fields
        }
        model_config = ModelConfig(**model_config_dict)

        config_dict = {
            k: v for k, v in config_dict.items() if k not in model_config_fields
        }
        config_dict["MODEL_CONFIG"] = model_config
        return cls(**config_dict)


def get_data_batch_loader(data_iter, data_loader, data_sampler, iter_num, device):
    new_data_iter = None
    try:
        x, y = next(data_iter)
    except StopIteration:
        if data_sampler is not None:
            data_sampler.set_epoch(iter_num)
        new_data_iter = iter(data_loader)
        x, y = next(new_data_iter)

    if data_loader.pin_memory is True:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y, new_data_iter


def get_torch_save_dict(raw_model, optimizer, train_config, iter_num, best_val_loss):
    return (
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(train_config.MODEL_CONFIG),
            "iter_num": iter_num,
            "config": asdict(train_config),
            "best_val_loss": best_val_loss,
            # "itoc": None,  # TODO: add decoder,
        },
    )


def save_checkpoint(filename, checkpoint, checkpoint_path, is_local):
    uncompressed_checkpoint_path = os.path.join(checkpoint_path, f"{filename}.pt")
    # Save the uncompressed checkpoint
    torch.save(checkpoint, uncompressed_checkpoint_path)

    # Creating a sagemaker model afterwards with the checkpoint requires it to be compressed
    if not is_local:
        compressed_checkpoint_path = os.path.join(checkpoint_path, f"{filename}.tar.gz")
        # Compress and save the checkpoint
        with tarfile.open(compressed_checkpoint_path, "w:gz") as tar:
            tar.add(uncompressed_checkpoint_path, arcname=f"{filename}.pt")


@torch.no_grad()
def estimate_loss(
    model,
    est_steps,
    device,
    ctx,
    using_DP,
    train_data_batch_args,
    val_data_batch_args,
    iter_num,
):
    mean_losses = []
    model.eval()
    new_data_iters = []
    for args in [train_data_batch_args, val_data_batch_args]:
        original_data_iter = args[0]
        data_iter = args[0]
        data_loader = args[1]
        data_sampler = args[2]
        losses = torch.zeros(est_steps, device=device)
        for i in range(est_steps):
            xb, yb, new_data_iter = get_data_batch_loader(
                data_iter, data_loader, data_sampler, iter_num, device
            )
            if new_data_iter is not None:
                data_iter = new_data_iter

            with ctx:
                _, loss, _, _ = model(xb, yb)
            if using_DP:
                loss = loss.mean()
            losses[i] = loss

        new_data_iters.append(data_iter if original_data_iter != data_iter else None)
        mean_losses.append(losses.mean().item())
    model.train()
    return (mean_losses, new_data_iters)


def train(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    logger = logging.getLogger()
    logger.info("Starting training script.")
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

    initialization_type = InitializationType.SCRATCH
    checkpoint_path = (
        args.local_checkpoint_path if args.is_local else args.checkpoint_path
    )
    ckpt_file_path = None
    if checkpoint_path is not None:
        assert os.path.isdir(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_path, "ckpt.pt")):
            initialization_type = InitializationType.RESUME
            ckpt_file_path = os.path.join(checkpoint_path, "ckpt.pt")

    torch.manual_seed(
        TRAIN_CONFIG.RANDOM_SEED + seed_offset
    )  # this allows for distributed training data
    np.random.seed(TRAIN_CONFIG.RANDOM_SEED + seed_offset)
    random.seed(TRAIN_CONFIG.RANDOM_SEED + seed_offset)

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
    directory = Path(args.train)
    [train_file_path] = list(directory.glob("*_train.bin"))
    [val_file_path] = list(directory.glob("*_val.bin"))
    train_data, train_sampler = MapLocalDataset.create_with_distributed_sampler(
        train_file_path,
        TRAIN_CONFIG.MODEL_CONFIG.context_size,
        TRAIN_CONFIG.BATCH_SIZE,
        using_DDP,
    )
    val_data, val_sampler = MapLocalDataset.create_with_distributed_sampler(
        val_file_path,
        TRAIN_CONFIG.MODEL_CONFIG.context_size,
        TRAIN_CONFIG.BATCH_SIZE,
        using_DDP,
    )
    train_data_loader = DataLoader(
        train_data,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        shuffle=(train_sampler is None),
        pin_memory=True if device_type == "cuda" else False,
    )
    val_data_loader = DataLoader(
        val_data,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        sampler=val_sampler,
        num_workers=0,
        shuffle=(val_sampler is None),
        pin_memory=True if device_type == "cuda" else False,
    )
    curr_train_iter = iter(train_data_loader)
    curr_val_iter = iter(val_data_loader)

    best_val_loss = None
    iter_num = 0
    if initialization_type == InitializationType.SCRATCH:
        meta_path = os.path.join(args.train, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        TRAIN_CONFIG.MODEL_CONFIG.alphabet_size = meta["alphabet_size"]
        model = DropoutTransformer(TRAIN_CONFIG.MODEL_CONFIG)
    else:
        print("Loading checkpoint...")
        assert initialization_type == InitializationType.RESUME
        checkpoint = torch.load(ckpt_file_path, map_location=TRAIN_CONFIG.DEVICE)
        TRAIN_CONFIG.MODEL_CONFIG = ModelConfig(**checkpoint["model_config"])
        # create the model
        model = DropoutTransformer(TRAIN_CONFIG.MODEL_CONFIG)
        state_dict = checkpoint["model"]
        # from https://github.com/karpathy/nanoGPT/blob/master/train.py
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"] + 1

    model.to(TRAIN_CONFIG.DEVICE)

    MODEL_PARAMS = model.get_num_params()
    scaler = torch.cuda.amp.GradScaler(enabled=(TRAIN_CONFIG.DTYPE == "float16"))
    optimizer = model.configure_optimizer(
        TRAIN_CONFIG.WEIGHT_DECAY,
        TRAIN_CONFIG.LR,
        (TRAIN_CONFIG.BETA1, TRAIN_CONFIG.BETA2),
        device_type,
    )
    if initialization_type == InitializationType.RESUME:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None

    # when using DDP and LearnedDropout, compiling the model causes a bug in sagemaker, so disabling for now
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
        wandb.init(
            project="transformer_dropout",
            config={
                **asdict(TRAIN_CONFIG),
                "params": MODEL_PARAMS,
                "using_DP": using_DP,
                "using_DDP": using_DDP,
                "world_size": ddp_world_size if using_DDP else None,
            },
            dir=Path(
                __file__
            ).parent,  # this must be in the same directory as the training script in order to make auto-resumption work
            mode="online",
            # resume=True, # enables resuming a previous run
        )

    raw_model = model.module if using_DP or using_DDP else model
    model.train()
    X, Y, _ = get_data_batch_loader(
        curr_train_iter,
        train_data_loader,
        train_sampler,
        -1,
        TRAIN_CONFIG.DEVICE,
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
            (train_loss, val_loss), (new_train_iter, new_val_iter) = estimate_loss(
                model,
                TRAIN_CONFIG.EST_STEPS,
                TRAIN_CONFIG.DEVICE,
                ctx,
                using_DP,
                (curr_train_iter, train_data_loader, train_sampler),
                (curr_val_iter, val_data_loader, val_sampler),
                iter_num,
            )
            if new_train_iter is not None:
                curr_train_iter = new_train_iter
            if new_val_iter is not None:
                curr_val_iter = new_val_iter

            should_save_best_val_loss_checkpoint = False
            if best_val_loss is None or val_loss <= best_val_loss:
                best_val_loss = val_loss
                should_save_best_val_loss_checkpoint = True

            wandb.log(
                {
                    "est_train_loss": train_loss,
                    "est_val_loss": val_loss,
                    "est_lr": lr,
                    "est_step": iter_num / TRAIN_CONFIG.EST_INTERVAL - 1,
                },
                step=iter_num,
                # commit=True,
            )
            save_checkpoint(
                "ckpt",
                get_torch_save_dict(
                    raw_model, optimizer, TRAIN_CONFIG, iter_num, best_val_loss
                ),
                (
                    "transformer_dropout/model_checkpoints/"
                    if args.is_local
                    else checkpoint_path
                ),
                args.is_local,
            )
            if should_save_best_val_loss_checkpoint:
                save_checkpoint(
                    "best_ckpt",
                    get_torch_save_dict(
                        raw_model, optimizer, TRAIN_CONFIG, iter_num, best_val_loss
                    ),
                    (
                        "transformer_dropout/model_checkpoints/"
                        if args.is_local
                        else checkpoint_path
                    ),
                    args.is_local,
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
                _, loss, entropy, dropout_l1_norm = model(X, Y)
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
            X, Y, new_train_iter = get_data_batch_loader(
                curr_train_iter,
                train_data_loader,
                train_sampler,
                -1,
                TRAIN_CONFIG.DEVICE,
            )
            if new_train_iter is not None:
                curr_train_iter = new_train_iter
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
                    "loss": running_loss,
                    "dropout_entropy": running_entropy,
                    "dropout_l1_norm": running_l1_norm,
                    "time": float(f"{dt*1000:.2f}"),
                },
                step=iter_num,
                # commit=False,
            )
        iter_num += 1

    if is_master_process:
        torch.save(
            get_torch_save_dict(
                raw_model, optimizer, TRAIN_CONFIG, iter_num, best_val_loss
            ),
            (
                f"transformer_dropout/model_weights/model_{datetime.now().strftime('%H-%M-%S-%d-%m-%y')}.pth"
                if args.is_local
                else os.path.join(os.environ["SM_MODEL_DIR"], "model.pth")
            ),
        )

    if using_DDP:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )
    parser.add_argument("--train", type=str)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--is_local", type=lambda v: bool(strtobool(v)), default=True)
    parser.add_argument("--checkpoint_path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--local_checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    if args.local_checkpoint_path is not None:
        assert args.is_local

    if not args.is_local:
        args.train = os.environ.get("SM_CHANNEL_TRAIN")
    else:
        assert args.train is not None

    train(args)
