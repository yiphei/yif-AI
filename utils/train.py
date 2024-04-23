import argparse
import inspect
import io
import logging
import math
import os
import pickle
import random
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from distutils.util import strtobool
from enum import Enum
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader

import wandb
from utils.data_loading import MapLocalDataset


# This is a hack to circumvent the dataclass requirement that fields with non-default values must precede those with them
def required_field_exception():
    raise ValueError("Missing required property")


class PlatformType(str, Enum):
    LOCAL = "LOCAL"
    SAGEMAKER = "SAGEMAKER"
    LAMBDA = "LAMBDA"
    PAPERSPACE = "PAPERSPACE"

    def __str__(self):
        return self.value


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    # NB: "mps" introduces non-deterministic behavior, despite explicitly setting random seeds.
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class TrainConfig:
    model_config: Optional[dataclass]
    random_seed: int = field(default=1337)
    # Training
    batch_size: int = field(
        default_factory=required_field_exception
    )  # this will be scaled by GRADIENT_ACCUMULATION_STEPS
    train_steps: int = field(default_factory=required_field_exception)
    gradient_accumulation_steps: int = field(
        default_factory=required_field_exception
    )  # used to simulate large batches. Must be a multiple of world_size (i.e. # of GPUs) if using DDP
    # Optimizer
    lr: float = field(default=6e-4)  # max learning rate
    weight_decay: float = field(default=1e-1)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    decay_lr: bool = True
    warmup_iters: int = field(default_factory=required_field_exception)
    lr_decay_iters: int = field(default_factory=required_field_exception)
    min_lr: float = field(default=6e-5)
    # Estimation
    est_interval: int = field(default_factory=required_field_exception)
    est_steps: int = field(default_factory=required_field_exception)
    # Other
    dtype: str = field(
        default_factory=lambda: (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    )
    compile: bool = True

    def __post_init__(self):
        self.validate_field_values()

    def validate_field_values(self):
        if self.train_steps <= self.est_interval:
            raise ValueError("EST_INTERVAL must be less than TRAIN_STEPS")
        if self.min_lr >= self.lr:
            raise ValueError("MIN_LR must be less than LR")
        if self.warmup_iters >= self.train_steps:
            raise ValueError("WARMUP_ITERS must be less than TRAIN_STEPS")
        if self.est_steps >= self.train_steps:
            raise ValueError("EST_STEPS must be less than TRAIN_STEPS")
        if self.lr_decay_iters > self.train_steps:
            raise ValueError("LR_DECAY_ITERS must be less than TRAIN_STEPS")
        if self.warmup_iters > self.lr_decay_iters:
            raise ValueError("WARMUP_ITERS must be less than LR_DECAY_ITERS")

    @classmethod
    def create_from_config_file(
        cls, config_file: str, model_config_cls, is_sweep=False
    ):
        config_dict = {}
        with open(config_file, "r") as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {
            k.lower(): v
            for k, v in config_dict.items()
            if not k.startswith("__") and not inspect.ismodule(v)
        }

        model_config_fields = [f.name for f in fields(model_config_cls)]
        if not is_sweep:
            model_config_dict = {
                k: v for k, v in config_dict.items() if k in model_config_fields
            }
            model_config = model_config_cls(**model_config_dict)
        else:
            model_config = None

        config_dict = {
            k: v for k, v in config_dict.items() if k not in model_config_fields
        }
        config_dict["model_config"] = model_config
        return cls(**config_dict)

    def update_from_sweep_config(self, sweep_config):
        for k, v in sweep_config.items():
            if k == "model_config":
                continue

            assert hasattr(self, k)

            # for some reason, sometimes wandb does not convert this to a float
            if k in ["lr", "min_lr"]:
                v = float(v)
            setattr(self, k, v)

        self.validate_field_values()


DEFAULT_BUCKET = "dropout-transformer"


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
    else:
        x, y = x.to(device), y.to(device)
    return x, y, new_data_iter


def get_torch_save_dict(raw_model, optimizer, train_config, iter_num, best_val_loss):
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(train_config.model_config),
        "iter_num": iter_num,
        "config": asdict(train_config),
        "best_val_loss": best_val_loss,
        # "itoc": None,  # TODO: add decoder,
    }


@torch.no_grad()
def estimate_loss(
    model,
    raw_model,
    est_steps,
    device,
    ctx,
    train_data_batch_args,
    val_data_batch_args,
    iter_num,
):
    mean_accuracies = []
    mean_losses = []
    model.eval()
    new_data_iters = []
    for args in [train_data_batch_args, val_data_batch_args]:
        original_data_iter = args[0]
        data_iter = args[0]
        data_loader = args[1]
        data_sampler = args[2]
        accuracies = torch.zeros(est_steps, device=device)
        losses = torch.zeros(est_steps, device=device)
        for i in range(est_steps):
            xb, yb, new_data_iter = get_data_batch_loader(
                data_iter, data_loader, data_sampler, iter_num, device
            )
            if new_data_iter is not None:
                data_iter = new_data_iter

            with ctx(i, False):
                logits, loss = model(xb, yb)

            losses[i] = loss

            accuracies[i] = raw_model.get_accuracy(logits, yb)

        new_data_iters.append(data_iter if original_data_iter != data_iter else None)
        mean_accuracies.append(accuracies.mean().item())
        mean_losses.append(losses.mean().item())
    model.train()
    return (mean_accuracies, mean_losses, new_data_iters)


def save_model_artifact(filenames, model_dict, dir_path, s3_client):
    for filename in filenames:
        file_path = os.path.join(dir_path, filename)
        if s3_client is None:
            torch.save(model_dict, file_path)
        else:
            buffer = io.BytesIO()
            torch.save(model_dict, buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, DEFAULT_BUCKET, file_path)


def broadcast_object(obj, local_rank, device, src_rank=0):
    if local_rank == src_rank:
        # Only the source process executes this block
        obj_bytes = pickle.dumps(obj)
        obj_size = torch.tensor(len(obj_bytes), dtype=torch.long, device=device)
    else:
        obj_size = torch.tensor(0, dtype=torch.long, device=device)

    # Broadcast the size of the byte stream to all processes
    torch.distributed.broadcast(obj_size, src_rank)

    # Allocate buffer for the object's byte stream
    obj_bytes = bytearray(obj_size.item())

    if local_rank == src_rank:
        # Only the source fills the byte buffer
        obj_bytes[:] = pickle.dumps(obj)

    # Create a tensor wrapper for the byte buffer for broadcasting
    obj_tensor = torch.ByteTensor(obj_bytes).to(device)
    # Broadcast the byte stream
    torch.distributed.broadcast(obj_tensor, src_rank)

    # Deserialize the byte stream back into the Python object
    obj = pickle.loads(obj_tensor.cpu().numpy().tobytes())
    return obj


def _train(
    args,
    model_cls,
    create_training_context_fn,
    local_dir,
    wandb_project,
):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    logger = logging.getLogger()
    logger.info("Starting training script.")
    TRAIN_CONFIG = TrainConfig.create_from_config_file(
        args.config_file, model_cls.model_config_cls, args.sweep_id is not None
    )
    DEVICE = get_default_device()

    using_DDP = DEVICE == "cuda" and torch.cuda.device_count() > 1
    ddp_world_size = None
    if using_DDP:
        init_process_group(backend="nccl")
        ddp_rank = torch.distributed.get_rank()
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = torch.distributed.get_world_size()
        DEVICE = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(DEVICE)
        is_master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert TRAIN_CONFIG.gradient_accumulation_steps % ddp_world_size == 0
        TRAIN_CONFIG.gradient_accumulation_steps //= ddp_world_size
    else:
        is_master_process = True
        seed_offset = 0

    if is_master_process and args.profile:
        wandb.init(
            project=(
                wandb_project
                if args.platform_type != PlatformType.LOCAL
                else "local_test"
            ),
            dir=local_dir,  # this must be in the same directory as the training script in order to make auto-resumption work
            mode="online" if args.sync_profile_live else "offline",
            # resume=True, # enables resuming a previous run
        )
        if args.save_code:
            wandb.run.log_code(".")

    if args.sweep_id is not None and is_master_process:
        TRAIN_CONFIG.model_config = model_cls.model_config_cls(
            **wandb.config.model_config
        )
        TRAIN_CONFIG.update_from_sweep_config(wandb.config)

    if using_DDP:
        TRAIN_CONFIG = broadcast_object(TRAIN_CONFIG, ddp_local_rank, DEVICE)

    s3_client = None
    # need to create locally scoped variables because sweep runs necessitate creating new paths for each run
    current_checkpoint_path = args.checkpoint_path
    current_model_path = args.model_path
    if (
        is_master_process
        and args.platform_type not in [PlatformType.SAGEMAKER, PlatformType.LOCAL]
        and (current_model_path is None or current_checkpoint_path is None)
        and (args.save_checkpoint or args.save_model)
    ):
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
        )
        sweep_tag = f"_sweep_id_{str(args.sweep_id)}" if args.sweep_id else ""
        training_run_dir = f"training/{args.platform_type.lower()}{sweep_tag}_training_run_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}/"
        if current_model_path is None and args.save_model:
            s3_client.put_object(Bucket=DEFAULT_BUCKET, Key=training_run_dir + "model/")
            current_model_path = training_run_dir + "model/"
        if current_checkpoint_path is None and args.save_checkpoint:
            s3_client.put_object(
                Bucket=DEFAULT_BUCKET, Key=training_run_dir + "checkpoints/"
            )
            current_checkpoint_path = training_run_dir + "checkpoints/"
        print(f"S3 folder is: {training_run_dir}")

    initialize_from_checkpoint = False
    ckpt_file_path = None
    if current_checkpoint_path is not None and args.resume_from_checkpoint:
        assert os.path.isdir(current_checkpoint_path)
        if os.path.isfile(os.path.join(current_checkpoint_path, "ckpt.pt")):
            initialize_from_checkpoint = True
            ckpt_file_path = os.path.join(current_checkpoint_path, "ckpt.pt")

    # seed_offset allows for distributed training data
    torch.manual_seed(TRAIN_CONFIG.random_seed + seed_offset)
    np.random.seed(TRAIN_CONFIG.random_seed + seed_offset)
    random.seed(TRAIN_CONFIG.random_seed + seed_offset)

    # From https://github.com/karpathy/nanoGPT/blob/master/train.py
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in DEVICE else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[TRAIN_CONFIG.dtype]

    # Load and prepare training data
    directory = Path(args.train)
    [train_file_path] = list(directory.glob("*_train.bin"))
    [val_file_path] = list(directory.glob("*_val.bin"))
    train_data, train_sampler = MapLocalDataset.create_with_distributed_sampler(
        train_file_path,
        TRAIN_CONFIG.model_config.context_size,
        TRAIN_CONFIG.batch_size,
        using_DDP,
    )
    val_data, val_sampler = MapLocalDataset.create_with_distributed_sampler(
        val_file_path,
        TRAIN_CONFIG.model_config.context_size,
        TRAIN_CONFIG.batch_size,
        using_DDP,
    )
    train_data_loader = DataLoader(
        train_data,
        batch_size=TRAIN_CONFIG.batch_size,
        sampler=train_sampler,
        num_workers=0,
        shuffle=(train_sampler is None),
        pin_memory=True if device_type == "cuda" else False,
    )
    val_data_loader = DataLoader(
        val_data,
        batch_size=TRAIN_CONFIG.batch_size,
        sampler=val_sampler,
        num_workers=0,
        shuffle=(val_sampler is None),
        pin_memory=True if device_type == "cuda" else False,
    )
    curr_train_iter = iter(train_data_loader)
    curr_val_iter = iter(val_data_loader)

    best_val_loss = None
    iter_num = 0
    if not initialize_from_checkpoint:
        meta_path = os.path.join(args.train, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        TRAIN_CONFIG.model_config.alphabet_size = meta["alphabet_size"]
        model = model_cls(
            TRAIN_CONFIG.model_config,
            gradient_accumulation_steps=TRAIN_CONFIG.gradient_accumulation_steps,
        )
    else:
        print("Loading checkpoint...")
        checkpoint = torch.load(ckpt_file_path, map_location=DEVICE)
        model = model_cls.init_from_checkpoint(
            checkpoint,
            gradient_accumulation_steps=TRAIN_CONFIG.gradient_accumulation_steps,
        )
        TRAIN_CONFIG.model_config = model.config
        iter_num = checkpoint["iter_num"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    model.to(DEVICE)
    ctx = create_training_context_fn(model, iter_num, device_type, ptdtype)

    MODEL_NUM_PARAMS = model.get_num_params()
    scaler = torch.cuda.amp.GradScaler(enabled=(TRAIN_CONFIG.dtype == "float16"))
    optimizer = model.configure_optimizer(
        TRAIN_CONFIG.weight_decay,
        TRAIN_CONFIG.lr,
        (TRAIN_CONFIG.beta1, TRAIN_CONFIG.beta2),
        device_type,
    )
    if initialize_from_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None

    # when using DDP and LearnedDropout, compiling the model causes a bug in sagemaker, so disabling for now
    if TRAIN_CONFIG.compile and using_DDP and False:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    if using_DDP:
        # NB: broadcast_buffers = False is fine here because there is no buffer 
        # that currently needs to be synced. But if the model uses BatchNorm
        # and the likes, the buffers will need to be synced.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank], broadcast_buffers=False
        )

    # learning rate decay scheduler (cosine with warmup). From https://github.com/karpathy/nanoGPT/blob/master/train.py
    def get_lr(training_step):
        adjusted_training_step = (
            training_step + 1
        )  # to avoid zero division when training_step = 0
        # 1) linear warmup for warmup_iters steps
        if adjusted_training_step < TRAIN_CONFIG.warmup_iters:
            return TRAIN_CONFIG.lr * adjusted_training_step / TRAIN_CONFIG.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if adjusted_training_step > TRAIN_CONFIG.lr_decay_iters:
            return TRAIN_CONFIG.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (adjusted_training_step - TRAIN_CONFIG.warmup_iters) / (
            TRAIN_CONFIG.lr_decay_iters - TRAIN_CONFIG.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return TRAIN_CONFIG.min_lr + coeff * (TRAIN_CONFIG.lr - TRAIN_CONFIG.min_lr)

    if is_master_process and args.profile:
        wandb.config.update(
            {
                **asdict(TRAIN_CONFIG),
                "num_params": MODEL_NUM_PARAMS,
                "using_DDP": using_DDP,
                "world_size": ddp_world_size if using_DDP else None,
                "device": DEVICE,
            }
        )

    raw_model = model.module if using_DDP else model
    model.train()
    X, Y, _ = get_data_batch_loader(
        curr_train_iter,
        train_data_loader,
        train_sampler,
        -1,
        DEVICE,
    )
    while iter_num < TRAIN_CONFIG.train_steps:
        # determine and set the learning rate for this iteration. From https://github.com/karpathy/nanoGPT/blob/master/train.py
        lr = get_lr(iter_num) if TRAIN_CONFIG.decay_lr else TRAIN_CONFIG.lr
        optimizer.change_lr(lr)

        if (
            (iter_num + 1) % TRAIN_CONFIG.est_interval == 0
            or iter_num == (TRAIN_CONFIG.train_steps - 1)
        ) and is_master_process:
            (
                (train_accuracy, val_accuracy),
                (train_loss, val_loss),
                (new_train_iter, new_val_iter),
            ) = estimate_loss(
                model,
                raw_model,
                TRAIN_CONFIG.est_steps,
                DEVICE,
                ctx,
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

            if args.profile:
                wandb.log(
                    {
                        "est_train_accuracy": train_accuracy,
                        "est_train_loss": train_loss,
                        "est_val_accuracy": val_accuracy,
                        "est_val_loss": val_loss,
                        "est_lr": lr,
                        "est_step": iter_num / TRAIN_CONFIG.est_interval - 1,
                    },
                    step=iter_num,
                    # commit=True,
                )

            filenames = (
                ["ckpt.pt"]
                if not should_save_best_val_loss_checkpoint
                else ["best_ckpt.pt", "ckpt.pt"]
            )

            if args.save_checkpoint:
                save_model_artifact(
                    filenames,
                    get_torch_save_dict(
                        raw_model, optimizer, TRAIN_CONFIG, iter_num, best_val_loss
                    ),
                    current_checkpoint_path,
                    s3_client,
                )

        t0 = time.time()
        running_loss = 0
        is_first_mini_batch = True
        raw_model.reset_running_stats()
        for micro_step in range(TRAIN_CONFIG.gradient_accumulation_steps):
            if using_DDP:
                # this defers gradient sync until the last micro_step
                model.require_backward_grad_sync = (
                    micro_step == TRAIN_CONFIG.gradient_accumulation_steps - 1
                )
            with ctx(iter_num, is_first_mini_batch):
                (_, loss) = model(X, Y)

                loss = (
                    loss / TRAIN_CONFIG.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
                running_loss += loss.item()

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, new_train_iter = get_data_batch_loader(
                curr_train_iter,
                train_data_loader,
                train_sampler,
                -1,
                DEVICE,
            )
            if new_train_iter is not None:
                curr_train_iter = new_train_iter
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            is_first_mini_batch = False

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        if is_master_process and args.profile:
            mfu = 0
            if iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    TRAIN_CONFIG.batch_size * TRAIN_CONFIG.gradient_accumulation_steps,
                    dt,
                )

            wandb.log(
                {
                    "loss": running_loss,
                    "time": float(f"{dt*1000:.2f}"),
                    "mfu": mfu,
                    **raw_model.dump_stats(),
                },
                step=iter_num,
                # commit=False,
            )
        iter_num += 1

    if is_master_process and args.save_model:
        save_model_artifact(
            [
                (
                    f"model_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.pth"
                    if args.platform_type == PlatformType.LOCAL
                    else "model.pth"
                )
            ],
            get_torch_save_dict(
                raw_model, optimizer, TRAIN_CONFIG, iter_num, best_val_loss
            ),
            current_model_path,
            s3_client,
        )

    if using_DDP:
        destroy_process_group()

    if is_master_process and args.profile and not args.sync_profile_live:
        wandb_run_dir = wandb.run._settings.sync_dir
        wandb.finish()
        result = subprocess.run(
            f"wandb sync {wandb_run_dir}", shell=True, stdout=subprocess.PIPE, text=True
        )
        print(result.stdout)


def get_default_args(args, local_dir):
    if args.platform_type == PlatformType.SAGEMAKER:
        if args.checkpoint_path is None:
            args.checkpoint_path = "/opt/ml/checkpoints"
        if args.train is None:
            args.train = os.environ.get("SM_CHANNEL_TRAIN")
        if args.model_path is None:
            args.model_path = os.environ["SM_MODEL_DIR"]
        if args.resume_from_checkpoint is None:
            args.resume_from_checkpoint = True
    elif args.platform_type == PlatformType.LOCAL:
        if args.checkpoint_path is None:
            args.checkpoint_path = local_dir + "model_checkpoints/"
        assert args.train is not None
        if args.model_path is None:
            args.model_path = local_dir + "model_weights/"
        if args.resume_from_checkpoint is None:
            args.resume_from_checkpoint = False
    elif args.platform_type in [PlatformType.LAMBDA, PlatformType.PAPERSPACE]:
        if (args.checkpoint_path is None or args.model_path is None) and (
            args.save_checkpoint or args.save_model
        ):
            assert (
                args.aws_access_key_id is not None
                and args.aws_secret_access_key is not None
            )
    if args.sweep_id is not None:
        assert args.profile
        assert args.sweep_count is not None
        assert not args.resume_from_checkpoint
        if args.sweep_count > 1 and args.platform_type != PlatformType.LOCAL:
            assert args.checkpoint_path is None and args.model_path is None
        if args.save_checkpoint is None:
            args.save_checkpoint = False
        if args.save_model is None:
            args.save_model = False
        if args.sync_profile_live is None and args.profile:
            args.sync_profile_live = False
    else:
        if args.save_checkpoint is None:
            args.save_checkpoint = True
        if args.save_model is None:
            args.save_model = True
        if args.sync_profile_live is None and args.profile:
            args.sync_profile_live = True
    if not args.profile:
        assert args.sync_profile_live is None


def train(model_cls, create_training_context_fn, local_dir, wandb_project):
    parser = argparse.ArgumentParser(
        description="Training script for transformer model."
    )
    parser.add_argument("--train", type=str)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "--platform_type", type=PlatformType, default=PlatformType.LOCAL
    )
    parser.add_argument("--resume_from_checkpoint", type=lambda v: bool(strtobool(v)))
    parser.add_argument("--aws_access_key_id", type=str)
    parser.add_argument("--aws_secret_access_key", type=str)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--sweep_count", type=int, default=None)
    parser.add_argument("--save_code", type=lambda v: bool(strtobool(v)), default=False)
    parser.add_argument("--profile", type=lambda v: bool(strtobool(v)), default=True)
    parser.add_argument("--sync_profile_live", type=lambda v: bool(strtobool(v)))
    parser.add_argument("--save_checkpoint", type=lambda v: bool(strtobool(v)))
    parser.add_argument("--save_model", type=lambda v: bool(strtobool(v)))
    args = parser.parse_args()

    get_default_args(args, local_dir)

    if args.sweep_id is not None and int(os.getenv("LOCAL_RANK", "0")) == 0:
        wandb.agent(
            args.sweep_id,
            function=lambda: _train(
                args,
                model_cls,
                create_training_context_fn,
                local_dir,
                wandb_project,
            ),
            project=wandb_project,
            count=args.sweep_count,
        )
    else:
        _train(
            args,
            model_cls,
            create_training_context_fn,
            local_dir,
            wandb_project,
        )
