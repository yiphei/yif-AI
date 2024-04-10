from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Optional

import torch

from utils.train import train
from utils.train_common import BatchStatsBase

try:
    from transformer_dropout.model import DropoutTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import DropoutTransformer


@dataclass
class BatchStats(BatchStatsBase):
    running_sigmoid_entropy: Optional[float]
    running_entropy: Optional[float]
    running_l1_norm: Optional[float]
    sigmoid_entropy: Optional[float] = None
    entropy: Optional[float] = None
    dropout_l1_norm: Optional[float] = None
    entropy_coefficient: Optional[float] = None
    dropout_l1_norm_coefficient: Optional[float] = None

    @classmethod
    def initialize(cls, train_config, model):
        return cls(
            running_sigmoid_entropy=(
                0.0 if train_config.model_config.use_learned_dropout else None
            ),
            running_entropy=(
                0.0 if train_config.model_config.use_learned_dropout else None
            ),
            running_l1_norm=(
                0.0 if train_config.model_config.use_learned_dropout else None
            ),
            model=model,
            train_config=train_config,
        )

    def add_mini_batch_stats(self, mini_batch_stats):
        (
            entropy,
            dropout_l1_norm,
            entropy_coefficient,
            dropout_l1_norm_coefficient,
            sigmoid_entropy,
        ) = mini_batch_stats
        self.sigmoid_entropy = sigmoid_entropy
        self.entropy = entropy
        self.dropout_l1_norm = dropout_l1_norm
        self.entropy_coefficient = entropy_coefficient
        self.dropout_l1_norm_coefficient = dropout_l1_norm_coefficient

    def scale(self):
        if self.train_config.model_config.use_learned_dropout:
            self.running_entropy += (
                self.entropy.item() / self.train_config.gradient_accumulation_steps
            )
            self.running_sigmoid_entropy += (
                self.sigmoid_entropy.item()
                / self.train_config.gradient_accumulation_steps
            )
            self.running_l1_norm += (
                self.dropout_l1_norm.item()
                / self.train_config.gradient_accumulation_steps
            )

    def get_wandb_batch_stats(self):
        return {
            "dropout_sigmoid_entropy": self.running_sigmoid_entropy,
            "dropout_entropy": self.running_entropy,
            "dropout_l1_norm": self.running_l1_norm,
            "mean_dropout_near_one_percent": self.model.get_mean_dropout_near_one_percent(),
            "mean_dropout_near_zero_percent": self.model.get_mean_dropout_near_zero_percent(),
            "dropout_entropy_coefficient": self.entropy_coefficient,
            "dropout_l1_norm_coefficient": self.dropout_l1_norm_coefficient,
        }


def create_autocast_context(device_type, ptdtype):
    @contextmanager
    def autocast_context():
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )
        with ctx:
            yield

    return autocast_context


def create_training_step_context(starting_training_step, model):
    @contextmanager
    def training_step_context(training_step, is_first_minibatch):
        if model.config.use_learned_dropout and model.training and is_first_minibatch:
            assert (
                model.training_step == training_step - 1
                if training_step != starting_training_step
                else model.training_step is None
            )
            model.training_step = training_step
        yield

    return training_step_context


def create_training_context(model, starting_training_step, device_type, ptdtype):
    autocast_context = create_autocast_context(device_type, ptdtype)
    entropy_lambda_context = create_training_step_context(starting_training_step, model)

    @contextmanager
    def training_context(training_step, is_first_minibatch):
        with ExitStack() as stack:
            stack.enter_context(autocast_context())
            stack.enter_context(
                entropy_lambda_context(training_step, is_first_minibatch)
            )
            yield

    return training_context


if __name__ == "__main__":
    train(
        BatchStats,
        DropoutTransformer,
        create_training_context,
        "transformer_dropout/",
        "transformer_dropout_5_attn",
    )
