
import torch
from utils.train import train
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
try:
    from transformer_dropout.model import DropoutTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import DropoutTransformer


@dataclass
class BatchStats:
    model: DropoutTransformer
    train_config: dataclass
    running_entropy: Optional[float]
    running_l1_norm: Optional[float]
    entropy: Optional[float] = None
    dropout_l1_norm: Optional[float] = None
    entropy_coefficient: Optional[float] = None
    dropout_l1_norm_coefficient: Optional[float] = None


    @classmethod
    def initialize(cls,train_config, model):
        return cls(
            running_entropy = 0.0 if train_config.model_config.use_learned_dropout else None,
            running_l1_norm = 0.0 if train_config.model_config.use_learned_dropout else None,
            model = model,
            train_config = train_config,
        )

    def add_mini_batch_stats(self, mini_batch_stats):
        entropy, dropout_l1_norm, entropy_coefficient, dropout_l1_norm_coefficient = mini_batch_stats
        self.entropy = entropy
        self.dropout_l1_norm = dropout_l1_norm
        self.entropy_coefficient = entropy_coefficient
        self.dropout_l1_norm_coefficient = dropout_l1_norm_coefficient

    def mean(self):
        if self.train_config.model_config.use_learned_dropout:
            self.dropout_l1_norm = self.dropout_l1_norm.mean()
            self.entropy = self.entropy.mean()

    def scale(self):
        if self.train_config.model_config.use_learned_dropout:
            self.running_entropy += (
                self.entropy.item() / self.train_config.gradient_accumulation_steps
            )
            self.running_l1_norm += (
                self.dropout_l1_norm.item()
                / self.train_config.gradient_accumulation_steps
            )

    def get_wandb_batch_stats(self):
        A_mean, A_std = self.model.get_A_stats()
        B_mean, B_std = self.model.get_B_stats()
        return {
                "dropout_entropy": self.running_entropy,
                "dropout_l1_norm": self.running_l1_norm,
                "mean_dropout_near_one_percent": self.model.get_mean_dropout_near_one_percent(),
                "mean_dropout_near_zero_percent": self.model.get_mean_dropout_near_zero_percent(),
                "dropout_entropy_coefficient": self.entropy_coefficient,
                "dropout_l1_norm_coefficient": self.dropout_l1_norm_coefficient,
                "A_mean": A_mean,
                "A_std": A_std,
                "B_mean": B_mean,
                "B_std": B_std,
            }


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
    get_data_batch_loader_fn
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
            xb, yb, new_data_iter = get_data_batch_loader_fn(
                data_iter, data_loader, data_sampler, iter_num, device
            )
            if new_data_iter is not None:
                data_iter = new_data_iter

            with ctx(i, False):
                logits, loss, _ = model(xb, yb)
            if using_DP:
                loss = loss.mean()
            losses[i] = loss

            probs = F.softmax(logits, dim=-1)
            accuracy = (
                (probs.max(dim=-1).indices.view(-1) != yb.view(-1)).float().mean()
            )
            accuracies[i] = accuracy

        new_data_iters.append(data_iter if original_data_iter != data_iter else None)
        mean_accuracies.append(accuracies.mean().item())
        mean_losses.append(losses.mean().item())
    model.train()
    return (mean_accuracies, mean_losses, new_data_iters)


if __name__ == "__main__":
    train(estimate_loss, BatchStats, DropoutTransformer)
