
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

    @classmethod
    def initialize(cls,train_config, model):
        return cls(
            model = model,
            train_config = train_config,
        )

    def add_mini_batch_stats(self, mini_batch_stats):
        return

    def mean(self):
        return

    def scale(self):
        return

    def get_wandb_batch_stats(self):
        return {}


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

            if (
                not model.config.use_cross_entropy_loss
                and model.config.use_new_output_layer
            ):
                accuracy = (
                    (logits.min(dim=-1).indices.view(-1) != yb.view(-1)).float().mean()
                )
            else:
                probs = F.softmax(logits, dim=-1)
                accuracy = (
                    (probs.max(dim=-1).indices.view(-1) != yb.view(-1)).float().mean()
                )

            accuracies[i] = accuracy
            losses[i] = loss

        new_data_iters.append(data_iter if original_data_iter != data_iter else None)
        mean_accuracies.append(accuracies.mean().item())
        mean_losses.append(losses.mean().item())
    model.train()
    return (mean_accuracies, mean_losses, new_data_iters)


if __name__ == "__main__":
    train(estimate_loss, BatchStats, DropoutTransformer)
