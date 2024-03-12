from contextlib import ExitStack, contextmanager, nullcontext
import torch

from utils.train import train
from utils.train_common import BatchStatsBase

try:
    from transformer_embed.model import DropoutTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import DropoutTransformer

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


def create_training_context(model, starting_training_step, device_type, ptdtype):
    autocast_context = create_autocast_context(device_type, ptdtype)

    @contextmanager
    def training_context(training_step, is_first_minibatch):
        with ExitStack() as stack:
            stack.enter_context(autocast_context())
            yield

    return training_context


if __name__ == "__main__":
    train(
        BatchStatsBase,
        DropoutTransformer,
        create_training_context,
        "transformer_embed/",
        "transformer_embed_2",
    )
