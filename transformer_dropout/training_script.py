from contextlib import ExitStack, contextmanager, nullcontext

import torch

from utils.train import train

try:
    from transformer_dropout.model import DropoutTransformer
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
        DropoutTransformer,
        create_training_context,
        "transformer_dropout/",
        "transformer_dropout_5_attn_cosine",
    )
