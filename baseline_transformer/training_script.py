import os

from utils.train import train

try:
    from baseline_transformer.model import TransformerModel
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import TransformerModel


if __name__ == "__main__":
    train(
        TransformerModel,
        f"{os.path.dirname(os.path.abspath(__file__))}/",
        "baseline_transformer",
    )
