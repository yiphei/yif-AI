import os

from utils.train import train

try:
    from learned_dropout.model import LearnedDropoutTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import LearnedDropoutTransformer

if __name__ == "__main__":
    train(
        LearnedDropoutTransformer,
        f"{os.path.dirname(os.path.abspath(__file__))}/",
        "ultimate_attention_dropout_transformer_new_embed",
    )
