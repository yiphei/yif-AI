import os

from utils.train import train

try:
    from attention_dropout_transformer.model import AttentionDropoutTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import AttentionDropoutTransformer

if __name__ == "__main__":
    train(
        AttentionDropoutTransformer,
        f"{os.path.dirname(os.path.abspath(__file__))}/",
        "ultimate_attention_dropout_transformer_diff_loc",
    )
