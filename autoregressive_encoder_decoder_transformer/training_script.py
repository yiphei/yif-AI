import os

from utils.train import train

try:
    from autoregressive_encoder_decoder_transformer.model import \
        AutoregressiveEncoderDecoderTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import AutoregressiveEncoderDecoderTransformer

if __name__ == "__main__":
    train(
        AutoregressiveEncoderDecoderTransformer,
        f"{os.path.dirname(os.path.abspath(__file__))}/",
        "serial_encoder_decoder_transformer",
    )
