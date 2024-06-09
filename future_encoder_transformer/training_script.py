from utils.train import train

try:
    from future_encoder_transformer.model import EncoderDecoderTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import EncoderDecoderTransformer

if __name__ == "__main__":
    train(
        EncoderDecoderTransformer,
        "future_encoder_transformer/",
        "future_encoder_transformer",
    )
