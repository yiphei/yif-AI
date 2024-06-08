from utils.train import train

try:
    from parallel_future_transformer.model import EncoderDecoderTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import EncoderDecoderTransformer

if __name__ == "__main__":
    train(
        EncoderDecoderTransformer,
        "parallel_future_transformer/",
        "three_times_transformer",
    )
