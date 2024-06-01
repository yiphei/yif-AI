from utils.train import train

try:
    from encoder_decoder_transformer.model import EncoderDecoderTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import EncoderDecoderTransformer

if __name__ == "__main__":
    train(
        EncoderDecoderTransformer,
        "encoder_decoder_transformer/",
        "serial_encoder_decoder_transformer",
    )
