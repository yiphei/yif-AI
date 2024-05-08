from utils.train import train

try:
    from baseline_transformer.model import TransformerModel
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import TransformerModel





if __name__ == "__main__":
    train(
        TransformerModel,
        "baseline_transformer/",
        "baseline_transformer",
    )
