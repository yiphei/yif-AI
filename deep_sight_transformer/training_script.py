from utils.train import train

try:
    from deep_sight_transformer.model import DeepSight
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import DeepSight

if __name__ == "__main__":
    train(
        DeepSight,
        "deep_sight_transformer/",
        "future_encoder_transformer",
    )
