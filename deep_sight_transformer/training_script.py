from utils.train import train
import os

try:
    from deep_sight_transformer.model import DeepSight
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import DeepSight

if __name__ == "__main__":
    train(
        DeepSight,
        f"{os.path.dirname(os.path.abspath(__file__))}/",
        "future_encoder_transformer",
    )
