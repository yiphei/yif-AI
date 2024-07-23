import os

from utils.train import train

try:
    from deep_plan_transformer.model import DeepPlan
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import DeepPlan

if __name__ == "__main__":
    train(
        DeepPlan,
        f"{os.path.dirname(os.path.abspath(__file__))}/",
        "future_encoder_transformer",
    )
