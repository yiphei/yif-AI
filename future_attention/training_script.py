from utils.train import train

try:
    from future_attention.model import FutureAttentionTransformer
except ImportError:
    # I only upload the direct parent module to sagemaker, so I need a different import path
    from model import FutureAttentionTransformer

if __name__ == "__main__":
    train(
        FutureAttentionTransformer,
        "future_attention/",
        "future_attention_transformer_v1",
    )
