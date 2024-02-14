from transformer_dropout.model import ModelConfig


import torch


from dataclasses import dataclass, field, fields


def require_prop_exception():
    raise ValueError("Missing required property")


@dataclass
class TrainConfig:
    DEVICE: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    MODEL_CONFIG: ModelConfig = field(default_factory=require_prop_exception)
    # Training
    BATCH_SIZE: int = field(
        default_factory=require_prop_exception
    )  # this will be scaled by GRADIENT_ACCUMULATION_STEPS
    TRAIN_STEPS: int = field(default_factory=require_prop_exception)
    GRADIENT_ACCUMULATION_STEPS: int = field(
        default_factory=require_prop_exception
    )  # used to simulate large batches
    # Optimizer
    LR: float = field(default=6e-4)  # max learning rate
    WEIGHT_DECAY: float = field(default=1e-1)
    BETA1: float = field(default=0.9)
    BETA2: float = field(default=0.95)
    DECAY_LR: bool = True
    WARMUP_ITERS: int = field(default_factory=require_prop_exception)
    LR_DECAY_ITERS: int = field(default_factory=require_prop_exception)
    MIN_LR: float = field(default=6e-5)
    # Estimation
    EST_INTERVAL: int = field(default_factory=require_prop_exception)
    EST_STEPS: int = field(default_factory=require_prop_exception)
    # Other
    DTYPE: str = field(
        default_factory=lambda: (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    )
    COMPILE: bool = True
    USE_DP: bool = False  # DataParallel
    USE_DDP: bool = True  # DistributedDataParallel

    def __post_init__(self):
        if self.USE_DDP and self.USE_DP:
            raise ValueError("cannot have both USE_DDP and USE_DP set to True")

    @classmethod
    def create_from_config_file(cls, config_file: str):
        config_dict = {}
        with open(config_file, "r") as file:
            exec(file.read(), {}, config_dict)
        # Filter out built-in items
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("__")}

        model_config_props = [f.name.upper() for f in fields(ModelConfig)]
        model_config_dict = {
            k.lower(): v for k, v in config_dict.items() if k in model_config_props
        }
        model_config = ModelConfig(**model_config_dict)

        config_dict = {
            k: v for k, v in config_dict.items() if k not in model_config_props
        }
        config_dict["MODEL_CONFIG"] = model_config
        return cls(**config_dict)