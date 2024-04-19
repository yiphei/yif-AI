import torch

# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.3
LEARNED_DROPOUT_LAYERS = 1
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": False,
    "use_bias": False,
    "shift_init": torch.pi/2,
    "softmax_dim": 2,
    "n_heads": 4,
    "use_canonical_entropy": False,
    "rounding_type": 1,
    "sigmoid_slope": 100,
    "use_detached_x_in_dropout_mask": False,
    "profile_dropout_mask": False,
    "dropout_entropy_lambda": {
        "max_lambda": 1,
    },
}

# Training config
BATCH_SIZE = 50  # 50 when run on 1x A10
TRAIN_STEPS = 1000
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 1000

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
