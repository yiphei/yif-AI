import torch

# ModelConfig
CONTEXT_SIZE = 50
N_EMBED = 50
N_LAYER = 2
N_HEAD = 2
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.3
LEARNED_DROPOUT_LAYERS = 1
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": False,
    "use_bias": False,
    "shift_init": 1.2,
    "softmax_dim": 2,
    "n_heads": 2,
    "use_canonical_entropy": False,
    "rounding_type": 1,
    "sigmoid_slope": 30,
    "use_detached_x_in_dropout_mask": False,
    "profile_dropout_mask": False,
    "dropout_entropy_lambda": {
        "max_lambda": 1,
    },
}

# Training config
BATCH_SIZE = 10  # 50 when run on 1x A10
TRAIN_STEPS = 1000
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 1000

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
