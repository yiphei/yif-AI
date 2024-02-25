# ModelConfig
CONTEXT_SIZE = 50
N_EMBED = 100
N_LAYER = 3
N_HEAD = 2
BIAS = False
USE_LEARNED_DROPOUT = True
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": True,
    "use_canonical_entropy": True,
    "use_detached_x_in_dropout_mask": True,
    "dropout_entropy_lambda": {
        "min_lambda": 0.1,
        "max_lambda": 2,
        "coefficient": 0.001,
    },
    "dropout_l1_norm_lambda": {
        "min_lambda": 0.1,
        "max_lambda": 2,
        "coefficient": 0.001,
    },
}

# Training config
BATCH_SIZE = 10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
