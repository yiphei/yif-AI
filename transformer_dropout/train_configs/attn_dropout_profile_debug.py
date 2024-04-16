# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.3
LEARNED_DROPOUT_LAYERS = 4
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": False,
    "l1_norm_pos": 2,
    "use_bias": False,
    "use_detached_x_in_dropout_mask": False,
    "profile_dropout_mask": True,
    "dropout_entropy_lambda": {
        "coefficient": 0.0006,
    },
}

# Training config
BATCH_SIZE = 50  # 50 when run on 1x A10
TRAIN_STEPS = 3000
LR = 3e-4
WARMUP_ITERS = 100
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 3000

# Estimation config
EST_INTERVAL = 200
EST_STEPS = 100