import torch

# ModelConfig
CONTEXT_SIZE = 100
N_EMBED = 90
N_LAYER = 3
N_HEAD = 3
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.1
LEARNED_DROPOUT_LAYERS = 1
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": False,
    "use_dropout_l1_norm_in_loss": True,
    "use_bias": False,
    "shift_init": torch.pi / 2,
    "use_detached_x_in_dropout_mask": False,
    "profile_dropout_mask": True,
    # "dropout_entropy_lambda": {
    #     "coefficient": 0.01,
    # },
}

# Training config
BATCH_SIZE = 15  # 50 when run on 1x A10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
