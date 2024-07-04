import torch

# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 144
N_LAYER = 26
N_HEAD = 9
USE_BIAS = False
DROPOUT_RATE = 0
START_LAYER = 1
END_LAYER = 26
USE_DROPOUT_ENTROPY_IN_LOSS = True
USE_DROPOUT_L1_NORM_IN_LOSS = True
ATTENTION_DROPOUT_CONFIG = {
    "use_bias": False,
    "shift_init": torch.pi / 2,
    "softmax_dim": 1,
    "n_head": 9,
    "use_canonical_entropy": False,
    "rounding_type": 2,
    "use_detached_x_in_dropout_mask": False,
}

# Training config
BATCH_SIZE = 50  # 50 when run on 1x A10
TRAIN_STEPS = 9000
LR = 9e-4
WARMUP_ITERS = 300
MIN_LR = 9e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 700000

# Estimation config
EST_INTERVAL = 500
EST_STEPS = 200
