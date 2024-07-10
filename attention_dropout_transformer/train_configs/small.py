import torch

# ModelConfig
CONTEXT_SIZE = 100
N_EMBED = 90
N_LAYER = 3
N_HEAD = 3
USE_BIAS = False
DROPOUT_RATE = 0
USE_DROPOUT_ENTROPY_IN_LOSS = True
USE_DROPOUT_L1_NORM_IN_LOSS = False
DROPOUT_ENTROPY_LAMBDA = {
    "max_lambda": 1,
    "exp_coefficient": 0.001,
}
ATTENTION_DROPOUT_CONFIG = {
    "use_bias": False,
    "softmax_dim": 1,
    "n_head": 3,
    "use_canonical_entropy": False,
    "use_detached_x_in_dropout_mask": False,
}

# Training config
BATCH_SIZE = 15  # 50 when run on 1x A10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
