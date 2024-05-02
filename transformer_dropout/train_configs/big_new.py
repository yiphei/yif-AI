import torch

# ModelConfig
CONTEXT_SIZE = 100
N_EMBED = 150
N_LAYER = 6
N_HEAD = 5
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.3
LEARNED_DROPOUT_CONFIG = {
    "start_layer": 1,
    "end_layer": 6,
    "use_bias": False,
    "n_heads": 5,
    "profile_dropout_mask": False,
}

# Training config
BATCH_SIZE = 50  # 50 when run on 1x A10
TRAIN_STEPS = 10000
LR = 3e-4
WARMUP_ITERS = 300
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 700000

# Estimation config
EST_INTERVAL = 500
EST_STEPS = 200
