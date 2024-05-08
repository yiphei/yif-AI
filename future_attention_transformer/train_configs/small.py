# ModelConfig
CONTEXT_SIZE = 100
N_EMBED = 90
N_LAYER = 3
N_HEAD = 3
USE_BIAS = False
DROPOUT_RATE = 0.1
LEARNED_DROPOUT_CONFIG = {
    "start_layer": 3,
    "use_bias": False,
    "n_heads": 3,
    "future_dim": 10,
    "profile_dropout_mask": False,
    "mask_loss_type": 1,
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