# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.1
LEARNED_DROPOUT_LAYERS = 3
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": False,
    "use_dropout_l1_norm_in_loss": False,
    "l1_norm_pos": 2,
    "use_bias": False,
    "use_noise": True,
    "use_detached_x_in_dropout_mask": False,
    "profile_dropout_mask": True,
    # "dropout_entropy_lambda": {
    #     "coefficient": 0.01,
    # },
}

# Training config
BATCH_SIZE = 50  # 50 when run on 1x A10
TRAIN_STEPS = 4000
LR = 3e-3
WARMUP_ITERS = 200
MIN_LR = 3e-4
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 4000

# Estimation config
EST_INTERVAL = 400
EST_STEPS = 100
