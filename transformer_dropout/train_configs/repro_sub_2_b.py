# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": False,
    "use_dropout_l1_norm_in_loss": True,
    "use_sigmoid_on_dropout_mask": True,
    "use_canonical_entropy": True,
    "use_detached_x_in_dropout_mask": True,
    "A_param_config": {
        "init_mean": 0,
        "init_std": 0,
        "optimizer_type": "SGD",
    },
    "B_param_config": {
        "init_mean": 1.3033,
        "init_std": 0,
        "optimizer_type": "ADAMW",
        "lr": 0.03,
    },
    "dropout_l1_norm_lambda": {
        "max_lambda": 2,
    },
    "profile_dropout_mask": True,
}

# Training config
BATCH_SIZE = 50  # 50 when run on 1x A10
TRAIN_STEPS = 500
LR = 3e-3
WARMUP_ITERS = 50
MIN_LR = 3e-4
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 50
EST_STEPS = 50
