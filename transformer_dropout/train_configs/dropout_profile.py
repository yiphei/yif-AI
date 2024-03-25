# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.1
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": False,
    "use_sigmoid_on_dropout_mask": False,
    "use_canonical_entropy": False,
    "use_detached_x_in_dropout_mask": True,
    "A_param_config": {
        "init_mean": 10,
        "init_std": 0,
        "optimizer_type": "ADAMW",
    },
    "B_param_config": {
        "init_mean": 3.14,
        "init_std": 0,
        "optimizer_type": "ADAMW",
    },
    "dropout_entropy_lambda": {
        "min_lambda": 0.1,
        "max_lambda": 5,
        "coefficient": 0.0005,
    },
}

# Training config
BATCH_SIZE = 15 # 50 when run on 1x A10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
