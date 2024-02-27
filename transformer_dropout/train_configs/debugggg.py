# ModelConfig
CONTEXT_SIZE = 3
N_EMBED = 2
N_LAYER = 1
N_HEAD = 1
BIAS = False
USE_LEARNED_DROPOUT = True
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": False,
    "use_canonical_entropy": True,
    "use_detached_x_in_dropout_mask": True,
    "a_param_mean": 100000,
    "a_param_std": 0.02,
    "dropout_entropy_lambda": {
        "max_lambda": 100,
    },
}
# Training config
BATCH_SIZE = 1
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 1
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
