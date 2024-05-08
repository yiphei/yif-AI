# ModelConfig
CONTEXT_SIZE = 1024
N_EMBED = 768
N_LAYER = 12
N_HEAD = 12
BIAS = False
USE_LEARNED_DROPOUT = True
LEARNED_DROPOUT_CONFIG = {
    "use_dropout_entropy_in_loss": True,
    "use_dropout_l1_norm_in_loss": True,
    "use_canonical_entropy": False,
    "use_detached_x_in_dropout_mask": True,
    "a_param_mean": 100000,
    "a_param_std": 0.02,
    "dropout_entropy_lambda": {
        "max_lambda": 2,
    },
    "dropout_l1_norm_lambda": {
        "max_lambda": 2,
    },
}


# Training config
BATCH_SIZE = 7  # workst with 6 and 7, but 12 and 8 failed with 2 ml.p4d.24 instances
TRAIN_STEPS = 22000
LR = 6e-4
WARMUP_ITERS = 2000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = (
    2 * 8
)  # gpt2_baseline has 5*8 but this uses 2*8 because it has bigger memory footprint
LR_DECAY_ITERS = 22000

# Estimation config
EST_INTERVAL = 1000
EST_STEPS = 200
