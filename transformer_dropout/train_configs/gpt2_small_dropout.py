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
}

# Training config
BATCH_SIZE = 6  # works at 6, but failed at 9 and 7 with 1 instance of ml.p4d.24
TRAIN_STEPS = 600000
LR = 6e-4
WARMUP_ITERS = 2000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = (
    1 * 8
)  # gpt2_baseline has 5*8 but this uses 2*8 because it has bigger memory footprint
LR_DECAY_ITERS = 600000

# Estimation config
EST_INTERVAL = 1000
EST_STEPS = 200
