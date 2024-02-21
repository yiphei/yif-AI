# ModelConfig
CONTEXT_SIZE = 1024
N_EMBED = 768
N_LAYER = 12
N_HEAD = 12
BIAS = False
USE_DROPOUT_ENTROPY_IN_LOSS = True
USE_DROPOUT_L1_NORM_IN_LOSS = True
USE_LEARNED_DROPOUT = True
DROPOUT_ENTROPY_LAMBDA = 2
DROPOUT_L1_NORM_LAMBDA = 2

# Training config
BATCH_SIZE = 7  # workst with 6 and 7, but 12 and 8 failed with 2 ml.p4d.24 instances
TRAIN_STEPS = 600000
LR = 6e-4
WARMUP_ITERS = 2000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = (
    2 * 8
)  # gpt2_baseline has 5*8 but this uses 2*8 because it has bigger memory footprint
LR_DECAY_ITERS = 600000

# Estimation config
EST_INTERVAL = 1000
EST_STEPS = 200
