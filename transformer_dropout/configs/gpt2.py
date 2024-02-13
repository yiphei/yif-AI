# ModelConfig
CONTEXT_SIZE = 1024
N_EMBED = 768
N_LAYER = 12
N_HEAD = 12
BIAS = False
USE_DROPOUT_ENTROPY_IN_LOSS = False
USE_DROPOUT_L1_NORM_IN_LOSS = False
USE_LEARNED_DROPOUT = False
DROPOUT_RATE = 0.1

# Training config
BATCH_SIZE = 12
TRAIN_STEPS = 600000
LR = 6e-4
WARMUP_ITERS = 2000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 5 * 8
LR_DECAY_ITERS = 600000

# Estimation config
EST_INTERVAL = 1000
EST_STEPS = 200
