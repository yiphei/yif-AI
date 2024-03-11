# ModelConfig
CONTEXT_SIZE = 1024
N_EMBED = 400
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = False
DROPOUT_RATE = 0.1

# Training config
BATCH_SIZE = 10
TRAIN_STEPS = 20000
LR = 6e-4
WARMUP_ITERS = 1000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 1 * 8
LR_DECAY_ITERS = 20000

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 100
