# ModelConfig
CONTEXT_SIZE = 50
N_EMBED = 100
N_LAYER = 3
N_HEAD = 2
BIAS = False
USE_LEARNED_DROPOUT = False
DROPOUT_RATE = 0.1
BIAS = True

# Training config
BATCH_SIZE = 10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
