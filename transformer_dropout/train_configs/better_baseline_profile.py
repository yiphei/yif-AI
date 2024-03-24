# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 5
N_HEAD = 5
BIAS = False
USE_LEARNED_DROPOUT = False
DROPOUT_RATE = 0.1
BIAS = True
USE_FLASH = False

# Training config
BATCH_SIZE = 15
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20