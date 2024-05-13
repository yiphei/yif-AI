# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 200
N_LAYER = 10
N_HEAD = 10
USE_BIAS = False
DROPOUT_RATE = 0

# Training config
BATCH_SIZE = 50
TRAIN_STEPS = 8000
LR = 3e-4
WARMUP_ITERS = 300
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 700000

# Estimation config
EST_INTERVAL = 500
EST_STEPS = 200
