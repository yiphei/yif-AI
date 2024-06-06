# ModelConfig
CONTEXT_SIZE = 4
N_EMBED = 6
N_LAYER = 3
N_HEAD = 2
USE_BIAS = False
DROPOUT_RATE = 0
FUTURE_DIM = 5
use_ln_on_up_future = False

# Training config
BATCH_SIZE = 2  # 50 when run on 1x A10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 1
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
