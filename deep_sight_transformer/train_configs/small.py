# ModelConfig
CONTEXT_SIZE = 200
N_EMBED = 100
N_LAYER = 10
N_HEAD = 5
USE_BIAS = True
DROPOUT_RATE = 0
future_context_size = 10

# Training config
BATCH_SIZE = 15  # 50 when run on 1x A10
TRAIN_STEPS = 1000
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 1000

# Estimation config
EST_INTERVAL = 200
EST_STEPS = 20
