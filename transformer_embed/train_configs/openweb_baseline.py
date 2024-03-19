# ModelConfig
CONTEXT_SIZE = 1024
N_EMBED = 768
N_LAYER = 12
N_HEAD = 12
BIAS = False
USE_NEW_OUTPUT_LAYER = False
USE_FINAL_LN_LAYER = True
DROPOUT_RATE = 0.1

# Training config
BATCH_SIZE = 39 # 18 is best for sagemaker, 39 is best for paperspace
TRAIN_STEPS = 22000
LR = 6e-4
WARMUP_ITERS = 2000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 2 * 8
LR_DECAY_ITERS = 22000

# Estimation config
EST_INTERVAL = 1000
EST_STEPS = 200