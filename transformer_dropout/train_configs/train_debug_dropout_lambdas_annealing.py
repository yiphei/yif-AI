# ModelConfig
CONTEXT_SIZE = 50
N_EMBED = 100
N_LAYER = 3
N_HEAD = 2
BIAS = False
USE_DROPOUT_ENTROPY_IN_LOSS = True
USE_DROPOUT_L1_NORM_IN_LOSS = True
USE_LEARNED_DROPOUT = True
DROPOUT_ENTROPY_LAMBDA = {"min_lambda": 0.1, "max_lambda": 2, "coefficient": 0.001}
DROPOUT_L1_NORM_LAMBDA = {"min_lambda": 0.1, "max_lambda": 2, "coefficient": 0.001}
USE_CANONICAL_ENTROPY = True
USE_DETACHED_X_IN_DROPOUT_MASK = True

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
