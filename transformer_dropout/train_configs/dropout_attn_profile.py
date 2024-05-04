import torch

# ModelConfig
CONTEXT_SIZE = 100
N_EMBED = 90
N_LAYER = 3
N_HEAD = 3
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0.1
LEARNED_DROPOUT_CONFIG = {
    "start_layer": 1,
    "end_layer": 3,
    "use_bias": False,
    "n_heads": 3,
    "profile_dropout_mask": False,
    "add_pos_embed": True,
    "order_type": 1,
    "sub_pos_embed": 1,
    "token_loss_type": 2,
    "token_loss_detach_type": 3,
}

# Training config
BATCH_SIZE = 15  # 50 when run on 1x A10
TRAIN_STEPS = 500
LR = 3e-4
WARMUP_ITERS = 50
MIN_LR = 3e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_DECAY_ITERS = 500

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 20
