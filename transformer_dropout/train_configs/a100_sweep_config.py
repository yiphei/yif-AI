import torch

# ModelConfig
CONTEXT_SIZE = 100
N_EMBED = 147
N_LAYER = 6
N_HEAD = 7
BIAS = False
USE_LEARNED_DROPOUT = True
DROPOUT_RATE = 0
LEARNED_DROPOUT_CONFIG = {
    "start_layer": 1,
    "end_layer": 6,
    "use_bias": False,
    "n_heads": 7,
    "profile_dropout_mask": False,
    "add_pos_embed": False,
    "order_type": 1,
    "sub_pos_embed": 1,
    "add_ln_before_pred_ff": True,
    
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
