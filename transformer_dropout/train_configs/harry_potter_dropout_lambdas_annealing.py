# ModelConfig
CONTEXT_SIZE = 1024
N_EMBED = 400
N_LAYER = 4
N_HEAD = 4
BIAS = False
USE_LEARNED_DROPOUT = True
LEARNED_DROPOUT_CONFIG = {
                    "use_dropout_entropy_in_loss": True,
                    "use_dropout_l1_norm_in_loss": True,
                    "use_sigmoid_on_dropout_mask": True,
                    "use_canonical_entropy": False,
                    "use_detached_x_in_dropout_mask": False,
                    "A_param_config": {
                            "init_mean": 100000,
                            "init_std": 10, 
                            "optimizer_type": "ADAMW",
                            'lr': 1,
                    },
                    "B_param_config": {
                            "init_mean": 1.5707,
                            "init_std": 0.02,
                            "optimizer_type": "ADAMW",
                            },
                    "dropout_entropy_lambda": {
                            "min_lambda": 0.1,
                            "max_lambda": 5,
                            "coefficient": 0.0005,
                    },
                                        "dropout_l1_norm_lambda": {
                            "min_lambda": 0.1,
                            "max_lambda": 5,
                            "coefficient": 0.0005,
                    },
}

# Training config
BATCH_SIZE = 15
TRAIN_STEPS = 20000
LR = 6e-4
WARMUP_ITERS = 1000
MIN_LR = 6e-5
GRADIENT_ACCUMULATION_STEPS = 1 * 8
LR_DECAY_ITERS = 20000

# Estimation config
EST_INTERVAL = 100
EST_STEPS = 100
