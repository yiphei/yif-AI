# ModelConfig
context_size = 80
n_embed = 100
n_layer = 3
n_head = 2
bias = False
use_dropout_entropy_in_loss = True
use_dropout_l1_norm_in_loss = True
use_learned_dropout = True

# Training config
batch_size = 10
training_steps = 1000
lr = 3e-4
warmup_iters = 50
min_lr = 6e-5
gradient_accumulation_steps = 7

# Estimation config
est_interval = 200
est_steps = 100