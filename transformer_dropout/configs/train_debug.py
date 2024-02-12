# ModelConfig
context_size = 100
n_embed = 100
n_layer = 3
n_head = 2
bias = False

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