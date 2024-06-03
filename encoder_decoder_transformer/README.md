# Auto-regressive Encoder-Decoder Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Current SOTA LLMs are all decoder-only models. Here, a new end-to-end auto-regressive encoder-decoder transformer is presented that outperforms the canonical dencoder-only transformer.

## Motivations

Though transformer models have a singular objective function (next token prediction), the attention mechanism implicitly introduces another one: general (contextual) understanding. Indeed, there is empirical evidence that the earlier layers of a model focus more on understanding linguistic features and the later layers more on task-specific features and thus more on prediction. In a decoder-only transformer, the model learns when to switch from understanding to predicting. In a encoder-decoder model, it is more imposed by design: the encoder generates an output and the decoder has to continuosly attend to the encoder output. Therefore, the encoder focuses more on local understanding and the decoder more on prediction.

The canonical enoder-decoder transformer is used for sequence-to-sequence tasks, like machine translation. Instead, the model here is used auto-regressively end-to-end. This, along with novel components described in sections that follow, beats the baseline of a decoder-only transformer.

## Architecture

At the high level, the architecture re-implements the canonical encoder-decoder model but for auto-regressive language generation. Furthermore, an additional "encoder" loss and a positional embedding operation are added and demonstrate improved performance.

### Encoder-Decoder

In the canonical encoder-decoder transformer, the encoder runs once on an input, and then the decoder runs auto-regressively on its own ouput while attending to the encoder output. It looks like this

<div align="center">
  <img src="assets/diagram.png" alt="diagram" width="500">
  <br>
  <em>From the <strong>Attention is All You Need</strong> paper. The modern encoder-decoder remains largely the same as the one above, with the major difference being the relocation of <strong>Norm</strong> to before attention and feed forward blocks.</em>
</div>
<br>

To use this architecture for an end-to-end auto-regressive task, the encoder and decoder run together serially. The encoder generates an output and the decoder generates the next token while attending to the encoder output. Then, the input is updated with the ouput and fed back to the model, which reruns the encoder and decoder. To make this work, the encoder attention has to be masked. Visually, the entire model looks like this

<div align="center">
    <img src="assets/new_diagram.png"
         alt="diagram" width="500">
    <br>
    <em>Just like the one before, <strong>Norm</strong> should be moved to before each block.</em>
</div>
<br>

Another way to think about it is this. It takes a regular decoder-only model with $L$ layers and makes the last $L_{decoder}$ layers do both self-attention and cross-attention on the output of the first $L_{encoder}$ layers.

### Encoder loss

In the canonical decoder-encoder model, the loss function is evaluated over the decoder's output (itself being a function of the encoder's output). In this implementation, a new loss on the encoder is introduced. The idea here is similar to weight tying the output layer with the token embedding. Weigh tying increases update frequency & magnitude of embedding weights, which then better compresses the entire forward pass into embedding weights. Ultimately, this permits hidden layers to compute more complex representations. The same effect can be achieved (in addition to output layer weight tying) with the encoder loss described as follows. Given the original input embedding ${E}$, you calculate the cumulative average along the token dimension (i.e. T dimension). Then, the encoder loss is calculated as a disaffinity score between the cumulative average and the encoder output. Stated more formally, you have

$$
\begin{aligned}
& out_{enc} \coloneqq \text{encoder output} \\
& E \coloneqq \text{model input embedding, comprised of token and positional embedding} \\
& E_{avg\\\_sum} \coloneqq \text{cumulative average of }E\text{ along T dimension, where } E_{avg\\\_sum_{(i,j)}} = \frac{1}{i} \sum_{z}^{i}E_{z,j} \\
& encoder\\\_loss = disaffinity\\\_score(out_{enc}, E_{avg\\\_sum})
\end{aligned}
$$

Two disaffinity scores are experimented with. One is mean squared error, and the other is cosine dissimilarity. Cosine dissimilarity is cosine similarity normalized such that zero represents the most similarity and 1 most dissimilarity. So the encoder loss with MSE is just

$$encoder\\\_loss = MSE(out_{enc}, E_{avg\\\_sum})$$

and the encoder loss with cosine dissimilarity is

$$encoder\\\_loss = 1- \frac{cosine\\\_similarity(out_{enc}, E_{avg\\\_sum}) + 1}{2}$$

### Positional embedding substraction

Before the output layer, positional embedding of the "next tokens" are subtracted from the latent representation. Again, the idea here is similar to weight tying of token embedding but for positional embedding. By subtracting positional embedding, you increase update frequency & magnitude of positional weights. When coupled with token embedding weight tying, this should improve latent separation between token and positional embedding.

## Analysis/experiments

All training runs below were done on a wikipedia dataset for 9k steps on a single A100 GPU.

The MSE encoder loss did better than cosine dissimilarity. Both types of encoder loss did better than without it.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/g_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/g_val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/g_encoder_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **with cosine-dissimilarity encoder loss** [(config)](#with-cosine-dissimilarity-encoder-loss) | 2.984 | 3.445 | 8.285e-9 |
| **with MSE encoder loss** [(config)](#with-mse-encoder-loss) | **2.981** | **3.439** | 3.656e-9 |
| **with no encoder loss** [(config)](#with-no-encoder-loss) | 2.997 | 3.449 | N/A |

Adding the positional embedding of the next tokens to the decoder helped the train loss but was detrimental to validation loss.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/g_pos_train.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/g_pos_val.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/g_pos_encoder.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **add_pos_embed_to_decoder=True** [(config)](#add_pos_embed_to_decodertrue) | **2.981** | 3.439 | 3.656e-9 |
| **add_pos_embed_to_decoder=False** [(config)](#add_pos_embed_to_decoderfalse) | 2.99 | **3.435** | 4.471e-9 |

Compared to a canonical decoder-only transformer (baseline), the new model outperformed it in validation loss but underperformed in train loss. Both completed in a similar amount of time with similar memory demands. The baseline did have more parameters because it was hard to exactly match the new model's.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/g_final_train.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/g_final_val.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Size (params) |
|---|----------|----------|----------|
| **parallel encoder-decoder transformer** [(config)](#parallel-encoder-decoder-transformer) | 2.981 | **3.439** | 15,698,400 |
| **baseline** [(config)](#baseline) | **2.934** | 3.449 | 15,850,380 |

## Next steps

These are some improvements to look forward to:
- instead of the encoder and decoder having equal depth, it would be better for the model to learn what depth is best for either 
- instead of MSE and cosine dissimilarity, some other disaffinity scores should be experimented with
- the cumulative embedding average $E_{avg\\\_sum}$ assumes equal contribution from every preceding token, so a different aggregation might be better (maybe convolution?)
- the first `decoder_x` is initialized with a feed forward layer on $E$. Ideally, the decoder would have its own embeddings, but that would add too many parameters. A different way to initialize `decoder_x` should be explored
- try bigger models, at least GPT-2 size

## Conclusions
TODO

---
## Appendix
### Run configs
#### "with cosine-dissimilarity encoder loss"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 5,
                  'n_embed': 200,
                  'n_layer': 5,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'add_pos_embed_to_decoder': True,
                  'sub_pos_embed_to_decoder': 1,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 3,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 3,
                  'add_ln_before_decoder_ff': False,
                  'encoder_embed_loss_coeff': 2,
                  'encoder_embed_detach_type': 3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```
#### "with MSE encoder loss"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 5,
                  'n_embed': 200,
                  'n_layer': 5,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'add_pos_embed_to_decoder': True,
                  'sub_pos_embed_to_decoder': 1,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 2,
                  'add_ln_before_decoder_ff': False,
                  'encoder_embed_loss_coeff': 0.25,
                  'encoder_embed_detach_type': 3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```
#### "with no encoder loss"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 5,
                  'n_embed': 200,
                  'n_layer': 5,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'add_pos_embed_to_decoder': True,
                  'sub_pos_embed_to_decoder': 1,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'use_ln_on_encoder_out': None,
                  'encoder_embed_loss_type': 1,
                  'add_ln_before_decoder_ff': False},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```

#### "add_pos_embed_to_decoder=True"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 5,
                  'n_embed': 200,
                  'n_layer': 5,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'add_pos_embed_to_decoder': True,
                  'sub_pos_embed_to_decoder': 1,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 2,
                  'add_ln_before_decoder_ff': False,
                  'encoder_embed_loss_coeff': 0.25,
                  'encoder_embed_detach_type': 3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "add_pos_embed_to_decoder=False"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 5,
                  'n_embed': 200,
                  'n_layer': 5,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'add_pos_embed_to_decoder': False,
                  'sub_pos_embed_to_decoder': 1,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 2,
                  'add_ln_before_decoder_ff': False,
                  'encoder_embed_loss_coeff': 0.25,
                  'encoder_embed_detach_type': 3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "parallel encoder-decoder transformer"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 5,
                  'n_embed': 200,
                  'n_layer': 5,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'add_pos_embed_to_decoder': True,
                  'sub_pos_embed_to_decoder': 1,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 2,
                  'add_ln_before_decoder_ff': False,
                  'encoder_embed_loss_coeff': 0.25,
                  'encoder_embed_detach_type': 3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```
#### "baseline"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 9000,
 'est_interval': 500,
 'model_config': {'n_head': 10,
                  'n_embed': 210,
                  'n_layer': 10,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```