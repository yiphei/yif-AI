# Auto-regressive Encoder-Decoder Transformer
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Current SOTA LLMs are all decoder-only models. Here, a new end-to-end auto-regressive encoder-decoder transformer is presented that outperforms, with fewer parameters, the canonical decoder-only transformer.

## Motivations

Though transformer models have a singular objective function (next token prediction), the attention mechanism implicitly introduces another one: general (contextual) understanding. Indeed, empirical evidence shows that the earlier layers of a model focus more on general linguistic features (when applied to language) and the later layers more on task-specific features and thus more on prediction. In a decoder-only transformer, the model learns when to switch from understanding to predicting. In an encoder-decoder model, it is more imposed by design: the encoder generates an output and the decoder has to continuously attend to the encoder output. Therefore, the encoder focuses more on understanding, and the decoder more on prediction. I was interested to discover how an encoder-decoder model would fare on auto-regressive language generation.

The canonical encoder-decoder transformer is used for sequence-to-sequence tasks, like machine translation. Instead, the model here is used auto-regressively end-to-end. This, along with novel components described in sections that follow, beats – with fewer parameters – the baseline of a decoder-only transformer.

## Architecture

At the high level, the architecture re-implements the canonical encoder-decoder model but for auto-regressive language generation. Furthermore, an additional "encoder" loss and a positional embedding operation are added and demonstrate improved performance.

### Encoder-Decoder

In the canonical encoder-decoder transformer, the encoder runs once on an input, and then the decoder runs auto-regressively on its own output while attending to the encoder output. It looks like this

<div align="center">
  <img src="assets/self_canon_diagram.svg" alt="diagram" width="500">
</div>
<br>

To use this architecture for an end-to-end auto-regressive task, the encoder and decoder are adapted to run together serially. The encoder generates an output and the decoder generates the next token while attending to the encoder output. When a new input is formed with the last decoder output, it gets fed back to the model, which reruns the encoder and decoder. To make this work, the encoder's attention has to be masked. The new architecture looks like this

<div align="center">
    <img src="assets/self_new_diagram.svg"
         alt="diagram" height="900">
</div>
<br>

Another way to think about it is this. It takes a regular decoder-only architecture with $L$ layers and makes the last $L_{decoder}$ layers do both self-attention and cross-attention on the output of the first $L_{encoder}$ layers. When transitioning from encoder to decoder, the input to the first decoder layer is a linear pass on the encoder output.

For simplicity, the new architecture consists of an equal number of encoder and decoder layers.

### Positional embedding subtraction

Before the (decoder) output layer, the positional embedding of the "next tokens" is subtracted from the latent representation. Again, the idea here is similar to weight tying of token embedding but for positional embedding. By subtracting positional embedding, you increase update frequency & magnitude of positional weights. When coupled with token embedding weight tying, this should improve latent separation between token and positional embedding (i.e. more contrastive learning).

<div align="center">
    <img src="assets/pos_diagram.svg"
         alt="diagram" height="300">
</div>

### Encoder loss

In the canonical decoder-encoder model, the loss function is evaluated over the decoder's output (itself being a function of the encoder's output). In this implementation, a new loss on the encoder is introduced, in addition to the regular loss. The idea here is similar to weight tying the output layer with the token embedding. Weigh tying increases update frequency & magnitude of embedding weights, which then better compresses the entire forward pass into embedding weights. Ultimately, this permits hidden layers to compute more complex representations. The same effect can be achieved on token weights (in addition to output layer weight tying) **and on positional weights** with the encoder loss described as follows. Given the full embedding (token + positional) ${E}$ of the input, you calculate the cumulative average along the token dimension (i.e. T dimension). Then, the encoder loss is calculated as a disaffinity score between the cumulative average and the encoder output. Stated more formally, you have

$$
\begin{aligned}
& out_{enc} \coloneqq \text{encoder output (detached)} \\
& E \coloneqq \text{model input embedding, comprised of token and positional embedding} \\
& E_{avg\\\_sum} \coloneqq \text{cumulative average of }E\text{ along T dimension, where } E_{avg\\\_sum_{(i,j)}} = \frac{1}{i} \sum_{z}^{i}E_{z,j} \\
& encoder\\\_loss = disaffinity\\\_score(out_{enc}, E_{avg\\\_sum})
\end{aligned}
$$

Two disaffinity scores are experimented with. One is mean squared error, and the other is cosine dissimilarity. Cosine dissimilarity is cosine similarity normalized such that zero represents the most similarity and 1 most dissimilarity. So the encoder loss with MSE is just

$$encoder\\\_loss = MSE(out_{enc}, E_{avg\\\_sum})$$

and the encoder loss with cosine dissimilarity is

$$encoder\\\_loss = 1- \frac{cosine\\\_similarity(out_{enc}, E_{avg\\\_sum}) + 1}{2}$$

## Results

> All training runs below were done on a wikipedia dataset for 9k steps on a single A100 GPU, unless otherwise stated.
> 
> Implementation of decoder-only transformer model (baseline) can be found in the `baseline_transformer` directory in this repo

The MSE encoder loss performed better than cosine dissimilarity in validation loss but worse in train loss. Both types of encoder loss did better than without it.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/e_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/e_val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/e_encoder_loss_2.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **with cosine-dissimilarity encoder loss** [(config)](#with-cosine-dissimilarity-encoder-loss) | **2.993** | 3.387 | 8.564e-9 |
| **with MSE encoder loss** [(config)](#with-mse-encoder-loss) | 2.998 | **3.385** | 4.138e-9 |
| **no encoder loss and no pos sub** [(config)](#no-encoder-loss-and-no-pos-sub) | 3.043 | 3.413 | N/A |

Adding the positional embedding subtraction strictly improved performance.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/pos_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/pos_val_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **with pos embed sub** [(config)](#with-pos-embed-sub) | **2.979** | **3.381** | N/A |
| **no encoder loss and no pos sub** [(config)](#no-encoder-loss-and-no-pos-sub) | 3.043 | 3.413 | N/A |


Combining both MSE encoder loss and positional embedding subtraction improved validation loss.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/both_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/both_val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/both_encoder_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **with pos embed sub** [(config)](#with-pos-embed-sub) | **2.979** | 3.381 | N/A |
| **with MSE encoder loss** [(config)](#with-mse-encoder-loss) | 2.998 | 3.385 | 4.138e-9 |
| **with MSE encoder loss and pos sub** [(config)](#with-mse-encoder-loss-and-pos-sub) | 2.982 | **3.378** | 4.673e-9 |

Compared to a canonical decoder-only transformer (baseline), the new model outperformed it in validation loss but underperformed in train loss. Both completed in a similar amount of time with similar memory demands, but the baseline had more parameters.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/f_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/f_val_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Size (params) |
|---|----------|----------|----------|
| **with MSE encoder loss and pos sub** [(config)](#with-mse-encoder-loss-and-pos-sub) | 2.982 | **3.378** | 15,763,500 |
| **baseline** [(config)](#baseline) | **2.937** | 3.424 | 16,036,800 |

Two more baselines are compared: "smaller baseline" and "0.2 dropout baseline". "smaller baseline" is a baseline smaller than "with MSE encoder loss and pos sub". By outperforming it, the new model's better validation loss can't be attributed to its smaller size. "0.2 dropout baseline" is a baseline with 0.2 dropout. By outperforming it, the new model also demonstrates its superiority over dropout.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/ff_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/ff_val_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Size (params) |
|---|----------|----------|----------|
| **with MSE encoder loss and pos sub** [(config)](#with-mse-encoder-loss-and-pos-sub) | 2.982 | **3.378** | 15,763,500 |
| **baseline** [(config)](#baseline) | **2.937** | 3.424 | 16,036,800 |
| **smaller baseline** [(config)](#smaller-baseline) | 2.958 | 3.416 | 15,441,192 |
| **0.2 dropout baseline** [(config)](#02-dropout-baseline) | 3.174 | 3.406 | 16,036,800 |

## Next steps

These are some further things to look forward to:
- experiment with unequal encoder and decoder layers, ideally allowing the model to learn it 
- instead of MSE and cosine dissimilarity, some other disaffinity scores should be experimented with
- the cumulative embedding average $E_{avg\\\_sum}$ assumes equal contribution from every preceding token, so a different aggregation might be better (maybe convolution?)
- try bigger models, at least GPT-2 size
- run training for longer to observe long-term behavior
- try different datasets
- dropout is known to improve validation loss, but it was not used here for simplicity. The new architecture should also be tested with dropout

## Conclusions

Even the bare-bones [no encoder loss and no pos sub](#no-encoder-loss-and-no-pos-sub) outperformed the baseline in validation loss with fewer parameters. This probably means that cross-attention on encoder output is enough for better performance (or at least prevents overfitting). When coupled with encoder loss and positional embedding subtraction, performance improved even more.

More informative, it would be very interesting to inspect the effect of encoder loss and positional embedding subtraction on token and positional embeddings. Perhaps interesting relationships can be observed between token and positional embedding. Furthermore, positional embedding subtraction should work even for decoder-only transformers, and experiments should validate this.

Alas, the principal limitation is my personal compute budget, so this project cannot avail itself of further analysis and experimentation.

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
                  'n_embed': 150,
                  'n_layer': 13,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 3,
                  'add_ln_before_decoder_ff': False,
                  'add_pos_embed_to_decoder': False,
                  'encoder_embed_loss_coeff': 1,
                  'sub_pos_embed_to_decoder': 1,
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
                  'n_embed': 150,
                  'n_layer': 13,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 2,
                  'add_ln_before_decoder_ff': False,
                  'add_pos_embed_to_decoder': False,
                  'encoder_embed_loss_coeff': 1,
                  'sub_pos_embed_to_decoder': 1,
                  'encoder_embed_detach_type': 3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```
#### "no encoder loss and no pos sub"
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
                  'n_embed': 150,
                  'n_layer': 13,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': None,
                  'use_ln_on_encoder_out': None,
                  'encoder_embed_loss_type': 1,
                  'add_ln_before_decoder_ff': False,
                  'add_pos_embed_to_decoder': False,
                  'encoder_embed_loss_coeff': None,
                  'sub_pos_embed_to_decoder': 1,
                  'encoder_embed_detach_type': None},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```

#### "with pos embed sub"
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
                  'n_embed': 150,
                  'n_layer': 13,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': None,
                  'use_ln_on_encoder_out': None,
                  'encoder_embed_loss_type': 1,
                  'add_ln_before_decoder_ff': False,
                  'add_pos_embed_to_decoder': False,
                  'encoder_embed_loss_coeff': None,
                  'sub_pos_embed_to_decoder': 2,
                  'encoder_embed_detach_type': None},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "with MSE encoder loss and pos sub"
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
                  'n_embed': 150,
                  'n_layer': 13,
                  'use_bias': False,
                  'order_type': 1,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'cross_attn_config': {'n_head': 10, 'use_bias': False},
                  'encoder_embed_ln_type': 2,
                  'use_ln_on_encoder_out': True,
                  'encoder_embed_loss_type': 2,
                  'add_ln_before_decoder_ff': False,
                  'add_pos_embed_to_decoder': False,
                  'encoder_embed_loss_coeff': 8,
                  'sub_pos_embed_to_decoder': 2,
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
                  'n_embed': 160,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "smaller baseline"
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
 'model_config': {'n_head': 12,
                  'n_embed': 156,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "0.2 dropout baseline"
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
                  'n_embed': 160,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0.2},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```