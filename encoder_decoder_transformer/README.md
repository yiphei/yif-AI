# Parallel Encoder-Decoder Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Current SOTA LLMs are all decoder-only models. Here, a new encoder-decoder transformer variant is presented where encoder and decoder run in parallel, unlike the canonical implementation where they run serially (encoder first and decoder after).

## Motivations

The largest motivation originates from the paradox posed by the atttention mechanism attending latent representation of prior tokens when they are solely optimized for next token prediction. More concretely, if the latent representation $\mathbf{h}^{l}\_{t}$ of token $\mathbf{x}\_{t}$ is trying to predict the next token $\mathbf{x}\_{t+1}$, then $\mathbf{h}^{l}\_{t}$ shouldn't be **entirely** useful to the latent representation $\mathbf{h}^{l}\_{t+1}$ of the next token (and any $\mathbf{h}^{l}\_{z}$ where $z > t$), which is trying to predict $\mathbf{x}\_{t+2}$. Yet, the attention mechanism makes $\mathbf{h}^{l}\_{t}$ attend to the entire $\mathbf{h}^{l}\_{z}$ (though with different weightings) where $z < t$. Now, we know empirically that the earlier layers of a decoder-only transformer are less focused on next-token prediction and more on just general understanding, so latent representation of earlier tokens at these layers are indeed more useful to later tokens. Though there is a singular objective function (next token prediction), the attention mechanism implitictly introduces another one: general (contextual) understanding. However, there is reason to conjecture that they become less useful at later layers as latent representation becomes increasingly attuned to next token prediction. Plausibly, attending to prior tokens at these layers could hurt performance. Therefore, it's worth exploring if separating this dual latent representation could improve performance.

In a encoder-decoder model, this dual nature is intrinsically separated. The encoder handles general understanding and decoder handles prediction. These run serially in the canonical implementation. Instead, the model presented here implements them in parallel. The parallel implementation also permits an additional loss on the encoder output.

## Architecture

At the high level, the architecture re-implements the canonical encoder-decoder model but in a parallel way. Furthemore, novel components were added to exploit the dual & parallel encoder-decoder representation.

### Encoder-Decoder layer

The canonical encoder-decoder transformer first runs the encoder and then the decoder, serially. It looks roughly like this

<div align="center">
  <img src="assets/diagram.png" alt="diagram" width="500">
  <br>
  <em>From the <strong>Attention is All You Need</strong> paper. The modern encoder-decoder remains largely the same as the one above, with the major difference being the relocation of <strong>Norm</strong> to before attention and feed forward blocks.</em>
</div>
<br>

The parallelized implementation simply has the following as a single layer that's stacked $N$ times.

<div align="center">
    <img src="assets/new_diagram.png"
         alt="diagram" width="500">
    <br>
    <em>Just like the one before, <strong>Norm</strong> should be moved to before each block.</em>
</div>
<br>

This new combined layer has two inputs, one for the encoder and decoder, and two outputs, one for the encoder and decoder. The decoder and encoder latent representations interact only at the second attention block on the decoder side. Stated in pseudocode, it becomes

```
def encoder_decoder_layer_forward(encoder_x, decoder_x):
    # encoder part
    encoder_x = encoder_x + encoder_multi_attn_head(
        encoder_layer_norm_1(encoder_x)
    )
    encoder_x = encoder_x + encoder_feed_forward(encoder_layer_norm_2(encoder_x))
    
    # decoder part
    decoder_x = decoder_x + decoder_multi_attn_head(
        decoder_layer_norm_1(decoder_x)
    )
    decoder_x = decoder_x + cross_multi_attn_head(
        encoder_cross_layer_norm(encoder_x), decoder_cross_layer_norm(decoder_x)
    )
    decoder_x = decoder_x + decoder_feed_forward(decoder_layer_norm_2(decoder_x))
    return encoder_x, decoder_x
```

The `encoder_x` input to the first layer is just the input embedding $E$ (token + positional), no different than decoder-only transformer. The `decoder_x` input to the first layer is obtained from a feed forward on $E$.

### Encoder loss

In the canonical decoder-encoder model, the loss function is evaluated over the decoder's output (itself being a function of the encoder's output). In this implementation, we end up with two outputs, one from the encoder and one from decoder. The loss over the decoder output constitutes the canonical loss function extant in decoder-only models, but the presence of a encoder output permits another loss function. In this implementation, it is used to update the token and positional embedding. The idea here is similar to weight tying the output layer with the token embedding. Weigh tying increases update frequency & magnitude of embedding weights, which then better compresses the entire forward pass into embedding weights. Ultimately, this permits hidden layers to compute more complex representation. The same effect can be achieved (in addition to output layer weight tying) with the encoder loss described as follows. Given the original input embedding ${E}$, which is also the encoder input to the first hidden layer, you calculate the cumulative average along the token dimension (i.e. T dimension). Then, the encoder loss is calculated as a disaffinity score between the cumulative average and the encoder output. Stated more formally, you have

$$
\begin{aligned}
& out_{enc} \coloneqq \text{encoder output (from the last layer)} \\
& E \coloneqq \text{model input embedding, comprised of token and positional embedding} \\
& E_{avg\\\_sum} \coloneqq \text{cumulative average of }E\text{ along T dimension, where } E_{avg\\\_sum_{(i,j)}} = \frac{1}{i} \sum_{z}^{i}E_{z,j} \\
& encoder\\\_loss = disaffinity\\\_score(out_{enc}, E_{avg\\\_sum})
\end{aligned}
$$

Two disaffinity scores are experimented. One is mean squared error, and the other is cosine dissimilarity. Cosine dissimilarity is cosine similarity normalized such that zero represents most similarity and 1 most dissimilarity. So the encoder loss with MSE is just

$$encoder\\\_loss = MSE(out_{enc}, E_{avg\\\_sum})$$

and the encoder loss with cosine dissimilarity is

$$encoder\\\_loss = 1- \frac{cosine\\\_similarity(out_{enc}, E_{avg\\\_sum}) + 1}{2}$$

#### Positional embedding in decoder

A small addition that proved useful is adding the positional embedding of the next tokens (the ones to be predicted) to `decoder_x` before the first hidden layer. In pseudocode, it becomes
```
# this is the forward of the model
def forward(self, x, targets):
    ...

    decoder_x += self.positional_embedding(
        torch.arange(
            start=1, end=x.shape[1] + 1, dtype=torch.long
        )
    )
    
    # beginning of hidden layers
    ...
```

Naturally, the positional embeddings would have to be initialized to have `context_size + 1` indices.
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

Compared to a canonical decoder-only transformer (baseline), it outperformed it in validation loss but underformed in train loss. Both completed in the similar amount of time with the similar memory demands. The baseline did have more parameters because it was hard to exactly match the new model's.

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
- instead of MSE and cosine dissimilarity, some other disaffinity score should be experimented with
- the cumulative embedding average $E_{avg\\\_sum}$ assumes equal contribution from every preceding token, so a different aggregation might be better (maybe convolution?)
- the first `decoder_x` is initialized with a feed forward layer on $E$. Ideally, the decoder would have its own embeddings, but that would add too many parameters. A different way to initialize `decoder_x` should be explored

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
                  'add_pos_embed': True,
                  'sub_pos_embed': 1,
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
                  'add_pos_embed': True,
                  'sub_pos_embed': 1,
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
                  'add_pos_embed': True,
                  'sub_pos_embed': 1,
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
                  'add_pos_embed': True,
                  'sub_pos_embed': 1,
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
                  'add_pos_embed': False,
                  'sub_pos_embed': 1,
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
                  'add_pos_embed': True,
                  'sub_pos_embed': 1,
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