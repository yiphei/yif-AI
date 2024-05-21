# Parallel Encoder-Decoder Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Current SOTA LLMs are all decoder-only models. Here, a new encoder-decoder transformer variant is presented where encoder and decoder run in parallel, unlike the canonical implementation where they run serially (encoder first and decoder after).

## Motivations

The largest motivation originates from the paradox posed by the atttention mechanism attending latent representation of prior tokens when they are solely optimized for next token prediction. More concretely, if the latent representation $\mathbf{h}^{l}\_{t}$ of token $\mathbf{x}\_{t}$ is trying to predict the next token $\mathbf{x}\_{t+1}$, then $\mathbf{h}^{l}\_{t}$ shouldn't be **entirely** useful to the latent representation $\mathbf{h}^{l}\_{t+1}$ of the next token (and any $\mathbf{h}^{l}\_{z}$ where $z > t$), which is trying to predict $\mathbf{x}\_{t+2}$. Yet, the attention mechanism makes $\mathbf{h}^{l}\_{t}$ attend to the entire $\mathbf{h}^{l}\_{z}$ (though with different weightings) where $z < t$. Now, we know empirically that the earlier layers of a decoder-only transformer are less focused on next-token prediction and more on just general understanding, so latent representation of earlier tokens at these layers are indeed more useful to later tokens. Though there is a singular objective function (next token prediction), the attention mechanism implitictly introduces another one: general (contextual) understanding. However, there is reason to conjecture that they become less useful at later layers as latent representation becomes increasingly attuned to next token prediction. Plausibly, attending to prior tokens at these layers could hurt performance. Therefore, it's worth exploring if separating this dual latent representation could improve performance.

In a encoder-decoder model, this dual nature is intrinsically separated. The encoder handles general understanding and decoder handles prediction. These run serially in the canonical implementation. Instead, the model presented here implements them in parallel. The parallel implementation also permits an additional loss on the encoder output.

## Architecture

At the high level, the architecture re-implements the canonical encoder-decoder model but in a parallel way. Furthemore, novel components were added to exploit the dual & parallel encoder-decoder representation.

### Encoder-Decoder layer

The canonical encoder-decoder model looks roughly like this

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

The `encoder_x` input to the first layer is just the input embedding $E$ (token + positional), no different than decoder-only models. The `decoder_x` input to the first layer is obtained from a feed forward on $E$.

### Encoder loss

In the canonical decoder-encoder model, the loss function is evaluated over the decoder's output (itself being a function of the encoder's output). In this implementation, we end up with two outputs, one from the encoder and one from decoder. The loss over the decoder output constitutes the canonical loss function extant in decoder-only models, but the presence of a encoder output permits another loss function. In this implementation, it is used to update the token and positional embedding. The idea here is similar to weight tying the output layer with the token embedding. Weigh tying both 1) increases update frequency & magnitude of embedding weights and 2) better compresses the entire forward pass into embedding weights, thus permitting hidden layers to compute more complex representation. The same 1) and 2) can be achieved with the encoder loss described as follows. Given the original input embedding ${E}$ (token + positional), which is also the encoder input to the first hidden layer, you calculate the cumulative average along the token dimension (i.e. T dimension). Then, the encoder loss is calculated as a disaffinity score between the cumulative average and the encoder output. Stated more formally, you have

$$
\begin{aligned}
& out_{enc} \coloneqq \text{encoder output (from the last layer)} \\
& E \coloneqq \text{model input embedding, comprised of token and positional embedding} \\
& E_{avg\\\_sum} \coloneqq \text{cumulative average of }E\text{ along T dimension, where } E_{avg\\\_sum_{(i,j)}} = \frac{1}{i} \sum_{z}^{i}E_{z,j} \\
& encoder\\\_loss = disaffinity\\\_score(out_{enc}, E_{avg\\\_sum})
\end{aligned}
$$

Two disaffinity scores are experimented. One is mean squared error, and the other is cosine dissimilarity. Cosine dissimilarity is cosine similarity normalized such that zero represents most similarity and 1 most dissimilarity. So the encoder loss with euclidian distance is just

$$encoder\\\_loss = MSE(out_{enc}, E_{avg\\\_sum})$$

and the encoder loss with cosine dissimilarity is

$$encoder\\\_loss = 1- \frac{cosine\\\_similarity(out_{enc}, E_{avg\\\_sum}) + 1}{2}$$

#### Positional embedding in decoder

A small addition that proved useful is adding the positional embedding of the next tokens to the output of the feed forward on the input embeddings. In pseudocode, it becomes
```
# this is the forward of the model
def forward(self, x, targets):
    token_embed = self.token_embedding(x)
    pos_embed = self.positional_embedding(
        torch.arange(x.shape[1], dtype=torch.long)
    )
    encoder_embed = token_embed + pos_embed
    encoder_x = encoder_embed

    decoder_x = self.decoder_feed_forward(encoder_x)

    # this is the novel part
    decoder_x += self.positional_embedding(
        torch.arange(
            start=1, end=x.shape[1] + 1, dtype=torch.long
        )
    )
    ...
```
## Analysis/experiments

All training runs below were done on a wikipedia dataset for 9k steps on a single A100 GPU.

The MSE encoder loss did better than cosine dissimilarity. Both types of encoder loss did better than without it.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/val_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/train_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/encoder_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **with cosine-dissimilarity encoder loss** | 2.984 | 3.445 | 8.285e-9 |
| **with MSE encoder loss** | **2.981** | **3.439** | 3.656e-9 |
| **with no encoder loss** | 2.997 | 3.449 | N/A |

Adding the positional embedding of the next tokens to the encoder helped the train loss but was to the detriment of validation loss.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/pos_val_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/pos_train_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/pos_encoder_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **add_pos_embed_to_decoder=True** | **2.981** | 3.439 | 3.656e-9 |
| **add_pos_embed_to_decoder=False** | 2.99 | **3.435** | 4.471e-9 |

Compared to a canonical decoder-only transformer model (baseline), it outperformed it in validation loss but underformed in train loss. Both completed in the same amount of time with the same memory demands.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/baseline_val_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/baseline_train_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Size (params) |
|---|----------|----------|----------|
| **encoder-decoder transformer** | 2.981 | **3.439** | 15,698,400 |
| **baseline** | **2.934** | 3.449 | 15,850,380 |

## Next steps

These are some improvements to look forward to:
- instead of the encoder and decoder having equal depth, it would be better for the model to learn what depth is best for either and at which layer should the decoder start (the encoder should always start from the beginning)
- instead of MSE and cosine dissimilarity, some other disaffinity score should be experimented with
- the cumulative embedding average $E_{avg\_sum}$ assumes equal contribution from every preceding token, so a different aggregation might be better (maybe convolution?)
- the first $x_{decoder}$ is initialized with a feed forward layer on $E$. Ideally, the decoder would have its own embeddings, but that would add too many parameters. A different way to initialize $x_{decoder}$ should be explored

## Conclusions

TODO