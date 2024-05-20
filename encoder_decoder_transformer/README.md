# Parallel Encoder-Decoder Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Current SOTA LLMs are all decoder-only models. Here, a new encoder-decoder transformer variant is presented where encoder and decoder run in parallel, unlike the typical implementation where they run serially (encoder first and decoder after).

## Motivations

The largest motivation originates from the paradox posed by the atttention mechanism attending latent representation of prior tokens when they are solely optimized for next token prediction. More concretely, if the latent representation $\mathbf{h}^{l}_{t}$ of token $\mathbf{x}_{t}$ is trying to predict the next token $\mathbf{x}_{t+1}$, then $\mathbf{h}^{l}_{t}$ shouldn't be **entirely** useful to the latent representation $\mathbf{h}^{l}_{t+1}$ of the next token (and any $\mathbf{h}^{l}_{z}$ where $z > t$), which is trying to predict $\mathbf{x}_{t+2}$. Yet, the attention mechanism makes $\mathbf{h}^{l}_{t}$ attend to all $\mathbf{h}^{l}_{z}$ where $z < t$. Now, we know empirically that the earlier layers of a decoder-only transformer are less focused on next-token prediction and more on just general understanding, so latent representation of earlier tokens at these layers are indeed more useful to later tokens. Though there is a singular objective function (next token prediction), the attention mechanism implitictly introduces another one: general (contextual) understanding. But there is reason to conjecture that they become less useful at later layers as latent representation becomes increasingly attuned to next token prediction. Plausibly, attending to prior tokens at these layers could hurt performance. Therefore, separating the dual latent representation could improve performance.

In a encoder-decoder model, this dual nature is separated. The encoder handles general understanding and decoder handles prediction. But the canonical implementation has them running serially. Instead, the model presented here implements them in parallel. The parallel implementation also permits an additional loss on the encoder output.

## Architecture

At the high level, the architecture re-implements the canonical encoder-decoder model but in a parallel way. Furthemore, novel components were added to exploit the dual & parallel encoder-decoder representation.

### Encoder-Decoder

The canonical encoder-decoder model looks roughly like this

<figure>
    <img src="assets/diagram.png"
         alt="diagram">
    <figcaption><em>From the Attention is All You Need paper. The modern encoder-decoder remains largely the same as the one above, with the major difference being the relocation of Add & Norm component to before attention and feed forward blocks.</em></figcaption>
</figure>


The parallelized implementation simply has the following as a single layer that's stacked $N$ times.

<figure>
    <img src="assets/new_diagram.png"
         alt="diagram">
    <figcaption><em>Just like the one before, Add & Norm should be moved to before each block.</em></figcaption>
</figure>


This new combined layer has two inputs, one for the encoder and decoder, and two outputs, one for the encoder and decoder. The decoder and encoder latent representations interact only at the second attention block on the decoder side. Stated in pseudocode, it becomes

```
def encoder_decoder_layer_forward(encoder_x, decoder_x):
    # encoder block
    encoder_x = encoder_x + encoder_multi_attn_head(
        encoder_layer_norm_1(encoder_x)
    )
    encoder_x = encoder_x + encoder_feed_forward(encoder_layer_norm_2(encoder_x))
    
    # decoder block
    decoder_x = decoder_x + decoder_multi_attn_head(
        decoder_layer_norm_1(decoder_x)
    )
    decoder_x = decoder_x + cross_multi_attn_head(
        encoder_cross_layer_norm(encoder_x), decoder_cross_layer_norm(decoder_x)
    )
    decoder_x = decoder_x + decoder_feed_forward(decoder_layer_norm_2(decoder_x))
    return encoder_x, decoder_x
```

The `decoder_x` input of the first layer is obtained from a feed forward layer on the model input embedding.

### Encoder loss

In the canonical decoder-encoder model, the loss function is evaluated over the decoder's output (itself being a function of the encoder's output). In this implementation, we end up with two outputs, one from the encoder and one from decoder. The loss over the decoder output constitutes the canonical loss function, but the presence of a encoder output permits another loss function. In this case, it was used to update the token and positional embedding. The idea here is similar to weight tying of the output layer with token embedding. Weigh tying both 1) increases update frequency & magnitude of embedding weights and 2) better compresses the entire forward pass into embedding weights, thus permitting hidden layers to compute more complex representation. The same 1) and 2) can be achieved with the encoder loss described as follows. Given the original embedding ${E}$ (token + positional) that is the encoder input to the first hidden layer, you calculate the cumulative average along the token dimension (i.e. T dimension). Finally, you calculate a disaffinity score between the cumulative average and the encoder output, and use that as the encoder loss. Stated more formally, you have

$$
out_{enc} \coloneqq \text{encoder output of the last layer} \\
E \coloneqq \text{model input embedding, comprised of token and positional embedding} \\
E_{avg\_sum} = CumAvg(E)\quad \text{where}\quad E_{avg\_sum_{i,j}} = CumAvg(E_{1:i,j}) = \frac{1}{i} \sum_{z}^{i}E_{z,j} \\
encoder\_loss\ = disaffinity\_score(out_{enc}, E_{avg\_sum})
$$

Two disaffinity scores are experimented. One is mean squared error, and the other is cosine dissimilarity based. Cosine dissimilarity is cosine similarity normalized such that zero represents most similarity and 1 most dissimilarity. So the encoder loss with euclidian distance is just

$$encoder\_loss\ = MSE(out_{enc}, E_{avg\_sum})$$

and the encoder loss with cosine dissimilarity is

$$encoder\_loss\ = 1- \frac{cosine\_similarity(out_{enc}, E_{avg\_sum}) + 1}{2}$$

#### Positional loss in decoder

TODO

## Analysis/experiments

All training runs below were trained on a wikipedia dataset for 9k steps on a single A100 GPU.

The MSE encoder loss did better than cosine similarity. Both types of encoder loss did better than without it.

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

Another thing that was tried was adding the positional embedding of the next token to the decoder embedding. This proved to be helpful to train loss but detrimental to val loss.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/pos_val_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/pos_train_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/pos_encoder_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Encoder loss |
|---|----------|----------|----------|
| **add_pos_embed_to_decoder=True** | **2.981** | 3.439 | 3.656e-9 |
| **add_pos_embed_to_decoder=False** | 2.99 | **3.435** | 4.471e-9 |

But compared to a baseline, it doesn not outperform it in train loss, but it did outperform it in validation loss. Both complete in the same amount of time.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/baseline_val_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/baseline_train_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Size (params) |
|---|----------|----------|----------|
| **encoder-decoder transformer** | 2.981 | **3.439** | 15,698,400 |
| **baseline** | **2.934** | 3.449 | 15,850,380 |

## Next steps

There are some improvements to be made:
- instead of the encoder and decoder having equal depth, it would be better for the model to learn what depth is best for either and at which layer should the decoder start (the encoder should always start from the beginning)
- instead of MSE and cosine similarity, some other similarity score should be experimented
- the cumulative embedding average $E_{avg\_sum}$ assumes equal contribution from every preceding token, so a different aggregation might be better (maybe convolution?)
- the first $x_{decoder}$ is initialized with a feed forward layer on $E$, which is unideal. Ideally, the decoder would have its own embeddings, but that would add too many paramters. A different way to initialize $x_{decoder}$ should be explored

## Conclusions

TODO