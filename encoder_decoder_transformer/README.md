# Parallel Encoder-Decoder model [WIP readme]
> NB: LaTeX is optimized for Github's Markdown. 

Current SOTA LLMs are all decoder-only models. Here, a new transformer variant is presented where encoder and decoder run in parallel, unlike the typical implementation where they are run serially (encoder first and decoder after). 

## Motivations

The first motivation originates from the realization that the objective function of every token's latent representation at each layer is to predict the next token. This introduces a theoretical paradox when training (and inference later) is done in parallel. If the latent reprensation ${h_l}$ of token at time ${t}$ is trying to predict the next token at time ${t+1}$, then that latent representation shouldn't be very useful to the latent representation of token at time ${t+1}$ trying to predict token at time ${t+2}$. Yet, the attention mechanism makes the latent representation of ${t+1}$ attend to every latent representation of tokens at time $<{t+1}$. Now, we know that the earlier layers of a decoder-only transformer of are less focused on next-token prediction and more on just general understanding, so latent representation of earlier tokens at these layers should prove useful to later tokens. But it's reasonable to believe that they become less useful at later layers. Therefore, it would be nice to have some disentanglement: a latent representation more for general understanding and one more for next token prediction. This disentaglement exists in encoder-decoder models, with the encoder handling the understanding and decoder handling the next token prediction. But this is canonically implemented serially, leading to very deep models. Instead, the model presented here implements them in parallel.

## Architecture

At the high level, the architecture re-implements the canonical encoder-decoder model in a parallel way. But novel components were added to exploit the dual encoder-decoder representation.

### Encoder-Decoder

The canonical encoder-decoder model looks roughly like this

<figure>
    <img src="assets/diagram.png"
         alt="diagram">
    <figcaption><em>From the Attention is All You Need paper. The modern encoder-decoder is largely the same as the one above, with the major difference being the relocation of Add & Norm component to before attention and feed forward.</em></figcaption>
</figure>


The parallelization simply has the following as a single layer that's stacked N times.

<figure>
    <img src="assets/new_diagram.png"
         alt="diagram">
    <figcaption><em>Just like the one before, Add & Norm should be moved to before each block.</em></figcaption>
</figure>


This layer haves two inputs, one for the encoder and decoder, and two outputs, one for the encoder and decoder. The decoder and encoder latent representation interact only at the second attention block on the decoder side. Or, stated in pseudo-code, it is

```
def encoder_decoder_layer(encoder_x, decoder_x):
    encoder_x = encoder_x + encoder_multi_attn_head(
        encoder_layer_norm_1(encoder_x)
    )
    encoder_x = encoder_x + encoder_feed_forward(encoder_layer_norm_2(encoder_x))

    decoder_x = decoder_x + decoder_multi_attn_head(
        decoder_layer_norm_1(decoder_x)
    )
    decoder_x = decoder_x + cross_multi_attn_head(
        encoder_cross_layer_norm(encoder_x), decoder_cross_layer_norm(decoder_x)
    )
    decoder_x = decoder_x + decoder_feed_forward(decoder_layer_norm_2(decoder_x))
    return encoder_x, decoder_x
```

The first decoder_x is obtained from a feed forward layer on the model input embedding.

### Encoder loss

In the canonical decoder-encoder model, the loss function is evaluated over the decoder's output (itself being a function of the encoder's output). In this implementation, we end up with two outputs, one from the encoder and one from decoder. The loss over the decoder output constitutes the canonical loss function, but the presence of a encoder output permits something. In this case, it was used to update the token and positional embedding. The idea here is similar to weight tying of the output layer with token embedding. Weigh tying both 1) increases update frequencies and magnitude and 2) kinda compresses an entire forward pass into embedding weights, thus permitting hidden layers to do more complex representation. The same 1) and 2) can be achieved with the encoder loss described as following. Given the original embedding ${E}$ (token + positional) that is the input to the first hidden layer, you calculate the cumulative average along the token dimension (i.e. T dimension). Finally, you calculate an dis-affinity score between the cumulative average and the encoder output, and use that as the encoder loss. Stated more formally, you have

$$
out_{enc} \coloneqq \text{encoder output of the last layer} \\
E \coloneqq \text{model input embedding, comprising of token and positional embedding} \\
E_{avg\_sum} = CumAvg(E)\quad \text{where}\quad E_{i,j} = CumAvg(E_{1:i,j}) = \frac{1}{i} \sum_{z}^{i}E_{z,j} \\
encoder\_loss\ = disaffinity\_score(out_{enc}, E_{avg\_sum})
$$

Two disaffinity scores are experimented. One is euclidian distance, and the other is cosine similarity. Cosine similarity needs to be normalize for 0 to represent the most similarity. So the encoder loss with euclidian distance is just

$$encoder\_loss\ = \|out_{enc} - E_{avg\_sum} \|_2$$

and the encoder loss with cosine similarity is

$$encoder\_loss\ = 1- \frac{cosine\_similarity(out_{enc}, E_{avg\_sum}) + 1}{2}$$

#### Positional loss in decoder

TODO

## Analysis/experiments

TODO

## Conclusions

TODO