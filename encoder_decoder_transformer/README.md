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

The parallelization sim

### Penalty terms

Two penalty terms are added to the loss function: dropout entropy $\mathrm{H}$ and dropout L1 norm ${L_1}$. The final loss function is

$$ loss = cross\\_entropy(\theta, X, Y) + \mathrm{H}(\mathbf{m}) + L_1(\mathbf{m})$$

Reasons for both are described below.

#### Dropout entropy

Dropout mask values in-between 0 and 1 just scale down the input, which is undesirable for many reasons, the chief one being it potentially causing vanishing gradients. Therefore, the model should be penalized for dropout mask values far from 0 and 1. The dropout mask entropy $\mathrm{H}$ does exactly that. The dropout mask entropy $\mathrm{H}$ applies Shannon's information entropy to the dropout mask
$$\mathrm{H}(\mathbf{m}) =  \sum_{i}-\mathbf{m}_i\log_2\mathbf{m}_i $$

This ensures that the dropout mask values are pushed as close as possible to $\\{0,1\\}$ since the function's global minima occur there (remember that $\mathbf{m}_i \in [0,1]$).

#### Dropout L1 norm

Intuitively, one should desire for fewer experts (i.e. more dropout) active per token. This intuition stems from the Occam's razor principle. Yet, solely adding learned dropout does not incentivize the model to favor more dropout. In fact, the opposite would happen because the loss function will incentivize the model to use less dropout (more experts, and thus more compute). The Dropout L1 norm serves to counter the otherwise degenerate tendency. The L1 norm ${L_1}$ is just the canonical function

$$ L_1(\mathbf{m}) = |\mathbf{m}|_1$$

which encourages more dropout (more zeroes, fewer ones).

#### Additional LearnedDropout hyperparameters

TODO

## Analysis/experiments

TODO

## Conclusions

TODO