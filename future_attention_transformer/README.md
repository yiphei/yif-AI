# Future Attention Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Decoder-only transformer models apply a causal mask to enable parallel training with teacher forcing. But that wastes half of the attention matrix. What if you could use the upper-right triangle while respecting the temporal causality? The model presented here takes advantage of this observation.

## Motivations

In the canonical decoder's multi attention head, an attention matrix is calculated for every head

![sdasd](assets/matrix_2.png)

and an upper-right triangular mask is applied on it to respect temporal causality

![sdasd](assets/matrix_3.png)

But it seems wasteful to throw away so much information. So we can ask the model to also predict the upper right triangle, whose true value we do have.

## Architecture

At the high-level, the architecture is just the canonical decoder-only transformer but with changes to the multi attention head block also trying to compute the upper right triangle. Finally, an attention loss is computed for every attention matrix and that loss is added to the final model loss.

### Multi Attention Head

In the regular attention head, attention works by computing $Q$, $K$, and $V$ tensors. In order to predict the upper right triangle, we need $K_{future}$ and $V_{future}$. Then, you compute a future attention matrix $Attn_{future} = Q \cdot K_{future}$ and a future attention out $out_{future} = Attn_{future} \cdot  V_{future}$, then the final output is $out = out_{present} + out_{future}$ where $out_{present}$ is just the normal output.

Then, the attention loss is computed with $out_{future}$ and the true out future, which can be trivially computed.

## Analysis/experiments

TODO

## Conclusions

TODO