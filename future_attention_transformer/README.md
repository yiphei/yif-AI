# Future Attention Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Decoder-only transformer models apply a causal mask to enable parallel training with teacher forcing. But that wastes half of the attention matrix. What if you could use the full attention matrix while respecting the temporal causality? The model presented here demonstrates a way to do it.

## Motivations

In the canonical decoder's multi attention head, an attention matrix is calculated for every head

![sdasd](assets/matrix_2.png)

and an upper-right triangular mask is applied on it to respect temporal causality.

![sdasd](assets/matrix_3.png)

However, it seems wasteful to throw away so much information. Instead, we can ask the model to "predict" the masked upper right triangle of the attention matrix by having it compute the output contribution from it. In other words, asumming that $out_{full}$ is the output of the attention operation if no mask were applied to the attention matrix (also known as $out_{encoder}$) and $out_{decoder}$ is the output if a mask were applied, then we are asking the model to compute $out_{mask} = out_{full/encoder} - out_{decoder}$. Once that is computed, we return the final output of the attention operation as $out_{mask} + out_{decoder}$. Moreover, we can calculate an attention loss on $out_{mask}$ since we know the true $out_{mask}$. This attention loss is then aggregated over all heads and added to the model's final loss.

## Architecture

At the high-level, the architecture is just the canonical decoder-only transformer but with a modified multi attention head block that also predicts the contribution from the masked part of the attention matrix. Finally, an attention loss is computed for every $out_{mask}$, which is added to the final model loss.

### Future Attention Head

In the regular attention head, attention works by computing $Q$, $K$, and $V$ tensors. That continues to be the case here. $out_{decoder}$ is just the output of the cannonical attention operation with a masked attention matrix. In parallel, the model also computes $out_{mask}$. To do so, we need $K_{mask}$ and $V_{mask}$. There are many ways to obtain these two but the easier way is to have $K_{mask}$ and $V_{mask}$ as model parameters, not computed tensors like $Q$, $K$, and $V$. Then, you compute a "future" attention matrix $Attn_{future} = Q \cdot K_{future}$. Afterwards, compute the full attention matrix $Attn_{full} = Attn_{decoder} + Attn_{future}$, where $Attn_{decoder}$ is just the regular decoder-only masked attention (note that $Attn_{decoder} + Attn_{future}$ is not precise because there are padding operations to have the shapes match. Please view the code as it is hard to explain verbally). Perform the softmax on $Attn_{full}$ to get $AttnSoftmax_{full}$ and separate $AttnSoftmax_{full}$ into the individual contributions $AttnSoftmax_{decoder}$ and $AttnSoftmax_{mask}$, where $AttnSoftmax_{decoder}$ is just the regular attention softmax that you would get from decoder attention. Finally, compute $out_{decoder}= AttnSoftmax_{decoder} \cdot V$ and $out_{mask}= AttnSoftmax_{mask} \cdot V_{future}$ and get the final output $out = out_{decoder} + out_{mask}$. 

The attention loss is computed with respect to $out_{mask}$ and the true $out_{mask}$, which can be trivially computed.

## Analysis/experiments

TODO

## Conclusions

TODO