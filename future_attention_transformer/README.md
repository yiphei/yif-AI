# Future Attention Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github.

Decoder-only transformer models apply a causal mask to enable parallel training with teacher forcing. This creates an imbalance in compute per context length and throws away good learning signal. The model presented here demonstrates a way to remedy these two issues.

## Motivations

In the canonical decoder's multi attention head, an attention matrix is calculated for every head

<div align="center">
  <img src="assets/matrix_2.png" alt="sdasd" width="400">
</div>

and an upper-right triangular mask is applied on it to respect temporal causality.

<div align="center">
  <img src="assets/matrix_3.png" alt="sdasd" width="400">
</div>

This is necessary to permit parallel training because you don't want the model to cheat by exposing it the answers. However, it creates two issues. First, it creates an imbalance of compute per token: bigger contexts have more compute. This is not bad per se but the benefit of imposing it by design is not obvious. Second, the masked part of the attention matrix contains good signal on the affinity of earlier tokens to future tokens. This signal can be used for training as a loss.

We can ask the model to indirectly "predict" the masked upper right triangle of the attention matrix by having it compute the output contribution from it. That would be like predicting the "future" (beyond just the next token) since that's precisely what the mask removes. In other words, asumming that $out_{full}$ is the output of the attention operation if no mask were applied to the attention matrix (also known as $out_{encoder}$) and $out_{decoder}$ is the output if a mask were applied, then we are asking the model to compute $out_{future} = out_{full/encoder} - out_{decoder}$. Once that is computed, we return the final output of the attention operation as $out_{future} + out_{decoder}$. Moreover, as alluded to earlier, we can calculate a future attention loss on $out_{future}$ since we know the true $out_{future}^{*}$ (calculating $out_{future}^{*}$ is straight-forward but too convoluted to express briefly verbally or mathematically, so the code is your best friend here). This future attention loss is then aggregated over all heads and added to the model's final loss.

## Architecture

At the high-level, the architecture is just the canonical decoder-only transformer but with a modified multi attention head block that also predicts the contribution from the masked part of the attention matrix. Finally, a future attention loss is computed for every $out_{future}$, which is added to the final model loss.

### Future Attention Head

In the regular attention head, attention works by computing $Q$, $K$, and $V$ tensors. That continues to be the case here, with $out_{decoder}$ as the output of the cannonical attention operation with a masked attention matrix. In parallel, the model also computes $out_{future}$. To do so, we need to first determine how much of the masked part (the "future") we want to predict, which is represented by the hyperparameter $future\\\_dim$. $future\_dim$ demarcates the part of the masked upper-right triangle that starts from first diagonal where values are masked till the $future\\\_dim^{th}$ diagonal. The range of $future\\\_dim$ is $\in [1, context\\\_size-1]$ since there are $context\\\_size-1$ diagonals in the masked triangle. 

Afterwards, you need $K_{future}$ and $V_{future}$, while reusing $Q$. There are many ways to obtain these two but the easier way is to have $K_{future}$ and $V_{future}$ as model parameters (and not computed tensors like $Q$, $K$, and $V$), each of size $E\text{x}(context\\\_size-1)$. Then, you compute a "future" attention matrix $Attn_{future} = Q \cdot K_{future}[:,:,:,min(T+future\\\_dim, context\\\_size)-1]$. After, compute the full attention matrix $Attn_{full} = Attn_{decoder} + Attn_{future}$, where $Attn_{decoder}$ is just the regular decoder-only masked attention (though $Attn_{decoder} + Attn_{future}$ is not precise because there are padding operations to have the shapes match, but it's tedious to explain verbally so please see the code). Perform the softmax on $Attn_{full}$ to get $AttnSoftmax_{full}$ and separate $AttnSoftmax_{full}$ into the individual contributions $AttnSoftmax_{decoder}$ and $AttnSoftmax_{future}$. Finally, compute $out_{decoder}= AttnSoftmax_{decoder} \cdot V$, then $out_{future}= AttnSoftmax_{future} \cdot V_{future}[:,:,:min(T+future\\\_dim, context\\\_size)-1,:]$, and get the final output $out = out_{decoder} + out_{future}$. 

The future attention loss is computed between $out_{future}$ and $out_{future}^{*}$. Two types of loss are experimented. One is mean squared error, and the other is cosine dissimilarity. Cosine dissimilarity is cosine similarity normalized such that zero represents most similarity and 1 most dissimilarity. So the future attention loss with MSE is just

$$future\\\_attn\\\_loss = MSE(out_{future}, out_{future}^{*})$$

and with cosine dissimilarity is

$$future\\\_attn\\\_loss = 1- \frac{cosine\\\_similarity(out_{future}, out_{future}^{*}) + 1}{2}$$

## Analysis/experiments

All training runs below were trained on a wikipedia dataset for 9k steps on a single A100 GPU.

First, the two types of future attention loss were compared. Cosine-dissimilarity did better in validation loss while MSE did better in train loss.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/future_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Future Attention loss |
|---|----------|----------|----------|
| **with cosine-dissimilarity future attention loss** | 2.954 | **3.408** | 0.2478 |
| **with MSE future attention loss** | **2.953** | 3.411 | 0.1635 |


Different future dim were also experimented. Remember: the higher the future dim, the father in the future each token tries to predict. Naturally, one would expect that the bigger future dim is, the better. Indeed, that is true. The highest future dim of 199 performed the best (with a max context size of 200).

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/dim_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/dim_val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/dim_future_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss | Future Attention loss |
|---|----------|----------|----------|
| **with future_dim = 50** | 2.96 | 3.413 | 0.2665 |
| **with future_dim = 100** | 2.957 | 3.412 | 0.2559 |
| **with future_dim = 150** | 2.957 | 3.415 | 0.2574 |
| **with future_dim = 199** | **2.954** | **3.408** | 0.249 |

Compared to a canonical decoder-only transformer (baseline), it outperformed it in validation loss but underformed in train loss. [TO CONFIRM] Both completed in the similar amount of time with the similar memory demands. [TO CONFIRM] The baseline did have more parameters because it was hard to exactly match the new model's.

<div style="display: flex; overflow-x: auto; white-space: nowrap;">
  <img src="assets/base_train_loss.svg" alt="Image 1" style="width: 45%;"/>
  <img src="assets/base_val_loss.svg" alt="Image 2" style="width: 45%;"/>
</div>

|   | Train loss | Val loss |
|---|----------|----------|
| **future attention transformer** | 2.954 | **3.408** |
| **baseline** | **2.934** | 3.449 |


## Next steps

These are some improvements to look forward to:
- Have $K_{future}$ and $V_{future}$ be computed tokens (just like $Q$, $K$, and $V$) – instead of being free parameters – and in an parameter efficient way
- Explore other ways to compute the future attention loss

## Conclusions

TODO