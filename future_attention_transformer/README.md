# Future Attention Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

Decoder-only transformer models apply a causal mask to enable parallel training with teacher forcing. However, the causally masked part of the attention matrix contains good signal on the affinities between present and future tokens. This project investigates how the masked part can be leveraged to improve model training.

## Motivations

In the canonical decoder transformer, the attention layer computes an attention matrix $A$ for each head, like the figure below.

<div align="center">
  <img src="assets/unmasked.svg" alt="sdasd" width="400">
</div>

Because transformer models are trained in a parallel way, a causal mask $M$ must be applied to the attention matrix $A$ to prevent the model from peeking at future tokens and thus from cheating. Stated more formally,

$$
\begin{aligned}
& A_{causal}[i,j] = 
\begin{cases} 
A[i,j] & \text{if } M[i,j] = 1 \\
-\infty & \text{if } M[i,j] = 0
\end{cases} \\
\end{aligned}
$$

This masking is illustrated in the figure below (the masked values are depicted with red squares).

<div align="center">
  <img src="assets/causal_mask.svg" alt="sdasd" width="400">
</div>


Before proceeding, let's identify two subsets of the original $A$

$$
\begin{aligned}
& A_{unmasked}[i,j] = A[i,j] \text{ where } (i,j) \in \\{ (i,j) \mid M[i,j] = 1 \\} \\
& A_{masked}[i,j] = A[i,j] \text{ where } (i,j) \in \\{ (i,j) \mid M[i,j] = 0 \\}
\end{aligned}
$$

Now, the masked part $A_{masked}$ contains good signal on the affinities between present and future tokens. Presumably, the model could improve next token prediction by leveraging these affinities in its computations. Since the masked part can't be directly used, the model can instead predict the masked part, and these predictions can be optimized with the true masked values via a new "future loss". In the figure below, for instance, the model can predict the affinity of each token to the next two tokens (the blue squares) while the rest is masked away (the red squares).

<div align="center">
  <img src="assets/future_mask.svg" alt="sdasd" width="400">
</div>

Let's call the blue part $A_{future}$, formally defined as

$A_{future}[i,j] = A_{masked}[i,j] \text{  where  } i < j \leq min(i + future\\_dim, context\\_size)$

$future\\_dim$ is a scalar hyperparameter that defines how many masked values to predict. In the example above, $future\\_dim = 2$.

Because $A_{causal}$ is later matrix multiplied with $V$ to produce the attention output $out_{causal}$

$out_{causal} = softmax(A_{causal}) \cdot V$

then "future" $V$ values need to predicted along with $A_{future}$. This is trickier because, unlike $A_{future}$ where the target future lies in the last dimension, $V$ has the target future in the penultimate dimension. 

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with a modified multi attention head block that also predicts the some portion of the masked attention matrix. A loss is created from these predictions and is added to the final model loss.

### Future Attention Head

Because the model needs to both predict $A_{future}$ and future $V$, it is expensive to do both (because it would require two losses) and, as stated before, tricky to do $V$. Instead, it becomes much simpler to predict the contributions of $A_{future}$ and $V$ to the attention output $out$ if no mask $M$ had been applied in the first place. Then, a single loss is computed. In other words, assuming that $out_{no\\_mask}$ is the output of attention matrix without any mask

$out_{no\\_mask} = softmax(A) \cdot V$

then, the output contribution of $A_{future}$ and $V$ is

$out_{future} = out_{no\\\_mask} - out_{causal}$

Now, the attention mechanism is based on three operands:
$Q$, $K$, and $V$. $A$ is already computed by $Q$ and $K$

$A = Q \cdot K^{T}$

Since we need to indirectly predict $A_{future}$, we should reuse $Q$ but need different $K_{future}$ and $V_{future}$ to emulate $K$ and $V$. There are many ways to construct $K_{future}$ and $V_{future}$, but here they are model parameters, not computed tensors, of shape $T\times context\\_size$. All of this sums up to
    
$$
\begin{aligned}
& A = Q \cdot K^{T}  \\
& A_{causal} = A[M.indices]  \\
& A_{future} = Q \cdot K_{future}^{T}  \\
& A_{full} = A_{causal} + A_{future} \\
& Softmax\\\_A_{full} = softmax(A_{full}) \\
& Softmax\\\_A_{causal} = Softmax\\\_A_{full}[A_{causal}.indices] \\
& Softmax\\\_A_{future} = Softmax\\\_A_{full}[A_{future}.indices] \\
& out_{causal} = Softmax\\\_A_{causal} \cdot V \\
& out_{future} = Softmax\\\_A_{future} \cdot V \\
& out_{full} = out_{future} + out_{causal}
\end{aligned}
$$

The future attention loss is computed between $out_{future}$ and $out_{future}^{*}$. Two types of loss are experimented. One is mean squared error, and the other is cosine dissimilarity. Cosine dissimilarity is cosine similarity normalized such that zero represents most similarity and 1 most dissimilarity. So the future attention loss with MSE is just

$$future\\\_attn\\\_loss = MSE(out_{future}, out_{future}^{*})$$

and with cosine dissimilarity is

$$future\\\_attn\\\_loss = 1- \frac{cosine\\\_similarity(out_{future}, out_{future}^{*}) + 1}{2}$$

## Results

> All training runs below were done on a wikipedia dataset for 9k steps on a single A100 GPU, unless otherwise stated.
> 
> Implementation of decoder-only transformer model (baseline) can be found in the `baseline_transformer` directory in this repo

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
- Try bigger models, at least GPT-2 size

## Conclusions

TODO