# Future Attention Transformer [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

Decoder-only transformer models apply a causal mask to enable parallel training with teacher forcing. However, the causally masked part of the attention matrix contains good signal on the affinities between present and future tokens. This project investigates how the masked part can be leveraged to improve model training.

## Motivations

In the canonical decoder-only transformer, the attention layer computes an attention matrix $A$ for each head, like the figure below.

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

The result of masking is illustrated in the figure below (the masked values are depicted with red squares).

<div align="center">
  <img src="assets/causal_mask.svg" alt="sdasd" width="400">
</div>

Afterwards, the following operation occurs

$$
\begin{aligned}
& out_{causal} = softmax(A_{causal}) \cdot V \\
\end{aligned}
$$

This concludes the attention mechanism. Although other subsequent operations on $out_{causal}$ usually follow (e.g. dropout, residual projection, etc.), those are not of concern here.

Before proceeding, let's identify two subsets of the original $A$

$$
\begin{aligned}
& A_{unmasked}[i,j] = A[i,j] \text{ where } (i,j) \in \\{ (i,j) \mid M[i,j] = 1 \\} \\
& A_{masked}[i,j] = A[i,j] \text{ where } (i,j) \in \\{ (i,j) \mid M[i,j] = 0 \\}
\end{aligned}
$$

Now, the masked part $A_{masked}$ contains good signal on the affinities between present and future tokens. If no mask $M$ were applied, subsequent operations would transform these affinities into $out_{masked}$, like so              

$$
\begin{aligned}
& Softmax\\\_A = softmax(A) \\
& Softmax\\\_A_{masked} = Softmax\\\_A[A_{masked}.indices] \\
& out_{masked} = Softmax\\\_A_{masked} \cdot V \\
\end{aligned}
$$

Presumably, the model performance would improve if it could use $out_{masked}$ (e.g. add it to $out_{causal}$). Since $out_{masked}$ can't be directly used because of masking, the model can instead predict $out_{masked}$ and then use it. Subsequently, these predictions can be optimized against the true $out_{masked}^{\*}$ with a new **future loss**. Furthermore, instead of predicting the entire $out_{masked}$, the model can also predict part of it, which is equivalent to limiting how much of $A_{masked}$ is considered. Let's call $out_{future}$ and $A_{future}$ the subsets of $out_{masked}$ and $A_{masked}$, respectively. Then, let $future\\\_dim$ be the scalar hyperparameter that defines how many masked values in $out_{masked}$ to consider, per token. Stated formally, 

$$
\begin{aligned}
& A_{future}[i,j] = A_{masked}[i,j] \text{  where  } i < j \leq min(i + future\\\_dim, context\\\_size) \\
& A_{omni} = A_{future} \cup A_{unmasked} \\\\[0.4cm]
& Softmax\\\_A_{omni} = softmax(A_{omni}) \\
& Softmax\\\_A_{future} = Softmax\\\_A_{omni}[A_{future}.indices] \\
& out_{future} = Softmax\\\_A_{future} \cdot V \\
\end{aligned}
$$

In the figure below, for instance, the model considers the affinity of each token to the next two tokens (the blue squares) while the rest is masked away (the red squares). So $future\\_dim = 2$.

<div align="center">
  <img src="assets/future_mask.svg" alt="sdasd" width="400">
</div>

**Note:** $future\\_dim$ only represents the max value. In fact, in the figure above, $q_4$ can only predict $k_5$.

Here's a visualization guide for all the different $A$'s define thus far.

[ADD VISUALIZATION HERE]

(The reason for the explicit definition of all these different attention matrices is the indexing-heavy nature of the implementation presented below)

Because the softmax of the attention matrix is later matrix multiplied with $V$ to produce the attention output $out$

$out = softmax(A) \cdot V$

then $V$ also need to be adjusted to match $softmax(A)$'s shape.

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with a modified multi attention head block that also predicts some portion of the masked attention matrix. A loss is created from these predictions and is added to the final model loss.

### Future Attention Head

The attention mechanism is based on three operands:
$Q$, $K$, and $V$. $A$ is already computed by $Q$ and $K$

$A = Q \cdot K^{T}$

Since we need to indirectly predict $A_{future}$, we should reuse $Q$ but need different $K_{future}$ and $V_{future}$ to emulate $K$ and $V$. There are many ways to construct $K_{future}$ and $V_{future}$, but here they are model parameters, not computed tensors, of shape $T\times context\\_size$. All of this sums up to

|||
|----------|----------|
| $$A = Q \cdot K^{T}$$ | $A_{future} = Q \cdot K_{future}^{T}$ |
| $$A_{unmasked} = A[A_{unmasked}.indices]$$ | |
| $$A_{omni} = A_{unmasked} \cup A_{future}$$ | |
| $$Softmax\\\_A_{omni} = softmax(A_{omni})$$ | |
| $$Softmax\\\_A_{unmasked} = Softmax\\\_A_{omni}[A_{unmasked}.indices]$$ | $$Softmax\\\_A_{future} = Softmax\\\_A_{omni}[A_{future}.indices]$$ |
| $$out_{unmasked} = Softmax\\\_A_{unmasked} \cdot V$$ | $$out_{future} = Softmax\\\_A_{future} \cdot V_{future}$$ |
| $$out_{omni} = out_{future} + out_{unmasked}$$ | |

Note that $A_{unmasked}$ and $A_{future}$ have different shapes, so merging the two requires padding operations, which is denoted by $\cup$.

To calculate the true $out_{future}^{*}$, you have

$$
\begin{aligned}
& A_{omni}^{\*} = A[A_{unmasked}.indices \cup A_{future}.indices]  \\
& Softmax\\\_A_{omni}^{\*} = softmax(A_{omni}^{\*})  \\
& Softmax\\\_A_{future}^{\*} = Softmax\\\_A_{omni}^{\*}[A_{future}.indices]  \\
& out_{future}^{\*} = Softmax\\\_A_{future}^{\*} \cdot V
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