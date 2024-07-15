# Learned Dropout [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

Dropout is a very effective yet simple regularization technique. However, its random implementation relegates it to model training only and renders it invariant to input. Here, I present LearnedDropout, a parametrized dropout module that learns the best dropout for each unique input (i.e. variant to input). Furthermore, LearnedDropout represents a strong contender to MoE (Mixture of Experts).

## Motivations

Dropout is a very popular technique that regularizes the model training to be more robust against overfitting and thus yields improved generalization. It simply works by randomly setting some values of a tensor to zero, with the ratio of zero values determined by a hyperparameter. When a value is set to zero, it becomes effectively detached from the computational graph, so all the parameters that contributed to that value will have a gradient of 0 w.r.t. that value. In doing so, Dropout essentially creates a subgraph of the model because setting values to zeroes practically turns off part of the model. Given the randomness, every forward pass results in a different (transient) subgraph. Then, the final pre-trained model constitutes the ensemble of all the different subgraphs Dropout created. Furthermore, observe that this outcome is not so conceptually removed from MoE's outcome. Each subgraph can be loosely though of an expert, and through these subgraphs, Dropout (very weakly) partitions the model into different experts, like MoE.

Yet, unlike MoE, the random implementation means that 1) it cannot be used during inference and 2) it is invariant to input. 1) limits the benefit of Dropout to pre-training only, but 2) is the larger reason why MoE produces better performance than Dropout. To overcome these deficits, Dropout needs to be parametrized to enable the model to learn the best dropout, for every unique input. This should make it a compelling alternative to MoE.

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with a modified dropout module. Two dropout-specific losses are introduced to encourage low dropout entropy and low dropout L1 norm, in addition to next token prediction loss.

<div align="center">
  <img src="assets/decoder_diagram.svg" alt="sdasd" width="40%">
</div>

### Learned Dropout

Like every dropout implementation, the new LearnedDropout module computes a dropout mask $M \in \\{0, 1\\}$ that is applied to the dropout input $X = \\{x_1, x_2, \ldots, x_n\\}$. The crux lies in the mask $M$'s computation. The canonical Dropout module randomly generates the dropout mask $M$ from a Bernoulli distribution $M \sim \text{Bernoulli}(r)$, where $r$ is the dropout rate hyperparameter. To enable learning, LearnedDropout needs to generate the mask in a fully differentiable way. Normally, differentiability comes at the cost of loosing the $\in \\{0, 1\\}$ guarantee in favor of $M \in \[0, 1\]$. However, the implementation here suffers no such fate.

First, for a dropout to be highly variant to input $X$, it needs to leverage the dependencies between the input constituents $\\{x_i \mid x_i \in X\\}$ (i.e. across the T dimension). Therefore, a multi-headed attention operation is performed on the dropout input (without residual connection and other secondary operations). Stated more formally,

$$
\begin{aligned}
& W_{Q}, W_{K} ,W_{V} \coloneqq \text{attention weights} \\
& X \coloneqq \text{input of LearnedDropout}\\\\[0.5cm]
& Q = X \cdot W_{Q} \\
& K = X \cdot W_{K} \\
& V = X \cdot W_{V} \\
& Attn = Q \cdot K^{T} \\
& out_{attn} = softmax(Attn) \cdot V \\
\end{aligned}
$$

Afterwards, the attention output $out_{attn}$ needs to be mapped to $\[0, 1\]$. For this, the following cosine function is employed

$$M =  0.5 \cos(out_{attn} + B) + 0.5$$

where $B \in \[0, \pi\]$ is a bias term. This function lies in the $\[0,1\]$ range, and its recurrent property eliminates the risk of dropout becoming stuck in a local minima, though at the cost of worse convergence.

Lastly, a scaling function is applied to bring $M$ to $\\{0,1\\}$. This scaling is important because, otherwise, the model might use the module for computational ends (e.g. scaling). LearnedDropout must remain a purely selective module. Here, the scaling essentially rounds up or down with the probability proportional to the $M$ values. Stated formally,

$$
\begin{aligned}
& N \coloneqq \text{a noise tensor where } N_\{i,j\} \sim Uniform[0,1] \\
& M_{complement} =  (1 - M).detached() \\
& M_{rounded_{(i,j)}} = 
\begin{cases} 
M_{i,j} + M_{complement_{(i,j)}}  & \text{if } N_{i,j} >= M_{complement_{(i,j)}} \\
M_{i,j} - M_{(i,j)}.detached()  & \text{if } N_{i,j} < M_{complement_{(i,j)}} \\
\end{cases} \\
\end{aligned}
$$


At the end, the output of the module is the element-wise product between $X$ and $M$

$$ out_{dropout} =  X \odot M $$

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
