# Learned Dropout [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

Dropout is a very effective yet simple regularization technique. However, its random implementation relegates it to model training only and renders it invariant to input. Here, I present LearnedDropout, a parametrized dropout module that learns the best dropout for each unique input (i.e. variant to input). Furthermore, LearnedDropout represents a strong contender to MoE (Mixture of Experts).

## Motivations

$Dropout$ is a very popular technique that regularizes the model training to be more robust against overfitting and thus yields improved generalization. It simply works by randomly setting some values of a tensor to zero, with the ratio of zero values determined by a hyperparameter. When a value is set to zero, it becomes effectively detached from the computational graph, so all the parameters that contributed to that value will have a gradient of 0 w.r.t. that value. In doing so, $Dropout$ essentially creates a subgraph of the model because setting values to zeroes practically turns off part of the model. Given the randomness, every forward pass results in a different (transient) subgraph. Then, the final pre-trained model constitutes the ensemble of all the different subgraphs $Dropout$ created. Furthermore, observe that this outcome is not so conceptually removed from MoE's outcome. Each subgraph can be loosely though of an expert, and through these subgraphs, $Dropout$ (very weakly) partitions the model into different experts, like MoE.

Yet, unlike MoE, the random implementation means that 1) it cannot be used during inference and 2) it is invariant to input. 1) limits the benefit of $Dropout$ to pre-training only, but 2) represents the larger reason why MoE produces better performance than $Dropout$. To overcome these deficits, $Dropout$ needs to be parametrized to enable the model to learn the best dropout, for every unique input. This should make it a compelling alternative to MoE.

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with a new dropout module $LearnedDropout$. To encourage more dropout, a dropout ${L_1}$ norm penalty is added to the model loss.

<div align="center">
  <img src="assets/decoder_diagram.svg" alt="sdasd" width="40%">
</div>

### LearnedDropout

Like every dropout implementation, the new $LearnedDropout$ module computes a dropout mask $M \in \\{0, 1\\}$ that is applied to the dropout input $X = \\{x_1, x_2, \ldots, x_n\\}$. The crux lies in the mask $M$'s computation. The canonical $Dropout$ module randomly generates the dropout mask $M$ from a Bernoulli distribution $M \sim \text{Bernoulli}(r)$, where $r$ is the dropout rate hyperparameter. To enable learning, $LearnedDropout$ needs to generate the mask in a fully differentiable way. Normally, differentiability comes at the cost of loosing the $\in \\{0, 1\\}$ guarantee in favor of $M \in \[0, 1\]$. However, the implementation presented below suffers no such fate.

First, for a dropout to be highly variant to input $X$, it needs to leverage the dependencies between the input constituents $\\{x_i \mid x_i \in X\\}$ (i.e. across the T dimension). Therefore, a multi-headed attention operation is performed on the dropout input (without residual connection and other secondary operations). Stated more formally,

$$
\begin{aligned}
& W_{Q}, W_{K} ,W_{V} \coloneqq \text{attention weights} \\
& X \coloneqq \text{input of }LearnedDropout\\\\[0.5cm]
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

Lastly, a rounding is applied to bring $M$ to $\\{0,1\\}$ to satisfy $M \in \\{0, 1\\}$. The rounding is important because, otherwise, the model might use the dropout module for computational ends (e.g. scaling of $X$). $LearnedDropout$ must remain a purely selective module. Here, the rounding rounds up or down $M$ with a probability proportional to its values. For instance, given $M_\{i,j\}$, $P(M_{rounded_{(i,j)}}=1) = M_\{i,j\}$ and $P(M_{rounded_{(i,j)}}=0) = 1-M_\{i,j\}$. Stated formally,

$$
\begin{aligned}
& N \coloneqq \text{a noise tensor where } N_\{i,j\} \sim Uniform[0,1] \\
& M_{complement} =  (1 - M).detached()\\\\[0.2cm]
& M_{rounded_{(i,j)}} = 
\begin{cases} 
M_{i,j} + M_{complement_{(i,j)}}  & \text{if } N_{i,j} >= M_{complement_{(i,j)}} \\
M_{i,j} - M_{(i,j)}.detached()  & \text{if } N_{i,j} < M_{complement_{(i,j)}} \\
\end{cases} \\
\end{aligned}
$$

The detachment serves to reduce gradient magnitude. Also, the probabilistic rounding is only used during training. During evaluation or inference, the rounding becomes deterministic in the following way

$$
\begin{aligned}
& M_{rounded_{(i,j)}} = 
\begin{cases} 
1  & \text{if } M_{i,j} >= 0.5  \\
0  & \text{if } M_{i,j} < 0.5  \\
\end{cases} \\
\end{aligned}
$$

At the end, the output of the module is the element-wise product between $X$ and $M_{rounded}$

$$ out_{dropout} =  X \odot M_{rounded} $$

### Dropout L1 norm penalty

Intuitively, more dropout (i.e. more 0s in $M$) is desirable. This intuition stems from the Occam's razor or Minimum Description Length principle. This is also analogous to desiring fewer experts per token in MoE. Yet, the model does not intrinsically favor more dropout. In fact, the opposite would happen because the next token prediction loss function incentivizes the model to use as much compute as possible, hence less dropout. To counter this, a dropout ${L_1}$ norm penalty is added to the final model loss, calculated in the following way

$$ L_{1}\\\_norm\\\_penalty = \left|\frac{M^2}{2}\right|_1$$

Note that the unrounded $M$ is used because it is more determinative of more dropout. The squaring of $M$ serves to create an non-linear penalty: as $M$ approaches 0, the penalty should decay. The scaling of $M$ serves to scale down the gradients.

## Results

> All training runs below were done on a wikipedia dataset for 26k steps on a single A100 GPU, unless otherwise stated.
> 
> Implementation of decoder-only transformer model (baseline) can be found in the `baseline_transformer` directory in this repo

First, we evaluate the difference between including the L1 norm penalty and excluding it.

|   | Train loss | Val loss | $M_{rounded}$'s % of 1s |
|---|----------|----------|----------|
| **with penalty** [(config)](#) | **2.993** | 3.387 | 8.564e-9 |
| **without penalty** [(config)](#) | 2.998 | **3.385** | 4.138e-9 |