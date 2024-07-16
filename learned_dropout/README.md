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

where $B \in \[0, \pi\]$ is a bias term. This function lies in the $\[0,1\]$ range, and its recurrent property eliminates the risk of dropout becoming stuck in a local minimum, though at the cost of worse convergence.

Lastly, a rounding is applied to bring $M$ to $\\{0,1\\}$ to satisfy $M \in \\{0, 1\\}$. The rounding is important because, otherwise, the model might use the dropout module for computational ends (e.g. scaling of $X$). $LearnedDropout$ must remain a purely selective module. Here, the rounding rounds up or down $M$ with a probability proportional to its values. For instance, given $M_\{i,j\}$, $P(M_{rounded_{(i,j)}}=1) = M_\{i,j\}$ and $P(M_{rounded_{(i,j)}}=0) = 1-M_\{i,j\}$. Stated formally,

$$
\begin{aligned}
& N \coloneqq \text{a noise tensor where } N_\{i,j\} \sim Uniform[0,1] \\
& M_{complement} =  (1 - M).detached()\\\\[0.2cm]
& M_{rounded_{(i,j)}} = 
\begin{cases} 
M_{i,j} + M_{complement_{(i,j)}}  & \text{if } N_{i,j} \leq M_{i,j} \\
M_{i,j} - M_{(i,j)}.detached()  & \text{if } N_{i,j} > M_{i,j} \\
\end{cases} \\
\end{aligned}
$$

The detachment serves to reduce gradient magnitude. Also, the probabilistic rounding is only used during training. During evaluation or inference, the rounding becomes deterministic in the following way

$$
\begin{aligned}
& M_{rounded_{(i,j)}} = 
\begin{cases} 
1  & \text{if } M_{i,j} \geq 0.5  \\
0  & \text{if } M_{i,j} < 0.5  \\
\end{cases} \\
\end{aligned}
$$

At the end, the output of the module is the element-wise product between $X$ and $M_{rounded}$

$$ out_{dropout} =  X \odot M_{rounded} $$

### Dropout L1 norm penalty

Intuitively, more dropout (i.e. more 0s in $M$) is desirable. This intuition stems from the Occam's razor or Minimum Description Length principle. This is also analogous to desiring fewer experts per token in MoE. Yet, the model does not intrinsically favor more dropout. In fact, the opposite would happen because the next token prediction loss function incentivizes the model to use as much compute as possible, hence less dropout. To counter this, a dropout ${L_1}$ norm penalty is added to the final model loss, calculated in the following way

$$ L_{1}\\\_norm\\\_penalty = \left|\frac{M^2}{2}\right|_1$$

Note that the unrounded $M$ is used because it is deterministic. The squaring of $M$ serves to create an non-linear penalty: as $M$ approaches 0, the penalty should decay. The decay and the $\frac{1}{2}$ scaling ensure that the next token prediction objective functions remains primary.

## Results

> All training runs below were done on a wikipedia dataset for 26k steps on a single A100 GPU, unless otherwise stated.
> 
> Implementation of decoder-only transformer model (baseline) can be found in the `baseline_transformer` directory in this repo

First, we evaluate the difference between including the L1 norm penalty and excluding it.

|   | Train loss | Val loss | $M_{rounded}$'s % of 1s |
|---|----------|----------|----------|
| **with penalty** [(config)](#) | **2.993** | 3.387 | 8.564e-9 |
| **without penalty** [(config)](#) | 2.998 | **3.385** | 4.138e-9 |

Next, using the ${L_1}$ norm penalty, different initialization values for $B$ (named shift_init in the charts) are evaluated. The initialization with $0$ performed the best, followed by $\frac{\pi}{2}$ and $\pi$. This matches intuition because initializing with $0$ means that $M$ starts with values closer to 1, and it is easier to go from no dropout to more dropout than viceversa.

<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/shift_train_loss.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/shift_val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/shift_l1_norm.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ |
|---|----------|----------|----------|
| **shift_init = 0** [(config)](#) | **2.937** | **3.384** | 0.6167 |
| **shift_init = pi/2** [(config)](#) | 2.967 | 3.405 | 0.5955 |
| **shift_init = pi** [(config)](#) | 3.055 | 3.457 | **0.2507** |

Compared to a canonical decoder-only transformer (baseline) with no dropout, the new model outperformed the baseline in validation loss only. Both completed in a similar amount of time with similar memory demands, but the baseline had more parameters.


<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/baseline_train_loss.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/baseline_val_loss.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ | Size (params) |
|---|----------|----------|----------|----------|
| **shift_init = 0** [(config)](#) | 2.937 | **3.384** | 0.6167 | 15,335,424 |
| **baseline** [(config)](#) | **2.845** | 3.475 | NA | 15,441,192 |

Three more baselines with $Dropout$ were compared: "0.2 dropout baseline", "0.3 dropout baseline", and "0.4 dropout baseline". The new model outperformed all in validation loss except for "0.2 dropout baseline". Note that even the new model has some $Dropout$ modules and at more places and they were not used. So the model demontrantes its competitiveness with $Dropout$.


<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/dropout_train_loss.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/dropout_val_loss.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ | Size (params) |
|---|----------|----------|----------|----------|
| **shift_init = 0** [(config)](#) | 2.937 | 3.384 | 0.6167 | 15,335,424 |
| **baseline** [(config)](#) | **2.845** | 3.475 | NA | 15,441,192 |
| **0.2 dropout baseline** [(config)](#) | 3.1 | **3.354** | NA | 15,441,192 |
| **0.3 dropout baseline** [(config)](#) | 3.213 | 3.425 | NA | 15,441,192 |
| **0.4 dropout baseline** [(config)](#) | 3.319 | 3.512 | NA | 15,441,192 |

## Next steps

These are some further things to look forward to:
- instead of a single cosine function to map values to $\[0, 1\]$, use a Fourier series
- try bigger models, at least GPT-2 size
- run training for longer to observe long-term behavior
- evaluate on different datasets
- evaluate on non-language tasks

## Results

the new dropout can be implemented in a way that reduces the actual FLOPS, 

---
## Appendix
### Run configs
#### "shift_init = 0"
#### "shift_init = pi/2"
#### "shift_init = pi"
#### "baseline"
#### "0.2 dropout baseline"
#### "0.3 dropout baseline"
#### "0.4 dropout baseline"