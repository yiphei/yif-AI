# Learned Dropout [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

$Dropout$ is a very effective yet simple regularization technique. However, its random implementation relegates it to training time only and renders it invariant to input. Here, I present $LearnedDropout$, a parametrized dropout module that learns the best dropout for each unique input (i.e. variant to input). Results demonstrate its efficacy and competitiveness with both the canonical $Dropout$ and MoE (Mixture of Experts).

## Motivations

$Dropout$ is a very popular technique that regularizes the model training to be more robust against overfitting and thus yields improved generalization. It simply works by randomly setting some values of a tensor to zero, with the ratio of zero values determined by a hyperparameter. When a value is set to zero, it becomes effectively detached from the computational graph, so all the parameters that contributed to that value will have a gradient of 0 w.r.t. that value. In doing so, $Dropout$ essentially creates a subgraph of the model because setting values to zeroes practically turns off part of the model. Given the randomness, every forward pass results in a different (transient) subgraph. Then, the final pre-trained model constitutes the ensemble of all the different subgraphs $Dropout$ created. Furthermore, observe that this outcome is not so conceptually removed from MoE's outcome. Each subgraph can be loosely thought of as an expert, and through these subgraphs, $Dropout$ (very weakly) partitions the model into different experts, like MoE.

Yet, unlike MoE, the random implementation means that 1) it is not useful during inference and 2) it is invariant to input. 1) limits the benefit of $Dropout$ to pre-training only, but 2) represents the larger reason why MoE produces better performance than $Dropout$. To overcome these deficits, the dropout module needs to be parametrized to permit the model to learn the best dropout, for every unique input. This change should make it a compelling alternative to both $Dropout$ and MoE.

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with the new dropout module $LearnedDropout$. To encourage more dropout, a dropout ${L_1}$ norm penalty is added to the model loss.

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

where $B \in \[0, \pi\]$ is a shift bias term. This function lies in the $\[0,1\]$ range, and its recurrent property eliminates the risk of dropout becoming stuck in a local minimum, though at the cost of worse convergence.

Lastly, a rounding is applied to bring $M$ to $\\{0,1\\}$ to satisfy $M \in \\{0, 1\\}$. The rounding is important because, otherwise, the model might use the dropout module for computational ends (e.g. scaling of $X$). $LearnedDropout$ must remain a purely selective module. The rounding rounds up or down $M$ with a probability proportional to its values. For instance, given $M_\{i,j\}$, $P(M_{rounded_{(i,j)}}=1) = M_\{i,j\}$ and $P(M_{rounded_{(i,j)}}=0) = 1-M_\{i,j\}$. Stated formally,

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

In the end, the output of the module is the element-wise product between $X$ and $M_{rounded}$

$$ out_{dropout} =  X \odot M_{rounded} $$

### Dropout L1 norm penalty

Intuitively, more dropout (i.e. more 0s in $M$) is desirable. This intuition stems from the Occam's razor or Minimum Description Length principle. This is also analogous to desiring fewer experts per token in MoE. Yet, the model does not intrinsically favor more dropout. In fact, the opposite could happen because the next token prediction loss function can incentivize the model to use as much compute as possible, hence less dropout. To counter this, a dropout ${L_1}$ norm penalty is added to the final model loss, calculated in the following way

$$ L_{1}\\\_norm\\\_penalty = \left|\frac{M^2}{2}\right|_1$$

Note that the unrounded $M$ is used because it is deterministic. The squaring of $M$ serves to create a non-linear penalty: as $M$ approaches 0, the penalty should decay. The decay and the $\frac{1}{2}$ scaling ensure that the penalty does not take precedence over next token prediction.

## Results

> All training runs below were done on a wikipedia dataset for 26k steps on a single A100 GPU, unless otherwise stated.
> 
> Implementation of decoder-only transformer model (baseline) can be found in the `baseline_transformer` directory in this repo

First, the inclusion and exclusion of ${L_1}$ norm penalty were compared. Both had the shift bias $B$ initialized to $0$. The inclusion of the penalty outperformed its exclusion in validation loss but underperformed it in train loss. Surprisingly, the penalty absence did not detract the model from having more dropout over time, though at a much slower rate.

<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/penalty_train_loss.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/penalty_val_loss.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/penalty_l1_norm.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ |
|---|----------|----------|----------|
| **with penalty** [(config)](#) | 2.937 | **3.384** | **0.6167** |
| **without penalty** [(config)](#without-penalty) | **2.911** | 3.403 | 0.9609 |

Next, using the ${L_1}$ norm penalty, different initialization values for the shift bias $B$ were evaluated. The $0$ initialization performed the best, followed by $\frac{\pi}{2}$ and $\pi$. This matches intuition because initializing with $0$ means that $M$ starts with values closer to 1, and it is easier to go from no dropout to better dropout than vice versa. Yet, the $\pi$ initialization performed very competitively despite dropping out about 75% of values on average.

<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/shift_train_loss_a.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/shift_val_loss_a.svg" alt="Image 2" style="width: 45%;"/>
    <img src="assets/shift_l1_norm.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ |
|---|----------|----------|----------|
| **shift_init = 0** [(config)](#shift_init--0) | **2.937** | **3.384** | 0.6167 |
| **shift_init = pi/2** [(config)](#shift_init--pi2) | 2.967 | 3.405 | 0.5955 |
| **shift_init = pi** [(config)](#shift_init--pi) | 3.055 | 3.457 | **0.2507** |

Compared to a canonical decoder-only transformer (baseline) with no dropout, the new model outperformed the baseline in validation loss only. Both completed in a similar amount of time with similar memory demands, but the baseline had more parameters.


<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/baseline_train_loss_a.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/baseline_val_loss_a.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ | Size (params) |
|---|----------|----------|----------|----------|
| **shift_init = 0** [(config)](#shift_init--0) | 2.937 | **3.384** | 0.6167 | 15,335,424 |
| **baseline** [(config)](#baseline) | **2.845** | 3.475 | NA | 15,441,192 |

Three more baselines with $Dropout$ were compared: "0.2 dropout baseline", "0.3 dropout baseline", and "0.4 dropout baseline". The new model outperformed all in validation loss except for "0.2 dropout baseline". Note that even the new model has some $Dropout$ modules and at more places and they were not used. So the model demonstrates its competitiveness with $Dropout$.


<div>
  <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; align-content: flex-start;">
    <img src="assets/dropout_train_loss_a.svg" alt="Image 1" style="width: 45%;"/>
    <img src="assets/dropout_val_loss_a.svg" alt="Image 2" style="width: 45%;"/>
  </div>
    <div align="center">
      <em>Safari may not render the charts above. Chrome is advised.</em>
    </div>
</div>
<br>

|   | Train loss | Val loss | average % of 1s in $M_{rounded}$ | Size (params) |
|---|----------|----------|----------|----------|
| **shift_init = 0** [(config)](#shift_init--0) | 2.937 | 3.384 | 0.6167 | 15,335,424 |
| **baseline** [(config)](#baseline) | **2.845** | 3.475 | NA | 15,441,192 |
| **0.2 dropout baseline** [(config)](#02-dropout-baseline) | 3.1 | **3.354** | NA | 15,441,192 |
| **0.3 dropout baseline** [(config)](#03-dropout-baseline) | 3.213 | 3.425 | NA | 15,441,192 |
| **0.4 dropout baseline** [(config)](#04-dropout-baseline) | 3.319 | 3.512 | NA | 15,441,192 |

## Next steps

These are some further things to look forward to:
- instead of a single cosine function that maps values to $\[0, 1\]$, use a Fourier series
- try bigger models, at least GPT-2 size
- run training for longer to observe long-term behavior
- evaluate on different datasets
- evaluate on non-language tasks

## Conclusions

Instead of setting a dropout rate (or top-k experts in MoE's case) hyperparameter, $LearnedDropout$ permits the model to learn the best dropout for each unique input. As demonstrated by the results, this produces both competitive performances and more efficient models.

The end result of $LearnedDropout$ is adaptive sparsity, very similar to MoE. Unlike MoE, though, it doesn't require dictating a pre-defined sparsity target. With more sparsity, fine-tuning can improve since sparsity can reduce catastrophic forgetting. Furthermore, the adaptive sparsity can also enable the training paradigm whereby an over-parametrized dense model is used and is sparsified over time by $LearnedDropout$.

Alas, the principal limitation is my personal compute budget, so this project cannot avail itself of further analysis and experimentation.

## Citation

If you use this codebase, or otherwise found my work valuable, please cite:

```
@misc{yan2024learned-dropout,
  title={Learned Dropout},
  author={Yan, Yifei},
  year={2024},
  url={https://github.com/yiphei/yif-AI/tree/main/learned_dropout}
}
```

---
## Appendix
### Run configs
#### "without penalty"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 9,
                  'n_embed': 144,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'l1_norm_penalty_type': None,
                  'learned_dropout_config': {'n_head': 9,
                                             'use_bias': False,
                                             'shift_init': 0,
                                             'dropout_input_type': 1,
                                             'mask_rounding_type': 3,
                                             'use_detached_input': False},
                  'use_dropout_entropy_penalty': False,
                  'use_dropout_l1_norm_penalty': False},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "shift_init = 0"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 9,
                  'n_embed': 144,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'l1_norm_penalty_type': 2,
                  'learned_dropout_config': {'n_head': 9,
                                             'use_bias': False,
                                             'shift_init': 0,
                                             'dropout_input_type': 1,
                                             'mask_rounding_type': 3,
                                             'use_detached_input': False},
                  'use_dropout_entropy_penalty': False,
                  'use_dropout_l1_norm_penalty': True,
                  'dropout_l1_norm_coeff_config': {'max_coeff': 0.1}},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "shift_init = pi/2"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 9,
                  'n_embed': 144,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'l1_norm_penalty_type': 2,
                  'learned_dropout_config': {'n_head': 9,
                                             'use_bias': False,
                                             'shift_init': 1.57079,
                                             'dropout_input_type': 1,
                                             'mask_rounding_type': 3,
                                             'use_detached_input': False},
                  'use_dropout_entropy_penalty': False,
                  'use_dropout_l1_norm_penalty': True,
                  'dropout_l1_norm_coeff_config': {'max_coeff': 0.1}},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "shift_init = pi"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 9,
                  'n_embed': 144,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0,
                  'l1_norm_penalty_type': 2,
                  'learned_dropout_config': {'n_head': 9,
                                             'use_bias': False,
                                             'shift_init': 3.14159,
                                             'dropout_input_type': 1,
                                             'mask_rounding_type': 3,
                                             'use_detached_input': False},
                  'use_dropout_entropy_penalty': False,
                  'use_dropout_l1_norm_penalty': True,
                  'dropout_l1_norm_coeff_config': {'max_coeff': 0.1}},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "baseline"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 12,
                  'n_embed': 156,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
 ```
#### "0.2 dropout baseline"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 12,
                  'n_embed': 156,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0.2},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "0.3 dropout baseline"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 12,
                  'n_embed': 156,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0.3},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```
#### "0.4 dropout baseline"
```
{'lr': 0.0009,
 'beta1': 0.9,
 'beta2': 0.95,
 'min_lr': 9e-05,
 'decay_lr': True,
 'est_steps': 200,
 'batch_size': 50,
 'train_steps': 26000,
 'est_interval': 500,
 'model_config': {'n_head': 12,
                  'n_embed': 156,
                  'n_layer': 26,
                  'use_bias': False,
                  'context_size': 200,
                  'dropout_rate': 0.4},
 'warmup_iters': 300,
 'weight_decay': 0.1,
 'lr_decay_iters': 700000,
 'gradient_accumulation_steps': 16}
```