# Learned Dropout [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

Dropout is a very effective yet simple regularization technique. However, its random implementation relegates it to model training only and renders it invariant to input. Here, I present LearnedDropout, a parametrized dropout that learns the best dropout for each unique input (i.e. variant to input). LearnedDropout represents a strong contender to MoE (Mixture of Experts).

## Motivations

Dropout is a very popular technique that regularizes the model training to be more robust against overfitting and thus yielding improved generalization. It simply works by randomly setting some values of a tensor to zero, with the ratio of zero values determined by a hyperparameter. When a value is set to zero, it becomes effectively detached from the computational graph, thus all the parameters that contributed to that value won't receive gradients from it. In doing so, Dropout essentially creates a subgraph of the model, and due to the randomness, every forward pass results in a different (transient) subgraph. Then, the final pre-trained model constitutes the ensemble of all the different subgraphs Dropout created. Furthermore, observe that this outcome is not so conceptually removed from MoE's outcome. Each subgraph can be loosely though of an expert, so Dropout (very weakly) partitions the model into different experts, like MoE.

Yet, unlike MoE, the random implementation means that 1) it cannot be used during inference and 2) it is invariant to input. 1) limits the benefit of Dropout to pre-training only, but 2) is the larger reason why MoE yields better performance than Dropout. To overcome this, Dropout needs to be parametrized to enable the model to learn the best dropout for every input. Only then can dropout become a strong competitor to MoE.

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with a modified dropout module. Two dropout-specific losses are introduced to encourage low dropout entropy and dropout L1 norm, in addition to next token prediction loss.

<div align="center">
  <img src="assets/decoder_diagram.svg" alt="sdasd" width="40%">
</div>

### LearnedDropout

Like every dropout, LearnedDropout computes a dropout mask $\mathbf{m}$ of 0s and 1s that is applied to the dropout input. The crux lies in the mask's computation. The canonical dropout randomly generates the dropout mask $\mathbf{m}$ from a Bernoulli distribution, with the probablity of 0 determined by the dropout rate hyperparameter. To enable learning, **LearnedDropout** needs to generate the mask in a differentiable way.

First, for a dropout to be effective, it needs to understand the dependencies between tokens. Therefore, the dropout input is passed through a multi-headed attention operation. Stated more formally, etc.

[insert latex]

Then, the attention output needs to be mapped to 0s and 1s. There are many ways to do so, and a two-fold function is implemented here. The first part is mapping it to the 0.5 * cos(out) + 0.5 function. This function lies in the [0,1] domain, and its recurrent property reduces the risk of getting stuck in a local minima, at the potential cost of worse convergence. However, 0.5 * cos(out) + 0.5 does not guarantees 1s and 0s, so the second part scales the output to be closer to 0 and 1. This is important because the dropout needs to remain a purely selective/filter layer, not computational. There are two scaling methods used. TODO

### Dropout penalties

Dropout mask values in-between 0 and 1 just scale down the input, which is undesirable for many reasons, the chief one being it potentially causing vanishing gradients. Therefore, the model should be penalized for dropout mask values far from 0 and 1. The dropout mask entropy $\mathrm{H}$ does exactly that. The dropout mask entropy $\mathrm{H}$ applies Shannon's information entropy to the dropout mask
$$\mathrm{H}(\mathbf{m}) =  \sum_{i}-\mathbf{m}_i\log_2\mathbf{m}_i $$

This ensures that the dropout mask values are pushed as close as possible to $\\{0,1\\}$ since the function's global minima occur there (remember that $\mathbf{m}_i \in [0,1]$).

#### Dropout L1 norm

Intuitively, one should desire for fewer experts (i.e. more dropout) active per token. This intuition stems from the Occam's razor principle. Yet, solely adding learned dropout does not incentivize the model to favor more dropout. In fact, the opposite would happen because the loss function will incentivize the model to use less dropout (more experts, and thus more compute). The Dropout L1 norm serves to counter the otherwise degenerate tendency. The L1 norm ${L_1}$ is just the canonical function

$$ L_1(\mathbf{m}) = |\mathbf{m}|_1$$

which encourages more dropout (more zeroes, fewer ones).
