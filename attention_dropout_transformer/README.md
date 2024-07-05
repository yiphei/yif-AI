# Learned Dropout [WIP readme]
> NB: LaTeX here is optimized for Github's Markdown, so please view it on Github. Also, Safari does not render Github's LaTeX and some SVG files well, so Chrome is advised.

Dropout is a very effective yet simple regularization technique. However, its random implementation relegates it to model training only and makes it invariant to input. Here, I present LearnedDropout, a parametrized dropout that learns the best dropout for each input (i.e. variant to input), and it represents a strong contender to MoE.

## Motivations

Dropout is a very good technique that regularizes the model training to be more robust against overfitting and thus yielding improved generalization. It simply works by randomly setting some values of a tensor to zero, with the ratio of zero values determined by a hyperparameter. When a value is set to zero, it becomes effectively detached from the computational graph, thus all the parameters that contributed to that value won't receive gradients from it. In doing so, Dropout essentially creates a subgraph of the model, and due to the randomness, every forward pass results in a different (transient) subgraph. Then, the final pre-trained model can be viewed as the ensemble of all the different subgraphs Dropout created. Furthermore, each subgraph can be loosely though of an expert, so Dropout also represents a simple method to (weakly) partition the model like MoE.

However, unlike MoE, the random implementation means that 1) it cannot be used during inference and 2) it is invariant to input. 1) limits the benefit of Dropout to pre-training only. 2) is one big reason why MoE yields better performance than Dropout. To overcome this, the model needs to learn the best dropout for every input. Then, the learned dropout becomes a strong competitor to MoE. To continue the MoE comparison, each dropout essentially becomes a expert gating.

## Architecture

At the high-level, the architecture consists of a canonical decoder-only transformer with a modified dropout module. Additional dropout losses are introduced to coerce dropout towards certain characteristics.

<div align="center">
  <img src="assets/decoder_diagram.svg" alt="sdasd" width="40%">
</div>