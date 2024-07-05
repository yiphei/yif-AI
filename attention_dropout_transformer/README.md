# Learned Dropout [WIP readme]
> NB: LaTeX is optimized for Github's Markdown. 

Dropout is a very effective and simple technique that reduces overfitting and thus improve model generalization. However, its random implementation relegates it to model training only and makes it invariant to input. Here, I present LearnedDropout, a parametrized dropout that learns the best dropout for each input (i.e. variant to input). This makes LearnedDropout very close to MoE, and perhaps even better than it.

## Motivations

Dropout is a very good technique that regularizes the model training to be more robust against overfitting and thus yielding improved generalization. It simply works by randomly setting some values of a tensor to zero, the ratio of zero values being determined by a hyperparameter. When a value is set to zero, it is effectively detached from the computational graph, thus all the parameters that gave rise to the value won't be updated with respect of it. In doing so, Dropout essentially creates a different (transient) subgraph of the model on every forward pass. The final pre-trained model can then be understood as the ensemble of all the different subgraphs. Each subgraph can be though of an expert. Therefore, using Dropout is one way to indirectly get highly diffused and weak MoE. 

However, unlike MoE, the random implementation means that it cannot be used during infererence and is invariant to input. To overcome this, we need to parametrized it. Once done so, then resulting dropout becomes even closer to MoE, if not better.