# Learned Dropout [WIP readme]
> NB: LaTeX is optimized for Github's Markdown. 

A new architecture that uses a novel dropout module to achieve adaptive MOE.

## Motivations

Current SOTA LLMs have one frequent thing in common: they are MOE models. The concept of MOE is not new. For instance, ensemble models were the precursor to MOE and they helped achieve better performance by having many models contribute to the final output. MOE constitutes a reincarnation of ensemble models in LLMs, and they have proved their worth. Notable SOTA models that leverage MOE are: [insert models here]. 

There are many benefits to MOE. Intuitively, it mimics how intelligence is organized in the real world. At the societal level, intelligence is dependent on specialization to produce great output. At the individual level, it has been known that the brain contains regions specialized in specific tasks. Empirically, MOE models have shown to yield greater performance while utilizing less compute per token.

Presently, these MOE models are trained with a hyperparameter that generally specifies how many experts to use per forward pass. This number usually either defines the upper bound on how many experts should be used or the actual constant number of experts to be used. This same number is used at inference time. But restricting the number of experts is very limiting. Therefore, the motivation of this new architecture is to remove this hyperparameter, permitting the model to learn the best number of experts per specific token, not per forward pass. The “per specific token” part is important because it means that two different tokens can have different numbers of experts being used.

Before I proceed on this can be done, i just want to briefly review dropout. Dropout is a very popular and simple technique that regularizes the training to be more robust against overfitting. It simply works by randomly setting some weights of the model to zero, so gradients will not flow through them. In doing so, it essentially creates a different transient subgraph of the model on every forward pass. In the end, the model training consists of training on many different subgraphs, each of which can be though of as an expert. So using dropout is one way to indirectly get MOE. But again, you have to specify a hyperparameter on the dropout percentage, and the dropout is not active at inference time.

Therefore, the new architecture below introduces a new dropout module called LearnedDropout. It essentially takes dropout and, instead of setting the dropout percentage and deactivating it during inference, it learns the best weights to drop, which in doing so creates a MOE model, and it remains active at inference.

## Architecture

The architecture consists of a vanilla transformer architecture with the new LearnedDropout module being applied. The specific vanilla transformer implementation is largely borrowed from the awesome https://github.com/karpathy/nanoGPT/blob/master/model.py implementation, with the new module being applied in the same places as the one present in the borrowed implementation. 

### LearnedDropout (LD)

At the high level, the LearnedDropout implementation computes a dropout mask from the input tokens and applies it onto input. The key part is how this dropout mask is computed. In the regular dropout module, the dropout mask $\mathbf{m}$ is simply a tensor of zeroes and ones, with the number of zeroes determined by the dropout rate hyperparamter. To make the dropout mask to be learned by the model, we need to find a differentiable way to compute the dropout mask. In LD, this is done with a sinusoidal function. More precisely, let's assume the input is a single token with $N$ channels. So given an input $X \in \R^{N}$, the dropout mask $\mathbf{m}$ is computed

$$\mathbf{m} =  0.5 cos(\Alpha * X + \Beta) + 0.5$$ 

where $\Alpha \in \R^{N}$ and $\Beta \in \R^{N}$ are free parameters that the model learns. Once the mask is computed, you just apply it to $X$

$$ X = X * \mathbf{m}$$

The $0.5$ terms in the cosine functions serve to bound the function domain to $[0,1]$, and the parameters $\Alpha$ and $\Beta$ change the angular frequency and phase angle, respectively. 

Also, the X used in computing m is detached, so A and B's gradients dont affect X.

#### Regularizing terms
There are two penalty terms present in this architecture: dropout mask entropy $\Eta$ and dropout mask L1 norm ${L_1}$.

The dropout mask entropy $\Eta$ just applies Shannon's information entropy to the dropout mask
$$\Eta(\mathbf{m}) =  \sum_{i}-\mathbf{m}_i\log_2\mathbf{m}_i $$

This ensures that the dropout mask values are pushed as close as possible to $\{0,1\}$ since those represent the function's global minima (remember that $\mathbf{m}$ is bounded by $[0,1]$).

${L_1}$ is just the normal l1 norm function

$$ L_1(\mathbf{m}) = |\mathbf{m}|_1$$

This penalty term is to encourage more dropout (more zeroes, fewer ones).Intuitively, you should desire that fewer experts (i.e. more dropout) are active per token. This intuition stems from the Occam's razor principle. Yet, solely adding learned dropout does not incentivize the model to favor more dropout. In fact, the opposite would happen because the loss function will incentivize the model to use less dropout (more experts, and thus more compute).

The final loss function is

$$ loss = cross\_entropy(\theta, X, Y) + \Eta(\mathbf{m}) + L_1(\mathbf{m})$$

#### Additional LearnedDropout hyperparameters

TODO

## Analysis/experiments

TODO

## Conclusions

TODO