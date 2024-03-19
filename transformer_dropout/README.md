# Learned Dropout [WIP readme]
> NB: LaTeX is optimized for Github's Markdown. 

Current SOTA LLMs have one frequent thing in common: they use Mixture of Experts (MoE). However, all MoE implementations require setting, in some form or other, a hyperparameter of how many experts to use, at pre-training time. This is limiting because it requires determining the hyperparamter value, which is usually found with a rather ad-hoc process. Ideally, the model itself learns the best number of experts. To this end, I created a new module that helps a model achieve MoE without the explicit constraint of number of experts.

## Motivations

MoE is not a new concept. In fact, a popular precursor to MoE was ensemble models. Both helped improve models' performance. Presently, notable models that use MoE include OpenAI GPT-4, Mixtral 8x7B, xAI Grok-1, and Google Gemini.

What makes MoE so popular? First of all, at the intuitive level, it mimics how intelligence is organized in the real world. Across all scale levels, intelligence is specialized. For instance, at the societal level, intelligence specialization manifests in highly specialized modern economies. At the biological level, it has been known that the brain contains regions specialized in specific tasks. Empirically, MoE have demonstrated a range of benefits: cheaper training & inference, increased model capacity, more efficient parameters, and improved generalization.

Now, all present MoE implementations require a hyperparameter that specifies how many experts to use per forward pass. This value is unchanged for inference. The goal of the new module introduced here is to remove this hyperparameter, permitting the model to learn the best number of experts per token type, not per forward pass. The “per token type” part is important because it means that two different tokens can have different numbers of experts being used.

Before I proceed on this was implemented, I want to briefly review Dropout. Dropout is a very popular and simple training technique that regularizes the model training to be more robust against overfitting and thus yielding improved generalization. It simply works by randomly setting some weights of the model to zero, so they will be ignored during backprop. In doing so, Dropout essentially creates a different (transient) subgraph of the model on every forward pass. The final pre-trained model can then be understood as the ensemble of all the different subgraphs, each of which can be thought of an expert. Therefore, using Dropout is one way to indirectly get MoE. But again, you have to set a hyperparameter for the dropout rate, and Dropout is not activated at inference time.

The new module borrows the Dropout idea but allows the model to learn the best weights to dropout. This new module remains active at inference time.

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