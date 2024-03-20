# Embed [WIP readme]
> NB: LaTeX is optimized for Github's Markdown. 

Weight tying of input embedding and output layer is a very popular technique that slightly reduces model size without detriments to performance. I propose a weight tying variant that, instead of using weight tying to compute embedding affinity, it uses it to compute embedding distance.

## Motivations

TODO

## Architecture

Traditional weight tying works like this. Given the input embedding matrix $E \in \mathbb{R}^{V \times D}$, where $V$ is the vocabulary size and $D$ is the embedding size, the final logits are computed in the following way

$$ logits = E \cdot X $$

where $X$ is the input to the output layer. What this does is placing an expectation that $X$ represents affinity scores to the output. The higher the affinity between $E$ and $X$ for a particular token, then the higher logit for that token will be, thus lowering the loss.

In my weight tying, it works like this. Once $X$ is computed, it is first subtracted from the position embedding

$$ X_{\text{token}} = X - P_{i+1} $$

then, $MSE$ between $E$ and $X$ is computed

$$ logits = \frac{1}{D} \sum_{D} (E - X_{\text{token}})^2 $$

What this does is placing the expectation that $X$ actually represents the true (input) embedding of the output.

## Analysis/experiments

TODO

## Conclusions

TODO