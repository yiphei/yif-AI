# Boolean neural net

The **Boolean Neural Net** (BNN) is a new architecture that consists of a novel forward pass and backward pass. More precisely, the forward pass is a weighted boolean conjuction of the input, and the backward pass uses a discrete optimizer (i.e. no gradient descent). So the 'neural net' suffix is a bit of a misnomer since the only major element BNN retains from canonical neural nets is the net shape. Perhaps Boolean Neural Circuit is more adequate. 

### Motivation
I wanted to investigate how a model could become a more rigorous reasoner. Part of the reason for human's ability to do so is symbolic reasoning (this is sometimes also referred as System 2 thinking). Now, I will refrain from opining on the actual symbolic representation in human brains (I actually dont think it's discrete), but I think it's fair to say that even in a continuous space, these symbols tend to be relatively stable. Thus, it became obvious that restricting the data domain of the hidden layers' outputs to just discrete values would be an interesting direction to experiment with.

### Architecture
For this implementation, I placed more restrictions beyond those dictated by the original motivation in order to make this a more lightweight and quicker experiment. In addition to the hidden layers, both the input and output layers are also discrete, and the discrete values are bits. The output of the model is a single bit. Each layer consists of a free parameter $W$, which also consists of bits.

##### Feed forward

Instead of the traditional feedforward algorithm, which is a weighted sum passed through a non-linear activation function, BNN uses a weighted boolean conjuction. Namely, for a single neuron, given an input $X \in \\{1,0\\}^N$ and weights $W \in \\{1,0\\}^{2N}$, we first construct the expanded input

$$ \tilde{X} = X \frown \neg{X} \quad \text{where} \frown \text{represents vector concatenation}$$

then, the output of that neuron is

$$h = \prod_{i} \tilde{X_i} \quad \text{where} \quad i \in \\{j : W_j = 1\\}$$

To offer an example, given input $X = [1,0,1]$ and weights $W = [1,1,0,0,0,1]$, the expanded input is

$$ \tilde{X} = [1,0,1] \frown [0,1,0] = [1,0,1,0,1,0] $$

then, the selected indices are $i \in \{1,2,6\}$, so the output is

$$h = \tilde{X_1} *\tilde{X_2} * \tilde{X_6} = 1 * 0 * 0 = 0$$

At the layer level, the equations become the following:

$$
\begin{align*}
\text{Given input } X &\in \{1,0\}^N \text{ and } W \in \{1,0\}^{2NM}, \text{ where } M \text{ is the width of the layer (i.e. \# of neurons)} \\
\tilde{X} &= X \frown \neg X \quad \text{where } \frown \text{ represents vector concatenation} \\
\mathbf{h} &= \left(h_z\right)_{M}, \quad \text{where} \quad h_z = \prod_{i} \tilde{X}_{i} \quad \text{for} \quad i \in \{j: W_{z,j} = 1\}
\end{align*}
$$

 
The idea behind this forwad pass is to simulate propositional logic using boolean algebra. For inputs, the bit value 0 corresponds to False and 1 corresponds to True. Then, the weights essentially select the inputs (1 = select, 0 = unselect/ignore) and then a boolean conjunction is computed on selected inputs. Just by doing this, each hidden layer can express any (first-order) propositional logic, and with many layers, the model can express any arbitrarily nested (first-order) propositional logic. 

Another way to understand this expressivity is to view the model as building a circuit of NAND gates (technically, it requires two layers to have a single NAND gate).

##### Back prop
The backprop algorithm essentially consists of a variant of the boolean satisfiability (BSAT) problem. In the canonical BSAT, you look for literal values for which a boolean expression consisting of those literals would evaluate to True. In our case, the literals are fixed (they are the inputs), so we look for expressions (i.e. weights) over the input literals that evaluate to the expected value. Unfortunately, BSAT and its variants are more than NP-hard; they are NP-complete. However, there are heuristical solutions that can solve it more quickly on average. Here, I implemented my own heuristically beam search for it. Nonetheless, I regret my implementation’s complexity, which precludes me from summarizing it in beautiful mathematical expressions.

### Evaluation
Because of the NAND gates, it can model non-linear relationships, provided that those relationships can be expressed in bits. There is an accompanying jupyter notebook that shows it learning to predict non-linear data.

### Conclusions
This is a proof of concept and there are many things that can be improved:
- Have a hybrid neural net that uses both this weighted boolean conjunction layer and the traditional weighted sum of continuous values
- Extend support for > 1 bit output shape
- Better backprop beam search

But ultimately, the architecture's achilles heel is also what makes canonical NN so appealing: gradient descent. In my simple experiments, the model simply didn’t scale because of bottleneck caused by the backprop algorithm. There is definitely room for improvements, and indeed, I was able to make incremental gains in algorithmic efficiency, but it is still dwarfed by gradient descent. Therefore, I just decided to end my experiment here.