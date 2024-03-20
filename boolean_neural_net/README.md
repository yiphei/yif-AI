# Boolean neural net
> NB: LaTeX is optimized for Github's Markdown. 

The **Boolean Neural Net** (BNN) is a new architecture that consists of a novel forward pass and backward pass. More precisely, the forward pass is a weighted boolean conjunction of the input, and the backward pass uses a discrete optimizer (i.e. no gradient descent). So the 'neural net' suffix is a bit of a misnomer since the only major element BNN retains from canonical neural nets is the net shape. Perhaps Boolean Neural Circuit is more adequate. 

## Motivation
I wanted to investigate how a model could become a more rigorous reasoner. Part of the reason for human's ability to do so is symbolic reasoning (this is sometimes also referred to as System 2 thinking). Now, I will refrain from opining on the actual symbolic representation in human brains (I actually dont think it's discrete), but I think it's fair to say that even in a continuous space, these symbols tend to be relatively stable. Thus, it became obvious that restricting the data domain of the hidden layers' outputs to just discrete values would be an interesting direction to experiment with.

## Architecture
For this implementation, I placed more restrictions beyond those dictated by the original motivation to make this a more lightweight and quicker experiment. In addition to the hidden layers, both the input and output layers are also discrete, and the discrete values are bits. The output of the model is a single bit. Each layer consists of a free parameter $W$, which also consists of bits.

### Feed forward

Instead of the traditional feedforward algorithm, which is a weighted sum passed through a non-linear activation function, BNN uses a weighted boolean conjunction. Namely, for a single neuron, given an input $X \in \\{1,0\\}^N$ and weights $W \in \\{1,0\\}^{2N}$, we first construct the expanded input

$$ \tilde{X} = X \frown \neg{X} \quad \text{where} \frown \text{represents vector concatenation}$$

then, the output of that neuron is

$$h = \prod_{i} \tilde{X_i} \quad \text{where} \quad i \in \\{j : W_j = 1\\}$$

To offer an example, let's say that we have input $X = [1,0,1]$ and weights $W = [1,1,0,0,0,1]$. The expanded input is

$$ \tilde{X} = [1,0,1] \frown [0,1,0] = [1,0,1,0,1,0] $$

then, the selected indices are $i \in \\{1,2,6\\}$, so the neuron output is

$$h = \tilde{X_1} \cdot \tilde{X_2} \cdot \tilde{X_6} = 1 \cdot 0 \cdot 0 = 0$$

The operations at the neuron level can be batched and optimized at the layer level. Given an input $X \in \\{1,0\\}^N$, you now have weights $W \in \\{1,0\\}^{M\times 2N}$, where $M$ is the width of the layer (i.e. # of neurons). Then, with $\tilde{X}$ being calculated the same way, you have the output of the layer as the vector

$$\mathbf{h} = \left(h_z\right)_{M}, \quad \text{where} \quad h_z = \prod\_{i} \tilde{X}\_{i} \quad \text{for} \quad i \in \\{j: W\_{z,j} = 1\\}$$

 
The idea behind this forward pass is to simulate propositional logic using boolean algebra. For inputs, the bit value 0 corresponds to False, and 1 corresponds to True. Then, inputs are expanded to include their negations. Finally, the weights select the inputs (1 = select, 0 = unselect/ignore), and then a boolean conjunction is constructed on selected inputs. Just by doing this, each hidden layer can express any (first-order) propositional logic, and with many layers, the model can express any arbitrarily nested (first-order) propositional logic. 

Another way to understand this expressivity is to view the model as building a circuit of NAND gates (technically, it requires two layers to have a single NAND gate). Remember that the NAND operation is functionally complete, meaning that any Boolean expression can be equivalently re-expressed with only NAND operations.

#### Example: XOR

The XOR operation on two inputs is one of the simplest non-linear relationships. The simplest BNN model that captures this relationship is the following

![XOR example](assets/BNN.svg)

(Note: this image won't render well if you have dark mode on)

where $x_1$ and $x_2$ are the inputs to the model. Converting the above model's forward pass into a single boolean expression, we get

$$ O = \neg (\neg x_1 \wedge \neg x_2) \wedge \neg (x_1 \wedge x_2)  $$

which precisely captures the XOR relationship using just conjunction and negation.

### Back prop
The backprop algorithm essentially consists of a variant of the boolean satisfiability (BSAT) problem. In the canonical BSAT, you look for literal values for which a boolean expression consisting of those literals would evaluate to True. In our case, it's the inverse. The literals are fixed (they are the inputs), so we look for expressions (i.e. weights) over the input literals that evaluate to the expected value. Unfortunately, BSAT and its variants are more than NP-hard; they are NP-complete. Yet, there are heuristical solutions that can solve it performantly on average. Here, I implemented my own heuristical beam search. Nonetheless, I regret my implementation’s complexity, which precludes me from distilling it into beautiful mathematical expressions.

## Evaluation
Because of the NAND gates, it can model non-linear relationships, provided that those relationships can be expressed in bits. There is an accompanying jupyter notebook that shows it learning to predict non-linear data.

The big advantage of the architecture is interpretability. Once the model has successfully learned from the dataset, you can always extract the boolean expression algorithm from the weights.

## Conclusions
This is a proof of concept, and many things that can be improved:
- Have a hybrid neural net that uses both this weighted boolean conjunction layer and the traditional weighted sum layer
- Extend support for > 1 bit output shape
- Better backprop beam search

But ultimately, the architecture's Achilles heel is also what makes canonical NN so appealing: gradient descent. In my simple experiments, the model simply didn’t scale because of the bottleneck caused by the backprop algorithm. There is definitely room for improvement, and indeed, I was able to make incremental gains in algorithmic efficiency, but it is still dwarfed by gradient descent. Therefore, I just decided to end my experiment here.