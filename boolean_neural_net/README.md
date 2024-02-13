# Boolean neural net

A novel architecture thats consists of a novel forward pass and backward pass. Specifically, the forward pass is a boolean conjuction of the input, and the backward pass uses a discrete optimizer (i.e. no gradient descent).

### Motivation
I wanted to investigate how a model can become a more rigorous reasoner. Part of the reason for human's ability to do so is symbolic reasoning (this is sometimes also referred as System 2 thinking). Now, I will refrain from opining on the actual symbolic representation in human brains (I actually dont think they are discrete), but I think its fair to say that even in a continuous space, these symbols tend to be relatively stable. Thus, it became obvious that restricting the data domain to just discrete values would be an interesting direction to experiment with.

### Architecture
The inputs to the model are bits, and the output of the model is a single bit (just my first implementation's restriction; it can definitely be extended). Each layer consists of a free parameter W, which is also consists of bits.

##### Feed forward

Instead of the traditional feedforward algorithm {insert also here}, BNN uses a boolean conduction {insert formula here}

The idea behind this implementation is to simulate propositional logic using boolean algebra. For inputs, a value 0 corresponds to True and value 1 corresponds to False. Then, the weights essentially select the inputs (1 = select, 0 = unselect) and then a boolean conjunction is computed. Just by doing this, you can express any arbitrarily nested (first-order) propositional logic. 

Because of this expressivity, you can also see the neural model as building an actual circuit of NAND gates (technically, it requires two layers to have a NAND gate)

##### Back prop
The backprop algorithm essentially consists of a version of the boolean satisfiability (BSAT) problem. In BSAT, you look for literal values for which a boolean expression would evaluate to True. In our case, the literals are fixed, so we look for expressions (i.e. weights) over the input literals that evaluate to the corresponding value. Unfortunately, BSAT and its variants are more than NP-hard; they are NP-complete. However, there are heuristical solutions that can solve it quickly on average. Here, I implemented my own heuristically beam search for it. Nonetheless, I regret my implementation’s complexity, which precludes me from summarizing it in beautiful mathematical expressions

### Evaluation
Because of the NAND gates, it can model non-linear relationships, provided that those relationships can be expressed in bits. There is a notebook that shows it learning to predict bits

### Conclusions
This is a proof of concept  and there are many things to take this further:
- Have a hybrid neural net that uses both this boolean layer and the traditional weighted sum of continuous values
- Extend support for > 1 bit output shape

But ultimately, its achilles heel is also what makes canonical NN so appealing: gradient descent. In my simple experiments, the model simply didn’t scale because of bottleneck caused by the backprop algorithms. There is definitely room for improvements, and indeed, I was able to make incremental gains in algorithmic efficiency, but it is still dwarfed by gradient descent. So I just decided to end my experiment.

