# Neural Networks

Neural networks are machine learning models inspired by the structure of biological neural networks. 
They are sometimes referred to as artificial neural networks to distinguish them from their biological inspirations.
They consist of connected nodes where the output of a neuron is calculated as a nonlinear function of its inputs.
The degree to which a neuron uses a particular input is determined by a weight, which is adjusted during the learning process.
A network is called a deep neural network if it has at least two hidden layers. 
Network depth has proved important is learning hierarchical representations of data, enabling improved performance on structured data.
Since the 2010s, neural networks have emerged as the most important and performant class of models for machine learning applied to structured data.

The outputs of layer $l$ of a fully-connected neural network are generally given by
\begin{equation}
\mathbf{z}^{(l)} = h^{(l)}(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)},
\end{equation}
where $h^{(l)}$ denotes the activation function associated with layer $l$ and $\mathbf{W}^{(l)}$ is the layer's weights and bias parameters.

<!-- Potentially replace above equation with more general version of eq. 6.22 from Bishop) -->

<!-- Possibility to add: refer to a neural network with n hidden layers as a n+1-layer neural network, since there are n+1 layers of learnable parameters -->

## General Considerations

<!-- Note: should add contrastive learning as a learning paradigm to general ML review. This is reviewed in 6.3.5 of Bishop -->

### Hidden Unit Activation Functions

The activation function for neural network outputs should be determined from the task it is performing.
For the hidden units, however, various activation functions could be reasonable and have been tried in practice. 
Empirically, one of the best-performing activation functions is the Rectified Linear Unit, or ReLU, defined by
\begin{equation}
h(a) = \max(0, a),
\end{equation}
where $a$ is the input.

### Learning with Gradient Descent

...

### Regularization

...

### Neural Networks are Universal Approximators

Various works have shown that two-layer neural networks can approximate any function defined over a continuous subset of R^D with arbitrary accuracy.
Thus, neural networks are said to be universal approximators.
Practically, however, deep neural neural networks have been found to be important in performance at various tasks and hierarchical representation learning.


## Architectures

There are a variety of different neural network architectures that have been used or investigated. 
Here, we review a few of the currently most important ones for applications.

### Transformers

<!-- Transformers can be viewed as GNNs: https://graphdeeplearning.github.io/post/transformers-are-gnns/ -->


## References

1. ["Deep Learning: Foundations and Concepts"](https://www.bishopbook.com) by Chris Bishop (2023).
2. [Stanford Machine Learning Courses Cheatsheets](https://stanford.edu/~shervine/teaching/cs-221/)
