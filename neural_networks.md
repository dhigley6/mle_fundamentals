# Neural Networks

Neural networks are machine learning models inspired by the structure of biological neural networks. 
They are sometimes referred to as artificial neural networks to distinguish them from their biological inspirations.
They consist of connected nodes where the output of a neuron is calculated as a nonlinear function of its inputs.
The degree to which a neuron uses a particular input is determined by a weight, which is adjusted during the learning process.
A network is called a deep neural network if it has at least two hidden layers. 
Network depth has proved important is learning hierarchical representations of data, enabling improved performance on structured data.
Since the 2010s, neural networks have emerged as the most important and performant class of models for machine learning applied to structured data.

The outputs of layer $l$ of a fully-connected neural network are generally given by

$$
\mathbf{z}^{(l)} = h^{(l)}(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)}),
$$

where $h^{(l)}$ denotes the activation function associated with layer $l$ and $\mathbf{W}^{(l)}$ is the layer's weights and bias parameters.

<!-- Potentially replace above equation with more general version of eq. 6.22 from Bishop) -->

<!-- Possibility to add: refer to a neural network with n hidden layers as a n+1-layer neural network, since there are n+1 layers of learnable parameters -->

## General Considerations

<!-- Note: should add contrastive learning as a learning paradigm to general ML review. This is reviewed in 6.3.5 of Bishop -->

### Hidden Unit Activation Functions

The activation function for neural network outputs should be determined from the task it is performing.
For the hidden units, however, various activation functions could be reasonable and have been tried in practice. 
Empirically, one of the best-performing activation functions is the Rectified Linear Unit, or ReLU, defined by

$$
h(a) = \max(0, a),
$$

where $a$ is the input.

### Residual Connections

Residual connections enable the training of deep networks by adding the input of a layer directly to its output. This allows networks to learn a residual mapping, which is the difference between the desired output and the input. Residual connections help solve the vanishing gradient problem by providing an alternative direct path for the gradients to flow.

### Learning with Gradient Descent

Minimizing neural network loss functions is a high dimensional optimization problem in the space of the weights of the neural network. 
Thus, gradient-based optimization methods are used since higher-order optimization methods will be more computationally intensive (the gradient has O(w) parameters, where w is the number of weights while the Hessian has O($w^2$)).

<!-- Add stochastic gradient descent with mini-batches here -->

<!-- Add ADAM here -->

<!-- Add He initialization here -->

<!-- Add batch/layer normalization here -->

To determine the relevant derivatives for the optimization processes above, one typically uses automatic differentiation.
In reverse-mode automatic differentiation, one augments intermediate variables in the calculation of a neural network output with adjoint variables which can be evaluated sequentially starting from the output of the network using the chain rule of calculus.
This is implemented as a core part of standard machine learning frameworks like Tensorflow and Pytorch.

<!-- TODO: Add elaboration on autograd -->

### Regularization

Regularization is a set of techniques used to prevent overfitting by biasing machine learning algorithms towards models which are more likely to generalize well, typically simpler models with higher levels of smoothness in feature space.

#### Accounting for Symmetries/Invariances of the Problem

One important form of regularization is to account for symmetries and invariances of the problem into the model selection process. Ways of accomplishing this include
1. Pre-processing: Use features that are invariant under the required transformations.
2. Regularized error function: The error function includes a regularization term that penalizes changes in model prediction while a desired symmetry is preserved.
3. Data augmentation: The training dataset is expanded to include additional observations which are transformed examples of original observations where the applied transformations should not change model predictions.
4. Network architecture choice: The symmetry properties of the problem are built into the network architecture. An example is convolutional neural networks, which are designed to be translation equivariant.

#### Weight Decay

Another common method of regularization is to add a regularization term to the error function proportional to the sum of squares of the weights in the model:

$$
\tilde{E}(\mathbf{w}) + \frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

This can be interpreted as adding a zero-mean Gaussian prior to the weight distribution and encourages weights to decay towards zero, as seen by the gradient of the error function,

$$
\nabla \tilde{E}(\mathbf{w}) = \nabla E(\mathbf{w}) + \lambda \mathbf{w}
$$

#### Early Stopping

Early stopping is another regularization technique which avoids overfitting by stopping the model training process when error on a validation set starts to consistently increase with further training.

#### Parameter Sharing

Parameter sharing is when groups of weights in a network share the same value learned from data. Convolutional neural networks can be viewed as one example of this. It is also possible to do soft parameter sharing by adding terms to error functions which encourage weights to be close to eachother

#### Dropout

In dropout, nodes are deleted at random from a network during training. Each time a data point is presented during training, a new choice of which nodes to drop is made. Dropout can be viewed as a method of approximate model averaging. After training, model predictions are made without nodes being dropped out.

### Neural Networks are Universal Approximators

Various works have shown that two-layer neural networks can approximate any function defined over a continuous subset of R^D with arbitrary accuracy.
Thus, neural networks are said to be universal approximators.
Practically, however, deep neural neural networks have been found to be important in performance at various tasks and hierarchical representation learning.


## Architectures

There are a variety of different neural network architectures that have been used or investigated. 
Here, we review a few of the currently most important ones for applications.

### Transformers

<!-- Transformers can be viewed as GNNs: https://graphdeeplearning.github.io/post/transformers-are-gnns/ -->

Transformers are composed of transformer layers that transform a set of vectors into another set of vectors having the same dimensionality. 
Since their introduction in 2017, transformers have greatly surpassed Recurrent Neural Networks (RNNs) on Natural Language Processing tasks, as well as outperformed convolutional neural networks on image processing and been used on multimodal datasets combining multiple types of data.

Transformers rely on attention mechanisms.
The algorithm for scaled dot product self-attention, one of the most important and used types is as follows (from algorithm 12.1 of Bishop)

**Inputs**: Set of tokens $\textbf{X} \in \mathbb{R}^{N\times D}$, Weight matricies $\{ \textbf{W}^{(q)}, \textbf{W}^{(k)}\} \in \mathbb{R}^{D\times D_k}$ and $\textbf{W}^{(v)} \in \mathbb{R}^{D \times D_V}$

**Outputs**: $Attention(\textbf{Q}, \textbf{K}, \textbf{V}) \in \mathbb{R}^{N \times D_V}$

**Steps**: 

Compute queries:

$$
\textbf{Q} = \textbf{XW}^{(q)}
$$

Compute keys:

$$
\textbf{K} = \textbf{XW}^{(k)}
$$

Compute values:

$$
\textbf{V} = \textbf{XW}^{(v)}
$$

Return 

$$
Attention(\textbf{Q}, \textbf{K}, \textbf{V}) = Softmax\left[\frac{\textbf{QK}^T}{\sqrt{D_k}}\right]\textbf{V}
$$

Multi-head attention combines attention heads in parallel in a single layer with the addition of another weight matrix for converting the concatenated attention head outputs into a matrix with the same dimensionality as the inputs, enabling composability.
To imporve training efficiency, residual connections and layer normalization are often added. 
To enchance the flexibility of the training process (so that outputs aren't just linear combinations of the inputs), the output of each layer is post-processed with a neural network which is applied to each of the output vectors, also including a residual connection.

This results in the following algorithm for a transformer layer (from algorithm 12.3 of Bishop):

**Input**: Set of tokens $\textbf{X} \in \mathbb{R}^{(N \times D)}$, multi-head self-attention parameters, feed-forward network parameters

**Output**: $\tilde{\mathbf{X}} \in \mathbb{R}^{N \times D}$

**Steps**:

Compute multi-head self-attention with layer normalization and residual connections:

$$
Z = LayerNorm[\textbf{Y}(\textbf{X})+\textbf{X}],
$$

where $\textbf{Y}(\textbf{X})$ is the result of applying multi-head self-attention to $\textbf{X}$.

Apply shared neural network:

$$
\tilde{\textbf{X}} = LayerNorm[MLP[\textbf{Z}]+\textbf{Z}]
$$

Return $\tilde{\textbf{X}}$


Adding position embedding vectors to the input vectors enables the transformer to take into account the position of its inputs.

<!-- Add something on encoder, decoder, encoder-decoder architectures -->

## References

1. ["Deep Learning: Foundations and Concepts"](https://www.bishopbook.com) by Chris Bishop (2023).
2. [Stanford Machine Learning Courses Cheatsheets](https://stanford.edu/~shervine/teaching/cs-221/)
