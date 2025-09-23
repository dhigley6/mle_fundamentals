# Machine Learning

A quick review of non-neural network Machine Learning (ML) algorithms. This review seeks to go over most of the most common/useful non-neural network-based machine learning problem types and algorithms. This review does not discuss in depth how different algorithms are fit, focusing instead on their other aspects. This review describes things succinctly without a focus on pedagogy. For going through the material with a more pedagogic focus, I have found the following resources useful (and used some of these in constructing this review:
1. "An Introduction to Statistical Learning". This is a quick read particularly great for a first exposure to machine learning
2. "CS229 at Stanford Lecture Notes"
3. "Probabilistic Machine Learning: An Introduction" by Kevin P. Murphy.
4. Scikit-Learn User Documentation

## What is Machine Learning

A popular definition of machine learning due to Tom Mitchell is "A computer program is said to learn from experience E with respect to some class of tasks T, and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

## Types of Machine Learning

### Supervised

The most common form of ML in industry is supervised learning. The task is to learn a mapping $f$ from inputs $\textbf{x}$ to outputs $\textbf{y}$. The inputs $x$ are also called the features, covariates or predictors. The output $y$ is also known as the label, target or response. The expericence is given in the form of a set of $N$ input-output pairs, $`D = \{(\textbf{x}_n, \textbf{y}_n)\}_{n=1}^N`$. The performance measure depends on the type of output.

#### Classification

In classification, the problem is to predict one of a set of mutually exclusive labels known as classes.

#### Regression

#### Learning to Rank

Learning to Rank reference: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

#### Methods for Dealing with Limited Labels

##### Transfer Learning

##### Active Learning

##### Semi-Supervised Learning

### Unsupervised Learning

#### Clustering

#### Dimensionality Reduction

##### Principal Component Analysis

Principal Component Analysis (PCA) reduces the dimensionality of data by linearly transforming the data into a new coordinate system such that the number of dimensions (principal components) retained in the new coordinate system maximizes the captured variation in the data.

The principal components are the eigenvectors of the empirical sample covariance matrix and the right singular vectors of the Singular Value Decomposition (SVD) of the data matrix.

One can apply PCA to data with the following procedure:

1. (where it makes sense) standardize the data
2. Perform SVD and from that extract the principal components
3. Sort eigenvalues in descending order and extract the top k corresponding principal components
4. Construct the projection matrix $W$ from the extracted eigenvectors
5. Transform the original dataset $X$ with $W$ to obtain the reduced dimensionality data

The number of principal components to use is often chosen with the proportion of explained variance, which is the proportion of variance in the data that can be explained with a selected number of principal components.

<!-- Reference: https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html -->

#### Novelty and Outlier Detection

#### Manifold Learning

### Reinforcement Learning

## Supervised Learning Algorithms

### Linear Regression

We have a collection of $N$ labeled examples,
$`\{ (\textbf{x}_i, y_i) \} _{i=1}^N`$.
where $\textbf{x}_i$ is a feature vector of example $i$, $y_i$ is a real-valued target. We want to build a model as a linear combination of features of example $x$,
$`f_{\textbf{w}, b}(\textbf{x}) = \textbf{w}\textbf{x} + b `$, where $w$ is a vector of parameters and $b$ is a real number.

### Generalized Linear Models

### Logistic Regression

### Decision Tree

Given training vectors $` x_i \in R^n `$, and label vector $y \in R^l$, a decision tree recursively partitions the feature space such that the samples with the same labels or similar target values are grouped together.

Let the data at node $m$ be represented by $Q_m$ with $n_m$ samples. For each candidate split $\theta = (j, t_m)$ consisting of a feature $j$ and threshold $t_m$, partition the data into $Q_m^{left}(\theta)$ and $Q_m^{right}(\theta)$ subsets. 

The quality of a candidate split of node $m$ is then computed using an impurity or loss function, $H()$.

$` G(Q_m, \theta) = \frac{n_m^{left}}{n_m}H(Q_m^{left}(\theta)) + \frac{n_m^{right}}{n_m}H(Q_m^{right}(\theta)) `$

Select the parameters that minimize the impurity:

$` \theta^* = \textrm{argmin}_{\theta}G(Q_m, \theta) `$

Common classification criteria include the Gini impurity

$` H(Q_m) = \sum_k p_{mk}(1-p_{mk}) `$

and the Log Loss or Entropy

$` H(Q_m) = -\sum_k p_{mk}\log(p_{mk}) `$.

Reference: https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation

### Naive Bayes

Set of classification algorithms with the "naive" assumption of conditional independence of pairs of feature values given the class value. For a class value of $y$ and feature values $x_1$, $x_2$, ..., $x_n$, this assumption gives

$P(x_i| y, x_1, x_2, ..., x{i-1}, x_{i+1}, ..., x_n) = P(x_i|y).$

From Bayes' Theorem, we have,

$P(y|x_1, x_2, ..., x_n) = \frac{P(y)P(x_1, x_2, ..., x_n | y)}{P(x_1, x_2, ..., x_n)}.$

Using the "naive" conditional independence assumption, we get

$P(y|x_1, x_2, ..., x_n) = \frac{P(y)\prod_{i=1}^n P(x_i|y)}{P(x_1, x_2, ..., x_n)}.$

Or

$P(y|x_1, x_2, ..., x_n) \propto P(y)\prod_{i=1}^n P(x_i|y).$

We can apply Maximum A Posteriori estimation $P(y)$ and $P(x_i|y)$. The former is then just the relative frequency of class $y$ in the training set. The different Naive Bayes classifiers differ mainly the assumed form of the distribution $P(x_i|y)$.

### Support Vector Machines

Good reference on kernel trick and svm: https://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes3.pdf

### Linear and Quadratic Discriminant Analysis

$P(y=k | x) = \frac{P(x|y = k)P(y=k)}{P(x)} = \frac{P(x|y=k)P(y=k)}{\sum_l P(x|y=l)P(y=l)}$

For Linear and Quadratic Discriminant Analysis, we model $P(y=k|x)$ as a multivariate Gaussian distribution with density

$P(x|y=k) = \frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}\exp(-\frac{1}{2}(x-\mu_k)^t\Sigma_k^{-1}(x-\mu_k))$,

where $d$ is the number of features.

#### QDA

According...

### Gaussian Process

Good overview of Gaussian Process regression/classification: https://distill.pub/2019/visual-exploration-gaussian-processes/

### kNN

### Ensemble Methods

#### Bagging

#### Random Forest

#### Gradient Boosting

<!-- Boosting reference: https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/ -->
<!-- Reference for algorithm: https://en.wikipedia.org/wiki/Gradient_boosting -->

Boosting combines weak learners (models with high bias which only weakly correlate with complex patterns to be learned) to form strong learners (models which are arbitrarily well-correlated with patterns to be learned). Gradient boosting is a particular type of boosting which allows optimization of an arbitrary differentiable loss function through a gradient descent-like procedure. Gradient boosting combines weak learners into a single strong learner iterartively.

Formally, we have an input training set $` \{(x_i, y_i)\}_{i=1}^n `$, a differentiable loss function $` L(y, F(x)) `$ and a number of iterations $` M `$. We then apply the following algorithm:
1. Initialize model with a constant value, $` F_0(x) = \textrm{argmin}_{\gamma} \sum_{i=1}^n L(y_i, \gamma) `$,
2. For $` m=1 `$ to $` M `$,
   - Compute the psuedo-residuals $` r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} `$ for $` i=1,...,n `$.
   - Fit a baselearner to the pseudo-residuals (the training set $` \{(x_i, r_{im})\}_{i=1}^n `$
   - Compute multiplier $` \gamma_m `$ by solivng the following one-dimensional optimization problem $` \gamma_m = \textrm{argmin}_{\gamma} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)) `$.
   - Update the model $` F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) `$.
3. Output the model $` F_m(x) `$

#### Stacking

<!-- Reference: https://www.geeksforgeeks.org/machine-learning/stacking-in-machine-learning/ -->
<!-- Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html -->

In stacking, a final model combines the predictions of multiple base models. The final model, such as logistic regression is trained on the predictions of the base models, enabling improved accuracy over any of the base models in isolation.

## Practical Techniques

### Extending Binary Classification to Multiclass Classification

Some binary classifiers do not have as natural of extensions to multiclass classification. There are some general approaches we can use in such cases:

#### One-Versus-One Classification

Fit a binary classifier for each pair of classes. Classify an observation as the most commonly predicted class among all the binary classifier predictions.

#### One-Versus-Rest Classification

For each class, fit a binary classifier which predicts whether an observation belongs to that class or not. Classify an observation as the class with the highest predicted probability among all the one-versus-rest classifiers.

### Overview of First-Order Optimization Methods

TODO: Probably remove this (out of scope)

https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c
