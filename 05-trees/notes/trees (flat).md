# Boosting Algorithms

This document contains the complete algorithmic pseudocode and key formulas for various boosting and tree-based algorithms.

---

## Table of Contents

1. [Decision Tree (CART)](#1-decision-tree-cart)
2. [AdaBoost](#2-adaboost-adaptive-boosting)
3. [Gradient Boosting](#3-gradient-boosting)
4. [Key Formulas](#key-formulas)

---

## 1. Decision Tree (CART)

**Algorithm:** Build Decision Tree with Gini or Entropy

Input: $X \in \mathbb{R}^{n \times d}$, $y \in \{0, 1, \ldots, K-1\}^n$, criterion $c$, max depth $D$

Output: decision tree $T$

Impurity measures:
- Gini: $\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2$
- Entropy: $\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)$
- where $p_k = \dfrac{|S_k|}{|S|}$ is the proportion of class $k$ in set $S$

Information gain:
- $\text{Gain}(S, j, v) = \text{Impurity}(S) - \dfrac{|S_L|}{|S|}\text{Impurity}(S_L) - \dfrac{|S_R|}{|S|}\text{Impurity}(S_R)$
- where $S_L = \{i \in S : x_{ij} \leq v\}$ and $S_R = \{i \in S : x_{ij} > v\}$

Algorithm (BuildTree):
1. Function BuildTree($S$, $d$):
	1. If $d \geq D$, or $|S| <$ `min_split`, or all $y_i$ are the same: return a leaf with the majority class $\arg\max_k |\{i \in S : y_i = k\}|$
	2. Find the best split: $(j^*, v^*) = \arg\max_{j,v} \text{Gain}(S, j, v)$
	3. If there is no valid split, or the gain is $\leq 0$: return a leaf with the majority class
	4. Split the data: $S_L = \{i \in S : x_{ij^*} \leq v^*\}$, $S_R = \{i \in S : x_{ij^*} > v^*\}$
	5. Create the node:
		- node.feature = $j^*$
		- node.threshold = $v^*$
		- node.left = BuildTree($S_L$, $d+1$)
		- node.right = BuildTree($S_R$, $d+1$)
	6. Return the node
2. Return BuildTree($\{1, \ldots, n\}$, 0)

**Implementation:** [`core_cart.py`](../scripts/core_cart.py)

**Key Properties:**

- **Gini:** Fast to compute, range [0, 1-1/K], biased toward larger partitions
- **Entropy:** Slower (requires log), range [0, log₂(K)], more balanced splits
- **Both:** Produce similar trees in practice, Gini slightly faster

---

## 2. AdaBoost (Adaptive Boosting)

**Algorithm:** SAMME (Stagewise Additive Modeling using Multi-class Exponential loss)

Input: $X \in \mathbb{R}^{n \times d}$, $y \in \{0, 1, \ldots, K-1\}^n$, weak learner $h$, rounds $M$

Output: strong classifier $H(x)$

Algorithm:
1. Initialize the sample weights: $w_i^{(1)} = \dfrac{1}{n}$ for all $i \in \{1, \ldots, n\}$
2. For $m = 1$ to $M$:
	1. Train a weak learner on the weighted samples: $h_m = \arg\min_h \sum_{i=1}^{n} w_i^{(m)} \cdot \mathbb{I}(h(x_i) \neq y_i)$
	2. Compute the weighted error: $\epsilon_m = \dfrac{\sum_{i=1}^{n} w_i^{(m)} \cdot \mathbb{I}(h_m(x_i) \neq y_i)}{\sum_{i=1}^{n} w_i^{(m)}}$
	3. If $\epsilon_m \geq 1 - \dfrac{1}{K}$: stop (worse than random)
	4. Compute the classifier weight: $\alpha_m = \log\left(\dfrac{1 - \epsilon_m}{\epsilon_m}\right) + \log(K - 1)$
	5. Update the sample weights for all $i$: $w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left(\alpha_m \cdot \mathbb{I}(h_m(x_i) \neq y_i)\right)$
	6. Normalize the weights for all $i$: $w_i^{(m+1)} \leftarrow \dfrac{w_i^{(m+1)}}{\sum_{j=1}^{n} w_j^{(m+1)}}$
3. Final classifier (weighted majority vote): $H(x) = \arg\max_{k \in \{0,\ldots,K-1\}} \sum_{m=1}^{M} \alpha_m \cdot \mathbb{I}(h_m(x) = k)$

**Implementation:** [`core_adaboost.py`](../scripts/core_adaboost.py)

**Key Insights:**

- **Weight updates:** Misclassified samples get higher weights → next learner focuses on them
- **Exponential loss:** Weights grow exponentially with number of misclassifications
- **Weak learners:** Even slightly better than random (>50% accuracy) combine to strong classifier
- **SAMME:** Multi-class extension of AdaBoost, reduces to AdaBoost.M1 for binary case

---

## 3. Gradient Boosting

**Algorithm:** Gradient Boosting for Classification

Input: $X \in \mathbb{R}^{n \times d}$, $y \in \{0, 1, \ldots, K-1\}^n$, loss $L$, rounds $M$, learning rate $\eta$

Output: ensemble model $F(x)$

Loss functions:
- Binary (logistic): $L(y, f) = \log(1 + e^{-yf})$, with $y \in \{-1, +1\}$
- Multi-class (softmax): $L(y, \mathbf{f}) = -\sum_{k=1}^{K} y_k \log(p_k)$, with $p_k = \dfrac{e^{f_k}}{\sum_{j=1}^{K} e^{f_j}}$

Algorithm (binary classification):
1. Initialize with a constant (the log-odds): $F_0(x) = \arg\min_c \sum_{i=1}^{n} L(y_i, c) = \log\left(\dfrac{\sum \mathbb{I}(y_i=1)}{\sum \mathbb{I}(y_i=0)}\right)$
2. For $m = 1$ to $M$:
	1. Compute the negative gradients (pseudo-residuals) for all $i$:
		- $r_{im} = -\dfrac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F=F_{m-1}} = y_i - p_i^{(m-1)}$
		- where $p_i^{(m-1)} = \dfrac{1}{1 + e^{-F_{m-1}(x_i)}}$
	2. Fit a regression tree to the residuals: $h_m = \arg\min_h \sum_{i=1}^{n} (r_{im} - h(x_i))^2$
	3. Update the model: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$
3. Final model:
	- $F(x) = F_0 + \sum_{m=1}^{M} \eta \cdot h_m(x)$
	- $P(y=1|x) = \dfrac{1}{1 + e^{-F(x)}}$

**Implementation:** [`core_gbm.py`](../scripts/core_gbm.py)

**Key Insights:**

- **Gradient descent in function space:** Each tree approximates the negative gradient
- **More general than AdaBoost:** Works with any differentiable loss function
- **Learning rate:** Shrinkage parameter η prevents overfitting (typical: 0.01-0.3)
- **Residual fitting:** Trees predict what previous ensemble got wrong
- **Multi-class:** Train K trees per round (one per class) using softmax gradient

---

## Key Formulas

### Regularization Term

$$\Omega(f) = \gamma T + \dfrac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

where $T$ is the number of leaves and $w_j$ is the weight of leaf $j$.

---

### Gradient and Hessian for Different Loss Functions

#### Regression (Squared Error)

**Loss:** $l(y, \hat{y}) = (y - \hat{y})^2$

**Gradient:** $g = \hat{y} - y$

**Hessian:** $h = 1$

#### Binary Classification (Logistic Loss)

**Loss:** $l(y, \hat{y}) = -[y \log(p) + (1-y) \log(1-p)]$ where $p = \dfrac{1}{1 + e^{-\hat{y}}}$

**Gradient:** $g = p - y$

**Hessian:** $h = p(1 - p)$

---

## References

1. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
2. Burges et al. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research Technical Report.
3. Qin et al. (2010). *A General Approximation Framework for Direct Optimization of Information Retrieval Measures*. Information Retrieval Journal. (ApproxNDCG)
4. Cao et al. (2007). *Learning to Rank: From Pairwise Approach to Listwise Approach*. ICML. (ListNet)
5. [XGBoost Documentation](https://xgboost.readthedocs.io/)
