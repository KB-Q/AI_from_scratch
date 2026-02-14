# Boosting Algorithms

This document contains the complete algorithmic pseudocode and key formulas for various boosting and tree-based algorithms.

---

## Table of Contents

### Core Tree Algorithms

1. [Decision Tree (CART)](#decision-tree-cart)
1. [AdaBoost](#adaboost-adaptive-boosting)
1. [Gradient Boosting](#gradient-boosting)
1. [XGBoost Training](#algorithm-4-xgboost-training-gradient-tree-boosting)
1. [Key Formulas](#key-formulas)

---

## Decision Tree (CART)

### Algorithm: Build Decision Tree with Gini or Entropy

$$
\begin{align*}
&\textbf{Input:} X \in \mathbb{R}^{n \times d}, y \in \{0, 1, \ldots, K-1\}^n, \text{ criterion } c, \text{ max depth } D \\
&\textbf{Output:} \text{Decision tree } T \\
\\
&\textbf{Impurity Measures:} \\
&\quad \text{Gini: } \text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2 \\
&\quad \text{Entropy: } \text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2(p_k) \\
&\quad \text{where } p_k = \frac{|S_k|}{|S|} \text{ (proportion of class } k \text{ in set } S\text{)} \\
\\
&\textbf{Information Gain:} \\
&\quad \text{Gain}(S, j, v) = \text{Impurity}(S) - \frac{|S_L|}{|S|}\text{Impurity}(S_L) - \frac{|S_R|}{|S|}\text{Impurity}(S_R) \\
&\quad \text{where } S_L = \{i \in S : x_{ij} \leq v\}, \quad S_R = \{i \in S : x_{ij} > v\} \\
\\
&\textbf{Algorithm (BuildTree):} \\
&\quad \text{1. Function } \textbf{BuildTree}(S, d): \\
&\quad\quad \text{a) If } d \geq D \text{ or } |S| < \text{min\_split} \text{ or all } y_i \text{ same:} \\
&\quad\quad\quad \text{Return Leaf with majority class: } \arg\max_k |\{i \in S : y_i = k\}| \\
\\
&\quad\quad \text{b) Find best split:} \\
&\quad\quad\quad (j^*, v^*) = \arg\max_{j,v} \text{Gain}(S, j, v) \\
\\
&\quad\quad \text{c) If no valid split or Gain} \leq 0: \\
&\quad\quad\quad \text{Return Leaf with majority class} \\
\\
&\quad\quad \text{d) Split data:} \\
&\quad\quad\quad S_L = \{i \in S : x_{ij^*} \leq v^*\}, \quad S_R = \{i \in S : x_{ij^*} > v^*\} \\
\\
&\quad\quad \text{e) Create node:} \\
&\quad\quad\quad \text{node.feature} = j^* \\
&\quad\quad\quad \text{node.threshold} = v^* \\
&\quad\quad\quad \text{node.left} = \textbf{BuildTree}(S_L, d+1) \\
&\quad\quad\quad \text{node.right} = \textbf{BuildTree}(S_R, d+1) \\
\\
&\quad\quad \text{f) Return node} \\
\\
&\quad \text{2. Return } \textbf{BuildTree}(\{1, \ldots, n\}, 0)
\end{align*}
$$

**Implementation:** [`decision_tree.py`](decision_tree.py)

**Key Properties:**

- **Gini:** Fast to compute, range [0, 1-1/K], biased toward larger partitions
- **Entropy:** Slower (requires log), range [0, log₂(K)], more balanced splits
- **Both:** Produce similar trees in practice, Gini slightly faster

---

## AdaBoost (Adaptive Boosting)

### Algorithm: SAMME (Stagewise Additive Modeling using Multi-class Exponential loss)

$$
\begin{align*}
&\textbf{Input:} X \in \mathbb{R}^{n \times d}, y \in \{0, 1, \ldots, K-1\}^n, \text{ weak learner } h, \text{ rounds } M \\
&\textbf{Output:} \text{Strong classifier } H(x) \\
\\
&\textbf{Algorithm:} \\
&\quad \text{1. Initialize sample weights: } w_i^{(1)} = \frac{1}{n} \quad \forall i \in \{1, \ldots, n\} \\
\\
&\quad \text{2. For } m = 1 \text{ to } M: \\
\\
&\quad\quad \text{a) Train weak learner on weighted samples:} \\
&\quad\quad\quad h_m = \arg\min_h \sum_{i=1}^{n} w_i^{(m)} \cdot \mathbb{I}(h(x_i) \neq y_i) \\
\\
&\quad\quad \text{b) Compute weighted error:} \\
&\quad\quad\quad \epsilon_m = \frac{\sum_{i=1}^{n} w_i^{(m)} \cdot \mathbb{I}(h_m(x_i) \neq y_i)}{\sum_{i=1}^{n} w_i^{(m)}} \\
\\
&\quad\quad \text{c) If } \epsilon_m \geq 1 - \frac{1}{K}: \text{ stop (worse than random)} \\
\\
&\quad\quad \text{d) Compute classifier weight:} \\
&\quad\quad\quad \alpha_m = \log\left(\frac{1 - \epsilon_m}{\epsilon_m}\right) + \log(K - 1) \\
\\
&\quad\quad \text{e) Update sample weights:} \\
&\quad\quad\quad w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left(\alpha_m \cdot \mathbb{I}(h_m(x_i) \neq y_i)\right) \quad \forall i \\
\\
&\quad\quad \text{f) Normalize weights:} \\
&\quad\quad\quad w_i^{(m+1)} \leftarrow \frac{w_i^{(m+1)}}{\sum_{j=1}^{n} w_j^{(m+1)}} \quad \forall i \\
\\
&\quad \text{3. Final classifier (weighted majority vote):} \\
&\quad\quad H(x) = \arg\max_{k \in \{0,\ldots,K-1\}} \sum_{m=1}^{M} \alpha_m \cdot \mathbb{I}(h_m(x) = k)
\end{align*}
$$

**Implementation:** [`adaboost.py`](adaboost.py)

**Key Insights:**

- **Weight updates:** Misclassified samples get higher weights → next learner focuses on them
- **Exponential loss:** Weights grow exponentially with number of misclassifications
- **Weak learners:** Even slightly better than random (>50% accuracy) combine to strong classifier
- **SAMME:** Multi-class extension of AdaBoost, reduces to AdaBoost.M1 for binary case

---

## Gradient Boosting

### Algorithm: Gradient Boosting for Classification

$$
\begin{align*}
&\textbf{Input:} X \in \mathbb{R}^{n \times d}, y \in \{0, 1, \ldots, K-1\}^n, \text{ loss } L, \text{ rounds } M, \text{ learning rate } \eta \\
&\textbf{Output:} \text{Ensemble model } F(x) \\
\\
&\textbf{Loss Functions:} \\
&\quad \text{Binary (Logistic): } L(y, f) = \log(1 + e^{-yf}), \quad y \in \{-1, +1\} \\
&\quad \text{Multi-class (Softmax): } L(y, \mathbf{f}) = -\sum_{k=1}^{K} y_k \log(p_k), \quad p_k = \frac{e^{f_k}}{\sum_{j=1}^{K} e^{f_j}} \\
\\
&\textbf{Algorithm (Binary Classification):} \\
&\quad \text{1. Initialize with constant:} \\
&\quad\quad F_0(x) = \arg\min_c \sum_{i=1}^{n} L(y_i, c) = \log\left(\frac{\sum \mathbb{I}(y_i=1)}{\sum \mathbb{I}(y_i=0)}\right) \quad \text{(log-odds)} \\
\\
&\quad \text{2. For } m = 1 \text{ to } M: \\
\\
&\quad\quad \text{a) Compute negative gradients (pseudo-residuals):} \\
&\quad\quad\quad r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F=F_{m-1}} = y_i - p_i^{(m-1)} \quad \forall i \\
&\quad\quad\quad \text{where } p_i^{(m-1)} = \frac{1}{1 + e^{-F_{m-1}(x_i)}} \\
\\
&\quad\quad \text{b) Fit regression tree to residuals:} \\
&\quad\quad\quad h_m = \arg\min_h \sum_{i=1}^{n} (r_{im} - h(x_i))^2 \\
\\
&\quad\quad \text{c) Update model:} \\
&\quad\quad\quad F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x) \\
\\
&\quad \text{3. Final model:} \\
&\quad\quad F(x) = F_0 + \sum_{m=1}^{M} \eta \cdot h_m(x) \\
&\quad\quad P(y=1|x) = \frac{1}{1 + e^{-F(x)}}
\end{align*}
$$

**Implementation:** [`gradient_boosting.py`](gradient_boosting.py)

**Key Insights:**

- **Gradient descent in function space:** Each tree approximates the negative gradient
- **More general than AdaBoost:** Works with any differentiable loss function
- **Learning rate:** Shrinkage parameter η prevents overfitting (typical: 0.01-0.3)
- **Residual fitting:** Trees predict what previous ensemble got wrong
- **Multi-class:** Train K trees per round (one per class) using softmax gradient

---

## Algorithm 4: XGBoost Training (Gradient Tree Boosting)

### Algorithm 4.1: Main XGBoost algorithm

$$
\begin{align*}
&\textbf{Input:} \text{Training data } \mathcal{D} = \{(x_i, y_i)\}_{i=1}^n, \text{ loss function } l, \text{ number of trees } K, \text{ learning rate } \eta \\
&\textbf{Output:} \text{Ensemble model } \hat{y}(x) \\
\\
&\text{1. Initialize base prediction:} \\
&\quad \hat{y}_i^{(0)} = \arg\min_{\gamma} \sum_{i=1}^n l(y_i, \gamma) \quad \text{(e.g., mean for regression, log-odds for classification)} \\
\\
&\text{2. For } t = 1 \text{ to } K: \\
&\quad \text{a) Compute gradients and Hessians:} \\
&\quad\quad g_i^{(t)} = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i^{(t)} = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2} \quad \forall i \\
\\
&\quad \text{b) Subsample data (optional):} \\
&\quad\quad \mathcal{D}_t \sim \text{Sample}(\mathcal{D}, \text{subsample ratio}) \\
\\
&\quad \text{c) Build tree } f_t \text{ by calling } \textbf{BuildTree}(\mathcal{D}_t, \{g_i^{(t)}\}, \{h_i^{(t)}\}, 0) \\
\\
&\quad \text{d) Update predictions:} \\
&\quad\quad \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i) \quad \forall i \\
\\
&\text{3. Return final model: } \hat{y}(x) = \hat{y}^{(0)} + \sum_{t=1}^K \eta \cdot f_t(x)
\end{align*}
$$

**Implementation:** [`XGBoost.fit()`](xgboost_manual.py#L476-L560)

---

### Algorithm 4.2: BuildTree (Recursive Tree Construction)

$$
\begin{align*}
&\textbf{Input:} \text{Instance set } I, \text{ gradients } \{g_i\}_{i \in I}, \text{ Hessians } \{h_i\}_{i \in I}, \text{ depth } d \\
&\textbf{Output:} \text{Tree node} \\
\\
&\text{1. If stopping condition met (} d \geq \text{max\_depth or } |I| = 0 \text{):} \\
&\quad \text{Return Leaf with weight } w = -\frac{G}{H + \lambda}, \text{ where } G = \sum_{i \in I} g_i, H = \sum_{i \in I} h_i \\
\\
&\text{2. Find best split using } \textbf{FindBestSplit}(I, \{g_i\}, \{h_i\}) \rightarrow (j^*, v^*, \text{Gain}^*) \\
\\
&\text{3. If no valid split found (Gain}^* \leq 0 \text{):} \\
&\quad \text{Return Leaf with weight } w = -\frac{G}{H + \lambda} \\
\\
&\text{4. Partition instances:} \\
&\quad I_L = \{i \in I : x_{ij^*} \leq v^*\}, \quad I_R = \{i \in I : x_{ij^*} > v^*\} \\
\\
&\text{5. Create internal node with:} \\
&\quad \text{feature = } j^*, \text{ threshold = } v^* \\
&\quad \text{left = } \textbf{BuildTree}(I_L, \{g_i\}_{i \in I_L}, \{h_i\}_{i \in I_L}, d+1) \\
&\quad \text{right = } \textbf{BuildTree}(I_R, \{g_i\}_{i \in I_R}, \{h_i\}_{i \in I_R}, d+1) \\
\\
&\text{6. Return internal node}
\end{align*}
$$

**Implementation:** [`XGBoostTree._build_tree()`](xgboost_manual.py#L274-L337)

---

### Algorithm 4.3: FindBestSplit (Exact Greedy Algorithm)

$$
\begin{align*}
&\textbf{Input:} \text{Instance set } I, \text{ gradients } \{g_i\}_{i \in I}, \text{ Hessians } \{h_i\}_{i \in I} \\
&\textbf{Output:} \text{Best split } (j^*, v^*, \text{Gain}^*) \\
\\
&\text{1. Initialize: Gain}^* \leftarrow -\infty, j^* \leftarrow \text{null}, v^* \leftarrow \text{null} \\
&\text{2. Compute parent statistics: } G = \sum_{i \in I} g_i, \quad H = \sum_{i \in I} h_i \\
\\
&\text{3. For each feature } j \in \{1, 2, \ldots, d\}: \\
&\quad \text{a) Sort instances by feature } j: I_{\text{sorted}} = \text{sort}(I, \text{by } x_{ij}) \\
\\
&\quad \text{b) For each split candidate } v \text{ (between consecutive unique values):} \\
&\quad\quad \text{i) Partition: } I_L = \{i \in I : x_{ij} \leq v\}, \quad I_R = \{i \in I : x_{ij} > v\} \\
\\
&\quad\quad \text{ii) Compute statistics:} \\
&\quad\quad\quad G_L = \sum_{i \in I_L} g_i, \quad H_L = \sum_{i \in I_L} h_i \\
&\quad\quad\quad G_R = \sum_{i \in I_R} g_i, \quad H_R = \sum_{i \in I_R} h_i \\
\\
&\quad\quad \text{iii) Check constraints:} \\
&\quad\quad\quad \text{If } H_L < \text{min\_child\_weight or } H_R < \text{min\_child\_weight: continue} \\
\\
&\quad\quad \text{iv) Calculate gain (Equation 7):} \\
&\quad\quad\quad \text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma \\
\\
&\quad\quad \text{v) If Gain} > \text{Gain}^*: \\
&\quad\quad\quad \text{Gain}^* \leftarrow \text{Gain}, \quad j^* \leftarrow j, \quad v^* \leftarrow v \\
\\
&\text{4. Return } (j^*, v^*, \text{Gain}^*)
\end{align*}
$$

**Implementation:** [`XGBoostTree._find_best_split()`](xgboost_manual.py#L202-L272)

---

## Key Formulas

### Optimal Leaf Weight (Equation 5)

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

where $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$ for leaf $j$.

**Implementation:** [`_calculate_leaf_weight()`](xgboost_manual.py#L88-L118)

---

### Split Gain (Equation 7)

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma
$$

**Implementation:** [`_calculate_split_gain()`](xgboost_manual.py#L162-L200)

---

### Regularization Term

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2
$$

where $T$ is the number of leaves and $w_j$ is the weight of leaf $j$.

---

### Gradient and Hessian for Different Loss Functions

#### Regression (Squared Error)

**Loss:** $l(y, \hat{y}) = (y - \hat{y})^2$

**Gradient:** $g = \hat{y} - y$

**Hessian:** $h = 1$

#### Binary Classification (Logistic Loss)

**Loss:** $l(y, \hat{y}) = -[y \log(p) + (1-y) \log(1-p)]$ where $p = \frac{1}{1 + e^{-\hat{y}}}$

**Gradient:** $g = p - y$

**Hessian:** $h = p(1 - p)$

---

## References

1. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
2. Burges et al. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research Technical Report.
3. Qin et al. (2010). *A General Approximation Framework for Direct Optimization of Information Retrieval Measures*. Information Retrieval Journal. (ApproxNDCG)
4. Cao et al. (2007). *Learning to Rank: From Pairwise Approach to Listwise Approach*. ICML. (ListNet)
5. [XGBoost Documentation](https://xgboost.readthedocs.io/)