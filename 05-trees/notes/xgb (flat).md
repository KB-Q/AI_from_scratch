# XGBoost Algorithm

### Algorithm 4.1: Main XGBoost algorithm

Input: training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, loss function $l$, number of trees $K$, learning rate $\eta$

Output: ensemble model $\hat{y}(x)$

1. Initialize the base prediction (e.g. the mean for regression, the log-odds for classification): $\hat{y}_i^{(0)} = \arg\min_{\gamma} \sum_{i=1}^n l(y_i, \gamma)$
2. For $t = 1$ to $K$:
	1. Compute gradients and Hessians for all $i$:
		- $g_i^{(t)} = \dfrac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$
		- $h_i^{(t)} = \dfrac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$
	2. Subsample the data (optional): $\mathcal{D}_t \sim \text{Sample}(\mathcal{D}, \text{subsample ratio})$
	3. Build tree $f_t$ by calling BuildTree($\mathcal{D}_t$, $\{g_i^{(t)}\}$, $\{h_i^{(t)}\}$, 0)
	4. Update the predictions for all $i$: $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$
3. Return the final model: $\hat{y}(x) = \hat{y}^{(0)} + \sum_{t=1}^K \eta \cdot f_t(x)$

**Implementation:** [`XGBoost.fit()`](../scripts/core_xgb.py#L63-L222)

---

### Algorithm 4.2: BuildTree (Recursive Tree Construction)

Input: instance set $I$, gradients $\{g_i\}_{i \in I}$, Hessians $\{h_i\}_{i \in I}$, depth $d$

Output: tree node

1. If a stopping condition is met ($d \geq$ `max_depth` or $|I| = 0$): return a leaf with weight $w = -\dfrac{G}{H + \lambda}$, where $G = \sum_{i \in I} g_i$ and $H = \sum_{i \in I} h_i$
2. Find the best split with FindBestSplit($I$, $\{g_i\}$, $\{h_i\}$), which returns $(j^*, v^*, \text{Gain}^*)$
3. If no valid split is found ($\text{Gain}^* \leq 0$): return a leaf with weight $w = -\dfrac{G}{H + \lambda}$
4. Partition the instances: $I_L = \{i \in I : x_{ij^*} \leq v^*\}$, $I_R = \{i \in I : x_{ij^*} > v^*\}$
5. Create an internal node with:
	- feature = $j^*$, threshold = $v^*$
	- left = BuildTree($I_L$, $\{g_i\}_{i \in I_L}$, $\{h_i\}_{i \in I_L}$, $d+1$)
	- right = BuildTree($I_R$, $\{g_i\}_{i \in I_R}$, $\{h_i\}_{i \in I_R}$, $d+1$)
6. Return the internal node

**Implementation:** [`XGBoostTree._build_tree()`](../scripts/core_xgb_tree.py#L163-L204)

---

### Algorithm 4.3: FindBestSplit (Exact Greedy Algorithm)

Input: instance set $I$, gradients $\{g_i\}_{i \in I}$, Hessians $\{h_i\}_{i \in I}$

Output: best split $(j^*, v^*, \text{Gain}^*)$

1. Initialize: $\text{Gain}^* \leftarrow -\infty$, $j^* \leftarrow$ null, $v^* \leftarrow$ null
2. Compute the parent statistics: $G = \sum_{i \in I} g_i$, $H = \sum_{i \in I} h_i$
3. For each feature $j \in \{1, 2, \ldots, d\}$:
	1. Sort the instances by feature $j$: $I_{\text{sorted}} = \text{sort}(I, \text{by } x_{ij})$
	2. For each split candidate $v$ (between consecutive unique values):
		1. Partition: $I_L = \{i \in I : x_{ij} \leq v\}$, $I_R = \{i \in I : x_{ij} > v\}$
		2. Compute the statistics:
			- $G_L = \sum_{i \in I_L} g_i$, $H_L = \sum_{i \in I_L} h_i$
			- $G_R = \sum_{i \in I_R} g_i$, $H_R = \sum_{i \in I_R} h_i$
		3. Check the constraints: if $H_L <$ `min_child_weight` or $H_R <$ `min_child_weight`, continue
		4. Calculate the gain (Equation 7): $\text{Gain} = \dfrac{1}{2}\left[\dfrac{G_L^2}{H_L + \lambda} + \dfrac{G_R^2}{H_R + \lambda} - \dfrac{G^2}{H + \lambda}\right] - \gamma$
		5. If $\text{Gain} > \text{Gain}^*$: $\text{Gain}^* \leftarrow \text{Gain}$, $j^* \leftarrow j$, $v^* \leftarrow v$
4. Return $(j^*, v^*, \text{Gain}^*)$

**Implementation:** [`XGBoostTree._find_best_split()`](../scripts/core_xgb_tree.py#L113-L162)

---

## Key Formulas

### Optimal Leaf Weight (Equation 5)

$$w_j^* = -\dfrac{G_j}{H_j + \lambda}$$

where $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$ for leaf $j$.

**Implementation:** [`_calculate_leaf_weight()`](../scripts/core_xgb_tree.py#L61-L75)

---

### Split Gain (Equation 7)

$$\text{Gain} = \dfrac{1}{2}\left[\dfrac{G_L^2}{H_L + \lambda} + \dfrac{G_R^2}{H_R + \lambda} - \dfrac{G^2}{H + \lambda}\right] - \gamma$$

**Implementation:** [`_calculate_split_gain()`](../scripts/core_xgb_tree.py#L91-L112)

---
