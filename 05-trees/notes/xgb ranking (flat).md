# Boosting Algorithms - Ranking

### Algorithm 1.1: LambdaMART (Learning to Rank)

Input: relevance labels $\{y_i\}_{i=1}^n$, predictions $\{\hat{y}_i\}_{i=1}^n$, query IDs $\{q_i\}_{i=1}^n$

Output: lambda gradients $\{g_i\}_{i=1}^n$ and Hessians $\{h_i\}_{i=1}^n$

1. Initialize: $g_i \leftarrow 0$, $h_i \leftarrow 0$ for all $i$
2. For each unique query $q$:
	1. Get the documents in the query: $I_q = \{i : q_i = q\}$
	2. If $|I_q| \leq 1$: continue
	3. Sort the documents by prediction: $\pi = \text{argsort}(-\{\hat{y}_i\}_{i \in I_q})$
	4. Compute the ideal DCG, where $y_{\text{sorted}} = \text{sort}(\{y_i\}_{i \in I_q}, \text{descending})$: $\text{IDCG}_q = \sum_{k=1}^{|I_q|} \frac{2^{y_{\text{sorted}[k]}} - 1}{\log_2(k + 1)}$
	5. If $\text{IDCG}_q = 0$: continue
	6. For each pair of documents $(i, j) \in I_q \times I_q$:
		1. If $i = j$ or $y_i = y_j$: continue
		2. Get the current ranks: $r_i = \pi^{-1}(i)$, $r_j = \pi^{-1}(j)$
		3. Compute the gain and discount at each position:
			- $G_i = 2^{y_i} - 1$, $G_j = 2^{y_j} - 1$
			- $D_i = \frac{1}{\log_2(r_i + 2)}$, $D_j = \frac{1}{\log_2(r_j + 2)}$
		4. Compute the change in NDCG if $i$ and $j$ are swapped:
			- $\Delta\text{DCG}_{ij} = (G_i - G_j) \cdot (D_i - D_j)$
			- $\Delta\text{NDCG}_{ij} = \frac{\Delta\text{DCG}_{ij}}{\text{IDCG}_q}$
		5. Compute the sigmoid of the score difference: $\sigma_{ij} = \frac{1}{1 + e^{-(\hat{y}_i - \hat{y}_j)}}$
		6. Compute the lambda value:
			- if $y_i > y_j$: $\lambda_{ij} = \sigma_{ij} \cdot |\Delta\text{NDCG}_{ij}|$
			- if $y_i < y_j$: $\lambda_{ij} = -\sigma_{ij} \cdot |\Delta\text{NDCG}_{ij}|$
		7. Accumulate the gradients and Hessians:
			- $g_i \leftarrow g_i + \lambda_{ij}$
			- $h_i \leftarrow h_i + \sigma_{ij} \cdot (1 - \sigma_{ij}) \cdot |\Delta\text{NDCG}_{ij}|$
3. Ensure positive Hessians: $h_i \leftarrow \max(h_i, 10^{-16})$ for all $i$
4. Return $\{g_i\}_{i=1}^n$, $\{h_i\}_{i=1}^n$

**Implementation:** [`ObjectiveFunctions.compute_lambda_gradients()`](../metrics.py#L171-L233)

**Key Insight:** LambdaMART computes gradients based on **pairwise comparisons** within each query. The gradient for document $i$ depends on how swapping it with other documents would affect the ranking metric (NDCG). Documents with higher relevance should be ranked higher, and the gradients push the model in that direction.

**Ranking Approach:** Pairwise - considers all O(n²) document pairs within each query.

---

### Algorithm 1.2: Listwise LambdaMART (ApproxNDCG)

Input: relevance labels $\{y_i\}_{i=1}^n$, predictions $\{\hat{y}_i\}_{i=1}^n$, query IDs $\{q_i\}_{i=1}^n$, temperature $\tau$

Output: lambda gradients $\{g_i\}_{i=1}^n$ and Hessians $\{h_i\}_{i=1}^n$

1. Initialize: $g_i \leftarrow 0$, $h_i \leftarrow 0$ for all $i$
2. For each unique query $q$:
	1. Get the documents in the query: $I_q = \{i : q_i = q\}$, $|I_q| = m$
	2. If $m \leq 1$: continue
	3. Compute a soft rank for each document $i \in I_q$ using a sigmoid approximation:
		- $s_i = \sum_{j \in I_q, j \neq i} \sigma\left(\frac{\hat{y}_j - \hat{y}_i}{\tau}\right) + 1$
		- where $\sigma(x) = \frac{1}{1 + e^{-x}}$ (smooth rank approximation)
	4. Compute the gain and discount at the soft ranks, for each document $i \in I_q$:
		- $G_i = 2^{y_i} - 1$
		- $D_i = \frac{1}{\log_2(s_i + 1)}$
	5. Compute the smooth DCG: $\text{DCG}_q = \sum_{i \in I_q} G_i \cdot D_i$
	6. Compute the ideal DCG, where $y_{\text{sorted}} = \text{sort}(\{y_i\}_{i \in I_q}, \text{descending})$: $\text{IDCG}_q = \sum_{k=1}^{m} \frac{2^{y_{\text{sorted}[k]}} - 1}{\log_2(k + 1)}$
	7. If $\text{IDCG}_q = 0$: continue
	8. Compute the smooth NDCG: $\text{NDCG}_q = \frac{\text{DCG}_q}{\text{IDCG}_q}$
	9. Compute the gradient for each document $i \in I_q$ (chain rule):
		- $\frac{\partial \text{NDCG}_q}{\partial \hat{y}_i} = \frac{1}{\text{IDCG}_q} \sum_{j \in I_q} G_j \cdot \frac{\partial D_j}{\partial s_j} \cdot \frac{\partial s_j}{\partial \hat{y}_i}$
		- where $\frac{\partial D_j}{\partial s_j} = -\frac{1}{\ln(2) \cdot (s_j + 1) \cdot \ln^2(s_j + 1)}$
		- and, if $i = j$: $\frac{\partial s_j}{\partial \hat{y}_i} = -\sum_{k \in I_q, k \neq j} \frac{\sigma_k}{\tau} \cdot (1 - \sigma_k)$
		- and, if $i \neq j$: $\frac{\partial s_j}{\partial \hat{y}_i} = \frac{\sigma_i}{\tau} \cdot (1 - \sigma_i)$
		- where $\sigma_k = \sigma\left(\frac{\hat{y}_k - \hat{y}_j}{\tau}\right)$
		- $g_i \leftarrow -\frac{\partial \text{NDCG}_q}{\partial \hat{y}_i}$ (negative gradient for minimization)
	10. Compute a Hessian approximation (diagonal, kept positive) for each document $i \in I_q$: $h_i \leftarrow \frac{1}{\text{IDCG}_q} \sum_{j \in I_q} |G_j| \cdot \frac{\sigma_j}{\tau^2} \cdot (1 - \sigma_j)$
3. Ensure positive Hessians: $h_i \leftarrow \max(h_i, 10^{-16})$ for all $i$
4. Return $\{g_i\}_{i=1}^n$, $\{h_i\}_{i=1}^n$

### Algorithm 1.3: Soft Ranking Function

The key innovation in listwise ranking is the **differentiable soft rank** approximation:

$$\text{SoftRank}(i) = 1 + \sum_{j \neq i} \sigma\left(\frac{\hat{y}_j - \hat{y}_i}{\tau}\right)$$

This smoothly approximates the discrete rank by counting how many documents have higher scores. As $\tau \to 0$, this converges to the true rank.

**Key Differences from Pairwise LambdaMART:**

1. **Gradient computation:** Considers the entire ranking list at once via soft ranks
2. **Complexity:** O(m²) per query for soft rank computation, but single pass (not nested loops over pairs)
3. **NDCG optimization:** Directly differentiable through soft ranks, rather than pairwise swap deltas
4. **Temperature parameter** $\tau$: Controls smoothness of rank approximation
   - Small $\tau$ → closer to true ranks, sharper gradients
   - Large $\tau$ → smoother gradients, more stable training
   - Typical: $\tau \in [0.1, 1.0]$

**Advantages of Listwise Approach:**

- Directly optimizes the ranking metric (NDCG) as a differentiable function
- Captures list-level interactions rather than just pairwise preferences
- Often more stable gradients due to smooth approximations
- Better alignment with evaluation metrics

**Disadvantages:**

- Computationally more expensive per query (O(m²) gradient computation)
- Requires careful tuning of temperature parameter
- Hessian approximation less accurate (diagonal only)

**Ranking Approach:** Listwise - considers the entire document list as a unit using soft rank approximations.

---

### NDCG (Normalized Discounted Cumulative Gain)

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

where:

- $\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$ (Discounted Cumulative Gain)
- $\text{IDCG@}k$ is the ideal DCG (DCG of perfect ranking by relevance)
- $\text{rel}_i$ is the relevance label of the document at position $i$

**Implementation:** [`ndcg()`](../metrics.py#L24-L39)

---
