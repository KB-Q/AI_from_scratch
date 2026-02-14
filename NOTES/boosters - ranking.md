# Boosting Algorithms - Ranking

### Algorithm 1.1: LambdaMART (Learning to Rank)

$$
\begin{align*}
&\textbf{Input:} \text{Relevance labels } \{y_i\}_{i=1}^n, \text{ predictions } \{\hat{y}_i\}_{i=1}^n, \text{ query IDs } \{q_i\}_{i=1}^n \\
&\textbf{Output:} \text{Lambda gradients } \{g_i\}_{i=1}^n \text{ and Hessians } \{h_i\}_{i=1}^n \\
\\
&\text{1. Initialize: } g_i \leftarrow 0, \quad h_i \leftarrow 0 \quad \forall i \\
\\
&\text{2. For each unique query } q: \\
&\quad \text{a) Get documents in query: } I_q = \{i : q_i = q\} \\
&\quad \text{b) If } |I_q| \leq 1: \text{ continue} \\
\\
&\quad \text{c) Sort documents by predictions: } \pi = \text{argsort}(-\{\hat{y}_i\}_{i \in I_q}) \\
\\
&\quad \text{d) Compute ideal DCG:} \\
&\quad\quad \text{IDCG}_q = \sum_{k=1}^{|I_q|} \frac{2^{y_{\text{sorted}[k]}} - 1}{\log_2(k + 1)} \quad \text{(} y_{\text{sorted}} = \text{sort}(\{y_i\}_{i \in I_q}, \text{descending}) \text{)} \\
\\
&\quad \text{e) If IDCG}_q = 0: \text{ continue} \\
\\
&\quad \text{f) For each pair of documents } (i, j) \in I_q \times I_q: \\
&\quad\quad \text{i) If } i = j \text{ or } y_i = y_j: \text{ continue} \\
\\
&\quad\quad \text{ii) Get current ranks: } r_i = \pi^{-1}(i), \quad r_j = \pi^{-1}(j) \\
\\
&\quad\quad \text{iii) Compute gain and discount at each position:} \\
&\quad\quad\quad G_i = 2^{y_i} - 1, \quad G_j = 2^{y_j} - 1 \\
&\quad\quad\quad D_i = \frac{1}{\log_2(r_i + 2)}, \quad D_j = \frac{1}{\log_2(r_j + 2)} \\
\\
&\quad\quad \text{iv) Compute change in NDCG if } i \text{ and } j \text{ are swapped:} \\
&\quad\quad\quad \Delta\text{DCG}_{ij} = (G_i - G_j) \cdot (D_i - D_j) \\
&\quad\quad\quad \Delta\text{NDCG}_{ij} = \frac{\Delta\text{DCG}_{ij}}{\text{IDCG}_q} \\
\\
&\quad\quad \text{v) Compute sigmoid of score difference:} \\
&\quad\quad\quad \sigma_{ij} = \frac{1}{1 + e^{-(\hat{y}_i - \hat{y}_j)}} \\
\\
&\quad\quad \text{vi) Compute lambda value:} \\
&\quad\quad\quad \lambda_{ij} = \begin{cases}
\sigma_{ij} \cdot |\Delta\text{NDCG}_{ij}| & \text{if } y_i > y_j \\
-\sigma_{ij} \cdot |\Delta\text{NDCG}_{ij}| & \text{if } y_i < y_j
\end{cases} \\
\\
&\quad\quad \text{vii) Accumulate gradients and Hessians:} \\
&\quad\quad\quad g_i \leftarrow g_i + \lambda_{ij} \\
&\quad\quad\quad h_i \leftarrow h_i + \sigma_{ij} \cdot (1 - \sigma_{ij}) \cdot |\Delta\text{NDCG}_{ij}| \\
\\
&\text{3. Ensure positive Hessians: } h_i \leftarrow \max(h_i, 10^{-16}) \quad \forall i \\
\\
&\text{4. Return } \{g_i\}_{i=1}^n, \{h_i\}_{i=1}^n
\end{align*}
$$

**Implementation:** [`XGBoost._compute_lambda_gradients()`](xgboost_manual.py#L464-L565)

**Key Insight:** LambdaMART computes gradients based on **pairwise comparisons** within each query. The gradient for document $i$ depends on how swapping it with other documents would affect the ranking metric (NDCG). Documents with higher relevance should be ranked higher, and the gradients push the model in that direction.

**Ranking Approach:** Pairwise - considers all O(n²) document pairs within each query.

---

### Algorithm 1.2: Listwise LambdaMART (ApproxNDCG)

$$
\begin{align*}
&\textbf{Input:} \text{Relevance labels } \{y_i\}_{i=1}^n, \text{ predictions } \{\hat{y}_i\}_{i=1}^n, \text{ query IDs } \{q_i\}_{i=1}^n, \text{ temperature } \tau \\
&\textbf{Output:} \text{Lambda gradients } \{g_i\}_{i=1}^n \text{ and Hessians } \{h_i\}_{i=1}^n \\
\\
&\text{1. Initialize: } g_i \leftarrow 0, \quad h_i \leftarrow 0 \quad \forall i \\
\\
&\text{2. For each unique query } q: \\
&\quad \text{a) Get documents in query: } I_q = \{i : q_i = q\}, \quad |I_q| = m \\
&\quad \text{b) If } m \leq 1: \text{ continue} \\
\\
&\quad \text{c) Compute soft rank for each document using sigmoid approximation:} \\
&\quad\quad \text{For each document } i \in I_q: \\
&\quad\quad\quad s_i = \sum_{j \in I_q, j \neq i} \sigma\left(\frac{\hat{y}_j - \hat{y}_i}{\tau}\right) + 1 \\
&\quad\quad\quad \text{where } \sigma(x) = \frac{1}{1 + e^{-x}} \text{ (smooth rank approximation)} \\
\\
&\quad \text{d) Compute gain and discount at soft ranks:} \\
&\quad\quad \text{For each document } i \in I_q: \\
&\quad\quad\quad G_i = 2^{y_i} - 1 \\
&\quad\quad\quad D_i = \frac{1}{\log_2(s_i + 1)} \\
\\
&\quad \text{e) Compute smooth DCG:} \\
&\quad\quad \text{DCG}_q = \sum_{i \in I_q} G_i \cdot D_i \\
\\
&\quad \text{f) Compute ideal DCG:} \\
&\quad\quad \text{IDCG}_q = \sum_{k=1}^{m} \frac{2^{y_{\text{sorted}[k]}} - 1}{\log_2(k + 1)} \quad \text{(} y_{\text{sorted}} = \text{sort}(\{y_i\}_{i \in I_q}, \text{descending}) \text{)} \\
\\
&\quad \text{g) If IDCG}_q = 0: \text{ continue} \\
\\
&\quad \text{h) Compute smooth NDCG:} \\
&\quad\quad \text{NDCG}_q = \frac{\text{DCG}_q}{\text{IDCG}_q} \\
\\
&\quad \text{i) Compute gradient for each document (chain rule):} \\
&\quad\quad \text{For each document } i \in I_q: \\
&\quad\quad\quad \frac{\partial \text{NDCG}_q}{\partial \hat{y}_i} = \frac{1}{\text{IDCG}_q} \sum_{j \in I_q} G_j \cdot \frac{\partial D_j}{\partial s_j} \cdot \frac{\partial s_j}{\partial \hat{y}_i} \\
\\
&\quad\quad\quad \text{where:} \\
&\quad\quad\quad \frac{\partial D_j}{\partial s_j} = -\frac{1}{\ln(2) \cdot (s_j + 1) \cdot \ln^2(s_j + 1)} \\
\\
&\quad\quad\quad \frac{\partial s_j}{\partial \hat{y}_i} = \begin{cases}
-\sum_{k \in I_q, k \neq j} \frac{\sigma_k}{\tau} \cdot (1 - \sigma_k) & \text{if } i = j \\
\frac{\sigma_i}{\tau} \cdot (1 - \sigma_i) & \text{if } i \neq j
\end{cases} \\
\\
&\quad\quad\quad \text{where } \sigma_k = \sigma\left(\frac{\hat{y}_k - \hat{y}_j}{\tau}\right) \\
\\
&\quad\quad\quad g_i \leftarrow -\frac{\partial \text{NDCG}_q}{\partial \hat{y}_i} \quad \text{(negative gradient for minimization)} \\
\\
&\quad \text{j) Compute Hessian approximation (diagonal approximation):} \\
&\quad\quad \text{For each document } i \in I_q: \\
&\quad\quad\quad h_i \leftarrow \frac{1}{\text{IDCG}_q} \sum_{j \in I_q} |G_j| \cdot \frac{\sigma_j}{\tau^2} \cdot (1 - \sigma_j) \quad \text{(positive approximation)} \\
\\
&\text{3. Ensure positive Hessians: } h_i \leftarrow \max(h_i, 10^{-16}) \quad \forall i \\
\\
&\text{4. Return } \{g_i\}_{i=1}^n, \{h_i\}_{i=1}^n
\end{align*}
$$

### Algorithm 1.3: Soft Ranking Function

The key innovation in listwise ranking is the **differentiable soft rank** approximation:

$$
\text{SoftRank}(i) = 1 + \sum_{j \neq i} \sigma\left(\frac{\hat{y}_j - \hat{y}_i}{\tau}\right)
$$

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

$$
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

where:

- $\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$ (Discounted Cumulative Gain)
- $\text{IDCG@}k$ is the ideal DCG (DCG of perfect ranking by relevance)
- $\text{rel}_i$ is the relevance label of the document at position $i$

**Implementation:** [`_compute_ndcg()`](xgboost_manual.py#L432-L462)

---