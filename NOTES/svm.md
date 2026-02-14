
### Linear SVM (Primal Formulation)

$$
\begin{align*}
&\textbf{Input:} X \in \mathbb{R}^{n \times d}, y \in \{-1, +1\}^n \\
&\textbf{Output:} w \in \mathbb{R}^d, b \in \mathbb{R} \\
\\
&\textbf{Parameters:} \\
&\quad \text{Learning rate } \eta \\
&\quad \text{Number of iterations } T \\
&\quad \text{Regularization parameter } \lambda \\
\\
&\textbf{Objective (Hinge Loss):} \\
&\quad\quad \min_{w,b} \, \lambda \|w\|^2 + \frac{1}{n}\sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b)) \\
\\
&\textbf{Decision Function:} \\
&\quad\quad f(x) = \text{sign}(w^T x + b) \\
\\
&\textbf{Algorithm (Stochastic Gradient Descent):} \\
&\quad \text{1. Initialize: } w \leftarrow 0 \in \mathbb{R}^d, \, b \leftarrow 0 \\
&\quad \text{2. Convert labels: } y_i \in \{-1, +1\} \quad \forall i \\
&\quad \text{3. For } t = 1 \text{ to } T: \\
&\quad\quad \text{a) Shuffle data indices} \\
&\quad\quad \text{b) For each sample } (x_i, y_i): \\
&\quad\quad\quad \text{i) Compute margin: } m_i = y_i(w^T x_i + b) \\
&\quad\quad\quad \text{ii) If } m_i \geq 1 \text{ (correctly classified):} \\
&\quad\quad\quad\quad w \leftarrow w - \eta (2\lambda w) \\
&\quad\quad\quad \text{iii) Else (violates margin):} \\
&\quad\quad\quad\quad w \leftarrow w - \eta (2\lambda w - y_i x_i) \\
&\quad\quad\quad\quad b \leftarrow b - \eta (-y_i) \\
&\quad \text{4. Return } w, b \\
\end{align*}
$$

**Implementation:** [`SVM._fit_primal()`](SVM.py)

---

## Support Vector Machine (Kernel)

### Kernel SVM (Dual Formulation)

For non-linear decision boundaries, we use the **kernel trick**.

#### Kernel Functions

$$
\begin{align*}
&\textbf{Linear:} && K(x_i, x_j) = x_i^T x_j \\
&\textbf{RBF/Gaussian:} && K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) \\
&\textbf{Polynomial:} && K(x_i, x_j) = (\gamma x_i^T x_j + r)^d \\
&\textbf{Sigmoid:} && K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)
\end{align*}
$$

#### Decision Function (Dual)

$$
f(x) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b\right)
$$

#### Algorithm (Dual SGD with Kernels)

$$
\begin{align*}
&\textbf{Input:} X \in \mathbb{R}^{n \times d}, y \in \{-1, +1\}^n, \text{ kernel function } K \\
&\textbf{Output:} \alpha \in \mathbb{R}^n, b \in \mathbb{R} \\
\\
&\textbf{Parameters:} \\
&\quad \text{Learning rate } \eta \\
&\quad \text{Number of iterations } T \\
&\quad \text{Regularization parameter } \lambda \\
&\quad \text{Kernel parameters (}\gamma, d, r\text{)} \\
\\
&\textbf{Algorithm:} \\
&\quad \text{1. Initialize: } \alpha \leftarrow 0 \in \mathbb{R}^n, \, b \leftarrow 0 \\
&\quad \text{2. Precompute kernel matrix: } K_{ij} = K(x_i, x_j) \quad \forall i,j \\
&\quad \text{3. For } t = 1 \text{ to } T: \\
&\quad\quad \text{a) Shuffle data indices} \\
&\quad\quad \text{b) For each sample index } i: \\
&\quad\quad\quad \text{i) Compute decision: } d_i = \sum_{j=1}^{n} \alpha_j y_j K_{ji} + b \\
&\quad\quad\quad \text{ii) Compute margin: } m_i = y_i \cdot d_i \\
&\quad\quad\quad \text{iii) If } m_i \geq 1: \\
&\quad\quad\quad\quad \alpha_i \leftarrow \alpha_i - \eta (2\lambda \alpha_i) \\
&\quad\quad\quad \text{iv) Else:} \\
&\quad\quad\quad\quad \alpha_i \leftarrow \alpha_i - \eta (2\lambda \alpha_i - y_i) \\
&\quad\quad\quad\quad b \leftarrow b - \eta (-y_i) \\
&\quad \text{4. Store training data } X, y \\
&\quad \text{5. Return } \alpha, b \\
\end{align*}
$$

**Implementation:** [`SVM._fit_dual()`](SVM.py)

---