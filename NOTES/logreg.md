

$$
\begin{align*}
&\textbf{Input:} X \in \mathbb{R}^{n \times d}, y \in \mathbb{R}^{n} \\
&\textbf{Output:} \hat{y} \in \mathbb{R}^{n} \\
\\
&\textbf{Parameters:} \\
&\quad \text{Learning rate } \eta \\
&\quad \text{Number of iterations } T \\
&\quad \text{Regularization parameter } \lambda \\
\\
&\textbf{Model:} \\
&\quad\quad \text{Sigmoid function: } \sigma(z) = \frac{1}{1 + e^{-z}} \\
&\quad\quad \text{Prediction: } \hat{y}_i = \sigma(w^T x_i + b) = \frac{1}{1 + e^{-(w^T x_i + b)}} \\
\\
&\textbf{Loss Function (Binary Cross-Entropy + L2):} \\
&\quad\quad L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right] + \frac{\lambda}{2} \|w\|^2 \\
\\
&\textbf{Gradients:} \\
&\quad\quad \frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)x_i + \lambda w \\
&\quad\quad \frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \\
\\
&\textbf{Algorithm (Gradient Descent):} \\
&\quad \text{1. Initialize: } w \leftarrow 0 \in \mathbb{R}^d, \, b \leftarrow 0 \\
&\quad \text{2. For } t = 1 \text{ to } T: \\
&\quad\quad \text{a) Compute predictions: } \hat{y}_i = \sigma(w^T x_i + b) \quad \forall i \\
&\quad\quad \text{b) Compute gradients: } \\
&\quad\quad\quad g_w = \frac{1}{n} X^T(\hat{y} - y) + \lambda w \\
&\quad\quad\quad g_b = \frac{1}{n} \mathbf{1}^T(\hat{y} - y) \\
&\quad\quad \text{c) Update parameters: } \\
&\quad\quad\quad w \leftarrow w - \eta g_w \\
&\quad\quad\quad b \leftarrow b - \eta g_b \\
&\quad\quad \text{d) Optional: Check convergence } |L^{(t)} - L^{(t-1)}| < \text{tol} \\
&\quad \text{3. Return } w, b \\
\end{align*}
$$

**Implementation:** [`LogisticRegressionNumpy`](logreg.py)