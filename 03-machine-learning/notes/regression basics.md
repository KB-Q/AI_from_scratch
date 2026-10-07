
# Regression Basics

**Regression** predicts a continuous target $y$ from features $X$. (Classification predicts a discrete label — but logistic/softmax "regression" in §6 borrow the same machinery.)

Notation:
- $m$ = number of training examples; $n$ = number of features.
- $x^{(i)} = (x_1^{(i)}, ..., x_n^{(i)})$ = feature vector of example $i$; $y^{(i)}$ = its target.
- Weights $w = (w_1, ..., w_n)$, bias $b$. Prediction $\hat{y}$.
- Design matrix $X$ is $m \times n$ (rows = examples). $\theta = (b, w)$ stacked, with a column of 1s prepended to $X$ for the bias.

---

## 1. Simple Linear Regression

One feature, fit a straight line:
$$\hat{y} = w x + b$$

Fit by minimizing **Mean Squared Error (MSE)** — average squared residual:
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

**(WHY? the $\tfrac{1}{2}$ is a convenience — it cancels the 2 from differentiating the square, keeping the gradient clean)**

MSE is convex in $(w, b)$, so there is a single global minimum.

---

## 2. Multiple Linear Regression

$n$ features, a hyperplane:
$$\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b = w^{\top} x + b$$

Vectorized cost over the whole dataset: $J(\theta) = \dfrac{1}{2m} \lVert X\theta - y \rVert^2$.

Two ways to solve:

**(a) Normal Equation** — closed-form, no iteration:
$$\theta = (X^{\top} X)^{-1} X^{\top} y$$
**(WHY? set $\nabla_\theta J = 0$ and solve; this is exactly the least-squares projection of $y$ onto the column space of $X$)**

- Pros: exact, no learning rate, no iterations. Cons: computing $(X^{\top}X)^{-1}$ is $O(n^3)$ — impractical for large $n$; fails/is unstable when $X^{\top}X$ is singular (collinear features).

**(b) Gradient Descent** — iterative, scales to large $n$ and $m$:

**Function GradientDescentLinReg(X, y, lr $\eta$, epochs):**
1. Initialize $\theta = 0$ (or small random)
2. For $t$ from 1 to epochs:
	1. Predict for all examples: $\hat{y} = X\theta$
	2. Compute residuals: $r = \hat{y} - y$
	3. Compute gradient: $$\nabla_\theta J = \frac{1}{m} X^{\top} r$$
	4. Update: $\theta \mathrel{-}= \eta \cdot \nabla_\theta J$
	5. Repeat for each epoch $t$
3. Return $\theta$

Gradient-descent variants — differ only in how many examples per gradient step:
- **Batch GD**: full dataset per step (as above). Stable, slow per step.
- **Stochastic GD (SGD)**: one example per step. Noisy, fast, can escape shallow regions.
- **Mini-batch GD**: a batch of $b$ examples per step. The usual compromise.

Practical note: **feature scaling** (standardize each feature to zero mean, unit variance) is important for GD — unequal feature scales make the cost surface elongated and slow convergence.

---

## 3. Polynomial Regression

Model nonlinear relationships while staying *linear in the parameters* by expanding the feature set with powers (and cross terms) of the inputs:
$$\hat{y} = b + w_1 x + w_2 x^2 + ... + w_p x^p$$

- It's still linear regression — just fit on transformed features $[x, x^2, ..., x^p]$, so the same normal equation / gradient descent applies.
- Degree $p$ is the key knob: too low → **underfit** (high bias); too high → **overfit** (high variance, wild wiggles). This motivates regularization (§5).

---

## 4. Bias–Variance & the Case for Regularization

- **Bias**: error from too-simple a model (underfitting). **Variance**: error from over-sensitivity to the training data (overfitting).
- Total expected error $\approx \text{bias}^2 + \text{variance} + \text{irreducible noise}$. Lowering one often raises the other — the **bias–variance tradeoff**.
- A model with many features / high polynomial degree has low bias but high variance. **Regularization** adds a penalty on weight magnitude to the cost, trading a little bias for a large drop in variance — it shrinks weights toward 0 so the model can't over-fit noise.

---

## 5. Regularization — L2, L1, Elastic Net

Add a penalty term to the MSE cost. $\lambda \ge 0$ controls strength (larger → more shrinkage). The bias $b$ is conventionally **not** penalized.

### L2 — Ridge Regression

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n} w_j^2$$

- Penalizes squared weights. Shrinks all weights smoothly toward 0 but rarely exactly to 0 → keeps all features, just smaller.
- Gradient step just adds a shrink term: $\nabla_{w} J = \frac{1}{m}X^{\top}r + \lambda w$, so each update does $w \leftarrow w(1 - \eta\lambda) - \eta \cdot \frac{1}{m}X^\top r$ — "weight decay".
- Closed form (Ridge normal equation): $$\theta = (X^{\top}X + \lambda I)^{-1} X^{\top} y$$ **(WHY? the $+\lambda I$ makes the matrix always invertible — fixes the collinearity/singularity problem of plain OLS)**

### L1 — Lasso Regression

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |w_j|$$

- Penalizes absolute weights. Drives many weights **exactly to 0** → automatic **feature selection** (sparse model).
- $|w_j|$ is non-differentiable at 0, so plain gradient descent doesn't apply directly; use **coordinate descent** (or sub-gradient methods). Coordinate descent optimizes one weight at a time with a **soft-thresholding** update.

**Function LassoCoordinateDescent(X, y, $\lambda$, iterations):** (features assumed standardized)
1. Initialize $w = 0$
2. For $t$ from 1 to iterations:
	1. For each feature $j$ in $1..n$:
		1. Compute partial residual excluding feature $j$: $r_j = y - \sum_{k \neq j} w_k x_k$
		2. Compute the correlation of feature $j$ with that residual: $\rho_j = x_j^{\top} r_j$
		3. Apply soft-thresholding **(WHY? this is the closed-form 1-D minimizer of the L1-penalized cost; it zeroes out weights whose correlation is smaller than $\lambda$)** $$w_j = \frac{\text{sign}(\rho_j) \cdot \max(|\rho_j| - \lambda,\ 0)}{x_j^{\top} x_j}$$
		4. Repeat for each feature $j$
	2. Repeat for each iteration $t$
3. Return $w$

### Elastic Net

Combines both penalties — useful when features are correlated (L1 alone arbitrarily picks one of a correlated group; L2 shares among them):
$$J(w) = \frac{1}{2m}\sum_i (\hat{y}^{(i)} - y^{(i)})^2 + \lambda \left( \alpha \sum_j |w_j| + \frac{1-\alpha}{2} \sum_j w_j^2 \right)$$
$\alpha \in [0,1]$ mixes L1 ($\alpha=1$) and L2 ($\alpha=0$).

**L1 vs L2 intuition**: the L1 constraint region is a diamond with corners on the axes, so the optimum tends to land on a corner (a weight = 0); the L2 region is a smooth ball, so the optimum rarely sits exactly on an axis.

---

## 6. Logistic & Multinomial Regression (classification via regression)

When the target is a **class** rather than a continuous value, feed the linear score through a squashing function and fit with a likelihood loss.

### Binary Logistic Regression

Pass the linear score through the **sigmoid** to get a probability in $[0,1]$:
$$p = \sigma(w^{\top} x + b) = \frac{1}{1 + e^{-(w^{\top} x + b)}}$$

Fit by minimizing **binary cross-entropy (log loss)**:
$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log (1 - p^{(i)}) \right]$$

The gradient has the same clean form as linear regression: $\nabla_w J = \frac{1}{m} X^{\top}(p - y)$ **(WHY? the sigmoid derivative cancels against the log-loss derivative, collapsing to the residual $p - y$)** — so the same gradient-descent loop (§2b) trains it, and the same L1/L2 penalties apply.

### Multinomial (Softmax) Regression

Generalizes logistic to $K$ classes. One weight vector $w_k$ per class; convert scores to a probability distribution with **softmax**:
$$p_k = \frac{e^{w_k^{\top} x}}{\sum_{j=1}^{K} e^{w_j^{\top} x}}$$

Fit by minimizing **categorical cross-entropy**: $J = -\frac{1}{m}\sum_i \sum_{k} y_k^{(i)} \log p_k^{(i)}$, where $y^{(i)}$ is a one-hot label. Gradient per class: $\nabla_{w_k} J = \frac{1}{m} \sum_i (p_k^{(i)} - y_k^{(i)}) x^{(i)}$ — again the residual form, trained by the same gradient descent.

---

## 7. Other Fundamentals (quick reference)

- **Assumptions of linear regression (OLS)**: linear relationship, independent errors, homoscedasticity (constant error variance), normally distributed errors, little multicollinearity. Violations bias inference (not always prediction).
- **$R^2$ (coefficient of determination)**: fraction of variance in $y$ explained by the model, $R^2 = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$. 1 = perfect, 0 = no better than predicting the mean.
- **Adjusted $R^2$**: $R^2$ penalized for the number of features — stops it from rising just because you added predictors.
- **MAE vs MSE vs RMSE**: MAE = mean absolute error (robust to outliers, linear penalty); MSE/RMSE square the error (penalize large misses more, sensitive to outliers).
- **Multicollinearity**: highly correlated features make OLS weights unstable and hard to interpret; Ridge is the standard remedy.
- **Train/validation/test split & cross-validation**: choose $\lambda$, polynomial degree, and other hyperparameters on validation data, never on the test set.
- **Normalization vs standardization**: min-max scale to $[0,1]$ vs zero-mean/unit-variance; important for GD and mandatory for L1/L2 so the penalty treats features fairly.
