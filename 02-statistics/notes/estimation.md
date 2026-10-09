
# Point Estimation — MLE, MAP, Conjugate Priors, Bias–Variance

## 0. Why estimation

An estimator $\hat\theta = g(x_1, ..., x_n)$ turns data into a guess of an unknown parameter $\theta$. Maximum likelihood picks the parameter under which the data are most probable; Bayesian estimation combines the likelihood with a prior. Estimators are compared by their bias, variance, and mean squared error.

Notation:
- Data $x_1, ..., x_n$ i.i.d. from $p(x \mid \theta)$; likelihood $L(\theta) = \prod_i p(x_i \mid \theta)$; log-likelihood $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$.
- Prior $p(\theta)$; posterior $p(\theta \mid x) \propto p(x \mid \theta)\, p(\theta)$.
- Bernoulli data: $k = \sum_i x_i$ successes in $n$ trials. Normal data: $x_i \sim N(\mu, \sigma^2)$; $SS = \sum_i (x_i - \bar{x})^2$.

---

## 1. Maximum likelihood (MLE)

$$\hat\theta_{MLE} = \arg\max_\theta \ell(\theta)$$
Solve the **score equation** $\partial \ell / \partial \theta = 0$ (and check it is a maximum).

**Function BernoulliMLE(x):**
1. $\ell(p) = k \log p + (n-k)\log(1-p)$
2. $\ell'(p) = \dfrac{k}{p} - \dfrac{n-k}{1-p} = 0$
3. Return $\hat{p} = k / n$

**Function NormalMLE(x):**
1. $\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2$
2. $\partial \ell / \partial \mu = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0 \Rightarrow \hat\mu = \bar{x}$
3. $\partial \ell / \partial \sigma^2 = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i (x_i - \mu)^2 = 0 \Rightarrow \hat\sigma^2 = SS / n$
4. Return $\hat\mu, \hat\sigma^2$

Properties:
- **Consistent**: $\hat\theta \to \theta$ as $n \to \infty$ (under regularity conditions).
- **Asymptotically normal and efficient**: $\sqrt{n}(\hat\theta - \theta) \to N(0, I(\theta)^{-1})$, which reaches the Cramér–Rao bound (§5).
- **Invariant**: the MLE of $g(\theta)$ is $g(\hat\theta)$.
- Can be **biased** in finite samples: $E[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$ (§4).
- Minimising cross-entropy / log loss in ML is maximum likelihood for a Bernoulli or categorical model; minimising MSE is maximum likelihood under Gaussian noise.

**Implementation:** [`mle_bernoulli()`](../estimation.py#L11-L13), [`mle_normal()`](../estimation.py#L16-L19)

---

## 2. MAP and the posterior

$$\hat\theta_{MAP} = \arg\max_\theta \left[\ell(\theta) + \log p(\theta)\right]$$
- The prior acts as a **regulariser** **(WHY? $-\log p(\theta)$ is added to the negative log-likelihood: a Gaussian prior $w \sim N(0, \tau^2 I)$ adds $\|w\|^2 / 2\tau^2$, i.e. L2 / ridge; a Laplace prior adds $\|w\|_1 / b$, i.e. L1 / lasso)**.
- As $n$ grows the likelihood dominates and MAP → MLE; a flat prior gives MAP = MLE.
- Point summaries of the posterior: mode (MAP, minimises 0–1 loss), mean (minimises squared loss), median (minimises absolute loss).
- MAP is **not** invariant to reparameterisation (the density of $g(\theta)$ picks up a Jacobian), unlike the MLE.
- **Credible vs confidence interval**: a 95% credible interval contains $\theta$ with posterior probability 0.95 given the data; a 95% confidence interval is produced by a procedure that covers the fixed $\theta$ in 95% of repeated experiments.

---

## 3. Conjugate priors

A prior is **conjugate** to a likelihood when the posterior is in the same family, so updating is arithmetic on the parameters.

**Beta–Binomial**: prior $p \sim \text{Beta}(a, b) \propto p^{a-1}(1-p)^{b-1}$, likelihood $\propto p^k (1-p)^{n-k}$:
$$p \mid x \sim \text{Beta}(a + k,\ b + n - k)$$
- $a$ and $b$ act as **pseudo-counts** of prior successes and failures.
- Posterior mean $\frac{a + k}{a + b + n}$ is a weighted average of the prior mean $\frac{a}{a+b}$ (weight $a+b$) and the MLE $\frac{k}{n}$ (weight $n$).
- MAP (posterior mode) $= \frac{k + a - 1}{n + a + b - 2}$; Beta(1, 1) (uniform) gives the MLE.
- Equal-tailed credible interval: the $\alpha/2$ and $1-\alpha/2$ quantiles of the Beta CDF $I_x(a + k, b + n - k)$.

**Normal–Normal** ($\sigma$ known): prior $\mu \sim N(\mu_0, \tau_0^2)$, data $x_i \sim N(\mu, \sigma^2)$:

**Function NormalNormalPosterior(x, σ, μ0, τ0):**
1. Precisions add: $\dfrac{1}{\tau_n^2} = \dfrac{1}{\tau_0^2} + \dfrac{n}{\sigma^2}$ **(WHY? the log-posterior $-\frac{(\mu - \mu_0)^2}{2\tau_0^2} - \sum_i \frac{(x_i - \mu)^2}{2\sigma^2}$ is quadratic in $\mu$; completing the square gives a Gaussian whose inverse variance is the sum of the quadratic coefficients)**
2. Mean is precision-weighted: $\mu_n = \tau_n^2\left(\dfrac{\mu_0}{\tau_0^2} + \dfrac{n\bar{x}}{\sigma^2}\right)$
3. Return $\mu_n, \tau_n$ (the posterior is $N(\mu_n, \tau_n^2)$; its mean is also the MAP)

- **Shrinkage**: $\mu_n = w\bar{x} + (1-w)\mu_0$ with $w = \frac{n/\sigma^2}{n/\sigma^2 + 1/\tau_0^2}$; with few observations the estimate is pulled toward the prior mean.
- Other conjugate pairs: Gamma–Poisson, Dirichlet–Multinomial (Laplace smoothing in naive Bayes is a Dirichlet prior), Normal-Inverse-Gamma for unknown $\sigma$.

**Implementation:** [`beta_binomial_posterior()`](../estimation.py#L22-L24), [`map_bernoulli()`](../estimation.py#L27-L30), [`beta_credible_interval()`](../estimation.py#L33-L37), [`normal_normal_posterior()`](../estimation.py#L40-L49)

---

## 4. Bias, variance, and MSE

$$\text{MSE}(\hat\theta) = E[(\hat\theta - \theta)^2] = \underbrace{(E[\hat\theta] - \theta)^2}_{\text{bias}^2} + \underbrace{\text{Var}(\hat\theta)}_{\text{variance}}$$
**(WHY? add and subtract $E[\hat\theta]$ inside the square; the cross term $2(E[\hat\theta] - \theta)\,E[\hat\theta - E\hat\theta]$ is zero)**

**Why $n - 1$ makes $s^2$ unbiased**: $SS = \sum_i (x_i - \mu)^2 - n(\bar{x} - \mu)^2$, so $$E[SS] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$ **(WHY? $\bar{x}$ is fitted to the sample, so the residuals around $\bar{x}$ are smaller than those around the true $\mu$ by exactly the variance of $\bar{x}$)**

**Which divisor minimises MSE** (normal data, $SS/\sigma^2 \sim \chi^2_{n-1}$ with mean $n-1$ and variance $2(n-1)$): for the estimator $c \cdot SS$,
$$\text{MSE}(c) = \sigma^4\left[(c(n-1) - 1)^2 + 2(n-1)c^2\right]$$
- $c = \frac{1}{n-1}$ (unbiased): MSE $= \frac{2\sigma^4}{n-1}$
- $c = \frac{1}{n}$ (MLE): MSE $= \frac{(2n-1)\sigma^4}{n^2}$
- $c = \frac{1}{n+1}$: MSE $= \frac{2\sigma^4}{n+1}$, the minimum **(WHY? setting $d\,\text{MSE}/dc = 0$ gives $c(n-1) - 1 + 2c = 0$, i.e. $c = 1/(n+1)$)**
- An unbiased estimator is not automatically the best one: accepting some bias for less variance can lower MSE — the same trade-off as regularisation.

**Function BiasVarianceSimulation(estimator, sampler, θ, S):**
1. For $s$ from 1 to $S$: draw a dataset from the sampler and compute $\hat\theta_s$
2. bias $= \text{mean}(\hat\theta_s) - \theta$; variance $= \text{var}(\hat\theta_s)$
3. Return bias, variance, bias² + variance

**Implementation:** [`bias_variance()`](../estimation.py#L52-L57)

---

## 5. Cramér–Rao lower bound

For any unbiased estimator, $$\text{Var}(\hat\theta) \ge \frac{1}{n\,I(\theta)}, \qquad I(\theta) = E\left[\left(\frac{\partial \log p(x \mid \theta)}{\partial\theta}\right)^2\right] = -E\left[\frac{\partial^2 \log p(x \mid \theta)}{\partial\theta^2}\right]$$
- $I(\theta)$ (Fisher information) is the expected curvature of the log-likelihood: a sharply peaked likelihood pins $\theta$ down.
- **Bernoulli**: $\frac{\partial^2}{\partial p^2}\left[x\log p + (1-x)\log(1-p)\right] = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}$, with expectation $-\frac{1}{p(1-p)}$, so $I(p) = \frac{1}{p(1-p)}$ and the bound is $\frac{p(1-p)}{n}$ — exactly $\text{Var}(\hat{p})$, so $\hat{p}$ is **efficient**.
- The inverse Fisher information is also the asymptotic variance of the MLE, which gives Wald confidence intervals $\hat\theta \pm z_{1-\alpha/2}/\sqrt{n I(\hat\theta)}$.

---

## 6. Other fundamentals (quick reference)

- **Unbiased vs consistent**: $s^2$ is unbiased and consistent; $\hat\sigma^2_{MLE}$ is biased but consistent; $x_1$ alone is unbiased for $\mu$ but not consistent.
- **Method of moments**: match sample moments to theoretical ones (e.g. $\hat\mu = \bar{x}$, $\hat\sigma^2 = \frac{1}{n}\sum x_i^2 - \bar{x}^2$); simple, usually less efficient than the MLE.
- **Sufficient statistic**: a function of the data that carries all information about $\theta$ (e.g. $k$ for Bernoulli, $(\sum x_i, \sum x_i^2)$ for the normal).
- **Standard error of the mean**: $\sigma/\sqrt{n}$; quadrupling $n$ halves it.
- **Frequentist vs Bayesian**: $\theta$ fixed and data random vs $\theta$ random with a prior; they agree asymptotically when the prior has support at the true $\theta$ (Bernstein–von Mises).
