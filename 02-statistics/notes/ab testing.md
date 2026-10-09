
# A/B Testing — Power, Peeking, CUPED

## 0. Why A/B tests

Randomly assigning units (users, sessions) to control A or treatment B makes the two groups identical in expectation in everything except the treatment, so a difference in the metric estimates the **causal** effect of the treatment. The statistics is a two-sample test; the design questions are how many units to collect, when to look, and how to reduce noise.

Notation:
- Arms A (control) and B (treatment), $n$ units per arm; metric $y$ with standard deviation $\sigma$.
- True effect $\delta = \mu_B - \mu_A$; estimate $\hat{d} = \bar{y}_B - \bar{y}_A$ with standard error $se = \sigma\sqrt{2/n}$ (equal arms, equal variance).
- $\alpha$ = significance level, $1 - \beta$ = power; $z_q = \Phi^{-1}(q)$ (e.g. $z_{0.975} = 1.96$, $z_{0.8} = 0.84$).
- $\rho$ = correlation between the metric and a pre-experiment covariate $x$.

---

## 1. Power and sample size

The two-sided test rejects when $|\hat{d}| > z_{1-\alpha/2}\, se$. Under $H_1$, $\hat{d} \sim N(\delta, se^2)$, so
$$\text{power} = \Phi\!\left(\frac{\delta}{se} - z_{1-\alpha/2}\right) + \Phi\!\left(-\frac{\delta}{se} - z_{1-\alpha/2}\right)$$
The second term (rejecting in the wrong direction) is negligible for any useful power. Setting the first term to $1 - \beta$ gives $\delta = (z_{1-\alpha/2} + z_{1-\beta})\, se$, and with $se = \sigma\sqrt{2/n}$:
$$n = \frac{2\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2} \text{ per arm}$$

**Function SampleSize(δ, σ, α, power):**
1. $z_a = \Phi^{-1}(1 - \alpha/2)$, $z_b = \Phi^{-1}(\text{power})$
2. Return $\lceil 2\sigma^2 (z_a + z_b)^2 / \delta^2 \rceil$

- **Rule of thumb** (Lehr): at $\alpha = 0.05$ and 80% power, $2(1.96 + 0.84)^2 \approx 15.7$, so $n \approx 16\sigma^2/\delta^2$ per arm.
- $n \propto \sigma^2 / \delta^2$: halving the effect you want to detect quadruples the sample; halving the variance halves it (the motivation for CUPED, §3).
- **Minimum detectable effect** (MDE) for a fixed $n$: $\delta_{min} = (z_{1-\alpha/2} + z_{1-\beta})\,\sigma\sqrt{2/n}$.
- **Proportions** (conversion rates $p_A, p_B$, $\bar{p} = (p_A + p_B)/2$): the variance differs under $H_0$ and $H_1$, so $$n = \frac{\left[z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{1-\beta}\sqrt{p_A(1-p_A) + p_B(1-p_B)}\right]^2}{(p_B - p_A)^2}$$ Detecting 10% → 11% needs about 14,800 users per arm.
- Low-powered tests that do reach significance overstate the effect (the "winner's curse"): only the noisy draws that happened to be large cross the threshold.

**Implementation:** [`sample_size_means()`](../scripts/ab_testing.py#L13-L21), [`power_means()`](../scripts/ab_testing.py#L24-L27), [`minimum_detectable_effect()`](../scripts/ab_testing.py#L30-L32), [`sample_size_proportions()`](../scripts/ab_testing.py#L35-L43)

---

## 2. Peeking

Checking the p-value repeatedly while data arrive, and stopping at the first $p < \alpha$, inflates the false-positive rate even when there is no effect.

**Function PeekingSimulation(looks, batch, sims):** (A/A tests, $\sigma = 1$ known)
1. For each of `sims` experiments:
	1. For each look $k = 1..K$:
		1. Add `batch` users to each arm; $n_k = k \cdot batch$
		2. $z_k = (\bar{y}_B - \bar{y}_A) / \sqrt{2/n_k}$
		3. If $|z_k| > z_{1-\alpha/2}$: record a false positive and stop
2. Return the fraction of experiments with a false positive

- **Why it inflates**: each look is a level-$\alpha$ test, but the experiment rejects if **any** of the $K$ correlated tests rejects. The sequence $z_k$ behaves like a scaled random walk, and a random walk crosses any fixed boundary eventually (law of the iterated logarithm), so with unlimited looks the false-positive rate tends to 1.
- With equally spaced looks at $\alpha = 0.05$ the false-positive rate is about 0.14 for 5 looks, 0.25 for 20, and 0.37 for 100 (Armitage 1969; the simulation reproduces these).
- **Remedies**:
	- fix the sample size in advance and analyse once;
	- group-sequential designs: spend $\alpha$ across planned looks with stricter early boundaries (O'Brien–Fleming, Pocock, Lan–DeMets $\alpha$-spending);
	- always-valid p-values / confidence sequences (mixture SPRT), which stay valid under continuous monitoring.

**Implementation:** [`peeking_false_positive_rate()`](../scripts/ab_testing.py#L46-L58)

---

## 3. CUPED (variance reduction with a pre-experiment covariate)

Use a covariate $x$ measured **before** the experiment (typically the same metric in the preceding weeks) that correlates with $y$. Define
$$y^{cv} = y - \theta(x - \bar{x})$$

- **Unbiased for any θ**: $x$ is pre-treatment and assignment is random, so $E[\bar{x}_B - \bar{x}_A] = 0$ and $E[\bar{y}^{cv}_B - \bar{y}^{cv}_A] = \delta$.
- **Optimal θ**: $\text{Var}(y - \theta x) = \text{Var}(y) - 2\theta\,\text{Cov}(x,y) + \theta^2\,\text{Var}(x)$; setting the derivative to 0 gives $$\theta^* = \frac{\text{Cov}(x, y)}{\text{Var}(x)}, \qquad \text{Var}(y^{cv}) = \text{Var}(y)(1 - \rho^2)$$ **(WHY? plugging $\theta^*$ back in leaves $\text{Var}(y) - \text{Cov}(x,y)^2/\text{Var}(x) = \text{Var}(y)(1 - \rho^2)$; $\theta^*$ is the OLS slope of $y$ on $x$)**
- A correlation of $\rho = 0.7$ removes 51% of the variance, which is the same as doubling the sample size.

**Function CUPED(y, x):** (both arms pooled)
1. $\theta = \sum_i (x_i - \bar{x})(y_i - \bar{y}) \,/\, \sum_i (x_i - \bar{x})^2$
2. $y^{cv}_i = y_i - \theta(x_i - \bar{x})$
3. Analyse $y^{cv}$ with the usual two-sample test

- The covariate must not be affected by the treatment; an in-experiment covariate biases the estimate.
- Equivalent to regression adjustment (ANCOVA): regress $y$ on the treatment indicator and $x$. Several covariates → multiple regression; using an ML prediction of $y$ from pre-period features as $x$ is known as CUPAC.
- Users with no pre-period data (new users) can get $x = $ a constant plus an indicator.

**Implementation:** [`cuped()`](../scripts/ab_testing.py#L61-L71)

---

## 4. Other fundamentals (quick reference)

- **Sample ratio mismatch (SRM)**: if a 50/50 split yields, say, 50.6/49.4 with millions of users, a chi-square test on the counts flags broken assignment or logging; results are untrustworthy until it is fixed.
- **A/A tests**: run the pipeline with no treatment to check the false-positive rate and the variance estimate.
- **Unit of randomisation vs unit of analysis**: randomising users but analysing sessions makes observations dependent; use user-level aggregates or cluster-robust / delta-method standard errors.
- **Ratio metrics** (e.g. clicks per session): $\text{Var}(\bar{Y}/\bar{X}) \approx \frac{1}{\mu_X^2}\left[\text{Var}(\bar{Y}) - 2R\,\text{Cov}(\bar{Y},\bar{X}) + R^2\,\text{Var}(\bar{X})\right]$ with $R = \mu_Y/\mu_X$ (delta method).
- **Interference / SUTVA violations**: in marketplaces and social networks, treated units affect control units; use cluster or switchback designs.
- **Novelty and primacy effects**: early behaviour differs from steady state; look at the effect over time.
- **Many metrics or segments**: apply multiple-testing control (see `hypothesis testing.md`) or pre-register a primary metric.
