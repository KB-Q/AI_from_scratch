
# Hypothesis Testing — t-tests, Permutation, Bootstrap, Multiple Testing

## 0. Why hypothesis tests

A test asks whether the observed data would be surprising if a **null hypothesis** $H_0$ (no effect, no difference) were true. It turns that question into one number, the **p-value**, and a decision rule with a controlled false-alarm rate.

Notation:
- Sample $x_1, ..., x_n$; sample mean $\bar{x}$; sample standard deviation $s = \sqrt{\frac{1}{n-1}\sum_i (x_i - \bar{x})^2}$.
- $H_0$ = null hypothesis, $H_1$ = alternative; $\alpha$ = significance level (false-positive rate we accept).
- $T$ = test statistic, $t_{obs}$ = its observed value.
- $\Phi$ = standard normal CDF; $t_\nu$ = Student-t distribution with $\nu$ degrees of freedom; $\chi^2_\nu$ = chi-square with $\nu$ degrees of freedom.

---

## 1. The testing framework

- **p-value** (two-sided): the probability, under $H_0$, of a statistic at least as extreme as the one observed: $$p = P_{H_0}(|T| \ge |t_{obs}|)$$
- **Decision**: reject $H_0$ when $p < \alpha$.
- **Errors**:
	- Type I (false positive): reject a true $H_0$; probability $\alpha$.
	- Type II (false negative): keep a false $H_0$; probability $\beta$. **Power** $= 1 - \beta$.
- Under $H_0$ a continuous p-value is **Uniform(0, 1)**, so $P(p < \alpha) = \alpha$ **(WHY? if $F$ is the null CDF of $T$, then $F(T) \sim \text{Uniform}(0,1)$ — the probability integral transform, the same fact behind inverse-CDF sampling — and $p$ is a monotone function of $F(T)$)**.
- What a p-value is **not**: not $P(H_0 \mid \text{data})$, not the size of the effect. A tiny effect with a huge $n$ gives a tiny p-value.
- **Duality with confidence intervals**: the $1-\alpha$ confidence interval for $\mu$ is the set of values $\mu_0$ that a level-$\alpha$ test would not reject.

---

## 2. Normal and Student-t distributions

**Normal**: $$\phi(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2 / 2\sigma^2}, \qquad \Phi(z) = \tfrac{1}{2}\left(1 + \text{erf}(z/\sqrt{2})\right)$$

**Why the t distribution**: with known $\sigma$, $Z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$. Replacing the unknown $\sigma$ by $s$ adds noise:
$$T = \frac{\bar{x} - \mu}{s/\sqrt{n}} = \frac{Z}{\sqrt{V/(n-1)}}, \qquad V = \frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}$$
- For normal data $Z$ and $V$ are independent **(WHY? for Gaussian samples $\bar{x}$ and $s^2$ are independent — $\bar{x}$ lives in the direction $\mathbf{1}$, the residuals $x - \bar{x}\mathbf{1}$ in the orthogonal complement)**, and $Z / \sqrt{\chi^2_\nu / \nu}$ is the definition of $t_\nu$.
- $\nu = n - 1$ because $s^2$ uses $n - 1$ independent residuals — one degree of freedom is spent on estimating $\bar{x}$ (see the $n-1$ divisor in `estimation.md`).
- Density: $$f_\nu(t) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})}\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$
- Heavier tails than the normal for small $\nu$ (critical value 4.30 at $\nu = 2$, 2.23 at $\nu = 10$, 1.96 for the normal); $t_\nu \to N(0,1)$ as $\nu \to \infty$.

**CDF via the regularised incomplete beta function** $I_x(a, b) = \frac{1}{B(a,b)} \int_0^x u^{a-1}(1-u)^{b-1} du$:
$$P(T > t) = \tfrac{1}{2} I_{\nu/(\nu + t^2)}\left(\tfrac{\nu}{2}, \tfrac{1}{2}\right) \quad (t > 0)$$
**(WHY? substituting $u = \nu / (\nu + s^2)$ in the tail integral $\int_t^\infty f_\nu(s)\,ds$ turns it into a Beta$(\nu/2, 1/2)$ integral up to $\nu/(\nu+t^2)$)**
- $I_x(a,b)$ is evaluated with a continued fraction (modified Lentz), which converges fast for $x < \frac{a+1}{a+b+2}$; otherwise use the symmetry $I_x(a, b) = 1 - I_{1-x}(b, a)$.

**Function Quantile(F, p):** (inverts any continuous, increasing CDF)
1. Set $lo = -1$, $hi = 1$
2. While $F(lo) > p$: $lo \leftarrow 2 \cdot lo$; while $F(hi) < p$: $hi \leftarrow 2 \cdot hi$ (now $F(lo) \le p \le F(hi)$)
3. Repeat 80 times:
	1. $mid = (lo + hi)/2$
	2. If $F(mid) < p$: $lo \leftarrow mid$, else $hi \leftarrow mid$
4. Return $(lo + hi)/2$

- Pros: works for any CDF, needs no derivative, cannot diverge. Cons: ~80 CDF evaluations per quantile; libraries use rational approximations or Newton steps instead.

**Implementation:** [`normal_cdf()`](../scripts/distributions.py#L18-L21), [`t_pdf()`](../scripts/distributions.py#L24-L28), [`betainc()`](../scripts/distributions.py#L50-L64), [`t_cdf()`](../scripts/distributions.py#L75-L77), [`quantile()`](../scripts/distributions.py#L80-L97)

---

## 3. t-tests

**One-sample** ($H_0: \mu = \mu_0$): $$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \qquad \nu = n - 1, \qquad p = 2\,F_\nu(-|t|)$$
- **Paired** test (before/after on the same units) = one-sample test on the differences $d_i = x_i - y_i$. Pairing removes between-unit variance.

**Two-sample** ($H_0: \mu_x = \mu_y$), $t = (\bar{x} - \bar{y}) / se$:
- **Student** (equal variances): pooled $s_p^2 = \frac{(n_x-1)s_x^2 + (n_y-1)s_y^2}{n_x + n_y - 2}$, $se^2 = s_p^2\left(\frac{1}{n_x} + \frac{1}{n_y}\right)$, $\nu = n_x + n_y - 2$.
- **Welch** (unequal variances): $se^2 = \frac{s_x^2}{n_x} + \frac{s_y^2}{n_y}$, with the Welch–Satterthwaite degrees of freedom $$\nu = \frac{\left(s_x^2/n_x + s_y^2/n_y\right)^2}{\frac{(s_x^2/n_x)^2}{n_x - 1} + \frac{(s_y^2/n_y)^2}{n_y - 1}}$$ **(WHY? $se^2$ is a sum of two scaled $\chi^2$ variables, which is not $\chi^2$ itself; Satterthwaite approximates it by a scaled $\chi^2_\nu$ with the same mean and variance, using $\text{Var}(s^2) = 2\sigma^4/(n-1)$ for each group)**

**Function WelchTTest(x, y):**
1. Compute $\bar{x}, \bar{y}, s_x^2, s_y^2$
2. $se = \sqrt{s_x^2/n_x + s_y^2/n_y}$
3. $t = (\bar{x} - \bar{y}) / se$
4. Compute $\nu$ with the Welch–Satterthwaite formula
5. Return $t$, $\nu$, $p = 2\,F_\nu(-|t|)$

- Use Welch by default: when variances and sample sizes both differ, Student's test has the wrong type-I error rate, and when variances are equal Welch loses almost nothing.
- Assumptions: independent observations; approximately normal sample means (exact for normal data, approximate by the CLT for moderate $n$). Heavy skew with small $n$ → prefer a permutation test or the bootstrap.
- **z-test** vs t-test: same statistic with $\sigma$ known (or $n$ large enough that $t_\nu \approx N(0,1)$).

**Implementation:** [`t_test_one_sample()`](../scripts/hypothesis_tests.py#L11-L21), [`t_test_two_sample()`](../scripts/hypothesis_tests.py#L24-L42)

---

## 4. Permutation test

Under $H_0$ "both samples come from the same distribution", the group labels are **exchangeable**: every reassignment of the pooled values into groups of sizes $n_x, n_y$ is equally likely. The null distribution of any statistic is then obtained by shuffling labels — no distributional assumption.

**Function PermutationTest(x, y, B):**
1. $T_{obs} = |\bar{x} - \bar{y}|$; pool $z = [x, y]$
2. For $b$ from 1 to $B$:
	1. Shuffle $z$; split into the first $n_x$ and the remaining $n_y$ values
	2. $T_b = |\text{mean}(\text{first}) - \text{mean}(\text{rest})|$
3. Return $p = \dfrac{1 + \#\{b : T_b \ge T_{obs}\}}{1 + B}$ **(WHY? the observed labelling is itself one of the equally likely permutations; counting it keeps $P(p \le \alpha) \le \alpha$ exact and the p-value above 0)**

- Pros: exact under exchangeability for any $n$; any statistic (median, trimmed mean, AUC, ...).
- Cons: cost $B$ statistic evaluations; it tests equality of whole distributions, so with unequal variances it is not a pure test of means.

**Implementation:** [`permutation_test()`](../scripts/hypothesis_tests.py#L45-L58)

---

## 5. Bootstrap confidence intervals

**Plug-in principle**: the empirical distribution $\hat{F}$ (mass $1/n$ on each observation) stands in for the unknown $F$. Resampling $n$ points from $\hat{F}$ — i.e. with replacement from the data — simulates new datasets, and the spread of the recomputed statistic $\theta^*$ approximates the sampling distribution of $\hat\theta$.

**Function BootstrapCI(x, statistic, B, α):**
1. $\hat\theta = \text{statistic}(x)$
2. For $b$ from 1 to $B$:
	1. Draw $x^*_b$: $n$ indices uniformly with replacement
	2. $\theta^*_b = \text{statistic}(x^*_b)$
3. Let $q_{\alpha/2}, q_{1-\alpha/2}$ be quantiles of $\{\theta^*_b\}$
4. Percentile interval: $[q_{\alpha/2},\ q_{1-\alpha/2}]$
5. Basic (pivotal) interval: $[2\hat\theta - q_{1-\alpha/2},\ 2\hat\theta - q_{\alpha/2}]$ **(WHY? it assumes $\theta^* - \hat\theta$ is distributed like $\hat\theta - \theta$; solving $q_{\alpha/2} - \hat\theta \le \hat\theta - \theta \le q_{1-\alpha/2} - \hat\theta$ for $\theta$ gives the reflected interval)**
6. Bootstrap standard error: $\text{sd}(\{\theta^*_b\})$

- Each resample omits a point with probability $(1 - 1/n)^n \to e^{-1} \approx 0.368$, so about 63% of distinct points appear — the same fact behind out-of-bag error in random forests.
- Works for statistics with no closed-form standard error (median, ratios, AUC).
- Fails or needs care for: very small $n$; extremes (max, min); heavy tails; dependent data (use a block bootstrap). BCa intervals correct the percentile interval for bias and skew.

**Implementation:** [`bootstrap_ci()`](../scripts/hypothesis_tests.py#L61-L77)

---

## 6. Multiple testing

Running $m$ tests at level $\alpha$ gives $m_0 \alpha$ expected false positives ($m_0$ = number of true nulls); for independent tests $P(\text{at least one}) = 1 - (1-\alpha)^m$, which is 0.64 at $m = 20$.

Two error rates (with $V$ = false rejections, $R$ = all rejections):
- **FWER** (family-wise error rate) $= P(V \ge 1)$ — strict; for confirmatory claims.
- **FDR** (false discovery rate) $= E[V / \max(R, 1)]$ — the expected fraction of discoveries that are false; for screening many hypotheses (genes, metrics, features).

**Bonferroni**: reject $p_i \le \alpha / m$ (adjusted $\tilde{p}_i = \min(1, m p_i)$). FWER $\le \alpha$ **(WHY? union bound: $P(\cup_{i \in \text{nulls}} \{p_i \le \alpha/m\}) \le m_0 \cdot \alpha/m \le \alpha$, for any dependence between tests)**. Holm's step-down version controls FWER too and is uniformly more powerful.

**Function BenjaminiHochberg(p, q):**
1. Sort the p-values: $p_{(1)} \le ... \le p_{(m)}$
2. Find the largest $k$ with $p_{(k)} \le \dfrac{k}{m} q$
3. Reject the hypotheses with the $k$ smallest p-values
4. Adjusted p-values: $\tilde{p}_{(k)} = \min_{j \ge k} \dfrac{m\, p_{(j)}}{j}$ (capped at 1); reject where $\tilde{p} \le q$

- FDR $\le \frac{m_0}{m} q \le q$ for independent or positively dependent tests **(WHY? intuition: null p-values are uniform, so about $m_0 \cdot \frac{kq}{m}$ of them fall below the threshold $\frac{kq}{m}$; among $k$ rejections that is a false fraction of $\frac{m_0}{m}q$)**.
- Trade-off: Bonferroni keeps FWER at $\alpha$ but loses power as $m$ grows; BH allows a controlled share of false discoveries and keeps much more power.

**Implementation:** [`bonferroni()`](../scripts/hypothesis_tests.py#L80-L84), [`benjamini_hochberg()`](../scripts/hypothesis_tests.py#L87-L100)

---

## 7. Other fundamentals (quick reference)

- **Statistical vs practical significance**: report the effect size and its confidence interval, not only $p$. Cohen's $d = (\bar{x} - \bar{y}) / s_p$.
- **One-sided vs two-sided**: one-sided doubles the power for the chosen direction but must be decided before seeing the data.
- **p-hacking**: trying many analyses and reporting the significant one is multiple testing without correction.
- **Chi-square test** for a 2×2 table of counts; equivalent to the two-proportion z-test.
- **Mann–Whitney U** (Wilcoxon rank-sum): a rank-based two-sample test; $U / (n_x n_y)$ estimates $P(X > Y)$, which is the ROC-AUC.
- **CLT**: $\bar{x}$ is approximately $N(\mu, \sigma^2/n)$ for large $n$ whatever the data distribution (finite variance); this is why t- and z-tests work beyond normal data.
