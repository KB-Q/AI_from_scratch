
# Monte Carlo Sampling — Inverse CDF, Rejection, Importance Sampling, Metropolis–Hastings

## 0. Why Monte Carlo

Many quantities are expectations, $E_p[f(X)] = \int f(x)\,p(x)\,dx$ — probabilities, posterior means, losses. Monte Carlo replaces the integral by an average over samples:
$$\hat\mu = \frac{1}{n}\sum_{i=1}^n f(x_i), \quad x_i \sim p, \qquad \text{sd}(\hat\mu) = \frac{\text{sd}_p(f)}{\sqrt{n}}$$
The error shrinks as $1/\sqrt{n}$ regardless of the dimension of $x$ **(WHY? $\hat\mu$ is a mean of i.i.d. terms, so its variance is $\text{Var}(f)/n$ by the CLT; dimension enters only through $\text{Var}(f)$)**. The hard part is drawing $x_i \sim p$; the methods below differ in what they need to know about $p$.

Notation:
- Target density $p$ (sometimes known only up to a constant, $\tilde{p} = Z p$); proposal density $q$ that is easy to sample.
- $F$ = CDF, $F^{-1}$ = quantile function; $U \sim \text{Uniform}(0, 1)$.
- Importance weight $w(x) = p(x)/q(x)$.

---

## 1. Inverse-CDF sampling

If $U \sim \text{Uniform}(0,1)$ then $X = F^{-1}(U)$ has CDF $F$ **(WHY? $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$, since $F$ is increasing)**.

**Function InverseCDFSample(F⁻¹, n):**
1. Draw $u_1, ..., u_n \sim \text{Uniform}(0,1)$
2. Return $F^{-1}(u_1), ..., F^{-1}(u_n)$

- Exponential($\lambda$): $F(x) = 1 - e^{-\lambda x}$, so $F^{-1}(u) = -\log(1-u)/\lambda$.
- Discrete distributions: return the smallest $k$ with $F(k) \ge u$ (binary search on the cumulative sums) — this is how categorical / multinomial sampling, e.g. LLM token sampling, works.
- Pros: exact, one uniform per sample, no rejection. Cons: needs $F^{-1}$ in closed form or a numerical inversion (the normal has none; `distributions.py` inverts by bisection); mostly limited to 1-D.

**Implementation:** [`inverse_cdf_sample()`](../scripts/monte_carlo.py#L12-L14), [`exponential_ppf()`](../scripts/monte_carlo.py#L17-L19)

---

## 2. Rejection sampling

Requires a proposal $q$ and a constant $M$ with $p(x) \le M q(x)$ for all $x$.

**Function RejectionSample(p, q, M, n):**
1. Repeat until $n$ samples are accepted:
	1. Draw $x \sim q$ and $u \sim \text{Uniform}(0,1)$
	2. Accept $x$ if $u < \dfrac{p(x)}{M q(x)}$
2. Return the accepted samples

- Accepted samples are exact draws from $p$, and the acceptance probability is $1/M$ **(WHY? $P(\text{accept}, X \in dx) = q(x)\,dx \cdot \frac{p(x)}{M q(x)} = \frac{p(x)\,dx}{M}$; integrating gives $1/M$, and dividing by it leaves $p(x)\,dx$)**. With an unnormalised $\tilde{p} \le M q$ the acceptance rate is $Z/M$, and the samples are still exact.
- Example: $N(0,1)$ from Laplace(0, 1): $\frac{p(x)}{q(x)} = \sqrt{2/\pi}\, e^{|x| - x^2/2}$ is maximised at $|x| = 1$, giving $M = \sqrt{2e/\pi} \approx 1.32$ and a 76% acceptance rate.
- Cons: $M$ must be a true bound (hard to find), and it grows exponentially with dimension for product-form proposals, so rejection sampling is a low-dimensional tool.

**Implementation:** [`rejection_sample()`](../scripts/monte_carlo.py#L22-L35)

---

## 3. Importance sampling

Sample from $q$ and reweight:
$$E_p[f(X)] = \int f(x)\frac{p(x)}{q(x)}q(x)\,dx = E_q[f(X)\,w(X)] \approx \frac{1}{n}\sum_i f(x_i)\,w(x_i), \quad x_i \sim q$$

**Function ImportanceSampling(f, p, q, n):**
1. Draw $x_1, ..., x_n \sim q$
2. $w_i = p(x_i) / q(x_i)$
3. Return $\hat\mu = \frac{1}{n}\sum_i f(x_i) w_i$ and its standard error $\text{sd}(f w)/\sqrt{n}$

- Unbiased whenever $q > 0$ wherever $f p \ne 0$.
- The **optimal proposal** is $q^* \propto |f|\,p$; for $f \ge 0$ it gives zero variance **(WHY? then $f w = f p / q^* = \int f p$ is constant)**. In practice: put the proposal's mass where $|f| p$ is large.
- **Rare events**: for $P(Z > 4) \approx 3.2 \times 10^{-5}$, naive Monte Carlo with $10^4$ draws sees 0 or 1 hits; a $N(4, 1)$ proposal estimates it to about 2% relative error with the same $10^4$ draws, where naive sampling would need about $7 \times 10^7$.
- **Self-normalised IS** for $p$ known only up to a constant: $\hat\mu = \sum_i w_i f(x_i) / \sum_i w_i$; biased by $O(1/n)$, consistent.
- **Weight degeneracy**: if $q$ has lighter tails than $p$, a few huge weights dominate and the variance can be infinite. Kish's effective sample size $(\sum w_i)^2 / \sum w_i^2$ diagnoses this for self-normalised estimates of general $f$; a proposal tailored to one $f$ (like the $N(4,1)$ above) can have a tiny ESS while estimating that $f$ well.

**Implementation:** [`importance_sampling()`](../scripts/monte_carlo.py#L38-L45)

---

## 4. Markov chain Monte Carlo — Metropolis–Hastings

Build a Markov chain whose stationary distribution is $p$; after a burn-in, the chain's states are (correlated) samples from $p$. Only ratios $p(x')/p(x)$ are needed, so the normalising constant $Z$ — the usual obstacle in Bayesian posteriors — cancels.

**Function MetropolisHastings(log p̃, x₀, steps, step size s):** (random-walk proposal)
1. $x = x_0$
2. For $t$ from 1 to steps:
	1. Propose $x' = x + s\,\varepsilon$, $\varepsilon \sim N(0, 1)$
	2. Accept with probability $\alpha = \min\left(1, \dfrac{\tilde{p}(x')}{\tilde{p}(x)}\right)$ **(WHY? the general ratio is $\frac{\tilde{p}(x')\,q(x \mid x')}{\tilde{p}(x)\,q(x' \mid x)}$; a symmetric proposal makes the $q$ terms cancel)**; compare in log space: accept if $\log u < \log\tilde{p}(x') - \log\tilde{p}(x)$
	3. If accepted: $x \leftarrow x'$ (otherwise $x$ is repeated in the chain)
	4. Append $x$ to the chain
3. Discard the burn-in and return the chain

- **Why $p$ is stationary**: the transition satisfies **detailed balance**, $p(x)\,T(x \to x') = p(x')\,T(x' \to x)$ **(WHY? $p(x)\,q(x' \mid x)\,\alpha(x \to x') = \min\left(p(x)q(x' \mid x),\ p(x')q(x \mid x')\right)$, which is symmetric in $x$ and $x'$)**; summing both sides over $x$ shows that one step of the chain leaves $p$ unchanged. Irreducibility and aperiodicity then make the chain converge to $p$ from any start.
- **Step size**: too small → almost every proposal is accepted but the chain moves slowly (high autocorrelation); too large → most proposals land in low-density regions and are rejected. Acceptance rates around 0.44 in 1-D and 0.234 in high-dimensional Gaussian targets are optimal for random-walk proposals (Roberts 1997).
- **Multimodal targets**: a small step never crosses the low-density gap between modes, so the chain reports only one mode and looks converged. Diagnose with several chains from dispersed starts ($\hat{R}$); remedies are larger or mixed proposals and tempering.
- **Autocorrelation** reduces the information per sample: the effective sample size of a chain is $n / (1 + 2\sum_{k \ge 1} \rho_k)$, with $\rho_k$ the lag-$k$ autocorrelation.
- **Variants**: Gibbs sampling draws each coordinate from its full conditional (an MH move that is always accepted); Hamiltonian Monte Carlo uses gradients of $\log p$ to make long, high-acceptance moves.

**Implementation:** [`metropolis_hastings()`](../scripts/monte_carlo.py#L48-L63)

---

## 5. Other fundamentals (quick reference)

- **Which method when**: closed-form $F^{-1}$ → inverse CDF; low dimension with a good bound → rejection; an expectation (especially a rare event) under a known density → importance sampling; high-dimensional or unnormalised targets (Bayesian posteriors) → MCMC.
- **Box–Muller**: two uniforms → two independent normals via $\sqrt{-2\log u_1}\,(\cos 2\pi u_2, \sin 2\pi u_2)$.
- **Variance reduction**: control variates (the same idea as CUPED in `ab testing.md`), antithetic variates ($u$ and $1-u$), stratification.
- **Reparameterisation trick** (VAEs): write $x = \mu + \sigma\varepsilon$ with $\varepsilon \sim N(0,1)$ so the Monte Carlo estimate is differentiable in $\mu, \sigma$.
- **Bootstrap** is Monte Carlo over the empirical distribution (see `hypothesis testing.md`).
