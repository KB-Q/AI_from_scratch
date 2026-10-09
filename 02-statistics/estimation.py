"""
Point estimation: maximum likelihood, MAP and full posteriors with conjugate priors
(Beta–Binomial, Normal–Normal), and the bias–variance decomposition of an estimator's error.

Simplifications: closed-form estimators only (no numerical MLE); the Normal–Normal model assumes a known σ.
"""
import numpy as np
from distributions import betainc, quantile


def mle_bernoulli(x):
    """Maximise k log p + (n - k) log(1 - p): setting k/p - (n - k)/(1 - p) = 0 gives p̂ = k / n."""
    return float(np.mean(x))


def mle_normal(x):
    """μ̂ = x̄ and σ̂² = Σ(x - x̄)² / n; the 1/n divisor makes σ̂² biased by a factor (n - 1) / n."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(np.mean((x - x.mean()) ** 2))


def beta_binomial_posterior(k, n, a, b):
    """Beta(a, b) prior × Binomial(n, p) likelihood ∝ p^(a+k-1) (1-p)^(b+n-k-1), i.e. Beta(a + k, b + n - k)."""
    return a + k, b + n - k


def map_bernoulli(x, a, b):
    """Mode of the Beta(a + k, b + n - k) posterior: (k + a - 1) / (n + a + b - 2). Beta(1, 1) gives the MLE."""
    x = np.asarray(x)
    return float((x.sum() + a - 1) / (len(x) + a + b - 2))


def beta_credible_interval(a, b, level=0.95):
    """Equal-tailed interval from the Beta CDF I_x(a, b), inverted by bisection."""
    cdf = np.vectorize(lambda x: betainc(a, b, x))
    lo, hi = quantile(cdf, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def normal_normal_posterior(x, sigma, mu0, tau0):
    """
    x_i ~ N(μ, σ²) with σ known and prior μ ~ N(mu0, tau0²). Precisions add:
        1/τ_n² = 1/τ0² + n/σ²,   μ_n = τ_n² (mu0/τ0² + n x̄/σ²).
    μ_n is a precision-weighted average of the prior mean and x̄; it is also the MAP estimate.
    """
    x = np.asarray(x, dtype=float)
    precision = 1 / tau0 ** 2 + len(x) / sigma ** 2
    mu_n = (mu0 / tau0 ** 2 + x.sum() / sigma ** 2) / precision
    return float(mu_n), float(np.sqrt(1 / precision))


def bias_variance(estimator, sampler, theta, n_sims=20_000, rng=None):
    """MSE(θ̂) = E[(θ̂ - θ)²] = (E[θ̂] - θ)² + Var(θ̂) = bias² + variance, estimated by simulation."""
    rng = rng or np.random.default_rng(0)
    estimates = np.array([estimator(sampler(rng)) for _ in range(n_sims)])
    bias, variance = estimates.mean() - theta, estimates.var()
    return float(bias), float(variance), float(bias ** 2 + variance)


def example():
    rng = np.random.default_rng(3)

    print("--- Bernoulli: MLE vs MAP vs posterior (true p = 0.3, prior Beta(4, 4)) ---")
    a, b, p_true = 4, 4, 0.3
    flips = rng.random(1000) < p_true
    for n in [5, 20, 100, 1000]:
        x = flips[:n]
        a_n, b_n = beta_binomial_posterior(int(x.sum()), n, a, b)
        lo, hi = beta_credible_interval(a_n, b_n)
        print(f"n = {n:4d}: MLE {mle_bernoulli(x):.3f}, MAP {map_bernoulli(x, a, b):.3f}, "
              f"posterior Beta({a_n}, {b_n}), 95% credible interval [{lo:.3f}, {hi:.3f}]")
    assert np.isclose(map_bernoulli(flips[:50], 1, 1), mle_bernoulli(flips[:50]))

    print("--- Normal–Normal: shrinkage toward the prior mean (μ = 2, σ = 1, prior N(0, 0.5²)) ---")
    data = rng.normal(2.0, 1.0, size=200)
    for n in [1, 5, 50, 200]:
        mu_n, tau_n = normal_normal_posterior(data[:n], 1.0, 0.0, 0.5)
        print(f"n = {n:3d}: x̄ = {data[:n].mean():.3f}, posterior mean {mu_n:.3f}, posterior sd {tau_n:.3f}")

    print("--- variance estimators, n = 10, σ² = 4: divisor n - 1 vs n (MLE) vs n + 1 ---")
    n, var_true = 10, 4.0
    sampler = lambda r: r.normal(0.0, 2.0, size=n)
    mse = {}
    for divisor in [n - 1, n, n + 1]:
        estimator = lambda x, d=divisor: np.sum((x - x.mean()) ** 2) / d
        bias, variance, mse[divisor] = bias_variance(estimator, sampler, var_true, rng=rng)
        print(f"divisor {divisor:2d}: bias {bias:+.3f}, variance {variance:.3f}, MSE {mse[divisor]:.3f}")
    print(f"theory: MSE = 2σ⁴/(n-1) = {2 * 16 / 9:.3f}, (2n-1)σ⁴/n² = {19 * 16 / 100:.3f}, 2σ⁴/(n+1) = {2 * 16 / 11:.3f}")
    assert mse[n + 1] < mse[n] < mse[n - 1]

    print("--- Cramér–Rao bound for the Bernoulli MLE (p = 0.3, n = 50) ---")
    _, variance, _ = bias_variance(mle_bernoulli, lambda r: r.random(50) < 0.3, 0.3, rng=rng)
    print(f"Var(p̂) = {variance:.5f}, bound p(1 - p)/n = {0.3 * 0.7 / 50:.5f}")


if __name__ == "__main__":
    example()
