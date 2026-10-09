"""
A/B-test design and analysis: power and sample size, the false-positive cost of peeking,
and CUPED variance reduction.

Simplifications: power and sample size use the two-sided z-approximation; peeking is simulated on
A/A tests with a known variance and equally spaced looks; CUPED uses one pre-experiment covariate.
"""
import numpy as np
from distributions import normal_cdf, normal_ppf
from hypothesis_tests import t_test_two_sample


def sample_size_means(delta, sigma, alpha=0.05, power=0.8):
    """
    Per-arm n for a two-sided two-sample test to detect a mean difference delta:
        n = 2 σ² (z_{1-α/2} + z_{power})² / δ².
    The estimate d̂ = ȳ_B - ȳ_A has standard error σ√(2/n). The test rejects when |d̂| > z_{1-α/2} σ√(2/n);
    requiring P(reject | δ) = power, and ignoring the far tail, gives δ = (z_{1-α/2} + z_power) σ√(2/n).
    """
    z_a, z_b = float(normal_ppf(1 - alpha / 2)), float(normal_ppf(power))
    return int(np.ceil(2 * sigma ** 2 * (z_a + z_b) ** 2 / delta ** 2))


def power_means(delta, sigma, n, alpha=0.05):
    """P(|d̂| > z_{1-α/2} se) = Φ(δ/se - z_{1-α/2}) + Φ(-δ/se - z_{1-α/2}), with se = σ√(2/n)."""
    se, z_a = sigma * np.sqrt(2 / n), float(normal_ppf(1 - alpha / 2))
    return float(normal_cdf(delta / se - z_a) + normal_cdf(-delta / se - z_a))


def minimum_detectable_effect(sigma, n, alpha=0.05, power=0.8):
    """δ_min = (z_{1-α/2} + z_power) σ √(2/n), the inverse of sample_size_means."""
    return float((normal_ppf(1 - alpha / 2) + normal_ppf(power)) * sigma * np.sqrt(2 / n))


def sample_size_proportions(p_a, p_b, alpha=0.05, power=0.8):
    """
    n = [z_{1-α/2} √(2 p̄(1 - p̄)) + z_power √(p_a(1 - p_a) + p_b(1 - p_b))]² / (p_b - p_a)², p̄ = (p_a + p_b) / 2.
    The first term is the standard error under H0 (pooled rate), the second under H1.
    """
    z_a, z_b = float(normal_ppf(1 - alpha / 2)), float(normal_ppf(power))
    p_bar = (p_a + p_b) / 2
    root = z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    return int(np.ceil(root ** 2 / (p_b - p_a) ** 2))


def peeking_false_positive_rate(n_looks=20, batch=100, n_sims=4000, alpha=0.05, rng=None):
    """
    A/A test (no true effect, σ = 1 known). After every batch of users per arm, run a z-test on all
    data so far and stop at the first rejection. One look rejects with probability α; k looks reject
    if any of k positively correlated statistics crosses the boundary, which happens more often as k
    grows. Returns (fixed-horizon rate, rate with peeking).
    """
    rng = rng or np.random.default_rng(0)
    diff = rng.normal(size=(n_sims, n_looks * batch)) - rng.normal(size=(n_sims, n_looks * batch))
    n_seen = batch * np.arange(1, n_looks + 1)
    z = np.cumsum(diff, axis=1)[:, n_seen - 1] / np.sqrt(2 * n_seen)
    crossed = np.abs(z) > float(normal_ppf(1 - alpha / 2))
    return float(crossed[:, -1].mean()), float(crossed.any(axis=1).mean())


def cuped(y, x):
    """
    CUPED (Deng 2013): y_cv = y - θ (x - x̄) with θ = cov(x, y) / var(x), where x is measured before
    the experiment (e.g. the same metric in the prior weeks), so it is independent of treatment.
    The correction has mean zero in both arms, so the treatment effect estimate stays unbiased, and
    var(y_cv) = var(y) - 2θ cov(x, y) + θ² var(x), minimised at this θ to var(y)(1 - ρ²).
    θ is estimated on both arms pooled.
    """
    xc, yc = x - x.mean(), y - y.mean()
    theta = np.dot(xc, yc) / np.dot(xc, xc)
    return y - theta * xc, float(theta)


def example():
    rng = np.random.default_rng(7)

    print("--- power and sample size ---")
    delta, sigma = 0.1, 1.0
    n = sample_size_means(delta, sigma)
    rejections = [t_test_two_sample(rng.normal(0, sigma, n), rng.normal(delta, sigma, n))[2] < 0.05
                  for _ in range(1000)]
    print(f"δ = {delta}σ: n = {n} per arm; analytic power {power_means(delta, sigma, n):.3f}, "
          f"simulated power {np.mean(rejections):.3f}")
    print(f"MDE at n = {n}: {minimum_detectable_effect(sigma, n):.4f}σ")
    print(f"conversion 10% → 11%: n = {sample_size_proportions(0.10, 0.11)} per arm")
    assert abs(np.mean(rejections) - 0.8) < 0.05

    print("--- peeking (A/A tests, α = 0.05) ---")
    for looks in [1, 5, 20, 100]:
        fixed, peeking = peeking_false_positive_rate(n_looks=looks, batch=2000 // looks, rng=rng)
        print(f"{looks:3d} looks: false-positive rate {peeking:.3f} (fixed horizon {fixed:.3f})")

    print("--- CUPED (ρ = 0.7 between pre-period and in-experiment metric, true effect 0.05) ---")
    n, rho, effect = 2000, 0.7, 0.05
    treated = np.arange(2 * n) < n
    plain, adjusted, metric_ratio = [], [], []
    for _ in range(2000):
        x = rng.normal(size=2 * n)
        y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=2 * n) + effect * treated
        y_cv, _ = cuped(y, x)
        plain.append(y[treated].mean() - y[~treated].mean())
        adjusted.append(y_cv[treated].mean() - y_cv[~treated].mean())
        metric_ratio.append(y_cv[treated].var() / y[treated].var())
    print(f"mean effect estimate over 2,000 experiments: plain {np.mean(plain):.4f}, CUPED {np.mean(adjusted):.4f}")
    print(f"var(y_cv) / var(y) within an experiment: {np.mean(metric_ratio):.3f} (theory 1 - ρ² = {1 - rho ** 2:.3f})")
    print(f"var of the effect estimate, CUPED / plain, across experiments: {np.var(adjusted) / np.var(plain):.3f}")
    assert abs(np.mean(metric_ratio) - (1 - rho ** 2)) < 0.01


if __name__ == "__main__":
    example()
