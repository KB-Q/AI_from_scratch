"""
Hypothesis tests from scratch: one- and two-sample t-tests, a permutation test, bootstrap
confidence intervals, and Bonferroni / Benjamini–Hochberg corrections for multiple tests.

Simplifications: all tests are two-sided; the bootstrap gives percentile and basic intervals only (no BCa).
"""
import numpy as np
from distributions import t_cdf, t_ppf


def t_test_one_sample(x, mu0=0.0):
    """
    H0: E[x] = mu0. Under H0 with normal data, t = (x̄ - mu0) / (s / √n) follows Student-t with
    n - 1 degrees of freedom; for other data the CLT makes this approximate for large n.
    A paired test is this test on the within-pair differences.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    t = (x.mean() - mu0) / (x.std(ddof=1) / np.sqrt(n))
    df = n - 1
    return float(t), df, float(2 * t_cdf(-abs(t), df))


def t_test_two_sample(x, y, equal_var=False):
    """
    H0: E[x] = E[y], with t = (x̄ - ȳ) / se.
    Welch (default): se² = s_x²/n_x + s_y²/n_y, with Welch–Satterthwaite degrees of freedom
        df = se⁴ / [(s_x²/n_x)² / (n_x - 1) + (s_y²/n_y)² / (n_y - 1)].
    Student (equal_var=True): pooled s_p² = [(n_x - 1) s_x² + (n_y - 1) s_y²] / (n_x + n_y - 2),
        se² = s_p² (1/n_x + 1/n_y), df = n_x + n_y - 2.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    if equal_var:
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        se2, df = pooled * (1 / nx + 1 / ny), nx + ny - 2
    else:
        se2 = vx / nx + vy / ny
        df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t = (x.mean() - y.mean()) / np.sqrt(se2)
    return float(t), float(df), float(2 * t_cdf(-abs(t), df))


def permutation_test(x, y, n_permutations=10_000, rng=None):
    """
    H0: x and y come from the same distribution, so the group labels are exchangeable.
    Shuffle the pooled sample, split it into groups of the original sizes, and count shuffles whose
    |mean difference| is at least the observed one. The +1 in numerator and denominator counts the
    observed split as one of the permutations, which keeps the test valid and the p-value above 0.
    """
    rng = rng or np.random.default_rng(0)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pooled = np.concatenate([x, y])
    observed = abs(x.mean() - y.mean())
    perms = rng.permuted(np.tile(pooled, (n_permutations, 1)), axis=1)
    diffs = np.abs(perms[:, :len(x)].mean(axis=1) - perms[:, len(x):].mean(axis=1))
    return float((np.sum(diffs >= observed) + 1) / (n_permutations + 1))


def bootstrap_ci(x, statistic=np.mean, alpha=0.05, n_boot=10_000, method="percentile", rng=None):
    """
    Resample x with replacement n_boot times and recompute the statistic θ*_b on each resample.
    Percentile: [q_{α/2}(θ*), q_{1-α/2}(θ*)].
    Basic (pivotal): [2θ̂ - q_{1-α/2}(θ*), 2θ̂ - q_{α/2}(θ*)], which assumes θ* - θ̂ is
    distributed like θ̂ - θ.
    `statistic` must accept an `axis` argument (np.mean, np.median, ...).
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    resamples = x[rng.integers(0, len(x), size=(n_boot, len(x)))]
    boot = statistic(resamples, axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    if method == "percentile":
        return float(lo), float(hi)
    theta = statistic(x)
    return float(2 * theta - hi), float(2 * theta - lo)


def bonferroni(p_values, alpha=0.05):
    """FWER = P(at least one false rejection) ≤ m · α/m = α by the union bound; adjusted p = min(1, m p)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.minimum(1.0, p * len(p))
    return adjusted <= alpha, adjusted


def benjamini_hochberg(p_values, q=0.05):
    """
    Step-up procedure that controls FDR = E[false rejections / max(rejections, 1)] ≤ q for
    independent (or positively dependent) tests: sort p ascending, find the largest k with
    p_(k) ≤ k q / m, and reject the k smallest. Equivalently, reject where the adjusted
    p-value p̃_(k) = min_{j ≥ k} m p_(j) / j is ≤ q.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return adjusted <= q, adjusted


def example():
    rng = np.random.default_rng(42)

    print("--- one effect: t-tests, permutation test, bootstrap ---")
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.6, 2.0, size=25)
    t_w, df_w, p_w = t_test_two_sample(a, b)
    t_s, df_s, p_s = t_test_two_sample(a, b, equal_var=True)
    print(f"Welch:       t = {t_w:.3f}, df = {df_w:.1f}, p = {p_w:.4f}")
    print(f"Student:     t = {t_s:.3f}, df = {df_s:.0f},   p = {p_s:.4f}")
    print(f"Permutation: p = {permutation_test(a, b, rng=rng):.4f}")
    half = float(t_ppf(0.975, len(a) - 1)) * a.std(ddof=1) / np.sqrt(len(a))
    intervals = {"t-interval": (a.mean() - half, a.mean() + half),
                 "percentile bootstrap": bootstrap_ci(a, rng=rng),
                 "basic bootstrap": bootstrap_ci(a, method="basic", rng=rng)}
    for name, (lo, hi) in intervals.items():
        print(f"95% CI for mean(a), {name:20s}: [{lo:.3f}, {hi:.3f}]")
    lo, hi = bootstrap_ci(b, np.median, rng=rng)
    print(f"95% CI for median(b), percentile bootstrap: [{lo:.3f}, {hi:.3f}]")

    print("--- type-I error under H0 (2,000 A/A tests, n = 15 per arm) ---")
    p_null = np.array([t_test_two_sample(rng.normal(size=15), rng.normal(size=15))[2] for _ in range(2000)])
    print(f"rejection rate at α = 0.05: {np.mean(p_null < 0.05):.3f}")
    assert 0.035 < np.mean(p_null < 0.05) < 0.065

    print("--- multiple testing (m = 200 tests, 20 true effects, 300 repetitions) ---")
    m, m1, reps = 200, 20, 300
    is_null = np.arange(m) >= m1
    results = {"none": [], "Bonferroni": [], "BH": []}
    for _ in range(reps):
        shift = np.where(is_null, 0.0, 1.2)
        p = np.array([t_test_one_sample(rng.normal(s, 1.0, size=20))[2] for s in shift])
        for name, reject in [("none", p < 0.05), ("Bonferroni", bonferroni(p)[0]), ("BH", benjamini_hochberg(p)[0])]:
            false = np.sum(reject & is_null)
            results[name].append((false > 0, false / max(reject.sum(), 1), np.sum(reject & ~is_null) / m1))
    for name, r in results.items():
        fwer, fdr, power = np.mean(r, axis=0)
        print(f"{name:10s}: FWER = {fwer:.3f}, FDR = {fdr:.3f}, power = {power:.3f}")
    assert np.mean(results["Bonferroni"], axis=0)[0] <= 0.08
    assert np.mean(results["BH"], axis=0)[1] <= 0.07


if __name__ == "__main__":
    example()
