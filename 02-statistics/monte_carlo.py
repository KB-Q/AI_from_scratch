"""
Monte Carlo sampling: inverse-CDF sampling, rejection sampling, importance sampling,
and random-walk Metropolis–Hastings.

Simplifications: 1-D targets; Metropolis–Hastings uses a symmetric Gaussian proposal with a fixed
step size (no adaptation) and reports the acceptance rate and lag-1 autocorrelation rather than a full ESS.
"""
import numpy as np
from distributions import normal_cdf, normal_pdf, normal_ppf


def inverse_cdf_sample(ppf, n, rng):
    """If U ~ Uniform(0, 1) then F⁻¹(U) ~ F, because P(F⁻¹(U) ≤ x) = P(U ≤ F(x)) = F(x)."""
    return ppf(rng.random(n))


def exponential_ppf(u, rate=1.0):
    """F(x) = 1 - e^(-λx), so F⁻¹(u) = -log(1 - u) / λ."""
    return -np.log1p(-u) / rate


def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M, n, rng):
    """
    Needs p(x) ≤ M q(x) everywhere. Draw x ~ q and u ~ Uniform(0, 1); keep x if u < p(x) / (M q(x)).
    Accepted points are exact draws from p, and a proposal is accepted with probability 1/M
    (for normalised p and q), so M should be as small as the bound allows.
    """
    samples, proposed = [], 0
    while sum(len(s) for s in samples) < n:
        x = proposal_sample(rng, 2 * n)
        u = rng.random(2 * n)
        samples.append(x[u < target_pdf(x) / (M * proposal_pdf(x))])
        proposed += 2 * n
    accepted = np.concatenate(samples)
    return accepted[:n], len(accepted) / proposed


def importance_sampling(f, target_pdf, proposal_sample, proposal_pdf, n, rng):
    """
    E_p[f(X)] = E_q[f(X) w(X)] with weights w = p / q, estimated by the mean of f(x_i) w(x_i), x_i ~ q.
    Unbiased whenever q > 0 wherever f p ≠ 0; the variance is small when q puts its mass where |f| p is large.
    """
    x = proposal_sample(rng, n)
    fw = f(x) * target_pdf(x) / proposal_pdf(x)
    return float(fw.mean()), float(fw.std() / np.sqrt(n))


def metropolis_hastings(log_target, x0, n_steps, step, rng):
    """
    Random-walk Metropolis–Hastings. Propose x' = x + step · ε, ε ~ N(0, 1), and accept with
    probability min(1, p(x') / p(x)); the proposal is symmetric, so the Hastings ratio q(x | x') / q(x' | x)
    cancels, and p only needs to be known up to a constant. Detailed balance p(x) T(x → x') = p(x') T(x' → x)
    makes p the stationary distribution of the chain.
    """
    chain = np.empty(n_steps)
    x, log_p, accepted = x0, log_target(x0), 0
    for i in range(n_steps):
        proposal = x + step * rng.normal()
        log_p_new = log_target(proposal)
        if np.log(rng.random()) < log_p_new - log_p:
            x, log_p, accepted = proposal, log_p_new, accepted + 1
        chain[i] = x
    return chain, accepted / n_steps


def example():
    rng = np.random.default_rng(11)

    print("--- inverse-CDF sampling ---")
    x = inverse_cdf_sample(lambda u: exponential_ppf(u, rate=2.0), 100_000, rng)
    print(f"Exponential(λ = 2): mean {x.mean():.4f} (1/λ = 0.5), var {x.var():.4f} (1/λ² = 0.25)")
    z = inverse_cdf_sample(normal_ppf, 5_000, rng)
    print(f"N(0, 1) via the bisection quantile: mean {z.mean():.3f}, sd {z.std():.3f}")

    print("--- rejection sampling: N(0, 1) from a Laplace(0, 1) proposal ---")
    laplace_pdf = lambda x: 0.5 * np.exp(-np.abs(x))
    laplace_sample = lambda r, n: r.laplace(0.0, 1.0, size=n)
    M = np.sqrt(2 * np.e / np.pi)
    z, rate = rejection_sample(normal_pdf, laplace_sample, laplace_pdf, M, 50_000, rng)
    print(f"acceptance rate {rate:.3f} (1/M = {1 / M:.3f}); samples: mean {z.mean():.3f}, sd {z.std():.3f}, "
          f"P(Z > 1.96) = {np.mean(z > 1.96):.4f} (0.0250)")
    assert abs(rate - 1 / M) < 0.01

    print("--- importance sampling: P(Z > 4) for Z ~ N(0, 1) ---")
    exact = 1 - float(normal_cdf(4.0))
    indicator = lambda x: (x > 4.0).astype(float)
    naive = np.mean(rng.normal(size=10_000) > 4.0)
    est, se = importance_sampling(indicator, normal_pdf, lambda r, n: r.normal(4.0, 1.0, size=n),
                                  lambda x: normal_pdf(x, 4.0, 1.0), 10_000, rng)
    print(f"exact {exact:.3e}; naive MC (n = 10k) {naive:.3e}; IS with N(4, 1) proposal (n = 10k) {est:.3e} ± {se:.1e}")
    print(f"naive MC needs n ≈ p(1 - p) / se² = {exact * (1 - exact) / se ** 2:.1e} draws for the same standard error")
    assert abs(est - exact) < 4 * se

    print("--- Metropolis–Hastings on 0.3 N(-2, 0.5²) + 0.7 N(2, 0.5²) (true mean 0.8) ---")
    log_normal = lambda x, mu, sd: -0.5 * ((x - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))
    log_target = lambda x: np.logaddexp(np.log(0.3) + log_normal(x, -2.0, 0.5), np.log(0.7) + log_normal(x, 2.0, 0.5))
    for step in [0.1, 2.5, 25.0]:
        chain, acc = metropolis_hastings(log_target, 0.0, 50_000, step, rng)
        kept = chain[5_000:]
        lag1 = np.corrcoef(kept[:-1], kept[1:])[0, 1]
        print(f"step {step:5.1f}: acceptance {acc:.2f}, mean {kept.mean():+.3f}, "
              f"P(x < 0) {np.mean(kept < 0):.3f} (0.300), lag-1 autocorrelation {lag1:.3f}")


if __name__ == "__main__":
    example()
