"""
Normal and Student-t distributions from scratch: pdf, CDF, and quantile.

The normal CDF uses math.erf. The t CDF goes through the regularised incomplete beta function,
evaluated with the modified Lentz continued fraction (Numerical Recipes §6.4). Quantiles invert
any continuous CDF by bisection, which is slower than the rational approximations libraries use.
"""
import math
import numpy as np
_erf = np.vectorize(math.erf)


def normal_pdf(x, mu=0.0, sigma=1.0):
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x, mu=0.0, sigma=1.0):
    """Φ(z) = (1 + erf(z / √2)) / 2."""
    z = (np.asarray(x, dtype=float) - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1.0 + _erf(z))


def t_pdf(x, df):
    """f(t) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + t²/ν)^(-(ν+1)/2)."""
    x = np.asarray(x, dtype=float)
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    return np.exp(log_c - (df + 1) / 2 * np.log1p(x ** 2 / df))


def _beta_continued_fraction(a, b, x, max_iter=300, eps=1e-15):
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        # even then odd term of the continued fraction
        for aa in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                   -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """
    Regularised incomplete beta I_x(a, b) = B(x; a, b) / B(a, b).
    The continued fraction converges fast for x < (a + 1) / (a + b + 2); otherwise use the
    symmetry I_x(a, b) = 1 - I_{1-x}(b, a).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b


def _t_cdf_scalar(t, df):
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


_t_cdf = np.vectorize(_t_cdf_scalar)


def t_cdf(x, df):
    """F(t) = 1 - I_{ν/(ν+t²)}(ν/2, 1/2) / 2 for t > 0, and F(-t) = 1 - F(t)."""
    return _t_cdf(np.asarray(x, dtype=float), df)


def quantile(cdf, p, iters=80):
    """
    Invert a continuous increasing CDF by bisection: double [lo, hi] until it brackets p,
    then halve it; 80 halvings take any bracket below double precision.
    """
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must lie in (0, 1)")
    lo, hi = -np.ones_like(p), np.ones_like(p)
    while np.any(cdf(lo) > p):
        lo = np.where(cdf(lo) > p, 2 * lo, lo)
    while np.any(cdf(hi) < p):
        hi = np.where(cdf(hi) < p, 2 * hi, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def normal_ppf(p, mu=0.0, sigma=1.0):
    return mu + sigma * quantile(normal_cdf, p)


def t_ppf(p, df):
    return quantile(lambda x: t_cdf(x, df), p)


def example():
    print("--- reference values ---")
    print(f"Φ(1.96)          = {float(normal_cdf(1.96)):.6f}   (0.975002)")
    print(f"Φ⁻¹(0.975)       = {float(normal_ppf(0.975)):.6f}   (1.959964)")
    print(f"F_t(2.0; ν=5)    = {float(t_cdf(2.0, 5)):.6f}   (0.949030)")
    print(f"F_t⁻¹(0.975; 10) = {float(t_ppf(0.975, 10)):.6f}   (2.228139)")
    assert np.isclose(normal_ppf(0.975), 1.959964, atol=1e-6)
    assert np.isclose(t_ppf(0.975, 10), 2.228139, atol=1e-6)

    print("--- consistency checks ---")
    grid = np.linspace(-30, 30, 200_001)
    for name, pdf, cdf in [("normal", normal_pdf, normal_cdf),
                           ("t, ν=3", lambda x: t_pdf(x, 3), lambda x: t_cdf(x, 3))]:
        mass = np.trapezoid(pdf(grid), grid)
        mass_cdf = float(cdf(30.0) - cdf(-30.0))
        x = np.array([-2.5, -0.3, 0.0, 1.1, 4.0])
        h = 1e-5
        slope_gap = np.max(np.abs((cdf(x + h) - cdf(x - h)) / (2 * h) - pdf(x)))
        print(f"{name:7s}: ∫pdf over [-30, 30] = {mass:.6f}, F(30) - F(-30) = {mass_cdf:.6f}, "
              f"max |dF/dx - pdf| = {slope_gap:.1e}")
        assert abs(mass - mass_cdf) < 1e-6 and slope_gap < 1e-6

    p = np.array([0.001, 0.2, 0.5, 0.9, 0.999])
    roundtrip = np.max(np.abs(t_cdf(t_ppf(p, 7), 7) - p))
    print(f"max |F(F⁻¹(p)) - p| for t, ν=7: {roundtrip:.1e}")
    print(f"t quantile → normal as ν grows: "
          + ", ".join(f"ν={df}: {float(t_ppf(0.975, df)):.4f}" for df in [2, 10, 100, 10_000])
          + f"; normal: {float(normal_ppf(0.975)):.4f}")


if __name__ == "__main__":
    example()
