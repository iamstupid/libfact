#!/usr/bin/env python3
"""ECM parameter optimizer.

1. Implements Dickman rho via delay-DE
2. Reads calibration CSV (stage 1 / stage 2 costs)
3. For each factor size, sweeps B1, derives B2 from cost balance,
   computes expected cost = 2*T_s1 / P(B1,B2,p), finds minimum.

Usage:
  python optimize_schedule.py < calibration.csv
"""
import sys
import math
import numpy as np
from pathlib import Path

# ── Dickman rho function ──
# rho(u) = 1                    for 0 <= u <= 1
# u*rho'(u) = -rho(u-1)        for u > 1
# Solve via Euler method with step h.

def compute_dickman_rho(u_max=20.0, h=0.0001):
    """Return (u_array, rho_array) for u in [0, u_max].

    Uses trapezoidal rule on the delay-DE: u*rho'(u) = -rho(u-1).
    """
    n = int(u_max / h) + 1
    rho = np.zeros(n)
    back = int(round(1.0 / h))  # index offset for "u - 1"

    for i in range(n):
        u_i = i * h
        if u_i <= 1.0 + 1e-12:
            rho[i] = 1.0
        elif u_i <= 2.0 + 1e-12:
            rho[i] = 1.0 - math.log(max(u_i, 1.0 + 1e-15))
        else:
            # Trapezoidal: rho[i] = rho[i-1] - h/2*(rho[i-1-back]/(u-h) + rho[i-back]/u)
            u_prev = (i - 1) * h
            rho[i] = rho[i-1] - (h / 2.0) * (
                rho[i - 1 - back] / u_prev + rho[i - back] / u_i)

    u_arr = np.linspace(0, u_max, n)
    return u_arr, rho


def make_rho_interp(u_max=20.0):
    """Return callable rho(u) via linear interpolation on precomputed table."""
    u_arr, rho_arr = compute_dickman_rho(u_max)
    def rho(u):
        u = float(u)
        if u <= 0:
            return 1.0
        if u >= u_max:
            return 0.0
        return float(np.interp(u, u_arr, rho_arr))
    return rho


def ecm_prob(B1, B2, p, rho_func):
    """Probability of ECM success per curve.

    P = rho(u) + integral_1^v rho(u-t)/t dt

    u = ln(p/12) / ln(B1)   (Z/6 torsion: |E|/12 must be smooth)
    v = ln(B2) / ln(B1)
    """
    if B1 < 2 or B2 <= B1:
        return 0.0

    # Effective smoothness divisor: torsion 12 + extra smoothness exp(3.134)≈23
    # from GMP-ECM (Suyama curves).
    EXTRA_SMOOTH = math.exp(3.134)  # ≈ 22.97
    ln_B1 = math.log(B1)
    u = math.log(p / EXTRA_SMOOTH) / ln_B1
    v = math.log(B2) / ln_B1

    if u <= 0:
        return 1.0  # trivially smooth

    # Stage 1 probability
    p1 = float(rho_func(u))

    # Stage 2: integral_1^v rho(u-t)/t dt via Simpson's rule
    n_steps = 200
    if v <= 1.0:
        return max(p1, 1e-30)

    ts = np.linspace(1.0, v, n_steps + 1)
    dt = ts[1] - ts[0]
    integrand = np.array([float(rho_func(u - t)) / t for t in ts])
    # Simpson's rule
    p2 = dt / 3.0 * (integrand[0] + integrand[-1]
                      + 4.0 * np.sum(integrand[1::2])
                      + 2.0 * np.sum(integrand[2:-1:2]))

    # Empirical correction: measured success rates on 100/120/160-bit composites
    # are ~2.2× higher than Dickman model predicts.  Likely from our Z/6 Edwards
    # parameterization having extra smoothness beyond exp(3.134).
    EMPIRICAL_BOOST = 2.2
    return max((p1 + p2) * EMPIRICAL_BOOST, 1e-30)


def parse_csv(lines):
    """Parse calibration CSV into dict: {(type, N): [(param, ms), ...]}"""
    data = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('type'):
            continue
        parts = line.split(',')
        typ = parts[0]
        N = int(parts[1])
        param = int(parts[2])
        ms = float(parts[3])
        key = (typ, N)
        if key not in data:
            data[key] = []
        data[key].append((param, ms))
    return data


def fit_linear(data, typ, N):
    """Fit cost = alpha * param (linear through origin). Returns alpha."""
    key = (typ, N)
    if key not in data:
        return None
    pts = sorted(data[key])
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    # Least-squares fit: y = alpha * x
    alpha = np.sum(x * y) / np.sum(x * x)
    return alpha


def fit_fft(data, typ, N):
    """Fit cost = gamma * sqrt(B2) * log2(B2)^2.

    Known asymptotic: FFT polyeval is O(sqrt(B2) * log^2(B2) * coeff_size).
    Only fit the constant gamma from data.
    """
    key = (typ, N)
    if key not in data:
        return None
    pts = sorted(data[key])
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    # model: y = gamma * sqrt(x) * log2(x)^2
    basis = np.sqrt(x) * np.log2(x)**2
    gamma = np.sum(basis * y) / np.sum(basis * basis)
    return gamma


def optimize_for_factor(factor_bits, alpha_s1, alpha_bsgs, gamma_fft, rho_func):
    """Find optimal (B1, B2) via scipy.optimize.minimize.

    Cost models (fitted):
      stage 1:      T_s1(B1) = alpha_s1 * B1
      stage 2 BSGS: T_s2(B2) = alpha_bsgs * B2
      stage 2 FFT:  T_s2(B2) = gamma_fft * sqrt(B2) * log2(B2)^2

    Minimize E[cost] = (T_s1 + T_s2) / P(B1, B2, p) over (log B1, log B2).
    """
    from scipy.optimize import minimize

    p = 2.0 ** factor_bits

    def objective(x):
        log_b1, log_b2 = x
        B1 = math.exp(log_b1)
        B2 = math.exp(log_b2)
        if B2 <= B1 * 2:
            return 1e30
        t_s1 = alpha_s1 * B1
        t_bsgs = alpha_bsgs * B2
        t_fft = gamma_fft * math.sqrt(B2) * math.log2(B2)**2
        t_s2 = min(t_bsgs, t_fft)
        prob = ecm_prob(B1, B2, p, rho_func)
        if prob <= 1e-30:
            return 1e30
        return (t_s1 + t_s2) / prob

    # Try multiple starting points to avoid local minima
    best_cost = float('inf')
    best = None
    # Seed near the expected optimum: B1 ~ 2^(factor_bits/3), B2 ~ B1*100
    log_b1_est = factor_bits * 0.23  # rough ln(B1) estimate
    for log_b1_init in [log_b1_est - 2, log_b1_est, log_b1_est + 2]:
        for log_b2_init in [log_b1_init + 4, log_b1_init + 7, log_b1_init + 10]:
            try:
                res = minimize(objective, [log_b1_init, log_b2_init],
                               method='Nelder-Mead',
                               options={'xatol': 0.05, 'fatol': 1e-6,
                                        'maxiter': 500})
                if res.fun < best_cost:
                    best_cost = res.fun
                    B1 = math.exp(res.x[0])
                    B2 = math.exp(res.x[1])
                    t_bsgs = alpha_bsgs * B2
                    t_fft = gamma_fft * math.sqrt(B2) * math.log2(B2)**2
                    method = 'bsgs' if t_bsgs <= t_fft else 'fft'
                    t_s1 = alpha_s1 * B1
                    t_s2 = min(t_bsgs, t_fft)
                    prob = ecm_prob(B1, B2, p, rho_func)
                    best = (int(B1), int(B2), method,
                            1.0/prob if prob > 0 else 1e18, t_s1 + t_s2)
            except Exception:
                pass

    return best


def main():
    lines = sys.stdin.readlines()
    data = parse_csv(lines)

    rho_func = make_rho_interp()

    print("# Dickman rho verification:")
    for u in [1, 2, 3, 4, 5, 6, 8, 10]:
        print(f"#   rho({u}) = {float(rho_func(u)):.2e}")
    print()

    for N in [2, 4]:
        # Fit cost models
        alpha_s1 = fit_linear(data, 's1', N)
        alpha_bsgs = fit_linear(data, 's2_bsgs', N)
        gamma_fft = fit_fft(data, 's2_fft', N)

        if alpha_s1 is None:
            print(f"# No data for N={N}, skipping")
            continue

        if gamma_fft is None:
            gamma_fft = 0

        print(f"# === N={N} ===")
        print(f"#   T_s1  = {alpha_s1:.4e} * B1                    (ms)")
        print(f"#   T_bsgs= {alpha_bsgs:.4e} * B2                    (ms)")
        print(f"#   T_fft = {gamma_fft:.4e} * sqrt(B2) * log2(B2)^2  (ms)")
        print(f"#")
        print(f"# {'fbits':>5} {'B1':>10} {'B2':>14} {'method':>6} {'curves':>8} {'ms/c':>8} {'total_s':>10}")

        for factor_bits in range(20, 205, 5):
            result = optimize_for_factor(factor_bits, alpha_s1, alpha_bsgs,
                                         gamma_fft, rho_func)
            if result is None:
                continue
            B1, B2, method, exp_curves, ms_c = result
            total_s = ms_c * exp_curves / 1000.0
            print(f"  {factor_bits:5d} {B1:10d} {B2:14d} {method:>6} {exp_curves:8.0f} {ms_c:8.2f} {total_s:10.1f}")

        print()


if __name__ == "__main__":
    main()
