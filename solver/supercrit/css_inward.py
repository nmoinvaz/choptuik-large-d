#!/usr/bin/env python3
"""Exact finite-n CSS equations integrated inward from the outer region of the supercritical branch toward beta^2 x^2 = 1.

CSS: d/dtau = 0 (the scalar field is psi = kappa tau + Psi(x); p, m, b, c depend on x only). In the variables of
expand.py (y = p - m, s = p + m, c) and h = 1/b (so (n-1)(b-1) = (n-1)(1-h)/h; h = 0 is the apparent horizon,
g^{rr} = |grad r|^2 = h):
    x h_x = (n-1)(1-h) - h P,                P = (y^2 + s^2)/2         (E2)
    x (ln c)_x = (n-1)(1-h)/h                                          (E1)
    [x^2, -x c; -x c, x^2] (y_x, s_x) = (c ((n-1)/h) s, c (u - 1) y),  u = (n-1)(1-h)/h      ((A), (B))
Independent variable lambda = ln x (decreasing). Initial data at z0 = beta^2 x0^2 - 1 from the leading order
(y = sqrt2 beta x/w, c = f w, u = beta^2 x^2/w^2, s = eps * s1), so the start is off the exact solution by O(1/n);
the inward direction damps the fast mode (rate ~ n c^2/(x(x^2-c^2)) per unit x, growing outward), so the solution
relaxes onto the slow manifold within O(1/n) in ln x. kappa_f = f beta < 1 keeps c < x (no self-similarity horizon) on
the whole interval; with kappa_f > 1 start inside the horizon x0 < x_lc = f/sqrt(f^2 beta^2 - 1).

Output per n: where the outer expansion stops tracking (|y/y_LO - 1| > 0.1), the minimum of h reached and where,
and how the integration ends (h -> 0, c -> 0, stiff blow-up).

    python3 css_inward.py [beta f z0]          (defaults 1, 0.8, 1.0)
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

beta, f, z0 = (float(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (1.0, 0.8, 1.0)
kap = f * beta


def rhs(lam, Z, n):
    y, s, h, lc = Z
    x = np.exp(lam)
    c = np.exp(lc)
    P = 0.5 * (y * y + s * s)
    u = (n - 1) * (1 - h) / h
    det = x * x * (x * x - c * c)
    a1, a2 = c * ((n - 1) / h) * s, c * (u - 1) * y
    yx = (x * x * a1 + x * c * a2) / det
    sx = (x * x * a2 + x * c * a1) / det
    return [x * yx, x * sx, (n - 1) * (1 - h) - h * P, u]


def lo(x, n):
    w = np.sqrt(beta ** 2 * x ** 2 - 1)
    y = np.sqrt(2) * beta * x / w
    u = beta ** 2 * x ** 2 / w ** 2
    s1 = -np.sqrt(2) * x ** 2 * beta / (f * w ** 4)
    return y, s1 / n, 1 / (1 + u / (n - 1)), f * w


def hzero(lam, Z, n):
    return Z[2] - 1e-6
hzero.terminal = True

print(f"beta = {beta}, f = {f}, kappa_f = f beta = {kap:.3f}, start z0 = {z0}")
x0 = np.sqrt(1 + z0) / beta
for n in (100, 400, 1600, 6400, 25600):
    y0, s0, h0, c0 = lo(x0, n)
    ev = [hzero]
    sol = solve_ivp(rhs, [np.log(x0), np.log(x0) - 3.0], [y0, s0, h0, np.log(c0)], args=(n,), method="Radau",
                    rtol=1e-10, atol=1e-13, events=ev, dense_output=False, max_step=0.002)
    lam = sol.t
    x = np.exp(lam)
    z = beta ** 2 * x ** 2 - 1
    y, s, h, lc = sol.y
    ok = z > 0
    ylo = np.where(ok, np.sqrt(2) * beta * x / np.sqrt(np.abs(z) + 1e-300), np.nan)
    dev = np.abs(y / ylo - 1)
    i_dev = np.argmax(dev > 0.1) if np.any(dev > 0.1) else None
    imin = int(np.argmin(h))
    end = sol.message if sol.status != 1 else "h reached 1e-6"
    zdev = z[i_dev] if i_dev is not None else float("nan")
    print(f"n = {n:6d}: outer stops tracking (|y/y_LO - 1| > 0.1) at z = {zdev:+.4e} (z sqrt(n) = {zdev * np.sqrt(n):+.3f});"
          f" min h = {h[imin]:.3e} at z = {z[imin]:+.4e} (z n = {z[imin] * n:+.3f}); c there = {np.exp(lc[imin]):.3e};"
          f" y there = {y[imin]:.3e}, s = {s[imin]:.3e}; end: {end} at z = {z[-1]:+.4e}", flush=True)
