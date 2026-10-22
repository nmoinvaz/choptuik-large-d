#!/usr/bin/env python3
"""CSS background of the HE kappa-model in D = n + 2: centre shooting (for initial guesses and an independent
check) and the collocation solve of css_model.

    python3 css_bg.py [kappa] [N]       D = 4 validation against the published values

Centre shooting uses the second-order form (css_equations.py, part D, checked against HE97 (22b-c)):
    Delta f'' = q(-2 i om x + 2x - (n-1) u x - (b^2/x)(n + (n-1)u) - 4 i kappa om x |f|^2/rho)
                - f(om^2 + i om (1 - (n-1) u)) - (2 kappa fbar/rho)(Delta q^2 + om^2 f^2),
    b' = (n-1) b u / x,   u = 2 x h/(n(n-1)) [x(|p|^2 + |q|^2) + 2 b Re(p qbar)],   p = (i om f - x q)/b,   Delta = b^2 - x^2.
"""
import sys, os
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_model as cm


def ode_rhs(x, y, n, kap, om):
    f = y[0] + 1j * y[1]; q = y[2] + 1j * y[3]; b = y[4]
    rho = 1 - kap * abs(f) ** 2
    h = 1 / rho ** 2
    p = (1j * om * f - x * q) / b
    u = 2 * x * h / (n * (n - 1)) * (x * (abs(p) ** 2 + abs(q) ** 2) + 2 * b * (p * np.conj(q)).real)
    Dl = b * b - x * x
    rhs = q * (-2j * om * x + 2 * x - (n - 1) * u * x - b * b / x * (n + (n - 1) * u) - 4j * kap * om * x * abs(f) ** 2 / rho) \
        - f * (om ** 2 + 1j * om * (1 - (n - 1) * u)) - 2 * kap * np.conj(f) / rho * (Dl * q * q + om ** 2 * f * f)
    fpp = rhs / Dl
    bp = (n - 1) * b * u / x
    return [q.real, q.imag, fpp.real, fpp.imag, bp]


def centre_start(n, kap, om, f0, x0):
    f0 = complex(f0)
    rho0 = 1 - kap * abs(f0) ** 2
    h0 = 1 / rho0 ** 2
    f2 = -f0 * (om ** 2 + 1j * om + 2 * kap * om ** 2 * abs(f0) ** 2 / rho0) / (2 * (n + 1))
    u2 = 2 * h0 / (n * (n - 1)) * (om ** 2 * abs(f0) ** 2 + 4 * (1j * om * f0 * np.conj(f2)).real)
    b2 = (n - 1) * u2 / 2
    f = f0 + f2 * x0 ** 2; q = 2 * f2 * x0; b = 1 + b2 * x0 ** 2
    return [f.real, f.imag, q.real, q.imag, b]


def shoot(n, kap, om, f0, xmax, x0=1e-4, npts=400):
    y0 = centre_start(n, kap, om, f0, x0)
    # stop at the horizon b = x
    ev = lambda x, y, *a: y[4] - x
    ev.terminal = True
    sol = solve_ivp(ode_rhs, (x0, xmax), y0, args=(n, kap, om), method='DOP853', rtol=1e-12, atol=1e-14,
                    dense_output=True, events=ev)
    return sol


def guess_from_shooting(g, n, kap, om, f0, X, wcut=0.93):
    """Collocation initial guess: shoot from the centre to x = sqrt(wcut) X, extrapolate linearly beyond."""
    sol = shoot(n, kap, om, f0, X * np.sqrt(wcut) * 1.0000001)
    xs = X * np.sqrt(np.clip(g.w, 1e-12, wcut))
    xs = np.minimum(xs, sol.t[-1])
    Y = sol.sol(xs)
    f = Y[0] + 1j * Y[1]; q = Y[2] + 1j * Y[3]; b = Y[4]
    p = (1j * om * f - xs * q) / b
    h = 1 / (1 - kap * abs(f) ** 2) ** 2
    u = 2 * xs * h / (n * (n - 1)) * (xs * (abs(p) ** 2 + abs(q) ** 2) + 2 * b * (p * np.conj(q)).real)
    v = u / np.maximum(g.w, 1e-12)
    # centre value from the series: u = u2 x^2 + ..., v(0) = u2 X^2
    fc = complex(f0); rho0 = 1 - kap * abs(fc) ** 2
    f2 = -fc * (om ** 2 + 1j * om + 2 * kap * om ** 2 * abs(fc) ** 2 / rho0) / (2 * (n + 1))
    u2 = 2 / rho0 ** 2 / (n * (n - 1)) * (om ** 2 * abs(fc) ** 2 + 4 * (1j * om * fc * np.conj(f2)).real)
    v[0] = u2 * X ** 2
    P = -p
    # linear extrapolation in w beyond wcut
    hi = g.w > wcut
    if hi.any():
        i1 = np.where(~hi)[0][-1]; i0 = i1 - 3
        for arr in (f, P, v, b):
            slope = (arr[i1] - arr[i0]) / (g.w[i1] - g.w[i0])
            arr[hi] = arr[i1] + slope * (g.w[hi] - g.w[i1])
    Z = np.concatenate([f.real, f.imag, P.real, P.imag, v, b, [om, X]])
    return Z


def solve(N, n, kap, Y0, g0=None, verbose=False, tol=1e-11, A=0.0):
    g = cm.Grid(N, A)
    if g0 is not None and (g0.N != N or g0.A != A):
        Y0 = cm.regrid(g0, Y0, g)
    Y, res, it = cm.newton_bg(g, Y0, n, kap, tol=tol, verbose=verbose)
    return g, Y, res


def observables(g, Y, n, kap):
    d = cm.unpack(g, Y)
    f0 = d["sR"][0] + 1j * d["sI"][0]
    fh = d["sR"][-1] + 1j * d["sI"][-1]
    u = g.w * d["v"]
    return dict(omega=d["omega"], X=d["X"], f0=abs(f0), fh=abs(fh), uh=u[-1], b0=d["b"][0],
                Delta=2 * np.pi / d["omega"])


if __name__ == "__main__":
    kap = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n = 2
    if kap == 1.0:
        # Hamade-Horne-Stewart (gr-qc/9511024) eqs. (4.6): omega, z+, |f(0)|, |f(z+)|
        om, X, f0 = 1.17695272200, 2.60909347510, 0.892555411872
        ref = dict(omega=1.17695272200, X=2.60909347510, f0=0.892555411872, fh=0.364210875022)
    elif kap == 0.0:
        # Hirschmann-Eardley 1995a (gr-qc/9412066): omega, z2, P = p(0) (8 pi normalization), u(z2)
        om, X = 1.9154446, 5.0035380
        f0 = np.sqrt(4 * np.pi) * 0.67217263 / om      # p(0) = i om f(0), F = sqrt(4 pi) phi
        ref = dict(omega=1.9154446, X=5.0035380, f0=f0, uh=0.39707205)
    else:
        raise SystemExit("give kappa = 0 or 1 for the D = 4 validation")
    g = cm.Grid(N)
    Y0 = guess_from_shooting(g, n, kap, om, f0, X)
    g, Y, res = solve(N, n, kap, Y0, verbose=True)
    ob = observables(g, Y, n, kap)
    print("N = %d  residual %.2e" % (N, res))
    for k in ref:
        print("  %-6s computed %.12f   published %.12f   diff %.2e" % (k, ob[k], ref[k], ob[k] - ref[k]))
