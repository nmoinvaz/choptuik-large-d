#!/usr/bin/env python3
"""Substitute the truncated 1/n series of the supercritical branch (expand.py) into the exact equations.

Independent of expand.py's operator algebra: the series coefficients are turned into the physical fields
p = (s + y)/2, m = (s - y)/2, b = 1 + u/(n-1), c as functions of (x, tau) for an explicit beta(tau), f(tau), and the
residuals of the exact equations of hp/src/model.rs,
    E1  x c_x - (n-1) c (b-1)
    E2  x b_x - b (p^2+m^2) + (n-1) b (b-1)
    E3  x (x-c) p_x + x p_tau - G p + c p - (n/2) c (p+m)
    E4  x (x+c) m_x + x m_tau + G m - c m + (n/2) c (p+m)
are evaluated with numerical derivatives (mpmath, 30 digits). Truncated at order K (y, c, u through eps^K and
p + m through eps^(K+1)), every residual must fall like n^-(K+1); the script fits the exponent over n = 100..1600 and
exits 1 if any equation's exponent is above -(K + 1) + 0.3.

    python3 formal_check.py logs/expand_K2_zero_css.pkl css           # K = 2, beta = 1.3, f = 0.9
    python3 formal_check.py logs/expand_K1_free.pkl tau               # K = 1, beta = 1.7 + 0.4 sin tau, f = 0.8 + 0.3 cos 2 tau
    python3 formal_check.py logs/expand_K2_zero.pkl tau               # K = 2, same profile
    python3 formal_check.py logs/expand_K2_zero_css.pkl css --mutate=drop   # Y_K and C_K dropped: must FAIL
"""
import pickle
import sys
import mpmath as mp
import numpy as np
import sympy as sp

args = [a for a in sys.argv[1:] if not a.startswith("--")]
mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
path, prof = args[0], args[1]
mp.mp.dps = 30
with open(path, "rb") as fh:
    E = pickle.load(fh)
K = E["K"]
z, eps, Bd, Fd, Ad, Cd, Lz, L1 = E["syms"]
tau, x, e = sp.symbols("tau x e", positive=True)
if prof == "css":
    beta_t, f_t = sp.Rational(13, 10), sp.Rational(9, 10)
else:
    beta_t = sp.Rational(17, 10) + sp.Rational(2, 5) * sp.sin(tau)
    f_t = sp.Rational(4, 5) + sp.Rational(3, 10) * sp.cos(2 * tau)
sub = {Bd[j]: sp.diff(beta_t, tau, j) for j in range(len(Bd))}
sub.update({Fd[j]: sp.diff(f_t, tau, j) for j in range(len(Fd))})
for k in Ad:
    sub.update({a: 0 for a in Ad[k]})
    sub.update({a: 0 for a in Cd[k]})
zz = beta_t ** 2 * x ** 2 - 1


def phys(ex):
    ex = sp.sympify(ex).subs(sub).subs({Lz: sp.log(z), L1: sp.log(z + 1)}).subs(z, zz)
    return ex


Yser = [phys(E["Y"][j]) for j in range(K + 1)]
Cser = [phys(E["C"][j]) for j in range(K + 1)]
User = [phys(E["U"][j]) for j in range(K + 1)]
Sser = [phys(E["S"][j]) for j in range(1, K + 2)]
if mut == "drop":
    Yser[K] = 0
    Cser[K] = 0
w = sp.sqrt(zz)
y = sp.sqrt(2) * x / w * sum(e ** j * Yser[j] for j in range(K + 1))
c = w * sum(e ** j * Cser[j] for j in range(K + 1))
u = sum(e ** j * User[j] for j in range(K + 1))
s = sp.sqrt(2) * x ** 2 * sum(e ** (j + 1) * Sser[j] for j in range(K + 1))
F = {name: sp.lambdify((x, tau, e), ex, "mpmath") for name, ex in (("y", y), ("c", c), ("u", u), ("s", s))}


def fields(xv, tv, ev):
    yv, cv, uv, sv = (F[k](xv, tv, ev) for k in ("y", "c", "u", "s"))
    n = 1 / ev
    return (sv + yv) / 2, (sv - yv) / 2, 1 + uv / (n - 1), cv


def residuals(xv, tv, n):
    ev = mp.mpf(1) / n
    p, m, b, cv = fields(xv, tv, ev)
    dx = [mp.diff(lambda xx: fields(xx, tv, ev)[i], xv) for i in range(4)]
    dt = [mp.diff(lambda tt: fields(xv, tt, ev)[i], tv) for i in range(2)]
    G = (n - 1) * cv * (b - 1)
    E1 = xv * dx[3] - G
    E2 = xv * dx[2] - b * (p * p + m * m) + (n - 1) * b * (b - 1)
    E3 = xv * (xv - cv) * dx[0] + xv * dt[0] - G * p + cv * p - n / 2 * cv * (p + m)
    E4 = xv * (xv + cv) * dx[1] + xv * dt[1] + G * m - cv * m + n / 2 * cv * (p + m)
    return [abs(E1 / cv), abs(E2), abs(E3), abs(E4)]


pts = []
for tv in (mp.mpf("0.3"), mp.mpf("1.7")):
    bt = float(beta_t.subs(tau, tv)) if prof != "css" else float(beta_t)
    for zv in (0.5, 1.0, 3.0, 10.0):
        pts.append((mp.sqrt((1 + zv)) / bt, tv))
print(f"{path}: K = {K}, profile {prof}" + (f", mutation {mut}" if mut else "") + "; max over 8 points (z = 0.5..10):")
ns = [100, 200, 400, 800, 1600]
tab = []
for n in ns:
    worst = np.zeros(4)
    for xv, tv in pts:
        worst = np.maximum(worst, [float(r) for r in residuals(xv, tv, mp.mpf(n))])
    tab.append(worst)
    print(f"   n = {n:5d}: |E1/c| {worst[0]:.3e}  |E2| {worst[1]:.3e}  |E3| {worst[2]:.3e}  |E4| {worst[3]:.3e}"
          f"   x n^{K + 1}: " + " ".join(f"{v * n ** (K + 1):.3g}" for v in worst), flush=True)
tab = np.array(tab)
slopes = [np.polyfit(np.log(ns[1:]), np.log(tab[1:, i]), 1)[0] for i in range(4)]
print("   fitted exponents (n = 200..1600): " + " ".join(f"{sl:.3f}" for sl in slopes) + f"   (expected {-(K + 1)})")
ok = all(sl < -(K + 1) + 0.3 for sl in slopes)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
