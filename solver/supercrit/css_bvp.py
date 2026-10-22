#!/usr/bin/env python3
"""Finite-n check of the layer (layer.py): the exact CSS equations solved as a boundary-value problem across beta^2 x^2 = 1.

Inward integration of the exact equations is ill-posed (css_inward.py fails at once): the b equation has a mode growing
like x^-(n-1) inward and the p + m equation one growing outward. So solve on lambda = ln x in [lambda_a, lambda_b] with
    at lambda_b (outer region, z_b = beta^2 x_b^2 - 1 of order 1): y, c and s = p + m from the outer series through NLO
        (expand.py, CSS, A_1 = B_1 = 0)   -- two slow modes and the outward-growing mode;
    at lambda_a (behind the front):  h = 1/b equal to its slaved value (n-1)/(n-1+P)  -- the inward-growing mode.
Equations: exact (css_inward.py's rhs: E1, E2 in h = 1/b, (A), (B) solved for y_x, s_x), no large-n approximation.
Shared with layer.py: only the field equations. The comparison is with the layer's prediction (CSS, kappa_f = f beta):
    y = n^{1/4} sqrt(kappa_f) Y1(theta),  c = n^{-1/4} sqrt(f/beta) C1(theta),  p + m = kappa_f (C1 Y1 - sqrt2),
    ln x - ln(1/beta) = n^{-1/2} theta / kappa_f,
and behind the front Y1 -> 1.8038..., so (n-1)(b-1) -> 1.627 kappa_f sqrt(n): 1 - 1/b ~ 1.63 kappa_f n^{-1/2}.

    python3 css_bvp.py [beta f]              (defaults 1, 0.8)
    python3 css_bvp.py 1.5 1.0               # kappa_f = 1.5 > 1 (a self-similarity horizon lies outside the interval)
    python3 css_bvp.py 1 0.8 --mutate=cubic  # compare with a wrong layer (u = y^2/4 in the lapse equation): must FAIL
Pass: the deviation from the layer falls with n (log-log slope < -0.15) and is below 0.06 at n = 6400.
"""
import pickle
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp, solve_ivp

args = [a for a in sys.argv[1:] if not a.startswith("--")]
mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
beta, f = (float(args[0]), float(args[1])) if len(args) >= 2 else (1.0, 0.8)
kap = f * beta
lam0 = np.log(1 / beta)

# ---- outer series through NLO (CSS), from expand.py's free run with A_1 = B_1 = 0
with open("logs/expand_K1_free.pkl", "rb") as fh:
    E = pickle.load(fh)
z, eps, Bd, Fd, Ad, Cd, Lz, L1 = E["syms"]
sub = {Bd[0]: beta, Fd[0]: f}
sub.update({Bd[j]: 0 for j in range(1, len(Bd))})
sub.update({Fd[j]: 0 for j in range(1, len(Fd))})
for k in Ad:
    sub.update({a: 0 for a in Ad[k]})
    sub.update({a: 0 for a in Cd[k]})
lam_ = lambda e: sp.lambdify(z, sp.sympify(e).subs(sub).subs({Lz: sp.log(z), L1: sp.log(z + 1)}), "numpy")
Yh = [lam_(E["Y"][0]), lam_(E["Y"][1])]
Ch = [lam_(E["C"][0]), lam_(E["C"][1])]
Sh = [lam_(E["S"][1]), lam_(E["S"][2])]


def outer(x, n):
    zz = beta ** 2 * x ** 2 - 1
    w = np.sqrt(zz)
    e = 1.0 / n
    y = np.sqrt(2) * x / w * (Yh[0](zz) + e * Yh[1](zz))
    c = w * (Ch[0](zz) + e * Ch[1](zz))
    s = np.sqrt(2) * x ** 2 * (e * Sh[0](zz) + e * e * Sh[1](zz))
    return y, s, c


# ---- universal layer solution (layer.py, reduced with the first integral S1 = C1 Y1 - sqrt2)
def lay_rhs(th, Z):
    Y, C = Z
    return [C * (C * Y - np.sqrt(2)), C * Y * Y / (4 if mut == "cubic" else 2)]


th_hi = 60.0
lay = solve_ivp(lay_rhs, [th_hi, -12.0], [th_hi ** -0.5, np.sqrt(2 * th_hi)], method="LSODA", rtol=1e-12,
                atol=1e-14, dense_output=True)
Yinf = lay.y[0, -1]
print(f"layer: Y1(-12) = {Yinf:.6f}, C1(-12) = {lay.y[1, -1]:.3e}   (beta = {beta}, f = {f}, kappa_f = {kap})")


def layer_pred(lam, n):
    th = (lam - lam0) * np.sqrt(n) * kap
    Y1, C1 = lay.sol(np.clip(th, -12.0, th_hi))
    return n ** 0.25 * np.sqrt(kap) * Y1, kap * (C1 * Y1 - np.sqrt(2)), n ** -0.25 * np.sqrt(f / beta) * C1, th


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
    return np.vstack([x * yx, x * sx, (n - 1) * (1 - h) - h * P, u])


ok = True
rows = []
for n in (100, 400, 1600, 6400):
    zb = 0.6
    lam_b = np.log(np.sqrt(1 + zb) / beta)
    lam_a = lam0 - 7.0 / (kap * np.sqrt(n))          # theta_a = -7
    yb, sb, cb = outer(np.exp(lam_b), n)
    lam = np.linspace(lam_a, lam_b, 4000)
    yg, sg, cg, th = layer_pred(lam, n)
    hg = (n - 1) / (n - 1 + 0.5 * (yg ** 2 + sg ** 2))
    guess = np.vstack([yg, sg, hg, np.log(cg)])

    def bc(Za, Zb):
        ya, sa, ha, _ = Za
        return np.array([Zb[0] - yb, Zb[1] - sb, Zb[3] - np.log(cb),
                         ha - (n - 1) / (n - 1 + 0.5 * (ya * ya + sa * sa))])

    sol = solve_bvp(lambda l, Z: rhs(l, Z, n), bc, lam, guess, tol=1e-8, max_nodes=400000)
    if sol.status != 0:
        print(f"n = {n}: solve_bvp: {sol.message}")
        ok = False
        continue
    ll = np.linspace(lam_a + 1.0 / (kap * np.sqrt(n)), lam_b, 3000)
    y, s, h, lc = sol.sol(ll)
    yp, sp_, cp, thp = layer_pred(ll, n)
    inl = (thp > -5) & (thp < 5)
    dy = np.max(np.abs(y[inl] / yp[inl] - 1))
    dc = np.max(np.abs(np.exp(lc[inl]) / cp[inl] - 1))
    ds = np.max(np.abs(s[inl] - sp_[inl])) / kap
    # behind the front
    Ya, Sa, ha, lca = sol.sol(lam_a + 1.0 / (kap * np.sqrt(n)))
    th_in = -6.0
    yin_pred = n ** 0.25 * np.sqrt(kap) * lay.sol(th_in)[0]
    uin = (n - 1) * (1 - ha) / ha
    rows.append((n, dy, dc, ds, Ya / yin_pred, uin / (kap * np.sqrt(n)), 1 - ha, np.min(h)))
    print(f"n = {n:5d}: |theta|<5: max|y/y_layer - 1| = {dy:.3e}, max|c/c_layer - 1| = {dc:.3e}, "
          f"max|s - s_layer|/kappa = {ds:.3e}; at theta = -6: y/y_layer = {Ya / yin_pred:.4f}, "
          f"(n-1)(b-1)/(kappa sqrt n) = {uin / (kap * np.sqrt(n)):.4f}, 1 - 1/b = {1 - ha:.4f}, "
          f"min h on the interval = {np.min(h):.4f}, c = {np.exp(lca):.2e}", flush=True)
if len(rows) >= 3:
    r = np.array(rows)
    slope = np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)[0]
    print(f"max|y/y_layer - 1| falls like n^{slope:.3f}  (layer corrections expected at relative n^(-1/4))")
    ok &= bool(slope < -0.15 and r[-1, 1] < 0.06)
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
