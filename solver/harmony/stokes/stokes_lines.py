#!/usr/bin/env python3
"""The two-endpoint zero law of S1 and the physical zero (#357; the test is fixed in notes/STOKES_LINES.md §1-§4).

If the Borel transform of S1(x), x = A^2, has two algebraic endpoints zeta and kappa, the zeros of one arm obey
    Lam x_j + q ln x_j = L + 2 pi i j,   Lam = kappa - zeta.
The arm rule picks x0 (near 7.4), x1, x2, x3; (Lam, q, L) solve the law at j = 1, 2, 3 and the law at j = 0 predicts x0.
A function given by sigma_0..sigma_K is summed by Borel-Pade [m/(K-m)], m = ceil(0.35 K) .. floor(K/2) + 1
(zerolib), its zeros in |x| < 30 found on a grid (h = 0.2) and refined; the prediction is the median over the sums
in which the arm rule finds all four zeros, the spread their range.

planted mode: P1 (the #214 arc family at w = 1.10), P2 (a straight two-endpoint segment) and control C (P1 with its
x0 moved by d = 1.0), with exact zeros from 200-term series at 160 digits and coefficients to K with 1e-14 relative
noise, at K = 22, 30 and 40. Decisive at K: P1 and P2 both give |x0_hat - x0| <= 0.15 and C fires
(|x0_hat - (x0 + d)| > 3 e1). data mode (only when decisive at K = 22): sigma_0..sigma_22 at omega_inf.

    cd solver/harmony/stokes && python3 stokes_lines.py planted [--K=22,30,40] [--mutate=d0] [--out=FILE]
    cd solver/harmony/stokes && python3 stokes_lines.py data [--out=FILE]
    cd solver/harmony/stokes && python3 stokes_lines.py explore      (post hoc: the law's error from exact zeros)
--cache=FILE keeps the exact zeros of P1 and P2 between runs. --d=VALUE sets the control's displacement (fixed: 1.0).

Exit codes: planted 0 if decisive at K = 22, 3 if not; data 0 if H holds, 4 if it fails, 3 if not decisive;
--mutate=d0 (d = 0, so C must not fire) exits 1 if C still fires.
"""
import json
import math
import os
import sys

import mpmath as mp
import numpy as np

import zerolib as Z

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "independent_second_zero"))
import arc_family  # noqa: E402

mp.mp.dps = 60
NSER, SDPS = 200, 160
X0_DATA = mp.mpf("7.394077628079895")
DATA = os.path.join(HERE, "..", "..", "inner_amp", "scan45J1500", "sigF_w1.0000_K45_J1500_p1280.json")
TOL_E, D_CTRL, NOISE, R, H = 0.15, mp.mpf(1), 1e-14, 30.0, 0.2
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
MODE = sys.argv[1] if len(sys.argv) > 1 else "planted"
if args.get("mutate") == "d0":
    D_CTRL = mp.mpf(0)
if "d" in args:                                   # a larger displacement, to show that the control can fire
    D_CTRL = mp.mpf(args["d"])


# ---- functions given by Taylor coefficients c_n (S = sum c_n x^n / n!) -----------------------------------------
class Series:
    def __init__(self, name, c):
        self.name, self.c = name, c
        self.cf = np.array([complex(v) for v in c])

    def sigma(self, K):
        return [self.c[n] / mp.factorial(n) for n in range(K + 1)]

    def f_mp(self, x):
        with mp.workdps(SDPS):
            x, s, t = mp.mpc(x), mp.mpc(0), mp.mpc(1)
            for n, cn in enumerate(self.c):
                s += cn * t
                t = t * x / (n + 1)
            return s

    def f_np(self, X):
        acc, t = np.zeros_like(X, dtype=complex), np.ones_like(X, dtype=complex)
        for n, cn in enumerate(self.cf):
            acc += cn * t
            t = t * X / (n + 1)
        return acc


def p1():
    c = arc_family.ArcBase().coeffs(mp.mpf("1.10"), NSER)
    return Series("P1 arc (w 1.10)", c)


P2_PAR = dict(zeta=mp.mpc("0.749", "0.155"), kappa=mp.mpc("1.703", "1.127"), alpha=mp.mpc("1.5", "0.5"),
              beta=mp.mpc("9.74", "-5.81"), g=mp.mpc("0.3", "-0.2"))


def p2():
    """moments of (t - zeta)^(alpha-1) (kappa - t)^(beta-2) (1 + g (t - zeta)) on the segment, by Beta functions"""
    with mp.workdps(SDPS):
        z, k, a, b, g = [P2_PAR[n] for n in ("zeta", "kappa", "alpha", "beta", "g")]
        lam = k - z
        Bj = [mp.beta(a + j, b - 1) + g * lam * mp.beta(a + j + 1, b - 1) for j in range(NSER + 1)]
        c = [mp.fsum(mp.binomial(n, j) * z ** (n - j) * lam ** j * Bj[j] for j in range(n + 1)) for n in range(NSER + 1)]
        c = [v / c[0] for v in c]
    return Series("P2 segment", c)


def p2_checks(s2):
    """two independent routes for P2: one moment by quadrature, and S at a point by Kummer functions"""
    z, k, a, b, g = [P2_PAR[n] for n in ("zeta", "kappa", "alpha", "beta", "g")]
    lam = k - z
    with mp.workdps(50):
        rho = lambda t: (t - z) ** (a - 1) * (k - t) ** (b - 2) * (1 + g * (t - z))
        m0 = mp.quad(lambda s: rho(z + s * lam) * lam, [0, 0.5, 1])
        m5 = mp.quad(lambda s: rho(z + s * lam) * (z + s * lam) ** 5 * lam, [0, 0.5, 1])
        q5 = m5 / m0
        x = mp.mpc(9, 2)
        kum = mp.exp(z * x) * (mp.beta(a, b - 1) * mp.hyp1f1(a, a + b - 1, lam * x)
                               + g * lam * mp.beta(a + 1, b - 1) * mp.hyp1f1(a + 1, a + b, lam * x))
        kum0 = mp.beta(a, b - 1) + g * lam * mp.beta(a + 1, b - 1)
        return abs(q5 / s2.c[5] - 1), abs(kum / kum0 / s2.f_mp(x) - 1)


def displaced(base, x0, d):
    """base * (x - x0 - d)/(x - x0), coefficients in the monomial basis then back to c_n = n! a_n"""
    with mp.workdps(SDPS):
        f = [cn / mp.factorial(n) for n, cn in enumerate(base.c)]
        gco, prev = [], mp.mpc(0)
        for fn in f:                                  # f = (x - x0) g  ->  g_n = (g_{n-1} - f_n)/x0
            prev = (prev - fn) / x0
            gco.append(prev)
        a = x0 + d
        h = [(gco[n - 1] if n else 0) - a * gco[n] for n in range(len(gco))]
        c = [h[n] * mp.factorial(n) for n in range(len(h))]
        c = [v / c[0] for v in c]
    return Series(base.name + f" moved by d = {mp.nstr(d, 3)}", c)


# ---- zeros, arm rule, law -------------------------------------------------------------------------------------
def zeros_exact(s):
    """exact zeros in |x| < R; cached in --cache=FILE (strings at 60 digits) since the search takes minutes"""
    path = args.get("cache")
    store = json.load(open(path)) if path and os.path.exists(path) else {}
    if s.name in store:
        return [mp.mpc(*v) for v in store[s.name]]
    zs, _ = Z.zeros_in_disc(s.f_np, s.f_mp, R=R, h=H, tol=mp.mpf(10) ** -40)
    if path:
        store[s.name] = [[mp.nstr(z.real, 60), mp.nstr(z.imag, 60)] for z in zs]
        json.dump(store, open(path, "w"), indent=1)
    return zs


def arm_long(zs, n=6):
    """the arm continued past x3 by the same rule (exploration only)"""
    xs = arm(zs)
    while xs is not None and len(xs) < n:
        c = [z for z in zs if z.real > xs[-1].real + 2]
        if not c:
            break
        xs.append(min(c, key=lambda z: abs(z - (2 * xs[-1] - xs[-2]))))
    return xs


def law_fit(pts, js, inv=False):
    """(Lam, q, L[, c]) from the law at the zeros pts with labels js; inv adds a term c/x"""
    rows = [[x, mp.log(x), -1] + ([1 / x] if inv else []) for x in pts]
    sol = mp.lu_solve(mp.matrix(rows), mp.matrix([2j * mp.pi * j for j in js]))
    return list(sol)


def law_solve(par, j, x):
    for _ in range(200):
        F = par[0] * x + par[1] * mp.log(x) - par[2] + (par[3] / x if len(par) > 3 else 0) - 2j * mp.pi * j
        dF = par[0] + par[1] / x - (par[3] / x ** 2 if len(par) > 3 else 0)
        dx = F / dF
        x -= dx
        if abs(dx) < mp.mpf(10) ** -30:
            break
    return x


def explore(name, zs):
    """post hoc (not part of the fixed test): the law's error at x0 from exact zeros farther out"""
    xs = arm_long(zs)
    print(f"  {name}: exact arm " + ", ".join(Z.cnum(x, 3) for x in xs))
    res = []
    for lo in range(1, len(xs) - 2):
        par = law_fit(xs[lo:lo + 3], [lo, lo + 1, lo + 2])
        x0 = law_solve(par, 0, xs[1] - (xs[2] - xs[1]))
        res.append((f"3-term law on x{lo}..x{lo + 2}", float(abs(x0 - xs[0])), par))
    for lo in range(1, len(xs) - 3):
        par = law_fit(xs[lo:lo + 4], [lo, lo + 1, lo + 2, lo + 3], inv=True)
        x0 = law_solve(par, 0, xs[1] - (xs[2] - xs[1]))
        res.append((f"law + c/x on x{lo}..x{lo + 3}", float(abs(x0 - xs[0])), par))
    for lab, e, par in res:
        print(f"    {lab}: |x0_hat - x0| = {e:.3g}; Lam = {Z.cnum(par[0], 4)}, q = {Z.cnum(par[1], 3)}")
    return res


def arm(zs):
    def pick(cands, target):
        return min(cands, key=lambda z: abs(z - target)) if cands else None
    c1 = [z for z in zs if 10 < z.real < 16]
    if not c1:
        return None
    x1 = min(c1, key=lambda z: abs(z.imag))
    x0 = pick([z for z in zs if 4 < z.real < x1.real - 2], x1 - mp.mpf("5.4"))
    if x0 is None:
        return None
    x2 = pick([z for z in zs if z.real > x1.real + 2], 2 * x1 - x0)
    if x2 is None:
        return None
    x3 = pick([z for z in zs if z.real > x2.real + 2], 2 * x2 - x1)
    if x3 is None:
        return None
    return [x0, x1, x2, x3]


def law_predict(xs):
    x1, x2, x3 = xs[1:]
    M = mp.matrix([[x, mp.log(x), -1] for x in (x1, x2, x3)])
    rhs = mp.matrix([2j * mp.pi * j for j in (1, 2, 3)])
    lam, q, L = mp.lu_solve(M, rhs)
    x = x1 - (x2 - x1)
    for _ in range(100):
        F = lam * x + q * mp.log(x) - L
        dx = F / (lam + q / x)
        x -= dx
        if abs(dx) < mp.mpf(10) ** -30:
            break
    return x, (lam, q, L)


def bp_predict(sig, K):
    rows = []
    for m in range(math.ceil(0.35 * K), K // 2 + 2):
        ts, rs, c0 = Z.borel_pade(sig, m, K - m)
        zs, _ = Z.zeros_in_disc(Z.bp_numpy(ts, rs, c0), Z.bp_func(ts, rs, c0), R=R, h=H)
        xs = arm(zs)
        if xs is None:
            rows.append((m, None, None, None))
            continue
        xh, par = law_predict(xs)
        rows.append((m, xs, xh, par))
    good = [r for r in rows if r[2] is not None]
    if not good:
        return None, None, rows
    re = sorted(float(r[2].real) for r in good)
    im = sorted(float(r[2].imag) for r in good)
    med = mp.mpc(np.median(re), np.median(im))
    spread = max(abs(r[2] - med) for r in good)
    return med, spread, rows


def noisy(sig, seed):
    rng = np.random.default_rng(seed)
    return [s * (1 + NOISE * mp.mpc(*rng.standard_normal(2))) for s in sig]


def report(name, xs_true, K, med, spread, rows):
    print(f"  {name}, K = {K}:")
    for m, xs, xh, par in rows:
        if xs is None:
            print(f"    [{m}/{K - m}] arm incomplete")
        else:
            print(f"    [{m}/{K - m}] x0..x3 = " + ", ".join(Z.cnum(x, 3) for x in xs) + f"; law x0 = {Z.cnum(xh, 4)}"
                  f"; Lam = {Z.cnum(par[0], 3)}, q = {Z.cnum(par[1], 3)}")
    if med is None:
        print("    no sum gives the full arm")
        return None
    e = float(abs(med - xs_true[0]))
    print(f"    median x0_hat = {Z.cnum(med, 4)}, spread {float(spread):.3g}; true x0 = {Z.cnum(xs_true[0], 5)}; e = {e:.3g}")
    return e


def run_planted(Ks):
    s1, s2 = p1(), p2()
    q5, kum = p2_checks(s2)
    print(f"P2 checks: moment 5 by quadrature, rel diff {mp.nstr(q5, 3)}; S(9+2i) by Kummer, rel diff {mp.nstr(kum, 3)}")
    out = {"K": {}, "p2_checks": [str(q5), str(kum)], "d": str(D_CTRL)}
    truth = {}
    for s in (s1, s2):
        zs = zeros_exact(s)
        xs = arm(zs)
        truth[s.name] = xs
        print(f"{s.name}: exact arm x0..x3 = " + ", ".join(Z.cnum(x, 5) for x in xs))
        xh, par = law_predict(xs)
        print(f"   law on the exact x1..x3: x0_hat = {Z.cnum(xh, 5)}, |x0_hat - x0| = {float(abs(xh - xs[0])):.3g}"
              f"; Lam = {Z.cnum(par[0], 4)}, q = {Z.cnum(par[1], 4)}, L = {Z.cnum(par[2], 4)}")
        out[s.name + " exact"] = {"arm": [str(x) for x in xs], "x0_hat": str(xh), "e_exact": float(abs(xh - xs[0]))}
    sc = displaced(s1, truth[s1.name][0], D_CTRL)
    xs_c = list(truth[s1.name])
    xs_c[0] = truth[s1.name][0] + D_CTRL
    decisive = {}
    for K in Ks:
        print(f"\nK = {K}")
        res = {}
        for s, xs, seed in ((s1, truth[s1.name], 1), (s2, truth[s2.name], 2), (sc, xs_c, 3)):
            med, spread, rows = bp_predict(noisy(s.sigma(K), seed + 10 * K), K)
            e = report(s.name, xs, K, med, spread, rows)
            res[s.name] = {"x0_hat": str(med), "spread": float(spread) if spread is not None else None, "e": e}
        e1, e2 = res[s1.name]["e"], res[s2.name]["e"]
        ec = res[sc.name]["e"]
        fires = e1 is not None and ec is not None and ec > 3 * e1
        dec = e1 is not None and e2 is not None and e1 <= TOL_E and e2 <= TOL_E and fires
        decisive[K] = dec
        print(f"  verdict K = {K}: e1 = {e1}, e2 = {e2}, control |x0_hat - (x0 + d)| = {ec} "
              f"({'fires' if fires else 'does not fire'}); decisive: {dec}")
        res["fires"], res["decisive"] = fires, dec
        out["K"][str(K)] = res
    return out, decisive


def run_data():
    d = json.load(open(DATA))
    sig = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["sigma_str"]]
    K = len(sig) - 1
    med, spread, rows = bp_predict(sig, K)
    e = report("data, omega_inf", [mp.mpc(X0_DATA)], K, med, spread, rows)
    return {"K": K, "x0_hat": str(med), "spread": float(spread) if spread is not None else None, "e": e}, e, spread


if __name__ == "__main__":
    if MODE == "planted":
        Ks = [int(k) for k in args.get("K", "22,30,40").split(",")]
        out, dec = run_planted(Ks)
        if "out" in args:
            json.dump(out, open(args["out"], "w"), indent=1)
        if args.get("mutate") == "d0":
            fired = any(out["K"][str(K)]["fires"] for K in Ks)
            print(f"\n--mutate=d0: the control {'fires (wrong)' if fired else 'does not fire (right)'}")
            sys.exit(1 if fired else 0)
        sys.exit(0 if dec.get(22) else 3)
    elif MODE == "explore":
        for s in (p1(), p2()):
            explore(s.name, zeros_exact(s))
        sys.exit(0)
    elif MODE == "data":
        out, dec = run_planted([22])
        if not dec[22]:
            print("\nnot decisive at K = 22: the data run is not made (STOKES_LINES.md §4, item 2)")
            sys.exit(3)
        res, e, spread = run_data()
        e1 = out["K"]["22"]["P1 arc (w 1.10)"]["e"]
        e2 = out["K"]["22"]["P2 segment"]["e"]
        bound = 3 * max(e1, e2, float(spread))
        print(f"\nH: e_data = {e} against 3 max(e1, e2, spread) = {bound}: {'holds' if e <= bound else 'fails'}")
        if "out" in args:
            json.dump({"planted": out, "data": res, "bound": bound}, open(args["out"], "w"), indent=1)
        sys.exit(0 if e <= bound else 4)
