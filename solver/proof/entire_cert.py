#!/usr/bin/env python3
"""The amplitude series of the harmonic-1 multiplier as an entire function of x = A^2, and what a root certificate
for S_1(A, w) = 0 would need (proof/D_entire.md). NOT itself rigorous: it uses the fitted sigma_k (ball radii ~1e-20,
but the late-order fit is not an enclosure) and estimates, not bounds, for the tail.

S_1/S_1^(1) = sum_k sigma_k x^k (x = A^2), sigma_k from scan25phys (K = 25, so sigma_0..sigma_12 at w/w_inf = 0.99,
1.00, 1.01). Printed:
  1. the ratio law sigma_k/sigma_{k+1} ~ (k + beta)/kappa_*: order 1, type |kappa_*| in x (an entire function of
     exponential type), from the ratio increments and a complex linear fit;
  2. Prony fits of a_k = k! sigma_k = sum_i c_i kappa_i^k: the exponents kappa_i do not settle as terms are added
     but spread along an arc (a continuous Laplace density, S_1 = int rho(kappa) e^{kappa x} dkappa, i.e. the
     Borel transform's singular support);
  3. the resummed root: G = e^{-kbar x} S_1 has the same zeros and, for kbar inside the arc, coefficients that are
     O(1) at x0 ~ 7.4 and decay (instead of terms ~240 that cancel); kbar chosen to minimise the last two terms of
     G_K at x0. Zeros of G_K across w give the w where the zero is real, per K;
  4. the other zeros of G_K (the next one is at |x| > 10 and not converged);
  5. the margins a Rouche (disc in x) and a Krawczyk/Miranda (box in (A, w/w_inf)) certificate would need;
  6. the type bound |sigma_k| <= C |kappa_*|^k / Gamma(k + Re beta) calibrated on the computed orders, the same for
     G's coefficients with the shifted arc, and the tails they predict at x0 beyond K against the Rouche margins
     (the hypothesis of the conditional Theorem A(inf) of proof/P5_scope.md; a calibration, not a bound).

    python3 entire_cert.py [K] [--scan DIR]        (default K 12 on scan25phys; --scan ../inner_amp/scan35J1000 17
                                                    runs the seventeen-order scans)
"""
import glob
import json
import os
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "..", "inner_amp", "scan25phys")
WFS = (0.99, 1.0, 1.01)
X0 = 7.397                      # x0 = A0^2 at w_inf, A0 ~ 2.7196


def load(wf):
    for fn in glob.glob(os.path.join(SCAN, "*.json")):
        r = json.load(open(fn))
        if abs(r["wf"] - wf) < 1e-9:
            return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]]
    raise FileNotFoundError(wf)


def gcoef(sig, kap, K):
    """Taylor coefficients of G = e^{-kap x} sum sigma_k x^k, through x^K"""
    return [sum(sig[m] * (-kap) ** (k - m) / mp.factorial(k - m) for m in range(k + 1)) for k in range(K + 1)]


def poly(c):
    return lambda x: sum(ci * x ** i for i, ci in enumerate(c))


def ratio_law(S):
    print("1. ratio law sigma_k/sigma_{k+1} = (k + beta)/kappa_*  (entire, order 1, type |kappa_*| in x)")
    for wf in WFS:
        sig = S[wf]
        K = len(sig) - 1
        rat = [complex(sig[k] / sig[k + 1]) for k in range(K)]
        inc = np.diff(rat)
        ks = np.arange(6, K)
        A = np.vstack([ks, np.ones_like(ks)]).T.astype(complex)
        c = np.linalg.lstsq(A, np.array(rat[6:K]), rcond=None)[0]
        kap = 1 / c[0]
        print(f"   w/w_inf {wf}: increments k = 7..{K - 1}: " + " ".join(f"{z:.4f}" for z in inc[6:])
              + f"\n      fit k = 6..{K - 1}: kappa_* = {kap:.4f} (|kappa_*| {abs(kap):.4f}, arg {np.angle(kap):.4f}),"
              f" beta = {c[1] * kap:.3f}")


def prony(S, ms=(3, 4, 5)):
    """returns the exponents of the largest fit (points along the arc)"""
    print("2. Prony: a_k = k! sigma_k = sum_i c_i kappa_i^k (w = w_inf); exponents spread along an arc")
    sig = S[1.0]
    K = len(sig) - 1
    a = [sig[k] * mp.factorial(k) for k in range(K + 1)]
    for m in ms:
        rows = [[a[k + i] for i in range(m)] for k in range(K - m + 1)]
        p, _ = mp.qr_solve(mp.matrix(rows), mp.matrix([-a[k + m] for k in range(K - m + 1)]))
        roots = mp.polyroots([mp.mpc(1)] + [p[m - 1 - i] for i in range(m)], maxsteps=200, extraprec=200)
        V = mp.matrix([[r ** k for r in roots] for k in range(K + 1)])
        c, res = mp.qr_solve(V, mp.matrix(a))
        print(f"   m {m}: rel. resid {mp.nstr(res / mp.norm(mp.matrix(a)), 2)}; kappa_i = "
              + ", ".join(mp.nstr(r, 4) for r in sorted(roots, key=lambda z: abs(z))))
    return roots


def ratio_fit(sig, K):
    """kappa_* and beta of the ratio law from a complex linear fit over k = 6..K-1"""
    rat = [complex(sig[k] / sig[k + 1]) for k in range(K)]
    ks = np.arange(6, K)
    A = np.vstack([ks, np.ones_like(ks)]).T.astype(complex)
    c = np.linalg.lstsq(A, np.array(rat[6:K]), rcond=None)[0]
    kap = 1 / c[0]
    return kap, c[1] * kap


def type_bound(S, K, kb, arc):
    """the type bound calibrated on k <= K and the tail it predicts beyond K at x0, in the S_1 and the G form"""
    print("6. type bound calibrated on the computed orders (a hypothesis, not a bound), and its tail at x0")
    sig = S[1.0]
    kap, beta = ratio_fit(sig, K)
    a, b = abs(kap), beta.real
    Ck = [float(abs(sig[k]) * mp.gamma(k + b) / a ** k) for k in range(4, K + 1)]
    C = max(Ck)
    print(f"   S_1 form, |sigma_k| Gamma(k + {b:.2f}) / {a:.4f}^k for k = 4..{K}: "
          + " ".join(f"{v:.1f}" for v in Ck) + f"; C = max = {C:.1f}")
    print(f"   its prediction for the next orders, |sigma_k| <= C a^k / Gamma(k + b): "
          + ", ".join(f"k {k}: {mp.nstr(C * mp.mpf(a) ** k / mp.gamma(k + b), 3)}" for k in (K + 1, K + 2)))
    print("   the tail sum_(k>K) C a^k x0^k / Gamma(k + b) at x0, for the truncation orders K:")
    ax = mp.mpf(a) * X0
    for Kt in sorted({12, 14, K}):
        if Kt <= K:
            tail = mp.fsum(C * ax ** k / mp.gamma(k + b) for k in range(Kt + 1, Kt + 400))
            print(f"      K {Kt}: {mp.nstr(tail, 3)}")
    # a bound calibrated on fewer orders, tested on the orders after them (the planted-truth question for P5: does a
    # per-coefficient bound fixed at one order hold at the next ones?)
    for Kc in (12, 14):
        if Kc < K:
            kc, bc = ratio_fit(sig, Kc)
            ac, bcr = abs(kc), bc.real
            Cc = max(float(abs(sig[k]) * mp.gamma(k + bcr) / ac ** k) for k in range(4, Kc + 1))
            ratios = [float(abs(sig[k]) / (Cc * ac ** k / mp.gamma(k + bcr))) for k in range(Kc + 1, K + 1)]
            print(f"   calibrated on k <= {Kc} (a {ac:.4f}, b {bcr:.2f}, C {Cc:.1f}): |sigma_k| / bound for"
                  f" k = {Kc + 1}..{K}: " + " ".join(f"{v:.3f}" for v in ratios)
                  + ("  (violated)" if max(ratios) > 1 else "  (holds)"))
    # G = e^{-kbar x} S_1 has the arc shifted by kbar; its type is the farthest arc point from kbar, taken over the
    # Prony exponents and kappa_*; C_G is calibrated on the computed g_k
    pts = list(arc) + [mp.mpc(kap)]
    r = max(float(abs(t - kb)) for t in pts)
    far = max(pts, key=lambda t: abs(t - kb))
    g = gcoef(sig, kb, K)
    CG = [float(abs(g[k]) * mp.factorial(k) / r ** k) for k in range(4, K + 1)]
    Cg = max(CG)
    print(f"   G form with kbar {mp.nstr(kb, 4)}: r = max |t - kbar| over the arc = {r:.3f} (at t = {mp.nstr(far, 4)});"
          f" |g_k| k! / r^k for k = 4..{K}: " + " ".join(f"{v:.2f}" for v in CG) + f"; C_G = max = {Cg:.2f}")
    rx = mp.mpf(r) * X0
    tail_fact = float(mp.fsum(Cg * rx ** k / mp.factorial(k) for k in range(K + 1, K + 400)))
    # the sharpened form: the density vanishes at the far end like (kappa_* - t)^(beta - 1), so the shifted
    # coefficients decay like r^k / Gamma(k + Re beta), as the sigma_k do
    CG2 = [float(abs(g[k]) * mp.gamma(k + b) / r ** k) for k in range(4, K + 1)]
    Cg2 = max(CG2)
    tail = float(mp.fsum(Cg2 * rx ** k / mp.gamma(k + b) for k in range(K + 1, K + 400)))
    print(f"   sharpened, |g_k| Gamma(k + {b:.2f}) / r^k for k = 4..{K}: " + " ".join(f"{v:.1f}" for v in CG2)
          + f"; C_G' = max = {Cg2:.1f}")
    f = poly(g)
    z0 = mp.findroot(f, mp.mpc(X0, 0))
    print(f"   the G tail beyond K = {K} at x0: {tail_fact:.2e} with r^k/k!, {tail:.2e} sharpened;"
          f" the sharpened tail against the Rouche minimum of |G_{K}| on the circles:")
    for rho in (0.001, 0.01, 0.1):
        m = float(min(abs(f(z0 + rho * mp.expjpi(2 * t / 240))) for t in range(240)))
        print(f"      rho {rho}: min |G_K| {m:.2e}, tail / min = {tail / m:.2f}"
              + ("  (the tail alone fits)" if tail < m else "  (the tail alone does not fit)"))
    # the S_1-form test at each node on its own series: the bound calibrated on k <= K - 1 tested on sigma_K (the
    # order just computed), and the k <= 12 calibration's ratio at k = K
    print(f"   at the nodes, each on its own series: |sigma_{K}| / (bound calibrated on k <= {K - 1}), and the k <= 12"
          f" calibration's ratio at k = {K}")
    for w in WFS:
        s = S[w]
        out = []
        for Kc in (K - 1, 12):
            kc, bc = ratio_fit(s, Kc)
            ac, bcr = abs(kc), bc.real
            Cc = max(float(abs(s[k]) * mp.gamma(k + bcr) / ac ** k) for k in range(4, Kc + 1))
            out.append((Kc, ac, bcr, Cc, float(abs(s[K]) / (Cc * ac ** K / mp.gamma(K + bcr)))))
        print(f"      wf {w}: " + "; ".join(f"k <= {Kc} (a {ac:.4f}, b {bcr:.2f}, C {Cc:.1f}): {r:.3f}"
                                            for Kc, ac, bcr, Cc, r in out))


def best_kbar(sig, K):
    def obj(v):
        g = gcoef(sig, mp.mpc(v[0], v[1]), K)
        return float(abs(g[K]) * X0 ** K + abs(g[K - 1]) * X0 ** (K - 1))
    r = minimize(obj, [0.93, 0.5], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-15, maxiter=3000))
    return mp.mpc(*r.x), r.fun


def resummed(S, Kmax):
    print("3. resummed zero: G_K = e^{-kbar x} S_1 truncated at x^K (kbar minimising the last two terms at x0)")
    print("   (the plain Taylor truncation, kbar = 0, gives A0 = 1.99, 2.15, ..., 2.53 for K = 6..12: useless)")
    out = {}
    for K in range(9, Kmax + 1):
        kb, last = best_kbar(S[1.0], K)
        zs = {wf: mp.findroot(poly(gcoef(S[wf], kb, K)), mp.mpc(X0, 0)) for wf in WFS}
        im = [float(mp.im(mp.sqrt(zs[w]))) for w in WFS]
        re = [float(mp.re(mp.sqrt(zs[w]))) for w in WFS]
        c = np.polyfit(WFS, im, 2)
        r = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.97 < x.real < 1.03]
        out[K] = (kb, r)
        print(f"   K {K}: kbar {mp.nstr(kb, 5)}, last terms {last:.1e}; A0(w_inf) = {re[1]:.6f}{im[1]:+.6f}i;"
              f" real zero at w/w_inf = {', '.join(f'{x:.5f}' for x in r)}")
    return out


def zeros_and_margins(S, K, kb):
    g = gcoef(S[1.0], kb, K)
    rts = sorted(mp.polyroots(g[::-1], maxsteps=300, extraprec=300), key=abs)
    print(f"4. zeros of G_{K} at w_inf with |x| < 16: " + ", ".join(mp.nstr(z, 5) for z in rts if abs(z) < 16))
    print("   (the second zero moves with K: 10.09, 10.90, 11.36 for K = 10, 11, 12 - not converged)")
    f = poly(g)
    z0 = mp.findroot(f, mp.mpc(X0, 0))
    print(f"   terms |g_k| x0^k: " + " ".join(f"{float(abs(c)) * X0 ** i:.1e}" for i, c in enumerate(g)))
    print("5. certificate margins")
    for rho in (0.001, 0.01, 0.1):
        m = min(abs(f(z0 + rho * mp.expjpi(2 * t / 240))) for t in range(240))
        print(f"   Rouche disc |x - x0| = {rho}: min |G_K| on the circle {mp.nstr(m, 3)}"
              f" -> the tail (and coefficient errors) must be below this on the circle")
    A0 = mp.re(mp.sqrt(z0))

    def G(A, w):
        return poly(gcoef(S[w], kb, K))(A * A)
    dA = mp.mpf("1e-6")
    GA = (G(A0 + dA, 1.0) - G(A0 - dA, 1.0)) / (2 * dA)
    Gw = (G(A0, 1.01) - G(A0, 0.99)) / mp.mpf("0.02")
    J = np.array([[float(mp.re(GA)), float(mp.re(Gw))], [float(mp.im(GA)), float(mp.im(Gw))]])
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"   2D Jacobian of (Re G, Im G) in (A, w/w_inf): {np.round(J, 4).tolist()}, singular values"
          f" {sv[0]:.3f}, {sv[1]:.3f}")
    for r in (1e-2, 1e-3, 1e-4):
        print(f"   box radius {r:g}: total enclosure error of G must be below ~{sv[1] * r:.1e}")


def main():
    global SCAN
    args = sys.argv[1:]
    if "--scan" in args:
        i = args.index("--scan")
        SCAN = os.path.abspath(args[i + 1])
        del args[i:i + 2]
    K = int(args[0]) if args else 12
    S = {w: load(w) for w in WFS}
    ratio_law(S)
    arc = prony(S)
    res = resummed(S, K)
    zeros_and_margins(S, K, res[K][0])
    type_bound(S, K, res[K][0], arc)


if __name__ == "__main__":
    main()
