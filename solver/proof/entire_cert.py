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
  5. the margins a Rouche (disc in x) and a Krawczyk/Miranda (box in (A, w/w_inf)) certificate would need.

    python3 entire_cert.py [K]        (K <= 12; default 12)
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
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    S = {w: load(w) for w in WFS}
    ratio_law(S)
    prony(S)
    res = resummed(S, K)
    zeros_and_margins(S, K, res[K][0])


if __name__ == "__main__":
    main()
