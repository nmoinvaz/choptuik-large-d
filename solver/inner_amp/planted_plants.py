#!/usr/bin/env python3
"""Plants for the planted-truth test of the selection estimators (notes/PLANTED_SELECTION.md, section 2).

Every plant is an entire function S(x, wf) of x = A^2, given at the three nodes wf = 0.99, 1.00, 1.01, in closed form
(a finite exponential sum times a polynomial, or a power series with an explicit tail), so its zeros are known to any
precision independently of the estimators.

  P-ARC   S = (1 - x/x0(wf)) L_wf(x) / L_wf(0),  L = int_0^1 rho(s) exp(x gamma(s)) gamma'(s) ds  (64-point Gauss-Legendre,
          so L is exactly a 64-term exponential sum). gamma: cubic through the [6/6] Borel-Pade poles of D_entire 7.1 and
          kappa_* (per node); rho = (1-s)^(beta-1) sum_m c_m T_m(2s-1), c_m by weighted least squares against the real
          sigma_0..sigma_12 of that node with the factor (1 - x/x_ref), x_ref = x0(wf; omega_t = 1). L is then fixed and
          only the planted zero x0(wf; omega_t) moves.
  P-ARC-B the same with the straight chord, the mirrored arc, or the endpoint exponent 1 (beta = 2).
  P-Z2    (1 - x/x0)(1 - x/x2) L2, L2 refitted with both factors.
  C-*     the real sigma_0..sigma_12 with a ratio-law tail (windows k = 6-11, 8-11, 9-11) or the exponential sum of the
          [6/6] Borel-Pade approximant; truth computed by root finding on the completed function.
  N1, N2  exp(bx)(1 - x/x0) (a lone zero), 1 and exp(bx) (no zero).

The planted zero: x0(wf) = XT + V (wf - wt) + W (wf - wt)^2, Im x0(wt) = 0 exactly.
"""
import glob
import json
import os

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "scan25phys")
NODES = (0.99, 1.0, 1.01)
XT = mp.mpf("7.400")
V = mp.mpc("4.0", "-7.8")
W = mp.mpc("-6.5", "-2.0")
OMEGA_T = ("0.9960", "0.9985", "0.9995", "1.0000", "1.00025", "1.0010", "1.0040")
KMAX = 40                      # coefficients computed per plant (estimators use <= 13; the rest for the ratio law)
DPS = 50

# D_entire 7.1 ([6/6] poles of the Borel transform) and section 1 (kappa_*, beta), per node
POLES = {0.99: ["0.739+0.160j", "0.781+0.370j", "0.853+0.603j", "0.984+0.805j", "1.186+0.997j"],
         1.0: ["0.749+0.155j", "0.793+0.360j", "0.863+0.588j", "0.986+0.786j", "1.178+0.975j"],
         1.01: ["0.760+0.151j", "0.806+0.350j", "0.876+0.573j", "0.992+0.766j", "1.179+0.947j"]}
KSTAR = {0.99: "1.271+1.081j", 1.0: "1.279+1.081j", 1.01: "1.289+1.080j"}
BETA = {0.99: "4.10-1.56j", 1.0: "4.16-1.60j", 1.01: "4.23-1.64j"}


def mpc(s):
    return mp.mpc(complex(s))


def x0_planted(wf, wt):
    d = mp.mpf(wf) - mp.mpf(wt)
    return XT + V * d + W * d * d


def load_real(scandir=REAL):
    out = {}
    for fn in sorted(glob.glob(os.path.join(scandir, "*.json"))):
        r = json.load(open(fn))
        out[float(r["wf"])] = ([mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]], list(r.get("sigma_err", [])))
    return out


def polymul(a, b, n):
    """first n coefficients of the product of two coefficient lists"""
    return [sum(a[j] * b[k - j] for j in range(max(0, k - len(b) + 1), min(k, len(a) - 1) + 1)) for k in range(n)]


def factor(zeros):
    """coefficients of prod (1 - x/z)"""
    p = [mp.mpc(1)]
    for z in zeros:
        p = polymul(p, [mp.mpc(1), -1 / mp.mpc(z)], len(p) + 1)
    return p


# ---------------------------------------------------------------- arcs and densities
class Arc:
    """gamma(s), s in [0, 1]; kind: 'arc' (cubic through the poles and kappa_*), 'chord', 'mirror'."""

    def __init__(self, wf, kind="arc"):
        pts = [mpc(p) for p in POLES[wf]] + [mpc(KSTAR[wf])]
        P0, P1 = pts[0], pts[-1]
        ch = [mp.mpf(0)]
        for a, b in zip(pts, pts[1:]):
            ch.append(ch[-1] + abs(b - a))
        s = [c / ch[-1] for c in ch]
        # cubic least squares in s, with the end points held: gamma = P0 + (P1 - P0) s + s (1 - s)(e0 + e1 s)
        A = mp.matrix([[si * (1 - si), si * si * (1 - si)] for si in s[1:-1]])
        b = mp.matrix([pt - P0 - (P1 - P0) * si for pt, si in zip(pts[1:-1], s[1:-1])])
        # complex least squares by normal equations (2 unknowns)
        AH = A.H
        e = mp.lu_solve(AH * A, AH * b)
        self.P0, self.P1, self.e0, self.e1, self.kind = P0, P1, e[0], e[1], kind
        self.fit_pts = list(zip(s, pts))

    def g(self, s):
        if self.kind == "chord":
            return self.P0 + (self.P1 - self.P0) * s
        base = self.P0 + (self.P1 - self.P0) * s + s * (1 - s) * (self.e0 + self.e1 * s)
        if self.kind == "mirror":
            u = (base - self.P0) / (self.P1 - self.P0)
            return self.P0 + (self.P1 - self.P0) * mp.conj(u)
        return base

    def dg(self, s):
        if self.kind == "chord":
            return self.P1 - self.P0
        d = (self.P1 - self.P0) + (1 - 2 * s) * (self.e0 + self.e1 * s) + s * (1 - s) * self.e1
        if self.kind == "mirror":
            return (self.P1 - self.P0) * mp.conj(d / (self.P1 - self.P0))
        return d


class Laplace:
    """L(x) = sum_i W_i exp(x t_i) with W_i = g_i (1 - s_i)^ex T-basis(s_i) . c * gamma'(s_i), 64 Gauss-Legendre nodes"""

    def __init__(self, arc, ex, M, N=64):
        xg, wg = np.polynomial.legendre.leggauss(N)
        self.s = [mp.mpf(float(v) + 1) / 2 for v in xg]
        self.gw = [mp.mpf(float(v)) / 2 for v in wg]
        self.t = [arc.g(s) for s in self.s]
        dg = [arc.dg(s) for s in self.s]
        self.M = M
        # basis weights per node: B[m][i]
        self.B = [[self.gw[i] * mp.power(1 - self.s[i], ex) * mp.chebyt(m, 2 * self.s[i] - 1) * dg[i]
                   for i in range(N)] for m in range(M)]
        self.mom = [[sum(self.B[m][i] * self.t[i] ** k for i in range(N)) / mp.factorial(k) for k in range(KMAX + 1)]
                    for m in range(M)]
        self.c = None

    def fit(self, sig_real, pref, K=12):
        """least squares, weights 1/|sigma_k|, of (pref * L)_k against sigma_k, k = 0..K. Returns relative residuals."""
        cols = [polymul(pref, self.mom[m], K + 1) for m in range(self.M)]
        A = np.array([[complex(cols[m][k] / abs(sig_real[k])) for m in range(self.M)] for k in range(K + 1)])
        b = np.array([complex(sig_real[k] / abs(sig_real[k])) for k in range(K + 1)])
        Ar = np.block([[A.real, -A.imag], [A.imag, A.real]])
        br = np.concatenate([b.real, b.imag])
        sol, *_ = np.linalg.lstsq(Ar, br, rcond=None)
        c = sol[:self.M] + 1j * sol[self.M:]
        self.c = [mp.mpc(complex(z)) for z in c]
        fitted = self.coeffs(pref, K + 1)
        return [float(abs(fitted[k] - sig_real[k]) / abs(sig_real[k])) for k in range(K + 1)]

    def lcoef(self, n=KMAX + 1):
        return [sum(self.c[m] * self.mom[m][k] for m in range(self.M)) for k in range(n)]

    def coeffs(self, pref, n=KMAX + 1, normalise=False):
        s = polymul(pref, self.lcoef(), n)
        if normalise:
            s = [z / s[0] for z in s]
        return s

    def weights(self):
        return [sum(self.c[m] * self.B[m][i] for m in range(self.M)) for i in range(len(self.t))]

    def L(self, x):
        w = self.weights() if not hasattr(self, "_w") else self._w
        self._w = w
        return mp.fsum(wi * mp.exp(x * ti) for wi, ti in zip(w, self.t))


def other_zeros(f, x0, R=16, grid=9):
    """zeros of f in |x| < R by secant iteration from a grid, plus the argument-principle count on |x| = R and on
    |x - x0| = 1. Returns (sorted zeros, count on |x| = R, count near x0)"""
    found = []
    for a in np.linspace(-R, R, grid):
        for b in np.linspace(-R, R, grid):
            if a * a + b * b > R * R:
                continue
            try:
                z = mp.findroot(f, mp.mpc(a, b), tol=mp.mpf(10) ** (-20), maxsteps=60)
            except Exception:                                    # noqa: BLE001  (no convergence from this start)
                continue
            if abs(z) < R and abs(f(z)) < mp.mpf(10) ** (-12) * max(1, abs(f(z + mp.mpf("0.1")))) \
                    and all(abs(z - y) > 1e-8 for y in found):
                found.append(z)

    def count(center, rad, n=1600):
        tot, prev = 0.0, None
        for j in range(n + 1):
            v = f(center + rad * mp.expjpi(2 * mp.mpf(j) / n))
            a = float(mp.arg(v))
            if prev is not None:
                d = a - prev
                d -= 2 * np.pi * round(d / (2 * np.pi))
                tot += d
            prev = a
        return round(tot / (2 * np.pi))
    return sorted(found, key=lambda z: abs(z - x0)), count(mp.mpc(0), R), count(x0, 1)


def ratio_law(sig, lo=6, hi=11):
    rat = [complex(sig[k] / sig[k + 1]) for k in range(lo, hi + 1)]
    ks = np.arange(lo, hi + 1)
    A = np.vstack([ks, np.ones_like(ks)]).T.astype(complex)
    c = np.linalg.lstsq(A, np.array(rat), rcond=None)[0]
    kap = 1 / c[0]
    return kap, c[1] * kap


def peak_terms(sig, x0, kmax=KMAX):
    t = [float(abs(sig[k]) * abs(x0) ** k) for k in range(min(len(sig), kmax + 1))]
    k = int(np.argmax(t))
    return k, t[k], t[12] if len(t) > 12 else None


# ---------------------------------------------------------------- families
class ArcFamily:
    """one L per node, fitted once; plant(wt) moves only the planted zero"""

    def __init__(self, kind="arc", ex_one=False, extra_zeros=(), M=6, real=None):
        mp.mp.dps = DPS
        self.real = real or load_real()
        self.kind, self.extra = kind, [mp.mpc(complex(z)) for z in extra_zeros]
        self.L, self.resid, self.M = {}, {}, {}
        for wf in NODES:
            arc = Arc(wf, "arc" if kind == "beta2" else kind)
            ex = mp.mpc(1) if (ex_one or kind == "beta2") else mpc(BETA[wf]) - 1
            xref = x0_planted(wf, "1.0000")
            for m in (M, 8):
                lap = Laplace(arc, ex, m)
                res = lap.fit(self.real[wf][0], factor([xref] + self.extra))
                if max(res) <= 1e-2 or m == 8:
                    break
            self.L[wf], self.resid[wf], self.M[wf] = lap, res, m

    def nodes(self, wt):
        return {wf: self.L[wf].coeffs(factor([x0_planted(wf, wt)] + self.extra), normalise=True) for wf in NODES}

    def S(self, wf, wt):
        lap, z = self.L[wf], [x0_planted(wf, wt)] + self.extra
        L0 = lap.L(0)
        return lambda x: lap.L(x) * mp.fprod(1 - x / zz for zz in z) / L0

    def checks(self, wt="1.0000", zeros=True):
        out = {}
        for wf in NODES:
            sig = self.nodes(wt)[wf]
            x0 = x0_planted(wf, wt)
            kap, beta = ratio_law(sig)
            kpk, tpk, t12 = peak_terms(sig, x0)
            rec = {"fit_resid_max": max(self.resid[wf]), "fit_resid": self.resid[wf], "M": self.M[wf],
                   "kappa_star": [kap.real, kap.imag], "type": abs(kap), "beta": [beta.real, beta.imag],
                   "peak_k": kpk, "peak_term": tpk, "term12": t12}
            if zeros:
                zs, nR, near = other_zeros(self.S(wf, wt), x0)
                others = [z for z in zs if abs(z - x0) > 1e-6]
                rec.update({"zeros_found": [[float(mp.re(z)), float(mp.im(z))] for z in zs[:8]],
                            "count_R16": nR, "count_near_x0": near, "nearest_other": float(abs(others[0] - x0))
                            if others else None,
                            "nearest_other_zero": [float(mp.re(others[0])), float(mp.im(others[0]))] if others else None,
                            "n_found": len(zs)})
            out[wf] = rec
        return out


def completion(kind, real=None, nmax=250, kc=12):
    """C-W1/C-W2/C-W3 (ratio-law tail) or C-EXP (Borel-Pade [6/6] exponential sum). Returns ({wf: sigma list},
    {wf: function}, info)."""
    mp.mp.dps = DPS
    real = real or load_real()
    nodes, funcs, info = {}, {}, {}
    for wf in NODES:
        sig = list(real[wf][0][:kc + 1])
        if kind.startswith("C-W"):
            lo = {"C-W1": 6, "C-W2": 8, "C-W3": 9}[kind]
            rat = [sig[k] / sig[k + 1] for k in range(lo, kc)]
            A = mp.matrix([[k, 1] for k in range(lo, kc)])
            b = mp.matrix(rat)
            AH = A.H
            c = mp.lu_solve(AH * A, AH * b)
            kap, beta = 1 / c[0], c[1] / c[0]
            for k in range(kc, nmax):
                sig.append(sig[k] * kap / (k + beta))
            coeffs = sig
            funcs[wf] = (lambda cs: (lambda x: mp.polyval(cs[::-1], x)))(coeffs)
            info[wf] = {"kappa": [float(mp.re(kap)), float(mp.im(kap))], "beta": [float(mp.re(beta)), float(mp.im(beta))]}
        else:
            a = [sig[k] * mp.factorial(k) for k in range(kc + 1)]
            p, q = mp.pade(a, 6, kc - 6)        # [6/6] (kc = 12), [6/7] (kc = 13)
            us = mp.polyroots(q[::-1], maxsteps=400, extraprec=400)
            dq = [k * q[k] for k in range(1, len(q))]
            ts, rs = [], []
            for u in us:
                r = -mp.polyval(p[::-1], u) / (u * mp.polyval(dq[::-1], u))
                ts.append(1 / u)
                rs.append(r)
            c0 = p[6] / q[6] if kc == 12 else mp.mpc(0)
            funcs[wf] = (lambda ts, rs, c0: (lambda x: c0 + mp.fsum(r * mp.exp(t * x) for t, r in zip(ts, rs))))(ts, rs, c0)
            coeffs = [(c0 if k == 0 else 0) + mp.fsum(r * t ** k for t, r in zip(ts, rs)) / mp.factorial(k)
                      for k in range(KMAX + 1)]
            info[wf] = {"poles": [[float(mp.re(t)), float(mp.im(t))] for t in ts],
                        "max_pole": max(float(abs(t)) for t in ts),
                        "reproduce_max_rel": max(float(abs(coeffs[k] - sig[k]) / abs(sig[k])) for k in range(kc + 1))}
        nodes[wf] = coeffs
    # truth: the zero near 7.4 at each node, then the crossing of the quadratic through Im x and Im sqrt(x)
    xr = {}
    for wf in NODES:
        xr[wf] = mp.findroot(funcs[wf], mp.mpc("7.40", "0"), tol=mp.mpf(10) ** -40, maxsteps=200)
        info[wf]["x_R"] = [float(mp.re(xr[wf])), float(mp.im(xr[wf]))]
    truth = {}
    for lab, fn in (("im_x", lambda z: mp.im(z)), ("im_sqrt", lambda z: mp.im(mp.sqrt(z)))):
        c = np.polyfit(NODES, [float(fn(xr[w])) for w in NODES], 2)
        rts = [r.real for r in np.roots(c) if abs(r.imag) < 1e-12 and 0.95 < r.real < 1.05]
        truth[lab] = min(rts, key=lambda r: abs(r - 1)) if rts else None
    return nodes, funcs, info, truth


def lone_zero(wt, b=mp.mpc("0.89", "0.55")):
    mp.mp.dps = DPS
    out = {}
    for wf in NODES:
        x0 = x0_planted(wf, wt)
        e = [b ** k / mp.factorial(k) for k in range(KMAX + 1)]
        out[wf] = polymul(e, [mp.mpc(1), -1 / x0], KMAX + 1)
    return out


def no_zero(kind, b=mp.mpc("0.89", "0.55")):
    mp.mp.dps = DPS
    if kind == "one":
        return {wf: [mp.mpc(1)] + [mp.mpc(0)] * KMAX for wf in NODES}
    return {wf: [b ** k / mp.factorial(k) for k in range(KMAX + 1)] for wf in NODES}


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "arc"
    fam = ArcFamily(kind)
    for wf, rec in fam.checks().items():
        print(wf, {k: v for k, v in rec.items() if k != "fit_resid"})
        print("   fit residuals:", " ".join(f"{r:.1e}" for r in rec["fit_resid"]))
