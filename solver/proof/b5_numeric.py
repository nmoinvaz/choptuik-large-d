#!/usr/bin/env python3
"""Lemma B5(1) (proof/B5_inner.md): double-precision assessment, NOT rigorous. Three things:

  lip       the Lipschitz coefficients of the W-system (W = Y_out - Z_in in the state (P, Phi, C, vt)) along the M = 1
            inner solution, and the contraction constant Lambda(Z, kappa) of the fixed-point map of section 3 (kernel
            factors exactly as in b5_bounds.py), with Perron weights; for choosing (Z, kappa) before the rigorous run.
  residual  the residual of the inner solution in the exact model, f_eps(Z_in) - Z_in', on a ray z -> 0 at fixed
            eps: the Phi component must be O(eps/|z|) (|r| |z|/eps bounded), not O(eps/|z|^2).
            --mutate=offsol evaluates it at Z_in + 1e-2 z (off the solution): |r| |z|/eps must then grow like 1/|z|.
  sizes     sup over |z| <= Z, 0 <= arg z <= pi/2 + kappa of the inner quantities that enter the bounds.

The inner solution is the formal series truncated at N terms (mpmath, the M = 1 zero x^ of proof/A_certify.md), which
on |z| <= 0.012 is within e^{-1/(2|z|)} ~ 1e-18 of the lateral sum: adequate for an assessment.

    python3 b5_numeric.py lip [--Z=0.008] [--kappa=0.2]
    python3 b5_numeric.py residual [--mutate=offsol]
    python3 b5_numeric.py sizes [--Z=0.008] [--kappa=0.2]
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b5_system as BS  # noqa: E402
from series_arb import SeriesArb  # noqa: E402
from flint import acb, arb, ctx  # noqa: E402

ODD, EVEN = BS.ODD, BS.EVEN
A_STAR, WF_STAR = "2.5421480475", "0.7769940136"
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
COMPS = [("P", m) for m in ODD] + [("F", m) for m in ODD] + [("C", k) for k in EVEN] + [("v", k) for k in EVEN]


def omega():
    return float(2 * mp.pi / mp.mpf("1.64389375") * mp.mpf(WF_STAR))


def inner_series(J=120):
    ctx.prec = 256
    W0 = 2 * arb.pi() / arb("1.64389375")
    A = arb(A_STAR)
    sa = SeriesArb({1: acb(0, -A / 2), -1: acb(0, A / 2)}, acb(W0 * arb(WF_STAR)))
    d, s, g = sa.series(J)
    cx = lambda c: complex(float(c.real.mid()), float(c.imag.mid()))
    D = [{m: cx(dj[m]) for m in ODD} for dj in d]
    S = [{m: cx(sj[m]) for m in ODD} for sj in s]
    G = [{k: cx(gj[k]) for k in EVEN} for gj in g]
    return D, S, G


class Inner:
    def __init__(self, N=None):
        self.D, self.S, self.G = inner_series()
        self.N = N

    def at(self, z):
        """P, Q, C and z-derivatives at z (complex), truncated at the smallest term"""
        N = self.N or min(len(self.S), max(8, int(0.5 / abs(z))))
        zp = [z ** j for j in range(N + 1)]
        P = {m: sum(self.D[j][m] * zp[j] for j in range(N)) for m in ODD}
        Q = {m: sum(self.S[j][m] * zp[j] for j in range(N)) for m in ODD}
        C = {k: sum(self.G[j][k] * zp[j] for j in range(N)) for k in EVEN}
        dP = {m: sum(j * self.D[j][m] * zp[j - 1] for j in range(1, N)) for m in ODD}
        dQ = {m: sum(j * self.S[j][m] * zp[j - 1] for j in range(1, N)) for m in ODD}
        dC = {k: sum(j * self.G[j][k] * zp[j - 1] for j in range(1, N)) for k in EVEN}
        return P, Q, C, dP, dQ, dC


class NP:
    """numpy versions of the pieces of the W-system (state (P, Phi, C, vt)); the eps -> 0 parts"""

    def __init__(self, w):
        self.mdl = BS.Model(w, 0, mp_=NPmp())
        self.w = w

    def rhs0(self, z, Y):
        """(P', Phi'-Phi/(2z^2), C') of the eps -> 0 system at the state Y (dict by component)"""
        P = {m: Y[("P", m)] for m in ODD}
        Phi = {m: Y[("F", m)] for m in ODD}
        C = {k: Y[("C", k)] for k in EVEN}
        vt = {k: Y[("v", k)] for k in EVEN}
        dP, dPhi, dC, G0 = self.mdl.f_0(z, P, Phi, C, vt)
        out = {("P", m): dP[m] for m in ODD}
        out.update({("F", m): G0[m] for m in ODD})
        out.update({("C", k): dC[k] for k in EVEN})
        return out


class NPmp:
    """minimal mpmath-like shim for Model.solve with numpy complex"""

    @staticmethod
    def matrix(A):
        return np.array(A, dtype=complex)

    @staticmethod
    def lu_solve(A, b):
        return np.linalg.solve(A, b)


def state_in(inner, z, w):
    P, Q, C, dP, dQ, dC = inner.at(z)
    s = {m: 1j * m * w for m in ODD}
    CQ = BS.proj(C, Q, ODD)
    Phi = {m: CQ[m] - (1 + s[m]) * P[m] for m in ODD}
    Y = {("P", m): P[m] for m in ODD}
    Y.update({("F", m): Phi[m] for m in ODD})
    Y.update({("C", k): C[k] for k in EVEN})
    Y.update({("v", k): 0j for k in EVEN})
    return Y, dict(P=P, Q=Q, C=C, dP=dP, dQ=dQ, dC=dC, Phi=Phi)


def jac(f, z, Y, h=1e-7):
    base = f(z, Y)
    J = {}
    for cj in COMPS:
        Yp = dict(Y)
        Yp[cj] = Y[cj] + h
        fp = f(z, Yp)
        for ci in base:
            J[(ci, cj)] = (fp[ci] - base[ci]) / h
    return J


def sample_domain(Z, kappa, nr=6, na=9):
    az = np.pi / 2 + kappa
    for r in np.geomspace(Z * 1e-3, Z, nr):
        for th in np.linspace(0, az, na):
            yield r * np.exp(1j * th)


def lip(Z, kappa, verbose=True):
    w = omega()
    inner = Inner()
    npm = NP(w)
    # scaled coefficients: Phi row  |z| |dG/dY| (all sources; the Phi source carries an extra |t| in its weight),
    # C row |dg/dY|
    gam = {}
    for z in sample_domain(Z, kappa):
        Y, _ = state_in(inner, z, w)
        J = jac(npm.rhs0, z, Y)
        for (ci, cj), v in J.items():
            if ci[0] == "P":
                continue
            sc = abs(z) * abs(v) if ci[0] == "F" else abs(v)
            gam[(ci, cj)] = max(gam.get((ci, cj), 0.0), sc)
    a = np.sin(kappa) / 2
    sk = np.sin(kappa)
    n = len(COMPS)
    L = np.zeros((n, n))
    idx = {c: i for i, c in enumerate(COMPS)}
    for m in ODD:
        L[idx[("P", m)], idx[("F", m)]] = Z / 2
    for (ci, cj), g in gam.items():
        i, j = idx[ci], idx[cj]
        if ci[0] == "F":
            if cj[0] == "F":
                L[i, j] = g * Z / (a - 2 * sk * Z)       # integrand g |t|^0 (w_Phi ~ |t|), fast kernel, weight |z|
            else:
                L[i, j] = g / (a - sk * Z)               # integrand g |t|^-1, fast kernel, weight |z|
        elif ci[0] == "C":
            L[i, j] = g * Z                              # radial slow integral of an O(1) integrand, weight 1
    ev = np.linalg.eigvals(L)
    rho = max(abs(ev))
    # Perron weights (positive eigenvector of L + tiny), weighted max-row-sum norm
    Lp = L + 1e-12
    evl, evc = np.linalg.eig(Lp)
    k = int(np.argmax(evl.real))
    u = np.abs(evc[:, k].real) + 1e-9
    wts = 1 / u
    lam = max(sum(wts[i] * L[i, j] / wts[j] for j in range(n)) for i in range(n))
    if verbose:
        print(f"Z = {Z}, kappa = {kappa} (a = sin(kappa)/2 = {a:.4f}): spectral radius {rho:.4f}, weighted norm {lam:.4f}")
        big = sorted(((L[idx[ci], idx[cj]], ci, cj) for (ci, cj) in gam), reverse=True)[:8]
        print("   largest entries: " + "; ".join(f"{ci[0]}{ci[1]:+d}<-{cj[0]}{cj[1]:+d} {v:.3g}" for v, ci, cj in big))
        print("   Phi-row coefficients |z||dG/dY| (max): "
              + ", ".join(f"<-{t} {max(v for (ci, cj), v in gam.items() if ci[0] == 'F' and cj[0] == t):.3g}"
                          for t in ("P", "F", "C")))
    return rho, lam


def residual(mutate=None):
    """r_Phi = f_eps(Z_in) - Z_in' on the ray arg z = 1 (and 0), |z| from 5e-3 to 5e-5, eps = 1e-10"""
    mp.mp.dps = 40
    w = mp.mpf(omega())
    inner = Inner()
    eps = mp.mpf("1e-10")
    mdl = BS.Model(w, eps, mp)
    print(f"residual of the inner solution in the exact model, eps = {eps}" + (" [MUTATION offsol]" if mutate else ""))
    out = []
    for th in (0.0, 1.0, 1.9):
        for r in (5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5):
            z = mp.mpc(r * np.cos(th), r * np.sin(th))
            zc = complex(z)
            Yd, q = state_in(inner, zc, float(w))
            P = {m: mp.mpc(q["P"][m]) for m in ODD}
            Phi = {m: mp.mpc(q["Phi"][m]) for m in ODD}
            if mutate == "offsol":
                Phi = {m: Phi[m] + mp.mpf("1e-2") * z for m in ODD}
            C = {k: mp.mpc(q["C"][k]) for k in EVEN}
            vt = {k: mp.mpc(0) for k in EVEN}
            dP, dPhi, dC, dvt = mdl.f_eps(z, P, Phi, C, vt)
            # Z_in' from the inner system at the same state (exact for the solution; the mutation compares like with like)
            _, dPhi0, dC0, _ = BS.Model(w, 0, mp).f_0(z, P, Phi, C, vt)
            rF = max(abs(dPhi[m] - dPhi0[m]) for m in ODD)
            rC = max(abs(dC[k] - dC0[k]) for k in EVEN)
            out.append((th, r, float(rF * abs(z) / eps), float(rC / eps)))
            print(f"  arg z = {th:.1f}, |z| = {r:.0e}: |r_Phi| |z|/eps = {float(rF * abs(z) / eps):10.4g}   "
                  f"|r_C|/eps = {float(rC / eps):.4g}")
    vals = [o[2] for o in out if o[1] <= 1e-3]
    growth = max(vals) / min(vals)
    return out, growth


def sizes(Z, kappa):
    w = omega()
    inner = Inner()
    mx = {}
    for z in sample_domain(Z, kappa, nr=8, na=11):
        Y, q = state_in(inner, z, w)
        for key in ("P", "Q", "Phi", "dP", "dQ"):
            mx[key] = max(mx.get(key, 0), max(abs(v) for v in q[key].values()))
        mx["(C-1)/z"] = max(mx.get("(C-1)/z", 0), max(abs(q["C"][k] - (1 if k == 0 else 0)) / abs(z) for k in EVEN))
        mx["Phi/z"] = max(mx.get("Phi/z", 0), max(abs(v) / abs(z) for v in q["Phi"].values()))
    print(f"sup over |z| <= {Z}, 0 <= arg z <= pi/2 + {kappa}: " + ", ".join(f"|{k}| {v:.4g}" for k, v in mx.items()))
    return mx


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "lip"
    Z = float(OPTS.get("Z", "0.008"))
    kap = float(OPTS.get("kappa", "0.2"))
    if cmd == "lip":
        for zz in [float(x) for x in OPTS.get("Zs", str(Z)).split(",")]:
            for kk in [float(x) for x in OPTS.get("kappas", str(kap)).split(",")]:
                lip(zz, kk)
    elif cmd == "residual":
        out, growth = residual(OPTS.get("mutate"))
        print(f"spread of |r_Phi| |z|/eps over |z| <= 1e-3: factor {growth:.3g} "
              f"({'bounded: O(eps/|z|)' if growth < 3 else 'GROWS: not O(eps/|z|)'})")
        if OPTS.get("mutate") == "offsol":
            sys.exit(0 if growth >= 3 else 3)
        sys.exit(0 if growth < 3 else 1)
    elif cmd == "sizes":
        sizes(Z, kap)
