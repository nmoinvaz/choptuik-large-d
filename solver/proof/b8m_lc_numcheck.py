#!/usr/bin/env python3
"""Lemma LC (proof/B_chain.md section 13): numerical checks of the light-cone-gauge bootstrap at moderate n (double
precision with mpmath residuals, not rigorous), M = 1, at the centre of the parameter box.

The solution in the light-cone gauge c(1) = 1 is started at x_0 = 1e-3 from the formal next orders of the series in
the light-cone gauge (c_k(1) = 0 for every k), R(x_0) = sum_{K < k <= K+4} eps^k Z_k(x_0), so that
d = c(0) - c^K(0) = sum_{K < k <= K+4} eps^k c_k(0) up to the omitted orders. The remainder equation
R' = J R + tau + N (b8m_numcheck.integrate_R) is integrated along the real axis to 1 - eta and then around the circle
|x - 1| = eta through the upper and the lower half plane, eta = 0.02 (inside the light-cone region L).
  (1) R_c(1), from the mean of R_c over the circle (Cauchy), must be small compared with d: the gauge is the
      light-cone gauge; the two continuations must agree at 1 + eta (the light-cone singular part is of size
      eta^(n/2));
  (2) d n^(K+1) must be below the certified d0 of b8m_lc.py (its log line);
  (3) on the sample points, |R|/bound < 1 for the per-band constants h of the certificate (shape check: the certificate
      is for n >= n_0, the check at moderate n tests the form of the bound), and on L |x R_S'| <= h_Sp n w.
Mutations: --mutate=d0 (the bound d0 replaced by 0: (2) must fail); --mutate=weight (the weight rho^(K+2) instead of
rho^(K+1): (3) must fail).

    python3 b8m_lc_numcheck.py n --log=b8m_lc_K4.log [--L0=73.89] [--mutate=...]
"""
import json
import sys
import numpy as np
import mpmath as mp
from flint import ctx
import b8m_formal as FM
import b8m_numcheck as NC

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
O, E = NC.O, NC.E


class SeriesLC(NC.Series):
    def __init__(self, K, n, extra=4):
        ctx.prec = 200
        F, om = FM.params()
        self.fs = FM.formal(K + extra, F, om)            # every order in the light-cone gauge c_k(1) = 0
        self.K, self.n, self.extra = K, mp.mpf(n), extra
        self.w = mp.mpf(om.real.mid().str(50, radius=False))
        tm = lambda z: mp.mpc(mp.mpf(z.real.mid().str(45, radius=False)), mp.mpf(z.imag.mid().str(45, radius=False)))
        self.co = {}
        for nm, keys in (("V", O), ("S", O), ("be", E), ("c", E)):
            for h in keys:
                self.co[(nm, h)] = [[tm(c) for c in self.fs[nm][k][h].coeffs()] for k in range(K + extra + 1)]


def band(x):
    if abs(x) <= 0.05:
        return "N0"
    if abs(x - 1) <= 0.025 + 1e-12 and x.imag >= -1e-14:
        return "L"
    if abs(x) <= 1.5:
        return "Na"
    if abs(x) <= 20:
        return "Nb"
    return "F"


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = float(pos[0])
    K = 4
    hyp, d0 = None, None
    for line in open(OPTS["log"]):
        r = json.loads(line)
        if r.get("closes") and abs(r["L0"] - float(OPTS.get("L0", "73.89"))) < 1e-9 and not r.get("mutate"):
            hyp, d0 = r["hyp"], r["diag"]["d0"]
    if hyp is None:
        print("no closing LC record in the log")
        return 2
    if MUT == "d0":
        d0 = 0.0
    ser = SeriesLC(K, n)
    w = float(ser.w)
    zk = NC.ZKfast(ser)
    eta, x0 = 0.02, 1e-3
    # d from the formal orders K < k <= K + 4
    eps = 1 / n
    dfor = np.array([complex(sum(ser.co[("c", h)][k][0] * mp.mpf(eps) ** k for k in range(K + 1, K + 5)
                                 if ser.co[("c", h)][k])) for h in E])
    print(f"n = {n:g}: formal light-cone offset d = c(0) - c^K(0): |d|_1 n^(K+1) = {NC.l1(dfor) * n ** (K + 1):.4e} "
          f"(c_5(0) part {NC.l1(np.array([complex(ser.co[('c', h)][5][0]) for h in E])):.4e})", flush=True)
    R0 = ser.nxt(x0)
    # path A: the real axis, t = ln x
    real = lambda t: (np.exp(t), np.exp(t))
    xs = np.unique(np.concatenate([np.geomspace(0.02, 1 - eta, 24), [0.05, 0.5, 0.9, 0.975, 1 - eta]]))
    ts = np.concatenate([[np.log(x0)], np.log(xs)])
    RA, e1 = NC.integrate_R(ser, zk, n, w, real, R0, ts, np.linspace(np.log(x0), np.log(1 - eta), 201), rtol=1e-10)
    pts = [(complex(x), R) for x, R in zip(xs, RA[1:])]
    Rstart = RA[-1]
    # the circle through the upper (s = +1) and the lower (s = -1) half plane, phi from pi to 0
    phis = np.linspace(0, np.pi, 41)
    circ = {}
    for sgn in (1, -1):
        path = lambda t, s=sgn: (1 + eta * np.exp(1j * s * (np.pi - t)), -1j * s * eta * np.exp(1j * s * (np.pi - t)))
        RB, e2 = NC.integrate_R(ser, zk, n, w, path, Rstart, phis, np.linspace(0, np.pi, 121), rtol=1e-10)
        circ[sgn] = RB
        if sgn == 1:
            pts += [(path(t)[0], R) for t, R in zip(phis[1:], RB[1:])]
    # (1) the gauge: R_c(1) by the mean over the circle (trapezoid on both halves)
    dphi = phis[1] - phis[0]
    # upper half: phi from pi to 0; lower: -pi to 0; the mean over [0, 2 pi)
    up_ = np.array([circ[1][i][7:10] for i in range(len(phis))])
    dn_ = np.array([circ[-1][i][7:10] for i in range(len(phis))])
    mean = (np.trapezoid(up_, dx=dphi, axis=0) + np.trapezoid(dn_, dx=dphi, axis=0)) / (2 * np.pi)
    jump = NC.l1(circ[1][-1] - circ[-1][-1])
    g1 = NC.l1(mean) / NC.l1(dfor)
    ok1 = g1 < 1e-3
    print(f"(1) R_c(1) (Cauchy mean on |x - 1| = {eta}): |R_c(1)|_1 = {NC.l1(mean):.3e} = {g1:.2e} |d|; "
          f"jump of the two continuations at 1 + eta: {jump:.2e}; spline errors {max(e1, e2):.1e}  "
          f"{'OK' if ok1 else 'FAIL'}", flush=True)
    # (2) d against the certified d0
    dn5 = NC.l1(dfor) * n ** (K + 1)
    ok2 = dn5 <= d0
    print(f"(2) |d| n^(K+1) = {dn5:.4e} <= d0 = {d0:.4e} (certified, b8m_lc.py){'   [MUTATION d0]' if MUT == 'd0' else ''}"
          f"  {'OK' if ok2 else 'FAIL'}", flush=True)
    # (3) the bounds per band
    kw = K + 2 if MUT == "weight" else K + 1
    worst = {}
    for x, R in pts:
        b = band(x)
        X = 1 + abs(x)
        rho, th = X * X / n, abs(x) / X
        for nm, sl, wt in (("V", slice(0, 2), rho ** kw), ("S", slice(2, 4), th * rho ** kw),
                           ("b", slice(4, 7), th * rho ** kw), ("c", slice(7, 10), rho ** kw)):
            key = f"{nm}.{b}"
            v = NC.l1(R[sl]) / (hyp[key] * wt)
            worst[key] = max(worst.get(key, 0.0), v)
    # the derivative on L: x R_S' from the remainder equation R' = J R + tau + N
    wd = 0.0
    for x, R in pts:
        if band(x) != "L":
            continue
        z = zk(x)
        dR = NC.jac(x, z, n, w) @ R + ser.tau(x) + (NC.F_np(x, z + R, n, w) - NC.F_np(x, z, n, w)
                                                      - NC.jac(x, z, n, w) @ R)
        X = 1 + abs(x)
        wd = max(wd, NC.l1(x * dR[2:4]) / (hyp["Sp"] * n * (abs(x) / X) * (X * X / n) ** kw))
    worst["Sp.L"] = wd
    ok3 = all(v < 1 for v in worst.values())
    print(f"(3) max |R|/bound per band (weight rho^{kw}): " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()))
          + f"  {'OK' if ok3 else 'FAIL'}", flush=True)
    ok = ok1 and ok2 and ok3
    if MUT:
        print(f"MUTATION {MUT}: a check must FAIL: {'caught' if not ok else 'NOT caught'}")
        return 0 if not ok else 3
    print("ALL CHECKS OK" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
