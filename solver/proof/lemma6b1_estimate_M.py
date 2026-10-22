#!/usr/bin/env python3
"""Lemma 6b_1 for M harmonics, planning estimates (proof/A_lemmas_M.md section 5). Double precision, NOT rigorous.

1. y+ in the variables of lemma6_usys_M.SysUM on the annulus, from the formal series (fixed_point_M's) at |u| = rho0
   on the start ray arg u = -3pi/4, integrated inward along the ray and along arcs |u| = rho to the real axis: the
   largest |P|, |V|, |C_0 - 1| on each arc.
2. The distance to the nearest singularity of y+ at points of the annulus, by the root test on Taylor coefficients
   (|p_k|^(-1/k) at k = 40..60, computed by Picard iteration on power series at 64 bits): it sets the disc radii.
3. The same at M = 1, against the branch point located in proof/A_certify.md 12.1 (u* = 9.6746 e^{-2.1198 i}): a check
   that the root test sees it.

    python3 lemma6b1_estimate_M.py [--M=3] [--rho0=262]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import acb, arb, acb_series, ctx  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
import lemma6_usys_M as UM  # noqa: E402

KAP = UM.KAPPA
E3 = np.exp(-3j * np.pi / 4)


def F_np(u, y, M, w):
    odd, even = UM.harmonics(M)
    no = len(odd)
    P = {m: y[i] for i, m in enumerate(odd)}
    V = {m: y[no + i] for i, m in enumerate(odd)}
    Cm1 = {n: y[2 * no + i] / KAP for i, n in enumerate(even)}
    C = {n: Cm1[n] + (1 if n == 0 else 0) for n in even}

    def prn(X, Y, out):
        return {k: sum(X[a] * Y[k - a] for a in X if (k - a) in Y) for k in out}
    s = {m: 1j * m * w for m in odd}
    Q = {m: V[m] + (1 + s[m]) * P[m] for m in odd}
    CQ = prn(Cm1, Q, odd)
    W = {m: V[m] + CQ[m] for m in odd}
    CW = prn(Cm1, W, odd)
    Eh = {k: v / 2 for k, v in prn(P, P, even).items()}
    CE = prn(C, Eh, even)
    CEP = prn(CE, P, odd)
    z = 1 / u
    dP = [-z * (V[m] + CQ[m]) / 2 for m in odd]
    dV = [-V[m] / 2 - (CQ[m] + CW[m]) / 2 - z * (-(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m]) / 2 for m in odd]
    dC = [-z * z * CE[n] / 2 * KAP for n in even]
    return np.array(dP + dV + dC)


def start(M, rho0):
    """the optimally truncated series at u0 = rho0 e^{-3 i pi/4} (midpoints), in SysUM variables"""
    ctx.prec = 256
    xs = FPM.ZEROS[M]
    r0 = f"{float(rho0) ** -0.5:.10f}"
    sa, N, P, Q, C, R = FPM.setup_M([acb(v) for v in xs], M, r0)
    u0 = acb(arb(rho0)) * (acb(0, -1) * 3 * arb.pi() / 4).exp()
    z0 = 1 / u0
    w = sa.w
    y = [P[m](z0) for m in sa.odd]
    y += [Q[m](z0) - (1 + acb(0, 1) * m * w) * P[m](z0) for m in sa.odd]
    y += [KAP * (C[n](z0) - (1 if n == 0 else 0)) for n in sa.even]
    return np.array([complex(v.mid()) for v in y]), complex(w.mid()), N


def integrate(path, y0, t0, t1, M, w):
    nd = len(y0)

    def f(t, yr):
        u, du = path(t)
        d = F_np(u, yr[:nd] + 1j * yr[nd:], M, w) * du
        return np.concatenate([d.real, d.imag])
    return solve_ivp(f, (t0, t1), np.concatenate([y0.real, y0.imag]), method="DOP853", rtol=1e-12, atol=1e-14,
                     dense_output=True)


def val(sol, t):
    y = sol.sol(t)
    n = len(y) // 2
    return y[:n] + 1j * y[n:]


def root_test(M, c, y, w, n=60):
    """|p_k|^(-1/k) (max over components) for k = 40..n: an estimate of the radius of convergence at c"""
    ctx.prec = 64
    L = n + 1
    ctx.cap = L
    zs = 1 / acb_series([acb(c.real, c.imag), 1], prec=L)
    zero = acb_series([0], prec=L)
    Y0 = [acb(v.real, v.imag) for v in y]
    Ys = [acb_series([v], prec=L) for v in Y0]
    wa = acb(w.real, w.imag)
    for _ in range(n):
        Fs = UM.rhs_gen_M(M, Ys, zs, wa, zero)
        Ys = [acb_series([Y0[i]], prec=L) + Fs[i].integral() for i in range(len(Y0))]
    est = []
    for k in range(40, n + 1, 5):
        mk = max(float(abs(s.coeffs()[k]).mid()) if len(s.coeffs()) > k else 0.0 for s in Ys)
        est.append(mk ** (-1.0 / k) if mk > 0 else float("inf"))
    return est


def arc_diagnostic(M, rho, h=1e-8, rho0=262.0, K=11, box0=None):
    """why a box started at the arc's ray end grows: along the arc |u| = rho (from the start ray, t: 0 -> 1 in K steps)
    (a) the true spread of y+ over the parameter box x^ +- h: sum_i |dy/dx_i| h, by central differences in each
        parameter (series at rho0, ray, arc);
    (b) the growth of a box restarted at the arc's start: the largest absolute row sum of the arc's flow map (real
        coordinates, by finite differences of the flow), times the start box radius box0 (the hull of a full box)."""
    import fixed_point_M as F
    xs0 = [float(v) for v in F.ZEROS[M]]
    ts = np.linspace(0, 1, K + 1)
    phis = -3 * np.pi / 4 * (1 - ts)

    def traj(xs):
        saved = F.ZEROS[M]
        F.ZEROS[M] = [repr(v) for v in xs]
        try:
            y0, w, N = start(M, rho0)
        finally:
            F.ZEROS[M] = saved
        sr = integrate(lambda t: (t * E3, E3), y0, rho0, rho, M, w)
        ya = val(sr, rho)
        sa = integrate(lambda t: (rho * np.exp(1j * t), 1j * rho * np.exp(1j * t)), ya, -3 * np.pi / 4, 0.0, M, w)
        return np.array([val(sa, p) for p in phis]), ya, w
    base, ya, w = traj(xs0)
    spread = np.zeros(len(ts))
    for i in range(len(xs0)):
        d = 1e-6 * max(abs(xs0[i]), 1e-2)
        xp, xm = list(xs0), list(xs0)
        xp[i] += d
        xm[i] -= d
        der = (traj(xp)[0] - traj(xm)[0]) / (2 * d)
        spread += np.abs(der).max(axis=1) * h
    # flow map of the arc from its start, in real coordinates
    nd = len(ya)
    eps = 1e-7
    cols = []
    for j in range(2 * nd):
        dy = np.zeros(nd, complex)
        dy[j % nd] = eps if j < nd else 1j * eps
        sa = integrate(lambda t: (rho * np.exp(1j * t), 1j * rho * np.exp(1j * t)), ya + dy, -3 * np.pi / 4, 0.0, M, w)
        cols.append(np.array([val(sa, p) for p in phis]) - base)
    rows = []
    for k in range(len(ts)):
        Phi = np.array([np.concatenate([c[k].real, c[k].imag]) / eps for c in cols]).T
        rows.append((np.abs(Phi).sum(axis=1).max(), np.linalg.svd(Phi, compute_uv=False)[0]))
    print(f"M = {M}, arc |u| = {rho}: true spread over x^ +- {h:g} vs the hull growth of a box restarted at the arc start"
          + (f" (radius {box0:g})" if box0 else ""))
    for k in range(len(ts)):
        rs, sv = rows[k]
        print(f"   t = {ts[k]:.3f}: true spread {spread[k]:.2e};  max abs row sum of the flow {rs:.2e}, largest singular "
              f"value {sv:.2e}" + (f";  restarted box -> {rs * box0:.2e}" if box0 else ""), flush=True)
    return spread, rows


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "3"))
    if "arc" in opts:
        arc_diagnostic(M, float(opts["arc"]), float(opts.get("h", "1e-8")), box0=float(opts.get("box0", "0")) or None)
        return
    rho0 = float(opts.get("rho0", "262"))
    t00 = time.time()
    y0, w, N = start(M, rho0)
    print(f"M = {M}: series at |u| = {rho0} on the start ray (N = {N}), w = {w.real:.6f}", flush=True)
    sr = integrate(lambda t: (t * E3, E3), y0, rho0, 8.0, M, w)
    if not sr.success:
        print(f"ray integration stopped: {sr.message}")
    no = M + 1
    out = {"M": M, "arcs": [], "radii": []}
    print("1. max over the arc |u| = rho of |P|, |V|, |C_0 - 1| (y+ continued along the start ray, then the arc):")
    arcs = {}
    for rho in (262, 150, 100, 60, 40, 30, 25, 20):
        sa = integrate(lambda t, r=rho: (r * np.exp(1j * t), 1j * r * np.exp(1j * t)), val(sr, rho), -3 * np.pi / 4, 0.0,
                       M, w)
        arcs[rho] = sa
        mP = mV = mC = 0.0
        for t in np.linspace(-3 * np.pi / 4, 0, 61):
            y = val(sa, t)
            mP = max(mP, np.abs(y[:no]).max())
            mV = max(mV, np.abs(y[no:2 * no]).max())
            mC = max(mC, abs(y[2 * no + (M + 1) // 2 + 0] / KAP) if False else abs(y[2 * no + (len(y) - 2 * no) // 2] / KAP))
        out["arcs"].append(dict(rho=rho, P=mP, V=mV, C0m1=mC, ok=bool(sa.success)))
        print(f"   |u| = {rho:3d} (|xi| = {rho ** -0.5:.4f}): max|P| {mP:.2f}, max|V| {mV:.2f}, max|C_0 - 1| {mC:.3f}"
              + ("" if sa.success else "  (integration failed)"), flush=True)
    print("2. root-test radius of convergence of y+ (u-plane) at points of the annulus (k = 40..60):")
    for rho in (100, 60, 40, 30, 25, 20):
        for t in (-3 * np.pi / 4, -3 * np.pi / 8, 0.0):
            c = rho * np.exp(1j * t)
            y = val(arcs[rho], t)
            est = root_test(M, c, y, w)
            out["radii"].append(dict(rho=rho, arg=t, est=est))
            print(f"   u = {rho:3d} e^({t:+.3f} i): radius estimates {', '.join(f'{e:.1f}' for e in est)}", flush=True)
    print(f"[{time.time() - t00:.0f} s]")
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
