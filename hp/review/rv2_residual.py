#!/usr/bin/env python3
"""Review RV2: an independent evaluation of the double-double solver's residual F(z) at a saved state.

Written from the field equations in solver/dss.py's header (not from hp/src), in python-flint ball arithmetic at 200
bits:
  * tau-derivative: d/dtheta from the definition (odd harmonics k < M of the antiperiodic field, exact DFT), not
    the kernel's folded cot matrix;
  * centre block: the kernel's centre series (Solver.series(.., 0)) is CHECKED, not recomputed: its leading
    coefficients must be the regular-centre data (p = p1 x + ..., m = -p1 x + ..., a = 1 + ..., c = c0 + ...) and the
    four equations, multiplied by x, must vanish order by order through order kc (arb_poly arithmetic per tau point);
    a regular centre series is unique given (p1, c0), so this identifies it. Then it is summed at x = nodes[0];
  * Taylor blocks: re-integrated by Picard iteration on power series (arb_series), order KT, with every kernel step
    split into SUB equal substeps;
  * horizon block: the kernel's horizon series in u = 1 - x is checked the same way (c = 1, m = m_h, a = a_h at
    u = 0, the equations through order ks - 1; analyticity at u = 0 is the regularity condition), then summed at
    u0 = 1 - nodes[-1]; with --kfreeze the p rows are Pi_low (p mismatch) + Pi_high p1;
  * phase row sum_j sin(pi j / M) p1_j.

    python3 hp/review/rv2_residual.py STATE.npz [--kfreeze 13|0] [--KT 40] [--SUB 2] [--perturb 1e-6]
                                       [--break=delta|eq] [--blocks 0,1,2,...]

--perturb e moves z away from the solution (every node field by e * sin(3 theta + q), Delta by e) so that F is O(e):
both evaluations must then agree to far below e (equation check). At the saved z (no --perturb) the difference
F_indep - F_kernel is the kernel's own evaluation error (double-double rounding and Taylor truncation); projected on
y = J^-T e_Delta it gives the Delta error the kernel cannot see: -y . F_indep vs the kernel's -y . F_kernel.
--break=delta adds 1e-24 to Delta in the independent evaluation only; --break=eq changes the n/2 c (p + m) coupling
by a relative 1e-15 in it. Either must make the agreement check fail.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import flint  # noqa: E402
import numpy as np  # noqa: E402
from flint import arb, arb_mat, arb_poly, arb_series, fmpq  # noqa: E402

from hpkernel.driver import load, make_solver  # noqa: E402

flint.ctx.prec = 200


def A(x):
    return arb(fmpq(*float(x).as_integer_ratio()))


def ddv(h, l):
    return A(h) + A(l)


def dmat(M):
    """d/dtheta on the half grid theta_j = pi j / M for antiperiodic fields: odd harmonics k < M"""
    N = 2 * M
    pi = arb.pi()
    rows = []
    for i in range(M):
        r = []
        for j in range(M):
            s = arb(0)
            for k in range(1, M, 2):
                s += k * (k * (i - j) * pi / M).sin()
            r.append(-4 * s / N)
        rows.append(r)
    return arb_mat(rows)


def plow(M, kf):
    N = 2 * M
    pi = arb.pi()
    return arb_mat([[sum(((k * (i - j) * pi / M).cos() for k in range(1, kf + 1, 2)), arb(0)) * 4 / N
                     for j in range(M)] for i in range(M)])


def mat_apply(Dm, vec):
    return [(Dm * arb_mat([[v] for v in vec]))[i, 0] for i in range(len(vec))]


def polys_dtau(Dm, polys, omega, ncoef):
    """tau-derivative of a list (over theta_j) of arb_poly: omega * D applied coefficient-wise"""
    M = len(polys)
    C = arb_mat([[(p.coeffs() + [arb(0)] * ncoef)[k] for k in range(ncoef)] for p in polys])
    R = Dm * C
    return [arb_poly([omega * R[j, k] for k in range(ncoef)]) for j in range(M)]


def coef(p, k):
    c = p.coeffs()
    return c[k] if k < len(c) else arb(0)


class Model:
    def __init__(self, D, M, Delta, brk=None):
        self.M = M
        self.n = A(D) - 2
        self.nh = self.n / 2 * (1 + arb("1e-15") if brk == "eq" else 1)
        self.n1 = self.n - 1
        self.Delta = Delta + (arb("1e-24") if brk == "delta" else 0)
        self.omega = (2 * arb.pi() / self.Delta).mid()
        # midpoints only: ball arithmetic through the Picard iteration and the tau mixing overestimates the error
        # by tens of orders (wrapping); 200-bit midpoints are what is needed here (precision, not rigour)
        self.D = dmat(M).mid()

    # ------------------------------------------------------------------ centre series check
    def check_centre(self, S, p1, c0, x0):
        """S: (kc+1, 4, M) arb coefficients. Returns (state at x0, worst equation residual weighted by x0^k,
        worst leading-coefficient error)."""
        kc1, _, M = len(S), 4, self.M
        P = [arb_poly([S[k][0][j] for k in range(kc1)]) for j in range(M)]
        Mm = [arb_poly([S[k][1][j] for k in range(kc1)]) for j in range(M)]
        Aa = [arb_poly([S[k][2][j] for k in range(kc1)]) for j in range(M)]
        C = [arb_poly([S[k][3][j] for k in range(kc1)]) for j in range(M)]
        lead = 0.0
        for j in range(M):
            for v in (coef(P[j], 0), coef(Mm[j], 0), coef(P[j], 1) - p1[j], coef(Mm[j], 1) + p1[j],
                      coef(Aa[j], 0) - 1, coef(C[j], 0) - c0[j]):
                lead = max(lead, abs(float(v.mid())))
        Pt = polys_dtau(self.D, P, self.omega, kc1)
        Mt = polys_dtau(self.D, Mm, self.omega, kc1)
        x = arb_poly([0, 1])
        worst = 0.0
        for j in range(M):
            p, m, a, c = P[j], Mm[j], Aa[j], C[j]
            cx, px, mx, ax = c.derivative(), p.derivative(), m.derivative(), a.derivative()
            E = [x * cx - self.n1 * c * (a * a - 1),
                 2 * x * ax - a * (p * p + m * m) - self.n1 * a * (1 - a * a),
                 x * Pt[j] + x * (x - c) * px - x * cx * p + c * p - self.nh * c * (p + m),
                 x * Mt[j] + x * (x + c) * mx + x * cx * m - c * m + self.nh * c * (p + m)]
            for e in E:
                for k in range(kc1):
                    worst = max(worst, abs(float(coef(e, k).mid())) * float(x0) ** k)
        X0 = A(x0)
        state = [[f[j](X0) for j in range(M)] for f in (P, Mm, Aa, C)]
        return state, worst, lead

    # ------------------------------------------------------------------ horizon series check
    def check_horizon(self, S, mh, ah, u0):
        ks1, M = len(S), self.M
        P = [arb_poly([S[k][0][j] for k in range(ks1)]) for j in range(M)]
        Mm = [arb_poly([S[k][1][j] for k in range(ks1)]) for j in range(M)]
        Aa = [arb_poly([S[k][2][j] for k in range(ks1)]) for j in range(M)]
        C = [arb_poly([S[k][3][j] for k in range(ks1)]) for j in range(M)]
        lead = 0.0
        for j in range(M):
            for v in (coef(C[j], 0) - 1, coef(Mm[j], 0) - mh[j], coef(Aa[j], 0) - ah[j]):
                lead = max(lead, abs(float(v.mid())))
        Pt = polys_dtau(self.D, P, self.omega, ks1)
        Mt = polys_dtau(self.D, Mm, self.omega, ks1)
        X = arb_poly([1, -1])            # x = 1 - u; d/dx = -d/du
        worst = 0.0
        for j in range(M):
            p, m, a, c = P[j], Mm[j], Aa[j], C[j]
            cx, px, mx, ax = -c.derivative(), -p.derivative(), -m.derivative(), -a.derivative()
            E = [X * cx - self.n1 * c * (a * a - 1),
                 2 * X * ax - a * (p * p + m * m) - self.n1 * a * (1 - a * a),
                 X * Pt[j] + X * (X - c) * px - X * cx * p + c * p - self.nh * c * (p + m),
                 X * Mt[j] + X * (X + c) * mx + X * cx * m - c * m + self.nh * c * (p + m)]
            for e in E:
                for k in range(ks1 - 1):
                    worst = max(worst, abs(float(coef(e, k).mid())) * float(u0) ** k)
        U0 = A(u0)
        state = [[f[j](U0) for j in range(M)] for f in (P, Mm, Aa, C)]
        return state, worst, lead

    # ------------------------------------------------------------------ Taylor step by Picard iteration
    def step(self, st, x0, x1, K):
        M = self.M
        flint.ctx.cap = K + 1
        X = arb_series([x0, 1])
        h = x1 - x0
        f0 = [[arb_series([st[q][j]]) for j in range(M)] for q in range(4)]
        p, m, a, c = [list(f) for f in f0]

        def dtau(fs):
            Cm = arb_mat([[(f.coeffs() + [arb(0)] * (K + 1))[k] for k in range(K + 1)] for f in fs])
            R = self.D * Cm
            return [arb_series([self.omega * R[j, k] for k in range(K + 1)]) for j in range(M)]

        for _ in range(K + 1):
            pt, mt = dtau(p), dtau(m)
            np_, nm, na, nc = [], [], [], []
            for j in range(M):
                pj, mj, aj, cj = p[j], m[j], a[j], c[j]
                a2 = aj * aj
                G = self.n1 * cj * (a2 - 1)
                cx = G / X
                ax = aj * (pj * pj + mj * mj + self.n1 * (1 - a2)) / (2 * X)
                cpm = self.nh * cj * (pj + mj)
                px = (-X * pt[j] + G * pj - cj * pj + cpm) / (X * (X - cj))
                mx = (-X * mt[j] - G * mj + cj * mj - cpm) / (X * (X + cj))
                np_.append(f0[0][j] + px.integral())
                nm.append(f0[1][j] + mx.integral())
                na.append(f0[2][j] + ax.integral())
                nc.append(f0[3][j] + cx.integral())
            p, m, a, c = [[arb_series([v.mid() for v in f.coeffs()]) for f in fs] for fs in (np_, nm, na, nc)]
        return [[arb_poly(f.coeffs())(h).mid() for f in fs] for fs in (p, m, a, c)]

    def integrate(self, st, xs, K, sub):
        for i in range(len(xs) - 1):
            xa, xb = A(xs[i]), A(xs[i + 1])
            for s_ in range(sub):
                t0 = xa + (xb - xa) * s_ / sub
                t1 = xa + (xb - xa) * (s_ + 1) / sub
                st = self.step(st, t0, t1, K)
        return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--kfreeze", type=int, default=None)
    ap.add_argument("--KT", type=int, default=40)
    ap.add_argument("--SUB", type=int, default=2)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--break", dest="brk", default=None)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    T0 = time.time()
    zh, zl, meta = load(a.state)
    D, M, nodes, kc, ks = meta["D"], meta["M"], list(meta["nodes"]), meta["kc"], meta["ks"]
    kf = a.kfreeze if a.kfreeze is not None else (13 if D > 35.3 else 0)
    K = len(nodes)
    L = 4 * M * (K + 1) + 1
    zh, zl = zh.copy(), zl.copy()
    if a.perturb:
        th = np.pi * np.arange(M) / M
        for q in range(4 * K):
            o = 2 * M + q * M
            zh[o:o + M] += a.perturb * np.sin(3 * th + q)
        zh[-1] += a.perturb
    s = make_solver(D, M, nodes=nodes, kt=36, kc=kc, ks=ks, threads=1, kfreeze=kf or None)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    segs = s.get_segs()
    Fh, Fl = s.residual(zh, zl)
    Fk = [ddv(h, l) for h, l in zip(Fh, Fl)]
    Z = [ddv(h, l) for h, l in zip(zh, zl)]
    mdl = Model(D, M, Z[-1], a.brk)
    off = lambda i: 2 * M + 4 * M * i          # noqa: E731
    node = lambda i: [Z[off(i) + q * M: off(i) + (q + 1) * M] for q in range(4)]   # noqa: E731
    blocks = [int(b) for b in a.blocks.split(",")] if a.blocks else list(range(K + 1))
    Fi = [None] * L
    report = {}
    # centre
    if 0 in blocks:
        sh, sl = s.series(zh, zl, 0)
        h3, l3 = np.asarray(sh).reshape(-1, 4, M), np.asarray(sl).reshape(-1, 4, M)
        S = [[[ddv(h3[k, q, j], l3[k, q, j]) for j in range(M)] for q in range(4)] for k in range(h3.shape[0])]
        st, worst, lead = mdl.check_centre(S, Z[:M], Z[M:2 * M], nodes[0])
        n0 = node(0)
        for q in range(4):
            for j in range(M):
                Fi[q * M + j] = st[q][j] - n0[q][j]
        report["centre"] = dict(eq_worst=worst, lead=lead)
        print(f"centre series: equations through order {kc}: worst |coef_k| x0^k {worst:.1e}; leading data {lead:.1e} "
              f"({time.time() - T0:.0f}s)", flush=True)
    # Taylor segments
    for b in range(1, K):
        if b not in blocks:
            continue
        st = mdl.integrate(node(b - 1), segs[b - 1], a.KT, a.SUB)
        nb = node(b)
        for q in range(4):
            for j in range(M):
                Fi[b * 4 * M + q * M + j] = st[q][j] - nb[q][j]
        print(f"segment {b}: {len(segs[b - 1]) - 1} kernel steps x {a.SUB} ({time.time() - T0:.0f}s)", flush=True)
    # horizon
    if K in blocks:
        sh, sl = s.series(zh, zl, 1)
        h3, l3 = np.asarray(sh).reshape(-1, 4, M), np.asarray(sl).reshape(-1, 4, M)
        S = [[[ddv(h3[k, q, j], l3[k, q, j]) for j in range(M)] for q in range(4)] for k in range(h3.shape[0])]
        oh = 2 * M + 4 * M * K
        mh, ah = Z[oh:oh + M], Z[oh + M:oh + 2 * M]
        st, worst, lead = mdl.check_horizon(S, mh, ah, 1 - nodes[-1])
        nK = node(K - 1)
        d = [[st[q][j] - nK[q][j] for j in range(M)] for q in range(4)]
        if kf:
            lo = plow(M, kf)
            hi = arb_mat([[(1 if i == j else 0) - lo[i, j] for j in range(M)] for i in range(M)])
            dp = mat_apply(lo, d[0])
            hp = mat_apply(hi, Z[:M])
            d[0] = [x + y for x, y in zip(dp, hp)]
        for q in range(4):
            for j in range(M):
                Fi[K * 4 * M + q * M + j] = d[q][j]
        report["horizon"] = dict(eq_worst=worst, lead=lead)
        print(f"horizon series: equations through order {ks - 1}: worst |coef_k| u0^k {worst:.1e}; leading data {lead:.1e} "
              f"({time.time() - T0:.0f}s)", flush=True)
    th = [arb.pi() * j / M for j in range(M)]
    Fi[L - 1] = sum((th[j].sin() * Z[j] for j in range(M)), arb(0))
    # compare
    s.newton(zh, zl)
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    y = [ddv(h, l) for h, l in zip(yh, yl)]
    print(f"{'block':>6s} {'max|F_kernel|':>14s} {'max|F_indep|':>13s} {'max|diff|':>10s} {'max rad':>9s} "
          f"{'-y.F_kernel':>12s} {'-y.F_indep':>12s}")
    tot_k = tot_i = arb(0)
    worst_rel = 0.0
    nbl = K + 1
    for b in list(range(nbl)) + ["phase"]:
        idx = range(L - 1, L) if b == "phase" else range(b * 4 * M, (b + 1) * 4 * M)
        if any(Fi[i] is None for i in idx):
            continue
        fk = max(abs(float(Fk[i].mid())) for i in idx)
        fi = max(abs(float(Fi[i].mid())) for i in idx)
        df = max(abs(float((Fi[i] - Fk[i]).mid())) for i in idx)
        rad = max(float(Fi[i].rad()) for i in idx)
        ek = -sum((y[i] * Fk[i] for i in idx), arb(0))
        ei = -sum((y[i] * Fi[i] for i in idx), arb(0))
        tot_k += ek
        tot_i += ei
        worst_rel = max(worst_rel, df / max(fk, 1e-300))
        report[str(b)] = dict(Fk=fk, Fi=fi, diff=df, rad=rad, est_k=float(ek.mid()), est_i=float(ei.mid()))
        print(f"{str(b):>6s} {fk:14.2e} {fi:13.2e} {df:10.2e} {rad:9.1e} {float(ek.mid()):+12.2e} {float(ei.mid()):+12.2e}")
    print(f"total linearised Delta correction: kernel {float(tot_k.mid()):+.2e}, independent {float(tot_i.mid()):+.2e}; "
          f"worst |diff| / |F_kernel| per block {worst_rel:.1e}  ({time.time() - T0:.0f}s)")
    report.update(dict(state=os.path.basename(a.state), D=D, perturb=a.perturb, brk=a.brk, kfreeze=kf,
                       est_k=float(tot_k.mid()), est_i=float(tot_i.mid()), worst_rel=worst_rel, KT=a.KT, SUB=a.SUB))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
