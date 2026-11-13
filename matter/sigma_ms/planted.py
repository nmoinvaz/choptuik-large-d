#!/usr/bin/env python3
"""Planted truth for the sigma-model kernel (#639 acceptance 1): a manufactured periodic solution through the same
residual and Newton as the real runs.

The planted fields, on the half tau grid (theta_j = pi j / M), polynomials in x with tau-dependent coefficients of the
model's parities (p, m, psi odd, a, c even):

    psi = q x,   p = q x + P2 x^2 + P3 x^3,   m = q x - P2 x^2 - P3 x^3,   a = 1 + A2 x^2,   c = c0 + (1 - c0) x^2,

with q = 0.4 cos th + 0.1 cos 3th + 0.05 sin 3th (no sin th, so the phase row holds), P2 = 0.15 sin th + 0.1 cos th,
P3 = 0.05 sin th - 0.08 cos 3th, A2 = 0.12 + 0.04 cos 2th, c0 = 0.65 + 0.06 cos 2th + 0.03 sin 2th (c = 1 at x = 1, and
x - c < 0 on [0, 1) since c0 > 1/2). They solve the model's equations with source terms S_f (planted.rs) that this
script forms in mpmath at 50 digits from the closed forms sigma^2 = (1 - cos 2 beta x)/(2 kappa^2) and
sigma gamma = sin(2 beta x)/(2 kappa), beta = kappa q, with the grid's own d/dtheta (the double-double matrix the kernel
gets) and omega = 2 pi_dd / Delta* as the kernel forms it, so the planted samples solve the collocation system to
rounding. Nothing here shares code with the kernel's recursions.

Plan, written before the runs: from z* + delta (delta 1e-3 relative on every unknown, Delta 1e-3 off), the same Newton as
the real runs (smdriver.newton_solve) must reach max |F| < 1e-28 within 10 steps, at max |z - z*| < 1e-26 and
|Delta - Delta*| < 1e-28; then a quad-double Newton (smdriver.newton_qd) from that point must reach max |F| < 1e-50 with
|Delta - Delta*| < 1e-30 (the sources are double-double, so z* solves the quad-double system only to about 1e-32).
Exit 0 when every check passes, 1 otherwise.

Controls (each must make the check fail, exit 1):
  --mutate gamma       gamma's recursion in the Taylor steps with kappa^2 scaled by 1 + 2^-30 (model.rs MUT_GAMMA)
  --mutate sigma_h     the 2n of the a^3 sigma^2 term of 2 x a_x in the Taylor steps scaled by 1 + 2^-30 (MUT_SIGMA_H)
  --mutate dropcol     the Jacobian without one column's block derivative (multishoot's Control::DropColumn)
  --mutate qd_state    the quad-double residual and Jacobian read the state in its double-double part (MUT_QD_STATE_DD)

    PYTHONPATH=<site> python3 planted.py [--M 8] [--mutate NAME] [--threads 4] [--out FILE.json]
"""
import argparse
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smdriver as sd  # noqa: E402
from multidouble import dd_from_mp  # noqa: E402

MUT = dict(gamma=1, sigma_h=2, qd_state=9)
mp.mp.dps = 50


# ------------------------------------------------------------------ polynomials (lists of mpf, lowest order first)
def padd(*ps):
    n = max(len(p) for p in ps)
    return [sum((p[i] if i < len(p) else 0) for p in ps) for i in range(n)]


def pscale(p, s):
    return [s * c for c in p]


def pmul(*ps):
    out = [mp.mpf(1)]
    for p in ps:
        r = [mp.mpf(0)] * (len(out) + len(p) - 1)
        for i, a in enumerate(out):
            for j, b in enumerate(p):
                r[i + j] += a * b
        out = r
    return out


def pder(p):
    return [i * p[i] for i in range(1, len(p))] or [mp.mpf(0)]


def xmul(p):
    return [mp.mpf(0)] + list(p)


def peval(p, x):
    x = mp.mpf(x)
    return sum(c * x ** i for i, c in enumerate(p))


def planted_problem(M, n=2, k2=1, Delta=1.3):
    """the planted fields' coefficients at each point, the sources (5, 3, deg + 1, M) and beta"""
    th = [mp.pi * j / M for j in range(M)]
    q = [mp.mpf("0.4") * mp.cos(t) + mp.mpf("0.1") * mp.cos(3 * t) + mp.mpf("0.05") * mp.sin(3 * t) for t in th]
    P2 = [mp.mpf("0.15") * mp.sin(t) + mp.mpf("0.1") * mp.cos(t) for t in th]
    P3 = [mp.mpf("0.05") * mp.sin(t) - mp.mpf("0.08") * mp.cos(3 * t) for t in th]
    A2 = [mp.mpf("0.12") + mp.mpf("0.04") * mp.cos(2 * t) for t in th]
    C0 = [mp.mpf("0.65") + mp.mpf("0.06") * mp.cos(2 * t) + mp.mpf("0.03") * mp.sin(2 * t) for t in th]
    dh, dl = sd.dmat_odd(M)
    D = [[mp.mpf(float(dh[i, j])) + mp.mpf(float(dl[i, j])) for j in range(M)] for i in range(M)]
    Dv = lambda f: [sum(D[i][j] * f[j] for j in range(M)) for i in range(M)]
    Dq, DP2, DP3 = Dv(q), Dv(P2), Dv(P3)
    two_pi = mp.mpf(6.283185307179586) + mp.mpf(2.4492935982947064e-16)       # multidouble's DD::TWO_PI
    om = two_pi / mp.mpf(Delta)
    n = mp.mpf(n)
    k2 = mp.mpf(k2)
    kap = mp.sqrt(k2)
    fields, src, beta = [], [], []
    for j in range(M):
        p = [0, q[j], P2[j], P3[j]]
        m = [0, q[j], -P2[j], -P3[j]]
        a = [1, 0, A2[j]]
        c = [C0[j], 0, 1 - C0[j]]
        psi = [0, q[j]]
        Dp = [0, Dq[j], DP2[j], DP3[j]]
        Dm = [0, Dq[j], -DP2[j], -DP3[j]]
        p, m, a, c, psi, Dp, Dm = ([mp.mpf(v) for v in f] for f in (p, m, a, c, psi, Dp, Dm))
        xcx = xmul(pder(c))
        a2 = pmul(a, a)
        ca2 = pmul(c, a2)
        a3 = pmul(a, a2)
        pm = padd(p, m)
        # x c_x = G + S_c, G = (n-1)(c a^2 - c) - n c a^2 (1 - cos 2bx)/kappa^2
        Sc0 = padd(xcx, pscale(padd(ca2, pscale(c, -1)), -(n - 1)), pscale(ca2, n / k2))
        Sc1 = pscale(ca2, -n / k2)
        # 2 x a_x = H + S_a, H = a (p^2 + m^2) + (n-1)(a - a^3) + n a^3 (1 - cos 2bx)/kappa^2
        Sa0 = padd(pscale(xmul(pder(a)), 2), pscale(pmul(a, padd(pmul(p, p), pmul(m, m))), -1),
                   pscale(padd(a, pscale(a3, -1)), -(n - 1)), pscale(a3, -n / k2))
        Sa1 = pscale(a3, n / k2)
        # x psi_x = (p + m)/2 + S_psi
        Sq0 = padd(xmul(pder(psi)), pscale(pm, mp.mpf(-0.5)))
        # x (x - c) p_x = -omega x Dp + (x c_x) p - c p + (n/2) c (p + m) - n c a^2 sin(2bx)/(2 kappa) + S_p
        xmc = padd([0, 0, 1], pscale(xmul(c), -1))
        xpc = padd([0, 0, 1], xmul(c))
        Sp0 = padd(pmul(xmc, pder(p)), pscale(xmul(Dp), om), pscale(pmul(xcx, p), -1), pmul(c, p),
                   pscale(pmul(c, pm), -n / 2))
        Sp2 = pscale(ca2, n / (2 * kap))
        # x (x + c) m_x = -omega x Dm - (x c_x) m + c m - (n/2) c (p + m) + n c a^2 sin(2bx)/(2 kappa) + S_m
        Sm0 = padd(pmul(xpc, pder(m)), pscale(xmul(Dm), om), pmul(xcx, m), pscale(pmul(c, m), -1),
                   pscale(pmul(c, pm), n / 2))
        Sm2 = pscale(ca2, -n / (2 * kap))
        zero = [mp.mpf(0)]
        src.append([[Sp0, zero, Sp2], [Sm0, zero, Sm2], [Sa0, Sa1, zero], [Sc0, Sc1, zero], [Sq0, zero, zero]])
        fields.append(dict(p=p, m=m, a=a, c=c, psi=psi))
        beta.append(kap * q[j])
    deg = max(len(P) for s in src for f in s for P in f)
    coef = [[[[(src[j][f][t][d] if d < len(src[j][f][t]) else mp.mpf(0)) for j in range(M)] for d in range(deg)]
             for t in range(3)] for f in range(5)]
    flat = [v for f in coef for t in f for d in t for v in d]
    ch, cl = dd_from_mp(flat)
    bh, bl = dd_from_mp(beta)
    return dict(q=q, C0=C0, fields=fields, coef=(ch.reshape(5, 3, deg, M), cl.reshape(5, 3, deg, M)), beta=(bh, bl),
                n=n, k2=k2, Delta=Delta, src_q_max=max(abs(v) for s in src for v in s[4][0]))


def planted_z(pp, nodes):
    """z* (hi, lo) of the planted fields at the nodes"""
    M = len(pp["q"])
    vals = list(pp["q"]) + list(pp["C0"])
    for x in nodes:
        for f in ("p", "m", "a", "c", "psi"):
            vals += [peval(pp["fields"][j][f], x) for j in range(M)]
    for f in ("m", "a", "psi"):
        vals += [peval(pp["fields"][j][f], 1) for j in range(M)]
    vals.append(mp.mpf(pp["Delta"]))
    return dd_from_mp(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--mutate", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    M = a.M
    nodes = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9]
    pp = planted_problem(M)
    assert pp["src_q_max"] < 1e-45, "psi's source must vanish"
    s = sd.Solver(M, 2.0, 0.0, 1.0, 0.0, *sd.dmat_odd(M), *sd.phase_weights(M), nodes, [], 30, 40, 40, a.threads)
    s.set_planted(pp["beta"][0], pp["beta"][1], *pp["coef"])
    zh, zl = planted_z(pp, nodes)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    steps = sum(len(g) - 1 for g in s.get_segs())
    F_star = sd.fnorm(s.residual([zh, zl]))
    print(f"planted M = {M}, L = {s.size}, {steps} Taylor steps; max |F(z*)| = {F_star:.2e}", flush=True)
    if a.mutate in MUT:
        s.set_mutate(MUT[a.mutate])
    if a.mutate == "dropcol":
        col = 2 * M + 5 * M * 3 + 2 * M + 3          # node 3's field a, point 3
        s.set_drop_col(col)
        print(f"control: Jacobian column {col} dropped", flush=True)
    rng = np.random.default_rng(1)
    pert = 1e-3 * rng.uniform(-1, 1, len(zh)) * np.maximum(np.abs(zh), 0.1)
    pert[-1] = 1e-3 * zh[-1]
    z0h = zh + pert
    rh, rl, hist = sd.newton_solve(s, z0h, np.zeros_like(zl), tol=1e-28, it_max=10)
    conv = bool(hist and hist[-1].get("converged"))
    err = float(np.max(np.abs((rh - zh) + (rl - zl))))
    dD = float((rh[-1] - zh[-1]) + (rl[-1] - zl[-1]))
    n_it = len(hist) - 1
    ok_dd = conv and n_it <= 10 and err < 1e-26 and abs(dD) < 1e-28
    print(f"double-double: converged {conv} in {n_it} steps, max |z - z*| {err:.2e}, Delta - Delta* {dD:+.2e} -> "
          f"{'PASS' if ok_dd else 'FAIL'}", flush=True)
    zq, hq = sd.newton_qd(s, sd.qparts(rh, rl), tol=1e-50, it_max=6)
    convq = bool(hq and hq[-1].get("converged"))
    dDq = float(mp.mpf(zq[0][-1]) + zq[1][-1] + zq[2][-1] + zq[3][-1] - (mp.mpf(zh[-1]) + zl[-1]))
    ok_qd = convq and abs(dDq) < 1e-30
    print(f"quad-double: converged {convq} (|F| {hq[-1]['F']:.2e}), Delta - Delta* {dDq:+.2e} -> "
          f"{'PASS' if ok_qd else 'FAIL'}", flush=True)
    rec = dict(M=M, mutate=a.mutate, steps=steps, F_star=F_star, dd=dict(converged=conv, steps=n_it, err=err, dDelta=dD,
               F=[h["F"] for h in hist], pass_=ok_dd), qd=dict(converged=convq, F=[h["F"] for h in hq], dDelta=dDq,
               pass_=ok_qd))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    ok = ok_dd and ok_qd
    print("PLANTED", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
