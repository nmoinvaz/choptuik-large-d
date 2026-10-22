#!/usr/bin/env python3
"""Review RV2: re-solve a saved double-double state with changed discretisation, and report Delta.

    python3 hp/review/rv2_resolve.py STATE.npz [--M 48] [--kfreeze 13|0] [--kt 36] [--kc K] [--ks K]
        [--tol_step 1e-32] [--hmax 0.05] [--nodes 0.3,0.4,...] [--perturb k:amp,...] [--its 12] [--threads 1]
        [--out FILE.jsonl] [--label NAME]

Starts from the saved z (re-noded / resampled if asked), runs plain Newton (hpkernel.driver.newton_solve, exact
Jacobian, backtracking) to the residual floor, and prints Delta (hi + lo), |F|, the linearised correction
-dDelta/dF . F, the per-block max |dDelta/dF| and the time. --perturb adds amp to the cos coefficient of p1's harmonic
k (sets a known displacement along the weak 'junk' directions before re-converging). --kfreeze 0 switches the freeze
off. The defaults are the settings of the cloud runs above D = 35.35 (M 32, kfreeze 13, kt 36, tol_step 1e-32,
hmax 0.05, the state's own kc, ks and nodes).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from hpkernel.driver import (dd_add, dd_from_mp, deflated_newton, delta_correction, load, make_solver,  # noqa: E402
                             newton_solve, renode, resample_z)


def resample_dd(zh, zl, M0, M, K):
    """Change the tau grid in double-double (driver.resample_z works in f64 and drops lo, which at D ~ 60 puts
    1e-16 into the weak directions and stalls Newton): trigonometric interpolation at 40 digits, field by field
    (p1, m-like odd fields antiperiodic, a, c even), keeping the harmonics |k| < M0."""
    parity = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    out_h, out_l = [], []
    with mp.workdps(40):
        N0, N = 2 * M0, 2 * M
        for f, par in enumerate(parity):
            v = [mp.mpf(float(h)) + mp.mpf(float(l)) for h, l in zip(zh[f * M0:(f + 1) * M0], zl[f * M0:(f + 1) * M0])]
            full = v + ([-x for x in v] if par == "odd" else v)
            ks = range(1, M0, 2) if par == "odd" else range(0, M0 + 1, 2)
            co = {}
            for k in ks:
                co[k] = sum(full[j] * mp.expj(-k * 2 * mp.pi * j / N0) for j in range(N0)) / N0
            vals = []
            for j in range(M):
                th = 2 * mp.pi * j / N
                s = mp.mpf(0)
                for k, c in co.items():
                    w = 1 if (k == 0 or k == M0) else 2
                    s += w * mp.re(c * mp.expj(k * th))
                vals.append(s)
            h, l = dd_from_mp(vals)
            out_h.append(h)
            out_l.append(l)
    out_h.append(zh[-1:])
    out_l.append(zl[-1:])
    return np.concatenate(out_h), np.concatenate(out_l)


def harmonic_field(k, amp, M):
    """amp * cos(k theta_j) on the half grid, double-double"""
    with mp.workdps(40):
        return dd_from_mp([mp.mpf(amp) * mp.cos(k * 2 * mp.pi * j / (2 * M)) for j in range(M)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--kfreeze", type=int, default=13)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kc", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--perturb", default="")
    ap.add_argument("--its", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--corrector", default="newton", choices=["newton", "nols", "dnp"])
    ap.add_argument("--resample", default="dd", choices=["dd", "f64"])
    a = ap.parse_args()
    zh, zl, meta = load(a.state)
    D, M0, nodes0 = meta["D"], meta["M"], list(meta["nodes"])
    kc = a.kc or meta["kc"]
    ks = a.ks or meta["ks"]
    M = a.M or M0
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else nodes0
    t0 = time.time()
    if nodes != nodes0:
        s0 = make_solver(D, M0, nodes=nodes0, kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        s0.set_segs(s0.plan(zh, zl, a.tol_step, a.hmax))
        zh, zl = renode(s0, zh, zl, nodes0, nodes, M0)
    if M != M0:
        if a.resample == "f64":
            zh, zl = resample_z(zh + zl, M0, M, len(nodes))
        else:
            zh, zl = resample_dd(zh, zl, M0, M, len(nodes))
    for item in filter(None, a.perturb.split(",")):
        k, amp = item.split(":")
        hh, hl = harmonic_field(int(k), float(amp), M)
        zh[:M], zl[:M] = dd_add(zh[:M], zl[:M], hh, hl)
    s = make_solver(D, M, nodes=nodes, kt=a.kt, kc=kc, ks=ks, threads=a.threads, kfreeze=a.kfreeze or None)
    s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
    if a.corrector == "dnp":
        # the scans' corrector: deflated Newton (weakest directions projected out), then plain Newton to the bottom
        zh, zl, h0 = deflated_newton(s, zh, zl, tol=1e-40, it_max=a.its, w=16, verbose=not a.quiet)
    elif a.corrector == "nols":
        # full Newton steps without line search (the weak directions make |F|_2 non-monotone near the floor)
        zh, zl, h0 = newton_solve(s, zh, zl, tol=1e-40, it_max=a.its, verbose=not a.quiet, ls=False)
    zh2, zl2, hist = newton_solve(s, zh, zl, tol=1e-40, it_max=a.its, verbose=not a.quiet)
    Fh, Fl = s.residual(zh2, zl2)
    F = float(np.abs(Fh + Fl).max())
    s.newton(zh2, zl2)                       # LU at the final point for the sensitivity / correction
    est = delta_correction(s, Fh, Fl)
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, _ = s.lu_solve_t(e, np.zeros(L))
    nb = (L - 1) // (4 * M)
    sens = [float(np.abs(yh[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)]
    rec = dict(label=a.label, state=os.path.basename(a.state), D=D, M=M, kfreeze=a.kfreeze, kt=a.kt, kc=kc, ks=ks,
               tol_step=a.tol_step, hmax=a.hmax, nodes=nodes, perturb=a.perturb, corrector=a.corrector, Delta=float(zh2[-1]), Delta_lo=float(zl2[-1]),
               F=F, est=est, its=len(hist), steps=sum(len(g) - 1 for g in s.get_segs()), sens=sens,
               seconds=round(time.time() - t0, 1))
    ref = (meta.get("Delta"), None)
    print(json.dumps(rec), flush=True)
    print(f"D {D}  Delta {zh2[-1]!r} {zl2[-1]:+.3e}  (saved {zh[-1]!r} {zl[-1]:+.3e}; diff "
          f"{(zh2[-1] - load(a.state)[0][-1]) + (zl2[-1] - load(a.state)[1][-1]):+.2e})  |F| {F:.1e}  est {est:+.1e}  "
          f"{time.time() - t0:.0f}s", flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    del ref


if __name__ == "__main__":
    main()
