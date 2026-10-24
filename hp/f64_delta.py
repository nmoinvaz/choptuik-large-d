#!/usr/bin/env python3
"""What does double precision do to Delta? For a converged double-double state, the solution of the same discretised
equations with the residual evaluated in plain f64 (Solver.residual_f64: every operation in f64, the inputs rounded
to f64), as a double-precision code would find it.

First the linearised estimate -y . F64(z_dd), y = J^-T e_Delta: the change in Delta that F64's rounding at the dd
solution implies. Then a deflated chord Newton with the stored LU of the double-double Jacobian at the state: z rounded
to f64, then z <- fl64(z + dz), dz = -J^-1 F64(z) with its component in the --w (16) weakest right singular directions
removed (driver.weak_subspace), as the scans' deflated Newton does: a plain step diverges, because F64's rounding,
divided by singular values down to 1e-18 of the largest, throws z out of the basin. z stays in f64 throughout, as in a
double-precision code. Printed per step: max|F64| and Delta_f64 - Delta_dd; the mean of the last half of the steps
minus Delta_dd is the bias, their spread the f64 noise in Delta.

Check, with a control: at the state perturbed by 1e-7 (relative, seed 1), F64 must agree with the double-double F to
at most 1e-6 of max|F| there (F is about 1e-7 in size, so the f64 rounding is far below); the control compares F64 at
the perturbed state with F at the unperturbed one, which must disagree by more. Exit 1 if either fails.

    cd hp && python3 f64_delta.py STATE.npz [STATE ...] [--its=8] [--w=16] [--threads=2] [--plan=1e-32,0.05] [--json=F]
States from hp/data/states (large D: p1 harmonics above 13 frozen from D = 35, as the scans ran) or
hp/data/lowD_states (their own plan tolerance); M <= 128 (the dense double-double LU).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver, weak_subspace  # noqa: E402


def run(path, its, threads, plan, w):
    zh, zl, m = load(path)
    D, M = m["D"], m["M"]
    kf = 13 if D >= 35 else None
    s = make_solver(D, M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, *plan))
    zh, zl = np.asarray(zh, float), np.asarray(zl, float)
    d_dd = float(zh[-1]) + float(zl[-1])
    # check and control
    rng = np.random.default_rng(1)
    ph = zh * (1 + 1e-7 * rng.standard_normal(zh.size))
    Fp = np.asarray(s.residual(ph, zl)[0]) + np.asarray(s.residual(ph, zl)[1])
    F0 = np.asarray(s.residual(zh, zl)[0]) + np.asarray(s.residual(zh, zl)[1])
    F64p = np.asarray(s.residual_f64(ph, zl))
    chk = float(np.abs(F64p - Fp).max() / np.abs(Fp).max())
    ctl = float(np.abs(F64p - F0).max() / np.abs(Fp).max())
    # the stored LU at the dd solution, and the sensitivity y = J^-T e_Delta
    t = time.time()
    s.newton(zh, zl)
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    y = np.asarray(s.lu_solve_t(e, np.zeros(L))[0])
    F64 = np.asarray(s.residual_f64(zh, zl))
    lin = float(-y @ F64)
    print(f"D {D:8.4f} M {M}: |F| dd {np.abs(F0).max():.1e}, f64 at the dd solution {np.abs(F64).max():.1e}; "
          f"linearised Delta shift {lin:+.2e}", flush=True)
    # deflated chord Newton in f64
    V = weak_subspace(s, w) if w else np.zeros((L, 0))
    z = zh + zl                                   # rounded to f64
    hist = []
    for _ in range(its):
        F = np.asarray(s.residual_f64(z, np.zeros(L)))
        if not np.all(np.isfinite(F)):
            hist.append((float("nan"), float("nan")))
            break
        dz = np.asarray(s.lu_solve(-F, np.zeros(L))[0])
        dz = dz - V @ (V.T @ dz)
        z = z + dz
        hist.append((float(np.abs(F).max()), float(z[-1]) - d_dd))
    tail = [h[1] for h in hist[len(hist) // 2:]]
    rec = dict(D=D, M=M, Delta_dd=[float(zh[-1]), float(zl[-1])], F_dd=float(np.abs(F0).max()),
               F64_at_dd=float(np.abs(F64).max()), lin=lin, hist=hist, bias=float(np.mean(tail)),
               spread=float(np.max(tail) - np.min(tail)), check=chk, control=ctl, secs=round(time.time() - t, 1))
    print(f"  deflated f64 chord Newton: Delta_f64 - Delta_dd = " +
          " ".join(f"{d:+.1e}" for _, d in hist) + f" (|F64| " + " ".join(f"{f:.0e}" for f, _ in hist) +
          f"); bias {rec['bias']:+.2e}, spread {rec['spread']:.1e}; check {chk:.1e} (control {ctl:.1e}); "
          f"{rec['secs']} s", flush=True)
    return rec


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    its, threads = int(opts.get("its", 8)), int(opts.get("threads", 2))
    plan = tuple(float(v) for v in opts.get("plan", "1e-32,0.05").split(","))
    out, bad = [], []
    for p in paths:
        r = run(p, its, threads, plan, int(opts.get("w", 16)))
        out.append(r)
        if not r["check"] <= 1e-6:
            bad.append(f"D {r['D']}: F64 disagrees with F by {r['check']:.1e}")
        if not r["control"] > 1e-6:
            bad.append(f"D {r['D']}: the control agrees ({r['control']:.1e})")
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)
    print("PASS" if not bad else "FAIL:\n  " + "\n  ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
