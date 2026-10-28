#!/usr/bin/env python3
"""Planted truth for the quad-double solver at a large-D state (AGENTS.md: planted truth before real data;
notes/QUAD_DOUBLE.md, section 12). G(z) = F(z) - F(z*) has the exact root z*, the given state, on the real system
(its context, its weakest directions). From z* + eps v (v_i = +-1, a fixed pattern, Delta included) the quad-double
Newton (Solver.newton_q, the step for G by Solver.qlu_solve) and the double-double Newton (Solver.newton, lu_solve)
each take --steps steps; after each, max |z - z*| and |Delta - Delta*| are recorded, in quad-double.

Pass criteria, fixed before the run: for eps = --eps_pass the quad-double Newton ends with |Delta - Delta*| below
1e-33 and max |z - z*| below 1e-25 (the error bound cond(J) 2^-209 |z| is about 1e-27 at D = 73, where u kappa is
1e3 in double-double; Delta weighs the weakest direction by about 1e-8). Control: the same run with the state read
in its double-double part (qsolve::QD_MUT_STATE_DD) must fail them. Exits 1 if a criterion or the control fails.
The default starts lie inside Newton's basin at D = 72.8 (radius about 4e-38): from 1e-24 every solver's first step
lands 2.3e-11 away (notes/QUAD_DOUBLE.md, 12.1).

    python3 qd_planted.py OUT.jsonl STATE.npz [--kfreeze 13] [--eps 1e-40,1e-36,1e-32,1e-28] [--eps_pass 1e-40]
        [--steps 5] [--threads N]

OUT must be in a raw-data folder (raw_out.py; --allow_unraw writes anyway).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from multidouble import qd_axpy  # noqa: E402
from hpkernel.driver import load_q, make_solver, qmax  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402

QD_MUT_STATE_DD = 9     # hp/src/qsolve.rs


def errors(z, zs):
    """(max |z - z*|, |Delta - Delta*|) in quad-double, rounded to f64"""
    d = qd_axpy(-1.0, *zs, *z)
    e = np.abs(d[0] + d[1])
    return float(e.max()), float(e[-1])


def run_qd(s, zs, fs, z, steps, mutate):
    """Newton in quad-double on G = F - F(z*): dz = -J^-1 F(z) + J^-1 F(z*)"""
    out = [errors(z, zs)]
    for _ in range(steps):
        d0, d1, d2, d3, _, _, _, _ = s.newton_q(*z, mutate=mutate)
        e = s.qlu_solve(*fs)
        z = qd_axpy(1.0, *e, *qd_axpy(1.0, d0, d1, d2, d3, *z))
        out.append(errors(z, zs))
    return out


def run_dd(s, zs, z, steps):
    """Newton in double-double on G = F - F(z*), F the double-double residual, J its double-double Jacobian"""
    fh, fl = s.residual(zs[0], zs[1])
    zh, zl = z[0].copy(), z[1].copy()
    zero = np.zeros(len(zh))
    out = [errors((zh, zl, zero, zero), zs)]
    for _ in range(steps):
        dh, dl, _, _, _, _ = s.newton(zh, zl)
        eh, el = s.lu_solve(np.ascontiguousarray(fh), np.ascontiguousarray(fl))
        w = qd_axpy(1.0, np.asarray(eh), np.asarray(el), zero, zero, *qd_axpy(1.0, dh, dl, zero, zero, zh, zl, zero, zero))
        zh, zl = np.ascontiguousarray(w[0] + 0.0), np.ascontiguousarray(w[1] + (w[2] + w[3]))
        out.append(errors((zh, zl, zero, zero), zs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("state")
    ap.add_argument("--kfreeze", type=int, default=13)
    ap.add_argument("--eps", default="1e-40,1e-36,1e-32,1e-28")
    ap.add_argument("--eps_pass", type=float, default=1e-40)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out, allow=a.allow_unraw)
    zs, meta = load_q(a.state)
    D, M = meta["D"], meta["M"]
    s = make_solver(D, M, nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 146), ks=meta.get("ks", 130),
                    threads=a.threads, kfreeze=a.kfreeze)
    s.set_segs(s.plan(zs[0], zs[1], a.tol_step, a.hmax))
    fs = s.residual_q(*zs)
    l = len(zs[0])
    v = np.where(np.arange(l) % 3 == 1, -1.0, 1.0)
    print(f"D {D}, M {M}, L {l}; |F(z*)| {qmax(fs):.2e} (the planted residual subtracts it)", flush=True)
    ok = True
    for eps in [float(x) for x in a.eps.split(",")]:
        z0 = qd_axpy(eps, v, np.zeros(l), np.zeros(l), np.zeros(l), *zs)
        for name, fn in (("qd", lambda: run_qd(s, zs, fs, z0, a.steps, 0)),
                         ("qd_state_dd", lambda: run_qd(s, zs, fs, z0, a.steps, QD_MUT_STATE_DD)),
                         ("dd", lambda: run_dd(s, zs, z0, a.steps))):
            t0 = time.time()
            errs = fn()
            rec = dict(state=a.state, D=D, kfreeze=a.kfreeze, eps=eps, solver=name, z_err=[e[0] for e in errs],
                       Delta_err=[e[1] for e in errs], seconds=round(time.time() - t0, 1))
            if eps == a.eps_pass and name != "dd":
                passed = errs[-1][1] < 1e-33 and errs[-1][0] < 1e-25
                rec["pass"] = passed
                if name == "qd" and not passed:
                    ok = False
                if name == "qd_state_dd" and passed:
                    ok = False
                    print("CONTROL FAILED: the state read in double-double met the criteria", flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"eps {eps:.0e} {name:>11}: max|z - z*| " + " ".join(f"{e[0]:.1e}" for e in errs)
                  + "; |Delta - Delta*| " + " ".join(f"{e[1]:.1e}" for e in errs)
                  + (f"; pass {rec['pass']}" if "pass" in rec else "") + f" ({rec['seconds']} s)", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
