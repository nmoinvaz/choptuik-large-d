#!/usr/bin/env python3
"""Why does qdh_delta_check.py's control fail? (comms 2026-10-25 22:22 UTC mac: Delta moved by 1e-14 at D 61.66 gave a
linearised correction of -1.1e-10.) At one state:

    cd hp && python3 qdh_delta_diag.py [STATE.npz] [--threads=N] [--out=data/large_d/qdh_delta_diag.jsonl]

1. Linearity: Delta moved by delta = 1e-16 .. 1e-12; the correction -y.F (driver.delta_correction, y = J^-T e_Delta
   from the double-double Jacobian's LU) from the double-double residual and from the quad-double-horizon one, each
   minus its value at the unmoved state, divided by -delta (1 if the estimator sees the move).
2. The Jacobian's Delta column: J e_Delta (Solver.jvp, the double-double Jacobian) against the central difference of
   the quad-double-horizon residual, (F_q(z + h e_Delta) - F_q(z - h e_Delta)) / 2h for h = 1e-14, 1e-12, 1e-10, per
   block (horizon rows, the others) and as y.(difference) (what it does to the correction).
3. The step solved with quad-double-horizon products (QdhSolver with gcr, the driver of wip/qd-horizon 5272dae): its
   Delta component at the unmoved state and at the state moved by 1e-14 (which must come back as -1e-14 more).
Prints a table per part; no pass/fail (a measurement).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import QdhSolver, dd_add, delta_correction, load, make_solver  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    path = args[0] if args else os.path.join(HERE, "data", "large_d", "up70_D61.6587.npz")
    out = opts.get("out", os.path.join(HERE, "data", "large_d", "qdh_delta_diag.jsonl"))
    zh, zl, meta = load(path)
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    M, K = meta["M"], len(meta["nodes"])
    s = make_solver(meta["D"], M, nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                    threads=int(opts.get("threads", 2)), kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    L = s.size
    hb = np.zeros(L, bool)
    hb[K * 4 * M:(K + 1) * 4 * M] = True
    z0 = np.zeros(L)
    rec = dict(state=os.path.basename(path), D=meta["D"], lin=[], col=[], gcr=[])

    def moved(delta):
        e = np.zeros(L)
        e[-1] = delta
        return tuple(np.ascontiguousarray(v) for v in dd_add(zh, zl, e, z0))

    s.newton(zh, zl)                  # the LU at the unmoved state, used throughout parts 1 and 2
    Fd0 = s.residual(zh, zl)
    Fq0 = s.residual_qdh(zh, zl, z0, z0)
    c_d0, c_q0 = delta_correction(s, *Fd0), delta_correction(s, *Fq0)
    print(f"D {meta['D']}: corrections at the state: dd {c_d0:+.3e}, qdh {c_q0:+.3e}", flush=True)
    print("1. Delta moved by delta: (correction - its value at the state) / (-delta), dd and qdh residual", flush=True)
    for delta in (1e-16, 1e-15, 1e-14, 1e-13, 1e-12):
        ph, pl = moved(delta)
        rd = (delta_correction(s, *s.residual(ph, pl)) - c_d0) / -delta
        rq = (delta_correction(s, *s.residual_qdh(ph, pl, z0, z0)) - c_q0) / -delta
        rec["lin"].append(dict(delta=delta, ratio_dd=rd, ratio_qdh=rq))
        print(f"   {delta:.0e}: dd {rd:+.4e}   qdh {rq:+.4e}", flush=True)
    e = np.zeros(L)
    e[-1] = 1.0
    jh, jl = s.jvp(zh, zl, e, z0)[:2]
    J = np.asarray(jh) + np.asarray(jl)
    ey = np.zeros(L)
    ey[-1] = 1.0
    yh, yl = s.lu_solve_t(ey, z0)
    y = np.asarray(yh) + np.asarray(yl)
    print(f"2. the Delta column: |J e_Delta| horizon {np.abs(J[hb]).max():.3e}, others {np.abs(J[~hb]).max():.3e}; "
          f"y.J e_Delta {float(y @ J):+.6e} (must be 1); |y| horizon {np.abs(y[hb]).max():.2e}, others "
          f"{np.abs(y[~hb]).max():.2e}", flush=True)
    for h in (1e-14, 1e-12, 1e-10):
        ph, pl = moved(h)
        mh, ml = moved(-h)
        a, b = s.residual_qdh(ph, pl, z0, z0), s.residual_qdh(mh, ml, z0, z0)
        g = ((np.asarray(a[0]) - np.asarray(b[0])) + (np.asarray(a[1]) - np.asarray(b[1]))) / (2 * h)
        d = g - J
        rec["col"].append(dict(h=h, horizon=float(np.abs(d[hb]).max()), others=float(np.abs(d[~hb]).max()),
                               y_dot=float(y @ d), y_dot_horizon=float(y[hb] @ d[hb])))
        print(f"   h {h:.0e}: |FD_q - J e_Delta| horizon {np.abs(d[hb]).max():.3e}, others {np.abs(d[~hb]).max():.3e}; "
              f"y.(FD_q - J e_Delta) {float(y @ d):+.4e} (horizon rows {float(y[hb] @ d[hb]):+.4e})", flush=True)
    print("3. the step with quad-double-horizon products (QdhSolver gcr): its Delta component", flush=True)
    for delta in (0.0, 1e-14):
        ph, pl = moved(delta) if delta else (zh, zl)
        qg = QdhSolver(s, horizon=slice(K * 4 * M, (K + 1) * 4 * M), gcr=dict(max_products=int(opts.get("products", 16)),
                                                                              step_tol=1e-12))
        dh, dl, _, _, _, _ = qg.newton(ph, pl)
        dD = float(dh[-1] + dl[-1])
        info = qg.last_gcr
        rec["gcr"].append(dict(delta=delta, dDelta=dD, products=info["products"], converged=info["converged"],
                               changes=info["changes"]))
        print(f"   moved by {delta:.0e}: Delta component {dD:+.4e} ({info['products']} products, converged "
              f"{info['converged']}, last change {info['changes'][-1]:.1e})", flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
