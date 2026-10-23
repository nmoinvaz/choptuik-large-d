#!/usr/bin/env python3
"""Which part of the horizon block has to be double-double for the mixed-precision Newton step to steer at large D
(notes/LARGE_D_MIXED.md section 1; notes/MIXED_PRECISION.md section 8 found that replacing the f64 Jacobian's horizon
rows by exact ones repairs the step)?

At each saved state (hp/data/states, nearest to each --D; kfreeze 13 from D = 35, as the scan ran), perturbed by
--eps relative (seeds --seeds, the perturbation of hp/mixed_largeD_blocks.py), the f64 Jacobian is built with each
`horizon_dd` variant of Solver.jacobian_f64 (bit flags):
    0   all f64 (the default)
    1   the horizon series' periodic solves in double-double (value and tangents; the f64 path uses the hi parts of
        the double-double LU factors there)
    2   no early end of the horizon series for the f64 tangents
    3   1 + 2
    8   the last segment's (block K - 1) columns in double-double
    16  the horizon block on a double-double base trajectory (f64 tangents, Dtn<4> lanes)
    17  16 + 1: double-double base trajectory and double-double solves, f64 tangent arithmetic
    4   the horizon block's columns in double-double (Du lanes, the exact Jacobian's columns), rounded to f64
and the Newton step solved with its f64 LU (scipy) is compared with the double-double step (Solver.newton): the
relative difference |dz - dz_dd| / |dz_dd| is the mixed iteration's contraction factor. Also printed: the horizon
rows' largest entry error against the double-double Jacobian's hi part, and the Jacobian's time (the dd one first).
Pass for a variant: step difference <= --pass (1e-5) at every state and seed.

    cd hp && python3 largeD_horizon.py [--D=50,55,60] [--seeds=1,2] [--variants=0,1,2,3,8,16,17,4] [--threads=2]
                                       [--out=data/mixed_precision/largeD_horizon.jsonl]
Each (state, seed) record is appended to --out as soon as it is done; a rerun skips the ones present.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import load, make_solver  # noqa: E402


def states():
    return {float(re.search(r"_D([0-9.]+)\.npz$", p).group(1)): p for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz"))}


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    Ds = [float(x) for x in opts.get("D", "50,55,60").split(",")]
    seeds = [int(x) for x in opts.get("seeds", "1,2").split(",")]
    variants = [int(x) for x in opts.get("variants", "0,1,2,3,8,16,17,4").split(",")]
    threads, eps, pas = int(opts.get("threads", 2)), float(opts.get("eps", 1e-12)), float(opts.get("pass", 1e-5))
    out = opts.get("out", os.path.join(HERE, "data", "mixed_precision", "largeD_horizon.jsonl"))
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.setdefault((round(r["D"], 4), r["seed"]), set()).update(r["step"])
    avail = states()
    worst = {v: 0.0 for v in variants}
    for D0 in Ds:
        p = avail[min(avail, key=lambda d: abs(d - D0))]
        zh, zl, m = load(p)
        kf = 13 if m["D"] >= 35 else None
        s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        n, M = s.size, m["M"]
        K = len(m["nodes"])
        hr = slice(4 * M * K, 4 * M * (K + 1))     # the horizon block's rows
        for seed in seeds:
            if {str(v) for v in variants} <= done.get((round(m["D"], 4), seed), set()):
                print(f"D {m['D']:.4f} seed {seed}: done, skipped", flush=True)
                continue
            rng = np.random.default_rng(seed)
            ph = np.asarray(zh) * (1 + eps * rng.standard_normal(n))
            pl = np.asarray(zl).copy()
            t = time.time()
            Jh, _ = s.jacobian(ph, pl)
            t_dd = time.time() - t
            Jh = np.asarray(Jh)
            dh, dl, Fh, Fl, *_ = s.newton(ph, pl)
            ddz = np.asarray(dh) + np.asarray(dl)
            b = -(np.asarray(Fh) + np.asarray(Fl))
            nr = np.linalg.norm(ddz)
            rec = dict(D=m["D"], M=M, N=n, kfreeze=kf, seed=seed, eps=eps, t_jac_dd=round(t_dd, 3), step={}, jac_horizon={}, t_jac={})
            line = []
            for v in variants:
                t = time.time()
                J = np.asarray(s.jacobian_f64(ph, pl, horizon_dd=v))
                tj = time.time() - t
                c = float(np.linalg.norm(sla.lu_solve(sla.lu_factor(J, check_finite=False), b, check_finite=False) - ddz) / nr)
                jd = float(np.max(np.abs(J[hr] - Jh[hr])) / np.max(np.abs(Jh[hr])))
                rec["step"][str(v)], rec["jac_horizon"][str(v)], rec["t_jac"][str(v)] = c, jd, round(tj, 3)
                worst[v] = max(worst[v], c)
                line.append(f"{v}: {c:.1e} (J {jd:.0e}, {tj:.2f} s)")
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"D {m['D']:.4f} seed {seed} (dd Jacobian {t_dd:.2f} s): " + "; ".join(line), flush=True)
    print("worst step difference per variant (this run): " + ", ".join(f"{v}: {worst[v]:.1e}{' PASS' if worst[v] <= pas else ''}" for v in variants))


if __name__ == "__main__":
    main()
