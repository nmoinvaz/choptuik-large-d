#!/usr/bin/env python3
"""The small check of the largest sizes (notes/HIGH_M.md 9.2): a converged state resampled to each M, its own Taylor
plan (1e-30, 0.05), the residual (|F|, seconds), the series tails and the peak RSS; with --newton also one Newton step
(Solver.newton_mixed, block storage) on the same plan or a coarser one (--newton_plan=tol,hmax): its seconds, the
Jacobian's and the factorization's, the block LU's MB and the peak RSS. Records in data/lowD_highM/check512.jsonl.

    cd hp && python3 highm_check512.py [--start=STATE.npz] [--newton] [--newton_plan=own|thinK|tol,hmax] [--kc=K --ks=K] M [M ...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lowD_dd  # noqa: E402
from hpkernel.driver import load, make_solver, resample_z  # noqa: E402


def rss():
    return lowD_dd.peak_rss_mb()


def main():
    opts = dict((a[2:].split("=", 1) + [True])[:2] for a in sys.argv[1:] if a.startswith("--"))
    Ms = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [384, 512]
    start = opts.get("start", "data/lowD_states/highm15_D3.7000_M256.npz")
    zh, zl, m = load(start)
    for M in Ms:
        z, _ = resample_z(np.asarray(zh) + np.asarray(zl), m["M"], M, len(m["nodes"]))
        t = time.time()
        kc, ks = int(opts.get("kc", m["kc"])), int(opts.get("ks", m["ks"]))
        s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=kc, ks=ks, threads=2)
        t_make = time.time() - t
        t = time.time()
        sg = s.plan(z, 0 * z, 1e-30, 0.05)
        t_plan = time.time() - t
        s.set_segs(sg)
        t = time.time()
        Fh, Fl = s.residual(z, 0 * z)
        t_res = time.time() - t
        F = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        tc, th = lowD_dd.tails(s, z, 0 * z, m["nodes"], M)
        rec = dict(M=M, D=m["D"], start=os.path.basename(start), nodes=len(m["nodes"]), kc=kc, ks=ks, N=s.size,
                   steps=sum(len(g) - 1 for g in sg), F=F, s_make=round(t_make, 1), s_plan=round(t_plan, 1),
                   s_residual=round(t_res, 1), tail_c=tc, tail_h=th, peak_rss_mb_residual=rss())
        print(json.dumps(dict(rec, peak_rss_mb=rss(), load=lowD_dd.load_avg())), flush=True)
        if opts.get("newton"):
            # "own": the residual's plan; a coarser plan can overflow (a non-finite pivot) near a first node at 0.000625
            if str(opts.get("newton_plan", "own")) == "own":
                tol, hmax, sg2 = 1e-30, 0.05, sg
            elif str(opts["newton_plan"]).startswith("thin"):
                # the residual's plan with every k-th step point kept (the segment ends always): a shorter check
                k = int(str(opts["newton_plan"])[4:])
                tol, hmax = 1e-30, 0.05
                sg2 = [[g[i] for i in range(len(g)) if i % k == 0 or i == len(g) - 1] for g in sg]
            else:
                tol, hmax = (float(v) for v in str(opts["newton_plan"]).split(","))
                sg2 = s.plan(z, 0 * z, tol, hmax)
            s.set_segs(sg2)
            t = time.time()
            dh, dl, _, _, p0, p1 = s.newton_mixed(z, 0 * z, block=True)
            t_nm = time.time() - t
            lu_bytes, _, t_jac, t_fac = s.lu_info()
            # the step's effect: |F(z + dz)| on the residual's plan (a Newton step from a resampled start: far below |F|)
            s.set_segs(sg)
            th, tl = np.asarray(dh, float), np.asarray(dl, float)
            F1 = float(np.abs(np.add(*s.residual(z + th, tl))).max())
            rec.update(newton_plan=[tol, hmax], newton_steps=sum(len(g) - 1 for g in sg2), s_newton=round(t_nm, 1),
                       s_jacobian=round(t_jac, 1), s_factor=round(t_fac, 2), lu_mb=round(lu_bytes / 2 ** 20, 1),
                       pivots=[p0, p1], dDelta=float(dh[-1]) + float(dl[-1]), F_after_step=F1)
        rec.update(peak_rss_mb=rss(), load=lowD_dd.load_avg())
        print(json.dumps(rec), flush=True)
        with open(os.path.join("data", "lowD_highM", "check512.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
