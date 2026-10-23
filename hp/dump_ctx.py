#!/usr/bin/env python3
"""Write a saved state and its solver set-up (as profile_newton.py builds it) to a flat little-endian float64 file
for the Rust benches and tests (model.rs: bench_jac, the f64-Jacobian agreement test):

    M, n_hi, n_lo, kt, kc, ks, K, nodes[K], K-1 segments as (len, xs[len]), dmat_hi[M*M], dmat_lo[M*M] (row-major),
    phase_hi[M], phase_lo[M], L, zh[L], zl[L]

    cd hp && python3 dump_ctx.py data/lowD_states/lowDv_D3.7000_M48.npz OUT.bin
"""
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import dd_from_mp, dmat_odd, load, make_solver, phase_weights  # noqa: E402


def main():
    path, out = sys.argv[1], sys.argv[2]
    zh, zl, m = load(path)
    M = m["M"]
    s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=1)
    segs = m.get("segs") or s.plan(zh, zl, 1e-30, 0.05)
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(m["D"]) - 2])
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    v = [M, nh[0], nl[0], m["kt"], m["kc"], m["ks"], len(m["nodes"])] + list(m["nodes"])
    for g in segs:
        v += [len(g)] + list(g)
    v += list(dh.ravel()) + list(dl.ravel()) + list(ph) + list(pl) + [len(zh)] + list(zh) + list(zl)
    np.asarray(v, dtype="<f8").tofile(out)
    print(f"{out}: M {M}, L {len(zh)}, {sum(len(g) - 1 for g in segs)} Taylor steps")


if __name__ == "__main__":
    main()
