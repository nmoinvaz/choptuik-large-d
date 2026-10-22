#!/usr/bin/env python3
"""Weakest singular values of the double-double Jacobian at saved states (notes/LOW_D.md, conditioning), with the Delta
component and the dominant block and tau-harmonic of each weak right singular vector.

    python3 lowD_weak.py OUT.jsonl STATE.npz [STATE.npz ...] [--w=16] [--threads=2]

One Jacobian and LU per state (Solver.newton at the state), then inverse subspace iteration on (J^T J)^-1 with the stored
LU (hpkernel.driver.weak_subspace, f64 directions) and a Rayleigh-Ritz step. s_max is the largest singular value of the
Jacobian (Solver.jacobian, rounded to f64) by 40 power steps on J^T J; ratios s/s_max are quoted.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import _normal_solve, load, make_solver, weak_subspace  # noqa: E402


def labels(M, K):
    names = ["p1(centre)", "c0(centre)"] + [f"{f}@node{i}" for i in range(K) for f in ("p", "m", "a", "c")] + ["mS", "aS"]
    pars = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    return names, pars


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    w = int(opts.get("w", 16))
    threads = int(opts.get("threads", 2))
    out = args[0]
    for f in args[1:]:
        zh, zl, meta = load(f)
        D, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
        K = len(nodes)
        t0 = time.time()
        s = make_solver(D, M, nodes=nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 60), ks=meta.get("ks", 100), threads=threads)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        jh, jl = s.jacobian(zh, zl)
        J = np.asarray(jh) + np.asarray(jl)
        L = J.shape[0]
        del jh, jl
        v = np.random.default_rng(0).standard_normal(L)
        for _ in range(40):
            v = J.T @ (J @ v)
            nv = np.linalg.norm(v)
            v /= nv
        smax = float(np.sqrt(nv))
        del J
        _, _, Fh, Fl, pmin, pmax = s.newton(zh, zl)
        V = weak_subspace(s, w, iters=10)
        W = _normal_solve(s, V)
        H = V.T @ W
        ev, U = np.linalg.eigh((H + H.T) / 2)
        X = V @ U
        X /= np.linalg.norm(X, axis=0)
        sig = 1 / np.sqrt(np.maximum(ev, 1e-300))
        names, pars = labels(M, K)
        rows = []
        for i in np.argsort(sig):
            x = X[:, i]
            blk = [float(np.sum(x[b * M:(b + 1) * M] ** 2)) for b in range(len(names))]
            b = int(np.argmax(blk))
            seg = x[b * M:(b + 1) * M]
            full = np.r_[seg, -seg] if pars[b] == "odd" else np.r_[seg, seg]
            c = np.abs(np.fft.rfft(full))
            k = int(np.argmax(c))
            rows.append(dict(s_rel=float(sig[i] / smax), delta=float(abs(x[-1])), block=names[b], k=k, block_weight=round(blk[b], 3)))
        rec = dict(state=os.path.basename(f), D=D, M=M, L=int(L), s_max=smax, F=float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max()),
                   piv=(pmin, pmax), weakest=rows, seconds=round(time.time() - t0, 1))
        print(f"{os.path.basename(f)}: D {D} M {M} s_max {smax:.2e}  ({rec['seconds']} s)", flush=True)
        for r in rows:
            print(f"   s/s_max {r['s_rel']:.2e}  |Delta comp| {r['delta']:.1e}  {r['block']} k={r['k']} (weight {r['block_weight']})", flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
