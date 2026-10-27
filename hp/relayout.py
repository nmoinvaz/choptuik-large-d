#!/usr/bin/env python3
"""A saved state moved to another layout, for second-layout checks (notes/GHOST_ADMIXTURE.md): re-noded
(driver.renode) and resampled in tau (driver.resample_z) as scan_hp does for a start file in another layout, p1's
harmonics above kfreeze set to zero, then converged by deflated Newton (driver.deflated_newton, w 24) in the new
layout, and saved (driver.save) with that layout in its meta.

    python3 relayout.py IN.npz OUT.npz [--nodes 0.1,0.2,...] [--M N] [--kc K] [--ks K] [--kfreeze 13|none]
        [--tol 1e-24] [--it_max 30] [--threads N]

kt, kc and ks default to IN's (its meta); --kfreeze defaults to 13 from D 35.35 on, none below (as the runs). Prints
|F| before and after and Delta in both layouts; a second layout that converges should reproduce Delta to the runs'
accuracy. Exit 1 if |F| stays above --tol. OUT must be in a raw/<task> worktree (raw_out.py; --allow_unraw writes
anyway).
"""
import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import deflated_newton, load, make_solver, plow_odd, renode, resample_z, save  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--nodes", default="")
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--kc", type=int, default=0)
    ap.add_argument("--ks", type=int, default=0)
    ap.add_argument("--kfreeze", default="")
    ap.add_argument("--tol", type=float, default=1e-24)
    ap.add_argument("--it_max", type=int, default=30)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out, allow=a.allow_unraw)
    zh, zl, meta = load(a.inp)
    D, M0, nodes0 = meta["D"], meta["M"], list(meta["nodes"])
    kt, kc, ks = meta.get("kt", 36), a.kc or meta.get("kc", 40), a.ks or meta.get("ks", 120)
    kf = (None if a.kfreeze == "none" else int(a.kfreeze)) if a.kfreeze else (13 if D >= 35.35 else None)
    M = a.M or M0
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else nodes0
    Delta0 = float(zh[-1]) + float(zl[-1])
    if M != M0:
        zh, zl = resample_z(zh, M0, M, len(nodes0))
    if nodes != nodes0:
        s0 = make_solver(D, M, nodes=nodes0, kt=kt, kc=meta.get("kc", 40), ks=meta.get("ks", 120), threads=a.threads)
        zh, zl = renode(s0, zh, zl, nodes0, nodes, M)
    if kf:
        lo, lo2 = plow_odd(M, kf)
        zh, zl = zh.copy(), zl.copy()
        zh[:M] = (lo + lo2) @ (zh[:M] + zl[:M])
        zl[:M] = 0.0
    s = make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=a.threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
    F0 = float(np.abs(np.asarray(s.residual(zh, zl)[0])).max())
    print(f"D {D}: {len(nodes0)} nodes, M {M0} -> {len(nodes)} nodes {nodes}, M {M}, kc {kc}, ks {ks}, kfreeze {kf}; "
          f"start |F| {F0:.2e}", flush=True)
    zh, zl, hist = deflated_newton(s, zh, zl, tol=a.tol, it_max=a.it_max, w=24, verbose=True)
    F = hist[-1]["F"] if hist else F0
    Delta = float(zh[-1]) + float(zl[-1])
    save(a.out, zh, zl, dict(D=D, M=M, nodes=nodes, kt=kt, kc=kc, ks=ks, F=F, kfreeze=kf, from_state=os.path.basename(a.inp)))
    print(f"|F| {F:.2e} after {len(hist)} steps; Delta {Delta:.16f} here, {Delta0:.16f} in the first layout, "
          f"difference {Delta - Delta0:+.2e}; saved {a.out}", flush=True)
    sys.exit(0 if F <= a.tol else 1)


if __name__ == "__main__":
    main()
