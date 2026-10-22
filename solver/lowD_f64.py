#!/usr/bin/env python3
"""f64 re-solves at one D for the low-D convergence tests (notes/LOW_D.md): the f64 solver (rust_solver.make +
ne_solver.ne_solve, as scan_ne.py) at a list of Fourier cut-offs kc, RK4 step multipliers and node sets, started from
a double-double state (hp/data/lowD_states/*.npz, converted to the f64 Fourier layout) or from an f64 solution.

    python3 lowD_f64.py OUT.jsonl (--dd STATE.npz | --seed F64.npy) --kcs 47,55,63 [--rks 2] [--nodes x,x,...]
                        [--Ks 60] [--threads 2] [--gamma] [--tag lowD]

The double-double state supplies every field on its half grid (M points); its harmonics k <= kc become the f64
unknowns (the phase convention of the two layouts is the same, hpkernel.driver.seed_from_f64 being the inverse map).
With --nodes different from the state's, the state is first re-noded by the double-double integrator
(hpkernel.driver.renode). For every (kc, rk): |F| of the start, the NE solve (ne_solve: strong Newton, weak by Schur
complement, junk frozen), Delta, the residual split, the weak singular values, and with --gamma lam = root of det M(lam)
(the Rust growth-mode matrix, as scan_ne.gamma_of). The state is saved as solver/data/lowD_f64_<tag>_kc<kc>_rk<rk>_D<D>.npy.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "hp"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import fast  # noqa: E402
fast.install()
from rust_solver import make  # noqa: E402
from spec_long import make_long  # noqa: E402
from ne_solver import ne_solve  # noqa: E402
from dd_solver import dd_solve  # noqa: E402

NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def dd_to_f64(zh, zl, meta, kc, nodes):
    """double-double state (half-grid samples) -> f64 spectral layout at kc on `nodes` (re-noded first if needed)"""
    M = meta["M"]
    if list(meta["nodes"]) != list(nodes):
        from hpkernel.driver import make_solver, renode
        s0 = make_solver(meta["D"], M, nodes=list(meta["nodes"]), kt=meta.get("kt", 36), kc=meta.get("kc", 60),
                         ks=meta.get("ks", 100), threads=2)
        s0.set_segs(s0.plan(zh, zl, 1e-32, 0.05))
        zh, zl = renode(s0, zh, zl, list(meta["nodes"]), list(nodes), M)
    z = np.asarray(zh) + np.asarray(zl)
    K = len(nodes)
    S = make_long(kc, meta["D"] - 2, nodes=list(nodes), Ks=60)
    pars = S.parities()
    out = []
    for j, p in enumerate(pars):
        f = z[j * M:(j + 1) * M]
        full = np.r_[f, -f] if p == "odd" else np.r_[f, f]
        c = np.fft.rfft(full) / (2 * M)
        if p == "odd":
            ks = S.kodd
            out.append(np.r_[c[ks].real, c[ks].imag])
        else:
            ks = S.keven[1:]
            out.append(np.r_[c[0].real, c[ks].real, c[ks].imag])
    zf = np.concatenate(out + [z[-1:]])
    if len(zf) != S.layout()[2]:
        raise ValueError("layout mismatch")
    return zf


def gamma_of(R, z, g0):
    cache = {}

    def sd(l):
        if l not in cache:
            cache[l] = np.linalg.slogdet(R.mode_matrix(z, l))
        return cache[l]
    for w in (0.003, 0.01, 0.03):
        grid = np.linspace(g0 - w, g0 + w, 5)
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(4) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(grid[i] - g0))
            ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            return brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-11, rtol=1e-14)
    return None


def selftest(path=os.path.join(HERE, "data", "neB_kc47_rk2_D4.0000.npy"), mutate=False):
    """round trip f64 -> double-double layout (hpkernel.driver.seed_from_f64) -> f64 (dd_to_f64) must reproduce the
    state to rounding; --mutate scales one harmonic of the converted state and must make it fail"""
    from hpkernel.driver import seed_from_f64
    z = np.load(path)
    D, zh, zl = seed_from_f64(path, 64, kc=47, nodes=NL)
    if mutate:
        zh = zh.copy()
        zh[64 * 3 + 5] *= 1.001          # one sample of node 0's p field
    back = dd_to_f64(zh, zl, dict(D=D, M=64, nodes=NL), 47, NL)
    err = float(np.abs(back - z).max())
    ok = err < 1e-12
    print(f"round trip f64 -> dd (M 64) -> f64 at D {D}: max difference {err:.1e}  {'ok' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def main():
    if "--selftest" in sys.argv:
        selftest(mutate="--mutate" in sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dd")
    ap.add_argument("--seed")
    ap.add_argument("--kcs", default="47")
    ap.add_argument("--rks", default="2")
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--Ks", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gamma", action="store_true")
    ap.add_argument("--lam0", type=float, default=None, help="growth-rate guess (default: from the hp/lowD_gamma table)")
    ap.add_argument("--tag", default="lowD")
    a = ap.parse_args()
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else NL
    log = lambda m: print(m, flush=True)
    if a.dd:
        from hpkernel.driver import load
        zh, zl, meta = load(a.dd)
        D, src = meta["D"], os.path.basename(a.dd)
    else:
        D = float(re.findall(r"D([0-9.]+)\.npy", a.seed)[0])
        z_seed = np.load(a.seed)
        m = re.search(r"kc(\d+)", os.path.basename(a.seed))
        kc_seed = int(m.group(1)) if m else 23
        src = os.path.basename(a.seed)
    lam0 = a.lam0
    if a.gamma and lam0 is None:
        from lowD_gamma import lam_guess
        lam0 = lam_guess(D)
    for kc in [int(v) for v in a.kcs.split(",")]:
        for rk in [int(v) for v in a.rks.split(",")]:
            t0 = time.time()
            if a.dd:
                z0 = dd_to_f64(zh, zl, meta, kc, nodes)
            else:
                Sf, St = make_long(kc_seed, D - 2, nodes=nodes, Ks=60), make_long(kc, D - 2, nodes=nodes, Ks=60)
                z0 = St.regrid(z_seed, Sf) if kc != kc_seed else z_seed.copy()
            S, R = make(D, z0[-1], nodes=nodes, threads=a.threads, kc=kc, Ks=a.Ks, rk=rk)
            F0 = float(np.abs(R(z0[None, :])[0]).max())
            log(f"D {D} kc {kc} rk {rk} nodes {nodes}: start |F| {F0:.1e} (seed Delta {z0[-1]:.12f})")
            if np.linalg.norm(R(z0[None, :])[0]) > 1e-8:
                z0, _, _, _ = dd_solve(R, z0, maxjac=12, use_dd=False, stall=3)
            z, info = ne_solve(R, z0)
            h_ = info["hist"][-1]
            lam = gamma_of(R, z, lam0) if (a.gamma and lam0) else None
            rec = dict(D=D, kc=kc, rk=rk, nodes=nodes, Ks=a.Ks, Delta=float(z[-1]), F=info["F"], strong=h_[1], weak=h_[2],
                       junk=h_[3], n_weak=info["n_weak"], weak_s=[float(f"{x:.2e}") for x in info["weak_s"]],
                       n_outer=len(info["hist"]), lam=lam, gamma=(1 / lam if lam else None), F_start=F0, source=src,
                       seconds=round(time.time() - t0))
            log(json.dumps(rec))
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            nn = "" if nodes == NL else "_n" + "-".join(f"{x:g}" for x in nodes)
            np.save(os.path.join(HERE, "data", f"lowD_f64_{a.tag}_kc{kc}_rk{rk}{nn}_D{D:.4f}.npy"), z)


if __name__ == "__main__":
    main()
