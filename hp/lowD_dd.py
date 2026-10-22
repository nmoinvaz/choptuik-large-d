#!/usr/bin/env python3
"""The double-double solver below D = 5: Delta at one D for a list of tau resolutions M, from an f64 seed or a saved
double-double state, with what is needed to judge convergence (notes/LOW_D.md).

    python3 lowD_dd.py OUT.jsonl --seed F64.npy [--kc_seed 47] --Ms 64,96 [options]
    python3 lowD_dd.py OUT.jsonl --start STATE.npz [--D D_new] [--history a.npz,b.npz,...] --Ms 96 [options]

--seed: a converged f64 solution (solver/data/neB_kc47_rk2_D*.npy: x_K = 0.6 layout, kc --kc_seed), resampled to the
first M (the recipe of hp/compare/dd_D5.py). --start: a double-double state (hpkernel.driver.save format), resampled
if its M differs; with --D the solve is at D_new (a continuation step; with --history, Lagrange extrapolation through
the given states and the start, oldest first, all at the same M and nodes).
Options: --nodes (default 0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6), --kt 36 (Taylor order in x), --kcs 60 (centre
series terms), --ks 100 (horizon series terms), --tol_step 1e-32 (Taylor step tolerance), --hmax 0.05, --tol 1e-29,
--it_max 14, --threads 2, --tag (state file prefix), --solver newton|dn (dn: deflated Newton, driver.deflated_newton),
--w 24 (dn: weak directions frozen in its first phase), --svd N (also the N weakest singular values of the Jacobian at
the solution, by inverse subspace iteration with the stored LU), --lm N (Levenberg-Marquardt iterations if Newton
stalls).
Each M appends one record to OUT.jsonl: Delta (hi, lo), |F|, iterations, seconds, the steps of the Taylor plan,
max |dDelta/dF| per block, the series tails, the linearised Delta correction, and the centre slope's odd harmonics
|F_k| (k = 1, 3, ..., M - 1; F = sqrt(D - 2) dp/dx at x = 0, normalised as amplitudes of cos/sin pairs). The state is saved
as hp/data/lowD_states/<tag>_D<D>_M<M>.npz.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DSS_SOLVER_DIR", os.path.normpath(os.path.join(HERE, "..", "solver")))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
from hpkernel.driver import (deflated_newton, delta_correction, load, lm_solve, make_solver, newton_solve, predict,  # noqa: E402
                             resample_z, save, seed_from_f64, weak_subspace, _normal_solve)

NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def harmonics(f):
    """odd harmonics |c_k| (k = 1, 3, ...) of an antiperiodic field given on the half grid (M points)"""
    M = len(f)
    c = np.fft.rfft(np.r_[f, -f]) / M          # amplitude of cos/sin pair: |2 c_k / N| with N = 2M
    return np.abs(c[1::2])


def tails(s, zh, zl, nodes, M):
    out = []
    for which, x in ((0, nodes[0]), (1, 1 - nodes[-1])):
        h, l = s.series(zh, zl, which)
        c = np.abs((h + l).reshape(-1, 4, M)).max(axis=(1, 2))
        k = np.arange(len(c))
        out.append(float((c * x ** k)[-3:].max()))
    return out


def sensitivity(s, M):
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    nb = (L - 1) // (4 * M)
    return [float(np.abs(yh[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)]


def weakest(s, w, iters=10):
    """w weakest singular values (and the Delta component of their right vectors) by inverse subspace iteration on
    (J^T J)^-1 with the stored LU (f64 directions; Rayleigh-Ritz at the end)"""
    V = weak_subspace(s, w, iters=iters)
    W = _normal_solve(s, V)
    H = V.T @ W
    ev, U = np.linalg.eigh((H + H.T) / 2)
    X = V @ U
    X /= np.linalg.norm(X, axis=0)
    sig = 1 / np.sqrt(np.maximum(ev, 1e-300))
    o = np.argsort(sig)
    return [float(f"{sig[i]:.3e}") for i in o], [float(f"{abs(X[-1, i]):.2e}") for i in o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed")
    ap.add_argument("--kc_seed", type=int, default=47)
    ap.add_argument("--start")
    ap.add_argument("--D", type=float, default=None)
    ap.add_argument("--history", default="")
    ap.add_argument("--Ms", default="64")
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kcs", type=int, default=60)
    ap.add_argument("--ks", type=int, default=100)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-29)
    ap.add_argument("--it_max", type=int, default=14)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="lowD")
    ap.add_argument("--solver", default="newton", choices=["newton", "dn"])
    ap.add_argument("--w", type=int, default=24)
    ap.add_argument("--svd", type=int, default=0)
    ap.add_argument("--lm", type=int, default=0)
    a = ap.parse_args()
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else NL
    Ms = [int(m) for m in a.Ms.split(",")]
    outdir = os.path.join(HERE, "data", "lowD_states")
    os.makedirs(outdir, exist_ok=True)
    log = lambda m: print(m, flush=True)

    if a.seed and "," in a.seed:
        # several f64 solutions at one kc: Lagrange interpolation in D to --D (a seed between computed points)
        pts = [seed_from_f64(f, Ms[0], kc=a.kc_seed, nodes=nodes) for f in a.seed.split(",")]
        if a.D is None:
            raise SystemExit("--seed with several files needs --D")
        D = a.D
        zh = predict([(p[0], p[1]) for p in pts], D)
        zl = np.zeros_like(zh)
        Mo, src = Ms[0], "interp:" + ",".join(os.path.basename(f) for f in a.seed.split(","))
        log(f"seed interpolated from D {[p[0] for p in pts]} to D {D}: Delta {zh[-1]:.12f}")
    elif a.seed:
        D, zh, zl = seed_from_f64(a.seed, Ms[0], kc=a.kc_seed, nodes=nodes)
        Mo, src = Ms[0], os.path.basename(a.seed)
        log(f"seed {a.seed}: D {D}, f64 Delta {zh[-1]:.12f}")
    else:
        zh, zl, meta = load(a.start)
        D, Mo, src = meta["D"], meta["M"], os.path.basename(a.start)
        if list(meta["nodes"]) != nodes:
            raise SystemExit(f"start nodes {meta['nodes']} != {nodes} (re-noding not supported here)")
        if a.D is not None and abs(a.D - D) > 1e-12:
            hist = []
            for f in filter(None, a.history.split(",")):
                hh, hl, hm = load(f)
                if hm["M"] != Mo or list(hm["nodes"]) != nodes:
                    raise SystemExit(f"{f}: M/nodes differ from the start")
                hist.append((hm["D"], hh + hl))
            hist.append((D, zh + zl))
            if len(hist) > 1:
                zh = predict(hist, a.D)
                zl = np.zeros_like(zh)
            log(f"start {a.start} (D {D}, M {Mo}); predictor through {len(hist)} point(s) to D {a.D}: Delta {zh[-1]:.12f}")
            D = a.D
    for M in Ms:
        if M != Mo:
            zh, zl = resample_z(np.asarray(zh) + np.asarray(zl), Mo, M, len(nodes))
            Mo = M
        t0 = time.time()
        s = make_solver(D, M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks, threads=a.threads)
        s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
        fh, fl = s.residual(zh, zl)
        F0 = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
        log(f"D {D} M {M}: L {s.size}, steps {sum(len(g) - 1 for g in s.get_segs())}, start |F| {F0:.1e}")
        if a.solver == "dn":
            zh, zl, hist = deflated_newton(s, zh, zl, tol=a.tol, it_max=a.it_max, w=a.w, verbose=True)
        else:
            zh, zl, hist = newton_solve(s, zh, zl, tol=a.tol, it_max=a.it_max, verbose=True)
        if a.lm and (not hist or hist[-1]["F"] > max(a.tol, 1e-25)):
            log("  Newton stalled; Levenberg-Marquardt")
            zh, zl, h2 = lm_solve(s, zh, zl, tol=a.tol, it_max=a.lm, verbose=True, newton_after=1e-22)
            hist = hist + h2
        zh, zl = np.asarray(zh), np.asarray(zl)
        # the LU stored in s must be at the final z for the sensitivities: newton_solve's last call factors at the
        # returned z when it stopped on tol (its last record has no step); otherwise refresh it (one more Jacobian)
        if hist and "lam" not in hist[-1] and "lm" not in hist[-1] and "failed" not in hist[-1] and a.solver == "newton":
            Fh, Fl = s.residual(zh, zl)
            pmin, pmax = hist[-1]["piv"]
        else:
            _, _, Fh, Fl, pmin, pmax = s.newton(zh, zl)
        F = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        dt = time.time() - t0
        sens = sensitivity(s, M)
        tc, th = tails(s, zh, zl, nodes, M)
        p1 = zh[:M] + zl[:M]
        hk = harmonics(np.sqrt(D - 2) * p1)
        rec = dict(D=D, M=M, Delta=float(zh[-1]), Delta_lo=float(zl[-1]), F=F, its=len(hist), seconds=round(dt, 1),
                   nodes=nodes, kt=a.kt, kcs=a.kcs, ks=a.ks, tol_step=a.tol_step, hmax=a.hmax,
                   steps=sum(len(g) - 1 for g in s.get_segs()), piv=(pmin, pmax), sens_max=max(sens),
                   sens=[float(f"{v:.2e}") for v in sens], tail_c=tc, tail_h=th,
                   dDelta_est=delta_correction(s, Fh, Fl), F_harm=[float(f"{v:.4e}") for v in hk],
                   solver=a.solver, source=src, threads=a.threads)
        if a.svd:
            t1 = time.time()
            sv, dc = weakest(s, a.svd)
            rec.update(weak_sv=sv, weak_delta=dc, svd_seconds=round(time.time() - t1, 1))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save(os.path.join(outdir, f"{a.tag}_D{D:.4f}_M{M}.npz"), zh, zl,
             dict(D=D, M=M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks))
        log(f"D {D} M {M}: Delta {zh[-1]:.16f} ({zl[-1]:+.2e}) |F| {F:.1e} its {len(hist)} {dt:.0f}s "
            f"dDelta/dF max {max(sens):.1e} tails {tc:.0e} {th:.0e} dDelta_est {rec['dDelta_est']:+.1e}")


if __name__ == "__main__":
    main()
