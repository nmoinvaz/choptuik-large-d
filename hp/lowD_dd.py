#!/usr/bin/env python3
"""The double-double solver below D = 5: Delta at one D for a list of tau resolutions M, from an f64 seed or a saved
double-double state, with what is needed to judge convergence (notes/LOW_D.md).

    python3 lowD_dd.py OUT.jsonl --seed F64.npy [--kc_seed 47] --Ms 64,96 [options]
    python3 lowD_dd.py OUT.jsonl --start STATE.npz [--D D_new] [--history a.npz,b.npz,...] --Ms 96 [options]

--seed: a converged f64 solution (solver/data/neB_kc47_rk2_D*.npy: x_K = 0.6 layout, kc --kc_seed), resampled to the
first M (the recipe of hp/compare/dd_D5.py); several comma-separated f64 solutions with --D: Lagrange interpolation
of them to D. --start: a double-double state (hpkernel.driver.save format), resampled if its M differs and re-noded
(driver.renode) if --nodes differs; with --D the solve is at D_new (a continuation step; with --history, Lagrange
extrapolation through the given states and the start, oldest first, all at the same M and nodes).
A solve that does not reach |F| < --accept (1e-24) is recorded with converged false and saved as *_fail.npz, and the
remaining M are skipped. Plain Newton from an f64 seed resampled straight to M = 64 fails at D = 4 (|F| 4e-2: the
seed lacks harmonics 49-63); go through M = 48 first, with --solver dn.
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
                             renode, resample_z, save, seed_from_f64, weak_subspace, _normal_solve)

NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def harmonics(f):
    """odd harmonics |c_k| (k = 1, 3, ...) of an antiperiodic field given on the half grid (M points)"""
    M = len(f)
    c = np.fft.rfft(np.r_[f, -f]) / M          # amplitude of cos/sin pair: |2 c_k / N| with N = 2M
    return np.abs(c[1::2])


def fill_tail(z, M, K, kc, klo=None):
    """Seed improvement: the f64 seed has no harmonics above kc; extend every field's harmonics k = kc+2, ..., < M by
    the geometric trend (modulus and phase linear in k) fitted on its harmonics klo..kc-2 of the same parity (the top
    one, kc, is depressed by the truncation). A field whose fitted modulus grows is left alone."""
    klo = klo if klo is not None else kc - 16
    pars = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    z = np.array(z, float)
    for j, p in enumerate(pars):
        f = z[j * M:(j + 1) * M]
        full = np.r_[f, -f] if p == "odd" else np.r_[f, f]
        c = np.fft.rfft(full)
        par = 1 if p == "odd" else 0
        kf = np.array([k for k in range(klo, kc - 1) if k % 2 == par])
        if len(kf) < 4 or np.any(np.abs(c[kf]) == 0):
            continue
        sl, ic = np.polyfit(kf, np.log(np.abs(c[kf])), 1)
        ph = np.unwrap(np.angle(c[kf]))
        ps, pc = np.polyfit(kf, ph, 1)
        if sl >= 0:
            continue
        kn = np.array([k for k in range(kc + 1, M) if k % 2 == par])
        c[kn] = np.exp(ic + sl * kn) * np.exp(1j * (pc + ps * kn))
        full2 = np.fft.irfft(c, n=2 * M)
        z[j * M:(j + 1) * M] = full2[:M]
    return z


def reintegrate(s, zh, zl, nodes, M, h=0.002):
    """Seed improvement: every node state recomputed from the seed's centre data, by the centre series at nodes[0]
    and the double-double Taylor integrator outward from node to node (as driver.renode), so that all centre-side
    blocks are consistent at this M and the mismatch sits in the horizon block."""
    zh = np.array(zh, float)
    zl = np.array(zl, float)
    ch, cl = s.series(zh, zl, 0)
    ch = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    v = np.zeros((4, M))
    for k in range(ch.shape[0] - 1, -1, -1):
        v = v * nodes[0] + ch[k]
    sh, sl = v.ravel(), np.zeros(4 * M)
    zh[2 * M:2 * M + 4 * M] = sh
    zl[2 * M:2 * M + 4 * M] = 0.0
    for i in range(1, len(nodes)):
        nst = max(2, int(np.ceil((nodes[i] - nodes[i - 1]) / h)))
        xs = list(np.linspace(nodes[i - 1], nodes[i], nst + 1))
        xs[-1] = nodes[i]
        sh, sl = s.integrate(zh, zl, sh, sl, xs)
        sh, sl = np.asarray(sh), np.asarray(sl)
        zh[2 * M + 4 * M * i:2 * M + 4 * M * (i + 1)] = sh
        zl[2 * M + 4 * M * i:2 * M + 4 * M * (i + 1)] = sl
    return zh, zl


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
    ap.add_argument("--seed_nodes", default=None, help="nodes of the f64 seed's layout (default the x_K = 0.6 set); the "
                    "seed is re-noded to --nodes when they differ")
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kcs", type=int, default=60)
    ap.add_argument("--ks", type=int, default=100)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-29)
    ap.add_argument("--accept", type=float, default=1e-24, help="a solve counts as converged below this |F|")
    ap.add_argument("--it_max", type=int, default=14)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="lowD")
    ap.add_argument("--solver", default="newton", choices=["newton", "dn"])
    ap.add_argument("--w", type=int, default=24)
    ap.add_argument("--dn_its", type=int, default=2, help="dn: iterations of the deflated phase before plain Newton")
    ap.add_argument("--resume", action="store_true", help="start each (D, M) from its checkpoint if there is one")
    ap.add_argument("--fill_tail", action="store_true", help="--seed: extend the seed's harmonics above kc_seed by "
                    "each field's geometric trend (fill_tail)")
    ap.add_argument("--start_residual_only", action="store_true", help="print the start residual per block and exit")
    ap.add_argument("--reintegrate", action="store_true", help="recompute every node state of the start from its centre "
                    "data (reintegrate)")
    ap.add_argument("--svd", type=int, default=0)
    ap.add_argument("--lm", type=int, default=0)
    a = ap.parse_args()
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else NL
    Ms = [int(m) for m in a.Ms.split(",")]
    outdir = os.path.join(HERE, "data", "lowD_states")
    os.makedirs(outdir, exist_ok=True)
    log = lambda m: print(m, flush=True)

    seed_nodes = [float(v) for v in a.seed_nodes.split(",")] if a.seed_nodes else NL
    if a.seed and "," in a.seed:
        # several f64 solutions at one kc: Lagrange interpolation in D to --D (a seed between computed points)
        pts = [seed_from_f64(f, Ms[0], kc=a.kc_seed, nodes=seed_nodes) for f in a.seed.split(",")]
        if a.D is None:
            raise SystemExit("--seed with several files needs --D")
        D = a.D
        zh = predict([(p[0], p[1]) for p in pts], D)
        zl = np.zeros_like(zh)
        Mo, src = Ms[0], "interp:" + ",".join(os.path.basename(f) for f in a.seed.split(","))
        log(f"seed interpolated from D {[p[0] for p in pts]} to D {D}: Delta {zh[-1]:.12f}")
    elif a.seed:
        D, zh, zl = seed_from_f64(a.seed, Ms[0], kc=a.kc_seed, nodes=seed_nodes)
        Mo, src = Ms[0], os.path.basename(a.seed)
        log(f"seed {a.seed}: D {D}, f64 Delta {zh[-1]:.12f}")
    if a.seed and a.fill_tail and Ms[0] > a.kc_seed + 1:
        zh = fill_tail(zh, Ms[0], len(seed_nodes), a.kc_seed)
        src += " (tail filled)"
        log(f"harmonics above {a.kc_seed} filled by the geometric trend of each field")
    if a.seed and seed_nodes != nodes:
        s0 = make_solver(D, Mo, nodes=seed_nodes, kt=a.kt, kc=a.kcs, ks=a.ks, threads=a.threads)
        s0.set_segs(s0.plan(zh, zl, a.tol_step, a.hmax))
        zh, zl = renode(s0, zh, zl, seed_nodes, nodes, Mo)
        log(f"seed re-noded {seed_nodes} -> {nodes}")
        src += " (re-noded)"
    if a.reintegrate:
        s0 = make_solver(D, Mo, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks, threads=a.threads)
        zh, zl = reintegrate(s0, zh, zl, nodes, Mo)
        src += " (node states re-integrated from the centre)"
        log("node states re-integrated from the centre data")
    if a.start_residual_only:
        for M in Ms:
            zz = resample_z(np.asarray(zh) + np.asarray(zl), Mo, M, len(nodes))[0] if M != Mo else np.asarray(zh) + np.asarray(zl)
            s = make_solver(D, M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks, threads=a.threads)
            s.set_segs(s.plan(zz, np.zeros_like(zz), a.tol_step, a.hmax))
            fh, fl = s.residual(zz, np.zeros_like(zz))
            F = np.abs(np.asarray(fh) + np.asarray(fl))
            nb = (len(F) - 1) // (4 * M)
            log(f"D {D} M {M}: start |F| {F.max():.2e}; per block " + " ".join(f"{F[b * 4 * M:(b + 1) * 4 * M].max():.1e}" for b in range(nb)))
        return
    else:
        zh, zl, meta = load(a.start)
        D, Mo, src = meta["D"], meta["M"], os.path.basename(a.start)
        if list(meta["nodes"]) != nodes:
            # re-node: every new node's state from the old solution (driver.renode: Taylor integration from the
            # nearest old node below, or the centre series)
            s0 = make_solver(D, Mo, nodes=list(meta["nodes"]), kt=a.kt, kc=a.kcs, ks=a.ks, threads=a.threads)
            s0.set_segs(s0.plan(zh, zl, a.tol_step, a.hmax))
            zh, zl = renode(s0, zh, zl, list(meta["nodes"]), nodes, Mo)
            log(f"re-noded {meta['nodes']} -> {nodes}")
            src += " (re-noded)"
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
        done = os.path.join(outdir, f"{a.tag}_D{D:.4f}_M{M}.npz")
        if a.resume and os.path.exists(done):
            zh, zl, _ = load(done)
            log(f"D {D} M {M}: already converged ({done}); skipped")
            continue
        ckpt = os.path.join(outdir, f"{a.tag}_D{D:.4f}_M{M}_ckpt.npz")
        meta_c = dict(D=D, M=M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks)
        if a.resume and os.path.exists(ckpt):
            zh, zl, mc = load(ckpt)
            log(f"resuming from checkpoint {ckpt} (iteration {mc.get('it')})")
        s = make_solver(D, M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks, threads=a.threads)
        segs = None
        if a.resume and os.path.exists(ckpt) and mc.get("segs"):
            segs = mc["segs"]             # keep the step plan of the interrupted run (the residual depends on it)
        s.set_segs(segs or s.plan(zh, zl, a.tol_step, a.hmax))
        meta_c["segs"] = s.get_segs()
        fh, fl = s.residual(zh, zl)
        F0 = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
        log(f"D {D} M {M}: L {s.size}, steps {sum(len(g) - 1 for g in s.get_segs())}, start |F| {F0:.1e}")
        hist = []
        if a.solver == "dn" and not (a.resume and os.path.exists(ckpt)):
            # deflated phase only (the weakest w directions removed from the step), then plain Newton below
            zh, zl, hist = deflated_newton(s, zh, zl, tol=a.tol, it_max=a.dn_its, w=a.w, verbose=True)
            save(ckpt, np.asarray(zh), np.asarray(zl), dict(meta_c, it=len(hist)))
        # plain Newton with backtracking, one iteration per call so that every accepted step is checkpointed
        for it in range(a.it_max):
            if hist and (hist[-1].get("accepted") or ("lam" not in hist[-1] and "piv" in hist[-1] and hist[-1]["F"] < a.tol)
                         or "failed" in hist[-1]):
                break
            zh, zl, h1 = newton_solve(s, zh, zl, tol=a.tol, it_max=1, verbose=True)
            hist += h1
            if h1 and "lam" in h1[-1]:
                save(ckpt, np.asarray(zh), np.asarray(zl), dict(meta_c, it=len(hist)))
        _rh, _rl = s.residual(zh, zl)
        if a.lm and not float(np.abs(np.asarray(_rh) + np.asarray(_rl)).max()) < a.accept:
            log("  Newton stalled; Levenberg-Marquardt")
            zh, zl, h2 = lm_solve(s, zh, zl, tol=a.tol, it_max=a.lm, verbose=True, newton_after=1e-22)
            hist = hist + h2
        zh, zl = np.asarray(zh), np.asarray(zl)
        Fh, Fl = s.residual(zh, zl)
        F = float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max())
        if not F < a.accept:
            # not converged: record it, keep the state under a _fail name (a start for a retry), no sensitivities
            dt = time.time() - t0
            rec = dict(D=D, M=M, Delta=float(zh[-1]), Delta_lo=float(zl[-1]), F=F, its=len(hist), seconds=round(dt, 1),
                       nodes=nodes, kt=a.kt, kcs=a.kcs, ks=a.ks, tol_step=a.tol_step, hmax=a.hmax, solver=a.solver,
                       source=src, threads=a.threads, converged=False)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            save(os.path.join(outdir, f"{a.tag}_D{D:.4f}_M{M}_fail.npz"), zh, zl,
                 dict(D=D, M=M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks))
            log(f"D {D} M {M}: NOT converged, |F| {F:.1e} after {len(hist)} iterations ({dt:.0f}s); stopping")
            break
        # the LU stored in s must be at the final z for the sensitivities: the solvers' last s.newton call is at the
        # returned z when they stopped on tol (last record without a step taken); otherwise refresh it (one Jacobian)
        last = hist[-1] if hist else {}
        if "piv" in last and ("lam" not in last or last.get("accepted") or "failed" in last):
            pmin, pmax = last["piv"]
        else:
            _, _, Fh, Fl, pmin, pmax = s.newton(zh, zl)
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
                   solver=a.solver, source=src, threads=a.threads, converged=True)
        if a.svd:
            t1 = time.time()
            sv, dc = weakest(s, a.svd)
            rec.update(weak_sv=sv, weak_delta=dc, svd_seconds=round(time.time() - t1, 1))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save(done, zh, zl, dict(D=D, M=M, nodes=nodes, kt=a.kt, kc=a.kcs, ks=a.ks))
        if os.path.exists(ckpt):
            os.remove(ckpt)
        log(f"D {D} M {M}: Delta {zh[-1]:.16f} ({zl[-1]:+.2e}) |F| {F:.1e} its {len(hist)} {dt:.0f}s "
            f"dDelta/dF max {max(sens):.1e} tails {tc:.0e} {th:.0e} dDelta_est {rec['dDelta_est']:+.1e}")


if __name__ == "__main__":
    main()
