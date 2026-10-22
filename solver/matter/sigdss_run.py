"""Drivers for the sigma-model DSS solver (sigdss_solver.py). Single process; run under nice.

  cont TAG SEED --par eps|D --start S --end E --targets t1,t2,.. --D D --eps EPS --kc KC [--kc_seed K]
       pseudo-natural continuation in eps = 1/eta (at fixed D) or in D (at fixed eps): tangent predictor
       dz/dpar = -J^-1 dF/dpar (complex step) plus a quadratic term from the previous point, adaptive steps landing
       on the targets. Every converged point -> data/sigdss_{TAG}.jsonl; targets -> data/sigdss_sol_{TAG}_*.npy.
  lam SOL --D D --eps EPS --kc KC --guess G [--tag T]
       growth rate of the unstable mode: root of det M(lam) near G (sign change of slogdet, brentq);
       appends to data/sigdss_lam.jsonl (or sigdss_lam_{T}.jsonl).
  roots SOL --D D --eps EPS --kc KC --lo a --hi b --ngrid m
       all sign changes of det M(lam) on a grid (real roots).
  refine SOL --D D --eps EPS --kc_seed K0 --kc KC [--guess G] [--tag T]
       regrid to kc, re-solve, optionally the growth rate; appends to data/sigdss_refine_{T}.jsonl.
Common options: --kappa (RK stiffness target, 0.7), --Ks (16), --u0 (0.01), --tol, --reg.
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sigdss_solver import SigDSS, NODES_STD, lm, lam_root, lam_secant, lam_model, real_roots   # noqa: E402

DATA = os.path.join(HERE, "data")

ap = argparse.ArgumentParser()
ap.add_argument("cmd"); ap.add_argument("a1"); ap.add_argument("a2", nargs="?")
ap.add_argument("--par", default="eps"); ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
ap.add_argument("--targets", default="")
ap.add_argument("--D", type=float, default=4.0); ap.add_argument("--eps", type=float, default=0.0)
ap.add_argument("--kc", type=int, default=15); ap.add_argument("--kc_seed", type=int, default=None)
ap.add_argument("--N", type=int, default=None)
ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--Ks", type=int, default=16)
ap.add_argument("--u0", type=float, default=0.01); ap.add_argument("--tol", type=float, default=1e-11)
ap.add_argument("--facc", type=float, default=None); ap.add_argument("--fmax", type=float, default=0.05)
ap.add_argument("--reg", type=float, default=0.0); ap.add_argument("--maxjac", type=int, default=8)
ap.add_argument("--dinit", type=float, default=0.1); ap.add_argument("--dmax", type=float, default=1.0)
ap.add_argument("--dmin", type=float, default=1e-3)
ap.add_argument("--guess", type=float, default=None); ap.add_argument("--tag", default="")
ap.add_argument("--lo", type=float); ap.add_argument("--hi", type=float); ap.add_argument("--ngrid", type=int, default=9)
ap.add_argument("--lams", default="", help="cont: growth rate at each target, starting guess")
a = ap.parse_args()


def mk(D, eps, z, kc=None):
    S = SigDSS(kc=kc or a.kc, nodes=NODES_STD, n=D - 2, eps=eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, N=a.N)
    S.freeze_from(z, a.kappa)
    return S


def eta_of(eps):
    return None if eps == 0 else 1.0 / eps


def describe(S, z, D, eps):
    um, pm = S.u_amplitude(z)
    return dict(D=D, eps=eps, eta=eta_of(eps), kc=S.kc, N=S.N, Ks=S.Ks, u0=S.u0, kappa=a.kappa, reg=a.reg,
                Delta=float(z[-1]), constraint=S.constraint(z), lam_gauge=S.gauge_lambda(z), umax=um, psimax=pm,
                steps=int(sum(S.steps_map.values())))


def converge(D, eps, zg):
    S = mk(D, eps, zg)
    F0 = float(np.linalg.norm(S.residual(zg[None, :])[0]))
    if not np.isfinite(F0) or F0 > a.fmax:
        return None, F0, None
    facc = a.facc if a.facc else 10 * a.tol
    S._lastJ = None
    zs, nF, _ = lm(S, zg, tol=a.tol, maxjac=a.maxjac, verbose=False, prep=lambda zz: S.refreeze(zz, a.kappa), reg=a.reg)
    if not nF < facc:
        return None, F0, nF
    return (S, zs), F0, nF


def fname(tag, D, eps):
    e = "inf" if eps == 0 else f"{1 / eps:.4f}"
    return os.path.join(DATA, f"sigdss_sol_{tag}_D{D:.4f}_eta{e}.npy")


def cmd_cont():
    tag, seed = a.a1, a.a2
    par = a.par
    up = a.end > a.start
    sgn = 1.0 if up else -1.0
    targets = sorted([float(v) for v in a.targets.split(",") if v], reverse=not up)
    targets = [t for t in targets if (t - a.start) * sgn > 1e-12 and (a.end - t) * sgn >= -1e-12]
    if not targets or abs(targets[-1] - a.end) > 1e-12:
        targets.append(a.end)
    z = np.load(seed)
    D, eps = (a.D, a.start) if par == "eps" else (a.start, a.eps)
    if a.kc_seed and a.kc_seed != a.kc:
        S0 = SigDSS(kc=a.kc_seed, nodes=NODES_STD, n=D - 2, eps=eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
        S1 = SigDSS(kc=a.kc, nodes=NODES_STD, n=D - 2, eps=eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, N=a.N)
        z = S1.regrid_from(z, S0)
    log = os.path.join(DATA, f"sigdss_{tag}.jsonl")
    lam_hist = {}
    if a.lams:          # "lam0" or "v1:lam1,v2:lam2,..." (known or estimated growth rates to start the extrapolation)
        if ":" in a.lams:
            lam_hist = {float(t.split(":")[0]): float(t.split(":")[1]) for t in a.lams.split(",")}
        else:
            lam_hist = {a.start: float(a.lams)}

    def point(v):
        return (a.D, v) if par == "eps" else (v, a.eps)

    def tangent(S, z):
        J = getattr(S, "_lastJ", None)            # Jacobian of the last Newton iterate (close enough for a predictor)
        if J is None:
            _, J = S.jacobian(z)
        return -np.linalg.lstsq(J, S.dF_dpar(z, "n" if par == "D" else "eps"), rcond=None)[0]

    t0 = time.time()
    res, F0, nF = converge(*point(a.start), z)
    assert res is not None, f"seed does not converge: start |F| {F0}, end {nF}"
    S, z = res
    tan = tangent(S, z)
    hist = [(a.start, z, tan)]
    print(f"seed {par} = {a.start}: Delta {z[-1]:.10f} |F| {nF:.2e} dDelta/d{par} {tan[-1]:.5f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    v = a.start
    step = a.dinit
    allv, allZ = [v], [z]
    while targets:
        nxt = v + sgn * step
        hit = (nxt - targets[0]) * sgn >= -1e-12
        if hit:
            nxt = targets[0]
        v1, z1, t1 = hist[-1]
        zg = z1 + (nxt - v1) * t1
        if len(hist) >= 2:
            v0, z0, _ = hist[-2]
            zg = zg + (z0 - z1 - t1 * (v0 - v1)) / (v0 - v1) ** 2 * (nxt - v1) ** 2
        t0 = time.time()
        res, F0, nF = converge(*point(nxt), zg)
        if res is None:
            step *= 0.5
            print(f"  {par} = {nxt:.6f}: failed (start |F| {F0:.2e}, end {nF}); step -> {step:.5f}", flush=True)
            if step < a.dmin:
                print("step below dmin; stopping", flush=True)
                break
            continue
        S, z = res
        tan = tangent(S, z)
        hist.append((nxt, z, tan)); hist = hist[-3:]
        v = nxt
        allv.append(v); allZ.append(z)
        Dp, ep = point(v)
        rec = describe(S, z, Dp, ep)
        rec.update(F=float(nF), F_start=F0, dDelta_dpar=float(tan[-1]), par=par, step=step, target=bool(hit),
                   seconds=round(time.time() - t0))
        if hit:
            np.save(fname(tag, Dp, ep), z)
            targets.pop(0)
            if a.lams:
                # gamma = 1/lam (smoother than lam) through the <= 3 history points nearest v (an entry at v itself,
                # e.g. an estimate from the l = 1 data, is returned as is); computed values replace estimates
                near = sorted(lam_hist, key=lambda u: abs(u - v))[:3]
                hv = np.array(near); hg = np.array([1 / lam_hist[u] for u in near])
                guess = 1 / np.polyval(np.polyfit(hv, hg, len(hv) - 1), v) if len(hv) > 1 else lam_hist[near[0]]
                tl = time.time()
                lam, nev, nch = lam_model(S, z, guess)
                rec.update(lam_guess=guess, lam=lam, gamma=(1 / lam if lam else None), n_eval=nev, last_step=nch,
                           lam_seconds=round(time.time() - tl))
                if lam:
                    lam_hist[v] = lam
        print(json.dumps(rec), flush=True)
        open(log, "a").write(json.dumps(rec) + "\n")
        if F0 < 0.1 * a.fmax:
            step = min(step * 1.5, a.dmax)
        np.savez_compressed(os.path.join(DATA, f"sigdss_{tag}.npz"), v=np.array(allv), Z=np.array(allZ), par=par)
    print("done", flush=True)


def load_solve(sol, kc_seed=None):
    z = np.load(sol)
    if kc_seed and kc_seed != a.kc:
        S0 = SigDSS(kc=kc_seed, nodes=NODES_STD, n=a.D - 2, eps=a.eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
        S1 = SigDSS(kc=a.kc, nodes=NODES_STD, n=a.D - 2, eps=a.eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, N=a.N)
        z = S1.regrid_from(z, S0)
    return z


def cmd_lam():
    z = load_solve(a.a1)
    S = mk(a.D, a.eps, z)
    F = float(np.linalg.norm(S.residual(z[None, :])[0]))
    t0 = time.time()
    lam, nev, nch = lam_model(S, z, a.guess)
    rec = describe(S, z, a.D, a.eps)
    rec.update(sol=os.path.basename(a.a1), F=F, lam_guess=a.guess, lam=lam, gamma=(1 / lam if lam else None),
               n_eval=nev, last_step=nch, seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    out = os.path.join(DATA, f"sigdss_lam{('_' + a.tag) if a.tag else ''}.jsonl")
    open(out, "a").write(json.dumps(rec) + "\n")


def cmd_roots():
    z = load_solve(a.a1)
    S = mk(a.D, a.eps, z)
    t0 = time.time()
    roots, nev = real_roots(S, z, a.lo, a.hi, a.ngrid)
    rec = describe(S, z, a.D, a.eps)
    rec.update(sol=os.path.basename(a.a1), lo=a.lo, hi=a.hi, ngrid=a.ngrid, roots=roots, n_eval=nev,
               seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    open(os.path.join(DATA, "sigdss_roots.jsonl"), "a").write(json.dumps(rec) + "\n")


def cmd_refine():
    z0 = np.load(a.a1)
    z = load_solve(a.a1, a.kc_seed)
    t0 = time.time()
    S = mk(a.D, a.eps, z)
    F0 = float(np.linalg.norm(S.residual(z[None, :])[0]))
    z, nF, _ = lm(S, z, tol=a.tol, maxjac=12, verbose=True, prep=lambda zz: S.refreeze(zz, a.kappa), reg=a.reg)
    S = mk(a.D, a.eps, z)
    z, nF, _ = lm(S, z, tol=a.tol, maxjac=4, verbose=True, reg=a.reg)
    rec = describe(S, z, a.D, a.eps)
    rec.update(sol=os.path.basename(a.a1), kc_seed=a.kc_seed, Delta_seed=float(z0[-1]), F_start=F0, F=float(nF),
               solve_seconds=round(time.time() - t0))
    tag = a.tag or f"kc{a.kc}"
    np.save(fname(f"ref_{tag}", a.D, a.eps), z)
    if a.guess and nF < 1e-8:
        tl = time.time()
        lam, nev, nch = lam_model(S, z, a.guess)
        rec.update(lam_guess=a.guess, lam=lam, gamma=(1 / lam if lam else None), n_eval=nev, last_step=nch,
                   lam_seconds=round(time.time() - tl))
    print(json.dumps(rec), flush=True)
    open(os.path.join(DATA, f"sigdss_refine_{tag}.jsonl"), "a").write(json.dumps(rec) + "\n")


{"cont": cmd_cont, "lam": cmd_lam, "roots": cmd_roots, "refine": cmd_refine}[a.cmd]()
