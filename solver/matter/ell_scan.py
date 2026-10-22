"""Solve the l-multiplet DSS problem along a list of D (continuation), with the growth rate at every point.

usage: python ell_scan.py out_tag l kc seed.npy D_seed "D1,D2,..." [--Ks 16] [--u0 0.01] [--rk 1] [--N N]
         [--lam lo:hi] [--kc_seed kc] [--no-gamma]
The seed is a solution vector (or guess) of DSSEll(l, kc_seed, NODES_STD) at D_seed. Each new point starts from a
polynomial extrapolation (in D) of the previous converged solutions. Solutions go to data/sol_{tag}_D{D}.npy, records
to data/{tag}.jsonl: Delta, |F|, the growth-rate roots (det M(lambda) sign changes) and the gauge rate.
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, lm, growth_rate

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("l", type=int); ap.add_argument("kc", type=int)
ap.add_argument("seed"); ap.add_argument("D_seed", type=float); ap.add_argument("Ds")
ap.add_argument("--Ks", type=int, default=16); ap.add_argument("--u0", type=float, default=0.01)
ap.add_argument("--rk", type=int, default=1); ap.add_argument("--N", type=int, default=None)
ap.add_argument("--lam", default="0.3:20"); ap.add_argument("--ngrid", type=int, default=40)
ap.add_argument("--kc_seed", type=int, default=None); ap.add_argument("--no-gamma", action="store_true")
ap.add_argument("--tol", type=float, default=1e-11); ap.add_argument("--maxjac", type=int, default=25)
ap.add_argument("--order", type=int, default=2)
ap.add_argument("--kappa", type=float, default=0.5)
ap.add_argument("--lam0", type=float, default=8.4)
a = ap.parse_args()


def make(D, Dref, kc=a.kc):
    return DSSEll(kc=kc, nodes=NODES_STD, n=D - 2, ell=a.l, Delta_ref=Dref, Ks=a.Ks, u0=a.u0, rk=a.rk, N=a.N)


z = np.load(a.seed)
if a.kc_seed and a.kc_seed != a.kc:
    S0 = make(a.D_seed, z[-1], a.kc_seed); S1 = make(a.D_seed, z[-1])
    z = S1.regrid_from(z, S0)
hist = [(a.D_seed, z, None)]
first = True
lo, hi = [float(v) for v in a.lam.split(":")]
lam_hist = []                                         # (D, lambda) of the physical mode


def lam_root(S, zs, guess):
    """Expanding bracket around the predicted lambda: sign change of slogdet M, then brentq."""
    from scipy.optimize import brentq
    cache = {}

    def sd(l):
        if l not in cache:
            cache[l] = np.linalg.slogdet(S.mode_matrix(zs, l))
        return cache[l]
    for w in (0.02, 0.06, 0.15, 0.35):
        grid = guess * (1 + w * np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(4) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - guess))
            ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            r = brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-11, rtol=1e-14)
            return r, len(cache), len(ch)
    return None, len(cache), 0


targets = [float(v) for v in a.Ds.split(",")]
depth = 0
while targets:
    D = targets[0]
    t0 = time.time()
    if first and abs(D - a.D_seed) < 1e-12:
        zg = z
    else:
        D1, z1, t1 = hist[-1]
        if t1 is not None:                             # tangent predictor, with a quadratic term from the point before
            zg = z1 + (D - D1) * t1
            if len(hist) >= 2:
                D0, z0, _ = hist[-2]
                cq = (z0 - z1 - t1 * (D0 - D1)) / (D0 - D1) ** 2
                zg = zg + cq * (D - D1) ** 2
        else:
            Dh = np.array([h[0] for h in hist[-(a.order + 1):]]); Zh = np.array([h[1] for h in hist[-(a.order + 1):]])
            zg = np.zeros_like(Zh[0])
            for i in range(len(Dh)):                   # Lagrange extrapolation
                w = np.prod([(D - Dh[j]) / (Dh[i] - Dh[j]) for j in range(len(Dh)) if j != i])
                zg = zg + w * Zh[i]
    first = False
    S = make(D, zg[-1]); S.freeze_from(zg, a.kappa)
    F0 = np.linalg.norm(S.residual(zg[None, :])[0])
    print(f"D = {D}: start |F| {F0:.2e}, Delta guess {zg[-1]:.8f}", flush=True)
    zs, nF, sec = lm(S, zg, tol=a.tol, maxjac=a.maxjac, prep=lambda zz: S.refreeze(zz, a.kappa))
    if not (nF < 1e-9):
        if depth < 4 and len(hist) >= 1:
            Dm = 0.5 * (hist[-1][0] + D)
            print(f"D = {D}: not converged (|F| {nF:.2e}); inserting D = {Dm}", flush=True)
            targets.insert(0, Dm); depth += 1
            continue
        print(f"D = {D}: not converged (|F| {nF:.2e}); stopping", flush=True)
        break
    depth = 0
    targets.pop(0)
    # re-freeze the RK steps at the converged solution and polish
    S = make(D, zs[-1]); steps = S.freeze_from(zs, a.kappa)
    zs, nF, sec2 = lm(S, zs, tol=a.tol, maxjac=8, verbose=False)
    rec = dict(D=D, l=a.l, kc=a.kc, N=S.N, Ks=a.Ks, u0=a.u0, rk=a.rk, kappa=a.kappa, steps=sum(steps.values()),
               Delta=float(zs[-1]), F=float(nF), lam_gauge=S.gauge_lambda(zs), constraint=S.constraint(zs))
    np.save(os.path.join(HERE, "data", f"sol_{a.tag}_D{D:.4f}.npy"), zs)
    if not a.no_gamma:
        if len(lam_hist) >= 2:
            (D1, l1), (D2, l2) = lam_hist[-2], lam_hist[-1]
            guess = l2 + (l2 - l1) * (D - D2) / (D2 - D1)
        elif lam_hist:
            guess = lam_hist[-1][1]
        else:
            guess = a.lam0
        lam, nev, nch = lam_root(S, zs, guess)
        rec.update(lam_guess=guess, n_eval=nev, n_signchanges=nch)
        if lam is not None:
            rec.update(lam=lam, gamma=1 / lam)
            lam_hist.append((D, lam))
    rec["seconds"] = round(time.time() - t0)
    print(json.dumps(rec), flush=True)
    open(os.path.join(HERE, "data", f"{a.tag}.jsonl"), "a").write(json.dumps(rec) + "\n")
    _, J = S.jacobian(zs)
    tan = -np.linalg.lstsq(J, S.dF_dn(zs), rcond=None)[0]
    hist = [h for h in hist if abs(h[0] - D) > 1e-12] + [(D, zs, tan)]
