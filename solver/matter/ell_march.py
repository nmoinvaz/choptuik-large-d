"""Adaptive D-continuation of the l-multiplet DSS solution (tangent predictor dz/dD = -J^-1 dF/dn, with a quadratic
term from the previous point; step grows after a success, halves after a failure; lands exactly on target D values).

usage: python ell_march.py tag l kc seed.npy D_seed D_end "targets" [--dinit 0.05] [--dmax 1] [--dmin 0.002]
         [--kappa 1.0] [--Ks 16] [--u0 0.01] [--rk 1] [--kc_seed kc] [--tol 1e-10] [--fmax 0.05]
Every converged point: data/march_{tag}.jsonl (D, Delta, |F|, |dz/dD|, dDelta/dD, constraint, start |F|, step);
target points also saved as data/sol_{tag}_D{D:.4f}.npy (all points are kept in data/march_{tag}.npz at the end).
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, lm, harmonic_weights

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("l", type=int); ap.add_argument("kc", type=int)
ap.add_argument("seed"); ap.add_argument("D_seed", type=float); ap.add_argument("D_end", type=float)
ap.add_argument("targets", default="")
ap.add_argument("--dinit", type=float, default=0.05); ap.add_argument("--dmax", type=float, default=1.0)
ap.add_argument("--dmin", type=float, default=0.002); ap.add_argument("--kappa", type=float, default=1.0)
ap.add_argument("--Ks", type=int, default=16); ap.add_argument("--u0", type=float, default=0.01)
ap.add_argument("--rk", type=int, default=1); ap.add_argument("--kc_seed", type=int, default=None)
ap.add_argument("--tol", type=float, default=1e-10); ap.add_argument("--fmax", type=float, default=0.05)
ap.add_argument("--maxjac", type=int, default=8)
ap.add_argument("--reg", type=float, default=0.0)
ap.add_argument("--facc", type=float, default=None)
a = ap.parse_args()


def tangent(S, z, J):
    Fn = S.dF_dn(z)
    if not a.reg:
        return -np.linalg.lstsq(J, Fn, rcond=None)[0]
    w = harmonic_weights(S)
    mu = a.reg * np.linalg.norm(J, 2)
    return -np.linalg.lstsq(np.vstack([J, mu * np.diag(w)]), np.concatenate([Fn, np.zeros(len(w))]), rcond=None)[0]


up = a.D_end > a.D_seed
sgn = 1.0 if up else -1.0
targets = sorted([float(v) for v in a.targets.split(",") if v], reverse=not up)
targets = [t for t in targets if (t - a.D_seed) * sgn > 1e-12 and (a.D_end - t) * sgn >= -1e-12]
if not targets or abs(targets[-1] - a.D_end) > 1e-12:
    targets.append(a.D_end)


def make(D, z):
    S = DSSEll(kc=a.kc, nodes=NODES_STD, n=D - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, rk=a.rk)
    S.freeze_from(z, a.kappa)
    return S


z = np.load(a.seed)
if a.kc_seed and a.kc_seed != a.kc:
    S0 = DSSEll(kc=a.kc_seed, nodes=NODES_STD, n=a.D_seed - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
    S1 = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D_seed - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
    z = S1.regrid_from(z, S0)
log = os.path.join(HERE, "data", f"march_{a.tag}.jsonl")


def converge(D, zg):
    S = make(D, zg)
    F0 = float(np.linalg.norm(S.residual(zg[None, :])[0]))
    if not np.isfinite(F0) or F0 > a.fmax:
        return None, F0, None
    facc = a.facc if a.facc else 10 * a.tol
    if a.reg:          # one regularised LM; its last Jacobian serves the tangent (no polish, no extra Jacobian)
        zs, nF, _, J = lm(S, zg, tol=a.tol, maxjac=a.maxjac, verbose=False,
                          prep=lambda zz: S.refreeze(zz, a.kappa), reg=a.reg, return_jac=True)
        if not nF < facc:
            return None, F0, nF
        return (S, zs, J), F0, nF
    zs, nF, _ = lm(S, zg, tol=a.tol, maxjac=a.maxjac, verbose=False, prep=lambda zz: S.refreeze(zz, a.kappa))
    if not nF < facc:
        return None, F0, nF
    S = make(D, zs)
    zs, nF, _ = lm(S, zs, tol=a.tol, maxjac=4, verbose=False)
    if not nF < facc:
        return None, F0, nF
    return (S, zs, None), F0, nF


# seed point
t0 = time.time()
res, F0, nF = converge(a.D_seed, z)
assert res is not None, f"seed does not converge: start |F| {F0}, end {nF}"
S, z, J = res
if J is None:
    _, J = S.jacobian(z)
tan = tangent(S, z, J)
hist = [(a.D_seed, z, tan)]
D = a.D_seed
step = a.dinit
print(f"seed D = {D}: Delta {z[-1]:.10f} |dz/dD| {np.linalg.norm(tan):.3e} dDelta/dD {tan[-1]:.5f}", flush=True)
allD, allZ = [D], [z]
while targets:
    nxt = D + sgn * step
    hit = (nxt - targets[0]) * sgn >= -1e-12
    if hit:
        nxt = targets[0]
    D1, z1, t1 = hist[-1]
    zg = z1 + (nxt - D1) * t1
    if len(hist) >= 2:
        D0, z0, _ = hist[-2]
        zg = zg + (z0 - z1 - t1 * (D0 - D1)) / (D0 - D1) ** 2 * (nxt - D1) ** 2
    t0 = time.time()
    res, F0, nF = converge(nxt, zg)
    if res is None:
        step *= 0.5
        print(f"  D = {nxt:.5f}: failed (start |F| {F0:.2e}, end {nF}); step -> {step:.4f}", flush=True)
        if step < a.dmin:
            print("step below dmin; stopping", flush=True)
            break
        continue
    S, z, J = res
    if J is None:
        _, J = S.jacobian(z)
    tan = tangent(S, z, J)
    hist.append((nxt, z, tan)); hist = hist[-3:]
    D = nxt
    allD.append(D); allZ.append(z)
    eD = np.zeros(len(z)); eD[-1] = 1.0
    sensD = float(np.linalg.norm(np.linalg.lstsq(J.T, eD, rcond=None)[0]))     # |dDelta/dF|, selection strength^-1
    rec = dict(D=D, l=a.l, kc=a.kc, kappa=a.kappa, reg=a.reg, u0=a.u0, Ks=a.Ks, Delta=float(z[-1]), F=float(nF), F_start=F0,
               dDelta_dF=sensD,
               tan_norm=float(np.linalg.norm(tan)), dDelta_dD=float(tan[-1]), constraint=S.constraint(z),
               lam_gauge=S.gauge_lambda(z), step=step, seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    open(log, "a").write(json.dumps(rec) + "\n")
    if hit:
        np.save(os.path.join(HERE, "data", f"sol_{a.tag}_D{D:.4f}.npy"), z)
        targets.pop(0)
    # step control from the predictor quality
    if F0 < 0.1 * a.fmax:
        step = min(step * 1.5, a.dmax)
    np.savez_compressed(os.path.join(HERE, "data", f"march_{a.tag}.npz"), D=np.array(allD), Z=np.array(allZ))
print("done", flush=True)
