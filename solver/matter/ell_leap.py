#!/usr/bin/env python3
"""D-continuation of the l-multiplet's DSS solution by secant steps (#1025).

ell_march.py's tangent predictor dz/dD = -J^-1 dF/dn follows the Jacobian's near-null directions past D ~ 10 (#810:
four of them at k_c 23, singular values 2e-12 to 1e-11), and at k_c 31 it left the branch by D 10.3 (psi-constraint
2.8e-4 at D 10.125, 2.2e-2 at 10.31; raw I1025-mac/march_kc31_D10_15_stopped.log). Here the predictor is the secant
through the two previous converged states, with no Jacobian, and each state is solved as ell_refine.py solves it
(lm with refrozen RK steps, then a polish at fixed steps). It stops when the psi-constraint exceeds --cmax or the
solve does not reach --fmax.

usage: python ell_leap.py tag l kc solA.npy DA solB.npy DB D_end h [--kappa 0.7] [--u0 0.04] [--Ks 30] [--tol 1e-12]
           [--cmax 1e-3] [--fmax 1e-9] [--every 0.5]
Every converged point: data/{tag}.jsonl (D, Delta, |F|, |F| at the secant start, psi-constraint, |dDelta/dF|);
states data/sol_{tag}_D{D:.4f}.npy at multiples of --every (and the last two, for a restart).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, lm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("l", type=int); ap.add_argument("kc", type=int)
ap.add_argument("solA"); ap.add_argument("DA", type=float); ap.add_argument("solB"); ap.add_argument("DB", type=float)
ap.add_argument("D_end", type=float); ap.add_argument("h", type=float)
ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--u0", type=float, default=0.04)
ap.add_argument("--Ks", type=int, default=30); ap.add_argument("--rk", type=int, default=1)
ap.add_argument("--tol", type=float, default=1e-12); ap.add_argument("--cmax", type=float, default=1e-3)
ap.add_argument("--fmax", type=float, default=1e-9); ap.add_argument("--every", type=float, default=0.5)
a = ap.parse_args()
out = os.path.join(HERE, "data", f"{a.tag}.jsonl")
zA, zB = np.load(a.solA), np.load(a.solB)
DA, DB = a.DA, a.DB
assert zA.shape == zB.shape, "the two states must share k_c and layout"
D = DB
while D < a.D_end - 1e-9:
    Dn = round(min(D + a.h, a.D_end), 10)
    zp = zB + (zB - zA) * (Dn - DB) / (DB - DA)
    t0 = time.time()
    mk = lambda z: DSSEll(kc=a.kc, nodes=NODES_STD, n=Dn - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, rk=a.rk)
    S = mk(zp)
    S.freeze_from(zp, a.kappa)
    F0 = float(np.linalg.norm(S.residual(zp[None, :])[0]))
    z, nF, _ = lm(S, zp, tol=a.tol, maxjac=12, verbose=False, prep=lambda zz: S.refreeze(zz, a.kappa))
    S = mk(z); steps = S.freeze_from(z, a.kappa)
    z, nF, _ = lm(S, z, tol=a.tol, maxjac=4, verbose=False)
    _, J = S.jacobian(z)
    eD = np.zeros(len(z)); eD[-1] = 1.0
    sensD = float(np.linalg.norm(np.linalg.lstsq(J.T, eD, rcond=None)[0]))
    con = S.constraint(z)
    rec = dict(D=Dn, l=a.l, kc=a.kc, kappa=a.kappa, u0=a.u0, Ks=a.Ks, Delta=float(z[-1]), Delta_pred=float(zp[-1]),
               F=float(nF), F_start=F0, steps=int(sum(steps.values())), constraint=con, dDelta_dF=sensD,
               lam_gauge=S.gauge_lambda(z), h=a.h, seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    if not (nF < a.fmax) or con > a.cmax:
        print(f"stop at D {Dn}: |F| {nF:.2e} (bar {a.fmax:.0e}), psi-constraint {con:.2e} (bar {a.cmax:.0e})", flush=True)
        sys.exit(1)
    open(out, "a").write(json.dumps(rec) + "\n")
    if abs(Dn / a.every - round(Dn / a.every)) < 1e-9 or Dn >= a.D_end - 1e-9:
        np.save(os.path.join(HERE, "data", f"sol_{a.tag}_D{Dn:.4f}.npy"), z)
    np.save(os.path.join(HERE, "data", f"sol_{a.tag}_last.npy"), z)
    np.save(os.path.join(HERE, "data", f"sol_{a.tag}_prev.npy"), zB)
    zA, zB, DA, DB, D = zB, z, DB, Dn, Dn
print(f"reached D {D}", flush=True)
