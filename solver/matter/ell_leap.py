#!/usr/bin/env python3
"""D-continuation of the l-multiplet's DSS solution by secant steps (#1025).

ell_march.py's tangent predictor dz/dD = -J^-1 dF/dn follows the Jacobian's near-null directions past D ~ 10 (#810:
four of them at k_c 23, singular values 2e-12 to 1e-11), and at k_c 31 it left the branch by D 10.3 (psi-constraint
2.8e-4 at D 10.125, 2.2e-2 at 10.31; raw I1025-mac/march_kc31_D10_15_stopped.log). Here the predictor is the secant
through the two previous converged states, with no Jacobian, and each state is solved as ell_refine.py solves it
(lm with refrozen RK steps, then a polish at fixed steps). It stops when the psi-constraint exceeds --cmax or the
solve does not reach --fmax.

usage: python ell_leap.py tag l kc solA.npy DA solB.npy DB D_end h [--kappa 0.7] [--u0 0.04] [--Ks 30] [--tol 1e-12]
           [--cmax 1e-3] [--fmax 1e-9] [--every 0.5] [--fold] [--hmin 0.015625]
Every converged point: data/{tag}.jsonl (D, Delta, |F|, |F| at the secant start, psi-constraint, |dDelta/dF|);
states data/sol_{tag}_D{D:.4f}.npy at multiples of --every (and the last two, for a restart).

--fold (a fold of the branch in D, #1025): each point also records the four smallest singular values of J (and the
largest), |cos| of the secant tangent t_z = z - z_prev with their right singular vectors, the smallest singular value
of the bordered matrix [[J, dF/dn], [t_z^T, t_D]] (regular at a regular fold, where sigma_1(J) -> 0 like
sqrt(D* - D)), and |J^+ dF/dn| (the tangent's size, which diverges at the fold). Steps adapt: a failed solve (|F| above
--fmax or the constraint above --cmax) halves h and retries from the same two states, and a fall of sigma_1(J) by
more than half over one step halves h for the next. That singular value is the fold candidate's, the weak mode
with the largest |cos| (sv_fold, cos_fold), because J also has near-null junk directions (#810). Below --hmin the
run stops. Also recorded: the number of singular values below 1e-12 sigma_max (n_null) and the four smallest above
it with the tangent's |cos| on their vectors (sv_reg_small, cos_t_reg), where a fold mode would show. Every state is saved
(sol_{tag}_D{D:.6f}.npy).
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
ap.add_argument("--fold", action="store_true"); ap.add_argument("--hmin", type=float, default=0.015625)
a = ap.parse_args()
out = os.path.join(HERE, "data", f"{a.tag}.jsonl")
zA, zB = np.load(a.solA), np.load(a.solB)
DA, DB = a.DA, a.DB
assert zA.shape == zB.shape, "the two states must share k_c and layout"
D = DB
h = a.h
sv1_prev = None
while D < a.D_end - 1e-9:
    Dn = round(min(D + h, a.D_end), 10)
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
               lam_gauge=S.gauge_lambda(z), h=h, seconds=round(time.time() - t0))
    ok = nF < a.fmax and con <= a.cmax
    if a.fold and ok:
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)
        tz, tD = z - zB, Dn - DB
        nt = np.linalg.norm(tz)
        Fn = S.dF_dn(z)
        B = np.block([[J, Fn[:, None]], [tz[None, :] / nt, np.array([[tD / nt]])]])
        svB = np.linalg.svd(B, compute_uv=False)
        rec.update(sv_J_small=sv[-4:][::-1].tolist(), sv_J_max=float(sv[0]),
                   cos_t_v=[float(abs(tz @ Vt[-1 - i]) / nt) for i in range(4)],
                   sv_B_min=float(svB[-1]), sv_B_max=float(svB[0]),
                   tan_norm=float(np.linalg.norm(np.linalg.lstsq(J, Fn, rcond=None)[0])))
        i_f = int(np.argmax(rec["cos_t_v"]))          # the fold candidate: the weak mode the tangent lies along
        rec.update(sv_fold=rec["sv_J_small"][i_f], cos_fold=rec["cos_t_v"][i_f], i_fold=i_f)
        # the smallest singular values above the numerical null space (sigma > 1e-12 sigma_max): at k_c 47 J has
        # several at the rounding floor (1e-16 against 7) that the tangent does not lie along (#1025)
        reg = np.nonzero(sv > 1e-12 * sv[0])[0]
        last = reg[-4:][::-1]
        rec.update(n_null=int(len(sv) - len(reg)), sv_reg_small=sv[last].tolist(),
                   cos_t_reg=[float(abs(tz @ Vt[i]) / nt) for i in last])
    print(json.dumps(rec), flush=True)
    if not ok:
        if a.fold and h / 2 >= a.hmin:
            h /= 2
            print(f"failed at D {Dn} (|F| {nF:.2e}, psi-constraint {con:.2e}); h -> {h}", flush=True)
            continue
        print(f"stop at D {Dn}: |F| {nF:.2e} (bar {a.fmax:.0e}), psi-constraint {con:.2e} (bar {a.cmax:.0e})", flush=True)
        sys.exit(1)
    open(out, "a").write(json.dumps(rec) + "\n")
    if a.fold:
        sv1 = rec["sv_fold"]
        if sv1_prev is not None and sv1 < 0.5 * sv1_prev and h / 2 >= a.hmin:
            h /= 2
            print(f"the fold candidate's singular value fell from {sv1_prev:.3e} to {sv1:.3e}; h -> {h}", flush=True)
        sv1_prev = sv1
        np.save(os.path.join(HERE, "data", f"sol_{a.tag}_D{Dn:.6f}.npy"), z)
    if abs(Dn / a.every - round(Dn / a.every)) < 1e-9 or Dn >= a.D_end - 1e-9:
        np.save(os.path.join(HERE, "data", f"sol_{a.tag}_D{Dn:.4f}.npy"), z)
    np.save(os.path.join(HERE, "data", f"sol_{a.tag}_last.npy"), z)
    np.save(os.path.join(HERE, "data", f"sol_{a.tag}_prev.npy"), zB)
    zA, zB, DA, DB, D = zB, z, DB, Dn, Dn
print(f"reached D {D}", flush=True)
