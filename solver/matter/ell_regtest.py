"""Sensitivity of Delta to the smoothness prior (lm reg) and to the RK stiffness target, at one D.
usage: python ell_regtest.py sol.npy kc_src D l kc reg kappa out_tag [u0=0.01] [Ks=16]
Re-solves from sol (regridded to kc) and appends Delta, |F|, constraint, the q spectrum tail to data/ell_regtest.jsonl;
saves the solution as data/sol_{out_tag}_D{D}.npy."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, lm

f, kc_src, D, l, kc, reg, kappa, tag = (sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]),
                                        int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8])
z0 = np.load(f)
U0 = float(sys.argv[9]) if len(sys.argv) > 9 else 0.01
KS = int(sys.argv[10]) if len(sys.argv) > 10 else 16
mk = lambda kc_: DSSEll(kc=kc_, nodes=NODES_STD, n=D - 2, ell=l, Delta_ref=z0[-1], Ks=KS, u0=U0)
S = mk(kc)
z = S.regrid_from(z0, mk(kc_src)) if kc != kc_src else z0.copy()
t0 = time.time()
S.freeze_from(z, kappa)
z, nF, _ = lm(S, z, tol=1e-11, maxjac=15, verbose=True, prep=lambda zz: S.freeze_from(zz, kappa), reg=reg)
S.freeze_from(z, kappa)
z, nF, _ = lm(S, z, tol=1e-11, maxjac=6, verbose=True, reg=reg)
fl = S.fields(z)
A = np.abs(np.fft.rfft(fl["d"])) / S.N * 2
rec = dict(D=D, l=l, kc=kc, reg=reg, kappa=kappa, u0=U0, Ks=KS, steps=int(sum(S.steps_map.values())), src=os.path.basename(f), Delta=float(z[-1]), F=float(nF),
           constraint=S.constraint(z), q_harm={int(k): float(A[k]) for k in range(1, kc + 1, 2)},
           seconds=round(time.time() - t0))
print(json.dumps(rec), flush=True)
open(os.path.join(HERE, "data", "ell_regtest.jsonl"), "a").write(json.dumps(rec) + "\n")
np.save(os.path.join(HERE, "data", f"sol_{tag}_D{D:.4f}.npy"), z)
