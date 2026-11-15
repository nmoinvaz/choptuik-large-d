"""Does the psi-constraint follow the solution's weakly determined directions?  (issue #810)

usage: python ell_constraint_dirs.py sol.npy D l kc [--kappa 0.7] [--u0 0.04] [--Ks 30] [--nweak 4] [--dC 1e-5]
                                     [--resolve]

At a converged state z, the Jacobian J = dF/dz is decomposed, J = U S V^T.  Along each of the nweak weakest right
singular vectors v (and, as the control, the strongest), the state is moved to z + t v with t chosen, from a finite
difference of the constraint, to change the psi-constraint by dC.  Each move reports the new |F|, the change of Delta
(t v[-1]) and the constraint's actual change.  A weak direction that changes the constraint by dC while raising |F| by
no more than 1e-12 and moving Delta by no more than 1e-8 explains how two converged solves can share Delta and differ
in the constraint.  The strongest direction must fail that test (it is the control).

--resolve: from each moved state, Newton (dss_ell.lm, as ell_refine.py solves) re-converges to tol 1e-13.  The solution
set is not isolated in the constraint if the re-converged state has |F| <= 1e-12, a constraint at least dC/2 from the
start's, and Delta within 1e-8; if it returns to the start's constraint, the root is isolated along that direction.
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, lm

ap = argparse.ArgumentParser()
ap.add_argument("sol"); ap.add_argument("D", type=float); ap.add_argument("l", type=int); ap.add_argument("kc", type=int)
ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--u0", type=float, default=0.04)
ap.add_argument("--Ks", type=int, default=30); ap.add_argument("--rk", type=int, default=1)
ap.add_argument("--nweak", type=int, default=4); ap.add_argument("--dC", type=float, default=1e-5)
ap.add_argument("--resolve", action="store_true")
a = ap.parse_args()

z = np.load(a.sol)
S = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, rk=a.rk)
S.freeze_from(z, a.kappa)
res = lambda x: float(np.linalg.norm(S.residual(x[None, :])[0]))
F0, C0 = res(z), S.constraint(z)
_, J = S.jacobian(z)
U, sv, Vt = np.linalg.svd(J, full_matrices=False)
print(json.dumps(dict(kind="state", sol=os.path.basename(a.sol), D=a.D, kc=a.kc, n=len(z), F=F0, constraint=C0,
                      Delta=float(z[-1]), sigma_max=float(sv[0]), sigma_min=float(sv[-1]),
                      sigma_weak=[float(s) for s in sv[-a.nweak:]])), flush=True)
cases = [("weak", len(sv) - 1 - j) for j in range(a.nweak)] + [("control_strongest", 0)]
for name, i in cases:
    v = Vt[i]
    h = 1e-6 / max(np.abs(v).max(), 1e-300)
    slope = (S.constraint(z + h * v) - S.constraint(z - h * v)) / (2 * h)
    if slope == 0:
        print(json.dumps(dict(kind="dir", case=name, index=int(i), sigma=float(sv[i]), slope=0.0)), flush=True)
        continue
    t = a.dC / abs(slope)
    zt = z + t * v
    rec = dict(kind="dir", case=name, index=int(i), sigma=float(sv[i]), slope=float(slope), t=float(t),
               dF=res(zt) - F0, F=res(zt), dDelta=float(t * v[-1]), dConstraint=S.constraint(zt) - C0)
    rec["explains"] = bool(rec["dF"] <= 1e-12 and abs(rec["dDelta"]) <= 1e-8 and abs(rec["dConstraint"]) >= 0.5 * a.dC)
    if a.resolve:
        zr, nF, _ = lm(S, zt.copy(), tol=1e-13, maxjac=12, verbose=False, prep=lambda zz: S.refreeze(zz, a.kappa))
        S.freeze_from(z, a.kappa)          # back to the start's frozen steps for the comparison
        Sr = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D - 2, ell=a.l, Delta_ref=zr[-1], Ks=a.Ks, u0=a.u0, rk=a.rk)
        Sr.freeze_from(zr, a.kappa)
        rec.update(resolved_F=float(np.linalg.norm(Sr.residual(zr[None, :])[0])),
                   resolved_dDelta=float(zr[-1] - z[-1]), resolved_dConstraint=Sr.constraint(zr) - C0,
                   resolved_distance=float(np.linalg.norm(zr - z)))
        rec["not_isolated"] = bool(rec["resolved_F"] <= 1e-12 and abs(rec["resolved_dDelta"]) <= 1e-8
                                   and abs(rec["resolved_dConstraint"]) >= 0.5 * a.dC)
    print(json.dumps(rec), flush=True)
