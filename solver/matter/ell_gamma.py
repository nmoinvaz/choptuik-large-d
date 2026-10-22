"""Growth rates of a converged l-multiplet DSS solution: real roots of det M(lambda) on [lo, hi] (sign changes of
slogdet on a grid, refined by brentq), and the gauge rate mean(1 + c_1).
usage: python ell_gamma.py sol.npy D l kc lo hi ngrid [--Ks 16] [--u0 0.01] [--rk 1] [--kappa 0.5] [--N N]
Appends to data/ell_gamma.jsonl."""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, growth_rate

ap = argparse.ArgumentParser()
ap.add_argument("sol"); ap.add_argument("D", type=float); ap.add_argument("l", type=int); ap.add_argument("kc", type=int)
ap.add_argument("lo", type=float); ap.add_argument("hi", type=float); ap.add_argument("ngrid", type=int)
ap.add_argument("--Ks", type=int, default=16); ap.add_argument("--u0", type=float, default=0.01)
ap.add_argument("--rk", type=int, default=1); ap.add_argument("--kappa", type=float, default=0.5)
ap.add_argument("--N", type=int, default=None)
a = ap.parse_args()
z = np.load(a.sol)
S = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, rk=a.rk, N=a.N)
S.freeze_from(z, a.kappa)
F = np.linalg.norm(S.residual(z[None, :])[0])
t0 = time.time()
roots, nev = growth_rate(S, z, a.lo, a.hi, ngrid=a.ngrid, verbose=True)
rec = dict(sol=os.path.basename(a.sol), D=a.D, l=a.l, kc=a.kc, rk=a.rk, kappa=a.kappa, Delta=float(z[-1]), F=float(F),
           lo=a.lo, hi=a.hi, ngrid=a.ngrid, roots=roots, lam_gauge=S.gauge_lambda(z), n_eval=nev,
           seconds=round(time.time() - t0))
print(json.dumps(rec), flush=True)
open(os.path.join(HERE, "data", "ell_gamma.jsonl"), "a").write(json.dumps(rec) + "\n")
