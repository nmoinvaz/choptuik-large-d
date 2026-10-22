"""l = 0 validation of dss_ell.DSSEll against the stored 4-field solutions of the main solver.

The seed (x_K = 0.6 long-series layout, kc as stored) is converted to the 5-field layout with psi = 0, re-solved with
DSSEll at l = 0 (psi decouples there), and Delta compared with the stored value and the reference value.
usage: python ell_validate.py D seedfile kc [Delta_ref_published]
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from dss_ell import DSSEll, NODES_L, lm, growth_rate
from spec_long import make_long

D, seed, kc = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ref = float(sys.argv[4]) if len(sys.argv) > 4 else None
z4 = np.load(seed)
S4 = make_long(kc, D - 2, nodes=NODES_L, Ks=60)
S = DSSEll(kc=kc, nodes=NODES_L, n=D - 2, ell=0, Delta_ref=z4[-1], Ks=60, long=True)
z = S.from_dss4(z4, S4)
F = S.residual(z[None, :])[0]
sizes, K, L = S.layout()
print(f"D = {D}, kc = {kc}: {L} unknowns; |F| at the converted seed {np.linalg.norm(F):.3e}", flush=True)
t0 = time.time()
z, nF, sec = lm(S, z, tol=1e-13, maxjac=8)
lam_g = S.gauge_lambda(z)
roots, nev = growth_rate(S, z, 1.2, 3.2, ngrid=11)
rec = dict(D=D, kc=kc, ell=0, Delta=float(z[-1]), Delta_seed=float(z4[-1]), F=float(nF), Delta_ref=ref,
           roots=roots, lam_gauge=lam_g, seconds=round(time.time() - t0))
print(json.dumps(rec), flush=True)
open(os.path.join(HERE, "data", "ell0_validation.jsonl"), "a").write(json.dumps(rec) + "\n")
