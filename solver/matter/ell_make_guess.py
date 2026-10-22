"""Build a DSSEll initial guess from stored near-critical slices (ell_guess.py run pass).
usage: python ell_make_guess.py n l slices.npz out.npy kc kfit triples ka
  triples i0:i1: consecutive crossing triples averaged for T* and Delta
  ka: the period used for the fit starts at crossing ka (tau_c = -ln(T* - T_ka)) and lasts Delta."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ell_guess import accumulation, transform, bvp_samples, make_guess
from dss_ell import DSSEll, NODES_STD

n, l, npz, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
kc, kfit = int(sys.argv[5]), int(sys.argv[6])
i0, i1 = [int(v) for v in sys.argv[7].split(":")]
ka = int(sys.argv[8])
d = np.load(npz)
Tk = d["Tk"]
acc = accumulation(Tk)
Tstar = float(np.median(acc[i0:i1, 0]))
Delta = float(np.median(-2 * np.log(acc[i0:i1, 1])))
tau_a = -np.log(Tstar - Tk[ka])
print(f"T* = {Tstar:.7f}, Delta = {Delta:.5f} (triples {i0}:{i1}); window tau_c [{tau_a:.4f}, {tau_a + Delta:.4f}]")
tr = transform(npz, Tstar, Delta, tau_a)
print(f"{len(tr['T'])} slices; SSH x_c in [{tr['xs'].min():.4f}, {tr['xs'].max():.4f}] (instantaneous root "
      f"{tr['x_inst']:.4f}); r_s in [{tr['rs'].min():.4f}, {tr['rs'].max():.4f}]")
samp = bvp_samples(tr, NODES_STD, l)
S = DSSEll(kc=kc, nodes=NODES_STD, n=n, ell=l, Delta_ref=Delta, Ks=16, u0=0.01)
z, rms = make_guess(S, samp, tr["tau"], Delta, kfit)
print("relative rms of the fits (worst over nodes):", {k: round(float(v), 4) for k, v in rms.items()})
S.freeze_from(z)
F = S.residual(z[None, :])[0]
print(f"|F| of the guess: {np.linalg.norm(F):.3e} (|z| {np.linalg.norm(z):.2e})")
q = samp["q"]
print(f"q range [{q.min():.4f}, {q.max():.4f}], c0 range [{samp['c0'].min():.4f}, {samp['c0'].max():.4f}], "
      f"a(1) range [{samp['aS'].min():.4f}, {samp['aS'].max():.4f}]")
np.save(out, z)
