"""SVD anatomy of the converged-but-floored solution: which directions carry the residual, and does Delta care?"""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, jacobian_c
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
t0 = time.time(); F, J = jacobian_c(S, z, 3e-7); print(f"J {J.shape} in {time.time()-t0:.0f}s |F|={np.linalg.norm(F):.2e}")
np.save(sys.argv[2].replace(".npy", "_J.npy"), J)
U, s, Vt = np.linalg.svd(J)
c = U.T @ F
print("smax", s[0])
print(" i   sigma/smax   |U.F|    |V_Delta|   GN dDelta contrib")
for i in list(range(len(s) - 25, len(s))):
    print(f"{i:4d} {s[i]/s[0]:.2e}  {abs(c[i]):.2e}  {abs(Vt[i,-1]):.2e}  {-c[i]/s[i]*Vt[i,-1]:+.2e}")
# residual energy by sigma band
for lo, hi in ((1e-3, 1), (1e-6, 1e-3), (1e-9, 1e-6), (0, 1e-9)):
    m = (s / s[0] >= lo) & (s / s[0] < hi); print(f"band [{lo:.0e},{hi:.0e}): n={m.sum()} |F_band|={np.linalg.norm(c[m]):.2e}")
for cut in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
    m = s / s[0] >= cut
    dz = -(Vt[m].T @ (c[m] / s[m]))
    print(f"TSVD cut {cut:.0e}: rank {m.sum()} dDelta {dz[-1]:+.3e} |dz| {np.linalg.norm(dz):.2e} pred residual {np.linalg.norm(c[~m]):.2e}")
