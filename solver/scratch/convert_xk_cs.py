"""Convert a long-series solution matched at x_K=0.6 into one matched at x_K=0.5 (drop the 0.6 node), re-solve."""
import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long, from_standard
from spec_c import lm_solve, freeze_steps, jacobian_c, jacobian_cs
D = float(sys.argv[1]); f = sys.argv[2]; KS = int(sys.argv[3]); out = sys.argv[4]
N6 = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]; N5 = N6[:-1]
S6 = make_long(23, D - 2, nodes=N6, Ks=60); z6 = np.load(f)
S = freeze_steps(make_long(23, D - 2, nodes=N5, Ks=KS), z6[-1])
z = from_standard(S6, z6, S)
t0 = time.time(); F0 = np.linalg.norm(S.residual(z[None, :])[0])
z, nF, _ = lm_solve(S, z, maxjac=int(sys.argv[5]) if len(sys.argv) > 5 else 25, jac=jacobian_cs, verbose=True, tol=1e-14)
F, J = jacobian_cs(S, z, 3e-7)
y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-13)[0]; sD = np.linalg.norm(J[:, -1] - J[:, :-1] @ y)
print(f"D={D} x_K=0.5 Ks={KS}: start |F|={F0:.1e} -> {nF:.1e}; Delta={z[-1]:.10f} (x_K=0.6: {z6[-1]:.10f}); s_Delta={sD:.2e}; dDelta~{nF/sD:.1e} ({time.time()-t0:.0f}s)", flush=True)
np.save(out, z)
