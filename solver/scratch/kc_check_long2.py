"""Truncation check at high D: regrid a converged kc23 long-series solution to kc31 and re-solve."""
import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import lm_solve, freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]; kc = int(sys.argv[3])
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
S23 = make_long(23, D - 2, nodes=NODES_L, Ks=60); z23 = np.load(f)
S = freeze_steps(make_long(kc, D - 2, nodes=NODES_L, Ks=60), z23[-1])
z = S.regrid(z23, S23)
t0 = time.time(); z, nF, _ = lm_solve(S, z, maxjac=20)
print(f"D={D} kc={kc}: |F|={nF:.1e} Delta={z[-1]:.10f} vs kc23 {z23[-1]:.10f}  diff {z[-1]-z23[-1]:+.2e} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"long_D{D:.4f}_kc{kc}.npy", z)
