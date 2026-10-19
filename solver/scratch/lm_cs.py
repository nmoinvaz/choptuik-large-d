import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, lm_solve, jacobian_cs
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2]); mj = int(sys.argv[3]); mu = float(sys.argv[4])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
z2, nF, t = lm_solve(S, z, mu_rel=mu, maxjac=mj, verbose=True, jac=jacobian_cs, tol=1e-14)
print(f"final |F|={nF:.2e} Delta={z2[-1]:.12f} (start {z[-1]:.12f}, diff {z2[-1]-z[-1]:+.2e}) {t:.0f}s")
np.save(sys.argv[2].replace(".npy", "_cs.npy"), z2)
