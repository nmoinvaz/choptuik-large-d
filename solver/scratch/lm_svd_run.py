import sys, numpy as np, warnings, json
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, lm_svd
D = float(sys.argv[1]); z = np.load(sys.argv[2]); xk = float(sys.argv[3]); Ks = int(sys.argv[4]); mj = int(sys.argv[5])
N6 = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]; nodes = [x for x in N6 if x <= xk + 1e-12]
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=Ks), z[-1])
z2, nF, t = lm_svd(S, z, maxjac=mj, verbose=True, tol=1e-14)
print(f"final |F|={nF:.2e} Delta={z2[-1]:.12f} (start {z[-1]:.12f}, diff {z2[-1]-z[-1]:+.2e}) {t:.0f}s")
np.save(sys.argv[6], z2)
