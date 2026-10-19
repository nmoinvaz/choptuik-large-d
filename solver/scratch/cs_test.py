import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, jacobian_c, jacobian_cs
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
F, Jc = jacobian_cs(S, z)
JR = np.load(sys.argv[2].replace(".npy", "_JR.npy"))
_, J3 = jacobian_c(S, z, 3e-7); _, J9 = jacobian_c(S, z, 1e-8)
for nm, M in (("FD 3e-7", J3), ("FD 1e-8", J9), ("Richardson", JR)):
    print(f"{nm:10s} vs complex-step: |dJ|={np.linalg.norm(M - Jc, 2):.2e}  Dcol {np.linalg.norm(M[:,-1]-Jc[:,-1]):.2e}")
