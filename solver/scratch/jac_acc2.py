import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, jacobian_c
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
sizes, K, L = S.layout(); blk = sum(sizes[2:6])
J = {e: jacobian_c(S, z, e)[1] for e in (6e-7, 3e-7, 1.5e-7, 3e-8, 1e-8)}
R1 = (4 * J[3e-7] - J[6e-7]) / 3; R2 = (4 * J[1.5e-7] - J[3e-7]) / 3
ref = R2
for name, M in (("3e-7", J[3e-7]), ("3e-8", J[3e-8]), ("1e-8", J[1e-8]), ("Rich(6e-7,3e-7)", R1)):
    d = M - ref
    rows = [np.linalg.norm(d[b*blk:(b+1)*blk], 2) for b in range(K + 1)]
    print(f"{name:16s} |dJ|={np.linalg.norm(d,2):.2e}  per row-block:", " ".join(f"{r:.0e}" for r in rows), f" Dcol {np.linalg.norm(d[:,-1]):.1e}")
np.save(sys.argv[2].replace(".npy", "_JR.npy"), R2)
