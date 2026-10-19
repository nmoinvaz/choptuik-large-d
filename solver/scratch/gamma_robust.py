"""Growth rate from the dip of the smallest singular value of the harmonic-regularised mode matrix [M(lam); mu W].
High-harmonic junk null directions are lifted by mu W; the smooth physical growth mode is barely affected."""
import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from newton_tsvd import harmonic_weights
D = float(sys.argv[1]); f = sys.argv[2]; xk = float(sys.argv[3]); KS = int(sys.argv[4])
lams = np.linspace(float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))
nodes = [x for x in [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6] if x <= xk + 1e-9]
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=KS), 2.2); z = np.load(f)
w = harmonic_weights(S, 4)[:-1]
for lam in lams:
    M = S.mode_matrix(z, lam); smax = np.linalg.norm(M, 2)
    out = []
    for mur in (1e-6, 1e-4):
        A = np.vstack([M, mur * smax * np.diag(w)])
        out.append(np.linalg.svd(A, compute_uv=False)[-1])
    print(f"x_K={xk} lam={lam:.4f}: sigma_min(mu=1e-6)={out[0]:.3e}  sigma_min(mu=1e-4)={out[1]:.3e}", flush=True)
