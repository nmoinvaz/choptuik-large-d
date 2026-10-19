"""Growth-rate diagnostics with the complex-step horizon block: slogdet sign and regularised sigma_min over a lambda grid."""
import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, mode_matrix_cs
from newton_tsvd import harmonic_weights
D = float(sys.argv[1]); f = sys.argv[2]; xk = float(sys.argv[3]); KS = int(sys.argv[4])
lams = np.linspace(float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))
nodes = [x for x in [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6] if x <= xk + 1e-9]
z = np.load(f); S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=KS), z[-1])
w = harmonic_weights(S, 4)[:-1]
for lam in lams:
    M = mode_matrix_cs(S, z, lam); smax = np.linalg.norm(M, 2)
    sg, ld = np.linalg.slogdet(M)
    out = [np.linalg.svd(np.vstack([M, mur * smax * np.diag(w)]), compute_uv=False)[-1] for mur in (1e-8, 1e-6, 1e-4)]
    print(f"x_K={xk} lam={lam:.5f}: sign {sg:+.0f} logdet {ld:.2f} | smin mu=1e-8 {out[0]:.3e}  1e-6 {out[1]:.3e}  1e-4 {out[2]:.3e}", flush=True)
