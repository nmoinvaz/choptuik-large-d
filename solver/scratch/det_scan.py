import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]; xk = float(sys.argv[3]); KS = int(sys.argv[4])
lams = np.linspace(float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]))
nodes = [x for x in [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6] if x <= xk + 1e-9]
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=KS), 2.2); z = np.load(f)
for lam in lams:
    M = S.mode_matrix(z, lam); s, ld = np.linalg.slogdet(M); sv = np.linalg.svd(M, compute_uv=False)
    print(f"x_K={xk} lam={lam:.4f}: sign={s:+.0f} log|det|={ld:9.3f}  smallest sv {sv[-1]:.2e} {sv[-2]:.2e}", flush=True)
