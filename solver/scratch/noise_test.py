"""Is the residual floor numerical noise in F? Second differences of F along random directions, and series term sizes."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
F = lambda z: S.residual(z[None, :])[0]
F0 = F(z); print("|F0| =", np.linalg.norm(F0))
rng = np.random.default_rng(0)
for d in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
    v = rng.standard_normal(len(z)); v[-1] = 0; v /= np.linalg.norm(v)
    s2 = F(z + d * v) + F(z - d * v) - 2 * F0
    print(f"d={d:.0e}: |second diff| = {np.linalg.norm(s2):.2e}  (smooth => ~d^2)")
# series term sizes
parts, _ = S.split(z[None, :])
