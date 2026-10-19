import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
import fast; fast.install()
from fast import lm_svd3
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L, jacobian_rust
from dd_solver import profile, profile_minimum
D = float(sys.argv[1]); z = np.load(sys.argv[2]); xm = float(sys.argv[3]); w = float(sys.argv[4])
nodes = NODES_L
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=60), z[-1]); R = RustResidual(S, threads=2, match_node=nodes.index(xm))
t = time.time()
z, nF, _ = lm_svd3(S, z.copy(), maxjac=15, tol=1e-13, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
print(f"D={D:.4f} match x={xm}: free solve |F|={nF:.2e} Delta={z[-1]:.10f}", flush=True)
prof, _, _ = profile(R, z, np.zeros_like(z), z[-1] + w * np.array([-2, -1, 0, 1, 2]), maxjac=12, use_dd=False)
print("  phi:", " ".join(f"{v:.2e}" for v in prof[:, 1]), f" -> Delta* {profile_minimum(prof)[0]:.10f}  ({time.time()-t:.0f}s)", flush=True)
