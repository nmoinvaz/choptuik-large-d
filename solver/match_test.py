"""Delta with the match node moved inward (segments beyond it integrated from the horizon side)."""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
import fast; fast.install()
from fast import lm_svd3
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L, jacobian_rust
D = float(sys.argv[1]); z = np.load(sys.argv[2]); x7 = sys.argv[3] == "0.7"; matches = [float(x) for x in sys.argv[4].split(",")]
nodes = NODES_L + ([0.7] if x7 else [])
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=60), z[-1])
for xm in matches:
    m = nodes.index(xm)
    R = RustResidual(S, threads=2, match_node=m); t = time.time()
    z2, nF, _ = lm_svd3(S, z.copy(), maxjac=15, tol=1e-13, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
    print(f"D={D:.4f} x_K={nodes[-1]} match at x={xm}: start |F|={np.linalg.norm(R(z)):.1e} -> {nF:.1e}  Delta={z2[-1]:.10f} (outward-match value {z[-1]:.10f}, diff {z2[-1]-z[-1]:+.2e})  {time.time()-t:.0f}s", flush=True)
    np.save(sys.argv[2].replace(".npy", f"_m{xm}.npy"), z2)
