"""Is Delta actually pinned at this D? Start from the converged solution with Delta displaced, re-solve, see where Delta lands."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import jacobian_rust, NODES_L
D = float(sys.argv[1]); z0 = np.load(sys.argv[2]); xk = float(sys.argv[3])
nodes = NODES_L + ([0.7] if xk > 0.65 else [])
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=60), z0[-1]); R = RustResidual(S, threads=2)
print(f"D={D:.4f} x_K={xk}: base Delta {z0[-1]:.10f} |F| {np.linalg.norm(R(z0)):.1e}", flush=True)
for dD in [float(s) for s in sys.argv[4].split(",")]:
    z = z0.copy(); z[-1] += dD
    zz, nF, _ = fast.lm_svd3(S, z, maxjac=25, tol=1e-12, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
    print(f"  start Delta {dD:+.0e}: converged |F|={nF:.1e} Delta={zz[-1]:.10f}  (vs base {zz[-1]-z0[-1]:+.2e})", flush=True)
