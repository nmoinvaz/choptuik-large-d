"""Cross-check of the D>26 trend with a different matching point: move the last node from x_K=0.6 to 0.7
(horizon series evaluated at u=0.3 instead of 0.4; segment 0.6->0.7 integrated). Rust kernel."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import jacobian_rust, NODES_L
D = float(sys.argv[1]); z6 = np.load(sys.argv[2]); Ks = int(sys.argv[3]) if len(sys.argv) > 3 else 60
S6 = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z6[-1])
N7 = NODES_L + [0.7]
S7 = freeze_steps(make_long(23, D - 2, nodes=N7, Ks=Ks), z6[-1])
parts, Dl = S6.split(z6[None, :]); parts = [p[0] for p in parts]
mS = S6.phys(parts[-2][None, :], "odd"); aS = S6.phys(parts[-1][None, :], "even"); om = np.array([2 * np.pi / z6[-1]])
P, M, A, C = S6.ssh_series(mS, aS, om)
st = S6.trunc(np.stack([S6.eval_series(Q, 0.3) for Q in (P, M, A, C)]))   # (4, 1, N)
node7 = np.concatenate([S6.spec(st[0], "odd")[0], S6.spec(st[1], "odd")[0], S6.spec(st[2], "even")[0], S6.spec(st[3], "even")[0]])
sizes6, K6, L6 = S6.layout(); offs6 = np.concatenate([[0], np.cumsum(sizes6)])
nodes_end = offs6[2 + 4 * K6]
z7 = np.concatenate([z6[:nodes_end], node7, z6[nodes_end:]])
assert len(z7) == S7.layout()[2]
R7 = RustResidual(S7, threads=2); R6 = RustResidual(S6, threads=2)
print(f"D={D:.4f}: |F| x_K=0.6 {np.linalg.norm(R6(z6)):.1e}; converted to x_K=0.7 start |F| {np.linalg.norm(R7(z7)):.1e}", flush=True)
z, nF, _ = fast.lm_svd3(S7, z7, maxjac=16, tol=1e-12, verbose=True, jac=lambda S_, z_, e: jacobian_rust(S_, R7, z_, e), res=R7)
print(f"x_K=0.7 Ks={Ks}: |F|={nF:.1e} Delta={z[-1]:.10f}   x_K=0.6: {z6[-1]:.10f}   diff {z[-1]-z6[-1]:+.2e}")
np.save(f"xk7_D{D:.4f}.npy", z)
