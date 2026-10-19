import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
D = float(sys.argv[1]); z = np.load(sys.argv[2]); x7 = len(sys.argv) > 3
nodes = NODES_L + ([0.7] if x7 else [])
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=60), z[-1]); R = RustResidual(S, threads=2)
F0 = R(z); print(f"D={D:.3f} x_K={'0.7' if x7 else '0.6'}: |F|={np.linalg.norm(F0):.2e}")
rng = np.random.default_rng(0); V = rng.standard_normal((6, len(z))); V[:, -1] = 0; V /= np.linalg.norm(V, axis=1)[:, None]
for d in (1e-8, 1e-10, 1e-12, 1e-14):
    s2 = R(z[None, :] + d * V) + R(z[None, :] - d * V) - 2 * F0
    print(f"  second difference d={d:.0e}: median |.| {np.median(np.linalg.norm(s2, axis=1)):.1e}")
_, J = R.jacobian(z); s = np.linalg.svd(J, compute_uv=False)
print(f"  |J| {s[0]:.2e}, sigma_min/sigma_max {s[-1]/s[0]:.1e}; #sigma < 1e-12 smax: {(s < 1e-12*s[0]).sum()}")
# selection: residual change per unit Delta along the constrained direction (Schur complement)
y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-14)[0]; sD = np.linalg.norm(J[:, -1] - J[:, :-1] @ y)
print(f"  s_Delta (rcond 1e-14) {sD:.2e}  -> Delta resolvable to ~ noise/s_Delta")
