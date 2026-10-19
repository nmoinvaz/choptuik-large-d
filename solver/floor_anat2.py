import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
import fast; fast.install()
from fast import lm_svd3
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L, jacobian_rust
D = float(sys.argv[1]); z = np.load(sys.argv[2]); xm = float(sys.argv[3])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1]); m = NODES_L.index(xm)
R = RustResidual(S, threads=2, match_node=m)
z, nF, _ = lm_svd3(S, z.copy(), maxjac=15, tol=1e-13, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
F = R(z); sizes, K, L = S.layout(); blk = sum(sizes[2:6]); no, ne = sizes[0], sizes[1]
nodes = S.nodes
names = ["centre->%g" % nodes[0]] + [(f"{a}->{b}" if b_ <= m else f"{b}->{a} (in)") for b_, (a, b) in enumerate(zip(nodes[:-1], nodes[1:]), start=1)] + ["series->%g" % nodes[-1]]
print(f"D={D:.3f} match {xm}: |F|={np.linalg.norm(F):.2e}")
for b in range(K + 1):
    Fb = F[b * blk:(b + 1) * blk]
    c = Fb[:no]; hp = np.hypot(c[:no//2], c[no//2:])
    print(f"  {names[b]:16s} |F_b|={np.linalg.norm(Fb):.1e}  p:{np.linalg.norm(Fb[:no]):.0e} m:{np.linalg.norm(Fb[no:2*no]):.0e} a:{np.linalg.norm(Fb[2*no:2*no+ne]):.0e} c:{np.linalg.norm(Fb[2*no+ne:]):.0e}  p harmonics: " + " ".join(f"{v:.0e}" for v in hp[:8]))
_, J = R.jacobian(z); U, s, Vt = np.linalg.svd(J); c = U.T @ F
print("  residual by singular-value band (sigma/smax):")
for lo, hi in ((1e-6, 1), (1e-10, 1e-6), (1e-13, 1e-10), (0, 1e-13)):
    mk = (s / s[0] >= lo) & (s / s[0] < hi); print(f"    [{lo:.0e},{hi:.0e}): n={mk.sum()} |F|={np.linalg.norm(c[mk]):.1e}")
