import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
from regular_newton import regular_newton, make_transform
D = float(sys.argv[1]); z = np.load(sys.argv[2]); x7 = sys.argv[3] == "0.7"
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L + ([0.7] if x7 else []), Ks=60), z[-1]); R = RustResidual(S, threads=2)
T, F0, J, info = make_transform(S, R, z)
print(f"D={D:.3f}: regular-manifold rank {info['nreg']}; singular values of the singular-mode block: max {info['s_perp'][0]:.1e} min {info['s_perp'][-1]:.1e}")
print("  s_perp:", " ".join(f"{v:.0e}" for v in info['s_perp'][::4]))
zz, nF, nG, h = regular_newton(S, R, z, maxit=int(sys.argv[4]) if len(sys.argv) > 4 else 10, verbose=True)
print(f"final |F|={nF:.2e} |G|={nG:.2e} Delta={zz[-1]:.12f} (start {z[-1]:.12f}, change {zz[-1]-z[-1]:+.2e})")
np.save(sys.argv[2].replace(".npy", "_rn.npy"), zz)
