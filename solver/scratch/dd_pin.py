"""Pinning test in double-double: converge, displace Delta, re-converge, see whether Delta returns."""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
from dd_solver import dd_solve
D = float(sys.argv[1]); z = np.load(sys.argv[2]); x7 = sys.argv[3] == "0.7"; shifts = [float(s) for s in sys.argv[4].split(",")]
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L + ([0.7] if x7 else []), Ks=60), z[-1]); R = RustResidual(S, threads=2)
t = time.time()
zh, zl, nF, h = dd_solve(R, z, maxjac=15, verbose=True)
print(f"base: |F|={nF:.2e} Delta={zh[-1]:.15f} ({time.time()-t:.0f}s)", flush=True)
np.save(f"dd_D{D:.4f}_x{'7' if x7 else '6'}.npy", np.stack([zh, zl]))
for d in shifts:
    z2 = zh.copy(); z2[-1] += d
    zh2, zl2, n2, _ = dd_solve(R, z2, zl.copy(), maxjac=15)
    print(f"  start Delta {d:+.0e}: |F|={n2:.2e} Delta={zh2[-1]:.15f}  returned to within {(zh2[-1]-zh[-1]) + (zl2[-1]-zl[-1]):+.2e}", flush=True)
