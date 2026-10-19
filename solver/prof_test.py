import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
from dd_solver import profile, profile_minimum
D = float(sys.argv[1]); z = np.load(sys.argv[2]); x7 = sys.argv[3] == "0.7"; w = float(sys.argv[4])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L + ([0.7] if x7 else []), Ks=60), z[-1]); R = RustResidual(S, threads=2)
D0 = z[-1]; deltas = D0 + w * np.array([-2, -1, 0, 1, 2])
t = time.time(); prof, h, l = profile(R, z, np.zeros_like(z), deltas, maxjac=int(sys.argv[5]) if len(sys.argv) > 5 else 10)
for d, f in prof: print(f"  Delta {d:.10f}: phi {f:.4e}")
dstar, c = profile_minimum(prof)
print(f"D={D:.4f}: profile minimum Delta* = {dstar:.10f} (start {D0:.10f}, shift {dstar-D0:+.2e}); phi_min^2 {c[2] - c[1]**2/(4*c[0]):.2e}  ({time.time()-t:.0f}s)")
