import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from run_spec import make, tail
from newton_tsvd import tikhonov_newton, damp_top
D = float(sys.argv[1]); f = sys.argv[2]
S = make(31, n=D - 2); z = np.load(f)
z, nF, w = tikhonov_newton(S, damp_top(S, z), verbose=True)
print(f"D={D}: |F| {nF:.2e}  Delta {z[-1]:.12f}  tail {tail(S, z):.2e}  ({w:.0f} s)")
np.save(f"/home/claude/dss/tik_D{D:.3f}.npy", z)
