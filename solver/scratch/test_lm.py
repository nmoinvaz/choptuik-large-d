import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import lm_solve, freeze_steps
Dt = float(sys.argv[1]); z0 = np.load(sys.argv[2])
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
z, nF, t = lm_solve(S, z0, verbose=True)
print(f"LM: |F| {nF:.3e} Delta {z[-1]:.12f} ({t:.0f}s)", flush=True)
np.save(f"lm_D{Dt:.3f}.npy", z)
