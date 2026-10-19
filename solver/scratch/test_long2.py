import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_long import make_long, from_standard
from spec_c import lm_solve, jacobian_c, freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]; kc = 23
S_std = make(kc, n=D - 2); z = np.load(f)
nodes = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=60), z[-1])
zl = from_standard(S_std, z, S) if len(z) != S.layout()[2] else z
zl, nF, t = lm_solve(S, zl, maxjac=int(sys.argv[3]), verbose=True)
print(f"final |F|={nF:.2e} Delta={zl[-1]:.10f} ({t:.0f}s)")
np.save(f"long_D{D:.3f}.npy", zl)
