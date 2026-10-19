import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_long import make_long, from_standard
from spec_c import lm_solve, freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]; kcf = int(sys.argv[3]); kc = int(sys.argv[4])
S_std = make(kcf, n=D - 2); z = np.load(f)
if kcf != kc: S2 = make(kc, n=D - 2); z = S2.regrid(z, S_std); S_std = S2
nodes = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=60), z[-1])
zl = from_standard(S_std, z, S)
t0 = time.time(); zl, nF, _ = lm_solve(S, zl, maxjac=20)
print(f"D={D} kc={kc}: |F|={nF:.1e} Delta={zl[-1]:.10f} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"long_D{D:.3f}_kc{kc}.npy", zl)
