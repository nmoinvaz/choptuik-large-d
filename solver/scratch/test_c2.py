import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import newton_c, freeze_steps
Dt = float(sys.argv[1]); z0 = np.load(sys.argv[2])
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
# smoothness check of F along Delta
F0 = S.residual(z0[None, :])[0]
for h in (1e-9, 1e-8, 1e-7, 1e-6):
    zp = z0.copy(); zp[-1] += h
    print(f"  |F(Delta+{h:.0e}) - F(Delta)| = {np.linalg.norm(S.residual(zp[None,:])[0]-F0):.3e}", flush=True)
z, nF, t = newton_c(S, z0, mu_rel=1e-10, verbose=True)
print(f"|F| {nF:.3e} Delta {z[-1]:.12f} ({t:.0f}s)", flush=True)
np.save(f"cfix_D{Dt:.3f}.npy", z)
