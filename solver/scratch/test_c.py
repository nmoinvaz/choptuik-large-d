import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import newton_c
Dt = float(sys.argv[1]); z0 = np.load(sys.argv[2])
S = make(23, n=Dt - 2)
for mu in (1e-10, 1e-9):
    z, nF, t = newton_c(S, z0, mu_rel=mu, verbose=True)
    print(f"mu_rel={mu}: |F| {nF:.3e} Delta {z[-1]:.12f} ({t:.0f}s)", flush=True)
    np.save(f"cdiag_D{Dt:.3f}_mu{mu:.0e}.npy", z)
