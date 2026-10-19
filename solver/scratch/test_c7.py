import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import newton_c, freeze_steps
Dt = 9.35; z0 = np.load("diag_D9.350_mu1e-08.npy")
for eps in (3e-7, 1e-6):
    S = freeze_steps(make(23, n=Dt - 2), z0[-1])
    z, nF, t = newton_c(S, z0, mu_rel=1e-10, eps=eps, maxit=8, verbose=True)
    print(f"eps={eps:.0e}: |F| {nF:.3e} Delta {z[-1]:.12f} ({t:.0f}s)", flush=True)
    np.save(f"cfix_D9.350_eps{eps:.0e}.npy", z)
