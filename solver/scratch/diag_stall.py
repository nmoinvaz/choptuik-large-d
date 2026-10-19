"""Why does continuation stall near D = 9.25?  Try one step with the solver verbose and record the residual floor."""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from run_spec import make
from newton_tsvd import tikhonov_newton, damp_top
Dt = float(sys.argv[1]); kc = 23
z1 = np.load("solD_9.203_kc23.npy"); z2 = np.load("solD_9.234_kc23.npy")
D1, D2 = 9.203125, 9.233642578125
g = z2 + (z2 - z1) * (Dt - D2) / (D2 - D1)
S = make(kc, n=Dt - 2)
g = damp_top(S, g)
for mu in (1e-7, 1e-8):
    t0 = time.time()
    z, nF, _ = tikhonov_newton(S, g, mu_rel=mu, verbose=True)
    print(f"mu_rel={mu}: final |F| {nF:.3e}  Delta {z[-1]:.10f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"diag_D{Dt:.3f}_mu{mu:.0e}.npy", z)
