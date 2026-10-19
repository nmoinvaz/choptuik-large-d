import numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import jacobian_c, freeze_steps
Dt = 9.35; z0 = np.load("diag_D9.350_mu1e-08.npy")
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
Fz = lambda z: S.residual(z[None, :])[0]
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
rng = np.random.default_rng(2)
v = np.zeros(L); cols = list(range(offs[-2], offs[-1])); v[cols] = rng.standard_normal(len(cols)); v /= np.linalg.norm(v)
ref = {}
for h in (1e-4, 1e-5, 1e-6, 1e-7):
    ref[h] = (Fz(z0 + h * v) - Fz(z0 - h * v)) / (2 * h)
for h, h2 in ((1e-4, 1e-5), (1e-5, 1e-6), (1e-6, 1e-7)): print(f"|FD({h:.0e}) - FD({h2:.0e})| = {np.linalg.norm(ref[h]-ref[h2]):.2e}")
for eps in (1e-5, 1e-6, 1e-7):
    F0, J = jacobian_c(S, z0, eps)
    print(f"jacobian eps={eps:.0e}: |Jv - FD(1e-6)| = {np.linalg.norm(J @ v - ref[1e-6]):.2e}   |Jv - FD(1e-5)| = {np.linalg.norm(J @ v - ref[1e-5]):.2e}", flush=True)
