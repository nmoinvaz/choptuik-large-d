import numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import jacobian_c, freeze_steps
from newton_tsvd import harmonic_weights
Dt = 9.35; z0 = np.load("diag_D9.350_mu1e-08.npy")
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
Fz = lambda z: S.residual(z[None, :])[0]
F0, J = jacobian_c(S, z0, 3e-7)
smax = np.linalg.norm(J, 2); w = harmonic_weights(S, 4); w[-1] = 0
print(f"|F0| {np.linalg.norm(F0):.3e}")
for p in (4, 0):
    wp = harmonic_weights(S, p); wp[-1] = 0 if p else 1e-3
    for mur in (1e-9, 1e-7, 1e-5):
        mu = mur * smax
        A = np.vstack([J, mu * np.diag(wp)]); b = np.concatenate([-F0, np.zeros(len(wp))])
        dz = np.linalg.lstsq(A, b, rcond=None)[0]
        pred = np.linalg.norm(F0 + J @ dz)
        acts = [np.linalg.norm(Fz(z0 + t * dz)) for t in (1, 0.5)]
        print(f"step-penalty p={p} mu_rel={mur:.0e}: |dz|={np.linalg.norm(dz):.2e} dDelta={dz[-1]:+.2e} pred={pred:.2e} actual t=1: {acts[0]:.2e}, t=0.5: {acts[1]:.2e}", flush=True)
