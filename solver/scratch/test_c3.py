import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import jacobian_c, freeze_steps
from newton_tsvd import harmonic_weights
Dt = 9.35; z0 = np.load("diag_D9.350_mu1e-08.npy")
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
F0, J = jacobian_c(S, z0)
rng = np.random.default_rng(1)
Fz = lambda z: S.residual(z[None, :])[0]
for trial in range(2):
    v = rng.standard_normal(len(z0)); v /= np.linalg.norm(v)
    for h in (1e-7, 1e-5, 1e-3):
        fd = (Fz(z0 + h * v) - Fz(z0 - h * v)) / (2 * h)
        print(f"random dir {trial}, h={h:.0e}: |Jv - FD| / |Jv| = {np.linalg.norm(J @ v - fd) / np.linalg.norm(J @ v):.2e}", flush=True)
smax = np.linalg.norm(J, 2); w = harmonic_weights(S, 4)
for mur in (1e-10, 1e-8, 1e-6):
    mu = mur * smax
    A = np.vstack([J, mu * np.diag(w)]); b = np.concatenate([-F0, -mu * w * z0])
    dz = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = np.linalg.norm(F0 + J @ dz)
    acts = [np.linalg.norm(Fz(z0 + t * dz)) for t in (1, 0.1, 0.01)]
    print(f"mu_rel={mur:.0e}: |dz|={np.linalg.norm(dz):.2e} dDelta={dz[-1]:.2e} predicted |F|={pred:.2e}; actual at t=1,0.1,0.01: " + ", ".join(f"{a:.2e}" for a in acts), flush=True)
print(f"|F0| = {np.linalg.norm(F0):.3e}")
