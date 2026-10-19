import numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import freeze_steps
Dt = 9.35; z0 = np.load("diag_D9.350_mu1e-08.npy")
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
parts, Dl = S.split(z0[None, :]); om = np.array([2 * np.pi / Dl[0]])
mS = S.phys(parts[-2], "odd"); aS0 = S.phys(parts[-1], "even")
rng = np.random.default_rng(3); v = rng.standard_normal(aS0.shape); v /= np.linalg.norm(v)
def series_state(aS):
    P, M, A, C = S.ssh_series(mS, aS, om)
    return P, M, A, C, S.trunc(np.stack([S.eval_series(Q, S.u0) for Q in (P, M, A, C)]))
def lastnode(aS):
    return S._ssh_to_last(mS, aS, om)
for name, fn in (("series state at u0", lambda a: series_state(a)[4]), ("state at last node", lastnode)):
    d = {h: (fn(aS0 + h * v) - fn(aS0 - h * v)) / (2 * h) for h in (1e-4, 1e-5, 1e-6)}
    print(f"{name}: |d| = {np.linalg.norm(d[1e-6]):.3e};  |d(1e-4)-d(1e-5)| = {np.linalg.norm(d[1e-4]-d[1e-5]):.2e};  |d(1e-5)-d(1e-6)| = {np.linalg.norm(d[1e-5]-d[1e-6]):.2e}")
P, M, A, C, _ = series_state(aS0)
for name, Q in (("P", P), ("M", M), ("A", A), ("C", C)):
    print(name, " ".join(f"{np.abs(q).max():.1e}" for q in Q))
# per-order sensitivity of P coefficients
Pp = series_state(aS0 + 1e-5 * v)[0]; Pm = series_state(aS0 - 1e-5 * v)[0]
Pp2 = series_state(aS0 + 1e-4 * v)[0]; Pm2 = series_state(aS0 - 1e-4 * v)[0]
for K in range(len(P)):
    d1 = (Pp[K] - Pm[K]) / 2e-5; d2 = (Pp2[K] - Pm2[K]) / 2e-4
    print(f"P_{K}: |dP/da| {np.abs(d1).max():.2e}  curvature proxy |d(1e-4)-d(1e-5)| {np.abs(d2-d1).max():.2e}")
