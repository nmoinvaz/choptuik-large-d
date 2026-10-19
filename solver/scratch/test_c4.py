import numpy as np, warnings, time
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import jacobian_c, freeze_steps
Dt = 9.35; z0 = np.load("diag_D9.350_mu1e-08.npy")
S = freeze_steps(make(23, n=Dt - 2), z0[-1])
t0 = time.time(); F0, J = jacobian_c(S, z0); print(f"jacobian {time.time()-t0:.1f}s")
t0 = time.time(); Fz = lambda z: S.residual(z[None, :])[0]; Fz(z0); print(f"residual {time.time()-t0:.2f}s")
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
groups = {"Delta": [L - 1], "centre p1": range(offs[0], offs[1]), "centre c0": range(offs[1], offs[2]),
          "node0 p": range(offs[2], offs[3]), "node0 a": range(offs[4], offs[5]), "node6 m": range(offs[2 + 4*6 + 1], offs[2 + 4*6 + 2]),
          "ssh m": range(offs[-3], offs[-2]), "ssh a": range(offs[-2], offs[-1])}
rng = np.random.default_rng(2)
for name, cols in groups.items():
    v = np.zeros(L); cols = list(cols); v[cols] = rng.standard_normal(len(cols)); v /= np.linalg.norm(v)
    h = 1e-5
    fd = (Fz(z0 + h * v) - Fz(z0 - h * v)) / (2 * h)
    e = J @ v - fd
    rows = np.argsort(-np.abs(e))[:3]
    print(f"{name:10s}: |Jv|={np.linalg.norm(J@v):.2e}  |Jv-FD|={np.linalg.norm(e):.2e}  worst rows {rows.tolist()} (block {[(r // sum(sizes[2:6])) for r in rows]})", flush=True)
