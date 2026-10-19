import sys, time
import numpy as np, mpmath as mp
sys.path.insert(0, "/home/claude/dss-residual-kernel/hp/proto")
sys.path.insert(0, "/home/claude/dss-residual-kernel/solver")
from taylor_proto import dmat_odd, ssh_series
from mp_series import mp_ssh_series
from hpkernel.driver import make_solver, seed_from_f64
M = 16
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M)
n = D - 2; K = 10
s = make_solver(D, M, kt=30, kc=40, ks=24)
mh, ah = zh[2 * M + 4 * M * K: 3 * M + 4 * M * K], zh[3 * M + 4 * M * K: 4 * M + 4 * M * K]
omega = 2 * np.pi / zh[-1]
hh, hl = s.series(zh, zl, 1); hh = (hh + hl).reshape(-1, 4, M)
Ph = ssh_series(mh, ah, n, omega, dmat_odd(M), 24)
mfull = np.r_[mh, -mh]; afull = np.r_[ah, ah]
t0 = time.time()
P, Mm, A, C = mp_ssh_series(mp.mpf(38.174) - 2, mfull, afull, mp.mpf(2) * mp.pi / mp.mpf(zh[-1]), 24, dps=40)
print("mp time", round(time.time() - t0, 1))
for k in (1, 2, 3, 5, 8, 12, 16, 20, 24):
    mpv = [np.array([float(v) for v in Q[k][:M]]) for Q in (P, Mm, A, C)]
    r = " ".join(f"{np.abs(hh[k, q] - mpv[q]).max() / np.abs(mpv[q]).max():.1e}" for q in range(4))
    p = " ".join(f"{np.abs(Ph[q][k] - mpv[q]).max() / np.abs(mpv[q]).max():.1e}" for q in range(4))
    print(f"order {k:2d}: rust-vs-mp {r} | f64proto-vs-mp {p} | max|mp coef| {max(np.abs(v).max() for v in mpv):.1e}")
