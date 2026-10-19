"""Rust double-double building blocks vs the f64 numpy prototype (same grid, same data)."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/dss-residual-kernel/hp/proto")
from taylor_proto import dmat_odd, taylor_step, ssh_series, centre_series, horner
from hpkernel.driver import make_solver, seed_from_f64, dmat_odd as dmat_dd, block_norms
M = 48
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M)
n = D - 2
s = make_solver(D, M, kt=30, kc=40, ks=100)
print("L", s.size, "len z", len(zh))
dh, dl = dmat_dd(M)
print("dmat vs f64 prototype:", np.abs(dh - dmat_odd(M)).max())
omega = 2 * np.pi / zh[-1]
Dm = dmat_odd(M)
K = 10
node = lambda i: zh[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)]
# Taylor: node 5 (x=0.3) -> 0.4 in 4 steps
xs = list(np.linspace(0.3, 0.4, 5))
t0 = time.time(); oh, ol = s.integrate(zh, zl, node(5), np.zeros(4 * M), xs); t1 = time.time()
st = node(5).reshape(4, M)
x = 0.3
for i in range(4):
    st, _ = taylor_step(st, xs[i], xs[i + 1] - xs[i], n, omega, Dm, 30)
print(f"taylor: rust vs proto {np.abs(oh.reshape(4, M) - st).max():.2e}  ({t1 - t0:.3f}s for 4 steps)")
# centre series
ch, cl = s.series(zh, zl, 0)
ch = ch.reshape(-1, 4, M)
Pc = centre_series(zh[:M], zh[M:2 * M], n, omega, Dm, 40)
print("centre coeffs: rust vs proto", max(np.abs(ch[k, q] - Pc[q][k]).max() / max(1e-300, np.abs(Pc[q][k]).max()) for k in range(2, 41) for q in range(4) if np.abs(Pc[q][k]).max() > 0))
# horizon series
t0 = time.time(); hh, hl = s.series(zh, zl, 1); t1 = time.time()
hh = hh.reshape(-1, 4, M)
mh, ah = zh[2 * M + 4 * M * K: 3 * M + 4 * M * K], zh[3 * M + 4 * M * K: 4 * M + 4 * M * K]
Ph = ssh_series(mh, ah, n, omega, Dm, 100)
for k in (0, 1, 2, 5, 10, 20, 40, 80, 100):
    print(f"  horizon order {k}: rel diff " + " ".join(f"{np.abs(hh[k, q] - Ph[q][k]).max() / max(1e-300, np.abs(Ph[q][k]).max()):.1e}" for q in range(4)))
print(f"horizon series time {t1 - t0:.3f}s")
t0 = time.time(); Fh, Fl = s.residual(zh, zl); t1 = time.time()
print(f"residual time {t1 - t0:.3f}s; block norms", " ".join(f"{v:.1e}" for v in block_norms(s, Fh, Fl)))
