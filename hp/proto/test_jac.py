import time
import numpy as np
from hpkernel.driver import make_solver, seed_from_f64, dd_add
M = 48
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M, kmax=11)
s = make_solver(D, M, kt=36, kc=40, ks=100)
s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
t0 = time.time(); Jh, Jl = s.jacobian(zh, zl); t1 = time.time()
print(f"jacobian {s.size}x{s.size} in {t1 - t0:.1f}s")
L = s.size; K = 10
cols = [3, M + 5, 2 * M + 7, 2 * M + 4 * M * 5 + 2 * M + 1, 2 * M + 4 * M * 9 + 3 * M + 11, 2 * M + 4 * M * K + 4, 2 * M + 4 * M * K + M + 9, L - 1]
h = 1e-10
for c in cols:
    e = np.zeros(L); e[c] = h
    a = dd_add(zh, zl, e, np.zeros(L)); b = dd_add(zh, zl, -e, np.zeros(L))
    Fa = s.residual(*a); Fb = s.residual(*b)
    fd = ((Fa[0] - Fb[0]) + (Fa[1] - Fb[1])) / (2 * h)
    J = Jh[:, c] + Jl[:, c]
    print(f"col {c:5d}: max|J| {np.abs(J).max():.2e}  max|J - FD| {np.abs(J - fd).max():.2e}")
