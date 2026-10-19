import numpy as np, mpmath as mp
from hpkernel.driver import make_solver, seed_from_f64, dmat_odd
mp.mp.dps = 40
M = 16; K = 10
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M)
s = make_solver(D, M, kt=30, kc=40, ks=4)
hh, hl = s.series(zh, zl, 1)
hh = hh.reshape(-1, 4, M); hl = hl.reshape(-1, 4, M)
mh, ah = zh[2 * M + 4 * M * K: 3 * M + 4 * M * K], zh[3 * M + 4 * M * K: 4 * M + 4 * M * K]
n = mp.mpf(D) - 2
om = 2 * mp.pi / mp.mpf(zh[-1])
dh, dl = dmat_odd(M)
Dm = mp.matrix(M, M)
for i in range(M):
    for j in range(M):
        Dm[i, j] = mp.mpf(dh[i, j]) + mp.mpf(dl[i, j])
c1 = [(n - 1) * (1 - mp.mpf(a) ** 2) for a in ah]
kap = [c + 1 - n / 2 for c in c1]
A = Dm * om
for i in range(M):
    A[i, i] += kap[i]
rhs = mp.matrix([n / 2 * mp.mpf(v) for v in mh])
p0 = mp.lu_solve(A, rhs)
err = max(abs(p0[i] - (mp.mpf(hh[0, 0, i]) + mp.mpf(hl[0, 0, i]))) for i in range(M))
print("order 0 p: |rust - mp| =", mp.nstr(err, 3), " max|p0| =", mp.nstr(max(abs(v) for v in p0), 3))
# also the c1 coefficient
err_c1 = max(abs(c1[i] - (mp.mpf(hh[1, 3, i]) + mp.mpf(hl[1, 3, i]))) for i in range(M))
print("order 1 c: |rust - mp| =", mp.nstr(err_c1, 3))
