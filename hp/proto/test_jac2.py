import numpy as np
from hpkernel.driver import make_solver, seed_from_f64, dd_add
M = 48; K = 10
D, zh, zl = seed_from_f64("/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy", M, kmax=11)
s = make_solver(D, M, kt=36, kc=40, ks=100)
s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
L = s.size
Jh, Jl = s.jacobian(zh, zl)
for c in (2 * M + 4 * M * K + 4, 2 * M + 4 * M * K + M + 9):
    J = Jh[:, c] + Jl[:, c]
    r = np.argmax(np.abs(J))
    for h in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16):
        e = np.zeros(L); e[c] = h
        a = dd_add(zh, zl, e, np.zeros(L)); b = dd_add(zh, zl, -e, np.zeros(L))
        Fa = s.residual(*a); Fb = s.residual(*b)
        fd = ((Fa[0] - Fb[0]) + (Fa[1] - Fb[1])) / (2 * h)
        print(f"col {c} h {h:.0e}: max|J-FD| {np.abs(J - fd).max():.2e} at row {np.argmax(np.abs(J - fd))} (J max at row {r}: {J[r]:.6e})")
# rounding level of the residual: kt 36 vs 48, ks 100 vs 130
F0 = s.residual(zh, zl)
for kt, ks in ((48, 100), (36, 130)):
    s2 = make_solver(D, M, kt=kt, kc=40, ks=ks); s2.set_segs(s2.plan(zh, zl, 1e-32, 0.05))
    F2 = s2.residual(zh, zl)
    d = (F0[0] - F2[0]) + (F0[1] - F2[1])
    nb = (L - 1) // (4 * M)
    print(f"kt {kt} ks {ks}: |dF| per block", " ".join(f"{np.abs(d[b * 4 * M:(b + 1) * 4 * M]).max():.0e}" for b in range(nb)))
