import sys, re
import numpy as np
sys.path.insert(0, "/home/claude/dss-residual-kernel/solver")
sys.path.insert(0, "/home/claude/dss-residual-kernel/hp/proto")
from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L
from taylor_proto import dmat_odd, taylor_step, rk4, resample

f = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy"
D = float(re.findall(r"D([0-9.]+)\.npy", f)[0]); n = D - 2
z = np.load(f); nodes = NODES_L + [0.7]
S = freeze_steps(make_long(23, n, nodes=nodes, Ks=60), z[-1])
parts, _ = S.split(z[None, :]); parts = [p[0] for p in parts]; pars = S.parities()
F = [S.phys(np.atleast_2d(c), p)[0] for c, p in zip(parts, pars)]
omega = 2 * np.pi / z[-1]
M = 48
Dm = dmat_odd(M)
st = lambda i: np.stack([resample(F[2 + 4 * i + q], M) for q in range(4)])
# odd-parity check of resampled data: compare with direct sample at theta_j for j < M of the 72 grid where aligned
for ia, ib in ((5, 6), (6, 7), (7, 8), (8, 9)):
    xa, xb = nodes[ia], nodes[ib]
    sa, sb = st(ia), st(ib)
    # Taylor in several steps
    for nst, K in ((4, 30), (8, 30), (16, 30)):
        s = sa.copy(); h = (xb - xa) / nst; x = xa
        for _ in range(nst):
            s, _c = taylor_step(s, x, h, n, omega, Dm, K); x += h
        print(f"{xa}->{xb} taylor steps {nst:2d} K {K}: |s - node| = {np.abs(s - sb).max():.2e}", end="  ")
        if nst == 16:
            print()
    r = rk4(sa.copy(), xa, xb, 2000, n, omega, Dm)
    print(f"   rk4 2000 steps: |rk - node| = {np.abs(r - sb).max():.2e}, |rk - taylor16| = {np.abs(r - s).max():.2e}")
