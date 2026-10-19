import sys, re
import numpy as np
sys.path.insert(0, "/home/claude/dss-residual-kernel/solver")
sys.path.insert(0, "/home/claude/dss-residual-kernel/hp/proto")
from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L
from taylor_proto import dmat_odd, taylor_step, resample, ssh_series, centre_series, horner

f = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/dss/ne7_js_rk3_s111111118_D38.1740.npy"
D = float(re.findall(r"D([0-9.]+)\.npy", f)[0]); n = D - 2
z = np.load(f); nodes = NODES_L + [0.7]
S = freeze_steps(make_long(23, n, nodes=nodes, Ks=60), z[-1])
parts, _ = S.split(z[None, :]); parts = [p[0] for p in parts]; pars = S.parities()
F = [S.phys(np.atleast_2d(c), p)[0] for c, p in zip(parts, pars)]
omega = 2 * np.pi / z[-1]
M = 48
Dm = dmat_odd(M)
p1, c0 = resample(F[0], M), resample(F[1], M)
mh, ah = resample(F[-2], M), resample(F[-1], M)
# horizon: series at u = 0.3 vs Taylor-stepped from u = 0.2 (x 0.8 -> 0.7, backwards)
P, Mm, A, C = ssh_series(mh, ah, n, omega, Dm, 120)
for Ks in (40, 60, 80, 100, 120):
    print(f"Ks {Ks}: |term_Ks| at u=0.3: {max(np.abs(Q[Ks]).max() for Q in (P, Mm, A, C)) * 0.3 ** Ks:.1e}")
sH = lambda u: np.stack([horner(Q, u) for Q in (P, Mm, A, C)])
s = sH(0.2); x = 0.8
for _ in range(8):
    s, _c = taylor_step(s, x, -0.0125, n, omega, Dm, 30); x -= 0.0125
print(f"horizon: |series(0.3) - taylor from series(0.2)| = {np.abs(s - sH(0.3)).max():.2e}  (x = {x:.4f})")
print(f"horizon vs old node 0.7: {np.abs(sH(0.3) - np.stack([resample(F[2 + 4 * 9 + q], M) for q in range(4)])).max():.2e}")
# centre: series at 0.02 -> Taylor to 0.05 vs series at 0.05
Pc, Mc, Ac, Cc = centre_series(p1, c0, n, omega, Dm, 60)
sC = lambda x: np.stack([horner(Q, x) for Q in (Pc, Mc, Ac, Cc)])
for K in (20, 40, 60):
    print(f"centre K {K}: |term| at x=0.05: {max(np.abs(Q[K]).max() for Q in (Pc, Mc, Ac, Cc)) * 0.05 ** K:.1e}")
s = sC(0.02); x = 0.02
for _ in range(6):
    s, _c = taylor_step(s, x, 0.005, n, omega, Dm, 30); x += 0.005
print(f"centre: |series(0.05) - taylor from series(0.02)| = {np.abs(s - sC(0.05)).max():.2e}")
print(f"centre series(0.02) vs old node 0.02: {np.abs(sC(0.02) - np.stack([resample(F[2 + q], M) for q in range(4)])).max():.2e}")
