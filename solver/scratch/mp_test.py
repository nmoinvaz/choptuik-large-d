"""Compare the f64 and mp horizon series at one D: coefficient growth and convergence away from the horizon."""
import sys, time
import numpy as np
import mpmath as mp
from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L
from mp_series import mp_ssh_series, harmonic, evaluate

D = float(sys.argv[1]); f = sys.argv[2]; K = int(sys.argv[3]); dps = int(sys.argv[4]); kc = int(sys.argv[5]) if len(sys.argv) > 5 else 23
nodes = NODES_L + ([0.7] if "ne7" in f else [])
z = np.load(f); n = D - 2
S = freeze_steps(make_long(kc, n, nodes=nodes, Ks=K), z[-1])
parts, _ = S.split(z[None, :]); parts = [p[0] for p in parts]; pars = S.parities()
F = [S.phys(c[None, :], p)[0] for c, p in zip(parts, pars)]
om = 2 * np.pi / z[-1]
t = time.time()
P, M, A, C = mp_ssh_series(n, F[-2], F[-1], om, K, dps=dps)
print(f"mp series K={K} dps={dps}: {time.time() - t:.0f}s", flush=True)
Pf, Mf, Af, Cf = S.ssh_series(F[-2][None, :], F[-1][None, :], np.array([om]))
for j in (5, 20, 40, 60, 80, 120, 160, 199):
    if j < len(M):
        hm = abs(harmonic(M[j])); hf = abs(np.fft.rfft(Mf[j][0])[1] / len(Mf[j][0]))
        print(f"j={j:3d} |M_j|_1 mp {mp.nstr(hm, 5):>12s}  f64 {hf:.4e}   ratio^(1/j) mp {mp.nstr(hm ** (-mp.mpf(1) / j), 4) if hm > 0 else '-'}")
for i, x in enumerate(nodes):
    if x < 0.09:
        continue
    node = np.stack(F[2 + 4 * i:6 + 4 * i])
    sm = np.array([[float(v) for v in evaluate(Q, mp.mpf(1 - x))] for Q in (P, M, A, C)])
    print(f"x={x}: |mp series - node| {np.abs(sm - node).max():.1e}")
