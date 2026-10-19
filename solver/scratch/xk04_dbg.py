import numpy as np, sys, time, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rustkernel import RustResidual
from spec_long import make_long
from spec_c import freeze_steps
from ne_solver import ne_solve
from dd_solver import dd_solve
NODES_ALL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7]
D = float(sys.argv[1]); xk = float(sys.argv[2]); Ks = int(sys.argv[3]); f = sys.argv[4]; xs = float(sys.argv[5])
nodes = [x for x in NODES_ALL if x <= xk + 1e-9]
S0 = make_long(23, D - 2, nodes=[x for x in NODES_ALL if x <= xs + 1e-9], Ks=60); sizes, K, L = S0.layout()
offs = np.concatenate([[0], np.cumsum(sizes)]); j0 = 2 + 4 * len(nodes); j1 = 2 + 4 * K
z = np.load(f); z = np.r_[z[:offs[j0]], z[offs[j1]:]]
t0 = time.time()
S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=Ks), z[-1]); R = RustResidual(S, threads=2)
F = R(z); print(f"setup {time.time()-t0:.1f}s, |F| start {np.linalg.norm(F):.2e}, one residual {time.time()-t0:.1f}s", flush=True)
sizes, K, L = S.layout(); blk = sum(sizes[2:6])
print("  block norms: " + " ".join(f"{np.linalg.norm(F[b*blk:(b+1)*blk]):.0e}" for b in range(K + 1)), flush=True)
parts, _ = S.split(z[None, :]); om = np.array([2 * np.pi / z[-1]])
P, M, A, C = S.ssh_series(S.phys(parts[-2], "odd"), S.phys(parts[-1], "even"), om)
mag = np.array([max(np.abs(Q[k]).max() for Q in (P, M, A, C)) for k in range(P.shape[0])])
if mag is not None:
    print("series |c_k| u^k at u=0.6, k=0,10,..: " + " ".join(f"{mag[k] * 0.6**k:.0e}" for k in range(0, len(mag), 10)), flush=True)
t1 = time.time(); _, J = R.jacobian(z); print(f"jacobian {time.time()-t1:.1f}s", flush=True)
s = np.linalg.svd(J, compute_uv=False); r = s / s[0]
print("smallest s/smax: " + " ".join(f"{x:.0e}" for x in r[-24:]), flush=True)
z, _, nF, hist = dd_solve(R, z, maxjac=15, use_dd=False, stall=3, cut=1e-15, verbose=True)
z1, info = ne_solve(R, z, verbose=True)
print(f"Delta {z1[-1]:.10f} |F| {info['F']:.1e}", flush=True)
np.save(f"x04dbg_{f}", z1)
