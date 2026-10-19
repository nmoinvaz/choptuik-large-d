"""Two NE-converged solutions at the same D with different Delta: where do they differ?"""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, NODES_L
from ne_solver import ne_solve
from dd_solver import dd_solve
D = float(sys.argv[1]); seed = np.load(sys.argv[2]); xk = float(sys.argv[3])
nodes = NODES_L + ([0.7] if xk > 0.65 else [])
S, R = make(D, seed[-1], nodes=nodes)
za, ia = ne_solve(R, seed)
zl, *_ = dd_solve(R, seed, maxjac=12, use_dd=False, stall=3)
zb, ib = ne_solve(R, zl)
for nm, z, i in (("NE", za, ia), ("LM+NE", zb, ib)):
    h = i["hist"][-1]
    print(f"{nm:6s} Delta {z[-1]:.10f}  |F| {i['F']:.1e} strong {h[1]:.1e} weak {h[2]:.1e} junk {h[3]:.1e}  n_weak {i['n_weak']}")
np.save("jd_a.npy", za); np.save("jd_b.npy", zb)
d = zb - za
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
names = ["c.p", "c.a"] + [f"n{x}.{f}" for x in S.nodes for f in "pmac"] + ["h.m", "h.a"]
print(f"dDelta {d[-1]:+.2e}; largest block differences:")
w = [(np.linalg.norm(d[offs[j]:offs[j + 1]]), names[j], j) for j in range(len(sizes))]
for v, nm, j in sorted(w, reverse=True)[:6]:
    blk = d[offs[j]:offs[j + 1]]
    print(f"  {nm:10s} |d|={v:.1e}  by index: " + " ".join(f"{abs(x):.0e}" for x in blk[:12]) + " | " + " ".join(f"{abs(x):.0e}" for x in blk[12:24]))
F, J = R.jacobian(za); U, s, Vt = np.linalg.svd(J); r = s / s[0]
c = Vt @ d
for lo, hi in ((1e-10, 2), (3e-14, 1e-10), (0, 3e-14)):
    m = (r >= lo) & (r < hi); print(f"  step in band [{lo:.0e},{hi:.0e}): |V^T d| = {np.linalg.norm(c[m]):.2e}  (n={m.sum()})")
