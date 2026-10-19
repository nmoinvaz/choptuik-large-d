"""Match closer to the centre (x_K = 0.4, u_K = 0.6) with a long horizon series: horizon regularity is boosted by
(u_K / 0.4)^s0 relative to x_K = 0.6. Does the near-null (junk) band move up, and do different starts agree?"""
import numpy as np, sys, json, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rustkernel import RustResidual
from spec_long import make_long
from spec_c import freeze_steps
from ne_solver import ne_solve
from dd_solver import dd_solve
NODES_ALL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7]
D = float(sys.argv[1]); xk = float(sys.argv[2]); Ks = int(sys.argv[3])
nodes = [x for x in NODES_ALL if x <= xk + 1e-9]
def shrink(z, nodes_src):
    """drop node blocks beyond x_K from a solution with nodes_src"""
    S0 = make_long(23, D - 2, nodes=nodes_src, Ks=60); sizes, K, L = S0.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
    keep = len(nodes); j0 = 2 + 4 * keep; j1 = 2 + 4 * K
    return np.r_[z[:offs[j0]], z[offs[j1]:]]
for spec in sys.argv[4:]:
    f, xs = spec.split(":"); xs = float(xs)
    z = shrink(np.load(f), [x for x in NODES_ALL if x <= xs + 1e-9])
    S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=Ks), z[-1]); R = RustResidual(S, threads=2)
    F0 = np.linalg.norm(R(z))
    if F0 > 1e-8:
        z, *_ = dd_solve(R, z, maxjac=15, use_dd=False, stall=3, cut=1e-15)
    z1, info = ne_solve(R, z)
    _, J = R.jacobian(z1); s = np.linalg.svd(J, compute_uv=False); r = s / s[0]
    h = info["hist"][-1]
    print(json.dumps(dict(src=f, D=D, xK=xk, Ks=Ks, F_start=F0, Delta=z1[-1], F=info["F"], strong=h[1], weak=h[2], junk=h[3],
                          n_weak=info["n_weak"], smallest=[float(f"{x:.1e}") for x in r[-22:]])), flush=True)
    np.save(f"x04_{f}", z1)
