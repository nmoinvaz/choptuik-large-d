"""Which discretisation parameter sets the residual floor at large D? Free solve from the same seed with
Ks (horizon series order), RK4 step counts, and centre start x0 varied; compare the converged |F|."""
import numpy as np, sys, time, json, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
import rustkernel as rk
from rustkernel import RustResidual
from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L
from dd_solver import dd_solve
D = float(sys.argv[1]); z0 = np.load(sys.argv[2]); variants = sys.argv[3].split(",")
orig_steps = rk.step_counts
for v in variants:
    Ks, mult = 60, 1
    if v.startswith("Ks"): Ks = int(v[2:])
    if v.startswith("rk"): mult = int(v[2:])
    rk.step_counts = (lambda S, m=mult: (lambda s: ([m * k for k in s[0]], s[1]))(orig_steps(S)))
    S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=Ks), z0[-1])
    if v.startswith("ls"): S.log_steps = int(v[2:])
    R = RustResidual(S, threads=2)
    t0 = time.time()
    F0 = np.linalg.norm(R(z0))
    zh, zl, nF, hist = dd_solve(R, z0.copy(), maxjac=20, use_dd=False, stall=4)
    F = R(zh); sizes, K, L = S.layout(); blk = sum(sizes[2:6])
    tail = np.linalg.norm(F[(K - 1) * blk:])
    print(json.dumps(dict(D=D, variant=v, F_seed=F0, F=nF, F_last2=float(tail), Delta=zh[-1], dDelta=zh[-1] - z0[-1], s=round(time.time() - t0))), flush=True)
    np.save(f"ft_{v}_D{D:.4f}.npy", zh)
