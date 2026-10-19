"""How well does the nonlinear-elimination solution fix Delta? Re-solve one point with the discretisation changed
(matching point, Fourier cutoff, series order, RK4 steps) and from displaced starts; compare Delta."""
import numpy as np, sys, json, time, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
import rustkernel as rk
from rustkernel import RustResidual
from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L, convert_06_to_07
from ne_solver import ne_solve
from dd_solver import valley_tangent
D = float(sys.argv[1]); z = np.load(sys.argv[2]); variants = sys.argv[3].split(",")
orig = rk.step_counts
def build(kc=23, Ks=60, nodes=NODES_L, mult=1):
    rk.step_counts = (lambda S, m=mult: (lambda s: ([m * k for k in s[0]], s[1]))(orig(S)))
    S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=Ks), z[-1])
    return S, RustResidual(S, threads=2)
S0, R0 = build()
for v in variants:
    t0 = time.time()
    if v == "base": S, R = S0, R0; z0 = z
    elif v == "xK07": S, R = build(nodes=NODES_L + [0.7]); z0 = convert_06_to_07(D, z)
    elif v.startswith("kc"): S, R = build(kc=int(v[2:])); z0 = S.regrid(z, S0)
    elif v.startswith("Ks"): S, R = build(Ks=int(v[2:])); z0 = z
    elif v.startswith("rk"): S, R = build(mult=int(v[2:])); z0 = z
    elif v.startswith("d"): S, R = S0, R0; z0 = z + float(v[1:]) * valley_tangent(R0, z)
    z1, info = ne_solve(R, z0)
    h = info["hist"][-1]
    print(json.dumps(dict(D=D, variant=v, Delta=z1[-1], diff=z1[-1] - z[-1], F=info["F"], strong=h[1], weak=h[2], junk=h[3],
                          n_weak=info["n_weak"], weak_s=[float(f"{x:.1e}") for x in info["weak_s"]], n_outer=len(info["hist"]),
                          s=round(time.time() - t0))), flush=True)
    rk.step_counts = orig
