import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, NODES_L
from ne_solver import ne_solve
from smooth_junk import smooth_junk
D = float(sys.argv[1])
for spec in sys.argv[2:]:
    f, xk = spec.split(":"); xk = float(xk)
    nodes = NODES_L + ([0.7] if xk > 0.65 else [])
    z = np.load(f); S, R = make(D, z[-1], nodes=nodes)
    out = [f"{f} (x_K {xk}): start {z[-1]:.10f}"]
    for it in range(3):
        z, h0, h1 = smooth_junk(R, S, z)
        z, info = ne_solve(R, z)
        hh = info["hist"][-1]
        out.append(f"it{it}: high harm {h0:.1e}->{h1:.1e}, Delta {z[-1]:.10f} |F| {info['F']:.1e} weak {hh[2]:.0e} junk {hh[3]:.0e}")
    print("\n   ".join(out), flush=True)
    np.save(f.replace(".npy", "_sj.npy"), z)
