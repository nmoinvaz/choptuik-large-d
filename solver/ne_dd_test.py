import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, NODES_L
from ne_dd import ne_dd
D = float(sys.argv[1]); xk = float(sys.argv[2]); files = sys.argv[3:]
nodes = NODES_L + ([0.7] if xk > 0.65 else [])
for f in files:
    z = np.load(f)
    S, R = make(D, z[-1], nodes=nodes)
    print(f"{f}: start Delta {z[-1]:.12f}", flush=True)
    zh, zl, info = ne_dd(R, z, verbose=True)
    print(f"  -> Delta {zh[-1]:.13f} {zl[-1]:+.1e} |F| {info['F']:.1e} ({info['seconds']}s)", flush=True)
    np.save(f.replace(".npy", "_dd.npy"), np.stack([zh, zl]))
