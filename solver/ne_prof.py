"""Profile with the nonlinear-elimination solver: hold Delta fixed, solve strong + weak equations exactly, and record
the residual left in the selection direction (the one equation a fixed Delta cannot satisfy)."""
import numpy as np, sys, json, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make
from ne_solver import ne_solve
from dd_solver import valley_tangent
D = float(sys.argv[1]); z = np.load(sys.argv[2]); disps = [float(x) for x in sys.argv[3].split(",")]
S, R = make(D, z[-1])
t = valley_tangent(R, z)
for d in disps:
    z1, info = ne_solve(R, z + d * t, fix_delta=z[-1] + d)
    h = info["hist"][-1]
    print(json.dumps(dict(D=D, disp=d, sel=info["sel"], F=info["F"], weak=h[2], strong=h[1], junk=h[3],
                          n_outer=len(info["hist"]), s=info["seconds"])), flush=True)
