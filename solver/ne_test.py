import numpy as np, sys, json, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make
from ne_solver import ne_solve
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S, R = make(D, z[-1])
z1, info = ne_solve(R, z, verbose=True)
print(json.dumps(dict(D=D, Delta0=z[-1], Delta=z1[-1], shift=z1[-1] - z[-1], F=info["F"], s=info["seconds"])), flush=True)
np.save(f"ne_D{D:.4f}.npy", z1)
