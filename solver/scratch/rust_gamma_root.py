import numpy as np, warnings, time, json, sys
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
def lam0(S, R, z, guess, w=0.004):
    cache = {}
    def sd(l):
        if l not in cache: cache[l] = np.linalg.slogdet(R.mode_matrix(z, l))
        return cache[l]
    grid = np.linspace(guess - w, guess + w, 5); sg = [sd(l)[0] for l in grid]
    ch = [i for i in range(4) if sg[i] != sg[i + 1]]
    i = min(ch, key=lambda i: abs(grid[i] - guess)); ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
    return brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-11, rtol=1e-14), len(cache), len(ch)
py = {round(json.loads(l)['D'], 4): json.loads(l) for l in open('gamma_cs.jsonl')}
for D, f in ((11.062399999999998, "long_D11.0624.npy"), (13.9774, "long_D13.9774.npy"), (24.235037500000008, "longB_D24.2350.npy")):
    z = np.load(f); S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1]); R = RustResidual(S, threads=2)
    p = py[round(D, 4)]['lam_det']
    t = time.time(); l, nev, nch = lam0(S, R, z, p); dt = time.time() - t
    print(f"D={D:.4f}: rust lambda {l:.10f} (gamma {1/l:.10f})  python {p:.10f}  diff {l-p:+.1e}  sign changes {nch}  {nev} evals {dt:.1f}s")
