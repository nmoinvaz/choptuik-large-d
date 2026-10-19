"""Scan for growing perturbation modes exp(lambda tau) of the critical solution: det M(lambda) = 0."""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from dss import DSSMulti
from refine import nodes

N = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else ""
z = np.load(f"/home/claude/dss/sol_ms_N{N}{tag}.npy")
S = DSSMulti(nodes=nodes, n=2.0, N=N, Ks=8, u0=0.03, seg_h=0.002, log_steps=150)
def sdet(lam):
    M = S.mode_matrix(z, lam)
    sg, ld = np.linalg.slogdet(M)
    sv = np.linalg.svd(M, compute_uv=False)
    return sg, ld, sv[-1] / sv[0]
t0 = time.time()
grid = np.concatenate([np.linspace(0.05, 0.95, 10), np.linspace(1.05, 5.0, 40)])
prev = None
for lam in grid:
    sg, ld, cond = sdet(lam)
    flag = "  <-- sign change" if prev is not None and sg != prev else ""
    print(f"lambda={lam:6.3f}  sign={int(sg):+d}  log|det|={ld:10.3f}  smin/smax={cond:.2e}{flag}", flush=True)
    prev = sg
print(f"({time.time()-t0:.0f} s)")
