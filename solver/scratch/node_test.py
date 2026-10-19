"""Is the weak Delta-selection a property of the equations or of where we match?  Drop the last matching node(s)
(0.9, then 0.8) so the horizon segment ends further from the SSH, re-converge, and recompute the Schur sensitivity.
If the weakness is the (1-x)^s0 suppression of the horizon-singular piece, s_Delta should scale like (1-x_K)^s0."""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from spec import DSSSpec
from run_spec import NODES
from spec_c import jacobian_c, freeze_steps, lm_solve
D = float(sys.argv[1]); f = sys.argv[2]; kc = 23; kcf = int(sys.argv[3]) if len(sys.argv) > 3 else 23
def mk(nodes): return DSSSpec(kc=kc, nodes=nodes, n=D - 2, Ks=8, u0=0.03, seg_h=0.002, log_steps=150)
S0 = mk(NODES); z0 = np.load(f)
if kcf != kc: z0 = S0.regrid(z0, DSSSpec(kc=kcf, nodes=NODES, n=D - 2, Ks=8, u0=0.03, seg_h=0.002, log_steps=150))
def drop_last(S, z):
    sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
    j = 2 + 4 * (K - 1)                      # first block of the last node
    keep = np.r_[0:offs[j], offs[j + 4]:L]
    return z[keep]
def s_delta(S, z):
    F, J = jacobian_c(S, z, 3e-7)
    Jd, Jo = J[:, -1], J[:, :-1]
    y = np.linalg.lstsq(Jo, Jd, rcond=1e-13)[0]
    return np.linalg.norm(F), np.linalg.norm(Jd - Jo @ y)
nodes, S, z = list(NODES), S0, z0
for step in range(3):
    S = freeze_steps(S, z[-1])
    t0 = time.time()
    z, nF, _ = lm_solve(S, z, maxjac=6)
    F, sD = s_delta(S, z)
    print(f"D={D}: last node x_K={nodes[-1]:.2f}  |F|={F:.1e}  Delta={z[-1]:.8f}  s_Delta={sD:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    if step < 2:
        z = drop_last(S, z); nodes = nodes[:-1]; S = mk(nodes)
