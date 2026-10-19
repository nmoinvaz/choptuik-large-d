"""Does the high-harmonic 'junk' in the centre data bias Delta? Zero harmonics k>=kcut of the centre unknowns
(p1, c0) and of the node states at x<=0.05, re-solve, compare Delta."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, jacobian_rust
D = float(sys.argv[1]); z = np.load(sys.argv[2]); kcut = int(sys.argv[3])
S, R = make(D, z[-1])
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)]); pars = S.parities()
def strip(z, blocks):
    z = z.copy()
    for b in blocks:
        c = z[offs[b]:offs[b + 1]]
        if pars[b] == "odd": ks = np.r_[S.kodd, S.kodd]
        else: ks = np.r_[S.keven[:1], S.keven[1:], S.keven[1:]]
        c[ks >= kcut] = 0.0; z[offs[b]:offs[b + 1]] = c
    return z
solve = lambda zz: fast.lm_svd3(S, zz, maxjac=14, tol=1e-12, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
z1, n1, _ = solve(z.copy())
print(f"as is:            |F|={n1:.1e} Delta={z1[-1]:.10f}", flush=True)
for nm, blocks in (("centre only", [0, 1]), ("centre + nodes 0.02,0.05", list(range(0, 10)))):
    zs = strip(z1, blocks); print(f"  {nm}: |F| after stripping = {np.linalg.norm(R(zs)):.1e}", flush=True)
    z2, n2, _ = solve(zs)
    tail = max(np.abs(z2[offs[b]:offs[b+1]][(np.r_[S.kodd, S.kodd] if pars[b]=='odd' else np.r_[S.keven[:1], S.keven[1:], S.keven[1:]]) >= kcut]).max() for b in (0,))
    print(f"stripped {nm:24s}: |F|={n2:.1e} Delta={z2[-1]:.10f}  change {z2[-1]-z1[-1]:+.2e}  centre high-k max after {tail:.1e}", flush=True)
