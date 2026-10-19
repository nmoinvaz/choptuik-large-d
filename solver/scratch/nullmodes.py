"""What are the near-null directions of the DSS Jacobian?  Harmonic content and block location of the right
singular vectors belonging to the smallest singular values."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make
from spec_c import jacobian_c, freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]; kcf = int(sys.argv[3]); kc = 23
S = make(kc, n=D - 2); z = np.load(f)
if kcf != kc: z = S.regrid(z, make(kcf, n=D - 2))
S = freeze_steps(S, z[-1])
F, J = jacobian_c(S, z, 3e-7)
U, s, Vt = np.linalg.svd(J)
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)]); pars = S.parities()
# harmonic index of every unknown
kidx = []
for par in pars:
    ks = S.kodd if par == "odd" else S.keven
    kidx.append(np.concatenate([ks, ks]) if par == "odd" else np.concatenate([ks[:1], ks[1:], ks[1:]]))
kidx = np.concatenate(kidx + [[-1]])
blockname = ["centre"] * 2 + [f"node{i}" for i in range(K) for _ in range(4)] + ["ssh"] * 2
bidx = np.concatenate([[i] * sz for i, sz in enumerate(sizes)] + [[len(sizes)]])
print(f"D={D}: smax={s[0]:.2e}; 10 smallest: " + " ".join(f"{v:.1e}" for v in s[-10:]))
for j in range(1, 11):
    v = Vt[-j] ** 2
    kmean = (v[:-1] * kidx[:-1]).sum() / v[:-1].sum()
    klow = v[:-1][kidx[:-1] <= 5].sum() / v[:-1].sum()
    where = {}
    for b in range(len(sizes)): where[blockname[b]] = where.get(blockname[b], 0) + v[bidx == b].sum()
    top = sorted(where.items(), key=lambda t: -t[1])[:3]
    print(f"  sigma={s[-j]:.2e}: mean harmonic {kmean:5.1f} (kc={kc}), weight in k<=5: {klow:.2f}, Delta weight {v[-1]:.1e}, mostly in " + ", ".join(f"{a} {b:.2f}" for a, b in top))
