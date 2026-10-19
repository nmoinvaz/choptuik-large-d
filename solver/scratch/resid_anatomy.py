import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
D = float(sys.argv[1]); f = sys.argv[2]
z = np.load(f); S = freeze_steps(make_long(23, D - 2, nodes=[0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], Ks=60), z[-1])
F = S.residual(z[None, :])[0]
sizes, K, L = S.layout(); blk = sum(sizes[2:6])
names = ["centre->0.02"] + [f"{S.nodes[i]}->{S.nodes[i+1]}" for i in range(K - 1)] + ["horizon series->0.6"]
for b in range(K + 1):
    r = F[b * blk:(b + 1) * blk]
    # split by field (p, m odd; a, c even) and by harmonic: within each field block the layout is [re..., im...]
    o = np.cumsum([0] + sizes[2:6]); parts = [r[o[i]:o[i + 1]] for i in range(4)]
    print(f"{names[b]:22s} |F| = {np.linalg.norm(r):.1e}   p {np.linalg.norm(parts[0]):.1e}  m {np.linalg.norm(parts[1]):.1e}  a {np.linalg.norm(parts[2]):.1e}  c {np.linalg.norm(parts[3]):.1e}")
# harmonic content of the worst block
b = int(np.argmax([np.linalg.norm(F[i * blk:(i + 1) * blk]) for i in range(K + 1)]))
r = F[b * blk:(b + 1) * blk]; no = sizes[2]; half = no // 2
pm = np.abs(r[:no][:half]) + np.abs(r[:no][half:])
print("worst block", names[b], "p-residual by odd harmonic k=1,3,...:", " ".join(f"{v:.0e}" for v in pm))
print("phase-condition residual", F[-1])
