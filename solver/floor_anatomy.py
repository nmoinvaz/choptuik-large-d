import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from rust_solver import make
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S, R = make(D, z[-1]); F = R(z)
sizes, K, L = S.layout(); blk = sum(sizes[2:6]); no, ne = sizes[0], sizes[1]
names = ["centre->0.02"] + [f"{a}->{b}" for a, b in zip(S.nodes[:-1], S.nodes[1:])] + ["horizon->x_K"]
print(f"D={D:.3f} |F|={np.linalg.norm(F):.2e}")
for b in range(K + 1):
    Fb = F[b * blk:(b + 1) * blk]
    parts = [Fb[:no], Fb[no:2*no], Fb[2*no:2*no+ne], Fb[2*no+ne:]]
    amp = lambda c, odd: (np.hypot(c[:len(c)//2], c[len(c)//2:]) if odd else np.r_[abs(c[0]), np.hypot(c[1:1+(len(c)-1)//2], c[1+(len(c)-1)//2:])])
    hp = amp(parts[0], True); hm = amp(parts[1], True)
    print(f"  {names[b]:14s} |F_b|={np.linalg.norm(Fb):.1e}  p:{np.linalg.norm(parts[0]):.0e} m:{np.linalg.norm(parts[1]):.0e} a:{np.linalg.norm(parts[2]):.0e} c:{np.linalg.norm(parts[3]):.0e}  p by harmonic k=1,3,..: " + " ".join(f"{v:.0e}" for v in hp[:12]))
