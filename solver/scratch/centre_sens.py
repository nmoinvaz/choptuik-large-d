"""How strongly does each centre-data harmonic reach the first node? Column norms of d(block 1)/d(centre p_k)."""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make
for spec in sys.argv[1:]:
    f, D = spec.split(":"); D = float(D); z = np.load(f)
    S, R = make(D, z[-1]); _, J = R.jacobian(z)
    sizes, K, L = S.layout(); blk = sum(sizes[2:6]); h = sizes[0] // 2
    Jb = J[:blk, :sizes[0]]
    cn = np.linalg.norm(Jb, axis=0); cn = np.hypot(cn[:h], cn[h:])
    print(f"D={D:6.2f} |d state(0.02)/d c_k|, k=1,3,..: " + " ".join(f"{x:.0e}" for x in cn), flush=True)
