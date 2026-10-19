"""Where do the junk (near-null) directions live? Norm of their components per node (aggregated over the junk set)
and per harmonic at each node."""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make
f, D = sys.argv[1], float(sys.argv[2]); z = np.load(f)
S, R = make(D, z[-1]); _, J = R.jacobian(z)
U, s, Vt = np.linalg.svd(J); r = s / s[0]
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
for lo, hi, name in ((0, 3e-14, "junk"), (3e-14, 1e-10, "weak"), (1e-10, 1e-8, "Delta-ish")):
    V = Vt[(r >= lo) & (r < hi)]
    print(f"{name}: {len(V)} directions")
    print("  centre p/a: " + " ".join(f"{np.linalg.norm(V[:, offs[j]:offs[j+1]]):.1e}" for j in (0, 1)))
    for b, x in enumerate(S.nodes):
        j = 2 + 4 * b
        blk = V[:, offs[j]:offs[j + 4]]
        pblk = V[:, offs[j]:offs[j + 1]]; h = pblk.shape[1] // 2
        harm = np.sqrt((pblk[:, :h] ** 2 + pblk[:, h:] ** 2).sum(0))
        print(f"  node {x:4.2f}: |all| {np.linalg.norm(blk):.1e}   p by harmonic k=1,3,..: " + " ".join(f"{v:.0e}" for v in harm))
    print("  horizon m/a: " + " ".join(f"{np.linalg.norm(V[:, offs[j]:offs[j+1]]):.1e}" for j in (len(sizes) - 2, len(sizes) - 1)) + f"   Delta {np.linalg.norm(V[:, -1]):.1e}")
