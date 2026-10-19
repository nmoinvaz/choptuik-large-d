"""Anatomy of the near-null directions at a stalled solution: what the weak right singular vectors are made of
(centre data / node states / horizon data / Delta), and the Gauss-Newton step they would need."""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from rustkernel import RustResidual
from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1]); R = RustResidual(S, threads=2)
F, J = R.jacobian(z); U, s, Vt = np.linalg.svd(J); c = U.T @ F
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
names = ["c.p", "c.a"] + [f"n{x}.{f}" for x in S.nodes for f in "pmac"] + ["h.m", "h.a", "Delta"]
print(f"D={D:.3f} |F|={np.linalg.norm(F):.2e} smax={s[0]:.2e} sizes={sizes[:2]} ... {sizes[-2:]}")
order = np.argsort(s)
for i in order[:26][::-1]:
    v = Vt[i]; w = np.array([np.linalg.norm(v[offs[j]:offs[j + 1]]) for j in range(len(sizes))] + [abs(v[-1])])
    top = np.argsort(w)[::-1][:4]
    # which harmonic of the dominant block
    j = top[0]; blkv = v[offs[j]:offs[j + 1]] if j < len(sizes) else None
    print(f"  s/smax={s[i]/s[0]:.1e} |c|={abs(c[i]):.1e} step={abs(c[i])/s[i]:.1e} Dl={v[-1]:+.2f} " +
          " ".join(f"{names[t]}:{w[t]:.2f}" for t in top) +
          ("" if blkv is None else f"  peak idx {np.argmax(abs(blkv))}/{len(blkv)}"))
