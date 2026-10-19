import numpy as np, warnings, time, json
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps, jacobian_cs, mode_matrix_cs
from rustkernel import RustResidual
from rust_solver import NODES_L
D = 24.235037500000008; z = np.load("longB_D24.2350.npy")
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1]); R = RustResidual(S, threads=2)
t = time.time(); F0, Jr = R.jacobian(z); tr = time.time() - t
t = time.time(); _, Jp = jacobian_cs(S, z); tp = time.time() - t
sizes, K, L = S.layout(); blk = sum(sizes[2:6])
d = Jr - Jp
print(f"Jacobian: rust {tr:.2f}s python {tp:.2f}s | |dJ|_2/|J|_2 = {np.linalg.norm(d,2)/np.linalg.norm(Jp,2):.1e}; per row-block max|d|:",
      " ".join(f"{np.abs(d[b*blk:(b+1)*blk]).max():.0e}" for b in range(K + 1)), f" Delta col {np.abs(d[:,-1]).max():.0e}")
for lam in (2.0, 2.0951194518):
    t = time.time(); Mr = R.mode_matrix(z, lam); tr = time.time() - t
    t = time.time(); Mp = mode_matrix_cs(S, z, lam); tp = time.time() - t
    d = Mr - Mp
    print(f"M({lam}): rust {tr:.2f}s python {tp:.2f}s | max|d| {np.abs(d).max():.1e} (max|M| {np.abs(Mp).max():.1e}); per block:",
          " ".join(f"{np.abs(d[b*blk:(b+1)*blk]).max():.0e}" for b in range(K + 1)),
          f"| slogdet rust {np.linalg.slogdet(Mr)} python {np.linalg.slogdet(Mp)}")
