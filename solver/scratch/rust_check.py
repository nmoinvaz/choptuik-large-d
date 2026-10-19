import numpy as np, warnings, time, json
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_c import jacobian_cs
from rust_solver import make, jacobian_rust
D = 29.360321400000007; z = np.load("longB_D29.3603.npy")
S, R = make(D, z[-1])
t = time.time(); Fp = S.residual(z[None, :])[0]; tp = time.time() - t
t = time.time(); Fr = R(z); tr = time.time() - t
print(f"residual: python {tp:.3f}s rust {tr:.4f}s  max|diff| {np.abs(Fp-Fr).max():.1e}  |F| {np.linalg.norm(Fp):.2e} vs {np.linalg.norm(Fr):.2e}")
t = time.time(); _, Jp = jacobian_cs(S, z); tjp = time.time() - t
t = time.time(); _, Jr = jacobian_rust(S, R, z); tjr = time.time() - t
d = Jp - Jr
print(f"jacobian: python {tjp:.1f}s rust {tjr:.2f}s  |dJ|_2 {np.linalg.norm(d, 2):.2e} (|J| {np.linalg.norm(Jp, 2):.2e})  Delta col diff {np.linalg.norm(d[:, -1]):.2e}")
sizes, K, L = S.layout(); blk = sum(sizes[2:6])
print("per row-block |dJ|:", " ".join(f"{np.linalg.norm(d[b*blk:(b+1)*blk], 2):.0e}" for b in range(K + 1)))
