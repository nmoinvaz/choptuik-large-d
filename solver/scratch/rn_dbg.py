import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
from regular_newton import make_transform, _res, _rows_perp
D = 24.53503750000001; z = np.load("longB_D24.5350.npy")
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1]); R = RustResidual(S, threads=2)
T, F0, J, info = make_transform(S, R, z)
F = _res(R, z, True); G = T(F); JG = T(J)
U, s, Vt = np.linalg.svd(JG, full_matrices=False)
print("sigma(JG): max %.1e, at cut 1e-13: %d kept; smallest kept %.1e; tiny ones:" % (s[0], (s >= 1e-13*s[0]).sum(), s[s >= 1e-13*s[0]][-1]), " ".join(f"{v:.0e}" for v in s[-25:]))
c = U.T @ G
for cut in (1e-6, 1e-9, 1e-11, 1e-13):
    keep = s >= cut * s[0]
    dz = -(Vt[keep].T @ (c[keep] / s[keep]))
    lin = np.linalg.norm(G + JG @ dz)
    acts = []
    for a in (1.0, 0.1, 0.01):
        acts.append(np.linalg.norm(T(_res(R, z + a * dz, True))))
    print(f"cut {cut:.0e}: |dz| {np.linalg.norm(dz):.1e} dDelta {dz[-1]:+.1e} | |G| {np.linalg.norm(G):.2e} linear pred {lin:.2e} actual a=1,0.1,0.01: " + " ".join(f"{v:.2e}" for v in acts))
rp = _rows_perp(S)
print("G singular rows", np.linalg.norm(G[rp]), " residual of G outside", np.linalg.norm(np.delete(G, np.r_[rp])))
