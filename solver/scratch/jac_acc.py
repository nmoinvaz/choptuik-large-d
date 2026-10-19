import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps, jacobian_c
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1])
Js = {}
for e in (3e-6, 3e-7, 3e-8, 3e-9):
    F, Js[e] = jacobian_c(S, z, e)
U, s, Vt = np.linalg.svd(Js[3e-8])
for e in Js:
    if e == 3e-8: continue
    d = Js[e] - Js[3e-8]
    print(f"eps {e:.0e} vs 3e-8: |dJ|_2={np.linalg.norm(d,2):.2e}  |dJ V_low| (last 25) max {max(np.linalg.norm(d@Vt[i]) for i in range(len(s)-25,len(s))):.2e}")
c = U.T @ F
for e in Js:
    U2, s2, Vt2 = np.linalg.svd(Js[e]); c2 = U2.T @ F
    print(f"eps {e:.0e}: sigma_min/smax {s2[-1]/s2[0]:.1e}  #<1e-9: {(s2/s2[0]<1e-9).sum()}  descent test:")
    for mu in (1e-4, 1e-6, 1e-8):
        J = Js[e]; m = mu * s2[0]
        dz = np.linalg.solve(J.T @ J + m * m * np.eye(len(z)), -J.T @ F)
        Ft = S.residual((z + dz)[None, :])[0]
        print(f"     mu {mu:.0e}: |dz| {np.linalg.norm(dz):.1e}  |F| {np.linalg.norm(F):.3e} -> {np.linalg.norm(Ft):.3e}  dDelta {dz[-1]:+.2e}")
