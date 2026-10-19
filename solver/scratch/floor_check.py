import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps, jacobian_cs
D = float(sys.argv[1]); z = np.load(sys.argv[2])
S = freeze_steps(make_long(23, D - 2, nodes=[0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6], Ks=60), z[-1])
F0 = S.residual(z[None, :])[0]; print("|F| =", np.linalg.norm(F0))
rng = np.random.default_rng(0); V = rng.standard_normal((4, len(z))); V[:, -1] = 0; V /= np.linalg.norm(V, axis=1)[:, None]
for d in (1e-9, 1e-10, 1e-11):
    Fp = S.residual(z[None, :] + d * V); Fm = S.residual(z[None, :] - d * V)
    print(f"d={d:.0e}: noise ~ {np.median(np.linalg.norm(Fp + Fm - 2 * F0, axis=1)):.1e}")
F, J = jacobian_cs(S, z); U, s, Vt = np.linalg.svd(J); c = U.T @ F
for cut in (1e-6, 1e-8, 1e-9, 1e-10, 1e-12):
    m = s / s[0] >= cut
    print(f"cut {cut:.0e}: |F| outside = {np.linalg.norm(c[~m]):.1e}, GN dDelta from kept = {-(Vt[m, -1] * c[m] / s[m]).sum():+.2e}")
