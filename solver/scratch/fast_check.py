import numpy as np, warnings, time
warnings.filterwarnings('ignore')
from spec_long import make_long
from spec_c import freeze_steps, jacobian_cs, mode_matrix_cs
import fast
D = 19.846037499999998; z = np.load('long_D19.8460_svd2.npy')
S = freeze_steps(make_long(23, D - 2, nodes=[0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6], Ks=60), z[-1])
rng = np.random.default_rng(1); Zb = z[None, :] + 1e-4 * rng.standard_normal((6, len(z)))
res = {}
for mode in ("orig", "fast"):
    (fast.install if mode == "fast" else fast.uninstall)()
    t = time.time(); F1 = S.residual(z[None, :]); t1 = time.time() - t
    t = time.time(); Fb = S.residual(np.repeat(Zb, 17, 0)); tb = time.time() - t
    t = time.time(); _, J = jacobian_cs(S, z); tj = time.time() - t
    t = time.time(); M = mode_matrix_cs(S, z, 2.12); tm = time.time() - t
    res[mode] = (F1, Fb, J, M)
    print(f"{mode}: single F {t1:.3f}s  batch-102 F {tb:.2f}s  jacobian_cs {tj:.1f}s  mode_matrix_cs {tm:.1f}s", flush=True)
o, f = res["orig"], res["fast"]
print("max |dF| single", np.abs(o[0]-f[0]).max(), " batch", np.abs(o[1]-f[1]).max())
print("max |dJ|", np.abs(o[2]-f[2]).max(), " (|J|max", np.abs(o[2]).max(), ")  max |dM|", np.abs(o[3]-f[3]).max())
