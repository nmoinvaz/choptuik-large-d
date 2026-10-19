import numpy as np, warnings, time
warnings.filterwarnings('ignore')
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps
Ds = [19.214399999999998, 19.489024999999998, 19.846037499999998]
zs = [np.load(f) for f in ("longB_D19.2144.npy", "longB_D19.4890.npy", "longB_D19.8460.npy")]
Dt = 20.146037499999998
def extrap(Ds, vals, Dt):
    o = 0
    for i in range(3):
        w = 1.0
        for j in range(3):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * vals[i]
    return o
z0 = extrap(Ds, zs, Dt)
S = freeze_steps(make_long(23, Dt - 2, nodes=[0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6], Ks=60), z0[-1])
print("predictor |F| =", np.linalg.norm(S.residual(z0[None, :])[0]), flush=True)
z, nF, t = fast.lm_svd3(S, z0, maxjac=10, tol=1e-12, verbose=True)
print(f"lm_svd3: |F|={nF:.2e} Delta={z[-1]:.12f} in {t:.0f}s")
np.save("step_D20.1460.npy", z)
