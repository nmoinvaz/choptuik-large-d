import numpy as np, warnings, sys, json
warnings.filterwarnings("ignore")
import fast; fast.install()
from fast import lm_svd3
from rust_solver import make, jacobian_rust
from dd_solver import profile, profile_minimum
Ds = [30.51032140000001, 30.84832140000001, 31.04832140000001, 31.30832140000001]
zs = [np.load(f"longB_D{d:.4f}.npy") for d in Ds]
Dt = float(sys.argv[1])
def lag(Ds, zs, Dt):
    o = 0
    for i in range(len(Ds)):
        w = 1.0
        for j in range(len(Ds)):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * zs[i]
    return o
z0 = lag(Ds, zs, Dt); S, R = make(Dt, z0[-1])
z, nF, _ = lm_svd3(S, z0, maxjac=12, tol=1e-12, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
print(f"free solve |F|={nF:.2e} Delta={z[-1]:.10f}")
for w in (1e-4, 3e-4, 1e-3):
    prof, _, _ = profile(R, z, np.zeros_like(z), z[-1] + w * np.array([-2, -1, 0, 1, 2]), maxjac=12, use_dd=False)
    print(f" w={w:.0e}: phi", " ".join(f"{v:.2e}" for v in prof[:, 1]), " -> Delta*", f"{profile_minimum(prof)[0]:.10f}")
