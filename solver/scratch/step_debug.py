import numpy as np, warnings, json, time, sys
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, jacobian_rust
from spec_c import jacobian_cs
rs = {round(json.loads(l)['D'], 6): json.loads(l)['D'] for l in open('scan_longB.jsonl')}
Ds = sorted(rs.values())[-3:]; zs = [np.load(f"longB_D{d:.4f}.npy") for d in Ds]
Dt = Ds[-1] + 0.4
def extrap(Ds, vals, Dt):
    o = 0
    for i in range(3):
        w = 1.0
        for j in range(3):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * vals[i]
    return o
z0 = extrap(Ds, zs, Dt)
S, R = make(Dt, z0[-1])
print("seeds", [f"{d:.3f}" for d in Ds], "target", f"{Dt:.4f}")
print("predictor |F| rust", np.linalg.norm(R(z0)), " python", np.linalg.norm(S.residual(z0[None, :])[0]), flush=True)
mode = sys.argv[1]
t = time.time()
if mode == "rust":
    z, nF, _ = fast.lm_svd3(S, z0, maxjac=12, tol=1e-12, verbose=True, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
else:
    z, nF, _ = fast.lm_svd3(S, z0, maxjac=12, tol=1e-12, verbose=True)
print(mode, f"|F|={nF:.2e} Delta={z[-1]:.10f} {time.time()-t:.0f}s")
