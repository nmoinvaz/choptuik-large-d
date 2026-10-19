"""D-continuation where each point's Delta is the minimum of the profile phi(Delta) = min |F| at fixed Delta.
usage: scan_profile.py out.jsonl D_end h hmax "D:file;D:file;D:file;D:file"   (x_K = 0.6, Rust kernel)"""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
import fast; fast.install()
from fast import lm_svd3
from rust_solver import make, jacobian_rust
from dd_solver import dd_solve, profile, profile_minimum
out, D_end, h, hmax = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
hist = []
for item in sys.argv[5].split(";"):
    Ds, f = item.split(":"); hist.append((float(Ds), np.load(f)))
def lag(Ds, zs, Dt):
    o = 0
    for i in range(len(Ds)):
        w = 1.0
        for j in range(len(Ds)):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * zs[i]
    return o
def point(Dt, z0, w):
    S, R = make(Dt, z0[-1])
    z, nF, _ = lm_svd3(S, z0, maxjac=12, tol=1e-12, jac=lambda S_, z_, e: jacobian_rust(S_, R, z_, e), res=R)
    if not np.isfinite(nF) or nF > 1e-5: return None
    deltas = z[-1] + w * np.array([-2, -1, 0, 1, 2])
    prof, _, _ = profile(R, z, np.zeros_like(z), deltas, maxjac=12, use_dd=False)
    dstar, c = profile_minimum(prof)
    vshape = min(prof[0, 1], prof[-1, 1]) > 3 * prof[2, 1]
    zs = z.copy(); zs[-1] = dstar
    zs, _, nFs, _ = dd_solve(R, zs, None, maxjac=12, fix_delta=True, use_dd=False)
    return dict(D=Dt, Delta=float(dstar), Delta_free=float(z[-1]), residual_free=float(nF), phi=[float(v) for v in prof[:, 1]],
                width=w, phi_at_star=float(nFs), vshape=bool(vshape)), zs
while hist[-1][0] < D_end - 1e-9:
    Dt = min(hist[-1][0] + h, D_end); t0 = time.time()
    Ds = [d for d, _ in hist[-4:]]; zs = [z for _, z in hist[-4:]]
    r = point(Dt, lag(Ds, zs, Dt), 1e-4)
    if r is None or not r[0]["vshape"]:
        h /= 2; print(f"step to D={Dt:.4f} failed ({'no convergence' if r is None else 'profile not V-shaped'}); h -> {h}", flush=True)
        if h < 0.04: print("giving up", flush=True); break
        continue
    rec, z = r; rec["seconds"] = round(time.time() - t0)
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"prof_D{Dt:.4f}.npy", z); hist.append((Dt, z)); h = min(h * 1.3, hmax)
