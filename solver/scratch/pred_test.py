import numpy as np, warnings, json
warnings.filterwarnings("ignore")
from rust_solver import make
rs = {round(json.loads(l)['D'], 6): json.loads(l)['D'] for l in open('scan_longB.jsonl')}
Ds = sorted(rs.values())[-4:-1]; zs = [np.load(f"longB_D{d:.4f}.npy") for d in Ds]
Dtrue = sorted(rs.values())[-1]; ztrue = np.load(f"longB_D{Dtrue:.4f}.npy")
print("seeds", [f"{d:.3f}" for d in Ds], "target (known solution)", f"{Dtrue:.3f}")
S, R = make(Dtrue, ztrue[-1])
print("|F| at known solution", f"{np.linalg.norm(R(ztrue)):.1e}")
def lag(Ds, vals, Dt, k):
    Ds, vals = Ds[-k:], vals[-k:]; o = 0
    for i in range(k):
        w = 1.0
        for j in range(k):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * vals[i]
    return o
for k, nm in ((1, "constant"), (2, "linear"), (3, "quadratic")):
    zp = lag(Ds, zs, Dtrue, k); d = zp - ztrue
    print(f"{nm:9s}: |F| {np.linalg.norm(R(zp)):.2e}  |z_pred - z_true| {np.linalg.norm(d):.2e}  Delta err {d[-1]:+.2e}")
# hybrid: quadratic for Delta and smooth part, constant for the rest?
sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
zq = lag(Ds, zs, Dtrue, 3); zl = lag(Ds, zs, Dtrue, 2)
for nm, blocks in (("centre p1,c0", (0, 1)), ("horizon mS,aS", (len(sizes) - 2, len(sizes) - 1))):
    for k in (1, 2, 3):
        zp = zq.copy()
        for b in blocks: zp[offs[b]:offs[b+1]] = lag(Ds, zs, Dtrue, k)[offs[b]:offs[b+1]]
        print(f"quadratic but {nm} with order-{k}: |F| {np.linalg.norm(R(zp)):.2e}")
Ds4 = sorted(rs.values())[-5:-1]; zs4 = [np.load(f"longB_D{d:.4f}.npy") for d in Ds4]
for k in (3, 4):
    zp = lag(Ds4, zs4, Dtrue, k); d = zp - ztrue
    print(f"order-{k} from {[round(x,3) for x in Ds4[-k:]]}: |F| {np.linalg.norm(R(zp)):.2e}  |dz| {np.linalg.norm(d):.2e}  Delta err {d[-1]:+.2e}")
# horizon-only least-squares fix: Gauss-Newton on horizon unknowns to match the node-0.6 state
from rust_solver import jacobian_rust
zp = lag(Ds, zs, Dtrue, 3)
hor = np.arange(offs[len(sizes) - 2], offs[len(sizes)]); blk = sum(sizes[2:6]); rows = np.arange(K * blk, (K + 1) * blk)
for it in range(6):
    F = R(zp); Zc = np.repeat(zp[None, :].astype(complex), len(hor), 0); Zc[np.arange(len(hor)), hor] += 1e-30j
    Jh = (R.residual_complex(Zc, dedup=True)[:, rows].imag / 1e-30).T
    dz, *_ = np.linalg.lstsq(Jh, -F[rows], rcond=1e-12); zp[hor] += dz
    print(f"  horizon fix it {it}: block |F| {np.linalg.norm(F[rows]):.2e} -> total |F| {np.linalg.norm(R(zp)):.2e}")
