"""Continuation in D with the long-horizon-series formulation (matching at x_K = 0.6 from a Ks-term SSH series).
Skips dimensions where the horizon exponent s0 = n/(2 lambda_g) - 1 is within RES of an integer (resonant series order).
usage: scan_long.py out D_end h HMAX  seeds "D:file:kcsrc;..." """
import sys, json, time, numpy as np, warnings
import fast; fast.install()
warnings.filterwarnings("ignore")
from run_spec import make, tail
from spec_long import make_long, from_standard
from spec_c import jacobian_cs, freeze_steps
from fast import lm_svd3
from rust_solver import make as make_rust, jacobian_rust
from gamma import growth_rate
kc = 23; NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]; KS = 60; RES = 0.0; TOL = 5e-11
out = sys.argv[1]; D_end = float(sys.argv[2]); h = float(sys.argv[3]); HMAX = float(sys.argv[4])
RUST = {}
def mk(D, Dref):
    S, R = make_rust(D, Dref, Ks=KS, nodes=NODES_L)
    RUST[id(S)] = R
    return S
def JR(S, z, eps=3e-7): return jacobian_rust(S, RUST[id(S)], z, eps)
def lam_g(S, z):
    parts, _ = S.split(z[None, :]); aS = S.phys(parts[-1], "even")[0]
    return float(np.mean(1 - (S.n - 1) * (aS ** 2 - 1)))
def s0_of(S, z): return S.n / (2 * lam_g(S, z)) - 1
def solve(D, guess, maxjac=12):
    S = mk(D, guess[-1])
    try:
        z, nF, _ = lm_svd3(S, guess, maxjac=maxjac, tol=1e-12, jac=JR, res=RUST[id(S)])
    except Exception as e:
        print(f"  solver exception: {e!r}", flush=True)
        return None, None, S
    return z, nF, S
LAST_CHECK = {}
def accept(S, z, nF):
    """|F| <= TOL, or |F| < 1e-9 with the whole leftover residual in numerically-null (junk) directions:
    Gauss-Newton Delta correction from all directions with sigma/smax >= 1e-10 below 1e-9."""
    LAST_CHECK.clear()
    if nF <= TOL: return True
    if nF > 1e-6: return False
    F, J = JR(S, z); U, s, Vt = np.linalg.svd(J); c = U.T @ F; m = s / s[0] >= 1e-10
    dD = float(-(Vt[m, -1] * c[m] / s[m]).sum()); kept = float(np.linalg.norm(c[m]))
    LAST_CHECK.update(dDelta_GN=dD, resid_nonjunk=kept)
    print(f"  floor check: |F|={nF:.1e} non-junk residual {kept:.1e} GN dDelta {dD:+.1e}", flush=True)
    return abs(dD) < 1e-9 and kept < 1e-8
def record(D, z, nF, S, lam_pred, t0):
    F, J = JR(S, z)
    y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-13)[0]; sD = float(np.linalg.norm(J[:, -1] - J[:, :-1] @ y))
    lam = None
    import os
    for w in (() if os.environ.get("NOGAMMA") else (0.01, 0.03, 0.08)):
        try:
            lam, _ = growth_rate(S, z, lam_pred - w, lam_pred + w, xtol=1e-11); break
        except Exception:
            continue
    parts, _ = S.split(z[None, :]); c0 = S.phys(parts[1], "even")[0]
    P, M, A, C = S.ssh_series(S.phys(parts[-2], "odd"), S.phys(parts[-1], "even"), np.array([2 * np.pi / z[-1]]))
    lastterm = float(max(np.abs(Q[-1]).max() for Q in (P, M, A, C)) * S.u0 ** KS)
    rec = dict(engine="rust", D=D, kc=kc, Ks=KS, xK=NODES_L[-1], Delta=float(z[-1]), residual=float(nF), s_Delta=sD, dDelta=float(nF / sD),
               lam=lam, gamma=(1 / lam if lam else None), lam_gauge=lam_g(S, z), s0=s0_of(S, z), c0_min=float(c0.min()),
               tail=float(tail(S, z)), series_last_term=lastterm, seconds=round(time.time() - t0), **LAST_CHECK)
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"longB_D{D:.4f}.npy", z)
    return lam
def extrap(Ds, vals, Dt):
    Ds, vals = Ds[-4:], vals[-4:]; o = 0          # cubic predictor (horizon data is extremely sensitive)
    for i in range(len(Ds)):
        w = 1.0
        for j in range(len(Ds)):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * vals[i]
    return o
hist = []
for item in sys.argv[5].split(";"):
    Dd, f, ks = item.split(":"); Dd = float(Dd); ks = int(ks); t0 = time.time()
    z = np.load(f); Sstd = make(ks, n=Dd - 2)
    if ks != kc: z = make(kc, n=Dd - 2).regrid(z, Sstd); Sstd = make(kc, n=Dd - 2)
    S = mk(Dd, z[-1])
    if len(z) != S.layout()[2]: z = from_standard(Sstd, z, S)
    z, nF, S = solve(Dd, z, maxjac=10)
    lam = record(Dd, z, nF, S, 2.24, t0) or 2.24
    hist.append([Dd, z, lam, s0_of(S, z)])
D = hist[-1][0]
while D < D_end - 1e-9:
    Dt = min(D + h, D_end)
    # skip near-resonant dimensions: predict s0 by extrapolation
    for _ in range(20):
        s0p = extrap([r[0] for r in hist], [r[3] for r in hist], Dt)
        if abs(s0p - round(s0p)) >= RES: break
        Dt += 0.05
    t0 = time.time()
    Ds = [r[0] for r in hist]; zs = [r[1] for r in hist]; ls = [r[2] for r in hist]
    z, nF, S = solve(Dt, extrap(Ds, zs, Dt))
    if z is None or not np.isfinite(nF) or not accept(S, z, nF):
        h /= 2; print(f"step to D={Dt:.4f} failed (|F|={nF}); halving step to {h}", flush=True)
        if h < 0.03: print("giving up", flush=True); break
        continue
    lam = record(Dt, z, nF, S, extrap(Ds, ls, Dt), t0)
    hist.append([Dt, z, lam if lam else extrap(Ds, ls, Dt), s0_of(S, z)]); D = Dt
    h = min(h * 1.3, HMAX)
