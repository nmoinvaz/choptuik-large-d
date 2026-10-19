"""Faster continuation in D: quadratic (3-point) predictor for the solution and the growth rate, larger steps,
optional regrid of the starting solutions from another truncation kc.
usage: scan_D3.py kc "D:file:lam:kcsrc;..." out D_end h HMAX"""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, tail
from gamma import growth_rate
from newton_tsvd import tikhonov_newton, damp_top

kc = int(sys.argv[1]); out = sys.argv[3]; D_end = float(sys.argv[4]); h = float(sys.argv[5]); HMAX = float(sys.argv[6]); MU = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-8; TOL = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-8

def attempt(D, guess):
    S = make(kc, n=D - 2)
    try:
        g = damp_top(S, guess)
        if not np.all(np.isfinite(S.residual(g[None, :])[0])):
            return None, None, S
        z, nF, _ = tikhonov_newton(S, g, mu_rel=MU, maxit=18, return_best=True)
        return (z, nF, S) if np.isfinite(nF) and nF < TOL else (None, nF, S)
    except Exception:
        return None, None, S

def record(D, z, nF, S, pred, t0):
    lam = None
    for wdt in (0.01, 0.03, 0.08, 0.2):
        try:
            lam, _ = growth_rate(S, z, pred - wdt, pred + wdt, xtol=1e-10); break
        except Exception:
            continue
    parts, _ = S.split(z[None, :]); aS = S.phys(parts[-1], "even")[0]; c0 = S.phys(parts[1], "even")[0]
    rec = dict(D=D, kc=kc, Delta=float(z[-1]), residual=float(nF), lam=lam, gamma=(1 / lam if lam else None),
               lam_gauge=float(np.mean(1 - (S.n - 1) * (aS ** 2 - 1))), c0_min=float(c0.min()), tail=float(tail(S, z)),
               seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"/home/claude/dss/solD_{D:.3f}_kc{kc}.npy", z)
    return lam

hist = []
for item in sys.argv[2].split(";"):
    f = item.split(":"); D0, z0, l0, ks = float(f[0]), np.load(f[1]), float(f[2]), int(f[3])
    if ks != kc:
        t0 = time.time()
        z0 = make(kc, n=D0 - 2).regrid(z0, make(ks, n=D0 - 2))
        z, nF, S = attempt(D0, z0)
        if z is None: sys.exit(f"could not re-solve D={D0} at kc={kc}")
        l0 = record(D0, z, nF, S, l0, t0) or l0; z0 = z
    hist.append([D0, z0, l0])

def extrap(Ds, vals, Dt):
    """Lagrange extrapolation through the last (up to) three points."""
    Ds, vals = Ds[-3:], vals[-3:]
    out = 0
    for i in range(len(Ds)):
        w = 1.0
        for j in range(len(Ds)):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        out = out + w * vals[i]
    return out

D = hist[-1][0]
while D < D_end - 1e-9:
    Dt = min(D + h, D_end); t0 = time.time()
    Ds = [r[0] for r in hist]; zs = [r[1] for r in hist]; ls = [r[2] for r in hist]
    z, nF, S = attempt(Dt, extrap(Ds, zs, Dt))
    if z is None and len(hist) >= 2:
        z, nF, S = attempt(Dt, extrap(Ds[-2:], zs[-2:], Dt))
    if z is None:
        h /= 2; print(f"step failed at D={Dt}; halving step to {h}", flush=True)
        if h < 0.02: print("giving up", flush=True); break
        continue
    pred = extrap(Ds, ls, Dt)
    lam = record(Dt, z, nF, S, pred, t0)
    if lam is None: print("warning: no growth rate found", flush=True)
    hist.append([Dt, z, lam if lam else pred]); D = Dt
    h = min(h * 1.3, HMAX)
