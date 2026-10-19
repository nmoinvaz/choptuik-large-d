"""Continuation in D with the Levenberg-Marquardt solver (central-difference Jacobian, frozen RK4 step counts),
a quadratic predictor, and an error bar on Delta: dDelta ~ |F| / s_Delta (Schur sensitivity).
usage: scan_D4.py kc "D:file:lam;..." out D_end h HMAX"""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, tail
from gamma import growth_rate
from spec_c import lm_solve, freeze_steps, jacobian_c
kc = int(sys.argv[1]); out = sys.argv[3]; D_end = float(sys.argv[4]); h = float(sys.argv[5]); HMAX = float(sys.argv[6])
TOL = 2e-9

def attempt(D, guess):
    S = freeze_steps(make(kc, n=D - 2), guess[-1])
    try:
        if not np.all(np.isfinite(S.residual(guess[None, :])[0])): return None, None, S
        z, nF, _ = lm_solve(S, guess, maxjac=10)
        return (z, nF, S) if np.isfinite(nF) and nF < TOL else (None, nF, S)
    except Exception:
        return None, None, S

def extrap(Ds, vals, Dt):
    Ds, vals = Ds[-3:], vals[-3:]; out = 0
    for i in range(len(Ds)):
        w = 1.0
        for j in range(len(Ds)):
            if j != i: w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        out = out + w * vals[i]
    return out

hist = []
for item in sys.argv[2].split(";"):
    f = item.split(":"); hist.append([float(f[0]), np.load(f[1]), float(f[2])])
D = hist[-1][0]
while D < D_end - 1e-9:
    Dt = min(D + h, D_end); t0 = time.time()
    Ds = [r[0] for r in hist]; zs = [r[1] for r in hist]; ls = [r[2] for r in hist]
    z, nF, S = attempt(Dt, extrap(Ds, zs, Dt))
    if z is None:
        z, nF, S = attempt(Dt, extrap(Ds[-2:], zs[-2:], Dt))
    if z is None:
        h /= 2; print(f"step failed at D={Dt} (|F|={nF}); halving step to {h}", flush=True)
        if h < 0.02: print("giving up", flush=True); break
        continue
    F, J = jacobian_c(S, z, 3e-7)
    y = np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=1e-13)[0]; sD = float(np.linalg.norm(J[:, -1] - J[:, :-1] @ y))
    pred = extrap(Ds, ls, Dt); lam = None
    for wdt in (0.01, 0.03, 0.08):
        try:
            lam, _ = growth_rate(S, z, pred - wdt, pred + wdt, xtol=1e-10); break
        except Exception:
            continue
    parts, _ = S.split(z[None, :]); aS = S.phys(parts[-1], "even")[0]; c0 = S.phys(parts[1], "even")[0]
    rec = dict(D=Dt, kc=kc, Delta=float(z[-1]), residual=float(nF), s_Delta=sD, dDelta=float(nF / sD), lam=lam,
               gamma=(1 / lam if lam else None), lam_gauge=float(np.mean(1 - (S.n - 1) * (aS ** 2 - 1))),
               c0_min=float(c0.min()), tail=float(tail(S, z)), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"/home/claude/dss/solD_{Dt:.3f}_kc{kc}.npy", z)
    hist.append([Dt, z, lam if lam else pred]); D = Dt
    h = min(h * 1.3, HMAX)
