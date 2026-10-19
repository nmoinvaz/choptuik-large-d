"""Adaptive continuation in D with the regularised Newton solver; halves the step on failure."""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, tail
from gamma import growth_rate
from newton_tsvd import tikhonov_newton, damp_top

kc = int(sys.argv[1]); out = sys.argv[3]; D_end = float(sys.argv[4]); h = float(sys.argv[5]); HMAX = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
hist = []
for item in sys.argv[2].split(";"):
    f = item.split(":"); hist.append([float(f[0]), np.load(f[1]), float(f[2])])

def attempt(D, guess):
    S = make(kc, n=D - 2)
    try:
        g = damp_top(S, guess)
        if not np.all(np.isfinite(S.residual(g[None, :])[0])):
            return None, None, S
        z, nF, _ = tikhonov_newton(S, g)
        return (z, nF, S) if np.isfinite(nF) and nF < 1e-9 else (None, nF, S)
    except Exception as e:
        return None, None, S

D = hist[-1][0]
while D < D_end - 1e-9:
    Dt = min(D + h, D_end); t0 = time.time()
    (D1, z1, l1), (D2, z2, l2) = hist[-2], hist[-1]
    z, nF, S = attempt(Dt, z2 + (z2 - z1) * (Dt - D2) / (D2 - D1))
    if z is None:
        z, nF, S = attempt(Dt, z2)
    if z is None:
        h /= 2; print(f"step failed at D={Dt}; halving step to {h}", flush=True)
        if h < 0.02: print("giving up", flush=True); break
        continue
    pred = l2 + (l2 - l1) * (Dt - D2) / (D2 - D1)
    lam = None
    for wdt in (0.02, 0.06, 0.15):
        try:
            lam, _ = growth_rate(S, z, pred - wdt, pred + wdt, xtol=1e-10); break
        except Exception:
            continue
    parts, _ = S.split(z[None, :]); aS = S.phys(parts[-1], "even")[0]; c0 = S.phys(parts[1], "even")[0]
    rec = dict(D=Dt, kc=kc, Delta=float(z[-1]), residual=float(nF), lam=lam, gamma=(1 / lam if lam else None),
               lam_gauge=float(np.mean(1 - (S.n - 1) * (aS ** 2 - 1))), c0_min=float(c0.min()), tail=float(tail(S, z)),
               seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"/home/claude/dss/solD_{Dt:.3f}_kc{kc}.npy", z)
    hist.append([Dt, z, lam if lam else pred]); D = Dt
    if lam is None: print("warning: no growth rate found", flush=True)
    h = min(h * 1.25, HMAX)
