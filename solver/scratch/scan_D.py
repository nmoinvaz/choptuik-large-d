"""Continuation of the critical solution and its growing mode in the spacetime dimension D = n + 2."""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, solve, tail
from gamma import growth_rate
from newton_tsvd import tsvd_newton, damp_top, tikhonov_newton

kc = int(sys.argv[1]); start = sys.argv[2]; Ds = [float(a) for a in sys.argv[3].split(",")]
out = sys.argv[4]
hist = []
for item in start.split(";"):                 # "D:file[:lam]" entries, oldest first
    f = item.split(":")
    hist.append((float(f[0]), np.load(f[1])) + ((float(f[2]),) if len(f) > 2 else ()))
lam_prev = float(sys.argv[5]) if len(sys.argv) > 5 else 2.6741
for D in Ds:
    S = make(kc, n=D - 2)
    # secant prediction of the starting guess from the last two solutions
    if len(hist) >= 2:
        D1, z1 = hist[-2][0], hist[-2][1]; D2, z2 = hist[-1][0], hist[-1][1]
        zg = z2 + (z2 - z1) * (D - D2) / (D2 - D1)
    else:
        zg = hist[-1][1]
    t0 = time.time()
    z, nF, w = tikhonov_newton(S, damp_top(S, zg))
    if not np.isfinite(nF) or nF > 1e-9:
        z, nF, w = tikhonov_newton(S, damp_top(S, hist[-1][1]), maxit=40)
    parts, _ = S.split(z[None, :])
    aS = S.phys(parts[-1], "even")[0]
    lam_gauge = float(np.mean(1 - (S.n - 1) * (aS ** 2 - 1)))
    # predicted growth rate from the last two points
    lams = [h[2] for h in hist if len(h) > 2]
    pred = lams[-1] + (lams[-1] - lams[-2]) * (D - hist[-1][0]) / (hist[-1][0] - hist[-2][0]) if len(lams) >= 2 else lam_prev
    lam = None
    for wdt in (0.03, 0.08, 0.2, 0.5):
        a, b = pred - wdt, pred + wdt
        try:
            lam, nev = growth_rate(S, z, a, b, xtol=1e-10)
            break
        except AssertionError:
            continue
    rec = dict(D=D, kc=kc, Delta=float(z[-1]), residual=float(nF), lam=lam, gamma=(1 / lam if lam else None),
               lam_gauge=lam_gauge, tail=float(tail(S, z)), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    np.save(f"/home/claude/dss/solD_{D:.3f}_kc{kc}.npy", z)
    hist.append((D, z.copy(), lam) if lam else (D, z.copy()))
