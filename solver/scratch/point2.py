import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, tail
from gamma import growth_rate
from newton_tsvd import tsvd_newton
D = float(sys.argv[1]); f = sys.argv[2]; kc = int(sys.argv[3]); lam_guess = float(sys.argv[4]); out = sys.argv[5]
S = make(kc, n=D - 2); z = np.load(f); t0 = time.time()
z, nF, _ = tsvd_newton(S, z)
parts, _ = S.split(z[None, :]); aS = S.phys(parts[-1], "even")[0]
lam = None
for wdt in (0.01, 0.04, 0.12):
    try:
        lam, nev = growth_rate(S, z, lam_guess - wdt, lam_guess + wdt, xtol=1e-11); break
    except AssertionError:
        continue
rec = dict(D=D, kc=kc, Delta=float(z[-1]), residual=float(nF), lam=lam, gamma=(1 / lam if lam else None),
           lam_gauge=float(np.mean(1 - (S.n - 1) * (aS ** 2 - 1))), tail=float(tail(S, z)), seconds=round(time.time() - t0))
print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
np.save(f"/home/claude/dss/solD_{D:.3f}_kc{kc}.npy", z)
