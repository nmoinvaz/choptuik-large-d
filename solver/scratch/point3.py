"""One dimension at a given resolution (regularised Newton), regridded from another resolution, plus gamma."""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, tail
from gamma import growth_rate
from newton_tsvd import tikhonov_newton
D = float(sys.argv[1]); kc_from = int(sys.argv[2]); f = sys.argv[3]; kc = int(sys.argv[4]); lam_guess = float(sys.argv[5]); out = sys.argv[6]
Sf = make(kc_from, n=D - 2); S = make(kc, n=D - 2)
z = S.regrid(np.load(f), Sf); t0 = time.time()
z, nF, _ = tikhonov_newton(S, z)
lam = None
for wdt in (0.003, 0.02, 0.1):
    try:
        lam, nev = growth_rate(S, z, lam_guess - wdt, lam_guess + wdt, xtol=1e-11); break
    except AssertionError:
        continue
rec = dict(D=D, kc=kc, Delta=float(z[-1]), residual=float(nF), lam=lam, gamma=(1 / lam if lam else None), tail=float(tail(S, z)), seconds=round(time.time() - t0))
print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
np.save(f"/home/claude/dss/solD_{D:.3f}_kc{kc}.npy", z)
