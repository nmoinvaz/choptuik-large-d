"""Continuation downward in D (Delta only) to locate the maximum of Delta(D)."""
import sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from run_spec import make, solve, tail
kc = int(sys.argv[1]); Ds = [float(a) for a in sys.argv[2].split(",")]; out = sys.argv[3]
hist = [(4.0, np.load(f"/home/claude/dss/spec_kc{kc}.npy"))]
for D in Ds:
    S = make(kc, n=D - 2)
    zg = hist[-1][1] if len(hist) < 2 else hist[-1][1] + (hist[-1][1] - hist[-2][1]) * (D - hist[-1][0]) / (hist[-1][0] - hist[-2][0])
    t0 = time.time(); z, nF, w = solve(S, zg, maxfev=30)
    rec = dict(D=D, kc=kc, Delta=float(z[-1]), residual=float(nF), tail=float(tail(S, z)), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"/home/claude/dss/solD_{D:.3f}_kc{kc}.npy", z); hist.append((D, z))
