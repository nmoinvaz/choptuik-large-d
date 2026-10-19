"""Refit the 1/(D-2) series on step-converged data: RK x2 points (and Richardson RK2 + (RK2 - RK1)/15 where the RK x1
value at the same D exists). D <= 17.3: x_K 0.6 plain NE (junk negligible); 17.3 < D <= Dmax: junk-solved RK x2."""
import json, glob, sys, numpy as np
from data_best import load as old_load
Dmax = float(sys.argv[1]) if len(sys.argv) > 1 else 25.5
o = old_load()
rk1 = {k: r["Delta"] for k, r in o.items() if k < 17.2}          # old values are biased above 17.2
for f in sorted(glob.glob("scan_ne6*.jsonl")):                    # NE RK x1, x_K 0.6, kc 23: junk-solved wins
    for l in open(f):
        r = json.loads(l)
        if r.get("ok") and r.get("rk", 1) == 1 and abs(r.get("xK", 0.6) - 0.6) < 1e-9 and r.get("kc", 23) == 23 and not r.get("smooth"):
            k = round(r["D"], 4)
            if r.get("js"):
                rk1[k] = r["Delta"]
            elif r["D"] <= 23.4 and k >= 17.2:
                rk1.setdefault(k, r["Delta"])
rk2 = {}
for f in glob.glob("scan_ne6_rk2*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("ok"):
            rk2[round(r["D"], 4)] = r["Delta"]
D = np.array(sorted(k for k in rk2 if 8 <= k <= Dmax))
y2 = np.array([rk2[k] for k in D])
yR = np.array([rk2[k] + (rk2[k] - rk1[k]) / 15 if k in rk1 else rk2[k] for k in D])
print(f"{len(D)} RK2 points D = {D.min():.2f}..{D.max():.2f}; Richardson available at {sum(k in rk1 for k in D)}")
for name, y in (("RK x2", y2), ("Richardson", yR)):
    for xn, x in (("1/n", 1 / (D - 2)), ("1/(D-1)", 1 / (D - 1))):
        out = []
        for m in (7, 8, 9, 10):
            A = np.vander(x / x.max(), m + 1, increasing=True)      # scaled: an unscaled Vandermonde loses ~1e-9
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            out.append(f"m={m}: c0 {c[0]:.6f} rms {np.sqrt(np.mean((A @ c - y) ** 2)):.0e}")
        print(f"  {name:10s} {xn:8s} " + " | ".join(out))
