"""Is gamma (or lambda0 = 1/gamma, or Delta) a sum of powers of 1/n, n = D-2?  Fit truncated series and check
whether the leading coefficients are stable when the truncation order and the D-range change."""
import json, glob, numpy as np
rows = {}
for fn in ["scan_up_kc31.jsonl", "scan_up_kc23.jsonl"]:
    try:
        for l in open(fn):
            r = json.loads(l)
            if r.get("gamma"): rows.setdefault(round(r["D"], 4), r)
    except FileNotFoundError: pass
rows[4.0] = dict(D=4.0, gamma=0.3739609, Delta=3.4454523972)
D = np.array(sorted(rows)); n = D - 2
g = np.array([rows[d]["gamma"] for d in D]); Dl = np.array([rows[d]["Delta"] for d in D]); lam = 1 / g
print(f"{len(D)} points, D = {D.min()} .. {D.max()}")
for name, y in (("gamma", g), ("lambda0=1/gamma", lam), ("Delta", Dl)):
    print(f"\n{name}: series sum_k a_k / n^k")
    for nmin in (2.0, 3.0, 4.0):
        m = n >= nmin
        for K in (3, 4, 5):
            if m.sum() <= K + 2: continue
            c = np.polyfit(1 / n[m], y[m], K)[::-1]
            r = y[m] - np.polyval(c[::-1], 1 / n[m])
            print(f"   n>={nmin} K={K}: a0={c[0]:+.4f} a1={c[1]:+.4f} a2={c[2]:+.4f} a3={c[3]:+.4f}   max resid {np.abs(r).max():.1e}")
