"""Combine all dimension-scan data (kc31 precise to D~9, kc23 LM scan beyond, with Delta error bars),
and examine the large-D trends of gamma and Delta."""
import json, numpy as np
rows = {}
for fn, kcpref in (("scan_up_kc31.jsonl", 0), ("scan_lm_kc23.jsonl", 1)):
    for l in open(fn):
        r = json.loads(l)
        if r.get("gamma"): rows[round(r["D"], 4)] = r
rows[4.0] = dict(D=4.0, gamma=0.3739609, Delta=3.4454523972)
D = np.array(sorted(rows)); n = D - 2
g = np.array([rows[d]["gamma"] for d in D]); Dl = np.array([rows[d]["Delta"] for d in D])
dD = np.array([rows[d].get("dDelta", 3e-5 if d > 7.9 else 1e-7) for d in D])
lam = 1 / g; ln = np.log(n)
q = -np.gradient(np.log(Dl), ln); p = -np.gradient(np.log(0.5 - g), ln)
print(f"{'D':>7} {'Delta':>9} {'+-':>7} {'gamma':>9} {'q=-dlnDelta/dln n':>17} {'p(1/2-g)':>8} {'Delta n^1/4':>11} {'(lam-2)sqrt n':>13}")
for i in range(len(D)):
    print(f"{D[i]:7.3f} {Dl[i]:9.5f} {dD[i]:7.1e} {g[i]:9.6f} {q[i]:17.3f} {p[i]:8.3f} {Dl[i]*n[i]**.25:11.4f} {(lam[i]-2)*np.sqrt(n[i]):13.4f}")
