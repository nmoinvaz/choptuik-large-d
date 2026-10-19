"""Same test, but in powers of 1/sqrt(n): the fast-time scaling tau~ = sqrt(n) tau suggests Delta ~ Delta~/sqrt(n)."""
import json, numpy as np
rows = {}
for fn in ["scan_up_kc23.jsonl", "scan_up_kc31.jsonl"]:
    for l in open(fn):
        r = json.loads(l)
        if r.get("gamma"): rows[round(r["D"], 4)] = r
rows[4.0] = dict(D=4.0, gamma=0.3739609, Delta=3.4454523972)
rows[5.0] = dict(D=5.0, gamma=0.4131269359, Delta=3.2217019415)
D = np.array(sorted(rows)); n = D - 2; e = 1 / np.sqrt(n)
g = np.array([rows[d]["gamma"] for d in D]); Dl = np.array([rows[d]["Delta"] for d in D])
print("Delta*sqrt(n):", dict(zip(D[::3].round(2), (Dl * np.sqrt(n))[::3].round(4))))
print("Delta*n      :", dict(zip(D[::3].round(2), (Dl * n)[::3].round(4))))
def table(name, y, var):
    print(f"\n{name}")
    for nmin in (2.0, 3.0, 4.0):
        m = n >= nmin
        for K in (3, 4, 5):
            if m.sum() <= K + 3: continue
            c = np.polyfit(var[m], y[m], K)[::-1]
            r = y[m] - np.polyval(c[::-1], var[m])
            print(f"   n>={nmin} K={K}: " + " ".join(f"a{k}={c[k]:+.4f}" for k in range(4)) + f"   max resid {np.abs(r).max():.1e}")
table("gamma = sum a_k n^(-k/2)", g, e)
table("Delta*sqrt(n) = sum a_k n^(-k/2)", Dl * np.sqrt(n), e)
table("Delta*n = sum a_k n^(-k/2)", Dl * n, e)
