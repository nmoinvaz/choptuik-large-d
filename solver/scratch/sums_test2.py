import json, numpy as np, warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
rows = {}
for fn in ("scan_down_kc47.jsonl", "scan_up_kc31.jsonl", "scan_lm_kc23.jsonl"):
    for l in open(fn):
        r = json.loads(l); rows[round(r["D"], 4)] = r
rows[4.0] = dict(D=4.0, Delta=3.4454523972, gamma=0.3739609)
rows[5.0] = dict(D=5.0, Delta=3.2217019415, gamma=0.4131269359)
D = np.array(sorted(d for d in rows if d <= 10.2)); n = D - 2
Y = np.array([rows[d]["Delta"] for d in D]); E = np.array([max(rows[d].get("dDelta", 0), 2e-6) for d in D])
G = np.array([rows[d].get("gamma") or np.nan for d in D])
EG = np.maximum(0.16 * E, 1e-6)
# 1) free leading exponent alpha in  Delta = n^-alpha (a0 + a1 n^-1/2 + a2 n^-1 + a3 n^-3/2)
f = lambda p, n: n ** (-p[0]) * (p[1] + p[2] * n ** -0.5 + p[3] / n + p[4] * n ** -1.5)
for lo, hi in ((3.7, 10.2), (4.5, 10.2), (3.7, 8.0), (5.0, 10.2)):
    m = (D >= lo) & (D <= hi)
    r = least_squares(lambda p: (f(p, n[m]) - Y[m]) / E[m], [0.25, 4.7, 0, 0, 0])
    print(f"Delta, D in [{lo},{hi}]: alpha = {r.x[0]:.4f}, a0 = {r.x[1]:.3f}, a1..a3 = {np.round(r.x[2:], 3)}, max err {np.abs(f(r.x, n[m]) - Y[m]).max():.1e}")
# 2) fixed alpha = 1/4, 4 powers, on several ranges -> coefficient stability
g4 = lambda p, n: n ** -0.25 * (p[0] + p[1] * n ** -0.5 + p[2] / n + p[3] * n ** -1.5)
print()
for lo, hi in ((3.7, 10.2), (4.5, 10.2), (3.7, 8.0), (3.7, 7.0), (5.0, 10.2)):
    m = (D >= lo) & (D <= hi)
    r = least_squares(lambda p: (g4(p, n[m]) - Y[m]) / E[m], [4.7, 0, 0, 0])
    print(f"Delta = n^-1/4 (a0 + a1/sqrt n + a2/n + a3/n^1.5), D in [{lo},{hi}]: a = {np.round(r.x, 4)}  max err {np.abs(g4(r.x, n[m]) - Y[m]).max():.1e}")
# 3) gamma: out-of-sample comparison, 1/2 - n^-beta * series
print()
mG = np.isfinite(G)
for beta_kind in ("1/n", "1/sqrt n", "free"):
    for train_max in (8.0, 7.0):
        m = mG & (D <= train_max); t = mG & (D > train_max)
        if beta_kind == "1/n": h = lambda p, n: 0.5 - (p[0] / n + p[1] / n ** 2 + p[2] / n ** 3); p0 = [0.3, 0, 0]
        elif beta_kind == "1/sqrt n": h = lambda p, n: 0.5 - n ** -0.5 * (p[0] + p[1] * n ** -0.5 + p[2] / n + p[3] * n ** -1.5); p0 = [0.1, 0, 0, 0]
        else: h = lambda p, n: p[4] - n ** (-p[0]) * (p[1] + p[2] * n ** -0.5 + p[3] / n); p0 = [0.5, 0.1, 0, 0, 0.5]
        r = least_squares(lambda p: (h(p, n[m]) - G[m]) / EG[m], p0, max_nfev=20000)
        pe = np.abs(h(r.x, n[t]) - G[t])
        print(f"gamma [{beta_kind:8s}] train D<={train_max}: train err {np.abs(h(r.x, n[m]) - G[m]).max():.1e}, predict D<=10.1: max err {pe.max():.1e}; params {np.round(r.x, 4)}")
