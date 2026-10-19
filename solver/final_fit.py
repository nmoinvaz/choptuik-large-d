"""Delta_inf from all step-converged data: x_K 0.6 RK x2 (Richardson with RK x1 where available, junk-solved above
17.2) up to D = 30.1, plus x_K 0.7 RK x4 junk-solved above that. Weighted fits in x / max(x) for x = 1/D, 1/(D-1),
1/(D-2), several degrees and lower cuts; candidate limits forced for comparison."""
import json, glob, sys, numpy as np
from data_best import load as old_load
o = old_load()
rk1 = {k: r["Delta"] for k, r in o.items() if k < 17.2}
rk2, hi = {}, {}
for f in sorted(glob.glob("scan_ne*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if not r.get("ok") or r.get("kc", 23) != 23 or r.get("smooth"):
            continue
        k = round(r["D"], 4); xk = r.get("xK", 0.6); rk = r.get("rk", 1)
        if abs(xk - 0.6) < 1e-9:
            if rk == 2:
                rk2[k] = r["Delta"]
            elif rk == 1 and (r.get("js") or (17.2 <= k <= 23.4)):
                if r.get("js") or k not in rk1:
                    rk1[k] = r["Delta"]
        elif abs(xk - 0.7) < 1e-9 and rk >= 4 and r.get("js") and k > 30.2:
            hi[k] = r["Delta"]
D, y, s = [], [], []
for k in sorted(rk2):
    if k < 8: continue
    v = rk2[k] + ((rk2[k] - rk1[k]) / 15 if k in rk1 else 0.0)
    D.append(k); y.append(v); s.append(3e-11 if k < 22 else 5e-10 if k < 26 else 1e-9 if k < 29.5 else 3e-8)
for k in sorted(hi):
    D.append(k); y.append(hi[k]); s.append(1e-7)
D, y, s = map(np.array, (D, y, s))
print(f"{len(D)} points: x_K 0.6 RK2 {sum(D <= 30.2)} (D {D.min():.2f}..{D[D <= 30.2].max():.2f}), x_K 0.7 RK4 {len(hi)}" + (f" (to {max(hi):.2f})" if hi else ""))
cands = {"pi^2/6": np.pi ** 2 / 6, "1.6438": 1.6438, "1.6440": 1.6440, "1.6442": 1.6442}
for lo in (8, 10, 12):
    m_ = D >= lo
    for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2))):
        x = xf(D[m_]); sc = x.max(); w = 1 / s[m_]
        row = []
        for m in (8, 9, 10, 11):
            A = np.vander(x / sc, m + 1, increasing=True)
            c, *_ = np.linalg.lstsq(A * w[:, None], y[m_] * w, rcond=None)
            chi = np.sum(((A @ c - y[m_]) * w) ** 2) / (m_.sum() - m - 1)
            row.append(f"{c[0]:.6f}({chi:.1f})")
        m = 10; A = np.vander(x / sc, m + 1, increasing=True)
        cc = []
        for name, L in cands.items():
            c, *_ = np.linalg.lstsq(A[:, 1:] * w[:, None], (y[m_] - L) * w, rcond=None)
            cc.append(f"{name}:{np.sum(((A[:, 1:] @ c - (y[m_] - L)) * w) ** 2) / (m_.sum() - m):.1f}")
        print(f"D>={lo:2d} {xn:8s} free c0(chi2/dof) m=8..11: " + " ".join(row) + "  | forced m=10 chi2/dof: " + " ".join(cc))
