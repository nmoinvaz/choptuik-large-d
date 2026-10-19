import json, numpy as np, warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
rows = {}
for fn in ("scan_up_kc31.jsonl", "scan_lm_kc23.jsonl"):
    for l in open(fn):
        r = json.loads(l)
        if r.get("gamma"): rows[round(r["D"], 4)] = r
rows[4.0] = dict(D=4.0, gamma=0.3739609); rows[5.0] = dict(D=5.0, gamma=0.4131269359)
D = np.array(sorted(rows)); n = D - 2; G = np.array([rows[d]["gamma"] for d in D])
m = D <= 10.2
fams = {
 "1/2 - n^-1/2 (c0 + c1 n^-1/2 + c2/n + c3 n^-3/2)": (lambda p, n: 0.5 - n ** -0.5 * (p[0] + p[1] * n ** -0.5 + p[2] / n + p[3] * n ** -1.5), 4),
 "1/2 - n^-1 (c1 + c2 n^-1/2 + c3/n + c4 n^-3/2)   [c0=0]": (lambda p, n: 0.5 - n ** -1 * (p[0] + p[1] * n ** -0.5 + p[2] / n + p[3] * n ** -1.5), 4),
 "1/2 - (c1/n + c2/n^2 + c3/n^3 + c4/n^4)": (lambda p, n: 0.5 - (p[0] / n + p[1] / n ** 2 + p[2] / n ** 3 + p[3] / n ** 4), 4),
}
for name, (h, k) in fams.items():
    for tmax in (8.0, 7.0, 6.0):
        tr = m & (D <= tmax); te = m & (D > tmax)
        r = least_squares(lambda p: h(p, n[tr]) - G[tr], np.zeros(k) + 0.1)
        print(f"{name:58s} train D<={tmax}: predict D<=10.1 max err {np.abs(h(r.x, n[te]) - G[te]).max():.1e}  coeffs {np.round(r.x, 4)}")
    print()
h = fams["1/2 - n^-1/2 (c0 + c1 n^-1/2 + c2/n + c3 n^-3/2)"][0]
r = least_squares(lambda p: h(p, n[m]) - G[m], np.zeros(4) + 0.1)
print("all D<=10.1 fit:", np.round(r.x, 5), " max err", f"{np.abs(h(r.x, n[m]) - G[m]).max():.1e}")
for d in D[D > 10.2]:
    print(f"  scan point D={d:.3f}: gamma={rows[d]['gamma']:.6f}, formula {h(r.x, d - 2):.6f}, diff {rows[d]['gamma'] - h(r.x, d - 2):+.1e}")
for d in (12, 16, 20, 30, 50, 100):
    print(f"  formula prediction D={d}: gamma = {h(r.x, d - 2):.4f}")
