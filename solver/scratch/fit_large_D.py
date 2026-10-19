"""Large-D behaviour of gamma(D) and Delta(D) from the scan."""
import json, glob, numpy as np
from scipy.optimize import curve_fit
rows = {}
for fn in glob.glob("/home/claude/dss/scan_up_kc31.jsonl"):
    for l in open(fn):
        r = json.loads(l); rows[round(r["D"], 4)] = r
D = np.array(sorted(rows)); g = np.array([rows[d]["gamma"] for d in D]); Dl = np.array([rows[d]["Delta"] for d in D])
lam = 1 / g
print("points:", len(D), "D range", D.min(), D.max())
out = {}
for Dmin in (6.0, 8.0, 10.0):
    m = D >= Dmin
    if m.sum() < 5: continue
    x = 1 / D[m]
    # gamma: free limit, quadratic and cubic in 1/D
    for deg in (2, 3):
        c, cov = np.polyfit(x, g[m], deg, cov=True)
        print(f"gamma, D>={Dmin}, poly deg {deg} in 1/D: gamma_inf = {c[-1]:.5f} +- {np.sqrt(cov[-1,-1]):.5f}; coeffs (high->low) {np.round(c, 4)}")
        out[f"g_D{Dmin}_deg{deg}"] = [float(c[-1]), float(np.sqrt(cov[-1, -1]))]
    # lambda = 1/gamma
    c, cov = np.polyfit(x, lam[m], 3, cov=True)
    print(f"lambda0, D>={Dmin}, cubic in 1/D: lambda_inf = {c[-1]:.5f} +- {np.sqrt(cov[-1,-1]):.5f}")
    # Bland et al. form A - B exp(-C D)
    try:
        p, pc = curve_fit(lambda d, A, B, C: A - B * np.exp(-C * d), D[m], g[m], p0=[0.47, 0.5, 0.4], maxfev=20000)
        res = g[m] - (p[0] - p[1] * np.exp(-p[2] * D[m]))
        print(f"   exponential form on D>={Dmin}: A = {p[0]:.5f}, B = {p[1]:.4f}, C = {p[2]:.4f}; max residual {np.abs(res).max():.1e}")
    except Exception as e:
        print("   exponential fit failed", e)
    # Delta: power law at large D
    s, i = np.polyfit(np.log(D[m]), np.log(Dl[m]), 1)
    print(f"Delta, D>={Dmin}: local log-log slope {s:.3f};  D*Delta at largest D = {D[m][-1]*Dl[m][-1]:.3f}; sqrt(D)*Delta = {np.sqrt(D[m][-1])*Dl[m][-1]:.3f}")
# local slopes of Delta
ls = np.gradient(np.log(Dl), np.log(D))
print("local d ln(Delta) / d ln(D):", list(zip(D.round(2), ls.round(3))))
json.dump(out, open("/home/claude/dss/fits.json", "w"), indent=1)
