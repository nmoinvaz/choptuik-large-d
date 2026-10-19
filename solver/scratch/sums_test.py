"""'Small sum + large sum'?  Compare model families for Delta(n), n = D-2, by OUT-OF-SAMPLE prediction:
fit on low D, predict the high-D points (which the fit never sees).
  large sums:  A  sum a_k n^-k            B  n^-1/4 sum a_k n^-k/2        C  n^-1/2 sum a_k n^-k/2
  large+small: same plus b e^{-kappa n}  (kappa fitted)"""
import json, numpy as np, warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
rows = {}
for fn in ("scan_down_kc47.jsonl", "scan_up_kc31.jsonl", "scan_lm_kc23.jsonl"):
    for l in open(fn):
        r = json.loads(l); rows[round(r["D"], 4)] = r
rows[4.0] = dict(D=4.0, Delta=3.4454523972)
D = np.array(sorted(d for d in rows if d <= 10.2)); n = D - 2
Y = np.array([rows[d]["Delta"] for d in D]); E = np.array([max(rows[d].get("dDelta", 0), 2e-6) for d in D])
def base(kind, n, a):
    if kind == "A": return sum(c * n ** (-k) for k, c in enumerate(a))
    if kind == "B": return n ** -0.25 * sum(c * n ** (-k / 2) for k, c in enumerate(a))
    if kind == "C": return n ** -0.5 * sum(c * n ** (-k / 2) for k, c in enumerate(a))
def model(kind, small, p, n):
    if not small: return base(kind, n, p)
    return base(kind, n, p[:-2]) + p[-2] * np.exp(-p[-1] * n)
def fit(kind, small, npow, m):
    p0 = np.r_[np.ones(npow), [1.0, 1.0]] if small else np.ones(npow)
    best = None
    for k0 in ((0.5, 1.0, 2.0) if small else (None,)):
        if small: p0[-1] = k0
        r = least_squares(lambda p: (model(kind, small, p, n[m]) - Y[m]) / E[m], p0, max_nfev=20000)
        if best is None or r.cost < best.cost: best = r
    return best.x
for train_max in (8.0, 7.0):
    m = D <= train_max; t = D > train_max
    print(f"\n== fit on D in [{D[m].min()}, {train_max}] ({m.sum()} pts), predict D in ({train_max}, {D.max():.2f}] ({t.sum()} pts) ==")
    for kind in "ABC":
        for small in (False, True):
            for npow in (3, 4, 5):
                if small and npow > 4: continue
                p = fit(kind, small, npow, m)
                fit_err = np.abs(model(kind, small, p, n[m]) - Y[m]).max()
                pred_err = np.abs(model(kind, small, p, n[t]) - Y[t])
                lab = f"{kind}{'+small' if small else '      '} ({npow} powers{', kappa=%.2f' % p[-1] if small else ''})"
                print(f"  {lab:34s} train max err {fit_err:.1e}   prediction err: max {pred_err.max():.1e}, at D={D[t][-1]:.2f}: {pred_err[-1]:.1e}   leading coeff {p[0]:.4f}")
