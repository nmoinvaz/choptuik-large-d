"""Compact rational (Pade-type) formulas in x = 1/(D-2) for Delta and gamma, fitted on D = 8..25.5 and checked on the
held-out D > 25.5 points. f(x) = (p0 + p1 x + ... + pL x^L) / (1 + q1 x + ... + qM x^M)."""
import numpy as np, json
from scipy.optimize import least_squares
from series_factory import best
from data_ne import load

def rat(p, x, L, M):
    num = np.polyval(p[:L + 1][::-1], x); den = 1 + x * np.polyval(np.r_[p[L + 1:][::-1], 0][:-1], x) if M else 1
    if M:
        den = 1 + sum(p[L + 1 + j] * x ** (j + 1) for j in range(M))
    return num / den

def fit(x, y, w, L, M, p0=None):
    p0 = np.r_[y[-1], np.zeros(L), np.zeros(M)] if p0 is None else p0
    best_ = None
    for trial in range(30):
        start = p0 + (np.random.default_rng(trial).normal(0, 1, len(p0)) * (0 if trial == 0 else 5))
        r = least_squares(lambda p: (rat(p, x, L, M) - y) * w, start, max_nfev=20000, x_scale="jac")
        if best_ is None or r.cost < best_.cost:
            best_ = r
    return best_.x

D, y, s = best(8.0, 25.5); x = 1 / (D - 2); w = 1 / s
hold = {k: r["Delta"] for k, r in load(xK=0.6, kc=31, smooth=True).items() if k > 25.5}
Dh = np.array(sorted(hold)); yh = np.array([hold[k] for k in Dh])
print("Delta:")
for L, M in ((2, 1), (2, 2), (3, 2), (3, 3), (4, 3)):
    p = fit(x, y, w, L, M)
    fr = rat(p, x, L, M) - y; ho = rat(p, 1 / (Dh - 2), L, M) - yh
    lim = p[0]
    print(f"  [{L}/{M}] Delta_inf = {lim:.6f}  fit rms {np.sqrt(np.mean(fr**2)):.0e} max {np.abs(fr).max():.0e}  hold-out max {np.abs(ho).max():.1e}  p = " +
          " ".join(f"{v:+.5g}" for v in p))
# gamma
rows = {}
for l in open("gamma_rust.jsonl"):
    r = json.loads(l)
    if r.get("gamma") and 8 <= r["D"] < 17.2: rows[round(r["D"], 4)] = r["gamma"]
for k, r in load().items():
    if 17.2 <= k <= 25.5 and r.get("gamma"): rows[k] = r["gamma"]
gh = {k: r["gamma"] for k, r in load(xK=0.6, kc=31, smooth=True).items() if k > 25.5 and r.get("gamma")}
Dg = np.array(sorted(rows)); g = np.array([rows[k] for k in Dg]); xg = 1 / (Dg - 2)
Dgh = np.array(sorted(gh)); ggh = np.array([gh[k] for k in Dgh])
print("gamma:")
for L, M in ((2, 1), (2, 2), (3, 2), (3, 3), (4, 3)):
    p = fit(xg, g, np.full(len(g), 1e8), L, M)
    fr = rat(p, xg, L, M) - g; ho = rat(p, 1 / (Dgh - 2), L, M) - ggh
    print(f"  [{L}/{M}] gamma_inf = {p[0]:.6f}  fit rms {np.sqrt(np.mean(fr**2)):.0e} max {np.abs(fr).max():.0e}  hold-out max {np.abs(ho).max():.1e}  p = " +
          " ".join(f"{v:+.5g}" for v in p))
