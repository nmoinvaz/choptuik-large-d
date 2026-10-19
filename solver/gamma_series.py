"""gamma(D) as a power series in 1/n (and 1/sqrt(n)): free limit vs gamma_inf = 1/2, hold-out on the D > 25.5 points."""
import numpy as np, json
from data_ne import load
from consensus import interp
rows = {}
for l in open("gamma_rust.jsonl"):                     # Rust mode matrix on the old solutions (good below 17.2)
    r = json.loads(l)
    if r.get("gamma") and 8 <= r["D"] < 17.2:
        rows[round(r["D"], 4)] = r["gamma"]
for k, r in load().items():                             # NE, x_K = 0.6 kc 23, up to 25.5
    if 17.2 <= k <= 25.5 and r.get("gamma"):
        rows[k] = r["gamma"]
hold = {k: r["gamma"] for k, r in load(xK=0.6, kc=31, smooth=True).items() if k > 25.5 and r.get("gamma")}
D = np.array(sorted(rows)); g = np.array([rows[k] for k in D]); n = D - 2
print(f"{len(D)} fit points D = {D.min():.2f}..{D.max():.2f}; {len(hold)} hold-out points up to {max(hold):.2f}")
r = np.polyfit(1 / n, g, 9); res = np.polyval(r, 1 / n) - g
print("smoothness: rms residual of a degree-9 fit in 1/n:", f"{np.sqrt(np.mean(res**2)):.1e}")
for xname, xf in (("1/n", lambda d: 1 / (d - 2)), ("1/sqrt(n)", lambda d: (d - 2) ** -0.5), ("1/(D-1)", lambda d: 1 / (d - 1))):
    x = xf(D)
    print(f"\nx = {xname}")
    for L in (None, 0.5):
        for m in (6, 7, 8, 9):
            A = np.vander(x, m + 1, increasing=True)
            if L is None:
                c, *_ = np.linalg.lstsq(A, g, rcond=None)
            else:
                c, *_ = np.linalg.lstsq(A[:, 1:], g - L, rcond=None); c = np.r_[L, c]
            fitres = np.sqrt(np.mean((A @ c - g) ** 2))
            ho = [np.polyval(c[::-1], xf(k)) - v for k, v in sorted(hold.items())]
            print(f"  {'free' if L is None else 'g0=1/2':6s} m={m}: g0 {c[0]:.5f} c1 {c[1]:+.4f}  fit rms {fitres:.0e}  hold-out max {np.abs(ho).max():.1e}")
