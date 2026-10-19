"""Delta(D) asymptotic formula: weighted fits in powers of n^-1/2 (n = D-2), hold-out validation, Delta_inf estimate."""
import numpy as np, itertools
from data_best import load
rows = load(); D = np.array(sorted(d for d in rows if d >= 6)); y = np.array([rows[d]["Delta"] for d in D]); n = D - 2
sig = np.minimum(2e-6, np.maximum(1e-8, 5e-8 * np.exp((D - 22.5) * 0.5)))
def fit(ps, m):
    A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1) / sig[m, None]
    c, *_ = np.linalg.lstsq(A, y[m] / sig[m], rcond=None); return c
def pred(c, ps, mm): return np.stack([np.ones(mm.sum())] + [n[mm] ** -p for p in ps], 1) @ c
families = {}
for k in range(2, 7):
    families[f"half-powers 1..{(k+1)/2:g} (no n^-1/2)"] = tuple(j / 2 for j in range(2, k + 2))
    families[f"half-powers 1/2..{k/2:g}"] = tuple(j / 2 for j in range(1, k + 1))
    families[f"integer powers 1..{k}"] = tuple(range(1, k + 1))
print(f"{len(D)} points, D={D.min()}..{D.max():.2f}")
print("hold-out: fit 8<=D<=Dcut, chi = max |pred-true|/sigma over D>Dcut")
res = []
for nm, ps in families.items():
    out = []
    for Dcut in (20, 24, 27):
        m = (D >= 8) & (D <= Dcut); t = D > Dcut
        if m.sum() <= len(ps) + 2: out.append((np.nan, np.nan)); continue
        c = fit(ps, m); e = np.abs(pred(c, ps, t) - y[t])
        out.append((c[0], e.max()))
    res.append((nm, out))
for nm, out in sorted(res, key=lambda r: r[1][1][1]):
    print(f"  {nm:34s} " + " | ".join(f"cut{c}: L={o[0]:.4f} err={o[1]:.1e}" for c, o in zip((20, 24, 27), out)))
