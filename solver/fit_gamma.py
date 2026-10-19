import numpy as np
from data_best import load
rows = load()
D = np.array(sorted(d for d in rows if rows[d].get("gamma") and d >= 6)); g = np.array([rows[d]["gamma"] for d in D]); n = D - 2; lam = 1 / g
print(f"{len(D)} gamma points D={D.min():.2f}..{D.max():.2f}")
models = {"1,1.5,2": (1, 1.5, 2), "1,1.5,2,2.5": (1, 1.5, 2, 2.5), "1,2,3,4": (1, 2, 3, 4), "0.5,1,1.5,2": (0.5, 1, 1.5, 2)}
for nm, y in (("lambda=1/gamma", lam), ("gamma", g)):
    print(nm)
    for Dmin in (8, 10, 12):
        m = D >= Dmin; line = f"  D>={Dmin}: "
        for k, ps in models.items():
            A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1); c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            line += f"[{k}] L={c[0]:.4f} rms={np.sqrt(np.mean((A@c-y[m])**2)):.0e} | "
        print(line)
# local exponent of lambda - 2 and of 1/2 - gamma
q = -np.gradient(np.log(lam - 2), np.log(n)); p = -np.gradient(np.log(0.5 - g), np.log(n))
for i in range(0, len(D), 4): print(f"  D={D[i]:6.2f} lam={lam[i]:.6f} gamma={g[i]:.6f}  -dln(lam-2)/dln n={q[i]:.3f}  -dln(1/2-gamma)/dln n={p[i]:.3f}")
