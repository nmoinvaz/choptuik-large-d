"""Delta_inf from the NE-corrected data: asymptotic families in 1/n (n = D - 2) on sliding windows, hold-out test."""
import numpy as np, sys
from data_ne import load
rows = load(); D = np.array(sorted(rows)); y = np.array([rows[d]["Delta"] for d in D]); n = D - 2
Dmax = float(sys.argv[1]) if len(sys.argv) > 1 else 99
m0 = D <= Dmax; D, y, n = D[m0], y[m0], n[m0]
print(f"{len(D)} points, D = {D.min():.2f} .. {D.max():.3f}")
models = {"1,1.5,2": (1, 1.5, 2), "1,2,3": (1, 2, 3), "1,1.5,2,2.5": (1, 1.5, 2, 2.5), "1,2,3,4": (1, 2, 3, 4),
          "0.5,1,1.5,2": (0.5, 1, 1.5, 2), "1,1.5,2,3": (1, 1.5, 2, 3)}
for Dmin in (8, 10, 12, 14, 16):
    m = D >= Dmin
    line = f"D>={Dmin:2d} ({m.sum():2d}): "
    for name, ps in models.items():
        A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1)
        c, *_ = np.linalg.lstsq(A, y[m], rcond=None); r = A @ c - y[m]
        line += f" [{name}] L={c[0]:.4f} rms={np.sqrt(np.mean(r**2)):.0e} |"
    print(line)
print("\nhold-out (fit 10 <= D <= Dcut, worst error on D > Dcut):")
for Dcut in (16, 18, 20):
    m = (D >= 10) & (D <= Dcut); t = D > Dcut
    if not t.any(): continue
    out = f"  Dcut={Dcut}: "
    for name, ps in models.items():
        A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1)
        c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        At = np.stack([np.ones(t.sum())] + [n[t] ** -p for p in ps], 1)
        out += f"[{name}] L={c[0]:.4f} err={np.abs(At @ c - y[t]).max():.0e} "
    print(out)
