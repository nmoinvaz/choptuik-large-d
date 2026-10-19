"""Delta(D -> inf) from asymptotic fits over sliding D windows."""
import numpy as np, itertools
from data_best import load
rows = load(); D = np.array(sorted(rows)); y = np.array([rows[d]["Delta"] for d in D]); n = D - 2
print(f"{len(D)} points, D = {D.min():.2f} .. {D.max():.3f}")
models = {"1,1.5,2": (1, 1.5, 2), "1,2,3": (1, 2, 3), "1,1.5,2,2.5": (1, 1.5, 2, 2.5), "1,2,3,4": (1, 2, 3, 4),
          "0.5,1,1.5,2": (0.5, 1, 1.5, 2), "1,1.5,2,3": (1, 1.5, 2, 3)}
for Dmin in (8, 10, 12, 14):
    m = D >= Dmin
    line = f"D>={Dmin:2d} ({m.sum():2d} pts): "
    for name, ps in models.items():
        A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1)
        c, *_ = np.linalg.lstsq(A, y[m], rcond=None); r = A @ c - y[m]
        line += f" [{name}] L={c[0]:.4f} a={c[1]:+.2f} rms={np.sqrt(np.mean(r**2)):.0e} |"
    print(line)
# hold-out: fit D<=Dcut, predict the highest points
print("\nhold-out test (fit 8<=D<=Dcut, predict max D):")
for Dcut in (15, 16.5, 18):
    m = (D >= 8) & (D <= Dcut); t = D > Dcut
    if not t.any(): continue
    for name, ps in models.items():
        A = np.stack([np.ones(m.sum())] + [n[m] ** -p for p in ps], 1)
        c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        At = np.stack([np.ones(t.sum())] + [n[t] ** -p for p in ps], 1)
        print(f"  Dcut={Dcut}: [{name:12s}] L={c[0]:.4f}  max|pred-true| over D>{Dcut} = {np.abs(At @ c - y[t]).max():.1e}")
