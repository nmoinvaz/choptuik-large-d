"""python3 largeD_Fratios.py: Ratios of F's odd harmonics F_{nu+2}/F_nu on the double-double solutions, D = 12-60, extrapolated in 1/n:
test of the amplitude-expansion prediction (nearly nu-independent ratio q A^2, |F3/F1| = 0.055 at D = infinity)."""
import glob, sys
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from largeD_profiles import solver_for, harmonics
H = "/home/claude/dss/hp_scan/"   # the runs' saved states (not in the repo; hp/data/seeds holds a few)
rows = []
for t in (12, 14, 16, 18, 20, 23, 26, 30, 34, 38, 42, 46, 50, 54, 58, 60):
    fs = sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz"))
    if not fs:
        continue
    s, zh, zl, meta = solver_for(fs[0])
    M, D = meta["M"], meta["D"]; n = D - 2
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    Fh = harmonics(C[1, 0] * np.sqrt(n), True)
    r = [Fh[k + 2] / Fh[k] for k in (1, 3, 5, 7, 9)]
    rows.append((D, n, abs(Fh[1]), *r))
    print(f"D {D:6.2f}  |F1| {abs(Fh[1]):.4f}  " + "  ".join(f"|F{k+2}/F{k}| {abs(x):.5f} arg {np.angle(x)/np.pi:+.3f}pi"
          for k, x in zip((1, 3, 5, 7, 9), r)), flush=True)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fratios.npy"), np.array(rows, dtype=complex))
R = np.array(rows, dtype=complex); x = 1 / R[:, 1].real
print("extrapolations to n = infinity (polynomial in 1/n, fit on D >= 20):")
m = R[:, 0].real >= 20
for j, k in enumerate((1, 3, 5, 7, 9)):
    y = np.abs(R[:, 3 + j])
    ex = [np.polyval(np.polyfit(x[m], y[m], deg), 0) for deg in (1, 2, 3)]
    print(f"  |F{k+2}/F{k}| -> " + ", ".join(f"{e:.4f}" for e in ex) + "  (deg 1, 2, 3)")
y = np.abs(R[:, 2]); print("  |F1| -> " + ", ".join(f"{np.polyval(np.polyfit(x[m], y[m], d), 0):.4f}" for d in (1, 2, 3)))
