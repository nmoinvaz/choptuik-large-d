"""light-cone exponent shift on the DD solutions: n <G(1)>/2 (and the exact local-exponent shift) vs <F^2>.
(Reads the runs' saved states from /home/claude/dss/hp_scan, which are not in the repo; hp/data/seeds holds a few.)"""
import os
import glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from largeD_profiles import solver_for
H = "/home/claude/dss/hp_scan/"
for t in (20, 25, 30, 35, 40, 45, 50, 55, 60):
    fs = sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz"))
    if not fs:
        continue
    s, zh, zl, meta = solver_for(fs[0])
    M, D = meta["M"], meta["D"]; n = D - 2
    Delta = float(zh[-1] + zl[-1]); w = 2 * np.pi / Delta
    ch, cl = s.series(zh, zl, 0); C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    hh, hl = s.series(zh, zl, 1); Hs = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
    a1, c1 = Hs[0, 2], Hs[0, 3]
    G1 = (n - 1) * c1 * (a1**2 - 1)
    # c_x at x = 1 from the horizon series (u = 1 - x): c_x = -H[1,3]
    cx1 = -Hs[1, 3]
    F2 = np.mean((C[1, 0] * np.sqrt(n)) ** 2)
    Gm = np.mean(G1)
    lam0 = n / 2 - 1 - 1j * w
    lam = (lam0 + Gm) / (1 - np.mean(cx1))
    print(f"D {D:7.3f}: max|c(1)-1| {np.abs(c1 - 1).max():.1e}  max|c_x(1) - G(1)| {np.abs(cx1 - G1).max():.1e}  "
          f"n<G(1)>/2 {n * Gm / 2:.4f}  <F^2> {F2:.4f}  Re(lam - lam0) {np.real(lam - lam0):.4f}  "
          f"Im(lam - lam0) {np.imag(lam - lam0):.4f}  n<G(1)>/(2<F^2>) {n * Gm / 2 / F2:.4f}", flush=True)
