import glob
import numpy as np
from largeD_profiles import solver_for, harmonics
from states import H   # hp/data/states (in the repo), else the runs' directory; see hp/states.py
picks = []
for t in (12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60):
    fs = sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz"))
    if fs:
        picks.append(fs[0])
rows = []
for f in picks:
    s, zh, zl, meta = solver_for(f)
    M, D = meta["M"], meta["D"]
    n = D - 2
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    Fh = harmonics(C[1, 0] * np.sqrt(n), True)       # F(theta) = sqrt(n) p_1(theta)
    amp = np.abs(Fh)
    c0 = harmonics(C[0, 3], False)                    # c at the centre: mean and harmonic 2
    rows.append((D, n, zh[-1] + zl[-1], amp[1], amp[3] / amp[1], amp[5] / amp[1], amp[7] / amp[1],
                 n * (1 - c0[0].real / 2), np.mean((C[1, 0] * np.sqrt(n)) ** 2)))
    print(f"D {D:7.3f}: |F_1| {amp[1]:.6f}  F3/F1 {amp[3]/amp[1]:.5f}  F5/F1 {amp[5]/amp[1]:.6f}  F7/F1 {amp[7]/amp[1]:.2e}  "
          f"n(1 - <c(0)>) {n * (1 - np.mean(C[0, 3])):.4f}  <F^2> {np.mean((C[1, 0] * np.sqrt(n)) ** 2):.4f}", flush=True)
np.save("Frows.npy", np.array([r[:3] + r[3:] for r in rows], dtype=float))
