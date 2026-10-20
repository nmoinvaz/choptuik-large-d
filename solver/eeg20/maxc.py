"""max over theta of c(x=0, theta) on the double-double solutions vs the completed-square prediction
1 - c_max(x) = (eps^3/2) omega^2 F'(theta0)^2 (1 - x^4) at leading order (Ecker-Ecker-Grumiller's Eq. (22) in our scaling,
beta = sqrt(2 eps) F). eps = 1/n."""
import os
import glob, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
from largeD_profiles import solver_for, harmonics
from states import H   # hp/data/states (in the repo), else the runs' directory; see hp/states.py
for t in (12, 20, 30, 40, 50, 60):
    f = (sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz")))[0]
    s, zh, zl, meta = solver_for(f)
    M, D = meta["M"], meta["D"]; n = D - 2; eps = 1 / n
    Delta = float(zh[-1] + zl[-1]); w = 2 * np.pi / Delta
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    ce = harmonics(C[0, 3], False); ks = np.arange(len(ce))
    cev = lambda th: np.real(ce[0] / 2 + np.sum(ce[1:] * np.exp(1j * ks[1:] * th)))
    Fh = harmonics(C[1, 0] * np.sqrt(n), True)
    Fev = lambda th, d=0: np.real(np.sum(Fh[1:] * (1j * ks[1:]) ** d * np.exp(1j * ks[1:] * th)))
    g = np.linspace(0, np.pi, 4001)
    cv = np.array([cev(q) for q in g]); i = int(np.argmax(cv)); thm = g[i]
    # zero of F near the max of c
    a, b = thm - 0.3, thm + 0.3
    for _ in range(60):
        cc = 0.5 * (a + b)
        if Fev(a) * Fev(cc) <= 0: b = cc
        else: a = cc
    z0 = 0.5 * (a + b); F1 = Fev(z0, 1)
    pred = eps ** 3 / 2 * w * w * F1 * F1
    print(f"D {D:7.3f}: 1 - max c(0) = {1 - cv[i]:.4e} at theta {thm:.4f} (zero of F at {z0:.4f}); "
          f"prediction (eps^3/2) w^2 F'^2 = {pred:.4e}; ratio {(1 - cv[i]) / pred:.4f}; mean 1 - c(0) = {1 - ce[0].real / 2:.4e}", flush=True)
