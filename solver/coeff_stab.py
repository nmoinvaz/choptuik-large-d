"""Is Delta(D) a clean power series? Fit Delta = sum_k c_k x^k on the precise data (D <= 17.5, ~1e-12) for several
expansion variables x and degrees m; stable leading coefficients across m mean the series form is right."""
import numpy as np
from data_ne import load
rows = load(); D = np.array(sorted(k for k in rows if 6 <= k <= 17.5)); y = np.array([rows[k]["Delta"] for k in D]); n = D - 2
om = 2 * np.pi / y
VARS = {"1/sqrt(n)": n ** -0.5, "1/n": 1 / n, "1/sqrt(D-1)": (D - 1) ** -0.5, "1/(D-1)": 1 / (D - 1)}
for name, x in VARS.items():
    print(f"x = {name}   ({len(D)} points, D = {D.min():.1f}..{D.max():.1f})")
    for m in range(4, 10):
        A = np.vander(x, m + 1, increasing=True)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = A @ c - y
        print(f"  m={m}: rms {np.sqrt(np.mean(r**2)):.0e}  c0..c5 = " + " ".join(f"{v:+9.4f}" for v in c[:6]))
