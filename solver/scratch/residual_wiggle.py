"""Fit a flexible smooth curve to all exact data and look at the residual pattern: is there oscillatory structure?"""
import numpy as np, json
from formula_zoo import load
rows = load()
for target, dmin in (("gamma", 4.0), ("Delta", 4.0)):
    D = np.array(sorted(d for d in rows if rows[d][target] is not None and d >= dmin)); y = np.array([rows[d][target] for d in D])
    n = D - 2; x = n ** -0.5
    for deg in (6, 8):
        c = np.polyfit(x, y, deg); r = y - np.polyval(c, x)
        print(f"\n{target}: polynomial of degree {deg} in 1/sqrt(n) on D in [{D.min()}, {D.max():.2f}]; max |resid| = {np.abs(r).max():.1e}")
        if deg == 8:
            # sign changes and spacing
            sc = np.where(np.sign(r[1:]) != np.sign(r[:-1]))[0]
            print("   residual sign changes near D =", np.round(0.5 * (D[sc] + D[sc + 1]), 2).tolist())
            for d, rr in zip(D[::2], r[::2]): print(f"   D={d:6.2f} resid={rr:+.2e}")
