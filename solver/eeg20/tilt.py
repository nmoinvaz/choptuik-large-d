"""The zero line of p - m tilts as the completed square predicts.

Near a zero theta0 of F the formal family gives P - M = 2 x F + eps x^3 omega (omega F'' + 3 F') + ..., so the zero of p - m
at radius x sits at theta0 + dtheta(x), dtheta = -eps omega x^2 (3 F' + omega F'')/(2 F') + O(eps^2): the NLO term of
Ecker-Ecker-Grumiller is the shift of the zero, and b - 1 = 2 eps^2 x^2 (F(theta - dtheta))^2 + 2 eps^4 omega^2 x^4 F'^2 + ... >= 0.
Measures the zero of p - m on the double-double solutions at x = 0.1 .. 1 and compares with dtheta(x)."""
import os
import glob, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
from largeD_profiles import profile, harmonics
from states import H   # hp/data/states (in the repo), else the runs' directory; see hp/states.py
xs = [0.05, 0.3, 0.5, 0.7, 0.9, 1.0]
for t in (20, 30, 45, 60):
    f = (sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz")))[0]
    D, Delta, M, out, (C, Hs, nodes) = profile(f, xs)
    n = D - 2; eps = 1 / n; w = 2 * np.pi / Delta
    def zero(fh):
        Fh = harmonics(fh, True); ks = np.arange(len(Fh))
        ev = lambda th, d=0: np.real(np.sum(Fh[1:] * (1j * ks[1:]) ** d * np.exp(1j * ks[1:] * th)))
        a, b = 1.2, 1.9
        if ev(a) * ev(b) > 0:
            g = np.linspace(0, np.pi, 400); v = [ev(q) for q in g]
            i = next(i for i in range(399) if v[i] * v[i + 1] < 0); a, b = g[i], g[i + 1]
        for _ in range(60):
            c = 0.5 * (a + b)
            if ev(a) * ev(c) <= 0: b = c
            else: a = c
        return 0.5 * (a + b), ev(0.5 * (a + b), 1), ev(0.5 * (a + b), 2)
    # F from the centre slope
    z0, F1, F2 = zero(C[1, 0] * np.sqrt(n))
    line = []
    for x in xs:
        st = out[x]
        zx, _, _ = zero(st[0] - st[1])
        pred = -eps * w * x * x * (3 * F1 + w * F2) / (2 * F1)
        line.append(f"x {x:4.2f}: {zx - z0:+.5f} (pred {pred:+.5f})")
    print(f"D {D:7.3f}  eps {eps:.4f}: shift of the zero of p - m relative to F's zero: " + "; ".join(line), flush=True)
