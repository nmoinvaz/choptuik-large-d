import sys
import numpy as np
from largeD_profiles import profile, harmonics
from states import H   # hp/data/states (in the repo), else the runs' directory; see hp/states.py
files = ["up2_D20.0437", "up6_D30.0688", "up7_D40.0201", "up8_D50.0075", "up13_D60.0020"]
xs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
rows = {}
for f in files:
    D, Delta, M, out, _ = profile(H + f + ".npz", xs)
    n = D - 2
    r = {}
    for x in xs:
        p, m, a, c = out[x]
        hp = np.abs(harmonics(p, True))
        r[x] = dict(p1=hp[1], p3=hp[3], p5=hp[5], npm=n * np.sqrt(np.mean((p + m) ** 2)), na=n * np.mean(a - 1),
                    nav=n * (a.max() - a.min()), c=np.mean(c), cv=c.max() - c.min(), xc=np.min(x - c),
                    pmax=np.abs(p).max())
    rows[D] = (n, Delta, r)
    print(f"D {D:.2f} Delta {Delta:.10f}", flush=True)
for key, lab in (("p1", "p: harmonic-1 amplitude"), ("p3", "p: harmonic-3 amplitude"), ("p5", "harmonic 5"),
                 ("npm", "n * rms(p + m)"), ("na", "n * mean(a - 1)"), ("nav", "n * (max - min) of a over tau"),
                 ("c", "mean c"), ("cv", "(max - min) of c over tau"), ("xc", "min over tau of x - c")):
    print(f"\n{lab}\n  {'D':>6s} " + " ".join(f"{x:>8g}" for x in xs))
    for D, (n, Delta, r) in rows.items():
        print(f"  {D:6.2f} " + " ".join(f"{r[x][key]:8.4f}" if abs(r[x][key]) >= 1e-3 or r[x][key] == 0 else f"{r[x][key]:8.1e}" for x in xs))
