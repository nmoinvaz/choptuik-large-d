"""Is the large-D echo shrinking to a sinusoid (a Hopf cycle at D = infinity) or tending to a finite harmonic content?
Fits the non-first-harmonic power fraction 1 - P1 of hp/data/echo_shapes.csv (D >= 40) by f_inf + c/D and by c D^-p,
and compares f_inf with (F3/F1)^2 from the README's extrapolated ratio 0.055 (finite-D limit of |F3/F1|).
--plant: replace the data by an exact c/D law (a Hopf-like decay to 0); the fitted f_inf must come out 0."""
import sys, csv
import numpy as np
rows = [r for r in csv.reader(l for l in open('/Users/nathan/Source/choptuik-large-d/hp/data/echo_shapes.csv') if not l.startswith('#'))]
hdr, data = rows[0], rows[1:]
D = np.array([float(r[0]) for r in data]); P1 = np.array([float(r[4]) for r in data]); A = np.array([float(r[3]) for r in data])
m = D >= 40
D, f, A = D[m], 1 - P1[m], A[m]
if '--plant' in sys.argv:
    f = 0.36 / D
X = np.vstack([np.ones_like(D), 1 / D]).T
(finf, c), *_ = np.linalg.lstsq(X, f, rcond=None)
p, lc = np.polyfit(np.log(D), np.log(f), 1)
print("D:", D, " 1-P1:", f, " A*sqrt(D):", A * np.sqrt(D))
print("fit f_inf + c/D: f_inf = %.5f, c = %.4f, max resid %.1e" % (finf, c, np.max(np.abs(X @ [finf, c] - f))))
print("fit c D^-p: p = %.3f" % -p)
print("(F3/F1)^2 with |F3/F1| -> 0.055 (README): %.5f;  with 0.055 and |F5/F3| = 0.059: %.5f" % (0.055 ** 2, 0.055 ** 2 * (1 + 0.059 ** 2)))
ok = finf > 0.0015 if '--plant' not in sys.argv else abs(finf) < 1e-6
sys.exit(0 if ok else 1)
