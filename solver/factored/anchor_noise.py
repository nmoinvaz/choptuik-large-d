"""How the horizon series amplifies rounding near the order sigma ~ n/2 (#1088, notes/LARGE_D_FACTORED.md section 4).

Two measurements against n, each in float64 at M harmonic points, from the D-100 hp state rescaled to each D
(p1 and m_h by sqrt(100/n)... see fsolve.py), which need not be a solution:
  (a) single-order gain: the change of U at z_K = 1/2 per unit change of one s-coefficient of beta at order k, for
      k = k* (the mass mode's s-peak, = sigma_R - 2 or so at x_K = 1/sqrt 2) and k = k* - n/5;
      predicted ~ 2^(n/2) / omega at k*;
  (b) reproducibility of the mass-mode (anchor) column of the Jacobian: complex step at h = 1e-30 against central
      differences at h = 1e-7 and 1e-9, relative difference; predicted ~ eps 2^(n/2) / omega with eps = 2^-53.
PASS when log10 of (b)'s discrepancy grows with n at a slope within 30% of log10(2)/2 = 0.1505 per unit n over the
n where it is between 1e-5 (above the central differences' own error, about 2e-6 here) and 10; the control
--mutate=flat takes the column from the c rows only and must FAIL if they do not carry the gain.

Usage: python3 anchor_noise.py STATE.npz [--M=4] [--Ds=22,32,42,52,62,72,82,100] [--mutate=flat]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries import Factored  # noqa: E402
from fsolve import load_start  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
M = int(opts.get("M", 4))
Ds = [float(v) for v in opts.get("Ds", "22,32,42,52,62,72,77,82,87,92,100").split(",")]
mut = opts.get("mutate", "")
y0, D0 = load_start(path, M)
rows = slice(0, 2 * M) if mut != "flat" else slice(2 * M, 3 * M)
res = []
print("%7s %6s %5s %12s %12s %12s %12s" % ("D", "n", "k*", "gain(k*)", "gain(k*-n/5)", "anchor disc", "pred eps2^(n/2)/om"))
for D in Ds:
    n, n0 = D - 2, D0 - 2
    y = y0.copy()
    s = np.sqrt(n0 / n)
    y[0:M] *= s
    y[2 * M:3 * M] *= s
    y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
    f = Factored(D, M)
    p1, c0, Q0, anc, Delta = f.unpack4(y)
    om = 2 * np.pi / Delta
    kst = f.kstar()
    ho = f.horizon_anchor(Q0, om, anc)
    U0, V0, c0s, b0 = ho["series"]
    gains = []
    for k in (kst, max(1, kst - int(n / 5))):
        b = b0.copy()
        b[k] = b[k] + 1e-8
        U1, V1, c1 = f.horizon_fields(Q0, om, b)
        gains.append(np.max(np.abs((U1 - U0).sum(0))) / 1e-8)
    # anchor column of the matching rows
    def R(yy):
        return f.residual4(yy)[rows]
    i = 3 * M
    yc = y.astype(complex)
    yc[i] += 1e-30j
    cs = R(yc).imag / 1e-30
    fds = []
    for h in (1e-7, 1e-9):
        yp = y.copy()
        yp[i] += h
        ym = y.copy()
        ym[i] -= h
        fds.append((R(yp) - R(ym)) / (2 * h))
    disc = max(np.linalg.norm(cs - fd) for fd in fds) / max(np.linalg.norm(cs), np.linalg.norm(fds[0]))
    pred = 2.0**-53 * 2 ** (n / 2) / om
    res.append((n, disc))
    print("%7.2f %6.2f %5d %12.3e %12.3e %12.3e %12.3e" % (D, n, kst, gains[0], gains[1], disc, pred), flush=True)

nn = np.array([r[0] for r in res])
dd = np.log10(np.array([r[1] for r in res]))
sel = (dd > -5) & (dd < 1)
ok = False
if sel.sum() >= 3:
    slope = np.polyfit(nn[sel], dd[sel], 1)[0]
    ok = abs(slope - np.log10(2) / 2) < 0.3 * np.log10(2) / 2
    print("slope of log10(discrepancy) against n over %d points: %.4f (log10(2)/2 = %.4f)" % (sel.sum(), slope, np.log10(2) / 2))
else:
    print("fewer than 3 points with 1e-5 < discrepancy < 10")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
