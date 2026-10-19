"""Fit g(n) = -n^2 dDelta/dn = a + b n^-1/2 + c n^-1 (+ d n^-3/2), then L = Delta(n) - [a/n + (2/3) b n^-3/2 + (1/2) c n^-2 + (2/5) d n^-5/2]."""
import numpy as np
nn, g = np.load("deriv_diag.npy").T
from data_best import load
rows = load(); D = np.array(sorted(rows)); y = np.array([rows[d]["Delta"] for d in D]); n = D - 2
for Dmin in (12, 14, 16, 18):
    for nt in (3, 4):
        m = nn + 2 >= Dmin
        A = np.stack([nn[m] ** (-k / 2) for k in range(nt)], 1); c, *_ = np.linalg.lstsq(A, g[m], rcond=None)
        rms = np.sqrt(np.mean((A @ c - g[m]) ** 2))
        integ = lambda x: sum(c[k] * x ** (-(k / 2 + 1)) / (k / 2 + 1) for k in range(nt))
        sel = D >= Dmin; Ls = y[sel] - integ(n[sel])
        print(f"D>={Dmin} {nt} terms: a={c[0]:.3f} b={c[1]:+.3f} rms(g)={rms:.1e}  L from each point: mean {Ls.mean():.5f} spread {Ls.max()-Ls.min():.1e}")
