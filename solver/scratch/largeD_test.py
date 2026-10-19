"""Leading-order large-D structure (derived from our equations with n = D-2 -> infinity):
   q = (p - m)/2 = r Pi  ~  x B(tau) / c,      c^2 ~ 1 - 2 B^2 (1 - x^2),      p + m = O(1/n).
Equivalent to Ecker-Ecker-Grumiller's  f = sqrt((1 + beta^2 x^2)/(1 + beta^2))  with 2B^2 = beta^2/(1+beta^2)."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from dense import dense
cases = [(4.0, 31, "spec_kc31.npy"), (5.0, 47, "solD_5.000_kc47.npy"), (6.0, 31, "solD_6.000_kc31.npy"),
         (7.0, 31, "solD_7.000_kc31.npy"), (8.0, 31, "solD_8.000_kc31.npy")] + [(float(a), 31, b) for a, b in (x.split(":") for x in sys.argv[1:])]
out = {}
for D, kc, f in cases:
    n = D - 2
    X, Fd, Dl, N = dense(D, kc, f)
    p, m, a, c = Fd[:, 0], Fd[:, 1], Fd[:, 2], Fd[:, 3]
    q = 0.5 * (p - m); s = 0.5 * n * (p + m)
    sel = (X > 0.05) & (X < 0.95)
    Bx = (c * q / X[:, None])[sel]                      # should be independent of x
    B = Bx.mean(axis=0)
    xvar = np.abs(Bx - B).max() / np.abs(B).max()
    c2_pred = 1 - 2 * B[None, :] ** 2 * (1 - X[sel, None] ** 2)
    cerr = np.abs(c[sel] ** 2 - c2_pred).max()
    a2n = (n * (a ** 2 - 1))[sel]; A_pred = (p ** 2 + m ** 2)[sel]
    print(f"D={D:5.2f} n={n:5.2f}: x-variation of cq/x {xvar:.3f};  max|c^2 - (1-2B^2(1-x^2))| {cerr:.4f};  "
          f"max|p+m| {np.abs(p+m).max():.3f};  max|n(a^2-1) - (p^2+m^2)| {np.abs(a2n - A_pred).max():.3f};  "
          f"B range [{B.min():.3f},{B.max():.3f}], Delta {Dl:.4f}", flush=True)
    out[D] = dict(n=n, Delta=Dl, B=B.tolist(), xvar=float(xvar), cerr=float(cerr))
json.dump(out, open("/home/claude/dss/largeD_B.json", "w"))
