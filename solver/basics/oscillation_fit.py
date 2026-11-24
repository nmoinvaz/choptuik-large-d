"""The order-293 chain's departure from the D <= 61 1/D series as a growing oscillation (#1086, LARGE_D_BASICS.md 7).

Reads eeg_gap.py --at output (columns D, Delta, ks, state-fit(d16), state-fit(d18), path), keeps the rows of order
KS on DMIN <= D <= DMAX, and fits d(D) = A exp(r (D - 260)) cos(2 pi (D - D0)/P) + b by least squares (scipy), from
starting periods 2, 3, 4, 5, 6 and 8, keeping the best. Prints A, r, P, D0, b, the rms residual against the rms of d,
and the extrema of d between sign changes.

Usage: python3 oscillation_fit.py FILE [--ks=293] [--dmin=254] [--dmax=263.6] [--mutate=shuffle]
  --mutate=shuffle: control; permutes d over the D values (seed 1), which must leave most of the rms unexplained.
"""
import sys

import numpy as np
from scipy.optimize import curve_fit


def model(D, A, r, P, D0, b):
    return A * np.exp(r * (D - 260)) * np.cos(2 * np.pi * (D - D0) / P) + b


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ks = int(opts.get("ks", 293))
    lo, hi = float(opts.get("dmin", 254)), float(opts.get("dmax", 263.6))
    Ds, ds = [], []
    for line in open(args[0]):
        if line.startswith("#"):
            continue
        w = line.split()
        if int(w[2]) == ks and lo <= float(w[0]) <= hi:
            Ds.append(float(w[0]))
            ds.append(float(w[3]))
    D, d = np.array(Ds), np.array(ds)
    o = np.argsort(D)
    D, d = D[o], d[o]
    if opts.get("mutate") == "shuffle":
        d = np.random.default_rng(1).permutation(d)
    best = None
    for P0 in (2, 3, 4, 5, 6, 8):
        for D00 in np.linspace(lo, lo + P0, 8):
            try:
                p, _ = curve_fit(model, D, d, p0=[2e-7, 0.5, P0, D00, 0.0], maxfev=20000)
            except RuntimeError:
                continue
            rms = np.sqrt(np.mean((model(D, *p) - d) ** 2))
            if best is None or rms < best[0]:
                best = (rms, p)
    rms, p = best
    print("# order %d, %d states on D %.2f..%.2f" % (ks, D.size, D.min(), D.max()))
    print("A %.3e  r %.3f per unit D  P %.3f  D0 %.3f  b %.2e  rms residual %.2e  rms of d %.2e"
          % (p[0], p[1], abs(p[2]), p[3], p[4], rms, np.sqrt(np.mean(d ** 2))))
    ext = []
    for i in range(1, D.size - 1):
        if (d[i] - d[i - 1]) * (d[i + 1] - d[i]) < 0 and abs(d[i]) > 2e-8:
            ext.append("%.2f:%+.2e" % (D[i], d[i]))
    print("extrema (|d| > 2e-8): " + " ".join(ext))


if __name__ == "__main__":
    main()
