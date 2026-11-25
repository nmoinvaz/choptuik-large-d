"""Shape of the departure d(D) = Delta_EEG - Delta_ours of arXiv:2611.07007's Table I on 60 <= D <= 200 (#1089).

Asks whether d turns on near D 157, where the horizon exponent sigma_R ~ n/2 + 2.6 (notes/LARGE_D_BASICS.md 2) passes
the paper's Chebyshev degree 80, or grows smoothly from lower D. Fits d by polynomials of degree 1-3, by a broken line
with a step whose knot is scanned over 100-185, and by straight lines on four windows. The noise is the six-decimal
rounding of the table, sigma = 1e-6/sqrt(12), a rounding model and not a bound.

Usage: python3 eeg_gap_shape.py hp/data/delta_published_compare.csv [--mutate=step]
  --mutate=step: control; adds a step of 1e-6 at D >= 157 to d. The knot scan must then return 157 with a step near
  1e-6, and the script exits 1 if it does not.
"""
import csv
import sys

import numpy as np


def main():
    path = sys.argv[1]
    mutate = '--mutate=step' in sys.argv
    D, d = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            x = float(row['D'])
            if row['published_ref'] == 'EEG2026' and 60 <= x <= 200:
                D.append(x)
                d.append(-float(row['diff']))  # diff = ours - published
    D = np.array(D)
    d = np.array(d)
    if mutate:
        d = d + np.where(D >= 157, 1e-6, 0.0)
    s = 1e-6 / np.sqrt(12)
    print(f"points {len(D)}, D {D.min():g}-{D.max():g}, rounding sigma {s:.3g}" + (", MUTATED (step)" if mutate else ""))
    for deg in (1, 2, 3):
        c = np.polyfit(D, d, deg)
        r = d - np.polyval(c, D)
        dc = np.polyder(c)
        print(f"poly deg {deg}: chi2/dof {np.sum((r / s) ** 2) / (len(D) - deg - 1):.3f}, "
              f"slope at 100 {np.polyval(dc, 100):.3e}, at 157 {np.polyval(dc, 157):.3e}, "
              f"at 190 {np.polyval(dc, 190):.3e} per unit D")
    best = None
    for k in np.arange(100, 186, 1.0):
        A = np.column_stack([np.ones_like(D), D - k, np.maximum(D - k, 0), (D >= k).astype(float)])
        c, *_ = np.linalg.lstsq(A, d, rcond=None)
        chi = np.sum(((d - A @ c) / s) ** 2)
        if best is None or chi < best[0]:
            best = (chi, k, c)
    chi, k, c = best
    print(f"broken line + step: best knot {k:.0f}, chi2/dof {chi / (len(D) - 4):.3f}, slope before {c[1]:.3e}, "
          f"after {c[1] + c[2]:.3e}, step {c[3]:.3e}")
    over = d > 5e-7  # more than half a unit of the sixth decimal
    first = next(D[i] for i in range(len(D)) if over[i:].all())
    k = int(np.argmax(d))
    print(f"every row from D {first:g} on exceeds half a unit (smallest there {d[D >= first].min():.2e}); "
          f"largest {d[k]:.2e} at D {D[k]:g}")
    c1 = np.polyfit(D, d, 1)
    print(f"linear fit: zero at D {-c1[1] / c1[0]:.1f}; extrapolated (no mechanism) to D 400, 2000, 10000: "
          + ", ".join(f"{np.polyval(c1, x):.2e}" for x in (400, 2000, 10000)))
    for lo, hi in ((60, 100), (100, 140), (140, 170), (170, 200)):
        m = (D >= lo) & (D <= hi)
        cw = np.polyfit(D[m], d[m], 1)
        print(f"window {lo}-{hi}: mean d {d[m].mean():.3e}, slope {cw[0]:.3e} per unit D")
    if mutate and not (abs(k - 157) <= 2 and 0.5e-6 < c[3] < 1.5e-6):
        print("CONTROL FAILED: the planted step was not found")
        sys.exit(1)


if __name__ == '__main__':
    main()
