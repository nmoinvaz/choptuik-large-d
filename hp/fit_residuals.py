#!/usr/bin/env python3
"""Delta minus its large-D series fit at every continuation point from D = 10 up, for plot_errors.py.

The fit is dd_fits_mp's: a degree-18 polynomial in 1/(D - 1) at 60 digits, least squares on the cleaned points
10 <= D <= Dfit (dd_fits_mp.clean drops the points that sit off their neighbours). Writes data/fit_residuals.csv with
columns D, Delta_minus_fit and fitted (1 for the points the fit used; 0 for the ones the cleaning dropped and the
ones above Dfit, where the difference is an extrapolation error).

    cd hp && python3 fit_residuals.py [Dfit=61.03] [degree=18]
"""
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dd_fits_mp as F  # noqa: E402

mp.mp.dps = 60
F.MODELS["1/(D-1)"] = lambda D: 1 / (mp.mpf(D) - 1)


def main():
    dfit = float(sys.argv[1]) if len(sys.argv) > 1 else 61.03
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    pts, _ = F.load_points()
    Ds = [d for d in sorted(pts) if d >= 10]
    fitD = [d for d in F.clean(pts, sorted(pts)) if 10 <= d <= dfit]
    ev, res, _ = F.fit(pts, fitD, "1/(D-1)", deg)
    used = set(fitD)
    with open(os.path.join(HERE, "data", "fit_residuals.csv"), "w") as out:
        out.write("D,Delta_minus_fit,fitted\n")
        for d in Ds:
            out.write(f"{d},{mp.nstr(pts[d] - ev(d), 6)},{int(d in used)}\n")
    print(f"degree {deg} in 1/(D-1) on {len(fitD)} cleaned points, D = {fitD[0]} .. {fitD[-1]}: largest residual "
          f"{mp.nstr(res, 3)}; {len(Ds) - len(fitD)} of the {len(Ds)} points from D = 10 up not fitted")


if __name__ == "__main__":
    main()
