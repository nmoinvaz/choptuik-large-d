#!/usr/bin/env python3
"""Harmonic ratios of the finite-D echo profiles against the geometric model of arXiv:2611.07007v1, Eq. (16)
(notes/EEG_PROFILE.md section 3.3).

For each double-double state of hp/data/states (D = 12-60), F = sqrt(n) d_x p(x = 0) as in hp/largeD_Fratios.py, its odd
harmonics F_nu, gauge-fixed by a shift of theta to F_1 = -i|F_1| (this repository's sin-theta gauge, so that the paper's
w = q e^{i delta} is r_1 = F_3/F_1; notes/EEG_PROFILE.md 2.1). Printed per D: |F_1|, r_nu = F_{nu+2}/F_nu for
nu = 1..11 (modulus and phase), the paper's fit (S61)-(S62) on the first four harmonics; then polynomial
extrapolations in 1/n (degree 1-3, D >= 20) of |r_nu|, arg r_nu and the fitted (q, delta).

Needs the compiled kernel (hpkernel; hp/build.sh). From D = 35 the runs pin harmonics above 13 to zero, so r_nu stops
at nu = 11.

    python3 finiteD_ratios.py [--csv OUT.csv]
    python3 finiteD_ratios.py --mutate=reverse     theta -> -theta (conjugate harmonics): arg r_1 changes sign, so the
                                                   orientation check must FAIL
Two checks: (orientation) the D = infinity extrapolations of arg r_1 (degrees 2, 3) are positive, as the tuned profile's
0.0988 is; (agreement, criterion set before the first run) they are within 0.01 of it. Exit status 1 if either fails.
On the states of hp/data/states the agreement check FAILS: the phase is not converged at D <= 60 (notes/EEG_PROFILE.md
3.3).
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hp"))
from largeD_profiles import solver_for, harmonics  # noqa: E402
from states import H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
ap.add_argument("--mutate", default=None, choices=[None, "reverse"])
args = ap.parse_args()

TUNED_ARG_R1 = 0.09879      # tuned D = infinity profile at A = 2.72 (tuned_profile.py, data/tuned_profile_A2.72.txt)


def paper_fit(F):
    """(S61)-(S62) on the first four harmonics of bt ~ F (normalization irrelevant for q, delta)."""
    nus = [nu for nu in sorted(F) if nu % 2]
    b = np.array([2 * (1j ** nu) * F[nu] for nu in nus])        # c_n - i s_n of the paper's gauge, up to a factor
    N0 = 0.5 * float(np.sum(np.abs(b) ** 2))                    # <bt^2> = (1/2) sum (c_n^2 + s_n^2)
    n = np.arange(4)

    def res(p):
        Astar = math.sqrt(2 * N0 * (1 - p[0] ** 2))
        d = b[:4] - Astar * (-p[0]) ** n * np.exp(1j * n * p[1])
        return np.r_[d.real, d.imag]
    r = least_squares(res, [0.08, 0.05], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return r.x


rows = []
for t in (12, 14, 16, 18, 20, 23, 26, 30, 34, 38, 42, 46, 50, 54, 58, 60):
    fs = sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz"))
    if not fs:
        continue
    s, zh, zl, meta = solver_for(fs[0])
    M, D = meta["M"], meta["D"]
    n = D - 2
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    Fh = harmonics(C[1, 0] * np.sqrt(n), True)
    if args.mutate == "reverse":
        Fh = np.conj(Fh)
    sh = -math.pi / 2 - np.angle(Fh[1])                         # shift: arg F_1 = -pi/2
    F = {nu: Fh[nu] * np.exp(1j * nu * sh) for nu in range(1, min(len(Fh), 16), 2)}
    r = {nu: F[nu + 2] / F[nu] for nu in (1, 3, 5, 7, 9, 11)}
    qf, df = paper_fit(F)
    rows.append((D, n, abs(F[1]), r, qf, df, F))
    print(f"D {D:7.3f} M {M:3d} |F1| {abs(F[1]):.4f}  " + "  ".join(f"r{nu} {abs(r[nu]):.5f}@{np.angle(r[nu]):+.4f}"
          for nu in (1, 3, 5, 7)) + f"  paper fit q {qf:.5f} delta {df:+.5f}", flush=True)

if args.csv:
    with open(args.csv, "w") as fo:
        fo.write("D,n,nu,ReF_over_F1,ImF_over_F1\n")
        for D, n, _, _, _, _, F in rows:
            for nu in sorted(F):
                x = F[nu] / F[1]
                fo.write(f"{D:.6f},{n:.6f},{nu},{x.real:.12e},{x.imag:.12e}\n")

D = np.array([row[0] for row in rows])
x = 1 / np.array([row[1] for row in rows])
xe = 1 / (np.array([row[1] for row in rows]) + 1)              # the paper's eps = 1/(D - 1)
m = D >= 20


def ext(y, xx):
    return [np.polyval(np.polyfit(xx[m], y[m], deg), 0) for deg in (1, 2, 3)]


print("\nextrapolations to D = infinity, polynomial in 1/n (and in 1/(D-1)), fit on D >= 20, degrees 1, 2, 3:")
lines = {}
for nu in (1, 3, 5, 7, 9, 11):
    ya = np.array([abs(row[3][nu]) for row in rows])
    yp = np.array([np.angle(row[3][nu]) for row in rows])
    a1, a2, p1 = ext(ya, x), ext(ya, xe), ext(yp, x)
    lines[nu] = (a1, p1)
    print(f"  |r_{nu:<2d}| -> " + ", ".join(f"{v:.4f}" for v in a1) + "   (1/(D-1): " + ", ".join(f"{v:.4f}" for v in a2)
          + ")   arg r_" + f"{nu}" + " -> " + ", ".join(f"{v:+.4f}" for v in p1))
qa = np.array([row[4] for row in rows])
da = np.array([row[5] for row in rows])
print("  paper-fit q     -> " + ", ".join(f"{v:.4f}" for v in ext(qa, x)) + "   (1/(D-1): "
      + ", ".join(f"{v:.4f}" for v in ext(qa, xe)) + ")")
print("  paper-fit delta -> " + ", ".join(f"{v:+.4f}" for v in ext(da, x)) + "   (1/(D-1): "
      + ", ".join(f"{v:+.4f}" for v in ext(da, xe)) + ")")
print("  (tuned D = infinity profile at A = 2.72: |r_1| 0.0551, arg 0.0988; |r_3| 0.0572, arg 0.0788; |r_inf| 0.0575, "
      "arg 0.073; the paper: q 0.0551, delta 0.0985)")

p1 = lines[1][1]
ok_or = all(v > 0 for v in p1[1:])
ok_ag = all(abs(v - TUNED_ARG_R1) < 0.01 for v in p1[1:])
print(f"\ncheck (orientation): arg r_1 at D = infinity (degrees 2, 3) positive: {'PASS' if ok_or else 'FAIL'}")
print(f"check (agreement, set before the run): arg r_1 at D = infinity (degrees 2, 3) within 0.01 of {TUNED_ARG_R1}: "
      f"{'PASS' if ok_ag else 'FAIL'}")
sys.exit(0 if (ok_or and ok_ag) else 1)
