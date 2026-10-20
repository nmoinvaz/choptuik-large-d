#!/usr/bin/env python3
"""Error budget for Delta_inf: how the large-D limit of the double-double points moves with the fit choices.

    python3 dinf_budget.py            (about 3-6 minutes; nothing is written)

Grid: model (series in 1/D, in 1/(D-2)) x degree (14, 16, 18, 20, 22) x lower end of the fit (D >= 10, 12, 15) x
upper end / data snapshot (D <= 45, 60.1, 61.03) x cleaning (dd_fits_mp.clean on / off). Each fit is the same
least-squares Chebyshev fit as hp/dd_fits_mp.py (60 digits), solved through the normal equations of the degree-22
basis so one Gram matrix serves every degree of a group (the Chebyshev basis on the fitted range is well conditioned,
and at 60 digits the squared condition number costs nothing; checked against dd_fits_mp.fit below). A fit "reaches
the noise floor" when its largest residual is within 2.5x of the smallest one among the degrees of its group; only
those enter the spread. The limit is the value at 1/D = 0.
"""
import os
import sys
import time

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dd_fits_mp as F  # noqa: E402

mp.mp.dps = 60
DEGS = (14, 16, 18, 20, 22)
DMAX = 22


def design(Ds, model):
    t = [F.MODELS[model](d) for d in Ds]
    a, b = min(t), max(t)
    return F.cheb_rows(t, a, b, DMAX), a, b


def fits_all_degrees(pts, Ds, model):
    """{deg: (limit, max residual)} for every degree in DEGS, from one Gram matrix."""
    X, a, b = design(Ds, model)
    y = [pts[d] for d in Ds]
    N, K = len(Ds), DMAX + 1
    G = [[mp.fsum(X[i, j] * X[i, k] for i in range(N)) for k in range(K)] for j in range(K)]
    r = [mp.fsum(X[i, j] * y[i] for i in range(N)) for j in range(K)]
    t0 = F.cheb_rows([mp.mpf(0)], a, b, DMAX)          # basis at 1/D = 0 (both models vanish as D -> infinity)
    out = {}
    for deg in DEGS:
        m = deg + 1
        c = mp.lu_solve(mp.matrix([row[:m] for row in G[:m]]), mp.matrix(r[:m]))
        res = max(abs(mp.fsum(X[i, j] * c[j] for j in range(m)) - y[i]) for i in range(N))
        lim = mp.fsum(t0[0, j] * c[j] for j in range(m))
        out[deg] = (lim, res)
    return out


def main():
    T0 = time.time()
    pts, _ = F.load_points()
    allD = sorted(pts)
    # consistency check of the normal-equation solve against dd_fits_mp.fit (QR) on one case
    Ds = [d for d in allD if 10 <= d <= 45]
    ev, res, _ = F.fit(pts, Ds, "1/D", 16)
    chk = fits_all_degrees(pts, Ds, "1/D")[16]
    print(f"check vs dd_fits_mp.fit (1/D, degree 16, 10 <= D <= 45): limit differs by {mp.nstr(abs(chk[0] - F.limit(ev)), 2)}, "
          f"max residual {mp.nstr(chk[1], 3)} vs {mp.nstr(res, 3)}", flush=True)
    rows = []
    for dmax in (45, 60.1, 61.03):
        for dmin in (10, 12, 15):
            base = [d for d in allD if dmin <= d <= dmax]
            for clean in (True, False):
                Ds = F.clean(pts, base, verbose=False) if clean else base
                for model in ("1/D", "1/n"):
                    r = fits_all_degrees(pts, Ds, model)
                    floor = min(v[1] for v in r.values())
                    for deg, (lim, res) in r.items():
                        rows.append((dmax, dmin, clean, model, deg, lim, res, res <= 2.5 * floor))
                    print(f"D {dmin:>2}..{dmax:<5}  {'clean' if clean else 'all  '}  {model:3s}  N {len(Ds):3d}  "
                          + "  ".join(f"{deg}:{mp.nstr(lim, 10)}{'*' if res <= 2.5 * floor else ' '}" for deg, (lim, res) in r.items())
                          + f"   floor {mp.nstr(floor, 2)}   ({time.time() - T0:.0f} s)", flush=True)
    print("\n* = reaches the noise floor (max residual within 2.5x of the group's best)")
    print("\nSpread of the limits that reach the noise floor:")
    for label, sel in (("all fits", lambda r: True),
                       ("D <= 45 (fit range of the held-out test)", lambda r: r[0] == 45),
                       ("D <= 60.1 snapshot", lambda r: r[0] == 60.1),
                       ("D <= 61.03 (all data)", lambda r: r[0] == 61.03),
                       ("D <= 61.03, cleaned", lambda r: r[0] == 61.03 and r[2]),
                       ("D <= 61.03, cleaned, D >= 10", lambda r: r[0] == 61.03 and r[2] and r[1] == 10)):
        L = [r[5] for r in rows if r[7] and sel(r)]
        if L:
            print(f"  {label:42s} {len(L):3d} fits: {mp.nstr(min(L), 10)} .. {mp.nstr(max(L), 10)}   "
                  f"spread {mp.nstr(max(L) - min(L), 2)}, centre {mp.nstr((max(L) + min(L)) / 2, 10)}")
    print(f"\n({time.time() - T0:.0f} s)")


if __name__ == "__main__":
    main()
