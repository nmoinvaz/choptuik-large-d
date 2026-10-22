#!/usr/bin/env python3
"""Large-D model fits on the double-double values, done in mpmath so the fit itself adds no f64 rounding.

    python3 dd_fits_mp.py [Dmin] [Dmax_fit] [--clean] [--pred=D1,D2,...] [--mixed] [--scan=DIR] [--dmax=D]

Each model is a polynomial in a variable t(D); it is fitted by least squares in a Chebyshev basis on the fitted
t-range (well conditioned) at 60 digits. Reports the fit residual, the held-out error on D > Dmax_fit, the limit
(value at t = 0) and predictions. --mixed instead fits the 1/D series plus one non-analytic term (D^-1/2, D^-3/2,
ln D / D, ln D / D^2, e^(-D/4)) and reports that term's coefficient, the held-out error and the limit.

The points come from hp/data/delta_dd.csv (export_dd.py), or with --scan=DIR from the runs' DIR/scan_up*.jsonl
directly (a D that appears in several runs keeps the last value read). --clean drops isolated points that sit off
their neighbours (see clean()). --dmax=D ignores every point above D (before cleaning), so a table quoted for an
earlier snapshot of the data (e.g. --dmax=60.1, the points before the climb past 60.9) reproduces after new points
are added.
"""
import glob
import json
import os
import sys

import mpmath as mp

mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "1/D": lambda D: 1 / mp.mpf(D),
    "1/n": lambda D: 1 / (mp.mpf(D) - 2),
    "n^-1/2": lambda D: (mp.mpf(D) - 2) ** -0.5,
    "D^-1/2": lambda D: mp.mpf(D) ** -0.5,
}

EXTRA = {
    "D^-1/2": lambda D: mp.mpf(D) ** -0.5,
    "D^-3/2": lambda D: mp.mpf(D) ** -1.5,
    "ln D / D": lambda D: mp.log(D) / D,
    "ln D / D^2": lambda D: mp.log(D) / mp.mpf(D) ** 2,
    "e^-D/4": lambda D: mp.exp(-mp.mpf(D) / 4),
}


def load_points(scan=None):
    """{D rounded to 1e-6: Delta (mpf)} and {D: run name}."""
    pts, src = {}, {}
    if scan:
        for f in sorted(glob.glob(os.path.join(scan, "scan_up*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                k = round(r["D"], 6)
                pts[k] = mp.mpf(r["Delta"]) + mp.mpf(r["Delta_lo"])
                src[k] = os.path.basename(f)[5:-6]
    else:
        for line in list(open(os.path.join(HERE, "data", "delta_dd.csv")))[1:]:
            c = line.strip().split(",")
            k = round(float(c[0]), 6)
            # hi and lo are written as repr(float): parse each to the double it denotes, then add exactly (60 digits
            # hold hi + lo). mp.mpf(c[1]) would read the decimal string itself, which differs from the double by up
            # to half an ulp (2e-16); check_numbers.py (loader_exact) fails if that comes back.
            pts[k] = mp.mpf(float(c[1])) + mp.mpf(float(c[2]))
            src[k] = c[6]
    return pts, src


def cheb_rows(ts, a, b, deg):
    rows = []
    for t in ts:
        u = (2 * t - (a + b)) / (b - a)
        T = [mp.mpf(1), u]
        for _ in range(deg - 1):
            T.append(2 * u * T[-1] - T[-2])
        rows.append(T[: deg + 1])
    return mp.matrix(rows)


def fit(pts, Dl, model, deg, extra=None):
    """Least squares of Delta on Dl: a degree-deg polynomial in MODELS[model](D), plus c * extra(D) if given.
    Returns (ev, max residual, coefficient of the extra term)."""
    tf = [MODELS[model](d) for d in Dl]
    a, b = min(tf), max(tf)
    X = cheb_rows(tf, a, b, deg)
    sc = mp.mpf(1)
    if extra:
        col = [extra(d) for d in Dl]
        sc = max(abs(v) for v in col)
        X = mp.matrix([[X[i, j] for j in range(X.cols)] + [col[i] / sc] for i in range(X.rows)])
    y = mp.matrix([pts[d] for d in Dl])
    c = mp.qr_solve(X, y)[0]

    def ev(D):
        row = cheb_rows([MODELS[model](D)], a, b, deg)
        v = sum(row[0, j] * c[j] for j in range(deg + 1))
        if extra:
            v += c[deg + 1] * extra(D) / sc
        return v

    res = max(abs((X * c)[i] - y[i]) for i in range(len(Dl)))
    return ev, res, (c[deg + 1] / sc if extra else mp.mpf(0))


def limit(ev):
    return ev(mp.mpf(10) ** 80)


def clean(pts, Ds, verbose=True):
    """Drop isolated points that sit off their neighbours (junk-contaminated solves in the unfrozen runs): residual
    of a global 1/D degree-18 fit minus the median residual of the four nearest points, flagged above 1e-15."""
    ev, _, _ = fit(pts, Ds, "1/D", 18)
    r = [ev(d) - pts[d] for d in Ds]
    drop = []
    for i in range(len(Ds)):
        nb = sorted(r[j] for j in range(max(0, i - 2), min(len(Ds), i + 3)) if j != i)
        med = (nb[len(nb) // 2] + nb[(len(nb) - 1) // 2]) / 2
        if abs(r[i] - med) > 1e-15:
            drop.append((Ds[i], float(r[i] - med)))
    if verbose:
        print(f"--clean drops {len(drop)}: " + ", ".join(f"{d:.3f} ({e:+.0e})" for d, e in drop))
    dd = {d for d, _ in drop}
    return [d for d in Ds if d not in dd]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Dmin = float(args[0]) if len(args) > 0 else 10.0
    Dfit = float(args[1]) if len(args) > 1 else 1e9
    pred = [40, 45, 50, 60, 80, 100]
    for a in sys.argv[1:]:
        if a.startswith("--pred="):
            pred = [float(v) for v in a[7:].split(",")]
    scan = next((a[7:] for a in sys.argv[1:] if a.startswith("--scan=")), None)
    dmax = next((float(a[7:]) for a in sys.argv[1:] if a.startswith("--dmax=")), float("inf"))
    pts, _ = load_points(scan)
    Ds = sorted(d for d in pts if Dmin <= d <= dmax)
    if "--clean" in sys.argv:
        Ds = clean(pts, Ds)
    fitD = [d for d in Ds if d <= Dfit]
    holdD = [d for d in Ds if d > Dfit]
    print(f"{len(Ds)} DD points, D = {Ds[0]:.2f} .. {Ds[-1]:.2f}; fit on {len(fitD)} with D <= "
          f"{min(Dfit, Ds[-1]):.2f}, {len(holdD)} held out")

    def held(ev):
        if not holdD:
            return ""
        h = [ev(d) - pts[d] for d in holdD]
        return f"  held-out max {mp.nstr(max(abs(v) for v in h), 2):>8s} (at D={holdD[-1]:.1f}: {mp.nstr(h[-1], 2)})"

    if "--mixed" in sys.argv:
        for deg in (14, 16):
            for name, g in [("none", None)] + list(EXTRA.items()):
                ev, res, coef = fit(pts, fitD, "1/D", deg, g)
                print(f"1/D deg {deg} + {name:10s} coef {mp.nstr(coef, 3):>10s}  res {mp.nstr(res, 2):>7s}"
                      + held(ev) + f"  limit {mp.nstr(limit(ev), 10)}", flush=True)
        return

    tables = {}
    for model, degs in (("1/D", (12, 14, 16, 18, 20)), ("1/n", (12, 14, 16, 18, 20)),
                        ("n^-1/2", (20, 24, 28, 32)), ("D^-1/2", (20, 24, 28, 32))):
        tables[model] = {}
        for deg in degs:
            ev, res, _ = fit(pts, fitD, model, deg)
            print(f"{model:7s} deg {deg:2d}  max|res| {mp.nstr(res, 2):>8s}" + held(ev)
                  + f"  limit {mp.nstr(limit(ev), 12)}", flush=True)
            tables[model][deg] = ev
    print()
    print(f"{'model':14s} " + " ".join(f"{'D=' + format(d, 'g'):>17s}" for d in pred))
    for model, evs in tables.items():
        for deg, ev in evs.items():
            print(f"{model + ' ' + str(deg):14s} " + " ".join(f"{mp.nstr(ev(d), 14):>17s}" for d in pred))


if __name__ == "__main__":
    main()
