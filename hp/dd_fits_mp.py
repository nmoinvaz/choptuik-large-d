#!/usr/bin/env python3
"""Large-D model fits on the double-double values, done in mpmath so the fit itself adds no f64 rounding.

    python3 dd_fits_mp.py [Dmin] [Dmax_fit] [--pred D1,D2,...]

Each model is a polynomial in a variable t(D); it is fitted by least squares in a Chebyshev basis on the fitted
t-range (well conditioned) at 60 digits. Reports the fit residual, the held-out error on D > Dmax_fit, the limit
(value at t = 0) and predictions. The scan files are read from /home/claude/dss/hp_scan/scan_up*.jsonl; a point
that appears in several files (same D to 1e-6) keeps the last value read.
"""
import glob
import json
import sys

import mpmath as mp

mp.mp.dps = 60
args = [a for a in sys.argv[1:] if not a.startswith("--")]
Dmin = float(args[0]) if len(args) > 0 else 10.0
Dfit = float(args[1]) if len(args) > 1 else 1e9
pred = [40, 45, 50, 60, 80, 100]
for a in sys.argv[1:]:
    if a.startswith("--pred="):
        pred = [float(v) for v in a[7:].split(",")]

pts = {}
src = {}
for f in sorted(glob.glob("/home/claude/dss/hp_scan/scan_up*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        k = round(r["D"], 6)
        pts[k] = mp.mpf(r["Delta"]) + mp.mpf(r["Delta_lo"])
        src[k] = f.split("/")[-1]
Ds = sorted(d for d in pts if d >= Dmin)
fitD = [d for d in Ds if d <= Dfit]
holdD = [d for d in Ds if d > Dfit]
print(f"{len(Ds)} DD points, D = {Ds[0]:.2f} .. {Ds[-1]:.2f}; fit on {len(fitD)} with D <= {min(Dfit, Ds[-1]):.2f}, "
      f"{len(holdD)} held out")

MODELS = {
    "1/D": lambda D: 1 / mp.mpf(D),
    "1/n": lambda D: 1 / (mp.mpf(D) - 2),
    "n^-1/2": lambda D: (mp.mpf(D) - 2) ** -0.5,
    "D^-1/2": lambda D: mp.mpf(D) ** -0.5,
}


def cheb_rows(ts, a, b, deg):
    rows = []
    for t in ts:
        u = (2 * t - (a + b)) / (b - a)
        T = [mp.mpf(1), u]
        for _ in range(deg - 1):
            T.append(2 * u * T[-1] - T[-2])
        rows.append(T[: deg + 1])
    return mp.matrix(rows)


def fit(model, deg, Dlist=None):
    Dl = fitD if Dlist is None else Dlist
    tf = [MODELS[model](d) for d in Dl]
    a, b = min(tf), max(tf)
    X = cheb_rows(tf, a, b, deg)
    y = mp.matrix([pts[d] for d in Dl])
    c = mp.qr_solve(X, y)[0]

    def ev(D):
        return (cheb_rows([MODELS[model](D)], a, b, deg) * c)[0]

    res = max(abs((X * c)[i] - y[i]) for i in range(len(Dl)))
    return ev, res


def clean():
    """Drop isolated points that sit off their neighbours (junk-contaminated solves in the unfrozen runs): residual
    of a global 1/D degree-18 fit minus the median residual of the four nearest points, flagged above 1e-15."""
    global Ds, fitD, holdD
    ev, _ = fit("1/D", 18, Ds)
    r = [ev(d) - pts[d] for d in Ds]
    drop = []
    for i in range(len(Ds)):
        nb = sorted(r[j] for j in range(max(0, i - 2), min(len(Ds), i + 3)) if j != i)
        med = (nb[len(nb) // 2] + nb[(len(nb) - 1) // 2]) / 2
        if abs(r[i] - med) > 1e-15:
            drop.append((Ds[i], float(r[i] - med)))
    print(f"--clean drops {len(drop)}: " + ", ".join(f"{d:.3f} ({e:+.0e})" for d, e in drop))
    dd = {d for d, _ in drop}
    Ds = [d for d in Ds if d not in dd]
    fitD = [d for d in Ds if d <= Dfit]
    holdD = [d for d in Ds if d > Dfit]
    print(f"after cleaning: fit on {len(fitD)}, {len(holdD)} held out")


def run(model, degs):
    out = {}
    for deg in degs:
        ev, res = fit(model, deg)
        line = f"{model:7s} deg {deg:2d}  max|res| {mp.nstr(res, 2):>8s}"
        if holdD:
            h = [ev(d) - pts[d] for d in holdD]
            line += f"  held-out max {mp.nstr(max(abs(v) for v in h), 2):>8s} (at D={holdD[-1]:.1f}: {mp.nstr(h[-1], 2)})"
        lim = ev(1e300)
        line += f"  limit {mp.nstr(lim, 12)}"
        print(line)
        out[deg] = ev
    return out


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    tables = {}
    for model, degs in (("1/D", (12, 14, 16, 18, 20)), ("1/n", (12, 14, 16, 18, 20)),
                        ("n^-1/2", (20, 24, 28, 32)), ("D^-1/2", (20, 24, 28, 32))):
        tables[model] = run(model, degs)
    print()
    print(f"{'model':14s} " + " ".join(f"{'D=' + format(d, 'g'):>17s}" for d in pred))
    for model, evs in tables.items():
        for deg, ev in evs.items():
            print(f"{model + ' ' + str(deg):14s} " + " ".join(f"{mp.nstr(ev(d), 14):>17s}" for d in pred))
