#!/usr/bin/env python3
"""Does the data want an "instanton" term? The 1/(D-1) series plus c eps^p exp(-a/eps), eps = 1/(D - 1).

    python3 transseries_fit.py [--Dmin=10] [--Dfit=45] [--degs=10,12,14,16,18] [--as=0.5,1,1.5,2] [--ps=0,1]
                               [--planted] [--noise=1e-17]

If Delta(D) is a transseries rather than a convergent series in eps, its first non-perturbative piece is
c eps^p exp(-a/eps) = c eps^p exp(-a (D - 1)) (a = 1 is the guess of a one-instanton sector). At D = 10 that is about
1e-4 c for a = 1, at D = 30 about 2.5e-13 c, so the double-double points (D >= 8.25, about 1e-17 apart from each other
in noise) constrain it most strongly at small D.

For each polynomial degree K in eps and each (a, p): the least-squares fit on Dmin <= D <= Dfit (60 digits, Chebyshev
basis, dd_fits_mp.fit) with and without the term; the coefficient c; the largest fit residual; the largest error on the
held-out points D > Dfit; and the limit eps -> 0 (Delta_inf). A term the data need lowers the held-out error and has a
stable c across K; a term they do not need gets a c that wanders with K and leaves the held-out error unchanged.

--planted: the planted-truth check of the procedure first. Synthetic data at the same D: the degree-18 series fitted to
all points (no instanton), plus c0 eps^p exp(-a/eps) with known c0, plus Gaussian noise of --noise (absolute); the same
fits must recover c0 when it is above the detection limit, and the held-out error must show its absence when the term
is left out. Exit 1 if a planted term at c0 >= 1 is not recovered to 10% at the right (a, p).
"""
import random
import sys

import mpmath as mp

import dd_fits_mp as F

mp.mp.dps = 60
F.MODELS["1/(D-1)"] = lambda D: 1 / (mp.mpf(D) - 1)
MODEL = "1/(D-1)"


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def inst(a, p):
    return lambda D: (1 / (mp.mpf(D) - 1)) ** p * mp.exp(-mp.mpf(a) * (mp.mpf(D) - 1))


def run(pts, Ds, Dmin, Dfit, degs, avals, pvals, out=print):
    fitD = [d for d in Ds if Dmin <= d <= Dfit]
    hold = [d for d in Ds if d > Dfit]

    def held(ev):
        return max(abs(ev(d) - pts[d]) for d in hold) if hold else mp.mpf(0)

    rows = []
    for K in degs:
        ev0, res0, _ = F.fit(pts, fitD, MODEL, K)
        lim0 = F.limit(ev0)
        out(f"K {K:2d}  series alone            : max|res| {mp.nstr(res0, 2):>8s}  held-out {mp.nstr(held(ev0), 2):>8s}"
            f"  Delta_inf {mp.nstr(lim0, 13)}")
        for a in avals:
            for p in pvals:
                ev, res, c = F.fit(pts, fitD, MODEL, K, extra=inst(a, p))
                rows.append((K, a, p, c, res, held(ev), F.limit(ev) - lim0))
                out(f"K {K:2d}  + c eps^{p} e^(-{a}/eps)  : c {mp.nstr(c, 3):>10s}  max|res| {mp.nstr(res, 2):>8s}"
                    f"  held-out {mp.nstr(held(ev), 2):>8s}  Delta_inf shift {mp.nstr(F.limit(ev) - lim0, 2)}")
    return rows


def main():
    Dmin = float(opt("Dmin", 10))
    Dfit = float(opt("Dfit", 45))
    degs = [int(v) for v in opt("degs", "10,12,14,16,18").split(",")]
    avals = [float(v) for v in opt("as", "0.5,1,1.5,2").split(",")]
    pvals = [int(v) for v in opt("ps", "0,1").split(",")]
    pts, _ = F.load_points()
    Ds = F.clean(pts, sorted(pts))
    print(f"{len(Ds)} points, D = {Ds[0]:.2f} .. {Ds[-1]:.2f}; fit on {Dmin} <= D <= {Dfit}, held out D > {Dfit}; "
          f"eps = 1/(D - 1)")
    bad = 0
    if "--planted" in sys.argv:
        noise = float(opt("noise", 1e-17))
        base, _, _ = F.fit(pts, Ds, MODEL, 18)
        rng = random.Random(1)
        for a, p in ((1.0, 0), (0.5, 1)):
            for c0 in (0, 1e-3, 1, 100):
                syn = {d: base(d) + c0 * inst(a, p)(d) + mp.mpf(rng.gauss(0, noise)) for d in Ds}
                print(f"\nplanted c0 = {c0} at a = {a}, p = {p} (noise {noise:g}):")
                rows = run(syn, Ds, Dmin, Dfit, [14, 18], [a], [p], out=lambda s: print("  " + s))
                if c0 >= 1:
                    for K, _a, _p, c, _res, _h, _s in rows:
                        if abs(c - c0) > 0.1 * c0:
                            bad += 1
                            print(f"  NOT RECOVERED at K {K}: c {mp.nstr(c, 4)} against {c0}")
        print("\nplanted truth: " + ("recovered" if not bad else f"{bad} FAILURES") + "\n\nreal data:")
    run(pts, Ds, Dmin, Dfit, degs, avals, pvals)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
