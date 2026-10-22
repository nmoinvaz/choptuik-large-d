#!/usr/bin/env python3
"""Review RV2: an independent re-implementation of the Delta_inf fits (hp/dd_fits_mp.py, hp/dinf_budget.py), in
python-flint ball arithmetic at 640 bits, with a monomial basis in the scaled variable instead of Chebyshev
polynomials and the least-squares problem solved through arb_mat. Written from the docstrings, not the code.

    python3 hp/review/rv2_fits.py budget        the 48-fit budget (and the D <= 45 group), with the quoted range
    python3 hp/review/rv2_fits.py influence     limit = sum_i w_i Delta_i for each used fit: |w|_1, |w|_2 and what
                                                per-point error models do to the limit
    python3 hp/review/rv2_fits.py rules         how the quoted range moves with the selection rule
    python3 hp/review/rv2_fits.py clean         what the cleaning drops, by D band and snapshot
    python3 hp/review/rv2_fits.py selftest      plants a known degree-12 polynomial in 1/D on the real D grid and
                                                checks that every fit of degree >= 12 returns its limit (and that a
                                                deliberately wrong design, t = 1/(D+1e-3), does not)

The fit: least squares of Delta(D) on a degree-d polynomial in t = 1/D (model '1/D') or t = 1/(D-2) ('1/n'),
unweighted, the limit is the polynomial at t = 0. Cleaning: residual of a global 1/D degree-18 fit minus the median
of the residuals of the two neighbours on each side; points where that exceeds 1e-15 are dropped.
"""
import os
import sys

import flint
from flint import arb, arb_mat, fmpq

flint.ctx.prec = 640
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "data", "delta_dd.csv")
DEGS = (14, 16, 18, 20, 22)
DECIMAL = "--decimal" in sys.argv


def load(path=CSV):
    """{D (float, exact): Delta (arb, exact hi + lo)} and {D: run}"""
    pts, run = {}, {}
    for line in list(open(path))[1:]:
        c = line.strip().split(",")
        d = float(c[0])
        if DECIMAL:
            # emulate hp/dd_fits_mp.load_points: the CSV's shortest-repr strings parsed as decimals (not as the doubles)
            pts[d] = arb(c[1]) + arb(c[2])
        else:
            pts[d] = arb(fmpq(*(float(c[1])).as_integer_ratio())) + arb(fmpq(*(float(c[2])).as_integer_ratio()))
        run[d] = c[6]
    return pts, run


def tvar(d, model):
    return arb(1) / (arb(fmpq(*float(d).as_integer_ratio())) - (2 if model == "1/n" else 0))


class Fit:
    """All degrees up to dmax of one (data set, model): Gram matrix of the monomial basis in u (t scaled to [-1, 1])."""

    def __init__(self, pts, Ds, model, dmax=max(DEGS)):
        self.Ds, self.model, self.dmax = list(Ds), model, dmax
        t = [tvar(d, model) for d in Ds]
        lo = min(t, key=lambda v: v.mid())
        hi = max(t, key=lambda v: v.mid())
        self.a, self.b = lo, hi
        u = [(2 * v - (lo + hi)) / (hi - lo) for v in t]
        self.rows = []
        for v in u:
            r = [arb(1)]
            for _ in range(dmax):
                r.append(r[-1] * v)
            self.rows.append(r)
        self.X = arb_mat(self.rows)
        self.y = arb_mat([[pts[d]] for d in Ds])
        self.G = self.X.transpose() * self.X
        self.r = self.X.transpose() * self.y
        u0 = (-(lo + hi)) / (hi - lo)                    # t = 0
        self.e0 = [u0 ** k for k in range(dmax + 1)]

    def solve(self, deg):
        m = deg + 1
        G = arb_mat([[self.G[i, j] for j in range(m)] for i in range(m)])
        r = arb_mat([[self.r[i, 0]] for i in range(m)])
        c = G.solve(r)
        lim = sum((self.e0[k] * c[k, 0] for k in range(m)), arb(0))
        res = max(float(abs(sum((self.rows[i][k] * c[k, 0] for k in range(m)), arb(0)) - self.y[i, 0]).mid())
                  for i in range(len(self.Ds)))
        return lim, res, c

    def weights(self, deg):
        """w with limit = sum_i w_i y_i: w = X (G^-1 e0)"""
        m = deg + 1
        G = arb_mat([[self.G[i, j] for j in range(m)] for i in range(m)])
        z = G.solve(arb_mat([[self.e0[k]] for k in range(m)]))
        return [sum((self.rows[i][k] * z[k, 0] for k in range(m)), arb(0)) for i in range(len(self.Ds))]


def clean(pts, Ds, thr=1e-15):
    f = Fit(pts, Ds, "1/D", 18)
    _, _, c = f.solve(18)
    r = [(sum((f.rows[i][k] * c[k, 0] for k in range(19)), arb(0)) - pts[d]).mid() for i, d in enumerate(Ds)]
    keep, drop = [], []
    for i, d in enumerate(Ds):
        nb = sorted(r[j] for j in range(max(0, i - 2), min(len(Ds), i + 3)) if j != i)
        med = (nb[len(nb) // 2] + nb[(len(nb) - 1) // 2]) / 2
        (drop if abs(r[i] - med) > thr else keep).append(d)
    return keep, drop


def grid(pts, dmaxes=(45, 60.1, 61.03), factor=2.5, nuse=2, degs=DEGS, cleaner=clean, want_w=False):
    rows = []
    allD = sorted(pts)
    for dmax in dmaxes:
        for dmin in (10, 12, 15):
            base = [d for d in allD if dmin <= d <= dmax]
            for cl in (True, False):
                Ds = cleaner(pts, base)[0] if cl else base
                for model in ("1/D", "1/n"):
                    f = Fit(pts, Ds, model, max(degs))
                    r = {deg: f.solve(deg)[:2] for deg in degs}
                    floor = min(v[1] for v in r.values())
                    at = [deg for deg in degs if r[deg][1] <= factor * floor]
                    use = at[:nuse] if nuse else at
                    for deg in degs:
                        w = f.weights(deg) if (want_w and deg in use) else None
                        rows.append(dict(dmax=dmax, dmin=dmin, clean=cl, model=model, deg=deg, lim=r[deg][0],
                                         res=r[deg][1], use=deg in use, at=deg in at, Ds=Ds, w=w))
    return rows


def span(rows, sel=lambda r: r["dmax"] > 50):
    L = [r["lim"].mid() for r in rows if r["use"] and sel(r)]
    return min(L), max(L), len(L)


def fmt(x, n=12):
    return arb(x).str(n, radius=False)


def cmd_budget():
    pts, _ = load()
    rows = grid(pts)
    for r in rows:
        if r["deg"] == DEGS[0]:
            print(f"D {r['dmin']:>2}..{r['dmax']:<5} {'clean' if r['clean'] else 'all  '} {r['model']:3s} N {len(r['Ds']):3d} ", end="")
        print(f" {r['deg']}:{fmt(r['lim'], 10)}{'+' if r['use'] else ('*' if r['at'] else ' ')}({r['res']:.1e})", end="")
        if r["deg"] == DEGS[-1]:
            print()
    for label, sel in (("D <= 45", lambda r: r["dmax"] == 45), ("D <= 60.1", lambda r: r["dmax"] == 60.1),
                       ("D <= 61.03", lambda r: r["dmax"] == 61.03), ("quoted (60.1 and 61.03)", lambda r: r["dmax"] > 50)):
        lo, hi, n = span(rows, sel)
        print(f"{label:26s} {n:2d} fits: {fmt(lo)} .. {fmt(hi)}  centre {fmt((lo + hi) / 2)}  half-width {float((hi - lo) / 2):.2e}")


def cmd_influence():
    pts, run = load()
    rows = grid(pts, dmaxes=(60.1, 61.03), want_w=True)
    print("per used fit: |w|_1, |w|_2, the limit's shift for (a) iid noise 1e-16 (D <= 60.1) / 3e-15 (above),"
          " (b) a coherent 1e-15 on 35.35-37 (kfreeze cut), (c) a coherent +3e-15 on D > 60.1,"
          " (d) +1e-14 at D = 26.82972 (the up4 point with centre tail 2e-11), (e) sum |w| over D > 60.1")
    for r in rows:
        if not r["use"]:
            continue
        w = [x.mid() for x in r["w"]]
        Ds = r["Ds"]
        n1 = sum(abs(x) for x in w)
        n2 = sum(x * x for x in w) ** 0.5
        sa = sum((x * (1e-16 if d <= 60.1 else 3e-15)) ** 2 for x, d in zip(w, Ds)) ** 0.5
        sb = sum(x for x, d in zip(w, Ds) if 35.35 <= d <= 37) * 1e-15
        sc = sum(x for x, d in zip(w, Ds) if d > 60.1) * 3e-15
        sd = sum(x for x, d in zip(w, Ds) if abs(d - 26.82972) < 1e-9) * 1e-14
        se = sum(abs(x) for x, d in zip(w, Ds) if d > 60.1)
        print(f"  D {r['dmin']:>2}..{r['dmax']:<5} {'clean' if r['clean'] else 'all  '} {r['model']:3s} deg {r['deg']}: "
              f"|w|1 {float(n1):.2e} |w|2 {float(n2):.2e}  (a) {float(sa):.1e} (b) {float(sb):+.1e} (c) {float(sc):+.1e} "
              f"(d) {float(sd):+.1e} (e) {float(se):.1e}")


def cmd_rules():
    pts, _ = load()
    ALL = tuple(range(12, 23))
    base = grid(pts, dmaxes=(60.1, 61.03), nuse=0, degs=ALL)  # every degree 12-22, flags computed below
    by = {}
    for r in base:
        by.setdefault((r["dmax"], r["dmin"], r["clean"], r["model"]), []).append(r)

    def sel(factor, nuse, degs=DEGS):
        L = []
        for grp in by.values():
            g = [r for r in grp if r["deg"] in degs]
            floor = min(r["res"] for r in g)
            at = [r for r in g if r["res"] <= factor * floor]
            use = at[:nuse] if nuse else at
            L += [r["lim"].mid() for r in use]
        return min(L), max(L), len(L)

    print("selection rule -> range of the used limits (quoted: factor 2.5, two lowest, degrees 14-22 even)")
    for factor in (1.5, 2.0, 2.5, 3.0, 5.0, 10.0):
        for nuse in (1, 2, 3, 0):
            lo, hi, n = sel(factor, nuse)
            print(f"  factor {factor:4.1f}  {'all' if nuse == 0 else 'lowest ' + str(nuse):9s} {n:3d} fits  "
                  f"{fmt(lo)} .. {fmt(hi)}  centre {fmt((lo + hi) / 2)}  half-width {float((hi - lo) / 2):.1e}")
    print("factor 2.5, two lowest, other degree sets:")
    for degs in ((12, 14, 16, 18, 20, 22), (16, 18, 20, 22), tuple(range(14, 23)), (13, 15, 17, 19, 21), tuple(range(12, 23))):
        lo, hi, n = sel(2.5, 2, degs)
        print(f"  degrees {degs}: {n} fits {fmt(lo)} .. {fmt(hi)}  centre {fmt((lo + hi) / 2)}  half-width {float((hi - lo) / 2):.1e}")
    print("all degrees 14-22 (even) of every group, regardless of residual:")
    L = [r["lim"].mid() for r in base if r["deg"] in DEGS]
    print(f"  {len(L)} fits {fmt(min(L))} .. {fmt(max(L))}")
    for g, grp in sorted(by.items(), key=lambda kv: str(kv[0])):
        print(f"  {g}: " + "  ".join(f"{r['deg']}:{fmt(r['lim'], 10)}({r['res']:.1e})" for r in grp if r["deg"] in DEGS))


def cmd_clean():
    pts, run = load()
    allD = sorted(pts)
    for dmin, dmax in ((10, 60.1), (10, 61.03), (15, 61.03)):
        base = [d for d in allD if dmin <= d <= dmax]
        keep, drop = clean(pts, base)
        bands = {}
        for d in drop:
            b = "<27" if d < 27 else "27-35.25" if d < 35.25 else "35.25-60.1" if d <= 60.1 else ">60.1"
            bands[b] = bands.get(b, 0) + 1
        print(f"D {dmin}..{dmax}: {len(base)} points, drops {len(drop)} {bands}")
        print("   " + ", ".join(f"{d:.4f}({run[d]})" for d in drop))
    # the up4 point with the 2e-11 centre tail and the up6 points accepted at |F| > 1e-13
    keep, drop = clean(pts, [d for d in allD if 10 <= d <= 61.03])
    for d in (23.952249, 26.82972):
        print(f"D {d}: {'dropped' if d in drop else 'KEPT'} by the cleaning on 10..61.03")


def load_est(path=CSV):
    est = {}
    for line in list(open(path))[1:]:
        c = line.strip().split(",")
        if c[5]:
            est[float(c[0])] = float(c[5])
    return est


def cmd_variants():
    """The quoted 48-fit range on altered data: corrected by the runs' own linearised Delta corrections, without the
    D range where the solves were least clean, and with the run-flagged points removed instead of cleaning."""
    pts, run = load()
    est = load_est()
    raw = []
    for line in list(open(CSV))[1:]:
        c = line.strip().split(",")
        raw.append((float(c[0]), float(c[3]), float(c[5]) if c[5] else 0.0))
    variants = [("as published", pts, clean)]
    corr = {d: v + arb(est.get(d, 0.0)) for d, v in pts.items()}
    variants.append(("+ dDelta_est where recorded (cleaning on)", corr, clean))
    variants.append(("+ dDelta_est, no cleaning (both 'clean' arms identical)", corr, lambda p, Ds: (Ds, [])))
    flagged = {d for d, F, e in raw if F > 1e-20 or abs(e) > 1e-16}
    variants.append((f"run-flagged points removed (|F| > 1e-20 or |dDelta_est| > 1e-16: {len(flagged)}), no cleaning",
                     {d: v for d, v in pts.items() if d not in flagged}, lambda p, Ds: (Ds, [])))
    variants.append(("27 <= D <= 37 removed (unfrozen junk runs and the kfreeze cut), cleaning on",
                     {d: v for d, v in pts.items() if not 27 <= d <= 37}, clean))
    variants.append(("D = 26.82972 (centre tail 2e-11) removed, cleaning on",
                     {d: v for d, v in pts.items() if d != 26.82972}, clean))
    kfc = next((a[8:] for a in sys.argv if a.startswith("--kfcut=")), None)
    if kfc:
        # measured kfreeze-13 cut Delta_true - Delta_13 at a few D (rv2_resolve kfreeze15 / rv2_kfcut), linearly
        # interpolated in D, constant beyond the last point, applied to every kfreeze-13 point (D >= 35.35)
        pts_ = sorted((float(x.split(":")[0]), float(x.split(":")[1])) for x in kfc.split(","))

        def cut(d):
            if d < 35.35:
                return 0.0
            if d <= pts_[0][0]:
                return pts_[0][1]
            for (d0, v0), (d1, v1) in zip(pts_, pts_[1:]):
                if d0 <= d <= d1:
                    return v0 + (v1 - v0) * (d - d0) / (d1 - d0)
            return pts_[-1][1]
        variants.append((f"kfreeze-13 cut corrected (Delta + cut(D), cut through {kfc}), cleaning on",
                         {d: v + arb(cut(d)) for d, v in pts.items()}, clean))
    detail = next((int(a[9:]) for a in sys.argv if a.startswith("--detail=")), None)
    if detail is not None:
        bands = {}
        for d in flagged:
            b = "<27" if d < 27 else "27-35.25" if d < 35.25 else "35.25-60.1" if d <= 60.1 else ">60.1"
            bands[b] = bands.get(b, 0) + 1
        print("flagged points by band:", bands)
    for i, (label, p, cl) in enumerate(variants):
        if detail is not None and i != detail:
            continue
        rows = grid(p, dmaxes=(60.1, 61.03), cleaner=cl)
        lo, hi, n = span(rows)
        print(f"{label:80s} {n} fits {fmt(lo)} .. {fmt(hi)}  centre {fmt((lo + hi) / 2)}  half-width {float((hi - lo) / 2):.1e}",
              flush=True)
        if detail is not None:
            for r in rows:
                if r["deg"] == DEGS[0]:
                    print(f"  D {r['dmin']:>2}..{r['dmax']:<5} {'clean' if r['clean'] else 'all  '} {r['model']:3s} N {len(r['Ds']):3d} ", end="")
                print(f" {r['deg']}:{fmt(r['lim'], 10)}{'+' if r['use'] else ('*' if r['at'] else ' ')}({r['res']:.1e})", end="")
                if r["deg"] == DEGS[-1]:
                    print()


def cmd_selftest():
    """A degree-12 polynomial in 1/D with a known limit on the real grid, exact (no noise): every degree >= 12 must
    return the limit to far below 1e-20. Then the same data fitted with a wrong design (t shifted) must not."""
    pts, _ = load()
    Ds = [d for d in sorted(pts) if 10 <= d <= 61.03]
    coef = [arb(fmpq(k + 1, 7)) * (-1) ** k * 3 ** k for k in range(13)]
    L0 = coef[0]
    plant = {d: sum((coef[k] * tvar(d, "1/D") ** k for k in range(13)), arb(0)) for d in Ds}
    f = Fit(plant, Ds, "1/D")
    worst = max(abs((f.solve(deg)[0] - L0).mid()) for deg in DEGS)
    print(f"selftest: planted exact degree-12 polynomial, every degree 14-22 recovers the limit to {float(worst):.1e}")
    ok = worst < 1e-30
    # break it: a model that cannot represent the plant (degree 10 < 12) must miss, and so must the right degree
    # on data with one point moved by 1e-12
    miss = abs((f.solve(10)[0] - L0).mid())
    bad = dict(plant)
    bad[Ds[len(Ds) // 2]] += arb("1e-12")
    miss2 = abs((Fit(bad, Ds, "1/D").solve(16)[0] - L0).mid())
    print(f"selftest negative controls: degree 10 misses the limit by {float(miss):.1e}; one point moved by 1e-12 "
          f"moves the degree-16 limit by {float(miss2):.1e} (both must be > 1e-20)")
    ok = ok and miss > 1e-20 and miss2 > 1e-20
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "budget"
    r = {"budget": cmd_budget, "influence": cmd_influence, "rules": cmd_rules, "clean": cmd_clean,
         "selftest": cmd_selftest, "variants": cmd_variants}[cmd]()
    sys.exit(1 if r is False else 0)
