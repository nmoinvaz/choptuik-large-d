#!/usr/bin/env python3
"""#1098: the type of Delta(D)'s nearest singularity in e = 1/(D - 2), from the ladder records and the low-D rows
together (sing_data.py's file), each method run first on planted truths at the same D and sigma (sing_core.Truth).

    cd hp && python3 sing_types.py profile DATA [--truth=KIND --seed=N] [--Dmin=36] [--deg=Lp,Lr,M] [--types=...]
    cd hp && python3 sing_types.py ratscan DATA [--truth=KIND --seed=N] [--Dmin=36] [--Ms=2,3,4,5,6]

profile: typed forms f = (P(e) + S(u) R(e)) / Q(e), u = 1 + e / r, S = u^alpha (alpha on a grid) or log u or u log u,
P, R, Q of degrees Lp, Lr, M (Q(0) = 1), fitted by weighted least squares (sing_core.Fit) with r profiled (a grid in
log r, 0.03 to 0.2, then golden-section refinement); beside them the rational fit with the same number of parameters,
[Lp + Lr + 2 / M]. Reported per type: the least chi^2 and its r. The reading: a type whose singular factor the data
need gives a chi^2 far below the rational fit's at the same count, with r stable; for a pole truth every typed form
does no better than the rational one (R absorbs nothing the extra pole cannot).
families: for each parameter count n (--n=LO:HI), the least chi^2 of the rational family (over [L/M] with L + M + 1 = n,
M <= 8) and of each typed family (over (Lp, Lr, M) with Lp + Lr + M + 3 = n, Lr <= 2, M <= 5, r profiled), with
acceptance as for ratscan; the reading is the family that is accepted at the smallest n, and its r.
Data: the ladder, qd and dd rows (--kinds; chain_fit is left out by default: it departs smoothly from the fits to the
others by up to 40% of its err, e.g. -8.3e-14 at D 90, so it serves as a check), D >= --Dmin, sigma_eff.
reach: the accepted rational fits on D >= --Dmin (two per M past acceptance, M in --Ms), evaluated at every row of
every kind down to D 8.25; per D band, the median over the fits of data - fit at the band's worst point, and the
fits' range there (their spread, an estimator range, not a bound), beside the rows' sigma.
ratscan: rational fits [L/M], L up to acceptance (chi^2 <= the 0.99 quantile, |residual| <= 4 sigma) and two degrees
past it, with every pole, its residue and every zero within |e| < 0.4.
"""
import json
import math
import sys
import time

import mpmath as mp
from scipy.stats import chi2 as CHI2

import sing_core as S

args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if
            a.startswith("--"))
REF = mp.mpf("1.6438937451253575137")
THIN = {"dd": int(args.get("thin_dd", "80")), "qd": int(args.get("thin_qd", "12")),
        "chain_fit": int(args.get("thin_cf", "20"))}
TYPES = {"pole": -1.0, "isqrt": -0.5, "icbrt": -1 / 3, "cbrt": 1 / 3, "sqrt": 0.5, "c23": 2 / 3, "s32": 1.5,
         "s52": 2.5, "log": "log", "ulog": "ulog", "pole2": -2.0}


def sfun(kind, r):
    a = TYPES[kind]
    if a == "log":
        return lambda e: mp.log(1 + e / r)
    if a == "ulog":
        return lambda e: (1 + e / r) * mp.log(1 + e / r)
    return lambda e: (1 + e / r) ** a


def data():
    kinds = tuple(args.get("kinds", "ladder,qd,dd").split(","))
    pts = S.load(sys.argv[2], kinds=kinds, Dmin=float(args.get("Dmin", "36")), Dmax=float(args.get("Dmax", "1e9")),
                 thin=THIN, drop_runs=tuple(r for r in args.get("drop_runs", "").split(",") if r))
    if "truth" in args:
        tr = S.Truth(args["truth"], e0=mp.mpf(args.get("e0", "-0.061")), frac=mp.mpf(args.get("frac", "1")))
        pts = S.planted(pts, tr, int(args.get("seed", "1")), args.get("pattern", "gauss"))
    return pts


def accepted(f):
    return f.ok and f.dof >= 2 and f.chi2 <= CHI2.ppf(0.99, f.dof) and f.maxres <= 4


def profile_r(pts, Lp, Lr, M, kind, lo=0.03, hi=0.2, ngrid=12, nref=10):
    """least chi^2 over r of the typed form; returns (chi2, r, fit)"""
    cache = {}

    def c2(lr):
        if lr not in cache:
            r = mp.e ** lr
            f = S.Fit(pts, Lp, M, ref=REF, Lr=Lr, S=sfun(kind, r))
            cache[lr] = (f.chi2 if f.ok else float("inf"), float(r), f)
        return cache[lr][0]
    grid = [math.log(lo) + (math.log(hi) - math.log(lo)) * i / (ngrid - 1) for i in range(ngrid)]
    vals = [c2(g) for g in grid]
    j = min(range(ngrid), key=lambda i: vals[i])
    a_, b_ = grid[max(j - 1, 0)], grid[min(j + 1, ngrid - 1)]
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
    for _ in range(nref):
        if c2(x1) < c2(x2):
            b_, x2 = x2, x1
            x1 = b_ - g * (b_ - a_)
        else:
            a_, x1 = x1, x2
            x2 = a_ + g * (b_ - a_)
    best = min(cache, key=lambda k: cache[k][0])
    return cache[best]


def cmd_profile():
    pts = data()
    Lp, Lr, M = (int(v) for v in args.get("deg", "6,3,2").split(","))
    kinds = args.get("types", ",".join(TYPES)).split(",")
    print("%s: %d points (%s), e to %.4f; typed (%d, %d, %d) against rational [%d/%d]" % (
        args.get("truth", "real"), len(pts), ", ".join("%s %d" % (k, sum(1 for p in pts if p["kind"] == k))
                                                       for k in ("ladder", "qd", "dd", "chain_fit")),
        float(max(p["e"] for p in pts)), Lp, Lr, M, Lp + Lr + 2, M), flush=True)
    t0 = time.time()
    fr = S.Fit(pts, Lp + Lr + 2, M, ref=REF)
    pr = sorted((p for p in (fr.poles() or []) if abs(p) < 0.3), key=abs)
    out = dict(truth=args.get("truth", "real"), seed=args.get("seed"), deg=[Lp, Lr, M], N=len(pts),
               rational=dict(chi2=fr.chi2, dof=fr.dof, poles=[[p.real, p.imag] for p in pr]), types={})
    print("  rational [%d/%d]: chi2 %.4g (dof %d), poles %s" % (Lp + Lr + 2, M, fr.chi2, fr.dof, ", ".join(
        "%+.4f%+.4fi" % (p.real, p.imag) for p in pr)), flush=True)
    for k in kinds:
        c2, r, f = profile_r(pts, Lp, Lr, M, k)
        A = f.amp(-mp.mpf(r))
        pk = sorted((p for p in (f.poles() or []) if abs(p) < 0.3), key=abs)
        out["types"][k] = dict(chi2=c2, r=r, dof=f.dof - 1, amp=float(A), poles=[[p.real, p.imag] for p in pk])
        print("  %-6s chi2 %.4g (dof %d) at e0 %+.5f, amplitude R/Q(e0) %+.3e, poles %s [%.0fs]" % (
            k, c2, f.dof - 1, -r, float(A), ", ".join("%+.4f%+.4fi" % (p.real, p.imag) for p in pk),
            time.time() - t0), flush=True)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


def cmd_families():
    pts = data()
    lo, hi = (int(v) for v in args.get("n", "9:17").split(":"))
    kinds = args.get("types", "sqrt,isqrt,log,cbrt,s32").split(",")
    print("%s (seed %s): %d points, e to %.4f; families %s" % (args.get("truth", "real"), args.get("seed", "-"),
                                                             len(pts), float(max(p["e"] for p in pts)),
                                                             ", ".join(["rat"] + kinds)), flush=True)
    t0 = time.time()
    out = dict(truth=args.get("truth", "real"), seed=args.get("seed"), N=len(pts), rows=[])
    first = {}
    for n in range(lo, hi + 1):
        row = dict(n=n)
        best = None
        for M in range(1, 9):
            L = n - 1 - M
            if L < 1:
                continue
            f = S.Fit(pts, L, M, ref=REF)
            if f.ok and (best is None or f.chi2 < best[0]):
                best = (f.chi2, "[%d/%d]" % (L, M), f.dof, accepted(f), None)
        row["rat"] = best
        for k in kinds:
            bt = None
            for Lr in range(0, 3):
                for M in range(0, 6):
                    Lp = n - 3 - Lr - M
                    if Lp < 1:
                        continue
                    c2, r, f = profile_r(pts, Lp, Lr, M, k, ngrid=8, nref=8)
                    if f.ok and (bt is None or c2 < bt[0]):
                        bt = (c2, "(%d,%d,%d)" % (Lp, Lr, M), f.dof - 1, f.ok and f.dof - 1 >= 2 and
                              c2 <= CHI2.ppf(0.99, f.dof - 1) and f.maxres <= 4, -r)
            row[k] = bt
        for k in ["rat"] + kinds:
            if row[k] and row[k][3] and k not in first:
                first[k] = (n, row[k][4])
        out["rows"].append(row)
        print("n %2d " % n + " | ".join("%s %.3g %s%s%s" % (k, row[k][0], row[k][1], "*" if row[k][3] else "",
                                                             "" if row[k][4] is None else " e0 %+.4f" % row[k][4])
                                         for k in ["rat"] + kinds if row[k]) + " [%.0fs]" % (time.time() - t0),
              flush=True)
        if len(first) == len(kinds) + 1:
            break
    out["first"] = first
    order = sorted(first.items(), key=lambda kv: kv[1][0])
    print("first accepted n: " + ", ".join("%s %d%s" % (k, v[0], "" if v[1] is None else " (e0 %+.4f)" % v[1])
                                           for k, v in order))
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


def accepted_fits(pts, Ms, extra):
    fits = []
    for M in Ms:
        nacc = 0
        for L in range(max(2, M - 2), 45):
            f = S.Fit(pts, L, M, ref=REF)
            if not f.ok or f.dof < 2:
                continue
            if accepted(f):
                fits.append(((L, M), f))
                nacc += 1
            if nacc >= extra:
                break
    return fits


def cmd_reach():
    pts = data()
    Ms = [int(v) for v in args.get("Ms", "2,3,4,5,6").split(",")]
    fits = accepted_fits(pts, Ms, int(args.get("extra", "2")))
    print("%s: fits on %d points, D >= %s: %d accepted (%s)" % (args.get("truth", "real"), len(pts),
                                                               args.get("Dmin", "36"), len(fits),
                                                               ", ".join("[%d/%d]" % k for k, _ in fits)))
    allp = S.load(sys.argv[2], Dmin=8.0)
    if "truth" in args:
        tr = S.Truth(args["truth"], e0=mp.mpf(args.get("e0", "-0.061")), frac=mp.mpf(args.get("frac", "1")))
        allp = S.planted(allp, tr, int(args.get("seed", "1")) + 1000)
    bands = [(8, 12), (12, 16), (16, 20), (20, 25), (25, 30), (30, 36), (36, 45), (45, 60), (60, 74), (74, 100),
             (100, 150), (150, 200), (200, 250), (250, 9000)]
    out = []
    for kind in ("dd", "qd", "chain_fit", "ladder"):
        for lo, hi in bands:
            sel = [p for p in allp if p["kind"] == kind and lo <= float(p["D"]) < hi]
            if not sel:
                continue
            devs = []
            for p in sel:
                v = sorted(float(p["y"] - f(p["e"])) for _, f in fits)
                devs.append((v[len(v) // 2], v[-1] - v[0], p))
            w = max(devs, key=lambda d: abs(d[0]))
            ratio = max(abs(d[0]) / d[2]["sigma"] for d in devs)
            out.append(dict(kind=kind, band=[lo, hi], n=len(sel), D=float(w[2]["D"]), dev=w[0], range=w[1],
                            sigma=w[2]["sigma"], max_dev_over_sigma=ratio))
            print("  %-9s D %4d-%4d: %4d rows; worst at D %8.3f: median data - fit %+.2e, fits' range %.1e, "
                  "sigma %.1e; max |median|/sigma %.2g" % (kind, lo, hi, len(sel), float(w[2]["D"]), w[0], w[1],
                                                           w[2]["sigma"], ratio), flush=True)
    if "out" in args:
        json.dump(dict(truth=args.get("truth", "real"), Dmin=args.get("Dmin", "36"),
                       fits=["[%d/%d]" % k for k, _ in fits], bands=out), open(args["out"], "w"), indent=1)


def cmd_ratscan():
    pts = data()
    Ms = [int(v) for v in args.get("Ms", "2,3,4,5,6").split(",")]
    extra = int(args.get("extra", "2"))
    print("%s: %d points, e to %.4f" % (args.get("truth", "real"), len(pts), float(max(p["e"] for p in pts))))
    rows = []
    for M in Ms:
        nacc = 0
        for L in range(max(2, M - 2), 45):
            f = S.Fit(pts, L, M, ref=REF)
            if not f.ok or f.dof < 2:
                continue
            acc = accepted(f)
            if not acc and nacc == 0 and f.chi2 > 1e3 * f.dof:
                continue
            rr = f.roots()
            if rr is None:
                continue
            ps = sorted([pr for pr in rr[0] if abs(pr[0]) < 0.4], key=lambda pr: abs(pr[0]))
            zs = sorted([z for z in rr[1] if abs(z) < 0.4], key=abs)
            rows.append(dict(L=L, M=M, chi2=f.chi2, dof=f.dof, acc=acc, poles=[[p.real, p.imag, c.real, c.imag]
                                                                               for p, c in ps],
                             zeros=[[z.real, z.imag] for z in zs]))
            print("[%d/%d] chi2 %.4g dof %d %s poles %s | zeros %s" % (
                L, M, f.chi2, f.dof, "ACC" if acc else "   ", ", ".join(
                    "%+.5f%+.5fi (%+.1e)" % (p.real, p.imag, c.real) for p, c in ps), ", ".join(
                    "%+.4f%+.4fi" % (z.real, z.imag) for z in zs)), flush=True)
            if acc:
                nacc += 1
            if nacc > extra:
                break
    summary = front_summary(rows)
    print(summary["text"])
    if "out" in args:
        json.dump(dict(truth=args.get("truth", "real"), seed=args.get("seed"), rows=rows, front=summary),
                  open(args["out"], "w"), indent=1)


def front_summary(rows, rmin=1e-8):
    """the front pole of each accepted fit: the pole of least |e| on the negative real axis (|Im| < 1e-6) whose
    residue exceeds rmin in size (smaller ones are Froissart doublets), and the next such pole; their medians and
    the spreads (max - min) over the accepted fits; the front pole's relative residue spread is the pole/cut
    statistic (a pole keeps its residue; the front pole of a cut's pole string does not)"""
    fr = []
    for r in rows:
        if not r["acc"]:
            continue
        neg = sorted((p for p in r["poles"] if abs(p[1]) < 1e-6 and p[0] < 0 and abs(p[2]) > rmin),
                     key=lambda p: -p[0])
        if neg:
            fr.append((r["L"], r["M"], neg[0][0], neg[0][2], neg[1][0] if len(neg) > 1 else None))
    if not fr:
        return dict(n=0, text="front pole: no accepted fit")
    pos = sorted(f[2] for f in fr)
    res = sorted(f[3] for f in fr)
    nxt = sorted(f[4] for f in fr if f[4] is not None)
    med = lambda v: v[len(v) // 2]                                         # noqa: E731
    rs = (res[-1] - res[0]) / abs(med(res))
    out = dict(n=len(fr), pos=med(pos), pos_spread=pos[-1] - pos[0], res=med(res), res_rel_spread=rs,
               next=med(nxt) if nxt else None, fits=[list(f) for f in fr])
    out["text"] = ("front pole over %d accepted fits: e %+.5f (spread %.1e), residue %+.2e (relative spread %.2f); "
                   "next pole %s" % (len(fr), out["pos"], out["pos_spread"], out["res"], rs,
                                     "%+.4f" % out["next"] if nxt else "none"))
    return out


if __name__ == "__main__":
    {"profile": cmd_profile, "ratscan": cmd_ratscan, "families": cmd_families, "reach": cmd_reach}[sys.argv[1]]()
