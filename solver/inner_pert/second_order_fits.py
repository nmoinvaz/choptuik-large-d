"""The fitted second coefficient of the 1/n series of Delta, and the fit of Delta_inf with Delta_1 and Delta_2 held
fixed (notes/SECOND_ORDER.md, section 5). Same points, cleaning and least squares as first_order_fits.py:
the cleaned double-double points (hp/data/delta_dd.csv, hp/dd_fits_mp.py --clean), polynomials of degree 12-18 in
t = 1/n or 1/D on D >= 10, 12, 16, at 60 digits.

    Delta = Delta_inf + Delta_1/n + Delta_2/n^2 + ...
The 1/D fits give the 1/n coefficients through 1/n = 1/D + 2/D^2 + ...: Delta_2 = Delta_2^(D) - 2 Delta_1. (EEG's
eps = 1/(D - 1) = 1/(n + 1): Delta_2^(eps) = Delta_2 + Delta_1.)

Fixed fits: Delta(D) - Delta_1/n - Delta_2/n^2 fitted by 1, t^3, ..., t^deg (t = 1/n), giving Delta_inf.

Planted truth (--planted), run before the real data, with the criteria fixed before the first run:
  P1  the estimator of Delta_2: on the README's rational formula in 1/n (known Taylor coefficients) at the same D,
      noise 1e-14 (as first_order_fits.py --planted), every fit with max|res| <= 1e-13 recovers Delta_2 within 2e-3.
  P2  the fixed fit: with the planted Delta_1, Delta_2 held fixed, every fixed fit with max|res| <= 1e-13 recovers
      Delta_inf within 1.5e-8 (the README's bar), and the spread of those Delta_inf is not larger than the spread of
      the free fits on the same data.
  P3  control, must fire: with Delta_1 held 0.027 too high (the M = 5 derivation's distance from the fit), every fixed
      fit misses the residual criterion (max|res| > 1e-13) or moves Delta_inf by more than 1.5e-8.
  P4  lambda_2: on 1/gamma from the README's rational gamma(D) at the double-double growth-rate points (as
      first_order_fits.py --planted, noise 1e-12), the estimator's fits (LAM_DEGS on D >= 24, 30, lambda_inf = 2
      fixed) recover the planted 1/n^2 coefficient within 0.02 (criterion fixed before the first run of P4).
History. The first run of P1-P3 (log second_order_fits_planted_first.log in the raw folder) FAILED P1 on the two
degree-18 fits on D >= 16 (errors 4.7e-3 and 1.35e-2; every other fit within 2.7e-3 of which those meeting the
residual criterion within 4.8e-4): 19 parameters on the shortest range overfit the noise, as degree 20 did at first
order. The estimator was changed (degree 18 dropped for D >= 16, DEG_MAX16), not the threshold, and the test rerun.
The second run (second_order_fits_planted_second.log) passed P1-P3 and FAILED P4 on the degree-9 fit on D >= 24
(error 0.027; 0.011-0.015 for the other degree 9-10 fits, all of the same sign and falling with the degree, the
failing fit's max|res| 11 times the noise): the second coefficient needs one more resolved term than the first. The
estimator was changed to degrees 10-11 (LAM_DEGS), not the threshold, and the test rerun.
The real-data part (default) fits Delta_2, then holds (Delta_1, Delta_2) fixed at the fitted centre, at the derived
values of the truncation sequence given with --fixed=D1,D2[;D1,D2...], and reports Delta_inf and max|res|.

    python3 second_order_fits.py --planted
    python3 second_order_fits.py [--fixed=10.4023,-12.3;...]
"""
import os
import random
import sys
import json

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, "..", "..", "hp")
sys.path.insert(0, HP)
sys.path.insert(0, HERE)
from dd_fits_mp import load_points, clean  # noqa: E402
from first_order_fits import polyfit, DEGS  # noqa: E402

mp.mp.dps = 60
RES = mp.mpf("1e-13")
DEG_MAX16 = 16          # the highest degree on D >= 16 (see History above)
LAM_DEGS = (10, 11)     # the lambda_2 estimator's degrees (see History above)


def degs_for(Dmin):
    return [d for d in DEGS if Dmin < 16 or d <= DEG_MAX16]


def fixedfit(ts, ys, deg, d1, d2):
    """Delta - d1 t - d2 t^2 fitted by 1, t^3 .. t^deg (scaled by t0 = max t); returns Delta_inf, max|res|"""
    t0 = max(ts)
    pw = [0] + list(range(3, deg + 1))
    X = mp.matrix([[(t / t0) ** k for k in pw] for t in ts])
    y = mp.matrix([v - d1 * t - d2 * t * t for t, v in zip(ts, ys)])
    c = mp.qr_solve(X, y)[0]
    coef = {k: c[i] / t0 ** k for i, k in enumerate(pw)}
    res = max(abs(d1 * t + d2 * t * t + sum(coef[k] * t ** k for k in pw) - v) for t, v in zip(ts, ys))
    return coef[0], res


def free_rows(Ds, vals, verbose=True, label=""):
    rows = []
    for var, tf in (("1/n", lambda D: 1 / (mp.mpf(D) - 2)), ("1/D", lambda D: 1 / mp.mpf(D))):
        for Dmin in (10, 12, 16):
            for deg in degs_for(Dmin):
                sel = [d for d in Ds if d >= Dmin]
                coef, res = polyfit([tf(d) for d in sel], [vals[d] for d in sel], deg)
                d0, d1, d2 = coef[0], coef[1], coef[2]
                if var == "1/D":
                    d2 = d2 - 2 * d1                     # to the 1/n coefficient
                rows.append(dict(var=var, Dmin=Dmin, deg=deg, D0=d0, D1=d1, D2=d2, res=res))
                if verbose:
                    print(f"  {label}{var} D >= {Dmin:2d} deg {deg:2d}: Delta_inf {mp.nstr(d0, 13)}  Delta_1 {mp.nstr(d1, 10)}  "
                          f"Delta_2 (1/n) {mp.nstr(d2, 9)}  max|res| {mp.nstr(res, 2)}", flush=True)
    return rows


def fixed_rows(Ds, vals, d1, d2, verbose=True, label=""):
    rows = []
    for Dmin in (10, 12, 16):
        for deg in degs_for(Dmin):
            sel = [d for d in Ds if d >= Dmin]
            ts = [1 / (mp.mpf(d) - 2) for d in sel]
            d0, res = fixedfit(ts, [vals[d] for d in sel], deg, d1, d2)
            rows.append(dict(Dmin=Dmin, deg=deg, D0=d0, res=res))
            if verbose:
                print(f"  {label}fixed D >= {Dmin:2d} deg {deg:2d}: Delta_inf {mp.nstr(d0, 13)}  max|res| {mp.nstr(res, 2)}",
                      flush=True)
    return rows


def lam_points():
    rows = {}
    for f in (os.path.join(HP, "data", "gamma_dd.jsonl"), os.path.join(HP, "data", "local", "gamma_dd.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("lam"):
                rows[round(r["D"], 6)] = r
    good = {}
    for d in sorted(rows):
        if rows[d]["F"] <= 1e-20 and all(abs(d - e) >= 0.5 for e in good):
            good[d] = rows[d]
    return good


def lam_rows(good, ysel=None, verbose=True, label=""):
    """polynomials in 1/n with lambda_inf = 2 fixed, degrees 8-12, D >= 20, 24, 30 (as first_order_fits.py)"""
    out = []
    Ds = sorted(good)
    for Dmin in (20, 24, 30):
        sel = [d for d in Ds if d >= Dmin]
        ts = [1 / (mp.mpf(d) - 2) for d in sel]
        ys = [ysel(d, t) if ysel else mp.mpf(good[d]["lam"]) for d, t in zip(sel, ts)]
        for deg in (8, 9, 10, 11, 12):
            c, res = polyfit(ts, ys, deg, fix0=mp.mpf(2))
            out.append(dict(Dmin=Dmin, deg=deg, L1=c[1], L2=c[2], res=res))
            if verbose:
                print(f"  {label}lambda D >= {Dmin} deg {deg:2d}: lambda_1 {mp.nstr(c[1], 9)}  lambda_2 {mp.nstr(c[2], 7)}  "
                      f"max|res| {mp.nstr(res, 2)}", flush=True)
    return out


def spread(vals):
    return (min(vals), max(vals)) if vals else (mp.nan, mp.nan)


def planted():
    random.seed(1)
    C = json.load(open(os.path.join(HERE, "..", "data", "formula_coefficients.json")))
    a = [mp.mpf(v) for v in C["a"]]
    b = [mp.mpf(1)] + [mp.mpf(v) for v in C["b"]]
    rat = lambda eps: sum(c * eps ** k for k, c in enumerate(a)) / sum(c * eps ** k for k, c in enumerate(b))
    tay = mp.taylor(rat, 0, 3)
    D0t, D1t, D2t = tay[0], tay[1], tay[2]
    print(f"planted: Delta_inf {mp.nstr(D0t, 13)}  Delta_1 {mp.nstr(D1t, 12)}  Delta_2 {mp.nstr(D2t, 12)} (Taylor of the "
          f"rational formula in 1/n)", flush=True)
    pts, _ = load_points()
    Ds = clean(pts, sorted(pts), verbose=False)
    syn = {d: rat(1 / (mp.mpf(d) - 2)) + mp.mpf(random.gauss(0, 1e-14)) for d in Ds}
    ok = True
    print("P1: free fits, Delta_2 recovered within 2e-3 by every fit with max|res| <= 1e-13", flush=True)
    fr = free_rows(Ds, syn, verbose=False)
    for r in fr:
        err2 = r["D2"] - D2t
        flag = ""
        if r["res"] <= RES:
            flag = "ok" if abs(err2) <= 2e-3 else "MISS"
            ok &= abs(err2) <= 2e-3
        print(f"  {r['var']} D >= {r['Dmin']:2d} deg {r['deg']:2d}: Delta_2 - planted {mp.nstr(err2, 3):>10s}  Delta_1 - planted "
              f"{mp.nstr(r['D1'] - D1t, 3):>10s}  Delta_inf - planted {mp.nstr(r['D0'] - D0t, 3):>10s}  max|res| {mp.nstr(r['res'], 2)}  {flag}")
    p1 = ok
    print(f"P1: {'PASS' if p1 else 'FAIL'}")
    good_free = [r["D0"] - D0t for r in fr if r["res"] <= RES]
    lo, hi = spread(good_free)
    print(f"P2: fixed fits with the planted Delta_1, Delta_2; free fits' Delta_inf - planted in [{mp.nstr(lo, 3)}, {mp.nstr(hi, 3)}]"
          f" (width {mp.nstr(hi - lo, 3)})", flush=True)
    fx = fixed_rows(Ds, syn, D1t, D2t, verbose=False)
    p2 = True
    for r in fx:
        err = r["D0"] - D0t
        flag = ""
        if r["res"] <= RES:
            flag = "ok" if abs(err) <= 1.5e-8 else "MISS"
            p2 &= abs(err) <= 1.5e-8
        print(f"  fixed D >= {r['Dmin']:2d} deg {r['deg']:2d}: Delta_inf - planted {mp.nstr(err, 3):>10s}  max|res| {mp.nstr(r['res'], 2)}  {flag}")
    gf = [r["D0"] - D0t for r in fx if r["res"] <= RES]
    flo, fhi = spread(gf)
    p2 &= bool(gf) and (fhi - flo) <= (hi - lo)
    print(f"  fixed fits passing the residual criterion: {len(gf)}; Delta_inf - planted in [{mp.nstr(flo, 3)}, {mp.nstr(fhi, 3)}] "
          f"(width {mp.nstr(fhi - flo, 3)}, free {mp.nstr(hi - lo, 3)})")
    print(f"P2: {'PASS' if p2 else 'FAIL'}")
    print("P3 (control, must fire): Delta_1 held 0.027 too high", flush=True)
    fx3 = fixed_rows(Ds, syn, D1t + mp.mpf("0.027"), D2t, verbose=False)
    fired = all(r["res"] > RES or abs(r["D0"] - D0t) > 1.5e-8 for r in fx3)
    for r in fx3:
        print(f"  fixed D >= {r['Dmin']:2d} deg {r['deg']:2d}: Delta_inf - planted {mp.nstr(r['D0'] - D0t, 3):>10s}  max|res| {mp.nstr(r['res'], 2)}")
    print(f"P3: {'fires (PASS)' if fired else 'does not fire (FAIL)'}")
    print(f"P4: lambda_2 from 1/gamma of the rational gamma(D), noise 1e-12, degree {LAM_DEGS} fits on D >= 24, 30", flush=True)
    e = [mp.mpf(v) for v in C["e"]]
    f = [mp.mpf(1)] + [mp.mpf(v) for v in C["f"]]
    lam = lambda t: sum(c * t ** k for k, c in enumerate(f)) / sum(c * t ** k for k, c in enumerate(e))
    lt = mp.taylor(lam, 0, 3)
    print(f"  planted: lambda_inf {mp.nstr(lt[0], 10)}  lambda_1 {mp.nstr(lt[1], 10)}  lambda_2 {mp.nstr(lt[2], 10)}")
    good = lam_points()
    lr = lam_rows(good, ysel=lambda d, t: lam(t) + mp.mpf(random.gauss(0, 1e-12)), verbose=False)
    p4 = True
    for r in lr:
        err = r["L2"] - lt[2]
        flag = ""
        if r["deg"] in LAM_DEGS and r["Dmin"] >= 24:
            flag = "ok" if abs(err) <= 0.02 else "MISS"
            p4 &= abs(err) <= 0.02
        print(f"  D >= {r['Dmin']} deg {r['deg']:2d}: lambda_2 - planted {mp.nstr(err, 3):>10s}  lambda_1 - planted "
              f"{mp.nstr(r['L1'] - lt[1], 3):>10s}  max|res| {mp.nstr(r['res'], 2)}  {flag}")
    print(f"P4: {'PASS' if p4 else 'FAIL'}")
    allok = p1 and p2 and fired and p4
    print("PLANTED: PASS" if allok else "PLANTED: FAIL")
    return allok


def real(fixed_sets):
    pts, _ = load_points()
    Ds = clean(pts, sorted(pts), verbose=False)
    print(f"Delta: {len(Ds)} cleaned double-double points, D = {Ds[0]:.2f} .. {Ds[-1]:.2f}", flush=True)
    fr = free_rows(Ds, pts)
    good = [r for r in fr if r["res"] <= RES]
    for key in ("D0", "D1", "D2"):
        lo, hi = spread([r[key] for r in good])
        print(f"  fits with max|res| <= 1e-13 ({len(good)}): {key} in [{mp.nstr(lo, 12)}, {mp.nstr(hi, 12)}]")
    for var in ("1/n", "1/D"):
        lo, hi = spread([r["D2"] for r in good if r["var"] == var])
        print(f"     Delta_2 from the {var} fits: [{mp.nstr(lo, 10)}, {mp.nstr(hi, 10)}]")
    d1c = (min(r["D1"] for r in good) + max(r["D1"] for r in good)) / 2
    d2s = sorted(r["D2"] for r in good)
    d2c = d2s[len(d2s) // 2]
    print("lambda = 1/gamma: double-double growth rates", flush=True)
    lr = lam_rows(lam_points())
    l2 = [r["L2"] for r in lr if r["deg"] in LAM_DEGS and r["Dmin"] >= 24]
    print(f"  lambda_2 (degrees {LAM_DEGS}, D >= 24, 30): [{mp.nstr(min(l2), 6)}, {mp.nstr(max(l2), 6)}]")
    sets = [("fitted (centre of D_1's range, median D_2)", d1c, d2c)] + fixed_sets
    lo0, hi0 = spread([r["D0"] for r in good])
    print(f"  free fits: Delta_inf in [{mp.nstr(lo0, 12)}, {mp.nstr(hi0, 12)}] (width {mp.nstr(hi0 - lo0, 3)})")
    for name, d1, d2 in sets:
        print(f"Fixed: {name}: Delta_1 = {mp.nstr(d1, 10)}, Delta_2 = {mp.nstr(d2, 10)}", flush=True)
        fx = fixed_rows(Ds, pts, d1, d2)
        gf = [r["D0"] for r in fx if r["res"] <= RES]
        if gf:
            lo, hi = spread(gf)
            print(f"  {len(gf)} fixed fits with max|res| <= 1e-13: Delta_inf in [{mp.nstr(lo, 12)}, {mp.nstr(hi, 12)}] "
                  f"(width {mp.nstr(hi - lo, 3)})")
        else:
            print(f"  no fixed fit reaches max|res| <= 1e-13 (smallest {mp.nstr(min(r['res'] for r in fx), 3)})")


if __name__ == "__main__":
    if "--planted" in sys.argv[1:]:
        sys.exit(0 if planted() else 1)
    fs = []
    for a in sys.argv[1:]:
        if a.startswith("--fixed="):
            for i, pair in enumerate(a.split("=", 1)[1].split(";")):
                d1, d2 = pair.split(",")
                fs.append((f"given {i + 1}", mp.mpf(d1), mp.mpf(d2)))
    real(fs)
