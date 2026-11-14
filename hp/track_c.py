#!/usr/bin/env python3
"""Track C (#801): Delta(D) at D 250-400 from a 1/n series matched to the data below D 245 (pre-registered on #801).

Data: data/delta_all.csv rows of kind dd (err_kind "check", cleaned as dd_fits_mp.clean), qd ("starts") and chain_fit,
D_min <= D <= 245.64; Delta = the sum of the four parts, sigma = err floored at 1e-16. Fit: Delta = sum_{j<=K} a_j eps^j,
eps in {1/(D-2), 1/(D-1), 1/D}, K in {12, 14, 16, 18, 20}, D_min in {10, 20, 30}: weighted least squares in arb at
256 bits (Chebyshev basis over the fitted eps range; normal equations), accepted when chi^2/dof <= 4. At each target:
the median of the accepted fits, E = F max(h, s), h half their range, s the central fit's propagated data error (from
the inverse normal matrix), F from the planted test.

    cd hp && python3 track_c.py planted [--out=FILE]         P0-P4 on the real D grid with noise N(0, sigma)
    cd hp && python3 track_c.py real [--F=1] [--out=FILE]     the real data
"""
import csv
import json
import math
import os
import random
import sys

import mpmath as mp
from flint import arb, arb_mat, ctx

import dd_fits_mp as F_

ctx.prec = 256
mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
DMAX = 245.64
VARS = {"1/(D-2)": lambda D: 1 / (D - 2), "1/(D-1)": lambda D: 1 / (D - 1), "1/D": lambda D: 1 / D}
DEGS = (12, 14, 16, 18, 20)
DMINS = (10, 20, 30, 45, 60, 80, 100, 130, 160)   # second amendment (#801): the fit starts as low as the data allow
CHI_MAX = 4
TARGETS = [245.64, 250, 255, 259.5, 300, 400]
# the README's rational formula for Delta (a known series in eps = 1/n)
RA = ["1.6438975180", "4.0378045077e1", "2.5233285989e2", "2.0673502321e2", "-3.5709850575e2", "3.1347320996e2",
      "-4.1891374574e2", "3.2695252782e2", "-1.6050467699e2"]
RB = ["1.8251453626e1", "6.8182050167e1"]


def readme_formula(D):
    e = 1 / (mp.mpf(D) - 2)
    num = mp.fsum(mp.mpf(a) * e ** k for k, a in enumerate(RA))
    return num / (1 + mp.mpf(RB[0]) * e + mp.mpf(RB[1]) * e ** 2)


def load_real():
    """{D: (Delta mpf, sigma float, kind)} for D <= DMAX, kinds dd(check), qd(starts), chain_fit"""
    out = {}
    dd = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))):
        D = float(r["D"])
        if D > DMAX + 1e-9:
            continue
        k, ek = r["kind"], r["err_kind"]
        if not ((k == "dd" and ek == "check") or (k == "qd" and ek == "starts") or k == "chain_fit"):
            continue
        v = mp.fsum(mp.mpf(r["Delta_q%d" % i]) for i in range(4) if r["Delta_q%d" % i])
        s = max(float(r["err"]), 1e-16)
        if k == "dd":
            dd[round(D, 6)] = (v, s)
        else:
            out[round(D, 6)] = (v, s, k)
    keep = F_.clean({d: v for d, (v, _) in dd.items()}, sorted(dd), verbose=False)
    for d in keep:
        out[d] = (dd[d][0], dd[d][1], "dd")
    return out


def cheb(t, a, b, K):
    u = (2 * t - (a + b)) / (b - a)
    T = [arb(1), u]
    for _ in range(K - 1):
        T.append(2 * u * T[-1] - T[-2])
    return T[:K + 1]


def fit(data, var, K, dmin):
    """weighted least squares; returns dict with ev(D), s(D), chi2dof, n"""
    Ds = sorted(d for d in data if d >= dmin)
    f = VARS[var]
    ts = [arb(mp.nstr(f(mp.mpf(d)), 70)) for d in Ds]
    a, b = min(ts, key=lambda x: float(x)), max(ts, key=lambda x: float(x))
    rows = [cheb(t, a, b, K) for t in ts]
    w = [arb(1) / arb(data[d][1]) for d in Ds]
    X = arb_mat([[rows[i][j] * w[i] for j in range(K + 1)] for i in range(len(Ds))])
    y = arb_mat([[arb(mp.nstr(data[d][0], 70)) * w[i]] for i, d in enumerate(Ds)])
    XT = X.transpose()
    N = XT * X
    Ninv = N.inv()
    c = Ninv * (XT * y)
    res = X * c - y
    chi2 = sum(float(res[i, 0]) ** 2 for i in range(len(Ds)))
    cf = [c[j, 0] for j in range(K + 1)]

    def ev(D):
        t = arb(mp.nstr(f(mp.mpf(D)), 70)) if D != "inf" else arb(0)
        row = cheb(t, a, b, K)
        return sum((row[j] * cf[j] for j in range(K + 1)), arb(0))

    def sd(D):
        t = arb(mp.nstr(f(mp.mpf(D)), 70)) if D != "inf" else arb(0)
        row = cheb(t, a, b, K)
        v = arb(0)
        for i in range(K + 1):
            for j in range(K + 1):
                v += row[i] * Ninv[i, j] * row[j]
        return math.sqrt(max(float(v), 0.0))

    def slope():                                    # d Delta / d eps at eps = 0 (the same a_1 in all three variables)
        h = arb("1e-30")
        row1, row0 = cheb(h, a, b, K), cheb(arb(0), a, b, K)
        return (sum((row1[j] * cf[j] for j in range(K + 1)), arb(0)) -
                sum((row0[j] * cf[j] for j in range(K + 1)), arb(0))) / h
    resid = {d: float(res[i, 0]) for i, d in enumerate(Ds)}         # (fit - data)/sigma
    return dict(ev=ev, sd=sd, slope=slope, chi2dof=chi2 / max(1, len(Ds) - K - 1), n=len(Ds), resid=resid,
                var=var, K=K, dmin=dmin)


def all_fits(data, log=None):
    fits = []
    for var in VARS:
        for K in DEGS:
            for dmin in DMINS:
                ft = fit(data, var, K, dmin)
                ft["ok"] = ft["chi2dof"] <= CHI_MAX
                fits.append(ft)
                if log:
                    log("  %-8s K %2d Dmin %2d: n %4d chi2/dof %.3g %s" % (var, K, dmin, ft["n"], ft["chi2dof"],
                                                                          "accepted" if ft["ok"] else ""))
    return fits


def levels(fits):
    """D0, the smallest D_min with at least 5 accepted fits, and the next D_min; the fits the estimate uses"""
    for i, dm in enumerate(DMINS):
        if sum(1 for f in fits if f["ok"] and f["dmin"] == dm) >= 5:
            use = {dm} | ({DMINS[i + 1]} if i + 1 < len(DMINS) else set())
            return dm, [f for f in fits if f["ok"] and f["dmin"] in use]
    return None, []


def estimate(fits, D, Fc=1.0):
    D0, acc = levels(fits)
    if len(acc) < 5:                                # an estimate needs at least 5 accepted fits
        return None
    vals = sorted((float(f["ev"](D) - arb("1.6")), f) for f in acc)  # offset for float resolution
    vs = [v for v, _ in vals]
    med_i = len(vs) // 2
    central = vals[med_i][1]
    c = central["ev"](D)
    h = (vs[-1] - vs[0]) / 2
    s = central["sd"](D)
    return dict(value=c, h=h, s=s, E=Fc * max(h, s), n=len(acc), central=central, D0=D0)


def planted_data(real, kind, seed):
    rng = random.Random(seed)
    out = {}
    for d, (v, s, k) in real.items():
        D = mp.mpf(d)
        t = readme_formula(D)
        if kind == "P1":
            t += plant_exp(D, 0.1)
        if kind == "P2":
            t += plant_exp(D, 0.05)
        if kind == "P3":
            t += mp.mpf("1e-9") * mp.sqrt(mp.mpf("259.903") - D)
        if kind == "P4" and k == "chain_fit":
            t += mp.mpf("-3.1e-12") * ((D - 74) / 141) ** 2
        out[d] = (t + mp.mpf(rng.gauss(0, s)), s, k)
    return out


def plant_exp(D, rate):
    c = mp.mpf("1e-9") * mp.exp(rate * 98)
    return c * mp.exp(-rate * (D - 2)) * mp.cos(mp.mpf("1.3") * (D - 2) + mp.mpf("0.4"))


def truth(kind, D):
    D = mp.mpf(D)
    t = readme_formula(D)
    if kind == "P1":
        t += plant_exp(D, 0.1)
    if kind == "P2":
        t += plant_exp(D, 0.05)
    if kind == "P3":
        t += mp.mpf("1e-9") * mp.sqrt(mp.mpf("259.903") - D) if D < mp.mpf("259.903") else mp.nan
    return t


def cmd_planted():
    real = load_real()
    print("real grid: %d points, D %.3f .. %.3f" % (len(real), min(real), max(real)))
    out = {}
    ratios = []
    for kind, seed in (("P0", 1), ("P1", 2), ("P2", 3), ("P3", 4)):
        data = planted_data(real, kind, seed)
        fits = all_fits(data)
        nacc = sum(f["ok"] for f in fits)
        D0, used = levels(fits)
        print("%s: %d of %d fits accepted (per D_min %s); D0 = %s, %d fits used" % (
            kind, nacc, len(fits), ", ".join("%d: %d" % (dm, sum(1 for f in fits if f["ok"] and f["dmin"] == dm))
                                             for dm in DMINS), D0, len(used)))
        rec = {}
        for D in TARGETS:
            est = estimate(fits, D)
            if est is None:
                continue
            tv = truth(kind, D)
            err = abs(float(est["value"] - arb(mp.nstr(tv, 70)))) if not (isinstance(tv, mp.mpf) and mp.isnan(tv)) \
                else float("nan")
            ratio = err / est["E"] if est["E"] > 0 else float("inf")
            if kind != "P3" and D in (250, 300, 400):
                ratios.append(ratio)
            rec[str(D)] = dict(err=err, h=est["h"], s=est["s"], E=est["E"], ratio=ratio)
            print("  D %7.2f: |fit - truth| %.2e  h %.2e  s %.2e  E %.2e  ratio %.2f" % (
                D, err, est["h"], est["s"], est["E"], ratio))
        out[kind] = rec
    Fc = max(1.0, max(ratios))
    print("calibration: largest |fit - truth| / E over P0-P2 at D 250, 300, 400 = %.2f -> F = %.2f%s" % (
        max(ratios), Fc, " (> 3: reported)" if Fc > 3 else ""))
    out["F"] = Fc
    # P4 (#313): series on dd points D <= 67.77 (1/D, 1/n; degrees 16, 18, 20) against the chain points at 215
    data = planted_data(real, "P4", 5)
    ref = planted_data(real, "P0", 5)              # the same noise without the bias
    low = {d: v for d, v in data.items() if v[2] == "dd" and 10 <= d <= 67.77}
    sers = [fit(low, var, K, 10) for var in ("1/D", "1/(D-2)") for K in (16, 18, 20)]
    D215 = min((d for d in data if data[d][2] == "chain_fit"), key=lambda d: abs(d - 215))
    diffs = [float(arb(mp.nstr(data[D215][0], 70)) - s_["ev"](D215)) for s_ in sers]
    diffs0 = [float(arb(mp.nstr(ref[D215][0], 70)) - s_["ev"](D215)) for s_ in sers]
    spread = max(diffs) - min(diffs)
    print("P4 (#313): at D %.2f the planted chain minus the low-D series: %s; without the bias %s; planted bias %.2e; "
          "the series' spread %.2e" % (D215, ", ".join("%.2e" % v for v in diffs), ", ".join("%.2e" % v for v in diffs0),
                                       -3.1e-12 * ((D215 - 74) / 141) ** 2, spread))
    out["P4"] = dict(D=D215, diffs=diffs, diffs_nobias=diffs0, spread=spread)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    return out


def chain_beyond(raw):
    """#603's converged states past the chain_fit rows: {D: Delta} from I603-win2/chain and chain_nsr (anm_pin3_*)"""
    import glob
    import numpy as np
    out = {}
    for d in ("chain", "chain_nsr"):
        for p_ in glob.glob(os.path.join(raw, "I603-win2", d, "anm_pin3_D*.npz")):
            z = np.load(p_)
            m = json.loads(str(z["meta"]))
            if m["D"] <= DMAX + 1e-9:
                continue
            parts = [z[k][-1] for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in z.files]
            out[m["D"]] = mp.fsum(mp.mpf(float(x)) for x in parts)
    return out


def cmd_real():
    Fc = float(args.get("F", "1"))
    real = load_real()
    fits = all_fits(real)
    D0, used = levels(fits)
    print("real data: %d points, D %.3f .. %.3f; %d of %d fits accepted (per D_min %s); D0 = %s, %d fits used" % (
        len(real), min(real), max(real), sum(f["ok"] for f in fits), len(fits),
        ", ".join("%d: %d" % (dm, sum(1 for f in fits if f["ok"] and f["dmin"] == dm)) for dm in DMINS), D0, len(used)))
    print("  used: " + ", ".join("%s K%d D%d (%.2f)" % (f["var"], f["K"], f["dmin"], f["chi2dof"]) for f in used))
    out = {"D0": D0, "used": [(f["var"], f["K"], f["dmin"], f["chi2dof"]) for f in used], "F": Fc, "targets": {}}
    for D in TARGETS + ["inf"]:
        est = estimate(fits, D, Fc)
        v = est["value"]
        print("  D %-7s: Delta = %s  +- %.2e   (h %.2e, s %.2e, %d fits)" % (
            D, v.mid().str(18, radius=False), est["E"], est["h"], est["s"], est["n"]))
        out["targets"][str(D)] = dict(Delta=v.mid().str(25, radius=False), E=est["E"], h=est["h"], s=est["s"])
    sl = sorted(float(f["slope"]()) for f in used)
    print("  slope Delta_1 = dDelta/deps at eps = 0: median %.6f, range %.6f .. %.6f (D = infinity, M = 5: 10.402; "
          "fitted 10.3753)" % (sl[len(sl) // 2], sl[0], sl[-1]))
    out["slope"] = dict(median=sl[len(sl) // 2], lo=sl[0], hi=sl[-1])
    # residuals of the central fit on D 150-245
    cen = estimate(fits, 300, Fc)["central"]
    rr = [(d, cen["resid"][d]) for d in sorted(cen["resid"]) if 150 <= d <= DMAX + 1e-9]
    nr = [abs(v) for _, v in rr]
    absr = [abs(v) * real[d][1] for d, v in rr]
    print("  central fit (%s K%d D%d) on D 150-245: %d points, |res|/sigma max %.2f, rms %.2f; |res| max %.2e" % (
        cen["var"], cen["K"], cen["dmin"], len(rr), max(nr), math.sqrt(sum(v * v for v in nr) / len(nr)), max(absr)))
    out["resid_150_245"] = [(d, v) for d, v in rr]
    # #313: the low-D series (dd points D 10-67.77) against the chain at 215
    low = {d: v for d, v in real.items() if v[2] == "dd" and 10 <= d <= 67.77}
    sers = [fit(low, var, K, 10) for var in ("1/D", "1/(D-2)") for K in (16, 18, 20)]
    D215 = min((d for d in real if real[d][2] == "chain_fit"), key=lambda d: abs(d - 215))
    diffs = [float(arb(mp.nstr(real[D215][0], 70)) - s_["ev"](D215)) for s_ in sers]
    print("  #313: chain_fit at D %.2f minus the low-D series (1/D, 1/n; 16, 18, 20): %s; spread %.2e; the track C "
          "series minus chain_fit there: %+.2e" % (D215, ", ".join("%+.2e" % v for v in diffs), max(diffs) - min(diffs),
                                                    float(estimate(fits, D215, Fc)["value"] - arb(mp.nstr(real[D215][0], 70)))))
    out["313"] = dict(D=D215, diffs=diffs)
    # the chain past 245.64 against the series
    raw = args.get("raw")
    if raw:
        cb = chain_beyond(raw)
        rows = []
        for D in sorted(cb):
            est = estimate(fits, D, Fc)
            d_ = float(arb(mp.nstr(cb[D], 70)) - est["value"])
            rows.append((D, d_, est["E"]))
        print("  #603's chain past 245.64 minus the series (%d states):" % len(rows))
        for D, d_, E in rows[::max(1, len(rows) // 25)] + [rows[-1]]:
            print("    D %9.4f: chain - series %+.3e  E %.2e  ratio %.2f" % (D, d_, E, abs(d_) / E))
        out["beyond"] = rows
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "real":
        cmd_real()
        sys.exit(0)
    if cmd == "planted":
        cmd_planted()
        sys.exit(0)
    raise SystemExit(__doc__)
