#!/usr/bin/env python3
"""The gauge rate lam_g past the comparison frontier (#807): #787's screen departures on #603's states, D 250.47 to
259.7, against #805's weak-pivot law (notes/LARGE_D_WALL.md "The chain past the comparison frontier"). Tests and pass
criteria were pre-registered on #807.

    cd hp && python3 lam_frontier.py jet RAWDIR [--out=F]       test J for lam_g: chain_nsr's jets continued to the states
    cd hp && python3 lam_frontier.py law RAWDIR [--out=F]       test L: lam_g - R(D) times p_min, bounded within 3?
    cd hp && python3 lam_frontier.py screen RAWDIR [--out=F]    #787's screen on lam_g and on lam_g - R
    cd hp && python3 lam_frontier.py pred RAWDIR [--out=F]      the proposed rule: lam_g(state) - lam_g(prediction)
    cd hp && python3 lam_frontier.py posthoc RAWDIR [--out=F]   post hoc: S and E past the frontier only, end windows

RAWDIR is a clone of the raw-data repository (I603-win2/ at 1a3c1d04 or later). lam_g is hp/lam_gauge.py's,
1 - (n - 1)<a_h^2 - 1>, n = D - 2. p_min is the state's smallest LU pivot, pivots[0] of the record of the step from it
(frontier_audit.record_info).

jet: each jet of chain_nsr (25 orders, eight parts) gives a_h(D_b + a) = sum z_n a^n and lam_J at 200 digits; every
other state with T <= 1e-12 (T: the change of lam_J from order 22 to 23 and from 23 to 24, the larger) passes when
|lam - lam_J| <= 10 T + 1e-30. Control: each jet's lam_J at its next state's D equals the prediction file's lam_g to
1e-60, and with --mutate=n (n in place of n - 1 in lam_J only) it must not. Plants (scalar, exact jets, the same
comparison): W and S must pass, N and N1 must fail. Exits 1 when a verdict differs from the expected one.
law: R, least squares in 1/(D - 2) of degree K on chain_fit's backbone states from A to 245.64 (central K 8, A 150;
spread over K 6, 8, 10 and A 150, 200); a state enters when |lam - R| >= 3 spread; the windowed maximum (windows of 3
in D) of |lam - R| p_min over 250.47-259.7 passes when its largest and smallest are within a factor 3. Plants first:
R + W and R + S scaled so that their screen departure at 256.5686 is -2.8e-9, and R + fixed per-state errors of 1e-10;
W must pass, S and the errors must fail. Exits 1 when a plant's verdict differs from the expected one.
screen: export_delta_all.lam_screen's polynomial (degree SCREEN_DEG through SCREEN_N neighbours) on lam_g and on
lam_g - R, per state, and the literal reading's statistic (the screen's departure times p_min).
pred: |lam_g(state) - lam_g(prediction file)| <= 1e-20 at every state with a prediction; --mutate=lam adds 1e-10 to
the lam_g of the state nearest D 240, and the check must fail. Exits 1 when it fails.
"""
import csv
import glob
import json
import math
import os
import random
import re
import sys

import mpmath as mp
import numpy as np

import frontier_audit as FA
import lam_gauge as LG
from export_delta_all import SCREEN_DEG, SCREEN_N

mp.mp.dps = 200
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
HERE = os.path.dirname(os.path.abspath(__file__))
FRONT = 245.6366177558899          # chain_fit's last row
LO, HI = 250.46, 259.71            # #787's departures, 250.47 to 259.7
SCREEN_AT = 256.5686               # #787's largest departure, -2.8e-9
SCREEN_DEP = -2.8e-9


def lam_of(path):
    parts, meta = LG.load(path)
    with mp.workprec(LG.PREC):
        v = LG.lam_g(parts, meta)
        D = mp.mpf(meta["D_str"]) if "D_str" in meta else mp.mpf(meta["D"])
    return +D, +v


def chain_states(raw):
    """[(D, lam_g, path)] of #603's states, every run"""
    out = []
    for d in FA.FOLDERS:
        for f in glob.glob(os.path.join(raw, "I603-win2", d, "anm_pin3_D*.npz")):
            D, v = lam_of(f)
            out.append((D, v, f))
    return sorted(out)


def backbone(raw, lo=0.0):
    """[(D, lam_g)] of chain_fit's backbone states (delta_all.csv's state column) from lo to the frontier"""
    out = []
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))):
        if r["kind"] == "chain_fit" and lo <= float(r["D"]) <= FRONT + 1e-9 and r["state"]:
            D, v = lam_of(os.path.join(raw, r["state"]))
            out.append((D, v))
    return sorted(out)


def pmin_of(raw):
    return {k: v["min_pivot"] for k, v in FA.record_info(raw).items() if v["min_pivot"]}


# ---- test J ----

def load_ah_jet(path):
    """(D_b, [[a_h coefficient n, point j]], M) from a jet's eight parts"""
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    ks = sorted(k for k in z.files if re.fullmatch(r"z\d", k))
    N1, L = z[ks[0]].shape
    M = int(m["M"])
    K = (L - 1) // (4 * M) - 1
    off = 3 * M + 4 * M * K
    c = [[mp.fsum(mp.mpf(float(z[k][n][off + j])) for k in ks) for j in range(M)] for n in range(N1)]
    return mp.mpf(m["D_str"]), c, M


def lam_from_ah(ah, D, mutate=""):
    k = (D - 2) if mutate == "n" else (D - 3)
    return 1 - k * mp.fsum((x - 1) * (x + 1) for x in ah) / len(ah)


def pred_ah(path):
    """(a, a_h) of a prediction file: the float step its series was summed at (meta a) and its horizon block"""
    parts, meta = LG.load(path)
    M, K, off = LG.layout(parts, meta)
    return mp.mpf(meta["a"]), [mp.fsum(mp.mpf(float(p[off + j])) for p in parts) for j in range(M)]


def lam_jet(Db, c, M, D, mutate=""):
    """lam_J at orders N - 2, N - 1 and N"""
    a = D - Db
    pw = [a ** n for n in range(len(c))]
    k = (D - 2) if mutate == "n" else (D - 3)
    out = []
    for order in (len(c) - 3, len(c) - 2, len(c) - 1):
        ah = [mp.fsum(c[n][j] * pw[n] for n in range(order + 1)) for j in range(M)]
        out.append(1 - k * mp.fsum((x - 1) * (x + 1) for x in ah) / M)
    return out


def compare(Db, J, pts, T_max=1e-12):
    """J(D) -> (value, T); pts [(D, v)]: every point other than the base within reach"""
    rows, ok = [], True
    for D, v in pts:
        if abs(D - Db) < mp.mpf("1e-20"):
            continue
        val, T = J(D)
        if T > T_max:
            continue
        d = v - val
        good = abs(d) <= 10 * T + mp.mpf("1e-30")
        ok &= bool(good)
        rows.append(dict(D=float(D), d=float(d), T=float(T), ratio=float(abs(d) / (10 * T + mp.mpf("1e-30")))))
    return ok, rows


def scalar_J(jet, Db):
    def J(D):
        a = D - Db
        t = [c * a ** n for n, c in enumerate(jet)]
        return mp.fsum(t), max(abs(t[-1]), abs(t[-2]))
    return J


def cmd_jet(raw):
    W = os.path.join(raw, "I603-win2", "chain_nsr")
    jets = sorted(glob.glob(os.path.join(W, "jet_pin3_D*.npz")))
    st = chain_states(raw)
    pm = pmin_of(raw)
    ri = FA.record_info(raw)
    tab = [dict(D=float(D), min_pivot=pm[round(float(D), 6)]) for D, _, _ in st
           if 245.6 <= D <= 255.8 and round(float(D), 6) in pm]
    pf = FA.plant_funcs(tab)
    verdict = True
    out = dict(plants={}, real={})
    # the plants: scalar truths on the real grid, each jet's base, exact jets
    rng = random.Random(807)
    grid = [D for D, _, _ in st if D > 245]
    sgn = {round(float(D), 6): rng.choice((-1, 1)) for D in grid}
    raw_eps = {k: ((ri.get(k, {}).get("refused") or 0) + 0.2) / pm[k] * s for k, s in sgn.items() if k in pm}
    win = [abs(v) for k, v in raw_eps.items() if LO <= k <= HI]
    cN = 1e-11 / float(np.median(win))
    moved = min(grid, key=lambda D: abs(D - SCREEN_AT))
    bases = [load_ah_jet(j)[0] for j in jets]
    print("plants (%d jets, bases %s to %s):" % (len(jets), mp.nstr(bases[0], 8), mp.nstr(bases[-1], 8)))
    for name, expect in (("W", True), ("S", True), ("N", False), ("N1", False)):
        f, fj = pf["S"] if name == "S" else pf["W"]
        n_in, worst, ok_all = 0, None, True
        for Db in bases:
            jet = fj(Db, 24)
            if name == "N":
                jet[0] += cN * raw_eps.get(round(float(Db), 6), 0)
            P = []
            for D in grid:
                v = f(D)
                if name == "N":
                    v += cN * raw_eps.get(round(float(D), 6), 0)
                if name == "N1" and D == moved:
                    v += mp.mpf("1e-10")
                P.append((D, v))
            ok, rows = compare(Db, scalar_J(jet, Db), P)
            ok_all &= ok
            n_in += len(rows)
            w = max(rows, key=lambda r: r["ratio"]) if rows else None
            if w and (worst is None or w["ratio"] > worst["ratio"]):
                worst = dict(w, base=float(Db))
        good = ok_all == expect
        verdict &= good
        print("  plant %-2s: %d comparisons; worst |d|/(10T + 1e-30) %.3g at D %.4f from the jet at %.4f: %s; expected %s: %s"
              % (name, n_in, worst["ratio"], worst["D"], worst["base"], "passes" if ok_all else "fails",
                 "pass" if expect else "fail", "as expected" if good else "NOT AS EXPECTED"))
        out["plants"][name] = dict(ok=ok_all, expected=expect, comparisons=n_in, worst=worst)
    # the real chain: controls first
    pts = [(D, v) for D, v, _ in st]
    print("real chain:")
    ctl_ok, ctl_mut, ctl = True, True, []
    for jp in jets:
        Db, c, M = load_ah_jet(jp)
        nxt = min((D for D, _ in pts if D > Db + mp.mpf("1e-20")), default=None)
        pf_ = os.path.join(W, "pred_pin3_D%.4f.npz" % float(nxt)) if nxt is not None else ""
        if not os.path.exists(pf_):
            continue
        # amended after the first run (#807): the jet at the prediction's own float a, both sides with n = D_b + a - 2
        # (the stored D strings carry the run's 50-digit rounding of D, 2.5e-48 to 6e-48 from base + a past 257.5)
        ap, ahp = pred_ah(pf_)
        Dp = Db + ap
        vp = lam_from_ah(ahp, Dp)
        e = abs(lam_jet(Db, c, M, Dp)[-1] - vp)
        em = abs(lam_jet(Db, c, M, Dp, mutate="n")[-1] - vp)
        ctl.append((float(Db), float(e), float(em)))
        ctl_ok &= e <= mp.mpf("1e-60")
        ctl_mut &= em <= mp.mpf("1e-60")
    print("  control: %d jets against their next prediction, largest |lam_J - lam_pred| %.1e (bound 1e-60): %s; "
          "with --mutate=n smallest %.1e: %s" % (len(ctl), max(x[1] for x in ctl), "passes" if ctl_ok else "FAILS",
                                                min(x[2] for x in ctl), "fails, as it must" if not ctl_mut else "PASSES (bad)"))
    verdict &= bool(ctl_ok) and not ctl_mut
    allrows, ok_all = [], True
    for jp in jets:
        Db, c, M = load_ah_jet(jp)

        def J(D, Db=Db, c=c, M=M):
            l22, l23, l24 = lam_jet(Db, c, M, D)
            return l24, max(abs(l24 - l23), abs(l23 - l22))
        near = [(D, v) for D, v in pts if abs(D - Db) < 6]
        ok, rows = compare(Db, J, near)
        ok_all &= ok
        for r in rows:
            r["base"] = float(Db)
        allrows += rows
    worst = max(allrows, key=lambda r: r["ratio"])
    reached = sorted(set(round(r["D"], 6) for r in allrows))
    print("  test J: %d comparisons from %d jets, %d states reached, D %.4f to %.4f; worst |d|/(10T + 1e-30) %.3g at "
          "D %.4f from the jet at %.4f (d %+.2e, T %.1e): %s" % (
              len(allrows), len(jets), len(reached), reached[0], reached[-1], worst["ratio"], worst["D"], worst["base"],
              worst["d"], worst["T"], "lam_g is one analytic function there" if ok_all else "NOT one analytic function"))
    big = max(allrows, key=lambda r: abs(r["d"]))
    print("  largest |d| %.2e at D %.4f (T %.1e)" % (abs(big["d"]), big["D"], big["T"]))
    out["real"] = dict(ok=ok_all, control=ctl, rows=allrows, reached=reached)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    sys.exit(0 if verdict else 1)


# ---- the screen (export_delta_all.lam_screen's polynomial) ----

def screen_dep(pts, i):
    """pts [(D, v)] sorted; v_i minus the degree-SCREEN_DEG polynomial in 1/(D - 2) through its SCREEN_N nearest others,
    as export_delta_all.lam_screen forms it (f64 polyfit on differences from the nearest)"""
    D = pts[i][0]
    others = pts[:i] + pts[i + 1:]
    near = sorted(others, key=lambda p: abs(p[0] - D))[:SCREEN_N]
    t = [1 / (p[0] - 2) for p in near]
    cf = np.polyfit([float(x - 1 / (D - 2)) for x in t], [float(p[1] - near[0][1]) for p in near], SCREEN_DEG)
    return float(pts[i][1] - (near[0][1] + mp.mpf(float(cf[-1]))))


# ---- test L ----

def fit_R(bb, K, A):
    """least squares polynomial of degree K in x = 1/(D - 2), centred and scaled, on bb's points with D >= A"""
    with mp.workdps(60):
        P = [(1 / (D - 2), v) for D, v in bb if D >= A]
        xs = [x for x, _ in P]
        xm, xw = (max(xs) + min(xs)) / 2, (max(xs) - min(xs)) / 2
        Am = mp.matrix([[((x - xm) / xw) ** p for p in range(K + 1)] for x in xs])
        c = mp.qr_solve(Am, mp.matrix([v for _, v in P]))[0]
        res = max(abs(v - sum(c[p] * ((x - xm) / xw) ** p for p in range(K + 1))) for x, v in P)
    return (lambda D: sum(c[p] * ((1 / (D - 2) - xm) / xw) ** p for p in range(K + 1))), float(res)


FITS = [(K, A) for K in (6, 8, 10) for A in (150.0, 200.0)]


def law(bb, past, pm, label, quiet=False):
    """bb [(D, v)] to the frontier, past [(D, v)] beyond it; returns (passes, stats)"""
    Rs = {ka: fit_R(bb, *ka) for ka in FITS}
    Rc = Rs[(8, 150.0)][0]
    rows = []
    for D, v in past:
        if not LO <= D <= HI or round(float(D), 6) not in pm:
            continue
        r0 = Rc(D)
        spread = max(abs(R(D) - r0) for R, _ in Rs.values())
        dl = v - r0
        rows.append(dict(D=float(D), dl=float(dl), spread=float(spread), p_min=pm[round(float(D), 6)],
                         enters=bool(abs(dl) >= 3 * spread)))
    ent = [r for r in rows if r["enters"]]
    wm = []
    for r0 in ent:
        if r0["D"] + 3 > ent[-1]["D"] + 1e-9:
            break
        w = [abs(r["dl"]) * r["p_min"] for r in ent if r0["D"] <= r["D"] < r0["D"] + 3]
        wm.append(max(w))
    rg = max(wm) / min(wm) if wm else float("nan")
    ok = bool(wm) and rg <= 3
    if not quiet:
        print("  %s: central fit residual on its range %.1e; %d states in 250.47-259.7, %d enter (|lam - R| >= 3 spread); "
              "%d windows; windowed max of |lam - R| p_min from %.2e to %.2e, a factor %.3g: %s" % (
                  label, Rs[(8, 150.0)][1], len(rows), len(ent), len(wm), min(wm) if wm else 0, max(wm) if wm else 0,
                  rg, "the law holds (<= 3)" if ok else "the law does NOT hold (> 3)"))
    return ok, dict(rows=rows, range=rg, windows=len(wm), fit_residual=Rs[(8, 150.0)][1])


def cmd_law(raw):
    st = chain_states(raw)
    pm = pmin_of(raw)
    bb = backbone(raw, 150.0)
    past = [(D, v) for D, v, _ in st if D > FRONT + 1e-9]
    grid = bb + past
    Rc = fit_R(bb, 8, 150.0)[0]
    tab = [dict(D=float(D), min_pivot=pm[round(float(D), 6)]) for D, _ in past if 245.6 <= D <= 255.8
           and round(float(D), 6) in pm]
    pf = FA.plant_funcs(tab)
    i_at = min(range(len(grid)), key=lambda i: abs(grid[i][0] - SCREEN_AT))
    verdict, out = True, dict(plants={})
    print("plants (truth R + term on the real grid: %d backbone states from %.2f, %d past the frontier):" % (
        len(bb), bb[0][0], len(past)))
    rng = random.Random(807)
    for name, expect in (("W", True), ("S", False), ("E", False)):
        if name == "E":
            vals = [(D, Rc(D) + mp.mpf("1e-10") * rng.choice((-1, 1))) for D, _ in grid]
        else:
            f = pf[name][0]
            unit = [(D, f(D)) for D, _ in grid]
            s = SCREEN_DEP / screen_dep(unit, i_at)
            vals = [(D, Rc(D) + s * u) for D, u in unit]
        nb = len(bb)
        ok, stats = law(vals[:nb], vals[nb:], pm, "plant " + name)
        good = ok == expect
        verdict &= good
        print("    expected %s: %s" % ("pass" if expect else "fail", "as expected" if good else "NOT AS EXPECTED"))
        out["plants"][name] = dict(ok=ok, expected=expect, range=stats["range"])
    print("real lam_g:")
    ok, stats = law(bb, past, pm, "real")
    out["real"] = dict(ok=ok, **stats)
    print("   D         lam - R      spread     p_min     |lam - R| p_min  enters")
    for r in stats["rows"]:
        print("  %9.4f %+.3e %.1e %9.2e %10.2e  %s" % (r["D"], r["dl"], r["spread"], r["p_min"],
                                                    abs(r["dl"]) * r["p_min"], "yes" if r["enters"] else "no"))
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    sys.exit(0 if verdict else 1)


def cmd_screen(raw):
    st = chain_states(raw)
    pm = pmin_of(raw)
    bb = backbone(raw, 0.0)
    past = [(D, v) for D, v, _ in st if D > FRONT + 1e-9]
    pts = sorted(bb + past)
    Rc = fit_R(backbone(raw, 150.0), 8, 150.0)[0]
    dl = [(D, v - Rc(D)) for D, v in pts]
    Rv = [(D, Rc(D)) for D, _ in pts]
    rows = []
    for i, (D, v) in enumerate(pts):
        if not LO <= D <= HI:
            continue
        rows.append(dict(D=float(D), screen=screen_dep(pts, i), screen_dl=screen_dep(dl, i), screen_R=screen_dep(Rv, i),
                         p_min=pm.get(round(float(D), 6))))
    print("   D        screen(lam)   screen(lam - R)  difference   screen(R)    screen(lam) p_min")
    for r in rows:
        print("  %9.4f %+.3e  %+.3e  %+.1e  %+.1e  %s" % (r["D"], r["screen"], r["screen_dl"], r["screen"] - r["screen_dl"],
                                                       r["screen_R"], "%.2e" % (abs(r["screen"]) * r["p_min"])
                                                       if r["p_min"] else "-"))
    agree = max(abs(r["screen"] - r["screen_dl"]) - abs(r["screen_R"]) for r in rows)
    print("  %d states; largest |screen(lam) - screen(lam - R)| - |screen(R)| = %.1e (<= 0 within rounding: the "
          "departures are lam - R's own)" % (len(rows), agree))
    rr = [r for r in rows if r["p_min"]]
    wm = []
    for r0 in rr:
        if r0["D"] + 3 > rr[-1]["D"] + 1e-9:
            break
        wm.append(max(abs(r["screen"]) * r["p_min"] for r in rr if r0["D"] <= r["D"] < r0["D"] + 3))
    print("  the literal reading, screen departure x p_min: windowed max from %.2e to %.2e, a factor %.3g: %s" % (
        min(wm), max(wm), max(wm) / min(wm), "within 3" if max(wm) / min(wm) <= 3 else "NOT within 3"))
    if "out" in args:
        json.dump(dict(rows=rows, literal_range=max(wm) / min(wm)), open(args["out"], "w"), indent=1)


def cmd_pred(raw):
    rows = []
    for d in FA.FOLDERS:
        for f in glob.glob(os.path.join(raw, "I603-win2", d, "pred_pin3_D*.npz")):
            sf = f.replace("pred_pin3", "anm_pin3")
            if os.path.exists(sf):
                Dp, vp = lam_of(f)
                Ds, vs = lam_of(sf)
                if abs(Dp - Ds) < mp.mpf("1e-30"):
                    rows.append([float(Ds), vs, vp])
    rows.sort()
    if args.get("mutate") == "lam":
        k = min(range(len(rows)), key=lambda i: abs(rows[i][0] - 240))
        rows[k][1] += mp.mpf("1e-10")
        print("mutated: lam_g of the state at D %.4f + 1e-10" % rows[k][0])
    bad = [(D, float(vs - vp)) for D, vs, vp in rows if abs(vs - vp) > mp.mpf("1e-20")]
    big = max(abs(vs - vp) for _, vs, vp in rows)
    print("prediction check: %d states with a prediction, D %.4f to %.4f; largest |lam(state) - lam(prediction)| %.1e; "
          "%s" % (len(rows), rows[0][0], rows[-1][0], float(big),
                  "every state within 1e-20" if not bad else "%d above 1e-20, first at D %.4f (%+.1e)" % (len(bad), *bad[0])))
    if "out" in args:
        json.dump([[D, float(vs - vp)] for D, vs, vp in rows], open(args["out"], "w"), indent=1)
    sys.exit(0 if not bad else 1)


# ---- post hoc (not pre-registered) ----

def windows(rows, trunc):
    """windowed maxima of |dl| p_min over windows of 3 in D; trunc: also the windows cut short by the last state"""
    wm = []
    for r0 in rows:
        if r0["D"] + 3 > rows[-1]["D"] + 1e-9 and not trunc:
            break
        wm.append(max(abs(r["dl"]) * r["p_min"] for r in rows if r0["D"] <= r["D"] < r0["D"] + 3))
    return wm


def cmd_posthoc(raw):
    """(a) plants S and E applied past the frontier only, so that states enter and the factor statistic itself is
    tested; (b) the windows cut short by the last state included, for the real lam_g and plant W"""
    st = chain_states(raw)
    pm = pmin_of(raw)
    bb = backbone(raw, 150.0)
    past = [(D, v) for D, v, _ in st if D > FRONT + 1e-9]
    grid = bb + past
    Rc = fit_R(bb, 8, 150.0)[0]
    tab = [dict(D=float(D), min_pivot=pm[round(float(D), 6)]) for D, _ in past if 245.6 <= D <= 255.8
           and round(float(D), 6) in pm]
    pf = FA.plant_funcs(tab)
    i_at = min(range(len(grid)), key=lambda i: abs(grid[i][0] - SCREEN_AT))
    rng = random.Random(807)
    out = {}
    nb = len(bb)
    for name in ("W", "S", "E", "real"):
        if name == "real":
            vals = grid
        elif name == "E":
            vals = [(D, Rc(D) + (mp.mpf("1e-10") * rng.choice((-1, 1)) if i >= nb else 0)) for i, (D, _) in enumerate(grid)]
        else:
            f = pf[name][0]
            unit = [(D, f(D)) for D, _ in grid]
            sc = SCREEN_DEP / screen_dep(unit, i_at)
            vals = [(D, Rc(D) + (sc * u if (i >= nb or name == "W") else 0)) for i, (D, u) in enumerate(unit)]
        ok, stats = law(vals[:nb], vals[nb:], pm, "%s%s" % (name, " (past the frontier only)" if name in "SE" else ""))
        ent = [r for r in stats["rows"] if r["enters"]]
        wt = windows(ent, True) if ent else []
        rt = max(wt) / min(wt) if wt else float("nan")
        print("    with the windows cut short by the last state too: %d windows, a factor %.3g" % (len(wt), rt))
        out[name] = dict(ok=ok, range=stats["range"], range_trunc=rt, entered=len(ent))
    # Delta's and lam_g's zeros (#805's chain - series against lam - R): the sign changes of each
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


if __name__ == "__main__":
    cmd = pos[0] if pos else ""
    if cmd in ("jet", "law", "screen", "pred", "posthoc"):
        dict(jet=cmd_jet, law=cmd_law, screen=cmd_screen, pred=cmd_pred, posthoc=cmd_posthoc)[cmd](pos[1])
        sys.exit(0)
    raise SystemExit(__doc__)
