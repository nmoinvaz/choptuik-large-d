"""The truncation of the 40 x 80 grid of Ecker, Ecker and Grumiller (EEG, arXiv:2611.07007v1) against D (issue #106):
delta(D) = Delta_RT - Delta_hp on rt_run.py's 40x79 grid, fitted in D, extrapolated to larger D, and compared with EEG's
published departure from our Delta.

    python3 solver/rt/rt_trunc_fit.py PATH.jsonl [--second B1.jsonl,B2.jsonl,...] [--at 80,100,150,190]
        [--check_D 20.043653,40.020065,60.001952] [--compare hp/data/delta_published_compare.csv]
    python3 solver/rt/rt_trunc_fit.py --planted [--mutate=rate]

PATH is the continuation's records (start A); --second names re-solves from other starts (B) at some of its D, whose
Delta against A's is the two-start spread. For each D the last converged unmutated mp record is used.

Fits (the criteria of the raw folder I106-mac/README.md, written before the runs): ln|delta| against D by the models
E: a + b D, P: a + b ln D, Q: a + b D + c D^2, on the windows [20, 60], [30, 60], [40, 60], [50, 60], each on the points
at and above the last sign change of delta, least squares in ln|delta|. Each fit is extrapolated to --at. The spread
over the 12 fits, and over the primary set (E and P on [30, 60] and [40, 60]), is an estimator range: a spread over fit
choices, not an error bar.

EEG's departure d_EEG(D) = Delta_EEG - Delta_ours is minus the column diff of the compare file (rows EEG2026), averaged
over the windows 55-65, 90-110, 140-160, 180-200, with the standard error of each mean from the scatter of the rows
(EEG print six decimals). The verdict at D = 60 uses the measured delta at the last check D; at the larger D the
estimator range. "reproduces": delta has EEG's sign (positive) and overlaps [d/3, 3d]; "far below": its upper end
< d/10; "far above": its lower end > 10 d; otherwise "between". As written in the criteria, a delta of the opposite sign
(negative) counts as "far below", since it cannot produce a positive departure; the table marks it "opposite sign".

--planted runs the same fit and verdict code on synthetic delta(D) (exponential and power growth, 1% noise) on the
continuation's D points and on synthetic EEG rows (a known departure plus six-decimal rounding of our values), and on
planted verdict cases; it exits 1 unless the rate comes back within 2%, the extrapolation to 190 within a factor 1.5,
the window means within 3 standard errors, and every planted verdict is classified as planted. --mutate=rate makes the
checker expect a rate 10% off the planted one, and the run must then exit 1.
"""
import os
import sys
import csv
import json
import math
import argparse
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
# the hp rows of the continuation's path (I106-mac/run_cont.sh), 20.043653 first
PATH_D = [20.043653, 20.991519, 21.912601, 23.043278, 23.952249, 25.07972, 26.07972, 26.983581, 27.95546, 28.99264,
          29.984746, 30.987328, 32.021319, 32.987798, 33.991236, 34.993266, 35.953229, 36.965729, 38.020417, 39.001863,
          40.020065, 40.952548, 42.04394, 42.998908, 43.98116, 44.963411, 46.001135, 47.008218, 48.015301, 49.022384,
          50.007491, 50.980148, 51.991398, 52.991398, 53.991398, 54.996085, 56.01171, 56.99989, 58.00599, 58.99805,
          60.001952]
MODELS = ("E", "P", "Q")
WINDOWS = ((20, 60.5), (30, 60.5), (40, 60.5), (50, 60.5))
PRIMARY = {("E", 1), ("E", 2), ("P", 1), ("P", 2)}
EEG_WINDOWS = ((55, 65, 60), (90, 110, 100), (140, 160, 150), (180, 200, 190))


def basis(model, D):
    D = np.asarray(D, float)
    if model == "E":
        return np.c_[np.ones_like(D), D]
    if model == "P":
        return np.c_[np.ones_like(D), np.log(D)]
    return np.c_[np.ones_like(D), D, D ** 2]


def same_sign_tail(D, d):
    """The points at and above the last sign change of d (D increasing)."""
    s = np.sign(d)
    k = 0
    for i in range(1, len(d)):
        if s[i] != s[i - 1]:
            k = i
    return D[k:], d[k:]


def fit_all(D, d, at):
    """Every (model, window) fit of ln|d|: a list of dicts with the coefficients, the rate and the extrapolations."""
    D, d = np.asarray(D, float), np.asarray(d, float)
    Dt, dt = same_sign_tail(D, d)
    sgn = float(np.sign(dt[-1]))
    out = []
    for m in MODELS:
        for wi, (lo, hi) in enumerate(WINDOWS):
            sel = (Dt >= lo) & (Dt <= hi)
            npar = 3 if m == "Q" else 2
            if sel.sum() < npar + 1:
                out.append(dict(model=m, window=wi, lo=lo, n=int(sel.sum()), ok=False))
                continue
            A = basis(m, Dt[sel])
            y = np.log(np.abs(dt[sel]))
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            rms = float(np.sqrt(np.mean((A @ c - y) ** 2)))
            ext = {float(x): float(sgn * np.exp(basis(m, [x])[0] @ c)) for x in at}
            out.append(dict(model=m, window=wi, lo=lo, n=int(sel.sum()), ok=True, coef=[float(v) for v in c],
                            rate=float(c[1]), rms=rms, ext=ext, primary=(m, wi) in PRIMARY, D0=float(Dt[sel][0])))
    return out, sgn, float(Dt[0])


def ranges(fits, at):
    """{D: (min, max) over all fits, (min, max) over the primary set}."""
    r = {}
    for x in at:
        allv = [f["ext"][x] for f in fits if f["ok"]]
        pri = [f["ext"][x] for f in fits if f["ok"] and f["primary"]]
        r[x] = ((min(allv), max(allv)) if allv else None, (min(pri), max(pri)) if pri else None)
    return r


def eeg_windows(path, override=None):
    """{centre: (mean, sem, n)} of d_EEG = Delta_EEG - Delta_ours over EEG_WINDOWS. override: {D: d} replaces the rows."""
    if override is None:
        D, d = [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["published_ref"] == "EEG2026":
                    D.append(float(r["D"]))
                    d.append(-float(r["diff"]))
        D, d = np.array(D), np.array(d)
    else:
        D, d = np.array(sorted(override)), np.array([override[k] for k in sorted(override)])
    out = {}
    for lo, hi, c in EEG_WINDOWS:
        m = (D >= lo) & (D <= hi)
        v = d[m]
        out[c] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))), int(m.sum()))
    return out


def classify(lo, hi, dE):
    """The verdict for delta in [lo, hi] (lo = hi for a measured value) against EEG's departure dE > 0."""
    lo, hi = min(lo, hi), max(lo, hi)
    if hi < dE / 10:
        return "far below"
    if lo > 10 * dE:
        return "far above"
    if lo > 0 and hi >= dE / 3 and lo <= 3 * dE:
        return "reproduces"
    return "between"


def overall(verdicts):
    """verdicts: {60: v, 100: v, 150: v, 190: v}."""
    big = [verdicts[c] for c in (100, 150, 190)]
    if verdicts[60] == "reproduces" and sum(v == "reproduces" for v in big) >= 2:
        return "evidence for truncation"
    if all(v == "far below" for v in verdicts.values()):
        return "evidence against truncation (far below EEG's departures)"
    if all(v == "far above" for v in verdicts.values()):
        return "far above: the collocation at EEG's grid is far less accurate than their values, so it cannot measure " \
               "their truncation"
    return "not decided (see the verdict per D)"


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("solver") == "mp" and r.get("ok") and not r.get("mutate") and r.get("diff") is not None:
                out[round(r["D"], 6)] = r
    return out


def report(D, d, at, eeg, measured_D):
    fits, sgn, Dtail = fit_all(D, d, at)
    print(f"\nfits of ln|delta| (points at and above the last sign change, D >= {Dtail}, sign {sgn:+.0f}):")
    print("| model | window | n | rate | rms of ln | " + " | ".join(f"delta({x:g})" for x in at) + " |")
    print("|---|---|---|---|---|" + "---|" * len(at))
    for f in fits:
        if not f["ok"]:
            print(f"| {f['model']} | [{f['lo']}, 60] | {f['n']} | too few points |" + " |" * len(at))
            continue
        rate = (f"{f['rate'] * 10 / math.log(10):.3f} decades per 10 D" if f["model"] != "P" else
                f"power {f['rate']:.2f}") if f["model"] != "Q" else f"b {f['coef'][1]:.3g}, c {f['coef'][2]:.3g}"
        print(f"| {f['model']}{' (primary)' if f['primary'] else ''} | [{f['lo']}, 60] | {f['n']} | {rate} | "
              f"{f['rms']:.2f} | " + " | ".join(f"{f['ext'][x]:+.2e}" for x in at) + " |")
    rg = ranges(fits, at)
    print("\nestimator range of delta (a spread over fit choices, not an error bar):")
    print("| D | primary set (E, P on [30, 60], [40, 60]) | all 12 fits |")
    print("|---|---|---|")
    for x in at:
        a_, p_ = rg[x]
        print(f"| {x:g} | {p_[0]:+.2e} to {p_[1]:+.2e} | {a_[0]:+.2e} to {a_[1]:+.2e} |")
    ver = {}
    i = int(np.argmin(np.abs(np.asarray(D) - measured_D)))
    if abs(D[i] - measured_D) > 1e-6:
        print(f"\nno record at D = {measured_D}; the verdict needs it")
        return fits, ranges(fits, at), None, None
    dm = d[i]
    print("\nEEG's departure d_EEG = Delta_EEG - Delta_ours (window mean, +- its standard error from the rounding scatter, "
          "not a bound) against delta:")
    print("| centre | window | rows | d_EEG | delta (measured at 60, else primary range; all-fit range) | verdict "
          "(primary) | verdict (all fits) |")
    print("|---|---|---|---|---|---|---|")
    ver_all = {}
    for lo, hi, c in EEG_WINDOWS:
        m, se, n = eeg[c]
        if c == 60:
            v = classify(dm, dm, m)
            ver[c] = ver_all[c] = v
            v += " (opposite sign)" if dm < 0 else ""
            print(f"| {c} | {lo}-{hi} | {n} | {m:+.2e} +- {se:.1e} | {dm:+.2e} (measured, D = {measured_D}) | {v} | {v} |")
        else:
            x = min(at, key=lambda t: abs(t - c))
            a_, p_ = rg[x]
            ver[c] = classify(p_[0], p_[1], m)
            ver_all[c] = classify(a_[0], a_[1], m)
            neg = lambda r: " (opposite sign)" if max(r) < 0 else ""
            print(f"| {c} | {lo}-{hi} | {n} | {m:+.2e} +- {se:.1e} | {p_[0]:+.2e} to {p_[1]:+.2e}; {a_[0]:+.2e} to "
                  f"{a_[1]:+.2e} | {ver[c]}{neg(p_)} | {ver_all[c]}{neg(a_)} |")
    print(f"\noverall (primary set): {overall(ver)}")
    print(f"overall (all fits): {overall(ver_all)}")
    return fits, rg, ver, ver_all


def beyond(D, d, paths, compare, half=5.0):
    """Outside the criteria: measured delta past the fit window against the 12 fits' extrapolation to the same D and
    against EEG's departure averaged over D +- half (its standard error from the rounding scatter)."""
    R = {}
    for p in paths.split(","):
        R.update({x: r for x, r in load(p).items() if x > WINDOWS[0][1]})
    if not R:
        return
    Db = sorted(R)
    fits, _, _ = fit_all(D, d, Db)
    rg = ranges(fits, Db)
    rows = []
    with open(compare) as f:
        for r in csv.DictReader(f):
            if r["published_ref"] == "EEG2026":
                rows.append((float(r["D"]), -float(r["diff"])))
    print("\npast the fit window (outside the criteria): measured delta, the fits' extrapolation, EEG's departure")
    print("| D | delta measured | primary range | all-fit range | d_EEG (D +- 5) | rows | delta / d_EEG |")
    print("|---|---|---|---|---|---|---|")
    for x in Db:
        v = [e for (De, e) in rows if abs(De - x) <= half]
        m, se = (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v)))) if len(v) > 1 else (float("nan"), 0)
        a_, p_ = rg[x]
        dm = R[x]["diff"]
        print(f"| {x} | {dm:+.3e} | {p_[0]:+.2e} to {p_[1]:+.2e} | {a_[0]:+.2e} to {a_[1]:+.2e} | {m:+.2e} +- {se:.1e} |"
              f" {len(v)} | {dm / m:.2f} |")


def planted(mutate):
    rng = np.random.default_rng(106)
    D = np.array(PATH_D)
    at = [80.0, 100.0, 150.0, 190.0]
    failed = False
    # 1. exponential growth, E recovers the rate
    A, b = 1e-11, 0.21
    d = A * np.exp(b * (D - 20)) * (1 + 0.01 * rng.standard_normal(len(D)))
    fits, sgn, _ = fit_all(D, d, at)
    f = next(f for f in fits if f["model"] == "E" and f["window"] == 1)
    expect = b * (1.1 if mutate == "rate" else 1.0)
    ok_r = abs(f["rate"] / expect - 1) < 0.02
    true190 = A * math.exp(b * 170)
    ok_x = 1 / 1.5 < f["ext"][190.0] / true190 < 1.5
    print(f"planted exponential: rate {f['rate']:.5f} against {expect:.5f} ({'ok' if ok_r else 'FAIL'}); delta(190) "
          f"{f['ext'][190.0]:.3e} against {true190:.3e} ({'ok' if ok_x else 'FAIL'})")
    failed |= not (ok_r and ok_x)
    # 2. power growth, P recovers the exponent; negative sign below D = 25 (a sign change, then the tail)
    A, p = 1e-12, 9.0
    d = A * (D / 20) ** p * (1 + 0.01 * rng.standard_normal(len(D)))
    d[D < 25] *= -1
    fits, sgn, Dt = fit_all(D, d, at)
    f = next(f for f in fits if f["model"] == "P" and f["window"] == 1)
    ok_r = abs(f["rate"] / p - 1) < 0.02 and Dt > 25 and sgn > 0
    true190 = A * (190 / 20) ** p
    ok_x = 1 / 1.5 < f["ext"][190.0] / true190 < 1.5
    print(f"planted power: exponent {f['rate']:.4f} against {p} ({'ok' if ok_r else 'FAIL'}), tail from D = {Dt}; "
          f"delta(190) {f['ext'][190.0]:.3e} against {true190:.3e} ({'ok' if ok_x else 'FAIL'})")
    failed |= not (ok_r and ok_x)
    # 3. EEG window means: a planted departure g(D) on our values, rounded to six decimals as EEG print
    ours = {}
    with open(os.path.join(REPO, "hp", "data", "delta_published_compare.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["published_ref"] == "EEG2026":
                ours[float(r["D"])] = float(r["ours_Delta"])
    g = lambda x: 3e-8 * max(0.0, x - 55.0)
    dep = {x: round(v + g(x), 6) - v for x, v in ours.items()}
    w = eeg_windows(None, override=dep)
    for lo, hi, c in EEG_WINDOWS:
        xs = [x for x in ours if lo <= x <= hi]
        tm = float(np.mean([g(x) for x in xs]))
        m, se, n = w[c]
        ok = abs(m - tm) < 3 * se
        failed |= not ok
        print(f"planted EEG window {lo}-{hi}: mean {m:+.3e} +- {se:.1e} against {tm:+.3e} ({'ok' if ok else 'FAIL'})")
    # 4. the verdict rule on planted cases
    cases = [((1e-6, 1e-6), 1e-6, "reproduces"), ((3e-7, 2e-6), 1e-6, "reproduces"), ((1e-9, 5e-8), 1e-6, "far below"),
             ((2e-5, 1e-3), 1e-6, "far above"), ((-1e-6, -1e-6), 1e-6, "far below"), ((1e-7, 2e-7), 1e-6, "between")]
    for (lo, hi), dE, want in cases:
        got = classify(lo, hi, dE)
        failed |= got != want
        print(f"planted verdict: delta {lo:+.1e} to {hi:+.1e} against d_EEG {dE:.1e}: {got} ({'ok' if got == want else 'FAIL'})")
    v = overall({60: "reproduces", 100: "reproduces", 150: "between", 190: "reproduces"})
    failed |= v != "evidence for truncation"
    v2 = overall({60: "far below", 100: "far below", 150: "far below", 190: "far below"})
    failed |= not v2.startswith("evidence against")
    print(f"planted overall: {v}; {v2}")
    print(f"exit {1 if failed else 0}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--second", default="")
    ap.add_argument("--at", default="80,100,150,190")
    ap.add_argument("--check_D", default="20.043653,40.020065,60.001952")
    ap.add_argument("--compare", default=os.path.join(REPO, "hp", "data", "delta_published_compare.csv"))
    ap.add_argument("--beyond", default="", help="records past D = 60 (outside the criteria): measured delta against "
                    "the fits' extrapolation and EEG's departure near each D")
    ap.add_argument("--planted", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["rate"])
    a = ap.parse_args()
    if a.planted:
        sys.exit(planted(a.mutate))
    if not a.path:
        ap.error("give PATH.jsonl or --planted")
    at = [float(v) for v in a.at.split(",")]
    A = load(a.path)
    if not A:
        print("no converged records")
        sys.exit(1)
    Bs = [load(p) for p in a.second.split(",") if p]
    D = np.array(sorted(x for x in A if x <= WINDOWS[0][1]))        # the fits use D <= 60 only
    d = np.array([A[x]["diff"] for x in D])
    grid = next(iter(A.values()))["grid"]
    print(f"grid {grid[0]}x{grid[1]}, {len(D)} converged D from {D[0]} to {D[-1]} ({a.path})")
    print("| D | Delta_RT | Delta_RT - Delta_hp | Newton steps | final max|F| |")
    print("|---|---|---|---|---|")
    for x in D:
        r = A[x]
        print(f"| {x} | {r['Delta_str'][:22]} | {r['diff']:+.3e} | {len(r['newton']) - 1} | {r['newton'][-1]:.1e} |")
    failed = False
    checks = [round(float(v), 6) for v in a.check_D.split(",") if v]
    print("\nat the check D, start A (this path) and the other starts:")
    print("| D | A - hp | other start - hp | A - other | file |")
    print("|---|---|---|---|---|")
    import flint
    flint.ctx.prec = 192
    for x in checks:
        if x not in A:
            print(f"| {x} | missing | | | |")
            failed = True
            continue
        found = False
        for p, B in zip([p for p in a.second.split(",") if p], Bs):
            if x in B:
                found = True
                dab = float((flint.arb(A[x]["Delta_str"]) - flint.arb(B[x]["Delta_str"])).mid())
                print(f"| {x} | {A[x]['diff']:+.3e} | {B[x]['diff']:+.3e} | {dab:+.2e} | {p.rsplit('/', 1)[-1]} |")
        if not found:
            print(f"| {x} | {A[x]['diff']:+.3e} | no second start | | |")
    eeg = eeg_windows(a.compare)
    report(D, d, at, eeg, checks[-1])
    if a.beyond:
        beyond(D, d, a.beyond, a.compare)
    print(f"exit {1 if failed else 0}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
