#!/usr/bin/env python3
"""Tie the headline numbers quoted in README.md, the paper and the proof notes to the files they come from.

Each check recomputes a quoted number from its source (a data file, a log, a saved run, or arithmetic on other quoted
numbers), compares it at the quoted precision, and confirms that the quoted text is really in the documents. Numbers
are compared as numbers, never as formatted strings. `--selftest` proves every check can fail: it shifts each
recomputed value by one unit in the last quoted digit and demands that the check then rejects it. `--full` adds the
slow checks (the Delta_inf fit grid, about 5 minutes).

    python3 check_numbers.py [--selftest] [--full]       # exit status 0 only if every check passes
"""
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = {name: open(os.path.join(ROOT, name), encoding="utf-8").read()
        for name in ("README.md", "paper/physics/main.tex", "paper/math/main.tex", "proof/A_certify.md", "GAMMA.md",
                     "notes/PLANTED_SELECTION.md", "notes/PROOF_QUEUE.md", "proof/P5_scope.md",
                     "notes/QUAD_DOUBLE.md",
                     "notes/SOLVER_HP.md", "notes/LOW_D.md", "notes/INNER_STOKES.md", "notes/FORMULA_FORMS.md", "notes/LARGE_D_WALL.md",
                     "notes/ECHO_ANALOGUES.md", "notes/REVIEW_GHMG25.md", "notes/MODE_COUNT.md", "matter/SIGMA_DSS.md",
                     "notes/HELD_HARMONICS.md", "notes/CONSISTENCY_AUDIT.md", "PROOF.md", "notes/FIRST_ORDER.md",
                     "matter/CSS.md", "matter/SIGMA.md")}


def quoted_in(doc, text):
    return text in DOCS[doc]


def rnd(x, places):
    return float(f"{x:.{places}f}")


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# Each check returns a list of items (label, computed value, quoted value, unit, documents-and-texts).
# An item passes when |computed - quoted| <= unit/2 (the quoted value is the computed one rounded) and every text is
# present in its document.

@check
def data_points():
    rows = [r for r in list(open(os.path.join(ROOT, "hp/data/delta_dd.csv")))[1:] if ",lowD" not in r]
    D = [float(r.split(",")[0]) for r in rows]
    txt = [("README.md", "Δ is computed at 1199 points from D = 8.25 to 67.77"),
           ("README.md", "The 857 points up to D = 61.03 fit smooth")]
    fit = [d for d in D if d <= 61.03]
    return [("double-double points", len(D), 1199, 1, txt),
            ("lowest D", min(D), 8.25, 0.01, txt),
            ("highest D", max(D), 67.77, 0.01, txt),
            ("points up to the fits' end", len(fit), 857, 1, txt),
            ("the fits' end", max(fit), 61.03, 0.01, txt)]


@check
def formula_dd():
    """README, first graph and "One formula": the rational formula (solver/formula_all.py) against the double-double
    points above the f64 fit's end (hp/data/delta_dd.csv), D = 30.2-67.77"""
    sys.path.insert(0, os.path.join(ROOT, "solver"))
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "solver", "data"))         # formula_all's loaders read the scan files from here
    try:
        import formula_all
        rows = [r.split(",") for r in list(open(os.path.join(ROOT, "hp/data/delta_dd.csv")))[1:]]
        miss = lambda lo, hi: max(abs(formula_all.delta(float(c[0])) - (float(c[1]) + float(c[2]))) for c in rows  # noqa: E731
                                  if lo <= float(c[0]) <= hi and ",lowD" not in ",".join(c))
        err = miss(30.2001, 99.0)
        bands = (miss(8.25, 21.0), miss(21.0, 30.0), miss(30.0, 61.03))
    finally:
        os.chdir(cwd)
    txt = [("README.md", "and Δ to 5.8×10⁻⁸ up to D = 67.77"),
           ("README.md", "above D = 30.2 it matches them to 5.8×10⁻⁸"),
           ("README.md", "| 30.2 – 67.77, never seen by the fit | ≤ 5.8×10⁻⁸ |"),
           ("paper/physics/main.tex", "$5.8\\times10^{-8}$ to $D = 61$")]
    t2 = [("README.md", "2.7×10⁻⁹ or better on 8.25 ≤ D ≤ 21, 7.5×10⁻⁹ up to D = 30, and 5.8×10⁻⁸ from there to"),
          ("paper/physics.main.tex".replace(".main", "/main"), "the formula holds to $2.7\\times10^{-9}$ on\n$8.25$--$21$, $7.5\\times10^{-9}$ to $D = 30$")]
    return [("formula against the double-double points, D 30.2-67.77", err, 5.8e-8, 1e-9, txt),
            ("formula against the double-double points, D 8.25-21", bands[0], 2.7e-9, 1e-10, t2),
            ("formula against the double-double points, D 21-30", bands[1], 7.5e-9, 1e-10, t2),
            ("formula against the double-double points, D 30-61.03", bands[2], 5.8e-8, 1e-9, t2)]


@check
def t79_values():
    """README "Values": Delta and gamma at D = 5, 6, 7, 8 from T79 (hp/data/delta_dd_lowD.csv: Delta at the best M,
    gamma at M = 48), and the M 48 -> 64 changes the text quotes"""
    rows = [r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv")) if r[:1].isdigit()]
    best = {float(c[0]): c for c in rows if c[8] == "1"}
    gam = {float(c[0]): float(c[10]) for c in rows if c[1] == "48" and c[10]}
    delta = lambda D: float(best[D][2]) + float(best[D][3])  # noqa: E731
    txt = [("README.md", "| 5 | 3.2217019446 | 0.413126936 |"), ("README.md", "| 6 | 3.009483220493 | 0.430079481 |"),
           ("README.md", "| 7 | 2.84275320647 | 0.439991242 |"), ("README.md", "| 8 | 2.71179005444440 | 0.446716389 |"),
           ("README.md", "change from M = 48 to 64 is 8.7×10⁻¹² at D = 5, 3×10⁻¹⁴ at 6 and 10⁻²⁰ at 8")]
    return [("Delta(5)", delta(5.0), 3.2217019446, 1e-10, txt), ("Delta(6)", delta(6.0), 3.009483220493, 1e-12, txt),
            ("Delta(7)", delta(7.0), 2.84275320647, 1e-11, txt), ("Delta(8)", delta(8.0), 2.71179005444440, 1e-14, txt),
            ("gamma(5)", gam[5.0], 0.413126936, 1e-9, txt), ("gamma(6)", gam[6.0], 0.430079481, 1e-9, txt),
            ("gamma(7)", gam[7.0], 0.439991242, 1e-9, txt), ("gamma(8)", gam[8.0], 0.446716389, 1e-9, txt),
            ("M 48 -> 64 at D 5", abs(float(best[5.0][7])), 8.7e-12, 1e-13, txt),
            ("M 48 -> 64 at D 6", abs(float(best[6.0][7])), 3e-14, 1e-14, txt),
            ("M 48 -> 64 at D 8", math.log10(abs(float(best[8.0][7]))), -20, 1, txt)]


@check
def d5_two_codes():
    """README, D = 5 in the comparison with Ecker et al.: the f64 value and its estimate (solver/data_all.py low_d(),
    kc 31, 39, 47 extrapolated geometrically), the double-double values at M = 48 and 64 (hp/compare/dd_D5.log), and
    the gaps between them that the text quotes"""
    sys.path.insert(0, os.path.join(ROOT, "solver"))
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "solver", "data"))         # low_d reads the scan files from here
    try:
        import data_all
        f64, est, _ = data_all.low_d()[5.0]
    finally:
        os.chdir(cwd)
    log = open(os.path.join(ROOT, "hp/compare/dd_D5.log")).read()
    dd = {int(m.group(1)): float(m.group(2)) + float(m.group(3))
          for m in re.finditer(r"^D 5\.0 M (\d+): Delta ([0-9.]+) \(\+([-0-9.e]+)\)", log, re.M)}
    t1 = [("README.md", "mode count (3.221701945 ± 7×10⁻¹¹, the size of that extrapolation step")]
    t2 = [("README.md", "collocation points (3.22170194461806,\n    3.22170194462672), which agree with each other to 10⁻¹¹")]
    t3 = [("README.md", "comes from two independent codes, which agree to 2×10⁻¹⁰")]
    t4 = [("README.md", "The f64 value lies 1.5×10⁻¹⁰ above the M = 64 value,\n    about twice its estimate")]
    return [("f64 Delta(5)", f64, 3.221701945, 1e-9, t1),
            ("f64 Delta(5) estimate, the extrapolation step", est, 7e-11, 1e-11, t1),
            ("double-double Delta(5) at M 48", dd[48], 3.22170194461806, 1e-14, t2),
            ("double-double Delta(5) at M 64", dd[64], 3.22170194462672, 1e-14, t2),
            ("double-double M 48 - 64 (log10)", math.log10(abs(dd[48] - dd[64])), -11, 1, t2),
            ("largest gap of the two codes, M 48 and 64", max(abs(f64 - dd[48]), abs(f64 - dd[64])), 2e-10, 1e-10, t3),
            ("f64 above double-double M 64", f64 - dd[64], 1.5e-10, 1e-11, t4),
            ("that gap over the f64 estimate", (f64 - dd[64]) / est, 2, 1, t4)]


@check
def qd_values():
    """README, the quad-double Newton: Delta at D = 68.5, 69.5, 70.5 (hp/data/delta_qd.csv, hp/export_qd.py) to the
    double-double digits quoted, and the largest two-start spread there, the difference of the exact sums"""
    rows = {float(c[0]): c for c in (r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/delta_qd.csv"))) if c[0][:1].isdigit()}
    txt = [("notes/SOLVER_HP.md", "Δ(68.5) =\n  1.7899636306159916 − 1.016822×10⁻¹⁶, Δ(69.5) = 1.7879302979617628 + 7.083175×10⁻¹⁷ and Δ(70.5) = 1.7859529321182939 +\n  1.087438×10⁻¹⁶"),
           ("notes/SOLVER_HP.md", "to 3×10⁻²⁸")]
    out = []
    for D, hi, lo in ((68.5, 1.7899636306159916, -1.016822e-16), (69.5, 1.7879302979617628, 7.083175e-17), (70.5, 1.7859529321182939, 1.087438e-16)):
        c = rows[D]
        out += [(f"Delta({D}) hi", float(c[1]), hi, 4e-16, txt), (f"Delta({D}) lo", float(c[2]), lo, 1e-22, txt)]
    out.append(("largest two-start spread at 68.5-70.5", max(float(rows[D][6]) for D in (68.5, 69.5, 70.5)), 3e-28, 0.5e-28, txt))
    return out


@check
def lowD_305():
    """README "Below D = 4": T45's lowest point, D = 3.05 at M = 384, its step from M = 320, and T82's D = 3.045
    (hp/data/delta_dd_lowD.csv, hp/export_lowD.py reading the raw-data folder T82-win)"""
    rows = [r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv")) if r[:1].isdigit()]
    r = [c for c in rows if float(c[0]) == 3.05 and c[1] == "384"][0]
    txt = [("notes/LOW_D.md", "run reached D = 3.05 at M = 320/384, Δ = 2.4849271637881, where the step from M = 320 to 384 is 2.1×10⁻¹¹"),
           ("notes/LOW_D.md", "and the step from 320 to 384 is 2.1×10⁻¹¹, 56 times smaller")]
    step = float(r[7])
    prev = [c for c in rows if float(c[0]) == 3.05 and c[1] == "320"][0]
    r2 = [c for c in rows if float(c[0]) == 3.045 and c[1] == "384"][0]
    t2 = [("notes/LOW_D.md", "reached D = 3.045 at M = 320/384, Δ = 2.4464747274662, with the step from M = 320 to 384 6.7×10⁻¹¹, 3.2 times the step at 3.05")]
    return [("D 3.05 Delta", float(r[2]) + float(r[3]), 2.4849271637881, 1e-13, txt),
            ("D 3.05 step 320 -> 384", abs(step), 2.1e-11, 1e-12, txt),
            ("ratio of the steps 256/320 and 320/384", abs(float(prev[7]) / step), 56, 1, txt),
            ("D 3.045 Delta (T82)", float(r2[2]) + float(r2[3]), 2.4464747274662, 1e-13, t2),
            ("D 3.045 step 320 -> 384", abs(float(r2[7])), 6.7e-11, 1e-12, t2),
            ("ratio of the 3.045 and 3.05 steps 320 -> 384", abs(float(r2[7]) / step), 3.2, 0.1, t2)]


@check
def lowD_limits():
    """README "Below D = 4", the limit graph's paragraph: our polynomial in sqrt(D - 3) (hp/plot_lowD_values.C_DELTA)
    against Ecker et al.'s power law and scaling exponent (hp/data/published_lowD_limits.json) at D = 3.04375, 3.02
    and 3.005, and the T105 point's Delta (hp/data/delta_dd_lowD.csv at D = 3.04375, the best M)"""
    import json
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    from plot_lowD_values import C_DELTA, formula
    lim = json.load(open(os.path.join(ROOT, "hp/data/published_lowD_limits.json")))
    e, sc, p305 = lim["EEGJ2026_eq110"], lim["EEGJ2026_scaling"], lim["EEGJ2026_D3.05"]
    pw = lambda D: e["a"] * (D - 3) ** e["alpha"]  # noqa: E731
    sl = lambda D: p305["Delta"] / (p305["D"] - 3) ** sc["alpha"] * (D - 3) ** sc["alpha"]  # noqa: E731
    txt = [("README.md", "| 2.4364 | 2.4265 | 2.3826 |"), ("README.md", "| 3.02 | 2.20 | 2.16 | 1.88 |"), ("README.md", "| 3.005 | 1.96 | 1.75 | 1.24 |")]
    rows = [r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv")) if r[:1].isdigit()]
    t105 = sorted([c for c in rows if float(c[0]) == 3.04375], key=lambda c: int(c[1]))
    out = [("ours at 3.04375", formula(C_DELTA, 3.04375), 2.4364, 1e-4, txt),
           ("power law at 3.04375", pw(3.04375), 2.4265, 1e-4, txt),
           ("ours at 3.02", formula(C_DELTA, 3.02), 2.20, 0.01, txt),
           ("power law at 3.02", pw(3.02), 2.16, 0.01, txt),
           ("ours at 3.005", formula(C_DELTA, 3.005), 1.96, 0.01, txt),
           ("power law at 3.005", pw(3.005), 1.75, 0.01, txt),
           ("exponent 0.3 at 3.04375", sl(3.04375), 2.3826, 1e-4, txt),
           ("exponent 0.3 at 3.02", sl(3.02), 1.88, 0.01, txt),
           ("exponent 0.3 at 3.005", sl(3.005), 1.24, 0.01, txt),
           ("Eq. (110) as quoted", e["a"], 3.88, 1e-9, [("README.md", "Δ ≈ 3.88 (D − 3)^0.15")]),
           ("Bland 2007 point as cited", lim["Bland2007_D3.02"]["Delta"], 0.1379, 1e-9, [("README.md", "Δ = 0.1379 ± 0.0042")])]
    if t105:
        out.append(("T105 Delta at 3.04375 (M %s)" % t105[-1][1], float(t105[-1][2]) + float(t105[-1][3]), 2.436459, 1e-6,
                    [("README.md", "the T105 leg, M = 384, Δ = 2.436459")]))
    return out


# The published gamma, Delta and wiggle periods below D = 14 (#620), as copied from the papers' arXiv sources: ref, D,
# quantity, lower and upper end (equal unless a range), uncertainty. A frozen copy of hp/compare_gamma_published.py's
# rows, which guards the CSV against later edits; that copy's fidelity to the papers is the producer's --verify.
PUBLISHED_GAMMA = """
GCD1999 6 gamma 0.424 0.424 none_stated; GCD1999 6 gamma_mass 1.27 1.27 none_stated
GCD1999 6 Delta 3.03 3.03 none_stated; Birukou2002 4 gamma 0.36 0.36 none_stated
Birukou2002 4 gamma 0.35 0.35 none_stated; Birukou2002 5 gamma 0.41 0.41 none_stated
Birukou2002 5 gamma 0.52 0.52 none_stated; Birukou2002 5 gamma 0.49 0.49 none_stated
Birukou2002 6 gamma 0.44 0.44 none_stated; Birukou2002 6 gamma 0.41 0.41 none_stated
Husain2003 4 gamma 0.370 0.375 range; Husain2003 4 gamma 0.37 0.38 range; Husain2003 4 gamma 0.37 0.39 range
Husain2003 4 gamma 0.36 0.38 range; Husain2003 4 gamma 0.37 0.38 range; Husain2003 4 gamma 0.36 0.38 range
Husain2003 4 gamma 0.37 0.40 range; Husain2003 4 gamma 0.3738 0.3738 none_stated
Husain2003 4 Delta 3.5 3.5 none_stated; SO2005 4 Delta 3.37 3.37 2%; SO2005 4 gamma 0.372 0.372 1%
SO2005 5 Delta 3.19 3.19 2%; SO2005 5 gamma 0.408 0.408 2%; SO2005 5 Delta_half 1.6 1.6 none_stated
SO2005 6 Delta 3.01 3.01 2%; SO2005 6 gamma 0.422 0.422 2%; SO2005 7 Delta 2.83 2.83 2%
SO2005 7 gamma 0.429 0.429 2%; SO2005 7 Delta_half 1.41 1.41 none_stated; SO2005 8 Delta 2.70 2.70 3%
SO2005 8 gamma 0.436 0.436 2%; SO2005 9 Delta 2.61 2.61 3%; SO2005 9 gamma 0.442 0.442 2%
SO2005 10 Delta 2.55 2.55 3%; SO2005 10 gamma 0.447 0.447 3%; SO2005 11 Delta 2.51 2.51 3%
SO2005 11 gamma 0.44 0.44 3%; Bland2005 3.5 gamma 0.349 0.349 +-0.003; Bland2005 4 gamma 0.374 0.374 +-0.002
Bland2005 4 Delta 3.40 3.40 +-0.1; Bland2005 5 gamma 0.412 0.412 +-0.004; Bland2005 5 Delta 3.10 3.10 +-0.1
Bland2005 6 gamma 0.430 0.430 +-0.003; Bland2005 6 Delta 2.98 2.98 +-0.1; Bland2005 7 gamma 0.441 0.441 +-0.004
Bland2005 7 Delta 2.96 2.96 +-0.1; Bland2005 8 gamma 0.446 0.446 +-0.004; Bland2005 8 Delta 2.77 2.77 +-0.1
Bland2005 9 gamma 0.453 0.453 +-0.003; Bland2005 9 Delta 2.63 2.63 +-0.1; Bland2005 10 gamma 0.456 0.456 +-0.004
Bland2005 10 Delta 2.50 2.50 +-0.1; Bland2005 11 gamma 0.459 0.459 +-0.004; Bland2005 11 Delta 2.46 2.46 +-0.1
Bland2005 12 gamma 0.462 0.462 +-0.005; Bland2005 12 Delta 2.44 2.44 +-0.1; Bland2005 13 gamma 0.463 0.463 +-0.004
Bland2005 13 Delta 2.40 2.40 +-0.1; Bland2005 14 gamma 0.465 0.465 +-0.004; BK2007 5 gamma 0.4131 0.4131 +-0.0001
BK2007 5 Delta 3.225 3.225 +-0.003; BK2007 5 gamma 0.4130 0.4130 +-0.0001; BK2007 5 Delta 3.2264 3.2264 +-0.0005
BK2007 5 gamma 0.4130 0.4130 +-0.0001; BK2007 5 Delta 3.2230 3.2230 +-0.0006; BK2007 5 gamma 0.4131 0.4131 +-0.0001
BK2007 5 Delta 3.2223 3.2223 +-0.0006; BK2007 5 gamma 0.4131 0.4131 +-0.0001; BK2007 5 Delta 3.2264 3.2264 +-0.0004
BK2007 5 gamma 0.4131 0.4131 +-0.0001; BK2007 5 Delta 3.2246 3.2246 +-0.0005; BK2007 5 gamma 0.4131 0.4131 +-0.0003
BK2007 5 gamma 0.4133 0.4133 +-0.0004; TK2011 4 gamma 0.378 0.378 +-0.002; TK2011 4 T 4.4 4.4 +-0.3
TK2011 4 gamma 0.379 0.379 +-0.002; TK2011 4 T 4.4 4.4 +-0.2; TK2011 5 gamma 0.413 0.413 +-0.002
TK2011 5 T 3.9 3.9 +-0.8; TK2011 5 gamma 0.416 0.416 +-0.002; TK2011 5 T 3.7 3.7 +-0.4
TK2011 6 gamma 0.429 0.429 +-0.003; TK2011 6 T 3.4 3.4 +-0.3; TK2011 6 gamma 0.428 0.428 +-0.002
TK2011 6 T 3.3 3.3 +-0.2; TK2011 7 gamma 0.440 0.440 +-0.005; TK2011 7 T 3.1 3.1 +-0.1
TK2011 7 gamma 0.440 0.440 +-0.006; TK2011 7 T 3.1 3.1 +-0.4; TK2011 8 gamma 0.44 0.44 +-0.02
TK2011 8 T 3.0 3.0 +-0.1; PVG2022 5 gamma 0.415 0.415 none_stated; PVG2022 5 Delta_half 1.6 1.6 none_stated
PVG2022 5 gamma 0.413 0.413 none_stated; PVG2022 5 gamma 0.4131 0.4131 none_stated
PVG2022 5 T_log10 1.7 1.7 none_stated; PVG2022 5 gamma 0.41 0.41 none_stated
"""

PUBLISHED_GAMMA_NOTE = {   # notes/REVIEW_GHMG25.md section 8: values, within their uncertainty, largest |diff| and its unit
    "GCD1999": ("| Garfinkle, Cutler & Duncan 1999 | 3 | 0 | 2.05×10⁻² (Δ(6)) |", 3, 0, 2.05e-2, 1e-4),
    "Birukou2002": ("| Birukou et al. 2002 | 7 | 1 | 1.07×10⁻¹ (γ(5), Gaussian data) |", 7, 1, 1.07e-1, 1e-3),
    "Husain2003": ("| Husain et al. 2003 | 9 | 7 | 5.45×10⁻² (Δ(4), Λ = −5, \"a rough estimate\") |", 9, 7, 5.45e-2, 1e-4),
    "SO2005": ("| Sorkin & Oren 2005 | 18 | 12 | 7.55×10⁻² (Δ(4)) |", 18, 12, 7.55e-2, 1e-4),
    "Bland2005": ("| Bland et al. 2005 | 22 | 19 | 1.22×10⁻¹ (Δ(5)) |", 22, 19, 1.22e-1, 1e-3),
    "BK2007": ("| Bland & Kunstatter 2007 | 14 | 7 | 4.70×10⁻³ (Δ(5), N = 512 and 1536) |", 14, 7, 4.70e-3, 1e-5),
    "TK2011": ("| Taves & Kunstatter 2011 | 18 | 12 | 2.07×10⁻¹ (T(4)) |", 18, 12, 2.07e-1, 1e-3),
    "PVG2022": ("| Porto Veronese & Gundlach 2022 | 6 | 5 | 1.09×10⁻² (Δ_Ψ(5)) |", 6, 5, 1.09e-2, 1e-4),
}


@check
def published_gamma():
    """notes/REVIEW_GHMG25.md, section 8 and D1, D8 and row 13: hp/data/gamma_published_compare.csv
    (hp/compare_gamma_published.py). Every published value and stated uncertainty against PUBLISHED_GAMMA, to one unit of
    its last printed digit; our value against the data files; diff and within recomputed; the counts the note quotes."""
    import csv
    num = lambda s: float(s.strip("+-%")) if re.fullmatch(r"[+-]*[0-9.]+%?", s.strip()) else float("nan")  # noqa: E731
    unit = lambda s: 10.0 ** -len(s.strip("+-%").partition(".")[2])  # noqa: E731
    want, seen = {}, {}
    for e in (e.split() for e in PUBLISHED_GAMMA.replace("\n", ";").split(";") if e.strip()):
        k = (e[0], float(e[1]), e[2])
        want[k + (seen.get(k, 0),)] = e[3:]
        seen[k] = seen.get(k, 0) + 1
    # ours, read here from the data files and not through the producer
    gam = {}
    for r in csv.DictReader(l for l in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv")) if not l.startswith("#")):
        if r["gamma"]:
            gam.setdefault(float(r["D"]), {})[int(r["M"])] = float(r["gamma"])
    gam = {D: byM[max(byM)] for D, byM in gam.items()}
    for fn in ("hp/data/gamma_dd.jsonl", "hp/data/local/gamma_dd.jsonl"):
        for l in open(os.path.join(ROOT, fn)):
            j = json.loads(l)
            if "gamma" in j:
                gam.setdefault(j["D"], j["gamma"])
    dl = {float(r["D"]): sum(float(r[f"Delta_q{i}"] or 0) for i in range(4))
          for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv")))}
    f = {"gamma": lambda D: gam[D], "gamma_mass": lambda D: (D - 3) * gam[D], "Delta": lambda D: dl[D],
         "Delta_half": lambda D: dl[D] / 2, "T": lambda D: dl[D] / (2 * gam[D]),
         "T_log10": lambda D: dl[D] / (2 * math.log(10) * gam[D])}
    rows = list(csv.DictReader(open(os.path.join(ROOT, "hp/data/gamma_published_compare.csv"))))
    out, seen, bad_ours, bad_diff, bad_within, by_ref, diffs = [], {}, 0, 0, 0, {}, {}
    for r in rows:
        k = (r["published_ref"], float(r["D"]), r["quantity"])
        n = seen.get(k, 0)
        seen[k] = n + 1
        lo, hi, unc = want.get(k + (n,), ("nan", "nan", "none_stated"))
        lab = f"{k[0]} {k[2]}({r['D']}) #{n + 1}"
        out.append((lab, num(r["published_lo"]), num(lo), unit(lo), []))
        if hi != lo:
            out.append((lab + " upper end", num(r["published_hi"]), num(hi), unit(hi), []))
        if unc not in ("none_stated", "range"):
            out.append((lab + " uncertainty", num(r["published_unc"]), num(unc), unit(unc), []))
        v = (num(lo) + num(hi)) / 2
        ua = ((num(hi) - num(lo)) / 2 if unc == "range" else 0.5 * unit(lo) if unc == "none_stated" else
              num(unc) / 100 * v if unc.endswith("%") else num(unc))
        ours = float(r["ours"])
        bad_ours += not abs(ours - f[k[2]](k[1])) <= 1e-10
        d = ours - v
        bad_diff += not abs(float(r["diff"]) - d) <= 0.5 * 10 ** (math.floor(math.log10(abs(d))) - 2) * (1 + 1e-6)
        bad_within += (r["within"] == "yes") != (abs(d) <= ua * (1 + 1e-9))
        c = by_ref.setdefault(k[0], [0, 0, 0.0])
        c[0] += 1
        c[1] += abs(d) <= ua * (1 + 1e-9)
        c[2] = max(c[2], abs(d))
        diffs.setdefault(k, []).append(d)
    out += [("rows of the CSV", len(rows), 97, 1, [("notes/REVIEW_GHMG25.md", "(97 values)")]),
            ("entries of PUBLISHED_GAMMA", len(want), 97, 1, []),
            ("rows whose ours differs from the data files by more than 1e-10", bad_ours, 0, 1, []),
            ("rows whose diff is not ours minus the published value", bad_diff, 0, 1, []),
            ("rows whose within is not |diff| <= the uncertainty", bad_within, 0, 1, [])]
    note = "notes/REVIEW_GHMG25.md"
    for ref, (text, n, k, m, u) in PUBLISHED_GAMMA_NOTE.items():
        c = by_ref.get(ref, [0, 0, float("nan")])
        out += [(f"{ref} values", c[0], n, 1, [(note, text)]), (f"{ref} within", c[1], k, 1, [(note, text)]),
                (f"{ref} largest |diff|", c[2], m, u, [(note, text)])]
    g6 = diffs[("GCD1999", 6.0, "gamma")][0]
    bland = max(abs(d) for (ref, D, q), ds in diffs.items() if ref == "Bland2005" and q == "gamma" and D >= 4 for d in ds)
    out += [("D1: gamma(6) above 0.424", g6, 0.0061, 1e-4, [(note, "0.0061 (1.4%) higher")]),
            ("D1: the same, percent", 100 * g6 / 0.424, 1.4, 0.1, [(note, "0.0061 (1.4%) higher")]),
            ("D8: Bland-Kunstatter Delta(5) above ours", -diffs[("BK2007", 5.0, "Delta")][0], 3.3e-3, 1e-4,
             [(note, "(hep-th/0702226v1, Eq. (3)) is 3.3×10⁻³ above")]),
            ("row 13: our Delta(5)/2", f["Delta_half"](5.0), 1.611, 1e-3, [(note, "our Δ(5)/2 = 1.611")]),
            ("row 13: our residual period Delta/(2 ln(10) gamma) at D 5", f["T_log10"](5.0), 1.6934, 1e-4,
             [(note, "gives Δ_res = Δ/(2 ln(10) γ) = 1.6934")]),
            ("Bland et al. gamma at D 4-14, largest |diff|", bland, 1.31e-3, 1e-5, [(note, "by at most 1.31×10⁻³ (D = 9)")]),
            ("Bland et al. gamma(3.5) above ours", -diffs[("Bland2005", 3.5, "gamma")][0], 0.0266, 1e-4,
             [(note, "is 0.0266 above our 0.3223856 (M = 128)")])]
    return out


@check
def echo_shapes():
    """README "What repeats with that period": the first-harmonic power fractions, the deviations from the sinusoid and
    the centre amplitudes of hp/data/echo_shapes.csv (hp/plot_echo.py on stored states of the raw-data repository)"""
    rows = [r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/echo_shapes.csv")) if r[:1].isdigit()]
    by = {float(r[0]): r for r in rows}
    txt = [("README.md", "| 45 – 73.27 | 99.2% to 99.4% | on the harmonic to within 0.14 of the peak"),
           ("README.md", "| 16.76 | 98% | skewed |"), ("README.md", "| 8 | 94% | skewed |"),
           ("README.md", "| 4 | 76% | sharpened, a deviation of 1.60 |"),
           ("README.md", "| 3.044 | 37% | a narrow pulse on a flat floor, with a centre amplitude of 674 against 0.32 at D = 73.27 |")]
    frac = lambda D: 100 * float(by[D][4])  # noqa: E731
    dev = lambda D: float(by[D][5])  # noqa: E731
    return [("first harmonic at 45.05 (%)", frac(45.050722), 99.2, 0.1, txt),
            ("first harmonic at 73.27 (%)", frac(73.265722), 99.4, 0.1, txt),
            ("largest deviation from D 45 up", max(dev(45.050722), dev(60.962646), dev(73.265722)), 0.14, 0.01, txt),
            ("first harmonic at 16.76 (%)", frac(16.76), 98, 1, txt),
            ("first harmonic at 8 (%)", frac(8.0), 94, 1, txt),
            ("first harmonic at 4 (%)", frac(4.0), 76, 1, txt),
            ("deviation at 4", dev(4.0), 1.60, 0.01, txt),
            ("first harmonic at 3.044 (%)", frac(3.044375), 37, 1, txt),
            ("centre amplitude at 3.044", float(by[3.044375][3]), 674, 1, txt),
            ("centre amplitude at 73.27", float(by[73.265722][3]), 0.32, 0.01, txt)]


@check
def hopf_law():
    """notes/ECHO_ANALOGUES.md, "The tests of C.1": the amplitude law and the shape law of hp/hopf_amp_mode.py's
    amplitude_law() on hp/data/echo_shapes.csv (the three states at D >= 40)"""
    import contextlib
    import io
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import hopf_amp_mode as H
    with contextlib.redirect_stdout(io.StringIO()):
        law = H.amplitude_law()
    f, rows = law["fits"], [r for r in law["rows"] if r["D"] >= 40]
    a, b = f["A_sqrtD_limit"], f["A_sqrtD_slope_1_over_D"]
    misfit = max(abs(r["A_sqrtD"] - (a + b / r["D"])) for r in rows)
    ratio = [r["rest"] / r["deviation"] ** 2 for r in rows]
    N = "notes/ECHO_ANALOGUES.md"
    txt = [(N, "gives A√D = 2.867, 2.790, 2.754"), (N, "A√D = 2.574 + 13.2/D to 2e-4, so A²D → 6.62 at 1/D = 0"),
           (N, "A raw power fit gives A ∝ D^−0.58"),
           (N, "0.0064, 0.0058, falling like D^−0.66, and 0.143, 0.129, 0.123, falling like D^−0.33"),
           (N, "fraction / deviation² = 0.383 to 0.385"), (N, "would be −0.044, −0.033, −0.027 at these D")]
    out = [("A sqrt(D) at %.2f" % r["D"], r["A_sqrtD"], q, 0.001, txt) for r, q in zip(rows, (2.867, 2.790, 2.754))]
    out += [("A sqrt(D) limit", a, 2.574, 0.001, txt), ("its 1/D slope", b, 13.2, 0.1, txt), ("line misfit", misfit, 2e-4, 1e-4, txt),
            ("A^2 D limit", a * a, 6.62, 0.01, txt), ("A exponent", f["A_exponent"], -0.58, 0.01, txt),
            ("power-fraction exponent", f["rest_exponent"], -0.66, 0.01, txt),
            ("deviation exponent", f["deviation_exponent"], -0.33, 0.01, txt)]
    out += [("power fraction at %.2f" % r["D"], r["rest"], q, 0.0001, txt) for r, q in zip(rows, (0.0079, 0.0064, 0.0058))]
    out += [("deviation at %.2f" % r["D"], r["deviation"], q, 0.001, txt) for r, q in zip(rows, (0.143, 0.129, 0.123))]
    out += [("fraction / deviation^2, least", min(ratio), 0.383, 0.001, txt), ("fraction / deviation^2, largest", max(ratio), 0.385, 0.001, txt)]
    out += [("-2/D at %.2f" % r["D"], -2 / r["D"], q, 0.001, txt) for r, q in zip(rows, (-0.044, -0.033, -0.027))]
    return out


@check
def formula_table():
    """README "The formula" and paper/physics Table "Coefficients": every tabulated coefficient equals the stored one
    (solver/data/formula_coefficients.json) in units of its last printed digit"""
    C = json.load(open(os.path.join(ROOT, "solver/data/formula_coefficients.json")))
    want = {"a": C["a"], "b": ["1"] + C["b"], "e": C["e"], "f": ["1"] + C["f"]}

    def parse(cell):
        s = cell.replace("$", "").replace("(fixed)", "").replace("−", "-").replace("\\times10^{", "e").replace("}", "")
        s = re.sub(r"\s*×\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "e" + m.group(1).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")), s)
        s = s.strip()
        return s if s else None

    def units_off(printed, exact):
        """|printed - exact| in units of the printed mantissa's last digit"""
        m = re.match(r"([+-]?)(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$", printed)
        frac, ex = (m.group(3) or ""), int(m.group(4) or 0)
        return abs(float(printed) - float(exact)) / 10.0 ** (ex - len(frac))

    out = []
    for doc, pat in (("README.md", r"^\| (\d) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| ([^|]*) \|$"),
                     ("paper/physics/main.tex", r"^(\d)\s*&\s*([^&]*?)\s*&\s*([^&]*?)\s*&\s*([^&]*?)\s*&\s*([^&]*?)\s*\\\\$")):
        rows = [m for m in (re.match(pat, l) for l in DOCS[doc].split("\n")) if m]
        rows = [m for m in rows if parse(m.group(2)) and re.match(r"[+-]\d\.\d{6,}", parse(m.group(2)))]
        worst, n = 0.0, 0
        for m in rows:
            k = int(m.group(1))
            for col, key in ((2, "a"), (3, "b"), (4, "e"), (5, "f")):
                v = parse(m.group(col))
                if v is None:
                    continue
                n += 1
                worst = max(worst, units_off(v, want[key][k]) if k < len(want[key]) else float("inf"))
        out.append((f"{doc}: {n} tabulated coefficients, worst distance in last-digit units", worst, 0.0, 1.0, [(doc, "1.6438975180")]))
        out.append((f"{doc}: tabulated coefficients counted", n, 23, 1, [(doc, "1.6438975180")]))
    return out


@check
def formula_refit():
    """README "The formula": a fresh fit stays within 1e-9 of the stored Delta and 1e-8 of the stored gamma on the fitted
    range and moves Delta(80) by about 1e-7 (0.5e-7 to 2e-7); the items are 1 when true, so a drift of the fit, or of
    the data behind it, fails them"""
    sys.path.insert(0, os.path.join(ROOT, "solver"))
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "solver", "data"))
    try:
        import formula_all
        r = formula_all.refit_distance()
    finally:
        os.chdir(cwd)
    txt = [("README.md", "the values by 10⁻⁹ on the fitted range and about 10⁻⁷ at\nD = 80"),
           ("paper/physics/main.tex", "moves the values by $10^{-9}$ on the fitted range and about $10^{-7}$ at $D = 80$")]
    return [("refit within 1e-9 of the stored Delta on 4-30.2 (1 = yes)", float(r["delta_fit_range"] <= 1e-9), 1, 0.1, txt),
            ("refit within 1e-8 of the stored gamma on 4-30.2 (1 = yes)", float(r["gamma_fit_range"] <= 1e-8), 1, 0.1, txt),
            ("refit - stored Delta(80) between 0.5e-7 and 2e-7 in size (1 = yes)", float(0.5e-7 <= abs(r["delta_80"]) <= 2e-7), 1, 0.1, txt)]


@check
def formula_cut():
    """README "One formula for every D": the cut-plane series of hp/data/formula_cut.json (hp/formula_cut_fit.py, 256
    bits; hp/formula_cut.py evaluates and checks it): its points, its largest residual in units of the points' errors,
    its number of coefficients, Delta_inf = the series at zeta = 0"""
    import json
    J = json.load(open(os.path.join(ROOT, "hp/data/formula_cut.json")))
    txt = [("README.md", "Its 29 coefficients"), ("README.md", "reproduce all 1214 points within 3.47 times their errors, and give Δ∞ = 1.643893745")]
    return [("formula_cut points", J["points"], 1214, 0.5, txt),
            ("formula_cut coefficients", len(J["c"]), 29, 0.5, txt),
            ("formula_cut largest |r|/sigma", J["max_abs_residual_over_sigma"], 3.47, 0.01, txt),
            ("formula_cut Delta_inf", float(J["Delta_inf"]), 1.643893745, 1e-9, txt)]


@check
def f64_dinf_range():
    """README "Main results": the f64 limit 1.64395 ± 0.00008, the centre of the range of the free limit c0 over the
    fits with chi2/dof <= 1 on D <= 30.2 and its half-width rounded up at the quoted digit (solver/final_fit.py
    --range, limit_range); the range's ends and the number of fits are those notes/CONSISTENCY_AUDIT.md quotes"""
    sys.path.insert(0, os.path.join(ROOT, "solver"))
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "solver", "data"))         # step_converged reads the scan files from here
    try:
        import final_fit
        D, y, s, _ = final_fit.step_converged()
        rows, ok = final_fit.limit_range(D, y, s)
    finally:
        os.chdir(cwd)
    lo, hi = min(ok), max(ok)
    up = math.ceil(round((hi - lo) / 2 / 1e-5, 6)) * 1e-5
    txt = [("README.md", "The f64 points gave 1.64395 ± 0.00008, the centre and\nhalf-width, rounded up, of the range"),
           ("README.md", "(variable 1/D, 1/(D − 1) or 1/(D − 2), degree 8–11,\nfrom D ≥ 10 or 12 to 30.2, χ²/dof ≤ 1)")]
    audit = [("notes/CONSISTENCY_AUDIT.md", "the range 1.643882–1.644027 of the free limit c₀ over the 19 of 24 fits")]
    return [("fit choices", len(rows), 24, 1, audit),
            ("fits with chi2/dof <= 1", len(ok), 19, 1, audit),
            ("f64 range, low end", lo, 1.643882, 1e-6, audit),
            ("f64 range, high end", hi, 1.644027, 1e-6, audit),
            ("f64 range, centre", (lo + hi) / 2, 1.64395, 1e-5, txt),
            ("f64 range, half-width rounded up", up, 0.00008, 1e-5, txt),
            ("1.64395 ± 0.00008 contains the range (1 = yes)", float(1.64395 - 0.00008 <= lo and hi <= 1.64395 + 0.00008),
             1, 0.1, txt)]


@check
def lowD_logslope():
    """README "The local exponent already separates them": the local exponent (D - 3) Delta'/Delta of the low-D points
    and of the sqrt(D - 3) formula (hp/lowD_logslope.py)"""
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import lowD_logslope as LS
    import plot_lowD_values as P
    rows = LS.table(3.3)
    peak = max(rows, key=lambda r: r[1])
    low = min(rows, key=lambda r: r[0])
    txt = [("README.md", "it rises to 0.163 near D = 3.1 and falls to 0.145 at D = 3.044"),
           ("README.md", "the √(D − 3) formula follows it to 8×10⁻⁴ and sends it to 0, as about 0.71 √(D − 3)"),
           ("README.md", "the\n  formula predicts 0.1441 at D = 3.0425 and 0.1418 at 3.04")]
    return [("local exponent peak", peak[1], 0.163, 0.001, txt), ("D of the peak", peak[0], 3.1, 0.1, txt),
            ("local exponent at the lowest point", low[1], 0.145, 0.001, txt),
            ("largest |measured - formula|", max(abs(L - Lf) for _, L, Lf in rows), 8e-4, 1e-4, txt),
            ("small-s slope c1 / (2 c0)", P.C_DELTA[1] / (2 * P.C_DELTA[0]), 0.71, 0.01, txt),
            ("formula at 3.0425", LS.formula_L(3.0425), 0.1441, 1e-4, txt), ("formula at 3.04", LS.formula_L(3.04), 0.1418, 1e-4, txt)]


@check
def lowD_formula_bound():
    """README, the low-D figure's caption: the sqrt(D - 3) formula (hp/plot_lowD_values.py C_DELTA) against every point
    of hp/data/delta_dd_lowD.csv at its largest M, D <= 4"""
    import csv
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import plot_lowD_values as P
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv")) if not l.startswith("#"))]
    best = {}
    for r in rows:
        D, M = float(r["D"]), int(r["M"])
        if D <= 4.0 and (D not in best or M > best[D][0]):
            best[D] = (M, float(r["Delta_hi"]) + float(r["Delta_lo"] or 0))
    err = sorted(abs(P.formula(P.C_DELTA, D) - v) for D, (M, v) in best.items())
    return [("worst point against the sqrt(D - 3) formula, D <= 4", err[-1], 1.6e-5, 1e-6, [("README.md", "lies within 1.6×10⁻⁵ of the √(D − 3) formula")]),
            ("fourth worst point (the rest)", err[-4], 8e-6, 1e-6, [("README.md", "the rest within 8×10⁻⁶")])]


@check
def gamma_formula_dd():
    """README "gamma": the rational gamma formula (the stored coefficients) against the double-double gamma points
    (hp/data/gamma_dd.jsonl) for D <= 30 and at D = 61"""
    sys.path.insert(0, os.path.join(ROOT, "solver"))
    import formula_all
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "hp/data/gamma_dd.jsonl"))]
    g = [(r["D"], r["gamma"]) for r in rows if "gamma" in r]
    e30 = max(abs(float(formula_all.gamma(D)) - v) for D, v in g if D <= 30)
    e61 = max(abs(float(formula_all.gamma(D)) - v) for D, v in g if 60 <= D <= 61.1)
    txt = [("README.md", "The rational γ formula is within 1.2×10⁻⁹"), ("README.md", "off by 1.0×10⁻⁷ at D = 61")]
    return [("gamma formula against the double-double gamma, D <= 30", e30, 1.2e-9, 1e-10, txt), ("the same at D = 61", e61, 1.0e-7, 1e-8, txt)]


@check
def loader_exact():
    """hp/dd_fits_mp.load_points must return hi + lo of every CSV row exactly. The CSV stores repr() of each double;
    reading those strings as decimals at 60 digits moves a point by up to 2e-16 (notes/NUMERICS_REVIEW.md F7,
    notes/LOADER_FIX.md). The control reads them that way here and must be caught by the same comparison."""
    from fractions import Fraction
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import dd_fits_mp as F
    mp = F.mp

    def exact(x):                                    # an mpf as a Fraction, exactly
        man, exp = x.man_exp
        return Fraction(man) * Fraction(2) ** exp

    pts, _ = F.load_points()
    rows = [c for c in (r.strip().split(",") for r in list(open(os.path.join(ROOT, "hp/data/delta_dd.csv")))[1:])
            if not c[6].startswith("lowD")]       # load_points reads the continuation rows only
    want = {round(float(c[0]), 6): Fraction(float(c[1])) + Fraction(float(c[2])) for c in rows}
    n_bad = sum(exact(pts[k]) != v for k, v in want.items()) + len(set(pts) ^ set(want))
    old = {round(float(c[0]), 6): mp.mpf(c[1]) + mp.mpf(c[2]) for c in rows}     # the parse before the fix
    n_old = sum(exact(old[k]) != v for k, v in want.items())
    return [("CSV rows whose loaded value differs from the exact hi + lo", n_bad, 0, 1, []),
            ("control: the decimal-string parse is caught (rows differing > 0)", int(n_old > 0), 1, 1, [])]


@check
def gamma_points():
    pts = {}
    for fn in ("hp/data/gamma_dd.jsonl", "hp/data/local/gamma_dd.jsonl"):
        for line in open(os.path.join(ROOT, fn)):
            r = json.loads(line)
            if r.get("lam"):
                pts[round(r["D"], 4)] = r["lam"]
    txt = [("README.md", "67 points, D = 8.25–61")]
    return [("gamma points", len(pts), 67, 1, txt),
            ("gamma lowest D", min(pts), 8.25, 0.01, txt),
            ("gamma highest D", max(pts), 61, 1, txt)]


DINF, BAR = 1.643893746, 1.5e-8
RANGE = (1.643893744, 1.643893763)
BIAS = 0.8e-8   # the correction applied to the centre of the 48-fit range (README, "Is the period analytic in 1/D?")


def planted_offsets():
    """Centre minus planted limit, from the repository's planted test (T1-T6) and the review's independent one (N1-N3)."""
    rep = [float(m.group(1)) for m in re.finditer(r"centre off by ([0-9.e+-]+)",
                                                   open(os.path.join(ROOT, "hp/data/planted_dinf.log")).read())]
    rev = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(N[0-9]) .*?centre off by ([0-9.e+-]+)",
                                                             open(os.path.join(ROOT, "hp/review/rv2_planted.log")).read(), re.M)}
    return rep, rev


@check
def dinf_quotes():
    """Delta_inf is the centre of the 48-fit range less the planted-truth offset; the bar contains the fit range but its
    top 2e-9 and every graded plant; every quote of Delta_inf in the README and the papers carries the same value and bar."""
    rep, rev = planted_offsets()
    # the 48-fit range to 11 digits, as the review's independent refit prints it (the quoted selection: factor 2.5, two
    # lowest degrees); the full mode below recomputes the 9-digit RANGE with the repository's own code
    m = re.search(r"factor\s+2\.5\s+lowest 2\s+48 fits\s+([0-9.]+) \.\. ([0-9.]+)",
                  open(os.path.join(ROOT, "hp/review/rv2_fits_rules.log")).read())
    lo, hi = float(m.group(1)), float(m.group(2))
    centre = (lo + hi) / 2
    readme = [("README.md", "1.643893744–1.643893763")]
    items = [("fit range, low end", lo, RANGE[0], 1e-9, readme), ("fit range, high end", hi, RANGE[1], 1e-9, readme),
             ("centre of the fit range", centre, 1.643893754, 1e-9, [("README.md", "centred on 1.643893754")]),
             ("planted offset, repository test, lowest (7.85e-9: half-way, so a 2e-10 unit)", min(rep), 0.79e-8, 0.02e-8, [("README.md", "0.79–0.88×10⁻⁸")]),
             ("planted offset, repository test, highest", max(rep), 0.88e-8, 0.01e-8, [("README.md", "0.79–0.88×10⁻⁸")]),
             ("planted offset, independent test (N1)", rev["N1"], 0.78e-8, 0.01e-8, [("README.md", "0.78×10⁻⁸")]),
             ("correction within the measured offsets", min(max(BIAS, min(rep + [rev["N1"]])), max(rep)), BIAS, 1e-12, []),
             ("Delta_inf = centre less the correction", centre - BIAS, DINF, 1e-9, readme),
             ("bar reaches the bottom of the fit range", max(DINF - BAR, RANGE[0]), RANGE[0], 1e-12, []),
             ("fit range above the bar (quoted 2e-9)", RANGE[1] - (DINF + BAR), 2e-9, 1e-9, []),
             ("bar contains every graded plant", max(BAR, max(abs(o - BIAS) for o in rep + [rev[k] for k in ("N1", "N2", "N3")])),
              BAR, 1e-12, [])]
    pat = re.compile(r"1\.643893746\s*(?:±|\\pm)\s*([0-9.]+)(?:\s*(?:×|\\times)\s*10\^?\{?([⁻\-−]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)\}?)?")
    old = re.compile(r"1\.64389375\s*(?:±|\\pm)")
    sup = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹−", "-0123456789-")
    for doc in ("README.md", "paper/physics/main.tex", "paper/math/main.tex"):
        for m in pat.finditer(DOCS[doc]):
            v = float(m.group(1)) * (10 ** int(m.group(2).translate(sup)) if m.group(2) else 1)
            items.append((f"bar quoted in {doc}: '{m.group(0)}'", BAR, v, 1e-10, []))
        items.append((f"old quote 1.64389375 ± left in {doc}", len(old.findall(DOCS[doc])), 0, 1, []))
    return items


@check
def omega_quotes():
    w = 2 * math.pi / DINF
    dw = 2 * math.pi * BAR / DINF ** 2
    items = [("omega_inf = 2 pi / Delta_inf", w, 3.82213590, 1e-8, [("README.md", "3.82213590")])]
    pat = re.compile(r"3\.82213590\s*(?:±|\\pm)\s*([0-9.]+)(?:\s*(?:×|\\times)\s*10\^?\{?([⁻\-−]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)\}?)?")
    sup = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹−", "-0123456789-")
    for doc in ("README.md", "paper/physics/main.tex", "paper/math/main.tex"):
        for m in pat.finditer(DOCS[doc]):
            v = float(m.group(1)) * (10 ** int(m.group(2).translate(sup)) if m.group(2) else 1)
            # an error bar may be rounded up, never down: accept dw <= quoted <= dw rounded up to one figure
            up = math.ceil(dw / 10 ** math.floor(math.log10(dw))) * 10 ** math.floor(math.log10(dw))
            ok = dw <= v <= up * (1 + 1e-9)
            items.append((f"omega bar in {doc}: '{m.group(0)}' (2 pi bar / Delta^2 = {dw:.2e}; allowed {dw:.2e}..{up:.0e})",
                          v if ok else float("nan"), v, 1e-9, []))
    return items


@check
def selection_range():
    """The selection rule's estimator range from the D = infinity inner problem (notes/PLANTED_SELECTION.md section 14.9,
    the headline of solver/inner_amp/planted_checks_k22.json, the union of the K = 21 headline and the K = 22 range),
    the K = 22, 21, 20, 19, 18, 17 and 16 ranges alone, the unions of each pair of neighbouring orders, and the Delta_inf
    range the headline gives with the finite-D value 1.643893746 (dinf_quotes): Delta = 1.643893746 / (omega* / omega_inf)."""
    c22 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k22.json")))["K22 combine"]
    lo, hi = c22["headline"]
    lo22, hi22 = c22["range22"]
    c21 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k21.json")))["K21 combine"]
    lo21, hi21 = c21["range21"]
    c20 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k20.json")))["K20 combine"]
    lo20, hi20 = c20["range20"]
    c19 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k19.json")))["K19 combine"]
    lo19, hi19 = c19["range19"]
    c = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k18.json")))["K18 combine"]
    lo18, hi18 = c["range18"]
    lo17, hi17 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k17.json")))["K17 combine"]["range17"]
    lo16, hi16 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k16.json")))["K16 combine"]["range16"]
    dlo, dhi = DINF / hi, DINF / lo
    txt = [("README.md", "ω*/ω∞ = 0.999975–1.00005, Δ∞ = 1.64387 ± 0.00006 (1.64381–1.64393)"),
           ("README.md", "puts the selected ω at 0.999975–1.00005 ω∞ (Δ∞ = 1.64387 ± 0.00006"),
           ("paper/math/main.tex", "\\Dinf = 1.64387 \\pm 0.00006 \\qquad (1.64381\\text{--}1.64393)"),
           ("paper/math/main.tex", "$\\omega^*/\\winf = 0.999975$--$1.00005$"),
           ("paper/physics/main.tex", "$\\omega/\\omega_\\infty = 0.999975$--$1.00005$"),
           ("paper/physics/main.tex", "\\Dinf = 1.64387 \\pm 0.00006")]
    t22 = [("README.md", "the twenty-second order alone gives 0.9999988–1.0000004"),
           ("README.md", "the twenty-two-order estimator range is 0.9999988–1.0000004"),
           ("notes/PLANTED_SELECTION.md", "  0.9999988–1.0000004.**"),
           ("notes/PROOF_QUEUE.md", "K = 22 range 0.9999988–1.0000004"),
           ("paper/math/main.tex", "the twenty-two-order estimator range is $0.9999988$--$1.0000004$")]
    t21 = [("README.md", "family, the ring-free sums) is 0.999998–1.000002 ([notes/PLANTED_SELECTION.md]"),
           ("notes/PLANTED_SELECTION.md", "**The hull is 0.999998–1.000002.**"),
           ("notes/PROOF_QUEUE.md", "K = 21 range 0.999998–1.000002"),
           ("paper/math/main.tex", "the twenty-one-order estimator range is $0.999998$--$1.000002$")]
    t20 = [("notes/PLANTED_SELECTION.md", "**The hull is 0.999997–1.000002.**"),
           ("notes/PROOF_QUEUE.md", "K = 20 range 0.999997–1.000002"),
           ("paper/math/main.tex", "the twenty-order estimator range is $0.999997$--$1.000002$")]
    t19 = [("notes/PLANTED_SELECTION.md", "**The hull is 0.999996–1.000004.**"),
           ("notes/PROOF_QUEUE.md", "K = 19 range 0.999996–1.000004"),
           ("paper/math/main.tex", "the nineteen-order estimator range is $0.999996$--$1.000004$")]
    t18 = [("notes/PLANTED_SELECTION.md", "**The hull is 0.999992–1.000004.**"),
           ("paper/math/main.tex", "the eighteen-order estimator range is $0.999992$--$1.000004$")]
    t17 = [("notes/PLANTED_SELECTION.md", "**The hull is 0.999990–1.000009.**"),
           ("paper/math/main.tex", "the seventeen-order estimator range is $0.999990$--$1.000009$")]
    t16 = [("notes/PLANTED_SELECTION.md", "**The hull is 0.999980–1.000014.**"),
           ("paper/math/main.tex", "the sixteen-order estimator range is $0.999980$--$1.000014$")]
    t2_22 = [("README.md", "(the last two orders alone give 0.9999984–1.0000016)"),
             ("notes/PLANTED_SELECTION.md", "the last two orders alone, reported as context by item 7, is 0.9999984–1.0000016")]
    t2_21 = [("notes/PLANTED_SELECTION.md", "the last two orders alone, reported as context by item 7, is 0.999997–1.000002")]
    t2 = [("notes/PLANTED_SELECTION.md", "the last two orders alone, reported as context by item 7, is 0.999996–1.000004")]
    t2_19 = [("notes/PLANTED_SELECTION.md", "the range from the last two orders alone, reported as context by item 7, is 0.999992–1.000004")]
    t2_18 = [("notes/PLANTED_SELECTION.md", "the range from the last two orders alone, reported as context by item 7, is 0.999990–1.000009")]
    t2_17 = [("notes/PLANTED_SELECTION.md", "the range from the last two orders alone, reported as context by item 7, is 0.999980–1.000014")]
    return [("selection range, lower end", lo, 0.999975, 1e-6, txt),
            ("selection range, upper end", hi, 1.00005, 1e-5, txt),
            ("K = 22 range, lower end", lo22, 0.9999988, 1e-7, t22),
            ("K = 22 range, upper end", hi22, 1.0000004, 1e-7, t22),
            ("K = 21 and 22 union, lower end", min(lo21, lo22), 0.9999984, 1e-7, t2_22),
            ("K = 21 and 22 union, upper end", max(hi21, hi22), 1.0000016, 1e-7, t2_22),
            ("K = 21 range, lower end", lo21, 0.999998, 1e-6, t21),
            ("K = 21 range, upper end", hi21, 1.000002, 1e-6, t21),
            ("K = 20 range, lower end", lo20, 0.999997, 1e-6, t20),
            ("K = 20 range, upper end", hi20, 1.000002, 1e-6, t20),
            ("K = 19 range, lower end", lo19, 0.999996, 1e-6, t19),
            ("K = 19 range, upper end", hi19, 1.000004, 1e-6, t19),
            ("K = 18 range, lower end", lo18, 0.999992, 1e-6, t18),
            ("K = 18 range, upper end", hi18, 1.000004, 1e-6, t18),
            ("K = 17 range, lower end", lo17, 0.999990, 1e-6, t17),
            ("K = 17 range, upper end", hi17, 1.000009, 1e-6, t17),
            ("K = 16 range, lower end", lo16, 0.999980, 1e-6, t16),
            ("K = 16 range, upper end", hi16, 1.000014, 1e-6, t16),
            ("K = 20 and 21 union, lower end", min(lo20, lo21), 0.999997, 1e-6, t2_21),
            ("K = 20 and 21 union, upper end", max(hi20, hi21), 1.000002, 1e-6, t2_21),
            ("K = 19 and 20 union, lower end", min(lo19, lo20), 0.999996, 1e-6, t2),
            ("K = 19 and 20 union, upper end", max(hi19, hi20), 1.000004, 1e-6, t2),
            ("K = 18 and 19 union, lower end", min(lo18, lo19), 0.999992, 1e-6, t2_19),
            ("K = 18 and 19 union, upper end", max(hi18, hi19), 1.000004, 1e-6, t2_19),
            ("K = 17 and 18 union, lower end", min(lo17, lo18), 0.999990, 1e-6, t2_18),
            ("K = 17 and 18 union, upper end", max(hi17, hi18), 1.000009, 1e-6, t2_18),
            ("K = 16 and 17 union, lower end", min(lo16, lo17), 0.999980, 1e-6, t2_17),
            ("K = 16 and 17 union, upper end", max(hi16, hi17), 1.000014, 1e-6, t2_17),
            ("Delta_inf range, lower end", dlo, 1.64381, 1e-5, txt),
            ("Delta_inf range, upper end", dhi, 1.64393, 1e-5, txt),
            ("Delta_inf range, centre", (dlo + dhi) / 2, 1.64387, 1e-5, txt),
            ("Delta_inf range, half-width", (dhi - dlo) / 2, 0.00006, 1e-5, txt)]


@check
def pilot_agreement():
    """The two sigma_19 pilots at wf 1.00, (a) J 1000 / 960 bits and (b) J 1250 / 1280 bits (notes/PLANTED_SELECTION.md
    section 11.0, solver/inner_amp/pilot_agreement.py): |sigma_19(a) - sigma_19(b)| over (a)'s err19, the pilot gate
    that fired, and over |sigma_19|."""
    d = os.path.join(ROOT, "solver/inner_amp")
    a = json.load(open(os.path.join(d, "scan39J1000/sigF_w1.0000_K39_J1000_p960.json")))
    b = json.load(open(os.path.join(d, "scan39J1250/sigF_w1.0000_K39_J1250_p1280.json")))
    sa, sb = (complex(float(r["sigma_str"][19][0]), float(r["sigma_str"][19][1])) for r in (a, b))
    diff = abs(sa - sb)
    t = [("notes/PLANTED_SELECTION.md", "a ratio of 1.21"), ("proof/P5_scope.md", "1.21 times that fit error"),
         ("notes/PROOF_QUEUE.md", "agreement gate by 1.21 at σ₁₉")]
    t2 = [("notes/PLANTED_SELECTION.md", "The difference is 1.3×10⁻¹¹ of |σ₁₉|"),
          ("proof/P5_scope.md", "differs from it by 1.3×10⁻¹¹ of |σ₁₉|")]
    out = [("sigma_19 pilot gate ratio", diff / a["sigma_err"][19], 1.21, 0.01, t),
           ("sigma_19 pilot difference, relative", diff / abs(sa), 1.3e-11, 1e-12, t2)]
    # T91's attribution pilot (d), J 1250 / 960 bits, and the amended err19 (section 11.0's amendment of 2026-10-29)
    sys.path.insert(0, d)
    import pilot_agreement
    r = pilot_agreement.attribution(39)
    sys.path.pop(0)
    P = "notes/PLANTED_SELECTION.md"
    out += [("sigma_19 J effect |a - d|", r["ad19"], 1.85e-27, 0.01e-27,
             [(P, "puts σ₁₉ at 1.85×10⁻²⁷ from (a)"), (P, "4.5×10⁻²⁸, 1.85×10⁻²⁷ (the J effect)"),
              ("notes/PROOF_QUEUE.md", "a J effect of 1.85×10⁻²⁷")]),
            ("sigma_19 precision effect |d - b|", r["db19"], 1.94e-27, 0.01e-27,
             [(P, "1.94×10⁻²⁷ from (b), with its own plateau error"), (P, "differ at σ₁₉ by 1.94×10⁻²⁷ with both"),
              (P, "4×10⁻³³, 1.94×10⁻²⁷ (the precision effect)"), ("notes/PROOF_QUEUE.md", "a precision effect of 1.94×10⁻²⁷")]),
            ("sigma_19 (d) plateau error", r["err_d19"], 1.85e-27, 0.01e-27,
             [(P, "with its own plateau error 1.85×10⁻²⁷"), (P, "err19 1.85×10⁻²⁷ (7.35×10⁻¹²")]),
            ("sigma_19 (d) plateau error, relative", r["err_d19"] / r["mag_a19"], 7.35e-12, 0.01e-12, [(P, "(7.35×10⁻¹²")]),
            ("J effect over (a)'s plateau error", r["ad19"] / r["err_a19"], 2 / 3, 0.1,
             [(P, "each move σ₁₉ by about two thirds of (a)'s plateau error")]),
            ("precision effect over (a)'s plateau error", r["db19"] / r["err_a19"], 2 / 3, 0.1,
             [(P, "each move σ₁₉ by about two thirds of (a)'s plateau error")]),
            ("sigma_16 (d) - (b) over (b)'s plateau step", r["db16"] / r["err_b16"], 1.0, 0.001,
             [(P, "differ at σ₁₆ by 2.94×10⁻²⁶, exactly (b)'s plateau step")]),
            ("sigma_16 (d) - (b)", r["db16"], 2.94e-26, 0.01e-26, [(P, "|d − b|: 2.94×10⁻²⁶, 2×10⁻³³")]),
            ("sigma_17 (d) - (b)", r["db17"], 2e-33, 1e-33, [(P, "to 2×10⁻³³ and 4×10⁻³³ at σ₁₇")]),
            ("sigma_18 (d) - (b)", r["db18"], 4e-33, 1e-33, [(P, "to 2×10⁻³³ and 4×10⁻³³ at σ₁₇")]),
            ("sigma_16..18 (a) - (d)", r["ad16"], 5.4e-27, 0.1e-27, [(P, "|a − d| at σ₁₆…σ₁₉: 5.4×10⁻²⁷, 1.6×10⁻²⁷, 4.5×10⁻²⁸")]),
            ("sigma_17 (a) - (d)", r["ad17"], 1.6e-27, 0.1e-27, [(P, "5.4×10⁻²⁷, 1.6×10⁻²⁷, 4.5×10⁻²⁸")]),
            ("sigma_18 (a) - (d)", r["ad18"], 4.5e-28, 0.1e-28, [(P, "1.6×10⁻²⁷, 4.5×10⁻²⁸, 1.85×10⁻²⁷")]),
            ("neighbour families picking a different Mf at order 33", r["ndiff33"], 16, 0.5,
             [(P, "at order 33 all 16 neighbour families pick Mf 8")]),
            ("neighbour families at order 33", r["nfam33"], 16, 0.5, [(P, "all 16 neighbour families pick Mf 8")]),
            ("tuning constants (d) vs (b) at order 33", r["fdb33"], 1.9e-5, 0.1e-5, [(P, "differ by 1.9×10⁻⁵ of themselves at order 33")]),
            ("tuning constants (d) vs (b) at order 37", r["fdb37"], 3.1e-2, 0.1e-2, [(P, "and 3.1×10⁻² at order 37")]),
            ("tuning constants (a) vs (d) at order 33", r["fad33"], 0.77, 0.01, [(P, "differ by 77% of themselves at order 33")]),
            ("(d) and (b) equal at sigma_12..15", float(r["exact_12_15"]), 1, 0.5, [(P, "agree exactly at σ₁₂…σ₁₅")]),
            ("amended err19 at wf 1.00", r["err_amended"], 6.17e-27, 0.01e-27,
             [(P, "err19 = 6.17×10⁻²⁷, 2.45×10⁻¹¹ of |σ₁₉|"), (P, "6.17×10⁻²⁷; the amended item 1(a)")]),
            ("amended err19 at wf 1.00, relative", r["err_amended_rel"], 2.45e-11, 0.01e-11, [(P, "2.45×10⁻¹¹ of |σ₁₉|, passing 1(a) by 4.1")]),
            ("amended err19 margin under 1e-10", r["margin_amended"], 4.1, 0.1, [(P, "passing 1(a) by 4.1")]),
            ("T91 minutes", r["seconds_d"] / 60, 293, 1, [(P, "J 1250 at 960 bits, 293 min")])]
    return out


@check
def k20_gate():
    """T89's two sigma_20 runs at wf 1.00, (a) K 41, J 1250, 1280 bits and (b) J 1500, 1280 bits (notes/PLANTED_SELECTION.md
    section 12.0, the files in solver/inner_amp/scan41J1250 and scan41J1500 from the raw-data repository's T89-mac):
    err20 from sigma_err_fit.py (a1 from the 960-bit scan39J1000 file), the pilot difference, the sigma_16 difference
    against (a)'s plateau error, the strings equal to T88's pilot (b), the hours."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import sigma_err_fit as SF
    sys.path.pop(0)
    a = json.load(open(os.path.join(d, "scan41J1250/sigF_w1.0000_K41_J1250_p1280.json")))
    b = json.load(open(os.path.join(d, "scan41J1500/sigF_w1.0000_K41_J1500_p1280.json")))
    p = json.load(open(os.path.join(d, "scan39J1000/sigF_w1.0000_K39_J1000_p960.json")))
    t88 = json.load(open(os.path.join(d, "scan39J1250/sigF_w1.0000_K39_J1250_p1280.json")))
    a1 = SF.a1_of(p)
    ra, rb = SF.recompute(a, a1), SF.recompute(b, a1)
    sa, ea = ra[20][1], ra[20][2]
    sb, eb = rb[20][1], rb[20][2]
    diff = float(abs(sa - sb))
    d16 = float(abs(ra[16][1] - rb[16][1]))
    equal = sum(a["sigma_str"][k] == t88["sigma_str"][k] for k in range(20))
    P = "notes/PLANTED_SELECTION.md"
    return [("err20 of (a)", ea, 1.335e-29, 0.001e-29, [(P, "err20(a) = 1.335×10⁻²⁹")]),
            ("err20 of (a), relative", ea / float(abs(sa)), 7.18e-13, 0.01e-13, [(P, "7.18×10⁻¹³ of |σ₂₀|")]),
            ("err20 margin under 1e-10", 1e-10 / (ea / float(abs(sa))), 139, 1, [(P, "a margin of 139")]),
            ("sigma_20 pilot difference", diff, 1.524e-31, 0.001e-31, [(P, "|σ₂₀(a) − σ₂₀(b)| = 1.524×10⁻³¹")]),
            ("sigma_20 pilot difference over err20(a)", diff / ea, 0.011, 0.001, [(P, "0.011 of (a)'s err20")]),
            ("err20 of (b)", eb, 6.431e-32, 0.001e-32, [(P, "(b)'s err20 = 6.431×10⁻³²")]),
            ("sigma_16 difference of (a) and (b)", d16, 2.98e-26, 0.01e-26, [(P, "differ by 2.98×10⁻²⁶")]),
            ("sigma_16 difference over (a)'s plateau error", d16 / ra[16][2], 1.01, 0.01,
             [(P, "1.01 times (a)'s plateau error")]),
            ("sigma_0..19 strings equal to T88's pilot (b)", equal, 20, 0.5, [(P, "20 of 20 strings")]),
            ("T89 (a) hours", a["seconds"] / 3600, 12.4, 0.1, [(P, "(a) 12.4 h")]),
            ("T89 (b) hours", b["seconds"] / 3600, 13.0, 0.1, [(P, "(b) 13.0 h")])]


@check
def k19_amend2():
    """notes/PLANTED_SELECTION.md section 11.0, the second amendment of item 1 (2026-10-31), and section 12.0's amendment: the
    (a') converge run's err19 at the nodes (solver/inner_amp/planted_checks_k19.json, "K19 converge under (a')") and the
    sigma_8, sigma_11 differences of the 1280-bit node files (scan41J1250) from the 960-bit scan39J1000 values over that
    file's plateau errors (sigma_err_fit.py recomputes the 1280-bit values' errors; the ratios use the 960-bit errors)."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import sigma_err_fit as SF
    sys.path.pop(0)
    import mpmath as mp
    P = "notes/PLANTED_SELECTION.md"
    c = json.load(open(os.path.join(d, "planted_checks_k19.json")))["K19 converge under (a')"]["nodes"]
    n99 = c["0.99"]
    out = [("(a') plateau error at wf 0.99", n99["err19_plateau"], 2.07e-26, 0.01e-26, [(P, "err19 = 2.07×10⁻²⁶ (the 960-bit")]),
           ("(a') pilot difference at wf 0.99", n99["pilot_d19"], 2.02e-26, 0.01e-26, [(P, "+ 2.02×10⁻²⁶ (the pilot difference)")]),
           ("(a') err19 at wf 0.99", n99["err19"], 4.09e-26, 0.01e-26, [(P, "= 4.09×10⁻²⁶ is 1.59×10⁻¹⁰ of |σ₁₉|")]),
           ("(a') err19 at wf 0.99, relative", n99["err19_rel"], 1.59e-10, 0.01e-10, [(P, "is 1.59×10⁻¹⁰ of |σ₁₉|")]),
           ("(a') margin at wf 0.99", 1e-10 / n99["err19_rel"], 0.63, 0.01, [(P, "a margin of 0.63")]),
           ("(a') relative err19 at wf 1.00", c["1.0"]["err19_rel"], 2.45e-11, 0.01e-11, [(P, "wf 1.00 (2.45×10⁻¹¹) and 1.01 (5.75×10⁻¹²)")]),
           ("(a') relative err19 at wf 1.01", c["1.01"]["err19_rel"], 5.75e-12, 0.01e-12, [(P, "wf 1.00 (2.45×10⁻¹¹) and 1.01 (5.75×10⁻¹²)")])]
    mp.mp.dps = 60
    rat = {}
    for w in ("0.9900", "1.0000", "1.0100"):
        p960 = json.load(open(os.path.join(d, f"scan39J1000/sigF_w{w}_K39_J1000_p960.json")))
        n = json.load(open(os.path.join(d, f"scan41J1250/sigF_w{w}_K41_J1250_p1280.json")))
        r = SF.recompute(n, SF.a1_of(p960))
        for k in range(1, 19):
            s0 = mp.mpc(mp.mpf(p960["sigma_str"][k][0]), mp.mpf(p960["sigma_str"][k][1]))
            rat[(w, k)] = float(abs(r[k][1] - s0)) / p960["sigma_err"][k]
    s11 = "the ratio is 1.00, 0.99 and 1.17 at σ₁₁ (wf 0.99, 1.00, 1.01) and 0.01, 0.01 and 1.01 at σ₈"
    out += [("sigma_11 ratio at wf 0.99", rat[("0.9900", 11)], 1.00, 0.01, [(P, s11)]),
            ("sigma_11 ratio at wf 1.00", rat[("1.0000", 11)], 0.99, 0.01, [(P, s11)]),
            ("sigma_11 ratio at wf 1.01", rat[("1.0100", 11)], 1.17, 0.01, [(P, s11), (P, "σ₁₁ differs from `scan39J1000`'s by 1.17 times")]),
            ("sigma_8 ratio at wf 0.99", rat[("0.9900", 8)], 0.01, 0.01, [(P, s11)]),
            ("sigma_8 ratio at wf 1.00", rat[("1.0000", 8)], 0.01, 0.01, [(P, s11)]),
            ("sigma_8 ratio at wf 1.01", rat[("1.0100", 8)], 1.01, 0.01, [(P, s11), (P, "and σ₈ by 1.01 times")]),
            ("largest ratio at another order", max(v for (w, k), v in rat.items() if k not in (8, 11)), 0.74, 0.01,
             [(P, "one at most 0.74")])]
    return out


@check
def k21_gate():
    """T94's pilot (a) at wf 1.00, K 43, J 1250, 1280 bits (notes/PLANTED_SELECTION.md section 13.0's gate reading; the
    file in solver/inner_amp/scan43J1250 from the raw-data repository's T94-mac at 630fbda): err21 from sigma_err_fit.py
    (a1 from the 960-bit scan39J1000 file), its margin under 1e-10, the strings equal to the K 41 file, |sigma_21| against
    P5's K = 20 bound (the wf 1.0 node line of solver/proof/logs/entire_cert_K21.log, the unrounded bound; proof/P5_scope.md)."""
    import re
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import sigma_err_fit as SF
    sys.path.pop(0)
    a = json.load(open(os.path.join(d, "scan43J1250/sigF_w1.0000_K43_J1250_p1280.json")))
    p = json.load(open(os.path.join(d, "scan39J1000/sigF_w1.0000_K39_J1000_p960.json")))
    L = open(os.path.join(ROOT, "solver/proof/logs/entire_cert_K21.log")).read()
    r20 = float(re.search(r"wf 1\.0: k <= 20 \(.*?\): ([0-9.]+);", L).group(1))
    k41 = json.load(open(os.path.join(d, "scan41J1250/sigF_w1.0000_K41_J1250_p1280.json")))
    ra = SF.recompute(a, SF.a1_of(p))
    sa, ea = float(abs(ra[21][1])), ra[21][2]
    equal = sum(a["sigma_str"][k] == k41["sigma_str"][k] for k in range(21))
    P, Q = "notes/PLANTED_SELECTION.md", "proof/P5_scope.md"
    return [("err21 of (a)", ea, 1.351e-30, 0.001e-30, [(P, "err21(a) = 1.351×10⁻³⁰")]),
            ("err21 of (a), relative", ea / sa, 1.02e-12, 0.01e-12, [(P, "1.02×10⁻¹² of |σ₂₁|"), (Q, "1.02×10⁻¹² of |σ₂₁|")]),
            ("err21 margin under 1e-10", 1e-10 / (ea / sa), 98, 1, [(P, "a margin of 98")]),
            ("sigma_0..20 strings of (a) equal to the K 41 file", equal, 21, 0.5, [(P, "21 of 21 strings; its run log")]),
            ("|sigma_21| at wf 1.00", sa, 1.32e-18, 0.01e-18, [(P, "|σ₂₁(a)| = 1.32×10⁻¹⁸"), (Q, "|σ₂₁| = 1.32×10⁻¹⁸")]),
            ("sigma_21 over the K = 20 bound at wf 1.00", r20, 0.899, 0.001,
             [(P, "0.899 of P5's"), (Q, "0.899 of the K = 20 bound")])]

@check
def k21_hedge_nodes():
    """win2's J 1500 node files at wf 0.99 and 1.01 (K 43, 1280 bits; solver/inner_amp/scan43J1500 from the raw-data
    repository's T94-win2 at 2e49f99), the context of notes/PLANTED_SELECTION.md section 13.2: err21 from sigma_err_fit.py
    (a1 from the 960-bit scan39J1000 files), sigma_21 against the J 1250 node files, the strings equal to the K 41 files,
    the run times."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import sigma_err_fit as SF
    sys.path.pop(0)
    P = "notes/PLANTED_SELECTION.md"
    out = []
    for w, e, ue, dlt, h, te in (("0.9900", 2.10e-13, 0.01e-13, 7.1e-14, 17.7, "err21 = 2.10×10⁻¹³ and"),
                                 ("1.0100", 2.30e-15, 0.01e-15, 1.5e-14, 17.1, "2.30×10⁻¹⁵ of |σ₂₁|")):
        f15 = json.load(open(os.path.join(d, f"scan43J1500/sigF_w{w}_K43_J1500_p1280.json")))
        f12 = json.load(open(os.path.join(d, f"scan43J1250/sigF_w{w}_K43_J1250_p1280.json")))
        k41 = json.load(open(os.path.join(d, f"scan41J1250/sigF_w{w}_K41_J1250_p1280.json")))
        a1 = SF.a1_of(json.load(open(os.path.join(d, f"scan39J1000/sigF_w{w}_K39_J1000_p960.json"))))
        r15 = {t[0]: t for t in SF.recompute(f15, a1)}
        r12 = {t[0]: t for t in SF.recompute(f12, a1)}
        s15, s12 = r15[21][1], r12[21][1]
        equal = sum(f15["sigma_str"][k] == k41["sigma_str"][k] for k in range(21))
        out += [(f"J 1500 node wf {w}: err21 relative", float(r15[21][2] / abs(s15)), e, ue, [(P, te)]),
                (f"J 1500 node wf {w}: sigma_21 against the J 1250 node", float(abs(s15 - s12) / abs(s12)), dlt, 0.1e-14,
                 [(P, "by 7.1×10⁻¹⁴ and 1.5×10⁻¹⁴ of |σ₂₁|")]),
                (f"J 1500 node wf {w}: strings equal to the K 41 file", equal, 8, 0.5,
                 [(P, "8 of 21 strings equal the K 41 J 1250 files' at each node")]),
                (f"J 1500 node wf {w}: hours", f15["seconds"] / 3600, h, 0.1, [(P, "after 17.7 and 17.1 h")])]
    return out


@check
def large_d_walls():
    """notes/QUAD_DOUBLE.md 12.7: the fitted exponents, the crossings of the horizon series' off-solution radius with the
    layouts' matching distances, the observed walls and the linear zero (hp/data/large_d_walls.json, written by
    hp/wall_model.py from the raw-data repository's T84-mac, T85-mac and T101-mac records)."""
    J = json.load(open(os.path.join(ROOT, "hp/data/large_d_walls.json")))
    f, c, w = J["fits"], J["cross"], J["walls"]
    Q = "notes/QUAD_DOUBLE.md"
    return [("R exponent, 4e-9 off", f["1e-04"]["p"], -0.80, 0.01, [(Q, "| 4×10⁻⁹ | -0.80 |")]),
            ("R exponent, on the solution", f["0e+00"]["p"], -0.34, 0.01, [(Q, "| on the solution | -0.34 |")]),
            ("crossing u0 0.20, 4e-9 off", c["0.8"]["1e-04"], 83, 1, [(Q, "crossings for 4×10⁻⁹ and 4×10⁻⁷ off (83 and 57)")]),
            ("crossing u0 0.20, 4e-7 off", c["0.8"]["1e-02"], 57, 1, [(Q, "crossings for 4×10⁻⁹ and 4×10⁻⁷ off (83 and 57)")]),
            ("crossing u0 0.30, 4e-13 off", c["0.7"]["1e-08"], 80, 1, [(Q, "crossing 80) could bind")]),
            ("crossing u0 0.15, 4e-9 off", c["0.85"]["1e-04"], 118, 1, [(Q, "| 4×10⁻⁹ | -0.80 | 0.172 | 0.098 | 50 | 83 | 118 |")]),
            ("wall, 0.8 layout", w["0.8 pinned"], 78.1, 0.1, [(Q, "the 0.8 layout stopped at 78.1 with")]),
            ("wall, 0.7 plain", w["0.7 plain"], 67.8, 0.1, [(Q, "stopped at 67.8 (plain, its floor)")]),
            ("wall, 0.7 pinned", w["0.7 pinned"], 73.3, 0.1, [(Q, "and 73.3 (pinned, the floor and the basin")]),
            ("walls, 0.85 layout", w["0.85 pinned"], 76.5, 0.1, [(Q, "left the branch at 76.5 and 78.6 along")]),
            ("walls, 0.85 ladder", w["0.85 ladder"], 78.6, 0.1, [(Q, "left the branch at 76.5 and 78.6 along")]),
            ("linear zero, 4e-9 off", f["1e-04"]["D_zero_linear"], 128, 1, [(Q, "puts R at zero near D 128 for 4×10⁻⁹ off")])]


@check
def pole_c_orders():
    """notes/QUAD_DOUBLE.md 13.6's addendum: the pole constant C(R136) at 100 and 250 series orders over D 40-60, and
    its rise from 200 to 250 orders (hp/data/pole_c_orders.csv, written by hp/pole_c_orders.py from the raw-data
    repository's I538-mac and I546-win summaries; #546)."""
    import csv
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(ROOT, "hp/data/pole_c_orders.csv"))
                                      if not l.startswith("#"))]
    C = {(float(r["D"]), int(r["orders"])): float(r["C_R136"]) for r in rows}
    Ds = sorted({d for d, _ in C})
    c100 = [C[(d, 100)] for d in Ds]
    c250 = [C[(d, 250)] for d in Ds]
    rise = [C[(d, 250)] - C[(d, 200)] for d in Ds]
    Q, W = "notes/QUAD_DOUBLE.md", "notes/LARGE_D_WALL.md"
    t100 = [(Q, "C(R136) is 0.869–0.918\nat 100 orders")]
    t250 = [(Q, "0.947–0.958 at 250"), (W, "to 0.947–0.958 at 250")]
    trise = [(Q, "to 250 orders it is still 0.004–0.009")]
    return [("C(R136) at 100 orders, least (D 40-60)", min(c100), 0.869, 0.001, t100),
            ("C(R136) at 100 orders, most", max(c100), 0.918, 0.001, t100),
            ("C(R136) at 250 orders, least", min(c250), 0.947, 0.001, t250),
            ("C(R136) at 250 orders, most", max(c250), 0.958, 0.001, t250),
            ("C(R136) rise from 200 to 250 orders, least", min(rise), 0.004, 0.001, trise),
            ("C(R136) rise from 200 to 250 orders, most", max(rise), 0.009, 0.001, trise)]


@check
def dropoffs():
    """notes/LARGE_D_WALL.md (from the README), "Where the drop-offs are": u*kappa's slope and crossings (hp/data/dropoffs.json, written by hp/plot_dropoffs.py
    from the raw-data repository's T76 records): the full system's fit n^38.9 crossing 1 at D 59.9, u*kappa at 67.74, the
    pinned systems' predicted crossings and quad-double's."""
    J = json.load(open(os.path.join(ROOT, "hp/data/dropoffs.json")))
    R = "notes/LARGE_D_WALL.md"      # the drop-offs text moved there from the README on 2026-11-04
    return [("u kappa slope", J["slope"], 38.9, 0.1, [(R, "grows\n  like n^38.9 and reaches 1 at D = 59.9")]),
            ("u kappa = 1, double-double, full system", J["cross_dd"], 59.9, 0.1,
             [(R, "u·κ = 1 at 59.9, 1.7 at 60.7"), (R, "like n^38.9 and reaches 1 at D = 59.9")]),
            ("u kappa at 67.74", J["u_kappa_67_74"], 168, 1, [(R, "(u·κ = 168 at 67.74)"), (R, "it is 1.7 at the stall at 60.96 and 168 at 67.74")]),
            ("u kappa = 1, pin 9", J["pinned"]["9"]["D_cross"], 79, 1, [(R, "u·κ reaches 1 at D = 79 with harmonics 11–13 held")]),
            ("u kappa = 1, pin 7", J["pinned"]["7"]["D_cross"], 94, 1, [(R, "94 with 9–13 held")]),
            ("u kappa = 1, pin 5", J["pinned"]["5"]["D_cross"], 113, 1, [(R, "113 with 7–13 held")]),
            ("u kappa = 1, quad-double, full system", J["cross_qd"], 372, 1, [(R, "372 for the full system in quad-double")]),
            ("u kappa at the 60.96 stall", J["u_kappa_60_97"], 1.7, 0.1, [(R, "was 1.7 at the stall")]),
            ("u kappa = 1.7 by the fit", 2 + (J["cross_dd"] - 2) * J["u_kappa_60_97"] ** (1 / J["slope"]), 60.7, 0.1,
             [(R, "1.7 at 60.7")]),
            ("u kappa at 35.02, every harmonic", [v for v in J["all_harmonics"] if abs(v["D"] - 35.0248) < 1e-3][0]["uk"], 0.010, 0.001, [(R, "u·κ was only 0.010 at\n35.02")]),
            ("basin kappa at 35.02", [v for v in J["all_harmonics"] if abs(v["D"] - 35.0248) < 1e-3][0]["kappa"], 26, 1, [(R, "at 26 against its limit of 2")])]

@check
def k21_stages():
    """notes/PLANTED_SELECTION.md sections 13.2-13.10: the K = 21 stages ("K21 converge", "K21 plants check", "K21 real",
    "K21 ring test", "K21 calib", "K21 recomplete", "K21 project", "K21 combine" in solver/inner_amp/planted_checks_k21.json
    and the record "N0 scan43J1250" in planted_results_k21.jsonl), with the README's twenty-one-order bullet and the
    mathematics paper's sentence on sigma_21."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import planted_k14 as K14
    sys.path.pop(0)
    import mpmath as mp
    P, R, M = "notes/PLANTED_SELECTION.md", "README.md", "paper/math/main.tex"
    c = json.load(open(os.path.join(d, "planted_checks_k21.json")))
    cv, pc, rl, rt, cal, rc, pj, cb = (c[k] for k in ("K21 converge", "K21 plants check", "K21 real", "K21 ring test", "K21 calib",
                                                      "K21 recomplete", "K21 project", "K21 combine"))
    out = [("pilot difference at sigma_21", cv["pilot_d21"], 1.59e-31, 0.01e-31, [(P, "err21 = plateau + 1.59×10⁻³¹"), (P, "|σ₂₁(a) − σ₂₁(b)| = 1.59×10⁻³¹, is 0.118")]),
           ("pilot difference at sigma_21 (README)", cv["pilot_d21"], 1.6e-31, 0.05e-31, [(R, "run 1.6×10⁻³¹ away")]),
           ("pilot difference over (a)'s plateau err21", cv["pilot"]["d21_over_err21a"], 0.118, 0.001, [(P, "|σ₂₁(a) − σ₂₁(b)| = 1.59×10⁻³¹, is 0.118")])]
    for w, e, r, m, pl, d2 in (("0.99", 1.40e-30, 1.03e-12, 97, 1.24e-30, 0.0070), ("1.0", 1.51e-30, 1.14e-12, 88, 1.35e-30, 0.0070),
                               ("1.01", 1.63e-30, 1.26e-12, 79, 1.47e-30, 0.0069)):
        n = cv["nodes"][w]
        cell = f"{e / 1e-30:.2f}×10⁻³⁰ ({r / 1e-12:.2f}×10⁻¹², {m})"
        out += [(f"err21 at wf {w}", n["err21"], e, 0.01e-30, [(P, cell)]), (f"relative err21 at wf {w}", n["err21_rel"], r, 0.01e-12, [(P, cell)]),
                (f"margin at wf {w} (K = 21)", 1e-10 / n["err21_rel"], m, 1, [(P, cell)]),
                (f"plateau err21 at wf {w}", n["err21_plateau"], pl, 0.01e-30, [(P, "1.24, 1.35 and 1.47×10⁻³⁰")]),
                (f"ratio-law second difference at k = 19, wf {w}", cv["ratiolaw_d2"][w][-1], d2, 0.0001, [(P, f"| {d2:.4f} |")])]
    mp.mp.dps = 40
    a = json.load(open(os.path.join(d, "scan43J1250/sigF_w1.0000_K43_J1250_p1280.json")))
    sg = K14.sigmas(a)
    q = complex(sg[20] / sg[21])
    out += [("sigma_20 / sigma_21 at wf 1.00, real part", q.real, 10.25, 0.01, [(P, "σ₂₀/σ₂₁ = 10.25 − 9.63i")]),
            ("sigma_20 / sigma_21 at wf 1.00, imaginary part", q.imag, -9.63, 0.01, [(P, "σ₂₀/σ₂₁ = 10.25 − 9.63i")])]
    sp = cv["spread"]
    out += [("spread of sigma_21 in wf", sp["21"]["half_range_rel"] * 100, 6.4, 0.1, [(P, "σ₂₁ changes by 6.4% of itself")]),
            ("spread of sigma_19 in wf (13.2)", sp["19"]["half_range_rel"] * 100, 6.2, 0.1, [(P, "(6.2%, 6.3% for σ₁₉, σ₂₀)")]),
            ("spread of sigma_20 in wf (13.2)", sp["20"]["half_range_rel"] * 100, 6.3, 0.1, [(P, "(6.2%, 6.3% for σ₁₉, σ₂₀)")]),
            ("err21 over the wf change", sp["21"]["err_over_d1"], 1.92e-11, 0.01e-11, [(P, "1.92×10⁻¹¹ of that change")]),
            ("err19 over the wf change (13.2)", sp["19"]["err_over_d1"], 8.68e-12, 0.01e-12, [(P, "(8.68×10⁻¹² at σ₁₉, 1.24×10⁻¹¹ at σ₂₀)")]),
            ("err20 over the wf change (13.2)", sp["20"]["err_over_d1"], 1.24e-11, 0.01e-11, [(P, "(8.68×10⁻¹² at σ₁₉, 1.24×10⁻¹¹ at σ₂₀)")])]
    recs = {x["name"]: x for x in (json.loads(l) for l in open(os.path.join(d, "planted_results_k21.jsonl")) if l.strip())}
    plants = [x for x in recs.values() if not x["name"].startswith(("N0", "C1", "C2"))]
    out += [("plants at K = 21", sum(pc["families"].values()), 51, 0.5, [(P, "51 plants at K = 19, 20, 21 (12.2 min in all)")]),
            ("plants' minutes at K = 21", sum(x["seconds"] for x in plants) / 60, 12.2, 0.1, [(P, "51 plants at K = 19, 20, 21 (12.2 min in all)")]),
            ("N1 worst error at K = 21", pc["n1_worst"], 3.5e-7, 0.1e-7, [(P, "Padé and G to 3.5×10⁻⁷ at worst")])]
    e = rl["est"]
    rows = {"arg": "| arg crossing | 1.0000004 | 1.0000009 | 1.0000015 |", "ratio": "| ratio | 0.999992 | 0.999988 | 0.999986 |",
            "G": "| G (production κ̄) | 1.0000001 | 1.0000001 | 0.9999995 |", "aitken": "| Aitken, k − 2 … k | 0.9999997 | 0.9999991 | 1.0000009 |"}
    for K, vals in (("21", {"arg": (1.0000004, 1e-7), "ratio": (0.999992, 1e-6), "G": (1.0000001, 1e-7), "aitken": (0.9999997, 1e-7)}),
                    ("20", {"arg": (1.0000009, 1e-7), "ratio": (0.999988, 1e-6), "G": (1.0000001, 1e-7), "aitken": (0.9999991, 1e-7)}),
                    ("19", {"arg": (1.0000015, 1e-7), "ratio": (0.999986, 1e-6), "G": (0.9999995, 1e-7), "aitken": (1.0000009, 1e-7)})):
        for est, (qv, u) in vals.items():
            out.append((f"{est} at K = {K} (section 13.4)", e[K][est], qv, u, [(P, rows[est])]))
    out += [("arg crossing at K = 21 (README)", e["21"]["arg"], 1.0000004, 1e-7, [(R, "The arg crossing falls to 1.0000004")]),
            ("G at K = 21 (README)", e["21"]["G"], 1.0000001, 1e-7, [(R, "the factored series stays at 1.0000001")])]
    rec = recs["N0 scan43J1250"]
    pr = rec["est"]["21"]["pade_central_range"]
    out += [("central Pade range at K = 21, low", pr[0], 0.9999987, 1e-7, [(P, "[3/17]–[16/4]: 0.9999987–1.0000001 (mid 0.9999994)")]),
            ("central Pade range at K = 21, high", pr[1], 1.0000001, 1e-7, [(P, "[3/17]–[16/4]: 0.9999987–1.0000001 (mid 0.9999994)")]),
            ("central Pade mid at K = 21", rec["est"]["21"]["pade_central_mid"], 0.9999994, 1e-7, [(P, "[3/17]–[16/4]: 0.9999987–1.0000001 (mid 0.9999994)")]),
            ("[3/17] below 1 at K = 21, in 1e-6", (1 - pr[0]) * 1e6, 1.3, 0.1, [(P, "**[3/17] is 1.3×10⁻⁶ below 1**"), (R, "([3/17] is 1.3×10⁻⁶ below)")])]
    ag = {int(k): v for k, v in rec["est"]["21"]["arg"].items()}
    out += [("arg crossing at k = 18 (section 13.4)", ag[18], 1.0000024, 1e-7, [(P, "1.0000024, 1.0000015, 1.0000009, 1.0000004 for k = 18–21")]),
            ("shrink of arg_k - 1, 18 to 19 (13.4)", (ag[19] - 1) / (ag[18] - 1), 0.63, 0.01, [(P, "shrank by ×0.63 from k = 18 to 19")]),
            ("shrink of arg_k - 1, 19 to 20 (13.4)", (ag[20] - 1) / (ag[19] - 1), 0.57, 0.01, [(P, "×0.57 from 19 to 20 and ×0.51 from 20 to 21")]),
            ("shrink of arg_k - 1, 20 to 21", (ag[21] - 1) / (ag[20] - 1), 0.51, 0.01, [(P, "×0.57 from 19 to 20 and ×0.51 from 20 to 21")]),
            ("G over 1 at K = 21", (e["21"]["G"] - 1) * 1e7, 0.6, 0.1, [(P, "0.6×10⁻⁷ over it")]),
            ("last two differences of arg_k (13.4)", abs(ag[20] - ag[19]) * 1e6, 0.65, 0.01, [(P, "are 0.65×10⁻⁶, then 0.42×10⁻⁶")]),
            ("last difference of arg_k (13.4)", abs(ag[21] - ag[20]) * 1e6, 0.42, 0.01, [(P, "are 0.65×10⁻⁶, then 0.42×10⁻⁶")])]
    sz = rt["simple-zero law from tau_19, tau_20"]["tau21_rel"]
    out += [("simple-zero law at K = 21, best node", min(sz), 3.2e-6, 0.1e-6, [(P, "misses τ₂₁ by 3.2–6.0×10⁻⁶")]),
            ("simple-zero law at K = 21, worst node", max(sz), 6.0e-6, 0.1e-6, [(P, "misses τ₂₁ by 3.2–6.0×10⁻⁶")])]
    for name, lo, hi, u, sub in (("C20-BP [3/17]", 1.4e-6, 1.5e-6, 0.1e-6, "| 1.4–1.5×10⁻⁶ | 6.4–8.0×10⁻⁵ |"),
                                 ("C20-BP [4/16]", 4.0e-7, 4.2e-7, 0.1e-7, "| 4.0–4.2×10⁻⁷ | 1.9–2.1×10⁻⁵ |"),
                                 ("C20-BP [5/15]", 4.0e-7, 4.1e-7, 0.1e-7, "| 4.0–4.1×10⁻⁷ | 1.8–2.2×10⁻⁵ |"),
                                 ("C20-BP [6/14]", 3.0e-8, 3.3e-8, 0.1e-8, "| 3.0–3.3×10⁻⁸ | 1.5–1.6×10⁻⁶ |"),
                                 ("C20-BP [8/12]", 6.1e-9, 6.4e-9, 0.1e-9, "| 6.1–6.4×10⁻⁹ | 2.7–3.4×10⁻⁷ |"),
                                 ("C20-BP [9/11]", 6.1e-10, 7.6e-10, 0.1e-10, "| 6.1–7.6×10⁻¹⁰ | 2.7–3.7×10⁻⁸ |"),
                                 ("C19-BP [4/15]", 0.8e-6, 1.4e-6, 0.1e-6, "| 0.8–1.4×10⁻⁶ | 1.3–3.7×10⁻⁵ |")):
        v = rt[name]["sigma21_rel"]
        out += [(f"{name} sigma_21 error, best node", min(v), lo, u, [(P, sub)]),
                (f"{name} sigma_21 error, worst node", max(v), hi, u, [(P, sub)])]
    v = rt["C20-BP [7/13]"]["sigma21_rel"]
    out += [("C20-BP [7/13] sigma_21 error, best node", min(v), 0.85e-8, 0.01e-8, [(P, "| 0.85–1.1×10⁻⁸ | 4.6–4.9×10⁻⁷ |")]),
            ("C20-BP [7/13] sigma_21 error, worst node", max(v), 1.1e-8, 0.1e-8, [(P, "| 0.85–1.1×10⁻⁸ | 4.6–4.9×10⁻⁷ |")])]
    RF20 = ("C20-BP [6/14]", "C20-BP [7/13]", "C20-BP [8/12]", "C20-BP [9/11]", "C20-BP [10/10]")
    FAM25 = RF20 + ("C19-BP [6/13]", "C19-BP [7/12]", "C19-BP [8/11]", "C19-BP [9/10]", "C18-BP [6/12]", "C18-BP [7/11]", "C18-BP [8/10]",
                    "C18-BP [9/9]", "C17-BP [6/11]", "C17-BP [7/10]", "C17-BP [8/9]", "C16-BP [6/10]", "C16-BP [7/9]", "C16-BP [8/8]",
                    "C15-BP [6/9]", "C15-BP [7/8]", "C14-BP [5/9]", "C14-BP [6/8]", "C13-BP [5/8]", "C13-BP [6/7]")
    rings = {n: v for n, v in rt.items() if isinstance(v, dict) and v.get("ring") and not v.get("context")}
    both = "All ten rings of the K = 20 generation are rejected under both readings of item 4"
    out += [("C19-BP [4/15] rejected by D-bracket alone at K = 21", int(rt["C19-BP [4/15]"]["extra_detectors"] == ["D-bracket"]), 1, 0.5, []),
            ("K = 20 rings counted", len(rings), 10, 0.5, [(P, both)]),
            ("K = 20 rings rejected under the primary reading", sum(1 for v in rings.values() if not v["survives"]), 10, 0.5, [(P, both)]),
            ("K = 20 rings surviving without D-bracket", sum(1 for v in rings.values() if v["survives_without_bracket"]), 0, 0.5, [(P, both)]),
            ("ring-free C20-BP sums, best sigma_21 prediction", min(min(rt[n]["sigma21_rel"]) for n in RF20), 6e-10, 1e-10,
             [(P, "predicted σ₂₁ to 6×10⁻¹⁰–3×10⁻⁸"), (R, "predicted it to 6×10⁻¹⁰–3×10⁻⁸")]),
            ("ring-free C20-BP sums, worst sigma_21 prediction", max(max(rt[n]["sigma21_rel"]) for n in RF20), 3e-8, 1e-8,
             [(P, "predicted σ₂₁ to 6×10⁻¹⁰–3×10⁻⁸"), (R, "predicted it to 6×10⁻¹⁰–3×10⁻⁸")]),
            ("BP family of 25: rejected by D-bracket alone", sum(1 for n in FAM25 if rt[n]["extra_detectors"] == ["D-bracket"]), 6, 0.5,
             [(P, "rejects six ring-free sums and the context ring")]),
            ("BP family of 25: survivors", sum(1 for n in FAM25 if rt[n]["survives"]), 13, 0.5, [(P, "the other thirteen sums of the family of twenty-five survive")])]
    z = lambda fam, est: cal["21"][fam]["cells"][est]["zero"]
    arc_arg = [(P, "−0.01…+0.02 → 0.999998–1.000002"), (R, "the arg crossing puts it at 0.999998–1.000002")]
    out += [("arc plants, arg, zero low (K = 21)", z("all 28 arc plants", "arg")[0], 0.999998, 1e-6, arc_arg),
            ("arc plants, arg, zero high (K = 21)", z("all 28 arc plants", "arg")[1], 1.000002, 1e-6, arc_arg),
            ("arc plants, arg error low (K = 21)", cal["21"]["all 28 arc plants"]["cells"]["arg"]["lo"] * 1e4, -0.01, 0.01, [(P, "−0.01…+0.02 → 0.999998–1.000002")]),
            ("arc plants, arg error high (K = 21)", cal["21"]["all 28 arc plants"]["cells"]["arg"]["hi"] * 1e4, 0.02, 0.01, [(P, "−0.01…+0.02 → 0.999998–1.000002")]),
            ("BP family (25), arg, zero low", z("BP family (25)", "arg")[0], 0.999999, 1e-6, [(P, "0.00…+0.02 → 0.999999–1.000000")]),
            ("BP family (25), arg, zero high", z("BP family (25)", "arg")[1], 1.000000, 1e-6, [(P, "0.00…+0.02 → 0.999999–1.000000")]),
            ("arc plants, ratio, zero low (K = 21)", z("all 28 arc plants", "ratio")[0], 1.000002, 1e-6, [(P, "−0.35…−0.10 → 1.000002–1.000027")]),
            ("arc plants, ratio, zero high (K = 21)", z("all 28 arc plants", "ratio")[1], 1.000027, 1e-6, [(P, "−0.35…−0.10 → 1.000002–1.000027")]),
            ("arc plants, Pade, zero low (K = 21)", z("all 28 arc plants", "pade")[0], 0.999998, 1e-6, [(P, "0.00…+0.02 → 0.999998–0.999999")]),
            ("arc plants, Pade, zero high (K = 21)", z("all 28 arc plants", "pade")[1], 0.999999, 1e-6, [(P, "0.00…+0.02 → 0.999998–0.999999")]),
            ("BP family (25), Pade, zero low", z("BP family (25)", "pade")[0], 0.999988, 1e-6, [(P, "−0.21…+0.11 → 0.999988–1.000020")]),
            ("BP family (25), Pade, zero high", z("BP family (25)", "pade")[1], 1.000020, 1e-6, [(P, "−0.21…+0.11 → 0.999988–1.000020")]),
            ("BP family (25), G, zero high", z("BP family (25)", "G")[1], 1.000001, 1e-6, [(P, "−0.01…+0.01 → 0.999999–1.000001")])]
    dv = cal["sigma_dev_summary"]
    arc = [dv[f] for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2")]
    dev = "by 2.2–3.9% and 3.0–5.6% (worst node)"
    out += [("arc plants' sigma_20 deviation, least (13.6)", min(x["sigma20"][0] for x in arc) * 100, 2.2, 0.1, [(P, dev)]),
            ("arc plants' sigma_20 deviation, most (13.6)", max(x["sigma20"][1] for x in arc) * 100, 3.9, 0.1, [(P, dev)]),
            ("arc plants' sigma_21 deviation, least", min(x["sigma21"][0] for x in arc) * 100, 3.0, 0.1, [(P, dev)]),
            ("arc plants' sigma_21 deviation, most", max(x["sigma21"][1] for x in arc) * 100, 5.6, 0.1, [(P, dev)]),
            ("C-EXP sigma_21 deviation", dv["C-EXP"]["sigma21"][1] * 100, 2.1, 0.1, [(P, "and 2.1%, the REFIT plants' by 1.5–21% and 1.7–27%")]),
            ("REFIT sigma_21 deviation, most", dv["REFIT"]["sigma21"][1] * 100, 27, 1, [(P, "and 2.1%, the REFIT plants' by 1.5–21% and 1.7–27%")])]
    for name, zq, u, pole, nz in (("C21-BP [3/18]", 0.99999997, 1e-8, 1.73, 9), ("C21-BP [4/17]", 0.99999999, 1e-8, 1.74, 8),
                                  ("C21-BP [6/15]", 1.000000009, 1e-9, 1.78, 1), ("C21-BP [7/14]", 1.000000004, 1e-9, 1.80, 1),
                                  ("C21-BP [8/13]", 0.999999998, 1e-9, 1.88, 1), ("C21-BP [9/12]", 1.000000002, 1e-9, 1.83, 1),
                                  ("C21-BP [10/11]", 1.000000002, 1e-9, 1.83, 1)):
        r = rc[name]
        digits = {1e-8: 8, 1e-9: 9}[u]
        out += [(f"{name} zero", r["truth"], zq, u, [(P, f"| {name} | {zq:.{digits}f} |")]),
                (f"{name} largest pole", max(r["max_pole"].values()), pole, 0.01, []),
                (f"{name} zeros in |x| < 12", r["zeros_lt_12"], nz, 0.5, []),
                (f"{name} type rule", int(bool(r["type_rule"])), 1, 0.5, [])]
    out += [("C21-BP [5/16] type rule fails", int(bool(rc["C21-BP [5/16]"]["type_rule"])), 0, 0.5, [(P, "| C21-BP [5/16] | 0.99097 | 12 / 34 / 54 | fails")]),
            ("C21-BP [5/16] zeros in |x| < 7.3", rc["C21-BP [5/16]"]["zeros_lt_7.3"], 12, 0.5, [(P, "| C21-BP [5/16] | 0.99097 | 12 / 34 / 54 | fails")])]
    W = [rc[f"C21-W{i}"]["truth"] for i in (1, 2, 3)]
    A = [rc[f"C21-ARC {k}"] for k in ("arc", "chord", "mirror", "beta2")]
    out += [("C21-W zeros, low", min(W), 0.999925, 1e-6, [(P, "| 0.999925–0.999944 | 5–6 / 19 / – |")]),
            ("C21-W zeros, high", max(W), 0.999944, 1e-6, [(P, "| 0.999925–0.999944 | 5–6 / 19 / – |")]),
            ("C21-ARC zeros, low", min(x["truth"] for x in A), 0.99996, 1e-5, [(P, "| 0.99996–0.99999 | 0–4 / 17–19 / – |")]),
            ("C21-ARC zeros, high", max(x["truth"] for x in A), 0.99999, 1e-5, [(P, "| 0.99996–0.99999 | 0–4 / 17–19 / – |")]),
            ("C21-ARC best omega_t, low", min(float(x["best_wt"]) for x in A), 0.99908, 1e-5, [(P, "best ω_t 0.99908–1.00059")]),
            ("C21-ARC best omega_t, high", max(float(x["best_wt"]) for x in A), 1.00059, 1e-5, [(P, "best ω_t 0.99908–1.00059")]),
            ("C21-ARC L2 floor, low", min(x["L2_floor"] for x in A), 2.0e-3, 0.1e-3, [(P, "L2 floor 2.0–5.2×10⁻³")]),
            ("C21-ARC L2 floor, high", max(x["L2_floor"] for x in A), 5.2e-3, 0.1e-3, [(P, "L2 floor 2.0–5.2×10⁻³")])]
    h = pj["hyp"]
    out += [("projection for a zero at 1 (K = 21)", h["1.0"]["pred21"] - 1, 0.50e-6, 0.01e-6, [(P, "predicts arg₂₁ − 1 = +0.50×10⁻⁶ for ω* = 1")]),
            ("projection for a zero at 0.999997", h["0.999997"]["pred21"] - 1, 0.31e-6, 0.01e-6, [(P, "+0.31×10⁻⁶ for")]),
            ("projection for a zero at 0.999975 (K = 21)", h["0.999975"]["pred21"] - 1, 0.23e-6, 0.01e-6, [(P, "0.999997 and +0.23×10⁻⁶ for 0.999975")]),
            ("real arg_21 - 1", pj["arg"]["21"] - 1, 0.44e-6, 0.01e-6, [(P, "the real value is +0.44×10⁻⁶")]),
            ("miss, zero at 1 (K = 21)", abs(h["1.0"]["miss"]), 0.05e-6, 0.01e-6, [(P, "0.05×10⁻⁶ from the first")]),
            ("miss, zero at 0.999997", abs(h["0.999997"]["miss"]), 0.13e-6, 0.01e-6, [(P, "0.13×10⁻⁶ from the second")]),
            ("miss, zero at 0.999975 (K = 21)", abs(h["0.999975"]["miss"]), 0.21e-6, 0.01e-6, [(P, "and 0.21×10⁻⁶ from the third")]),
            ("model error (K = 21)", pj["model_err"], 2.7e-6, 0.1e-6, [(P, "The model's own error is 2.7×10⁻⁶")]),
            ("back-test, least (K = 21)", min(pj["back_real_max"].values()), 0.4e-6, 0.1e-6, [(P, "0.4–2.7×10⁻⁶ depending on ω*")]),
            ("back-test, most (K = 21)", max(pj["back_real_max"].values()), 2.7e-6, 0.1e-6, [(P, "0.4–2.7×10⁻⁶ depending on ω*")]),
            ("plant misses, low (K = 21)", min(p["miss"] for p in pj["plants"]), -1.1e-6, 0.1e-6, [(P, "−1.1×10⁻⁶…+0.1×10⁻⁶")]),
            ("plant misses, high (K = 21)", max(p["miss"] for p in pj["plants"]), 0.1e-6, 0.1e-6, [(P, "−1.1×10⁻⁶…+0.1×10⁻⁶")]),
            ("trend per order, k = 4-21", pj["trend_4_21"]["per_order"], 0.543, 0.001, [(P, "×0.543 per order (×0.542 over k = 4–20)")]),
            ("trend per order, k = 4-20 (13.8)", pj["trend_4_20"]["per_order"], 0.542, 0.001, [(P, "×0.543 per order (×0.542 over k = 4–20)")]),
            ("trend's arg_21 - 1", pj["trend_4_20"]["at21"], 0.42e-6, 0.01e-6, [(P, "puts arg₂₁ − 1 at 0.42×10⁻⁶")]),
            ("noise shift of arg_21", max(abs(x) for x in pj["noise_shift"]), 1.0e-9, 0.1e-9, [(P, "moves by at most 1.0×10⁻⁹")]),
            ("projection decisive (K = 21)", int(bool(pj["decisive"])), 0, 0.5, [])]
    rf = list(cb["ring_free"].values())
    r20 = json.load(open(os.path.join(d, "planted_checks_k20.json")))["K20 combine"]["range20"]
    r19 = json.load(open(os.path.join(d, "planted_checks_k19.json")))["K19 combine"]["range19"]
    rfs = [(P, "the ring-free completions, 0.999999998–1.000000009"), (R, "0.999999998–1.000000009. Calibrated on the planted functions"),
           (M, "put the zero at $0.999999998$--$1.000000009$")]
    out += [("K = 21 hull, low", cb["range21"][0], 0.999998, 1e-6, [(P, "**The hull is 0.999998–1.000002.**")]),
            ("K = 21 hull, high", cb["range21"][1], 1.000002, 1e-6, [(P, "**The hull is 0.999998–1.000002.**")]),
            ("K = 20 hull, low (13.9)", r20[0], 0.999997, 1e-6, [(P, "The K = 21 range lies inside the K = 20 one (0.999997–1.000002)")]),
            ("ring-free completions, low (K = 21)", min(rf), 0.999999998, 1e-9, rfs),
            ("ring-free completions, high (K = 21)", max(rf), 1.000000009, 1e-9, rfs),
            ("ring-free completions counted (K = 21)", len(rf), 5, 0.5, [(P, "C21-BP [6/15], [7/14], [8/13], [9/12], [10/11],")]),
            ("headline, low (K = 21)", cb["headline"][0], 0.999975, 1e-6, [(P, "**0.999975–1.000050, unchanged.**")]),
            ("headline, high (K = 21)", cb["headline"][1], 1.000050, 1e-6, [(P, "**0.999975–1.000050, unchanged.**")]),
            ("headline overlap (K = 21)", int(bool(cb["overlap_k20"])), 1, 0.5, []),
            ("last two orders, low (K = 21)", min(r20[0], cb["range21"][0]), 0.999997, 1e-6, [(P, "the last two orders alone, reported as context by item 7, is 0.999997–1.000002")]),
            ("last two orders, high (K = 21)", max(r20[1], cb["range21"][1]), 1.000002, 1e-6, [(P, "the last two orders alone, reported as context by item 7, is 0.999997–1.000002")]),
            ("K = 19 to 21, low", min(r19[0], r20[0], cb["range21"][0]), 0.999996, 1e-6, [(P, "0.999996–1.000004. An estimator range, not a bound."), (P, "and K = 19–21 would give 0.999996–1.000004.")]),
            ("K = 19 to 21, high", max(r19[1], r20[1], cb["range21"][1]), 1.000004, 1e-6, [(P, "0.999996–1.000004. An estimator range, not a bound."), (P, "and K = 19–21 would give 0.999996–1.000004.")]),
            ("K = 21 hull width", (cb["range21"][1] - cb["range21"][0]) * 1e5, 0.3, 0.1, [(P, "a width of 0.3×10⁻⁵ against 0.5×10⁻⁵ at K = 20")]),
            ("K = 20 hull width (13.10)", (r20[1] - r20[0]) * 1e5, 0.5, 0.1, [(P, "a width of 0.3×10⁻⁵ against 0.5×10⁻⁵ at K = 20")]),
            ("G production at K = 21", cb["G_minima"]["21"]["production"], 1.0000001, 1e-7, [(P, "production 1.0000001 at K = 21 (1.0000001 at K = 20)")]),
            ("G production at K = 20 (13.9)", cb["G_minima"]["20"]["production"], 1.0000001, 1e-7, [(P, "production 1.0000001 at K = 21 (1.0000001 at K = 20)")]),
            ("G minima span at K = 21, low", cb["G_minima"]["21"]["span"][0], 1.0000000, 1e-7, [(P, "factor 5 of the best, 1.0000000–1.0000001 (1.000000–1.000003 at K = 20)")]),
            ("G minima span at K = 21, high", cb["G_minima"]["21"]["span"][1], 1.0000001, 1e-7, [(P, "factor 5 of the best, 1.0000000–1.0000001 (1.000000–1.000003 at K = 20)")]),
            ("G minima span at K = 20, high (13.9)", cb["G_minima"]["20"]["span"][1], 1.000003, 1e-6, [(P, "factor 5 of the best, 1.0000000–1.0000001 (1.000000–1.000003 at K = 20)")]),
            ("G global minimum at K = 21", cb["G_minima"]["21"]["global"], 1.0000001, 1e-7, [(P, "its global minimum 1.0000001 (1.000000)")]),
            ("G global minimum at K = 20 (13.9)", cb["G_minima"]["20"]["global"], 1.000000, 1e-6, [(P, "its global minimum 1.0000001 (1.000000)")])]
    nodes = cv["nodes"]
    conv = [(P, "σ₂₁ is converged to 1.0–1.3×10⁻¹² relative"), (R, "1.0–1.3×10⁻¹² relative at the three ω")]
    out += [("relative err21 at wf 0.99 (verdict)", nodes["0.99"]["err21_rel"], 1.0e-12, 0.1e-12, conv),
            ("relative err21 at wf 1.01 (verdict)", nodes["1.01"]["err21_rel"], 1.3e-12, 0.1e-12, conv),
            ("margins at K = 21, least", min(1e-10 / nodes[w]["err21_rel"] for w in nodes), 79, 1, [(P, "with margins of 79–97 under item 1")]),
            ("margins at K = 21, most", max(1e-10 / nodes[w]["err21_rel"] for w in nodes), 97, 1, [(P, "with margins of 79–97 under item 1")])]
    return out


@check
def k22_stages():
    """notes/PLANTED_SELECTION.md sections 14.2-14.10: the K = 22 stages ("K22 converge" ... "K22 combine" in
    solver/inner_amp/planted_checks_k22.json and the record "N0 scan45J1250" in planted_results_k22.jsonl), with the
    README's twenty-second-order paragraph and the mathematics paper's sentence on sigma_22."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import planted_k14 as K14
    sys.path.pop(0)
    import mpmath as mp
    P, R, M = "notes/PLANTED_SELECTION.md", "README.md", "paper/math/main.tex"
    c = json.load(open(os.path.join(d, "planted_checks_k22.json")))
    cv, pc, rl, rt, cal, rc, pj, cb = (c[k] for k in ("K22 converge", "K22 plants check", "K22 real", "K22 ring test", "K22 calib",
                                                      "K22 recomplete", "K22 project", "K22 combine"))
    out = [("pilot difference at sigma_22", cv["pilot_d22"], 8.99e-33, 0.01e-33, [(P, "err22 = plateau + 8.99×10⁻³³")]),
           ("pilot difference at sigma_22 (README)", cv["pilot_d22"], 9.0e-33, 0.1e-33, [(R, "with the J = 1500 run 9.0×10⁻³³ away")]),
           ("pilot difference over (a)'s plateau err22", cv["pilot"]["d22_over_err22a"], 0.069, 0.001, [(P, "is 0.069 of (a)'s")])]
    for w, e, r, m, pl in (("0.99", 1.28e-31, 1.38e-12, 73, 1.19e-31), ("1.0", 1.39e-31, 1.53e-12, 65, 1.30e-31), ("1.01", 1.50e-31, 1.70e-12, 59, 1.41e-31)):
        n = cv["nodes"][w]
        cell = f"{e / 1e-31:.2f}×10⁻³¹ ({r / 1e-12:.2f}×10⁻¹², {m})"
        out += [(f"err22 at wf {w}", n["err22"], e, 0.01e-31, [(P, cell)]), (f"relative err22 at wf {w}", n["err22_rel"], r, 0.01e-12, [(P, cell)]),
                (f"margin at wf {w} (K = 22)", 1e-10 / n["err22_rel"], m, 1, [(P, cell)]),
                (f"plateau err22 at wf {w}", n["err22_plateau"], pl, 0.01e-31, [(P, "The plateau errors are 1.19, 1.30 and 1.41×10⁻³¹")]),
                (f"ratio-law second difference at k = 20, wf {w}", cv["ratiolaw_d2"][w][-1], 0.0066, 0.0001, [(P, "| 0.0066 |")]),
                (f"sigma_0..21 bitwise at wf {w}", int(bool(n["ref_bitwise"])), 1, 0.5, [(P, "bitwise equal, 22/22")])]
    lowr = {k: v[0] / v[1] for k, v in cv["pilot"]["low"].items()}
    out += [("pilot (b) sigma_19 over (a)'s plateau error", lowr["19"], 0.013, 0.001, [(P, "0.013, 0.011 and 0.118")]),
            ("pilot (b) sigma_20 over (a)'s plateau error", lowr["20"], 0.011, 0.001, [(P, "0.013, 0.011 and 0.118")]),
            ("pilot (b) sigma_21 over (a)'s plateau error", lowr["21"], 0.118, 0.001, [(P, "0.013, 0.011 and 0.118")])]
    mp.mp.dps = 40
    sg = K14.sigmas(json.load(open(os.path.join(d, "scan45J1250/sigF_w1.0000_K45_J1250_p1280.json"))))
    q = complex(sg[21] / sg[22])
    out += [("sigma_21 / sigma_22 at wf 1.00, real part", q.real, 10.67, 0.01, [(P, "σ₂₁/σ₂₂ = 10.67 − 9.93i")]),
            ("sigma_21 / sigma_22 at wf 1.00, imaginary part", q.imag, -9.93, 0.01, [(P, "σ₂₁/σ₂₂ = 10.67 − 9.93i")])]
    sp = cv["spread"]
    out += [("spread of sigma_22 in wf", sp["22"]["half_range_rel"] * 100, 6.5, 0.1, [(P, "σ₂₂ changes by 6.5% of itself")]),
            ("spread of sigma_20 in wf (14.2)", sp["20"]["half_range_rel"] * 100, 6.3, 0.1, [(P, "(6.3%, 6.4% for σ₂₀, σ₂₁)")]),
            ("spread of sigma_21 in wf (14.2)", sp["21"]["half_range_rel"] * 100, 6.4, 0.1, [(P, "(6.3%, 6.4% for σ₂₀, σ₂₁)")]),
            ("err22 over the wf change", sp["22"]["err_over_d1"], 2.55e-11, 0.01e-11, [(P, "2.55×10⁻¹¹ of that change")]),
            ("err20 over the wf change (14.2)", sp["20"]["err_over_d1"], 1.24e-11, 0.01e-11, [(P, "(1.24×10⁻¹¹ at σ₂₀, 1.73×10⁻¹¹ at σ₂₁)")]),
            ("err21 over the wf change (14.2)", sp["21"]["err_over_d1"], 1.73e-11, 0.01e-11, [(P, "(1.24×10⁻¹¹ at σ₂₀, 1.73×10⁻¹¹ at σ₂₁)")])]
    recs = {x["name"]: x for x in (json.loads(l) for l in open(os.path.join(d, "planted_results_k22.jsonl")) if l.strip())}
    plants = [x for x in recs.values() if not x["name"].startswith(("N0", "C1", "C2"))]
    out += [("plants at K = 22", sum(pc["families"].values()), 51, 0.5, [(P, "51 plants at K = 20, 21, 22 (10.2 min in all)")]),
            ("plants' minutes at K = 22", sum(x["seconds"] for x in plants) / 60, 10.2, 0.1, [(P, "51 plants at K = 20, 21, 22 (10.2 min in all)")]),
            ("plants' reproduction at K = 22, worst", pc["repro_worst"][0], 1.4e-12, 0.1e-12, [(P, "records to 1.4×10⁻¹² at worst")]),
            ("N1 worst error at K = 22", pc["n1_worst"], 3.5e-7, 0.1e-7, [(P, "Padé and G to 3.5×10⁻⁷ at worst")])]
    e = rl["est"]
    rows = {"arg": "| arg crossing | 1.0000002 | 1.0000004 | 1.0000009 |", "ratio": "| ratio | 0.999996 | 0.999992 | 0.999988 |",
            "G": "| G (production κ̄) | 1.0000000 | 1.0000001 | 1.0000001 |", "aitken": "| Aitken, k − 2 … k | 1.0000000 | 0.9999997 | 0.9999991 |"}
    for K, vals in (("22", {"arg": (1.0000002, 1e-7), "ratio": (0.999996, 1e-6), "G": (1.0000000, 1e-7), "aitken": (1.0000000, 1e-7)}),
                    ("21", {"arg": (1.0000004, 1e-7), "ratio": (0.999992, 1e-6), "G": (1.0000001, 1e-7), "aitken": (0.9999997, 1e-7)}),
                    ("20", {"arg": (1.0000009, 1e-7), "ratio": (0.999988, 1e-6), "G": (1.0000001, 1e-7), "aitken": (0.9999991, 1e-7)})):
        for est, (qv, u) in vals.items():
            out.append((f"{est} at K = {K} (section 14.4)", e[K][est], qv, u, [(P, rows[est])]))
    rec = recs["N0 scan45J1250"]["est"]["22"]
    pr, vals = rec["pade_central_range"], rec["pade_central_vals"]
    pc_text = [(P, "[3/18]–[17/4]: 0.9999998–1.0000000 (mid 0.9999999)")]
    ag = {int(k): v for k, v in rec["arg"].items()}
    out += [("arg crossing at K = 22 (README)", e["22"]["arg"], 1.0000002, 1e-7, [(R, "The arg crossing falls to 1.0000002")]),
            ("G at K = 22 (README)", e["22"]["G"], 0.99999999, 1e-8, [(R, "the factored series to 0.99999999")]),
            ("G below 1 at K = 22, in 1e-9", (1 - e["22"]["G"]) * 1e9, 7, 1, [(P, "7×10⁻⁹ below 1")]),
            ("central Pade range at K = 22, low", pr[0], 0.9999998, 1e-7, pc_text),
            ("central Pade range at K = 22, high", pr[1], 1.0000000, 1e-7, pc_text),
            ("central Pade mid at K = 22", rec["pade_central_mid"], 0.9999999, 1e-7, pc_text),
            ("[3/18] below 1 at K = 22, in 1e-7", (1 - vals[0]) * 1e7, 2.1, 0.1, [(P, "is 2.1×10⁻⁷ below"), (R, "([3/18] is 2.1×10⁻⁷ below)")]),
            ("[4/17]-[17/4] within 5e-8 of 1 (1 = yes)", float(max(abs(v - 1) for v in vals[1:]) <= 5e-8), 1, 0.5,
             [(P, "lie within 5×10⁻⁸ of 1"), (R, "lie within 5×10⁻⁸ of 1")]),
            ("shrink of arg_k - 1, 20 to 21 (14.4)", (ag[21] - 1) / (ag[20] - 1), 0.51, 0.01, [(P, "shrank by ×0.51 from k = 20 to 21")]),
            ("shrink of arg_k - 1, 21 to 22", (ag[22] - 1) / (ag[21] - 1), 0.52, 0.01, [(P, "×0.52 from 21 to 22")]),
            ("last two differences of arg_k (14.4)", abs(ag[21] - ag[20]) * 1e6, 0.42, 0.01, [(P, "are 0.42×10⁻⁶, then 0.21×10⁻⁶")]),
            ("last difference of arg_k (14.4)", abs(ag[22] - ag[21]) * 1e6, 0.21, 0.01, [(P, "are 0.42×10⁻⁶, then 0.21×10⁻⁶")])]
    sz = rt["simple-zero law from tau_20, tau_21"]["tau22_rel"]
    out += [("simple-zero law at K = 22, best node", min(sz), 3.8e-6, 0.1e-6, [(P, "misses τ₂₂ by 3.8–4.1×10⁻⁶")]),
            ("simple-zero law at K = 22, worst node", max(sz), 4.1e-6, 0.1e-6, [(P, "misses τ₂₂ by 3.8–4.1×10⁻⁶")])]
    for name, lo, hi, u, sub in (("C21-BP [4/17]", 5.0e-7, 6.0e-7, 0.1e-7, "| 5.0–6.0×10⁻⁷ | 1.4×10⁻⁵ |"),
                                 ("C21-BP [6/15]", 2.6e-8, 2.8e-8, 0.1e-8, "| 2.6–2.8×10⁻⁸ | 6.2–8.0×10⁻⁷ |"),
                                 ("C21-BP [8/13]", 4.5e-9, 5.4e-9, 0.1e-9, "| 4.5–5.4×10⁻⁹ | 1.3×10⁻⁷ |"),
                                 ("C21-BP [9/12]", 4.4e-10, 6.6e-10, 0.1e-10, "| 4.4–6.6×10⁻¹⁰ | 1.2–1.9×10⁻⁸ |"),
                                 ("C21-BP [10/11]", 3.5e-10, 8.7e-10, 0.1e-10, "| 3.5–8.7×10⁻¹⁰ | 0.84–2.5×10⁻⁸ |")):
        v = rt[name]["sigma22_rel"]
        out += [(f"{name} sigma_22 error, best node", min(v), lo, u, [(P, sub)]), (f"{name} sigma_22 error, worst node", max(v), hi, u, [(P, sub)])]
    RF21 = ("C21-BP [6/15]", "C21-BP [7/14]", "C21-BP [8/13]", "C21-BP [9/12]", "C21-BP [10/11]")
    FAM30 = RF21 + ("C20-BP [6/14]", "C20-BP [7/13]", "C20-BP [8/12]", "C20-BP [9/11]", "C20-BP [10/10]", "C19-BP [6/13]", "C19-BP [7/12]",
                    "C19-BP [8/11]", "C19-BP [9/10]", "C18-BP [6/12]", "C18-BP [7/11]", "C18-BP [8/10]", "C18-BP [9/9]", "C17-BP [6/11]",
                    "C17-BP [7/10]", "C17-BP [8/9]", "C16-BP [6/10]", "C16-BP [7/9]", "C16-BP [8/8]", "C15-BP [6/9]", "C15-BP [7/8]",
                    "C14-BP [5/9]", "C14-BP [6/8]", "C13-BP [5/8]", "C13-BP [6/7]")
    rings = {n: v for n, v in rt.items() if isinstance(v, dict) and v.get("ring") and n != "C21-BP [5/16]"}
    both = "All nine rings of the K = 21 generation are rejected under both readings of item 4"
    rfp = [(P, "predicted σ₂₂ to 3.5×10⁻¹⁰–2.8×10⁻⁸"), (R, "predicted it to 3.5×10⁻¹⁰–2.8×10⁻⁸")]
    out += [("K = 21 rings counted", len(rings), 9, 0.5, [(P, both), (R, "All nine rings built from σ₀…σ₂₁ are")]),
            ("K = 21 rings rejected under the primary reading", sum(1 for v in rings.values() if not v["survives"]), 9, 0.5, [(P, both)]),
            ("K = 21 rings surviving without D-bracket", sum(1 for v in rings.values() if v["survives_without_bracket"]), 0, 0.5, [(P, both)]),
            ("C21-BP [5/16] rejected under both readings", int(not rt["C21-BP [5/16]"]["survives"] and not rt["C21-BP [5/16]"]["survives_without_bracket"]), 1, 0.5, []),
            ("rings' smallest sigma_22 misprediction", min(min(v["sigma22_rel"]) for v in rings.values()), 5e-7, 1e-7, [(R, "they mispredict σ₂₂ by 5×10⁻⁷–10⁻²")]),
            ("ring-free C21-BP sums, best sigma_22 prediction", min(min(rt[n]["sigma22_rel"]) for n in RF21), 3.5e-10, 0.1e-10, rfp),
            ("ring-free C21-BP sums, worst sigma_22 prediction", max(max(rt[n]["sigma22_rel"]) for n in RF21), 2.8e-8, 0.1e-8, rfp),
            ("BP family of 30: rejected by D-bracket alone", sum(1 for n in FAM30 if rt[n]["extra_detectors"] == ["D-bracket"]), 4, 0.5,
             [(P, "rejects four ring-free sums of the family of thirty")]),
            ("BP family of 30: survivors", sum(1 for n in FAM30 if rt[n]["survives"]), 21, 0.5, [(P, "the other twenty-one survive")])]
    z = lambda fam, est: cal["22"][fam]["cells"][est]["zero"]  # noqa: E731
    err6 = lambda fam, est, k: cal["22"][fam]["cells"][est][k] * 1e6  # noqa: E731
    arc_arg = [(P, "+0.1…+1.4 → 0.9999988–1.0000002"), (R, "the arg crossing puts it at 0.9999988–1.0000002")]
    out += [("arc plants, arg, zero low (K = 22)", z("all 28 arc plants", "arg")[0], 0.9999988, 1e-7, arc_arg),
            ("arc plants, arg, zero high (K = 22)", z("all 28 arc plants", "arg")[1], 1.0000002, 1e-7, arc_arg),
            ("arc plants, arg error low (K = 22)", err6("all 28 arc plants", "arg", "lo"), 0.1, 0.1, arc_arg[:1]),
            ("arc plants, arg error high (K = 22)", err6("all 28 arc plants", "arg", "hi"), 1.4, 0.1, arc_arg[:1]),
            ("BP family (30), arg, zero low", z("BP family (30)", "arg")[0], 0.9999995, 1e-7, [(P, "−0.1…+0.8 → 0.9999995–1.0000004")]),
            ("BP family (30), arg, zero high", z("BP family (30)", "arg")[1], 1.0000004, 1e-7, [(P, "−0.1…+0.8 → 0.9999995–1.0000004")]),
            ("arc plants, ratio, zero low (K = 22)", z("all 28 arc plants", "ratio")[0], 0.9999657, 1e-7, [(P, "−12.4…+30.1 → 0.9999657–1.0000081")]),
            ("arc plants, ratio, zero high (K = 22)", z("all 28 arc plants", "ratio")[1], 1.0000081, 1e-7, [(P, "−12.4…+30.1 → 0.9999657–1.0000081")]),
            ("arc plants, Pade, zero high (K = 22)", z("all 28 arc plants", "pade")[1], 1.0000010, 1e-7, [(P, "−1.1…+0.6 → 0.9999993–1.0000010")]),
            ("BP family (30), Pade, zero high", z("BP family (30)", "pade")[1], 1.0000050, 1e-7, [(P, "−5.1…+6.6 → 0.9999933–1.0000050")])]
    dv = cal["sigma_dev_summary"]
    arc = [dv[f] for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2")]
    dev = [(P, "by 3.0–5.6% and 4.0–7.9% (worst node)")]
    out += [("arc plants' sigma_21 deviation, least (14.6)", min(x["sigma21"][0] for x in arc) * 100, 3.0, 0.1, dev),
            ("arc plants' sigma_21 deviation, most (14.6)", max(x["sigma21"][1] for x in arc) * 100, 5.6, 0.1, dev),
            ("arc plants' sigma_22 deviation, least", min(x["sigma22"][0] for x in arc) * 100, 4.0, 0.1, dev),
            ("arc plants' sigma_22 deviation, most", max(x["sigma22"][1] for x in arc) * 100, 7.9, 0.1, dev),
            ("C-EXP sigma_22 deviation", dv["C-EXP"]["sigma22"][1] * 100, 3.5, 0.1, [(P, "C-EXP's by 2.1% and")]),
            ("REFIT sigma_22 deviation, most", dv["REFIT"]["sigma22"][1] * 100, 33, 1, [(P, "the REFIT plants' by 1.7–27% and 2.0–33%")])]
    for name, zq, u, pole, nz in (("C22-BP [3/19]", 1.000000007, 1e-9, 1.74, 9), ("C22-BP [4/18]", 0.999999986, 1e-9, 1.80, 7),
                                  ("C22-BP [5/17]", 1.000000023, 1e-9, 1.84, 4), ("C22-BP [6/16]", 1.0000000044, 1e-10, 1.79, 1),
                                  ("C22-BP [7/15]", 1.0000000046, 1e-10, 1.86, 1), ("C22-BP [8/14]", 1.0000000019, 1e-10, 1.82, 1),
                                  ("C22-BP [9/13]", 1.0000000029, 1e-10, 1.84, 1), ("C22-BP [10/12]", 1.0000000031, 1e-10, 1.84, 1),
                                  ("C22-BP [11/11]", 1.0000000031, 1e-10, 1.84, 1)):
        r = rc[name]
        digits = {1e-9: 9, 1e-10: 10}[u]
        out += [(f"{name} zero", r["truth"], zq, u, [(P, f"| {name} | {zq:.{digits}f} |")]),
                (f"{name} largest pole", max(r["max_pole"].values()), pole, 0.01, [(P, f"| {name} | {zq:.{digits}f} | 0 / {nz} /")]),
                (f"{name} zeros in |x| < 12", r["zeros_lt_12"], nz, 0.5, [(P, f"| {name} | {zq:.{digits}f} | 0 / {nz} /")]),
                (f"{name} type rule", int(bool(r["type_rule"])), 1, 0.5, [])]
    W = [rc[f"C22-W{i}"]["truth"] for i in (1, 2, 3)]
    A = [rc[f"C22-ARC {k}"] for k in ("arc", "chord", "mirror", "beta2")]
    out += [("C22-W zeros, low", min(W), 0.999949, 1e-6, [(P, "| 0.999949–0.999963 | 3–5 / 20 / – |")]),
            ("C22-W zeros, high", max(W), 0.999963, 1e-6, [(P, "| 0.999949–0.999963 | 3–5 / 20 / – |")]),
            ("C22-ARC zeros, low", min(x["truth"] for x in A), 0.999976, 1e-6, [(P, "| 0.999976–0.999991 | 0–3 / 17–20 / – |")]),
            ("C22-ARC zeros, high", max(x["truth"] for x in A), 0.999991, 1e-6, [(P, "| 0.999976–0.999991 | 0–3 / 17–20 / – |")]),
            ("C22-ARC best omega_t, low", min(float(x["best_wt"]) for x in A), 0.99847, 1e-5, [(P, "best ω_t 0.99847–1.00095")]),
            ("C22-ARC best omega_t, high", max(float(x["best_wt"]) for x in A), 1.00095, 1e-5, [(P, "best ω_t 0.99847–1.00095")]),
            ("C22-ARC L2 floor, low", min(x["L2_floor"] for x in A), 2.6e-3, 0.1e-3, [(P, "L2 floor 2.6–7.6×10⁻³")]),
            ("C22-ARC L2 floor, high", max(x["L2_floor"] for x in A), 7.6e-3, 0.1e-3, [(P, "L2 floor 2.6–7.6×10⁻³")])]
    h = pj["hyp"]
    out += [("projection for a zero at 1 (K = 22)", (h["1.0"]["pred22"] - 1) * 1e7, 2.27, 0.01, [(P, "predicts arg₂₂ − 1 = +2.27×10⁻⁷ for ω* = 1")]),
            ("projection for a zero at 0.999998", (h["0.999998"]["pred22"] - 1) * 1e7, 0.82, 0.01, [(P, "+0.82×10⁻⁷ for 0.999998")]),
            ("projection for a zero at 0.999975 (K = 22)", (h["0.999975"]["pred22"] - 1) * 1e7, 0.27, 0.01, [(P, "+0.27×10⁻⁷ for 0.999975")]),
            ("real arg_22 - 1", (pj["arg"]["22"] - 1) * 1e7, 2.30, 0.01, [(P, "the real value is +2.30×10⁻⁷")]),
            ("miss, zero at 1 (K = 22)", abs(h["1.0"]["miss"]) * 1e9, 2.5, 0.1, [(P, "2.5×10⁻⁹")]),
            ("miss, zero at 0.999998", abs(h["0.999998"]["miss"]) * 1e7, 1.47, 0.01, [(P, "1.47×10⁻⁷ from the second")]),
            ("miss, zero at 0.999975 (K = 22)", abs(h["0.999975"]["miss"]) * 1e7, 2.03, 0.01, [(P, "2.03×10⁻⁷ from the third")]),
            ("model error (K = 22)", pj["model_err"], 3.9e-6, 0.1e-6, [(P, "The model's own error is 3.9×10⁻⁶")]),
            ("back-test, least (K = 22)", min(pj["back_real_max"].values()), 0.25e-6, 0.01e-6, [(P, "0.25–1.1×10⁻⁶ depending on ω*")]),
            ("back-test, most (K = 22)", max(pj["back_real_max"].values()), 1.1e-6, 0.1e-6, [(P, "0.25–1.1×10⁻⁶ depending on ω*")]),
            ("plant misses, low (K = 22)", min(p["miss"] for p in pj["plants"]), -3.9e-6, 0.1e-6, [(P, "−3.9×10⁻⁶…+0.4×10⁻⁶")]),
            ("plant misses, high (K = 22)", max(p["miss"] for p in pj["plants"]), 0.4e-6, 0.1e-6, [(P, "−3.9×10⁻⁶…+0.4×10⁻⁶")]),
            ("three times the model error", 3 * pj["model_err"], 1.2e-5, 0.1e-5, [(P, "three times that, 1.2×10⁻⁵")]),
            ("ratio of the two misses", abs(h["0.999998"]["miss"]) / abs(h["1.0"]["miss"]), 58, 1, [(P, "ω* = 1 is 58 times")]),
            ("model error over the second miss", pj["model_err"] / abs(h["0.999998"]["miss"]), 26, 1, [(P, "is 26 times the second")]),
            ("trend per order, k = 4-22", pj["trend_4_22"]["per_order"], 0.543, 0.001, [(P, "falls ×0.543 per\norder") if False else (P, "×0.543 per")]),
            ("trend's arg_22 - 1", pj["trend_4_21"]["at22"] * 1e7, 2.29, 0.01, [(P, "put arg₂₂ − 1 at 2.29×10⁻⁷")]),
            ("trend's distance from arg_22", (pj["arg"]["22"] - 1 - pj["trend_4_21"]["at22"]) * 1e9, 0.7, 0.1, [(P, "0.7×10⁻⁹ from where it is")]),
            ("noise shift of arg_22", max(abs(x) for x in pj["noise_shift"]), 1.2e-9, 0.1e-9, [(P, "moves by at most 1.2×10⁻⁹")]),
            ("projection decisive (K = 22)", int(bool(pj["decisive"])), 0, 0.5, [])]
    rf = list(cb["ring_free"].values())
    r21 = json.load(open(os.path.join(d, "planted_checks_k21.json")))["K21 combine"]["range21"]
    r20 = json.load(open(os.path.join(d, "planted_checks_k20.json")))["K20 combine"]["range20"]
    rfs = [(P, "the ring-free completions, 1.0000000019–1.0000000046"), (R, "1.0000000019–1.0000000046. Calibrated on the planted"),
           (M, "put the zero at $1.0000000019$--$1.0000000046$")]
    out += [("K = 22 hull, low", cb["range22"][0], 0.9999988, 1e-7, [(P, "  0.9999988–1.0000004.**")]),
            ("K = 22 hull, high", cb["range22"][1], 1.0000004, 1e-7, [(P, "  0.9999988–1.0000004.**")]),
            ("K = 22 arc range, low", cb["arc"][0], 0.9999988, 1e-7, [(P, "over the arc plants, 0.9999988–1.0000002")]),
            ("K = 22 BP range, high", cb["bp"][1], 1.0000004, 1e-7, [(P, "thirty, 0.9999995–1.0000004")]),
            ("K = 21 hull at seven digits, low", r21[0], 0.9999984, 1e-7, [(P, "inside the K = 21 one (0.9999984–1.0000016)")]),
            ("K = 21 hull at seven digits, high", r21[1], 1.0000016, 1e-7, [(P, "inside the K = 21 one (0.9999984–1.0000016)")]),
            ("ring-free completions, low (K = 22)", min(rf), 1.0000000019, 1e-10, rfs),
            ("ring-free completions, high (K = 22)", max(rf), 1.0000000046, 1e-10, rfs),
            ("ring-free completions counted (K = 22)", len(rf), 6, 0.5, [(P, "C22-BP [6/16], [7/15], [8/14], [9/13], [10/12], [11/11],")]),
            ("headline, low (K = 22)", cb["headline"][0], 0.999975, 1e-6, [(P, "**0.999975–1.000050, unchanged.**")]),
            ("headline, high (K = 22)", cb["headline"][1], 1.000050, 1e-6, [(P, "**0.999975–1.000050, unchanged.**")]),
            ("headline overlap (K = 22)", int(bool(cb["overlap_k21"])), 1, 0.5, []),
            ("K = 20 to 22, low", min(r20[0], r21[0], cb["range22"][0]), 0.9999971, 1e-7, [(P, "would give 0.9999971–1.0000017")]),
            ("K = 20 to 22, high", max(r20[1], r21[1], cb["range22"][1]), 1.0000017, 1e-7, [(P, "would give 0.9999971–1.0000017")]),
            ("K = 22 hull width", (cb["range22"][1] - cb["range22"][0]) * 1e6, 1.6, 0.1, [(P, "a width of 1.6×10⁻⁶ against")]),
            ("K = 21 hull width (14.10)", (r21[1] - r21[0]) * 1e6, 3.2, 0.1, [(P, "3.2×10⁻⁶ at K = 21")]),
            ("G production at K = 22", cb["G_minima"]["22"]["production"], 0.99999999, 1e-8, [(P, "production 0.99999999 at K = 22")]),
            ("G production at K = 21 (14.9)", cb["G_minima"]["21"]["production"], 1.00000006, 1e-8, [(P, "(1.00000006 at K = 21)")]),
            ("G minima span at K = 22, low", cb["G_minima"]["22"]["span"][0], 0.99999990, 1e-8, [(P, "0.99999990–1.00000000 (0.99999997–1.00000007 at K = 21)")]),
            ("G minima span at K = 22, high", cb["G_minima"]["22"]["span"][1], 1.00000000, 1e-8, [(P, "0.99999990–1.00000000 (0.99999997–1.00000007 at K = 21)")]),
            ("G minima span at K = 21, low (14.9)", cb["G_minima"]["21"]["span"][0], 0.99999997, 1e-8, [(P, "0.99999990–1.00000000 (0.99999997–1.00000007 at K = 21)")]),
            ("G minima span at K = 21, high (14.9)", cb["G_minima"]["21"]["span"][1], 1.00000007, 1e-8, [(P, "0.99999990–1.00000000 (0.99999997–1.00000007 at K = 21)")]),
            ("G global minimum at K = 22", cb["G_minima"]["22"]["global"], 0.99999990, 1e-8, [(P, "its global minimum 0.99999990")])]
    nodes = cv["nodes"]
    conv = [(P, "σ₂₂ is converged to 1.4–1.7×10⁻¹² relative"), (R, "1.4–1.7×10⁻¹² relative at the three ω")]
    out += [("relative err22 at wf 0.99 (verdict)", nodes["0.99"]["err22_rel"], 1.4e-12, 0.1e-12, conv),
            ("relative err22 at wf 1.01 (verdict)", nodes["1.01"]["err22_rel"], 1.7e-12, 0.1e-12, conv),
            ("margins at K = 22, least", min(1e-10 / nodes[w]["err22_rel"] for w in nodes), 59, 1, [(P, "with margins of 59–73 under item 1")]),
            ("margins at K = 22, most", max(1e-10 / nodes[w]["err22_rel"] for w in nodes), 73, 1, [(P, "with margins of 59–73 under item 1")])]
    return out


@check
def p5_sigma22():
    """proof/P5_scope.md item 3: sigma_22 at the nodes against the type bound (solver/proof/logs/entire_cert_K22.log,
    from `entire_cert.py 22 --scan ../inner_amp/scan45J1250`), |sigma_22| and its fit errors from sigma_err_fit.py on the
    K 45 files (a1 from the 960-bit scan39J1000 files; notes/PLANTED_SELECTION.md section 14.0's stage-1 reading), the
    k <= 12 calibration's ratio at k = 22, the K = 22 recalibration and its prediction for sigma_23."""
    import re
    P, Q, N = "proof/P5_scope.md", "notes/PROOF_QUEUE.md", "notes/PLANTED_SELECTION.md"
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import sigma_err_fit as SF
    sys.path.pop(0)
    L = open(os.path.join(ROOT, "solver/proof/logs/entire_cert_K22.log")).read()
    node = {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"wf (\S+): k <= 21 \(.*?\): ([0-9.]+); k <= 12 \(.*?\): ([0-9.]+)", L)}
    cal = re.search(r"\|sigma_k\| Gamma\(k \+ ([0-9.]+)\) / ([0-9.]+)\^k for k = 4\.\.22:.*C = max = ([0-9.]+)", L)
    pred = re.search(r"k 23: ([0-9.e+-]+),", L)
    rel = {}
    for w in ("0.9900", "1.0000", "1.0100"):
        a = json.load(open(os.path.join(d, f"scan45J1250/sigF_w{w}_K45_J1250_p1280.json")))
        p = json.load(open(os.path.join(d, f"scan39J1000/sigF_w{w}_K39_J1000_p960.json")))
        r = SF.recompute(a, SF.a1_of(p))
        rel[w] = (float(abs(r[22][1])), r[22][2])
    sa, ea = rel["1.0000"]
    k12 = "1.050, 1.027, 1.003 at wf 0.99, 1.00, 1.01"
    cs = "(a = 1.7609, b = 4.81, C = 86.8)"
    both = "0.898 and 0.894 of each node's own K = 21 bound"
    return [("|sigma_22| at wf 1.00", sa, 9.08e-20, 0.01e-20, [(P, "|σ₂₂| = 9.08×10⁻²⁰")]),
            ("err22 of (a)", ea, 1.297e-31, 0.001e-31, [(N, "err22 = 1.297×10⁻³¹ = 1.43×10⁻¹²")]),
            ("err22 of (a), relative", ea / sa, 1.43e-12, 0.01e-12,
             [(N, "err22 = 1.297×10⁻³¹ = 1.43×10⁻¹²"), (P, "1.43×10⁻¹² of |σ₂₂|")]),
            ("err22 margin under 1e-10", 1e-10 / (ea / sa), 70, 1, [(N, "a margin of 70")]),
            ("err22 at wf 0.99, relative", rel["0.9900"][1] / rel["0.9900"][0], 1.28e-12, 0.01e-12, [(N, "1.28×10⁻¹² (wf 0.99)")]),
            ("err22 at wf 1.01, relative", rel["1.0100"][1] / rel["1.0100"][0], 1.59e-12, 0.01e-12, [(N, "1.59×10⁻¹² (wf 1.01)")]),
            ("sigma_22 over the K = 21 bound at wf 1.00", node["1.0"][0], 0.896, 0.001, [(P, "0.896 of the K = 21 bound")]),
            ("sigma_22 over the K = 21 bound at wf 0.99", node["0.99"][0], 0.898, 0.001, [(P, both)]),
            ("sigma_22 over the K = 21 bound at wf 1.01", node["1.01"][0], 0.894, 0.001, [(P, both)]),
            ("k <= 12 calibration's ratio at k = 22, wf 0.99", node["0.99"][1], 1.050, 0.001,
             [(P, k12), (Q, "1.050, 1.027, 1.003: above 1 at all three nodes")]),
            ("k <= 12 calibration's ratio at k = 22, wf 1.00", node["1.0"][1], 1.027, 0.001, [(P, k12)]),
            ("k <= 12 calibration's ratio at k = 22, wf 1.01", node["1.01"][1], 1.003, 0.001, [(P, k12)]),
            ("K = 22 calibration a", float(cal.group(2)), 1.7609, 0.0001, [(P, cs)]),
            ("K = 22 calibration b", float(cal.group(1)), 4.81, 0.01, [(P, cs)]),
            ("K = 22 calibration C", float(cal.group(3)), 86.8, 0.1, [(P, cs)]),
            ("predicted bound on sigma_23", float(pred.group(1)), 6.74e-21, 0.01e-21, [(P, "|σ₂₃| ≤ 6.74×10⁻²¹")])]

@check
def p5_sigma21():
    """proof/P5_scope.md item 3: sigma_21 at the nodes against the type bound (solver/proof/logs/entire_cert_K21.log,
    from `entire_cert.py 21 --scan ../inner_amp/scan43J1250`), the k <= 12 calibration's ratio at k = 21, the K = 21
    recalibration and its prediction for sigma_22."""
    import re
    P, Q = "proof/P5_scope.md", "notes/PROOF_QUEUE.md"
    L = open(os.path.join(ROOT, "solver/proof/logs/entire_cert_K21.log")).read()
    node = {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"wf (\S+): k <= 20 \(.*?\): ([0-9.]+); k <= 12 \(.*?\): ([0-9.]+)", L)}
    cal = re.search(r"\|sigma_k\| Gamma\(k \+ ([0-9.]+)\) / ([0-9.]+)\^k for k = 4\.\.21:.*C = max = ([0-9.]+)", L)
    pred = re.search(r"k 22: ([0-9.e+-]+),", L)
    k12 = "1.016, 0.996, 0.975 at wf 0.99, 1.00, 1.01"
    cs = "(a = 1.7497, b = 4.72, C = 73.1)"
    return [("sigma_21 over the K = 20 bound at wf 0.99", node["0.99"][0], 0.901, 0.001, [(P, "0.901 and 0.896 of each node's own K = 20 bound")]),
            ("sigma_21 over the K = 20 bound at wf 1.01", node["1.01"][0], 0.896, 0.001, [(P, "0.901 and 0.896 of each node's own K = 20 bound")]),
            ("k <= 12 calibration's ratio at k = 21, wf 0.99", node["0.99"][1], 1.016, 0.001, [(P, k12), (Q, "1.016 at the 0.99 node at k = 21")]),
            ("k <= 12 calibration's ratio at k = 21, wf 1.00", node["1.0"][1], 0.996, 0.001, [(P, k12)]),
            ("k <= 12 calibration's ratio at k = 21, wf 1.01", node["1.01"][1], 0.975, 0.001, [(P, k12)]),
            ("K = 21 calibration a", float(cal.group(2)), 1.7497, 0.0001, [(P, cs)]),
            ("K = 21 calibration b", float(cal.group(1)), 4.72, 0.01, [(P, cs)]),
            ("K = 21 calibration C", float(cal.group(3)), 73.1, 0.1, [(P, cs)]),
            ("predicted bound on sigma_22", float(pred.group(1)), 1.01e-19, 0.01e-19, [(P, "|σ₂₂| ≤ 1.01×10⁻¹⁹")])]

@check
def p5_sigma20():
    """proof/P5_scope.md item 3: sigma_20 against the type bound (solver/proof/logs/entire_cert_K20.log, from
    `entire_cert.py 20 --scan ../inner_amp/scan41J1250`), and its fit error from sigma_err_fit.py (section 12.0's gate)."""
    import re
    P = "proof/P5_scope.md"
    L = open(os.path.join(ROOT, "solver/proof/logs/entire_cert_K20.log")).read()
    node = {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"wf (\S+): k <= 19 \(.*?\): ([0-9.]+); k <= 12 \(.*?\): ([0-9.]+)", L)}
    cal = re.search(r"\|sigma_k\| Gamma\(k \+ ([0-9.]+)\) / ([0-9.]+)\^k for k = 4\.\.20:.*C = max = ([0-9.]+)", L)
    pred = re.search(r"k 21: ([0-9.e+-]+),", L)
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import sigma_err_fit as SF
    sys.path.pop(0)
    a = json.load(open(os.path.join(d, "scan41J1250/sigF_w1.0000_K41_J1250_p1280.json")))
    p = json.load(open(os.path.join(d, "scan39J1000/sigF_w1.0000_K39_J1000_p960.json")))
    r = SF.recompute(a, SF.a1_of(p))
    s20, e20 = float(abs(r[20][1])), r[20][2]
    k12 = "0.988, 0.971, 0.953 at wf 0.99, 1.00, 1.01"
    cs = "(a = 1.7386, b = 4.63, C = 62.0)"
    return [("sigma_20 over the K = 19 bound at wf 1.00", node["1.0"][0], 0.901, 0.001, [(P, "0.901 of the K = 19 bound")]),
            ("sigma_20 over the K = 19 bound at wf 0.99", node["0.99"][0], 0.905, 0.001, [(P, "0.905 and 0.896 of each node's own K = 19 bound")]),
            ("sigma_20 over the K = 19 bound at wf 1.01", node["1.01"][0], 0.896, 0.001, [(P, "0.905 and 0.896 of each node's own K = 19 bound")]),
            ("k <= 12 calibration's ratio at k = 20, wf 0.99", node["0.99"][1], 0.988, 0.001, [(P, k12)]),
            ("k <= 12 calibration's ratio at k = 20, wf 1.00", node["1.0"][1], 0.971, 0.001, [(P, k12)]),
            ("k <= 12 calibration's ratio at k = 20, wf 1.01", node["1.01"][1], 0.953, 0.001, [(P, k12)]),
            ("K = 20 calibration a", float(cal.group(2)), 1.7386, 0.0001, [(P, cs)]),
            ("K = 20 calibration b", float(cal.group(1)), 4.63, 0.01, [(P, cs)]),
            ("K = 20 calibration C", float(cal.group(3)), 62.0, 0.1, [(P, cs)]),
            ("predicted bound on sigma_21", float(pred.group(1)), 1.47e-18, 0.01e-18, [(P, "|σ₂₁| ≤ 1.47×10⁻¹⁸")]),
            ("|sigma_20| at wf 1.00", s20, 1.86e-17, 0.01e-17, [(P, "|σ₂₀| = 1.86×10⁻¹⁷ at ω∞")]),
            ("sigma_20 fit error at wf 1.00, relative", e20 / s20, 7.2e-13, 0.1e-13, [(P, "7.2×10⁻¹³ of |σ₂₀|")])]


@check
def k19_converge_nodes():
    """notes/PLANTED_SELECTION.md sections 11.1 and 11.2: the converge stage under the second amendment ("K19 converge" in
    solver/inner_amp/planted_checks_k19.json), the 960-bit plateau errors of the (a') run, the sigma_0..18 relative
    differences (scan41J1250 against scan39J1000), the node runs' minutes, and the plants stage's checks and records."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import planted_k14 as K14
    sys.path.pop(0)
    import mpmath as mp
    P = "notes/PLANTED_SELECTION.md"
    c = json.load(open(os.path.join(d, "planted_checks_k19.json")))
    n, a = c["K19 converge"], c["K19 converge under (a')"]
    out = [("J 1500 pilot difference at sigma_19", n["pilot_j1500_d19"], 1.57e-30, 0.01e-30, [(P, "err19 = plateau + 1.57×10⁻³⁰")])]
    for w, e, r, m, pl, p9, q in (("0.99", 1.16e-28, 4.49e-13, 223, 1.14e-28, 2.07e-26, 181), ("1.0", 1.26e-28, 5.01e-13, 199, 1.25e-28, 2.79e-27, 22),
                                  ("1.01", 1.38e-28, 5.60e-13, 179, 1.36e-28, 1.20e-27, 9)):
        x, y = n["nodes"][w], a["nodes"][w]
        cell = f"{e / 1e-28:.2f}×10⁻²⁸ ({r / 1e-13:.2f}×10⁻¹³, {m})"
        out += [(f"err19 at wf {w} under (a'')", x["err19"], e, 0.01e-28, [(P, cell)]),
                (f"relative err19 at wf {w} under (a'')", x["err19_rel"], r, 0.01e-13, [(P, cell)]),
                (f"margin at wf {w} under (a'')", 1e-10 / x["err19_rel"], m, 1, [(P, cell)]),
                (f"1280-bit plateau error at wf {w}", x["err19_plateau"], pl, 0.01e-28, [(P, "1.14, 1.25 and 1.36×10⁻²⁸")]),
                (f"960-bit plateau error at wf {w}", y["err19_plateau"], p9, 0.01 * p9 / float(f"{p9:.2e}".split("e")[0]) if False else 0.01e-26 if w == "0.99" else 0.01e-27,
                 [(P, "2.07×10⁻²⁶, 2.79×10⁻²⁷ and 1.20×10⁻²⁷")]),
                (f"960 over 1280 plateau error at wf {w}", y["err19_plateau"] / x["err19_plateau"], q, 1, [(P, "(181, 22 and 9 times larger)")])]
    for w, v in (("0.99", 0.0079), ("1.0", 0.0078), ("1.01", 0.0078)):
        out.append((f"ratio-law second difference at k = 17, wf {w}", n["ratiolaw_d2"][w][-1], v, 0.0001, [(P, f"| {v:.4f} |")]))
    sp = n["spread"]
    out += [("spread of sigma_19 in wf", sp["19"]["half_range_rel"] * 100, 6.2, 0.1, [(P, "σ₁₉ changes by 6.2% of itself")]),
            ("spread of sigma_17 in wf", sp["17"]["half_range_rel"] * 100, 6.0, 0.1, [(P, "(6.0%, 6.1% for σ₁₇, σ₁₈)")]),
            ("spread of sigma_18 in wf", sp["18"]["half_range_rel"] * 100, 6.1, 0.1, [(P, "(6.0%, 6.1% for σ₁₇, σ₁₈)")]),
            ("err19 over the wf change", sp["19"]["err_over_d1"], 8.78e-12, 0.01e-12, [(P, "8.78×10⁻¹² of that change")]),
            ("err17 over the wf change", sp["17"]["err_over_d1"], 3.65e-12, 0.01e-12, [(P, "(3.65×10⁻¹² at σ₁₇, 5.96×10⁻¹² at σ₁₈)")]),
            ("err18 over the wf change", sp["18"]["err_over_d1"], 5.96e-12, 0.01e-12, [(P, "(3.65×10⁻¹² at σ₁₇, 5.96×10⁻¹² at σ₁₈)")])]
    mp.mp.dps = 50
    p960, nd = K14.raw_scan(os.path.join(d, "scan39J1000"), "*_p960.json"), K14.raw_scan(os.path.join(d, "scan41J1250"), "*.json")
    mx = max(float(abs(K14.sigmas(nd[w])[k] - K14.sigmas(p960[w])[k]) / abs(K14.sigmas(p960[w])[k])) for w in p960 for k in range(1, 19))
    out.append(("largest relative sigma_0..18 difference, 1280 against 960 bits", mx, 1.7e-13, 0.1e-13, [(P, "at most 1.7×10⁻¹³ of")]))
    for w in ("0.9900", "1.0100"):
        f = json.load(open(os.path.join(d, f"scan41J1250/sigF_w{w}_K41_J1250_p1280.json")))
        out.append((f"node run minutes at wf {w}", f["seconds"] / 60, 1460, 1, [(P, "took 1460 min each")]))
    pc = c["K19 plants check"]
    recs = [json.loads(l) for l in open(os.path.join(d, "planted_results_k19.jsonl")) if l.strip()]
    pl = [r for r in recs if not r["name"].startswith(("N0", "C13", "C14", "C15", "C16", "C17", "C18", "C19"))]
    out += [("plants at K = 19", sum(pc["families"].values()), 51, 0.5, [(P, "51 plants at K = 17, 18, 19 (11.8 min in all)")]),
            ("plants' minutes", sum(r["seconds"] for r in pl) / 60, 11.8, 0.1, [(P, "51 plants at K = 17, 18, 19 (11.8 min in all)")]),
            ("N1 worst error at K = 19", pc["n1_worst"], 3.5e-7, 0.1e-7, [(P, "Padé and G to 3.5×10⁻⁷ at worst")]),
            ("plants reproduce the K = 18 records", pc["repro_worst"][0], 0.0, 1e-12, [(P, "equal the stored K = 18 records exactly")])]
    return out


@check
def k19_real():
    """notes/PLANTED_SELECTION.md section 11.3: the real estimates at K = 17, 18, 19 ("K19 real" in
    solver/inner_amp/planted_checks_k19.json and the record "N0 scan41J1250" in planted_results_k19.jsonl) and item 3's
    noise comparison (the third amendment)."""
    d = os.path.join(ROOT, "solver/inner_amp")
    P = "notes/PLANTED_SELECTION.md"
    c = json.load(open(os.path.join(d, "planted_checks_k19.json")))["K19 real"]
    rec = {r["name"]: r for r in (json.loads(l) for l in open(os.path.join(d, "planted_results_k19.jsonl")) if l.strip())}["N0 scan41J1250"]
    e = c["est"]
    out = []
    for K, vals in (("19", {"arg": (1.0000015, 1e-7), "ratio": (0.999986, 1e-6), "pade": (1.0000029, 1e-7), "G": (0.9999995, 1e-7), "aitken": (1.0000009, 1e-7)}),
                    ("18", {"arg": (1.0000024, 1e-7), "ratio": (0.999966, 1e-6), "pade": (0.9999995, 1e-7), "G": (0.9999987, 1e-7), "aitken": (1.0000011, 1e-7)}),
                    ("17", {"arg": (1.0000046, 1e-7), "ratio": (0.999912, 1e-6), "pade": (0.999990, 1e-6), "G": (1.0000026, 1e-7), "aitken": (0.9999925, 1e-7)})):
        row = {"arg": "| arg crossing | 1.0000015 | 1.0000024 | 1.0000046 |", "ratio": "| ratio | 0.999986 | 0.999966 | 0.999912 |",
               "pade": "(mid 1.0000029) | [3/14]–[13/4]: 0.999997–1.000002 (mid 0.9999995) | [3/13]–[12/4]: 0.999978–1.000002 (mid 0.999990) |",
               "G": "| G (production κ̄) | 0.9999995 | 0.9999987 | 1.0000026 |", "aitken": "| Aitken, k − 2 … k | 1.0000009 | 1.0000011 | 0.9999925 |"}
        for est, (q, u) in vals.items():
            out.append((f"{est} at K = {K}", e[K][est], q, u, [(P, row[est])]))
    pr = rec["est"]["19"]["pade_central_range"]
    out += [("central Pade range at K = 19, low", pr[0], 0.9999996, 1e-7, [(P, "[3/15]–[14/4]: 0.9999996–1.0000061")]),
            ("central Pade range at K = 19, high", pr[1], 1.0000061, 1e-7, [(P, "[3/15]–[14/4]: 0.9999996–1.0000061")])]
    a = {int(k): v for k, v in rec["est"]["19"]["arg"].items()}
    out += [("arg_16 - 1 over arg_15 - 1", 0, 0, 1, [])] if False else []
    out += [("arg crossing at k = 16", a[16], 1.0000104, 1e-7, [(P, "1.0000104, 1.0000046, 1.0000024, 1.0000015 for k = 16–19")]),
            ("shrink of arg_k - 1, 16 to 17", (a[17] - 1) / (a[16] - 1), 0.44, 0.01, [(P, "×0.44 from k = 16 to 17")]),
            ("shrink of arg_k - 1, 17 to 18", (a[18] - 1) / (a[17] - 1), 0.53, 0.01, [(P, "×0.53 from 17 to 18")]),
            ("shrink of arg_k - 1, 18 to 19", (a[19] - 1) / (a[18] - 1), 0.63, 0.01, [(P, "×0.63 from 18 to 19")]),
            ("last two differences of arg_k", abs(a[18] - a[17]) * 1e6, 2.2, 0.1, [(P, "are 2.2×10⁻⁶, then 0.9×10⁻⁶")]),
            ("last difference of arg_k", (a[18] - a[19]) * 1e6, 0.9, 0.1, [(P, "are 2.2×10⁻⁶, then 0.9×10⁻⁶")])]
    nc = c["noise_comparison"]
    out += [("largest noise ratio, K = 17", nc["17"]["worst_ratio"][0], 0.61, 0.01, [(P, "0.61 at most over all (arg₁₁)")]),
            ("largest noise ratio, K = 18", nc["18"]["worst_ratio"][0], 0.61, 0.01, [(P, "0.61 at most over all (arg₁₁)")]),
            ("largest difference at K = 17", nc["17"]["largest_difference"][1], 9.3e-9, 0.1e-9, [(P, "at most 9.3×10⁻⁹ (K = 17,")]),
            ("largest difference at K = 18", nc["18"]["largest_difference"][1], 3.3e-8, 0.1e-8, [(P, "`pade10/6`) and 3.3×10⁻⁸ (K = 18")]),
            ("largest ratio-estimate noise ratio", max(v["ratio"] for K in ("17", "18") for k, v in nc[K]["rows"].items() if k.startswith("ratio")), 0.58, 0.01, [(P, "0.58 at most for")]),
            ("largest Pade noise ratio", max(v["ratio"] for K in ("17", "18") for k, v in nc[K]["rows"].items() if k.startswith("pade")), 0.07, 0.01, [(P, "0.01–0.07 of the noise rms for every Padé crossing")]),
            ("G noise ratio at K = 17", nc["17"]["rows"]["G"]["ratio"], 0.03, 0.01, [(P, "0.03 and 0.05 for G")]),
            ("G noise ratio at K = 18", nc["18"]["rows"]["G"]["ratio"], 0.05, 0.01, [(P, "0.03 and 0.05 for G")])]
    return out


@check
def k19_stages():
    """notes/PLANTED_SELECTION.md sections 11.4-11.9: the ring test, calibration, next generation, projection and
    combination at K = 19 ("K19 ring test", "K19 calib", "K19 recomplete", "K19 project", "K19 combine" in
    solver/inner_amp/planted_checks_k19.json) and the verdict's numbers."""
    d = os.path.join(ROOT, "solver/inner_amp")
    P = "notes/PLANTED_SELECTION.md"
    c = json.load(open(os.path.join(d, "planted_checks_k19.json")))
    rt, cal, rc, pj, cb = c["K19 ring test"], c["K19 calib"], c["K19 recomplete"], c["K19 project"], c["K19 combine"]
    out = []
    # 11.4 the ring test
    sz = rt["simple-zero law from tau_17, tau_18"]["tau19_rel"]
    out += [("simple-zero law, best node", min(sz), 1.6e-5, 0.1e-5, [(P, "misses τ₁₉ by 1.6–2.9×10⁻⁵")]),
            ("simple-zero law, worst node", max(sz), 2.9e-5, 0.1e-5, [(P, "misses τ₁₉ by 1.6–2.9×10⁻⁵")])]
    for name, hi, u, sub in (("C18-BP [3/15]", 3.3e-6, 0.1e-6, "by 3.3×10⁻⁶, 5.5×10⁻⁷ and 3.2×10⁻⁷ at worst"),
                             ("C18-BP [4/14]", 5.5e-7, 0.1e-7, "by 3.3×10⁻⁶, 5.5×10⁻⁷ and 3.2×10⁻⁷ at worst"),
                             ("C18-BP [5/13]", 3.2e-7, 0.1e-7, "by 3.3×10⁻⁶, 5.5×10⁻⁷ and 3.2×10⁻⁷ at worst"),
                             ("C17-BP [6/11]", 6.1e-7, 0.1e-7, "| C17-BP [6/11] | 1.0000004 | 0 / 1 / 10 | 6.0–6.1×10⁻⁷ |")):
        out.append((f"{name} sigma_19 error, worst node", max(rt[name]["sigma19_rel"]), hi, u, [(P, sub)]))
        out.append((f"{name} rejected", int(not rt[name]["survives"]), 1, 0.5, []))
    free = [rt[n]["sigma19_rel"] for n in ("C18-BP [6/12]", "C18-BP [7/11]", "C18-BP [8/10]", "C18-BP [9/9]")]
    out += [("ring-free C18-BP sums, best sigma_19 prediction", min(min(f) for f in free), 4e-9, 1e-9, [(P, "predicted σ₁₉ to 4×10⁻⁹–6×10⁻⁸"), ("paper/math/main.tex", "predicted $\\sigma_{19}$ to $4\\times10^{-9}$--$6\\times10^{-8}$")]),
            ("ring-free C18-BP sums, worst sigma_19 prediction", max(max(f) for f in free), 6e-8, 1e-8, [(P, "predicted σ₁₉ to 4×10⁻⁹–6×10⁻⁸"), ("paper/math/main.tex", "predicted $\\sigma_{19}$ to $4\\times10^{-9}$--$6\\times10^{-8}$")]),
            ("ring-free C18-BP sums surviving", sum(rt[n]["survives"] for n in ("C18-BP [6/12]", "C18-BP [7/11]", "C18-BP [8/10]", "C18-BP [9/9]")), 4, 0.5, []),
            ("K = 18 rings rejected", sum(1 for n, v in rt.items() if isinstance(v, dict) and v.get("ring") and not v["survives"]), 10, 0.5,
             [(P, "All ten rings of the K = 18 generation are rejected by σ₁₉ under both readings")]),
            ("K = 18 rings surviving without D-bracket", sum(1 for n, v in rt.items() if isinstance(v, dict) and v.get("ring") and v["survives_without_bracket"]), 0, 0.5, [])]
    # 11.5 calibration
    z = lambda fam, e: cal["19"][fam]["cells"][e]["zero"]
    out += [("arc plants, arg, zero low", z("all 28 arc plants", "arg")[0], 0.999996, 1e-6, [(P, "−0.03…+0.05 → 0.999996–1.000004")]),
            ("arc plants, arg, zero high", z("all 28 arc plants", "arg")[1], 1.000004, 1e-6, [(P, "−0.03…+0.05 → 0.999996–1.000004")]),
            ("BP family, arg, zero low", z("BP family (16)", "arg")[0], 0.999996, 1e-6, [(P, "+0.01…+0.06 → 0.999996–1.000000")]),
            ("BP family, arg, zero high", z("BP family (16)", "arg")[1], 1.000000, 1e-6, [(P, "+0.01…+0.06 → 0.999996–1.000000")]),
            ("arc plants, ratio, zero low", z("all 28 arc plants", "ratio")[0], 1.000006, 1e-6, [(P, "−0.87…−0.20 → 1.000006–1.000072")]),
            ("arc plants, ratio, zero high", z("all 28 arc plants", "ratio")[1], 1.000072, 1e-6, [(P, "−0.87…−0.20 → 1.000006–1.000072")]),
            ("arc plants, Pade, zero low", z("all 28 arc plants", "pade")[0], 0.999995, 1e-6, [(P, "−0.01…+0.08 → 0.999995–1.000004")]),
            ("arc plants, Pade, zero high", z("all 28 arc plants", "pade")[1], 1.000004, 1e-6, [(P, "−0.01…+0.08 → 0.999995–1.000004")]),
            ("BP family, Pade, zero low", z("BP family (16)", "pade")[0], 0.999983, 1e-6, [(P, "−0.35…+0.20 → 0.999983–1.000038")]),
            ("BP family, Pade, zero high", z("BP family (16)", "pade")[1], 1.000038, 1e-6, [(P, "−0.35…+0.20 → 0.999983–1.000038")]),
            ("BP family, G, zero high", z("BP family (16)", "G")[1], 1.000003, 1e-6, [(P, "−0.03…+0.01 → 0.999999–1.000003")]),
            ("arc plants, arg error low", cal["19"]["all 28 arc plants"]["cells"]["arg"]["lo"] * 1e4, -0.03, 0.01, [(P, "−0.03…+0.05 → 0.999996–1.000004")]),
            ("arc plants, arg error high", cal["19"]["all 28 arc plants"]["cells"]["arg"]["hi"] * 1e4, 0.05, 0.01, [(P, "−0.03…+0.05 → 0.999996–1.000004")]),
            ("K = 18 arc calibration reproduced, low", cal["18"]["all 28 arc plants"]["cells"]["arg"]["zero"][0], 0.999992, 1e-6, [(P, "section 10.5 (arc 0.999992–1.000002, BP family of twelve")]),
            ("K = 18 arc calibration reproduced, high", cal["18"]["all 28 arc plants"]["cells"]["arg"]["zero"][1], 1.000002, 1e-6, [(P, "section 10.5 (arc 0.999992–1.000002, BP family of twelve")])]
    dv = cal["sigma_dev_summary"]
    arc = [dv[f] for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2")]
    out += [("arc plants' sigma_18 deviation, least", min(x["sigma18"][0] for x in arc) * 100, 0.5, 0.1, [(P, "by 0.5–1.8% and 1.4–2.7% (worst node)")]),
            ("arc plants' sigma_18 deviation, most", max(x["sigma18"][1] for x in arc) * 100, 1.8, 0.1, [(P, "by 0.5–1.8% and 1.4–2.7% (worst node)")]),
            ("arc plants' sigma_19 deviation, least", min(x["sigma19"][0] for x in arc) * 100, 1.4, 0.1, [(P, "by 0.5–1.8% and 1.4–2.7% (worst node)")]),
            ("arc plants' sigma_19 deviation, most", max(x["sigma19"][1] for x in arc) * 100, 2.7, 0.1, [(P, "by 0.5–1.8% and 1.4–2.7% (worst node)")])]
    # 11.6 the next generation
    for name, zq, u, pole, nz in (("C19-BP [3/16]", 0.9999994, 1e-7, 1.70, 9), ("C19-BP [4/15]", 1.00000003, 1e-8, 1.74, 5), ("C19-BP [5/14]", 1.00000007, 1e-8, 1.72, 4),
                                  ("C19-BP [6/13]", 1.00000009, 1e-8, 1.75, 1), ("C19-BP [7/12]", 0.99999998, 1e-8, 1.78, 1), ("C19-BP [8/11]", 0.99999999, 1e-8, 1.78, 1),
                                  ("C19-BP [9/10]", 0.99999999, 1e-8, 1.79, 1)):
        r = rc[name]
        out += [(f"{name} zero", r["truth"], zq, u, [(P, f"| {name} | {zq:.{8 if u == 1e-8 else 7}f} |")]),
                (f"{name} largest pole", max(r["max_pole"].values()), pole, 0.01, []),
                (f"{name} zeros in |x| < 12", r["zeros_lt_12"], nz, 0.5, []),
                (f"{name} type rule", int(bool(r["type_rule"])), 1, 0.5, [])]
    W = [rc[f"C19-W{i}"]["truth"] for i in (1, 2, 3)]
    A = [rc[f"C19-ARC {k}"] for k in ("arc", "chord", "mirror", "beta2")]
    out += [("C19-W zeros, low", min(W), 1.00010, 1e-5, [(P, "| 1.00010–1.00016 | 7 / 18 / – |")]),
            ("C19-W zeros, high", max(W), 1.00016, 1e-5, [(P, "| 1.00010–1.00016 | 7 / 18 / – |")]),
            ("C19-ARC zeros, low", min(a["truth"] for a in A), 0.99993, 1e-5, [(P, "| 0.99993–1.00001 | 2–5 / 16 / – |")]),
            ("C19-ARC zeros, high", max(a["truth"] for a in A), 1.00001, 1e-5, [(P, "| 0.99993–1.00001 | 2–5 / 16 / – |")]),
            ("C19-ARC best omega_t, low", min(float(a["best_wt"]) for a in A), 0.99996, 1e-5, [(P, "best ω_t 0.99996–1.00032")]),
            ("C19-ARC best omega_t, high", max(float(a["best_wt"]) for a in A), 1.00032, 1e-5, [(P, "best ω_t 0.99996–1.00032")]),
            ("C19-ARC L2 floor, low", min(a["L2_floor"] for a in A), 1.1e-3, 0.1e-3, [(P, "L2 floor 1.1–2.5×10⁻³")]),
            ("C19-ARC L2 floor, high", max(a["L2_floor"] for a in A), 2.5e-3, 0.1e-3, [(P, "L2 floor 1.1–2.5×10⁻³")])]
    # 11.7 the projection
    h = pj["hyp"]
    out += [("projection for a zero at 1", h["1.0"]["pred19"] - 1, 1.27e-6, 0.01e-6, [(P, "predicts arg₁₉ − 1 = +1.27×10⁻⁶ for ω* = 1")]),
            ("projection for a zero at 0.999992", h["0.999992"]["pred19"] - 1, 0.62e-6, 0.01e-6, [(P, "+0.62×10⁻⁶ for")]),
            ("projection for a zero at 0.999975", h["0.999975"]["pred19"] - 1, 0.41e-6, 0.01e-6, [(P, "0.999992 and +0.41×10⁻⁶ for 0.999975")]),
            ("real arg_19 - 1", pj["arg"]["19"] - 1, 1.52e-6, 0.01e-6, [(P, "the real value is +1.52×10⁻⁶")]),
            ("miss, zero at 1", abs(h["1.0"]["miss"]), 0.25e-6, 0.01e-6, [(P, "0.25×10⁻⁶ from the first")]),
            ("miss, zero at 0.999992", abs(h["0.999992"]["miss"]), 0.90e-6, 0.01e-6, [(P, "0.90×10⁻⁶ from the second")]),
            ("miss, zero at 0.999975", abs(h["0.999975"]["miss"]), 1.11e-6, 0.01e-6, [(P, "and 1.11×10⁻⁶ from the third")]),
            ("model error", pj["model_err"], 3.4e-6, 0.1e-6, [(P, "The model's own error is 3.4×10⁻⁶")]),
            ("back-test, least", min(pj["back_real_max"].values()), 2.7e-6, 0.1e-6, [(P, "2.7–3.4×10⁻⁶ depending on ω*")]),
            ("plant misses, low", min(p["miss"] for p in pj["plants"]), -2.6e-6, 0.1e-6, [(P, "−2.6…+0.9×10⁻⁶")]),
            ("plant misses, high", max(p["miss"] for p in pj["plants"]), 0.9e-6, 0.1e-6, [(P, "−2.6…+0.9×10⁻⁶")]),
            ("trend per order, k = 4-19", pj["trend_4_19"]["per_order"], 0.540, 0.001, [(P, "falls ×0.540 per order (×0.539 over")]),
            ("trend per order, k = 4-18", pj["trend_4_18"]["per_order"], 0.539, 0.001, [(P, "falls ×0.540 per order (×0.539 over")]),
            ("trend's arg_19 - 1", pj["trend_4_18"]["at19"], 1.3e-6, 0.1e-6, [(P, "puts arg₁₉ − 1 at 1.3×10⁻⁶")]),
            ("noise shift of arg_19", max(abs(x) for x in pj["noise_shift"]), 2.6e-10, 0.1e-10, [(P, "moves by at most 2.6×10⁻¹⁰")]),
            ("projection decisive", int(bool(pj["decisive"])), 0, 0.5, [])]
    # 11.8 combination and 11.9 verdict
    rf = list(cb["ring_free"].values())
    k18 = (min(cal["18"]["all 28 arc plants"]["cells"]["arg"]["zero"][0], cal["18"]["BP family (12)"]["cells"]["arg"]["zero"][0]),
           max(cal["18"]["all 28 arc plants"]["cells"]["arg"]["zero"][1], cal["18"]["BP family (12)"]["cells"]["arg"]["zero"][1]))
    out += [("K = 19 hull, low", cb["range19"][0], 0.999996, 1e-6, [(P, "**The hull is 0.999996–1.000004.**")]),
            ("K = 19 hull, high", cb["range19"][1], 1.000004, 1e-6, [(P, "**The hull is 0.999996–1.000004.**")]),
            ("ring-free completions, low", min(rf), 0.99999998, 1e-8, [(P, "the ring-free completions, 0.99999998–1.00000009"), ("paper/math/main.tex", "put the zero at $0.99999998$--$1.00000009$")]),
            ("ring-free completions, high", max(rf), 1.00000009, 1e-8, [(P, "the ring-free completions, 0.99999998–1.00000009"), ("paper/math/main.tex", "put the zero at $0.99999998$--$1.00000009$")]),
            ("ring-free completions counted", len(rf), 4, 0.5, [(P, "C19-BP [6/13], [7/12], [8/11], [9/10],")]),
            ("headline, low", cb["headline"][0], 0.999975, 1e-6, [(P, "**0.999975–1.000050, unchanged.**")]),
            ("headline, high", cb["headline"][1], 1.000050, 1e-6, [(P, "**0.999975–1.000050, unchanged.**")]),
            ("headline overlap", int(bool(cb["overlap_k18"])), 1, 0.5, []),
            ("last two orders, low", min(k18[0], cb["range19"][0]), 0.999992, 1e-6, [(P, "reported as context by item 7, is 0.999992–1.000004")]),
            ("last two orders, high", max(k18[1], cb["range19"][1]), 1.000004, 1e-6, [(P, "reported as context by item 7, is 0.999992–1.000004")]),
            ("K = 19 hull width", (cb["range19"][1] - cb["range19"][0]) * 1e5, 0.8, 0.1, [(P, "a width of 0.8×10⁻⁵ against 1.2×10⁻⁵ at K = 18")]),
            ("K = 18 hull width", (k18[1] - k18[0]) * 1e5, 1.2, 0.1, [(P, "a width of 0.8×10⁻⁵ against 1.2×10⁻⁵ at K = 18")]),
            ("G production at K = 19", cb["G_minima"]["19"]["production"], 0.9999995, 1e-7, [(P, "production 0.9999995 at K = 19 (0.9999987 at K = 18)")]),
            ("G production at K = 18", cb["G_minima"]["18"]["production"], 0.9999987, 1e-7, [(P, "production 0.9999995 at K = 19 (0.9999987 at K = 18)")]),
            ("G minima span at K = 19, low", cb["G_minima"]["19"]["span"][0], 0.999998, 1e-6, [(P, "factor 5 of the best, 0.999998–1.000001 (0.999999–1.000010 at K = 18)")]),
            ("G minima span at K = 19, high", cb["G_minima"]["19"]["span"][1], 1.000001, 1e-6, [(P, "factor 5 of the best, 0.999998–1.000001 (0.999999–1.000010 at K = 18)")]),
            ("G minima span at K = 18, high", cb["G_minima"]["18"]["span"][1], 1.000010, 1e-6, [(P, "factor 5 of the best, 0.999998–1.000001 (0.999999–1.000010 at K = 18)")]),
            ("G global minimum at K = 19", cb["G_minima"]["19"]["global"], 0.999998, 1e-6, [(P, "its global minimum 0.999998 (1.000010)")]),
            ("G global minimum at K = 18", cb["G_minima"]["18"]["global"], 1.000010, 1e-6, [(P, "its global minimum 0.999998 (1.000010)")])]
    cv = c["K19 converge"]["nodes"]
    out += [("err19 relative at wf 0.99 (verdict)", cv["0.99"]["err19_rel"], 4.5e-13, 0.1e-13, [(P, "σ₁₉ is converged to 4.5–5.6×10⁻¹³ relative")]),
            ("err19 relative at wf 1.01 (verdict)", cv["1.01"]["err19_rel"], 5.6e-13, 0.1e-13, [(P, "σ₁₉ is converged to 4.5–5.6×10⁻¹³ relative")])]
    return out


@check
def k20_stages():
    """notes/PLANTED_SELECTION.md sections 12.2-12.10: the K = 20 stages ("K20 converge", "K20 plants check", "K20 real",
    "K20 ring test", "K20 calib", "K20 recomplete", "K20 project", "K20 combine" in solver/inner_amp/planted_checks_k20.json
    and the record "N0 scan41J1250" in planted_results_k20.jsonl)."""
    d = os.path.join(ROOT, "solver/inner_amp")
    sys.path.insert(0, d)
    import planted_k14 as K14
    sys.path.pop(0)
    import mpmath as mp
    P = "notes/PLANTED_SELECTION.md"
    c = json.load(open(os.path.join(d, "planted_checks_k20.json")))
    cv, pc, rl, rt, cal, rc, pj, cb = (c[k] for k in ("K20 converge", "K20 plants check", "K20 real", "K20 ring test", "K20 calib",
                                                      "K20 recomplete", "K20 project", "K20 combine"))
    out = [("pilot difference at sigma_20", cv["pilot_d20"], 1.52e-31, 0.01e-31, [(P, "err20 = plateau + 1.52×10⁻³¹"), (P, "|σ₂₀(a) − σ₂₀(b)| = 1.52×10⁻³¹, is 0.011")]),
           ("pilot difference over (a)'s plateau err20", cv["pilot"]["d20_over_err20a"], 0.011, 0.001, [(P, "|σ₂₀(a) − σ₂₀(b)| = 1.52×10⁻³¹, is 0.011")])]
    for w, e, r, m, pl, d2 in (("0.99", 1.24e-29, 6.50e-13, 154, 1.22e-29, 0.0074), ("1.0", 1.35e-29, 7.26e-13, 138, 1.33e-29, 0.0074),
                               ("1.01", 1.47e-29, 8.10e-13, 123, 1.46e-29, 0.0073)):
        n = cv["nodes"][w]
        cell = f"{e / 1e-29:.2f}×10⁻²⁹ ({r / 1e-13:.2f}×10⁻¹³, {m})"
        out += [(f"err20 at wf {w}", n["err20"], e, 0.01e-29, [(P, cell)]), (f"relative err20 at wf {w}", n["err20_rel"], r, 0.01e-13, [(P, cell)]),
                (f"margin at wf {w}", 1e-10 / n["err20_rel"], m, 1, [(P, cell)]),
                (f"plateau err20 at wf {w}", n["err20_plateau"], pl, 0.01e-29, [(P, "1.22, 1.33 and 1.46×10⁻²⁹")]),
                (f"ratio-law second difference at k = 18, wf {w}", cv["ratiolaw_d2"][w][-1], d2, 0.0001, [(P, f"| {d2:.4f} |")])]
    mp.mp.dps = 40
    a = json.load(open(os.path.join(d, "scan41J1250/sigF_w1.0000_K41_J1250_p1280.json")))
    sg = K14.sigmas(a)
    q = complex(sg[19] / sg[20])
    out += [("sigma_19 / sigma_20 at wf 1.00, real part", q.real, 9.82, 0.01, [(P, "σ₁₉/σ₂₀ = 9.82 − 9.32i")]),
            ("sigma_19 / sigma_20 at wf 1.00, imaginary part", q.imag, -9.32, 0.01, [(P, "σ₁₉/σ₂₀ = 9.82 − 9.32i")])]
    sp = cv["spread"]
    out += [("spread of sigma_20 in wf", sp["20"]["half_range_rel"] * 100, 6.3, 0.1, [(P, "σ₂₀ changes by 6.3% of itself")]),
            ("spread of sigma_18 in wf", sp["18"]["half_range_rel"] * 100, 6.1, 0.1, [(P, "(6.1%, 6.2% for σ₁₈, σ₁₉)")]),
            ("spread of sigma_19 in wf", sp["19"]["half_range_rel"] * 100, 6.2, 0.1, [(P, "(6.1%, 6.2% for σ₁₈, σ₁₉)")]),
            ("err20 over the wf change", sp["20"]["err_over_d1"], 1.25e-11, 0.01e-11, [(P, "1.25×10⁻¹¹ of that change")]),
            ("err18 over the wf change", sp["18"]["err_over_d1"], 5.96e-12, 0.01e-12, [(P, "(5.96×10⁻¹² at σ₁₈, 8.68×10⁻¹² at σ₁₉)")]),
            ("err19 over the wf change", sp["19"]["err_over_d1"], 8.68e-12, 0.01e-12, [(P, "(5.96×10⁻¹² at σ₁₈, 8.68×10⁻¹² at σ₁₉)")])]
    recs = {x["name"]: x for x in (json.loads(l) for l in open(os.path.join(d, "planted_results_k20.jsonl")) if l.strip())}
    plants = [x for x in recs.values() if not x["name"].startswith(("N0", "C1", "C2"))]
    out += [("plants at K = 20", sum(pc["families"].values()), 51, 0.5, [(P, "51 plants at K = 18, 19, 20 (13.5 min in all)")]),
            ("plants' minutes at K = 20", sum(x["seconds"] for x in plants) / 60, 13.5, 0.1, [(P, "51 plants at K = 18, 19, 20 (13.5 min in all)")]),
            ("N1 worst error at K = 20", pc["n1_worst"], 3.5e-7, 0.1e-7, [(P, "Padé and G to 3.5×10⁻⁷ at worst")])]
    e = rl["est"]
    rows = {"arg": "| arg crossing | 1.0000009 | 1.0000015 | 1.0000024 |", "ratio": "| ratio | 0.999988 | 0.999986 | 0.999966 |",
            "G": "| G (production κ̄) | 1.0000001 | 0.9999995 | 0.9999987 |", "aitken": "| Aitken, k − 2 … k | 0.9999991 | 1.0000009 | 1.0000011 |"}
    for K, vals in (("20", {"arg": (1.0000009, 1e-7), "ratio": (0.999988, 1e-6), "G": (1.0000001, 1e-7), "aitken": (0.9999991, 1e-7)}),
                    ("19", {"arg": (1.0000015, 1e-7), "ratio": (0.999986, 1e-6), "G": (0.9999995, 1e-7), "aitken": (1.0000009, 1e-7)}),
                    ("18", {"arg": (1.0000024, 1e-7), "ratio": (0.999966, 1e-6), "G": (0.9999987, 1e-7), "aitken": (1.0000011, 1e-7)})):
        for est, (qv, u) in vals.items():
            out.append((f"{est} at K = {K} (section 12.4)", e[K][est], qv, u, [(P, rows[est])]))
    rec = recs["N0 scan41J1250"]
    pr = rec["est"]["20"]["pade_central_range"]
    out += [("central Pade range at K = 20, low", pr[0], 0.9999999, 1e-7, [(P, "[3/16]–[15/4]: 0.9999999–1.0000002 (mid 1.0000000)")]),
            ("central Pade range at K = 20, high", pr[1], 1.0000002, 1e-7, [(P, "[3/16]–[15/4]: 0.9999999–1.0000002 (mid 1.0000000)")]),
            ("central Pade mid at K = 20", rec["est"]["20"]["pade_central_mid"], 1.0000000, 1e-7, [(P, "[3/16]–[15/4]: 0.9999999–1.0000002 (mid 1.0000000)")])]
    ag = {int(k): v for k, v in rec["est"]["20"]["arg"].items()}
    out += [("arg crossing at k = 17 (section 12.4)", ag[17], 1.0000046, 1e-7, [(P, "1.0000046, 1.0000024, 1.0000015, 1.0000009 for k = 17–20")]),
            ("shrink of arg_k - 1, 17 to 18 (12.4)", (ag[18] - 1) / (ag[17] - 1), 0.53, 0.01, [(P, "shrank by ×0.53 from k = 17 to 18")]),
            ("shrink of arg_k - 1, 18 to 19 (12.4)", (ag[19] - 1) / (ag[18] - 1), 0.63, 0.01, [(P, "×0.63 from 18 to 19 and ×0.57 from 19 to 20")]),
            ("shrink of arg_k - 1, 19 to 20", (ag[20] - 1) / (ag[19] - 1), 0.57, 0.01, [(P, "×0.63 from 18 to 19 and ×0.57 from 19 to 20")]),
            ("G over 1 at K = 20", (e["20"]["G"] - 1) * 1e7, 1.5, 0.1, [(P, "1.5×10⁻⁷ over it")]),
            ("last two differences of arg_k (12.4)", abs(ag[19] - ag[18]) * 1e6, 0.89, 0.01, [(P, "are 0.89×10⁻⁶, then 0.65×10⁻⁶")]),
            ("last difference of arg_k (12.4)", abs(ag[20] - ag[19]) * 1e6, 0.65, 0.01, [(P, "are 0.89×10⁻⁶, then 0.65×10⁻⁶")])]
    sz = rt["simple-zero law from tau_18, tau_19"]["tau20_rel"]
    out += [("simple-zero law at K = 20, best node", min(sz), 3.2e-6, 0.1e-6, [(P, "misses τ₂₀ by 3.2–6.3×10⁻⁶")]),
            ("simple-zero law at K = 20, worst node", max(sz), 6.3e-6, 0.1e-6, [(P, "misses τ₂₀ by 3.2–6.3×10⁻⁶")])]
    for name, lo, hi, u, sub in (("C19-BP [3/16]", None, 2.2e-6, 0.1e-6, "| 2.2×10⁻⁶ | 1.8–2.1×10⁻⁴ |"),
                                 ("C19-BP [4/15]", 0.6e-7, 1.3e-7, 0.1e-7, "| 0.6–1.3×10⁻⁷ | 0.5–1.2×10⁻⁵ |"),
                                 ("C19-BP [5/14]", 2.5e-7, 2.9e-7, 0.1e-7, "| 2.5–2.9×10⁻⁷ | 2.3–2.4×10⁻⁵ |"),
                                 ("C19-BP [6/13]", 4.6e-8, 5.4e-8, 0.1e-8, "| 4.6–5.4×10⁻⁸ | 4.4–4.5×10⁻⁶ |"),
                                 ("C19-BP [7/12]", 0.7e-8, 1.1e-8, 0.1e-8, "| 0.7–1.1×10⁻⁸ | 6.5–8.7×10⁻⁷ |"),
                                 ("C19-BP [8/11]", 4.7e-9, 5.2e-9, 0.1e-9, "| 4.7–5.2×10⁻⁹ | 4.2–4.5×10⁻⁷ |"),
                                 ("C19-BP [9/10]", 0.8e-9, 1.3e-9, 0.1e-9, "| 0.8–1.3×10⁻⁹ | 0.8–1.0×10⁻⁷ |")):
        v = rt[name]["sigma20_rel"]
        if lo is not None:
            out.append((f"{name} sigma_20 error, best node", min(v), lo, u, [(P, sub)]))
        out.append((f"{name} sigma_20 error, worst node", max(v), hi, u, [(P, sub)]))
    out += [("K = 19 rings rejected under the primary reading", sum(1 for n, v in rt.items() if isinstance(v, dict) and v.get("ring") and not v["survives"]), 10, 0.5,
             [(P, "All ten rings of the K = 19 generation are rejected under the primary reading of item 4")]),
            ("K = 19 rings surviving without D-bracket", sum(1 for n, v in rt.items() if isinstance(v, dict) and v.get("ring") and v["survives_without_bracket"]), 1, 0.5,
             [(P, "C19-BP [4/15] is rejected by D-bracket alone and survives under the second reading")]),
            ("C19-BP [4/15] rejected by D-bracket alone", int(rt["C19-BP [4/15]"]["extra_detectors"] == ["D-bracket"]), 1, 0.5, []),
            ("ring-free C19-BP sums, best sigma_20 prediction", min(min(rt[n]["sigma20_rel"]) for n in ("C19-BP [6/13]", "C19-BP [7/12]", "C19-BP [8/11]", "C19-BP [9/10]")), 8e-10, 1e-10,
             [(P, "predicted σ₂₀ to 8×10⁻¹⁰–5×10⁻⁸")]),
            ("ring-free C19-BP sums, worst sigma_20 prediction", max(max(rt[n]["sigma20_rel"]) for n in ("C19-BP [6/13]", "C19-BP [7/12]", "C19-BP [8/11]", "C19-BP [9/10]")), 5e-8, 1e-8,
             [(P, "predicted σ₂₀ to 8×10⁻¹⁰–5×10⁻⁸")])]
    z = lambda fam, est: cal["20"][fam]["cells"][est]["zero"]
    out += [("arc plants, arg, zero low (K = 20)", z("all 28 arc plants", "arg")[0], 0.999997, 1e-6, [(P, "0.00…+0.04 → 0.999997–1.000001")]),
            ("arc plants, arg, zero high (K = 20)", z("all 28 arc plants", "arg")[1], 1.000001, 1e-6, [(P, "0.00…+0.04 → 0.999997–1.000001")]),
            ("arc plants, arg error high (K = 20)", cal["20"]["all 28 arc plants"]["cells"]["arg"]["hi"] * 1e4, 0.04, 0.01, [(P, "0.00…+0.04 → 0.999997–1.000001")]),
            ("BP family (20), arg, zero low", z("BP family (20)", "arg")[0], 0.999998, 1e-6, [(P, "−0.01…+0.03 → 0.999998–1.000002")]),
            ("BP family (20), arg, zero high", z("BP family (20)", "arg")[1], 1.000002, 1e-6, [(P, "−0.01…+0.03 → 0.999998–1.000002")]),
            ("arc plants, ratio, zero low (K = 20)", z("all 28 arc plants", "ratio")[0], 0.999930, 1e-6, [(P, "−0.28…+0.58 → 0.999930–1.000016")]),
            ("arc plants, ratio, zero high (K = 20)", z("all 28 arc plants", "ratio")[1], 1.000016, 1e-6, [(P, "−0.28…+0.58 → 0.999930–1.000016")]),
            ("arc plants, Pade, zero low (K = 20)", z("all 28 arc plants", "pade")[0], 0.999998, 1e-6, [(P, "−0.01…+0.02 → 0.999998–1.000001")]),
            ("arc plants, Pade, zero high (K = 20)", z("all 28 arc plants", "pade")[1], 1.000001, 1e-6, [(P, "−0.01…+0.02 → 0.999998–1.000001")]),
            ("BP family (20), Pade, zero low", z("BP family (20)", "pade")[0], 0.999953, 1e-6, [(P, "−0.28…+0.47 → 0.999953–1.000028")]),
            ("BP family (20), Pade, zero high", z("BP family (20)", "pade")[1], 1.000028, 1e-6, [(P, "−0.28…+0.47 → 0.999953–1.000028")]),
            ("BP family (20), G, zero high", z("BP family (20)", "G")[1], 1.000002, 1e-6, [(P, "−0.01…+0.01 → 0.999999–1.000002")])]
    dv = cal["sigma_dev_summary"]
    arc = [dv[f] for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2")]
    out += [("arc plants' sigma_19 deviation, least (12.6)", min(x["sigma19"][0] for x in arc) * 100, 1.4, 0.1, [(P, "by 1.4–2.7% and 2.2–3.9% (worst node)")]),
            ("arc plants' sigma_19 deviation, most (12.6)", max(x["sigma19"][1] for x in arc) * 100, 2.7, 0.1, [(P, "by 1.4–2.7% and 2.2–3.9% (worst node)")]),
            ("arc plants' sigma_20 deviation, least", min(x["sigma20"][0] for x in arc) * 100, 2.2, 0.1, [(P, "by 1.4–2.7% and 2.2–3.9% (worst node)")]),
            ("arc plants' sigma_20 deviation, most", max(x["sigma20"][1] for x in arc) * 100, 3.9, 0.1, [(P, "by 1.4–2.7% and 2.2–3.9% (worst node)")]),
            ("C-EXP sigma_20 deviation", dv["C-EXP"]["sigma20"][1] * 100, 1.1, 0.1, [(P, "C-EXP's by 0.55%\n  and 1.1%".split("\\n")[0] if False else "and 1.1%, the REFIT plants' by 0.9–16% and 1.5–21%")]),
            ("REFIT sigma_20 deviation, most", dv["REFIT"]["sigma20"][1] * 100, 21, 1, [(P, "and 1.1%, the REFIT plants' by 0.9–16% and 1.5–21%")])]
    for name, zq, u, pole, nz in (("C20-BP [3/17]", 0.9999998, 1e-7, 1.71, 9), ("C20-BP [4/16]", 1.00000001, 1e-8, 1.73, 6), ("C20-BP [5/15]", 1.00000001, 1e-8, 1.73, 6),
                                  ("C20-BP [6/14]", 1.000000028, 1e-9, 1.77, 1), ("C20-BP [7/13]", 1.000000000, 1e-9, 1.78, 1), ("C20-BP [8/12]", 0.999999997, 1e-9, 1.79, 1),
                                  ("C20-BP [9/11]", 1.000000001, 1e-9, 1.81, 1), ("C20-BP [10/10]", 1.000000001, 1e-9, 1.81, 1)):
        r = rc[name]
        digits = {1e-7: 7, 1e-8: 8, 1e-9: 9}[u]
        out += [(f"{name} zero", r["truth"], zq, u, [(P, f"| {name} | {zq:.{digits}f} |")]),
                (f"{name} largest pole", max(r["max_pole"].values()), pole, 0.01, []),
                (f"{name} zeros in |x| < 12", r["zeros_lt_12"], nz, 0.5, []),
                (f"{name} type rule", int(bool(r["type_rule"])), 1, 0.5, [])]
    W = [rc[f"C20-W{i}"]["truth"] for i in (1, 2, 3)]
    A = [rc[f"C20-ARC {k}"] for k in ("arc", "chord", "mirror", "beta2")]
    out += [("C20-W zeros, low", min(W), 0.999965, 1e-6, [(P, "| 0.999965–0.999970 | 6–7 / 18 / – |")]),
            ("C20-W zeros, high", max(W), 0.999970, 1e-6, [(P, "| 0.999965–0.999970 | 6–7 / 18 / – |")]),
            ("C20-ARC zeros, low", min(x["truth"] for x in A), 0.99995, 1e-5, [(P, "| 0.99995–0.99999 | 0–5 / 16–18 / – |")]),
            ("C20-ARC zeros, high", max(x["truth"] for x in A), 0.99999, 1e-5, [(P, "| 0.99995–0.99999 | 0–5 / 16–18 / – |")]),
            ("C20-ARC best omega_t, low", min(float(x["best_wt"]) for x in A), 0.99957, 1e-5, [(P, "best ω_t 0.99957–1.00040")]),
            ("C20-ARC best omega_t, high", max(float(x["best_wt"]) for x in A), 1.00040, 1e-5, [(P, "best ω_t 0.99957–1.00040")]),
            ("C20-ARC L2 floor, low", min(x["L2_floor"] for x in A), 1.5e-3, 0.1e-3, [(P, "L2 floor 1.5–3.6×10⁻³")]),
            ("C20-ARC L2 floor, high", max(x["L2_floor"] for x in A), 3.6e-3, 0.1e-3, [(P, "L2 floor 1.5–3.6×10⁻³")])]
    h = pj["hyp"]
    out += [("projection for a zero at 1 (K = 20)", h["1.0"]["pred20"] - 1, 0.96e-6, 0.01e-6, [(P, "predicts arg₂₀ − 1 = +0.96×10⁻⁶ for ω* = 1")]),
            ("projection for a zero at 0.999996", h["0.999996"]["pred20"] - 1, 0.76e-6, 0.01e-6, [(P, "+0.76×10⁻⁶ for")]),
            ("projection for a zero at 0.999975 (K = 20)", h["0.999975"]["pred20"] - 1, 0.66e-6, 0.01e-6, [(P, "0.999996 and +0.66×10⁻⁶ for 0.999975")]),
            ("real arg_20 - 1", pj["arg"]["20"] - 1, 0.87e-6, 0.01e-6, [(P, "the real value is +0.87×10⁻⁶")]),
            ("miss, zero at 1 (K = 20)", abs(h["1.0"]["miss"]), 0.09e-6, 0.01e-6, [(P, "0.09×10⁻⁶ from the first")]),
            ("miss, zero at 0.999996", abs(h["0.999996"]["miss"]), 0.11e-6, 0.01e-6, [(P, "0.11×10⁻⁶ from the second")]),
            ("miss, zero at 0.999975 (K = 20)", abs(h["0.999975"]["miss"]), 0.21e-6, 0.01e-6, [(P, "and 0.21×10⁻⁶ from the third")]),
            ("model error (K = 20)", pj["model_err"], 1.7e-5, 0.1e-5, [(P, "The model's own error is 1.7×10⁻⁵")]),
            ("back-test, least (K = 20)", min(pj["back_real_max"].values()), 1.1e-6, 0.1e-6, [(P, "1.1–2.7×10⁻⁶ depending on ω*")]),
            ("back-test, most (K = 20)", max(pj["back_real_max"].values()), 2.7e-6, 0.1e-6, [(P, "1.1–2.7×10⁻⁶ depending on ω*")]),
            ("plant misses, low (K = 20)", min(p["miss"] for p in pj["plants"]), -1.7e-5, 0.1e-5, [(P, "−1.7×10⁻⁵…+1.0×10⁻⁶")]),
            ("plant misses, high (K = 20)", max(p["miss"] for p in pj["plants"]), 1.0e-6, 0.1e-6, [(P, "−1.7×10⁻⁵…+1.0×10⁻⁶")]),
            ("trend per order, k = 4-20", pj["trend_4_20"]["per_order"], 0.542, 0.001, [(P, "falls\n×0.542 per order".split("\\n")[0] if False else "×0.542 per order (×0.540 over k = 4–19)")]),
            ("trend per order, k = 4-19 (12.8)", pj["trend_4_19"]["per_order"], 0.540, 0.001, [(P, "×0.542 per order (×0.540 over k = 4–19)")]),
            ("trend's arg_20 - 1", pj["trend_4_19"]["at20"], 0.74e-6, 0.01e-6, [(P, "puts arg₂₀ − 1 at 0.74×10⁻⁶")]),
            ("noise shift of arg_20", max(abs(x) for x in pj["noise_shift"]), 4.2e-10, 0.1e-10, [(P, "moves by at most 4.2×10⁻¹⁰")]),
            ("projection decisive (K = 20)", int(bool(pj["decisive"])), 0, 0.5, [])]
    rf = list(cb["ring_free"].values())
    k19 = json.load(open(os.path.join(d, "planted_checks_k19.json")))
    r19 = k19["K19 combine"]["range19"]
    c18 = k19["K19 calib"]["18"]
    k18 = (min(c18["all 28 arc plants"]["cells"]["arg"]["zero"][0], c18["BP family (12)"]["cells"]["arg"]["zero"][0]),
           max(c18["all 28 arc plants"]["cells"]["arg"]["zero"][1], c18["BP family (12)"]["cells"]["arg"]["zero"][1]))
    out += [("K = 20 hull, low", cb["range20"][0], 0.999997, 1e-6, [(P, "**The hull is 0.999997–1.000002.**")]),
            ("K = 20 hull, high", cb["range20"][1], 1.000002, 1e-6, [(P, "**The hull is 0.999997–1.000002.**")]),
            ("ring-free completions at K = 20, low", min(rf), 0.999999997, 1e-9, [(P, "the ring-free completions, 0.999999997–1.000000028"), ("paper/math/main.tex", "put the zero at $0.999999997$--$1.000000028$")]),
            ("ring-free completions at K = 20, high", max(rf), 1.000000028, 1e-9, [(P, "the ring-free completions, 0.999999997–1.000000028"), ("paper/math/main.tex", "put the zero at $0.999999997$--$1.000000028$")]),
            ("ring-free completions counted at K = 20", len(rf), 5, 0.5, [(P, "C20-BP [6/14], [7/13], [8/12], [9/11], [10/10],")]),
            ("headline overlap at K = 20", int(bool(cb["overlap_k19"])), 1, 0.5, []),
            ("last two orders at K = 20, low", min(r19[0], cb["range20"][0]), 0.999996, 1e-6, [(P, "reported as context by item 7, is 0.999996–1.000004")]),
            ("last two orders at K = 20, high", max(r19[1], cb["range20"][1]), 1.000004, 1e-6, [(P, "reported as context by item 7, is 0.999996–1.000004")]),
            ("K = 18-20 union, low", min(k18[0], r19[0], cb["range20"][0]), 0.999992, 1e-6, [(P, "K = 18–20 would give\n  0.999992–1.000004".split("\\n")[0] if False else "K = 18–20 would give 0.999992–1.000004")]),
            ("K = 20 hull width", (cb["range20"][1] - cb["range20"][0]) * 1e5, 0.5, 0.1, [(P, "a width of 0.5×10⁻⁵ against 0.8×10⁻⁵ at K = 19")]),
            ("K = 19 hull width (12.10)", (r19[1] - r19[0]) * 1e5, 0.8, 0.1, [(P, "a width of 0.5×10⁻⁵ against 0.8×10⁻⁵ at K = 19")]),
            ("G production at K = 20", cb["G_minima"]["20"]["production"], 1.0000001, 1e-7, [(P, "production 1.0000001 at K = 20 (0.9999995 at K = 19)")]),
            ("G production at K = 19 (12.9)", cb["G_minima"]["19"]["production"], 0.9999995, 1e-7, [(P, "production 1.0000001 at K = 20 (0.9999995 at K = 19)")]),
            ("G minima span at K = 20, low", cb["G_minima"]["20"]["span"][0], 1.000000, 1e-6, [(P, "factor 5 of the best, 1.000000–1.000003 (0.999998–1.000001 at K = 19)")]),
            ("G minima span at K = 20, high", cb["G_minima"]["20"]["span"][1], 1.000003, 1e-6, [(P, "factor 5 of the best, 1.000000–1.000003 (0.999998–1.000001 at K = 19)")]),
            ("G global minimum at K = 20", cb["G_minima"]["20"]["global"], 1.000000, 1e-6, [(P, "its global minimum 1.000000 (0.999998)")]),
            ("G global minimum at K = 19 (12.9)", cb["G_minima"]["19"]["global"], 0.999998, 1e-6, [(P, "its global minimum 1.000000 (0.999998)")]),
            ("relative err20 at wf 0.99 (verdict)", cv["nodes"]["0.99"]["err20_rel"], 6.5e-13, 0.1e-13, [(P, "σ₂₀ is converged to 6.5–8.1×10⁻¹³ relative")]),
            ("relative err20 at wf 1.01 (verdict)", cv["nodes"]["1.01"]["err20_rel"], 8.1e-13, 0.1e-13, [(P, "σ₂₀ is converged to 6.5–8.1×10⁻¹³ relative")]),
            ("margins at K = 20, least", min(1e-10 / cv["nodes"][w]["err20_rel"] for w in cv["nodes"]), 123, 1, [(P, "with margins of 123–154 under item 1")]),
            ("margins at K = 20, most", max(1e-10 / cv["nodes"][w]["err20_rel"] for w in cv["nodes"]), 154, 1, [(P, "with margins of 123–154 under item 1")])]
    return out


@check
def homega():
    """proof/P5_scope.md section 4.2 item 2's result: the (H_omega) interpolation measurement on the five nodes
    (solver/proof/logs/homega_interp.json, from `homega_interp.py real` in solver/proof/)."""
    P, Q = "proof/P5_scope.md", "notes/PROOF_QUEUE.md"
    rows = json.load(open(os.path.join(ROOT, "solver/proof/logs/homega_interp.json")))["rows"]
    r19, r1 = rows[19], rows[1]
    k1e6 = next(r["k"] for r in rows if r["k"] >= 1 and max(r["m2m"], r["m2p"]) > 1e-6)
    hi = [r["R34"] for r in rows if r["k"] >= 11]
    incons = [r["k"] for r in rows if not r["consistent"]]
    return [("three-node midpoint error at k = 19, 0.995", r19["m2m"], 1.52e-5, 0.01e-5, [(P, "1.52×10⁻⁵ and 1.55×10⁻⁵ of σ₁₉")]),
            ("three-node midpoint error at k = 19, 1.005", r19["m2p"], 1.55e-5, 0.01e-5, [(P, "1.52×10⁻⁵ and 1.55×10⁻⁵ of σ₁₉")]),
            ("three-node midpoint error at k = 19, rounded", r19["m2m"], 1.5e-5, 0.1e-5,
             [(P, "midpoint error 1.5×10⁻⁵ of σ₁₉"), (Q, "halfway between the nodes is 1.5×10⁻⁵ of σ₁₉")]),
            ("three-node midpoint error at k = 1", r1["m2m"], 9.2e-8, 0.1e-8, [(P, "rises from 9.2×10⁻⁸ of σ₁")]),
            ("first k with the midpoint error above 1e-6", k1e6, 5, 1, [(P, "above 10⁻⁶ from k = 5 on")]),
            ("four-node centre error at k = 1", r1["m3"], 5.5e-10, 0.1e-10, [(P, "5.5×10⁻¹⁰ of σ₁")]),
            ("four-node centre error at k = 19", r19["m3"], 1.66e-7, 0.01e-7, [(P, "1.66×10⁻⁷ of σ₁₉")]),
            ("radius |a3/a4| at k = 19", r19["R34"], 61.8, 0.1, [(P, "61.8 at k = 19")]),
            ("radius in omega at k = 19", 0.01 * r19["R34"], 0.62, 0.01, [(P, "|ω/ω∞ − 1| ≈ 0.62"), (P, "the radius of analyticity 0.62 in ω/ω∞")]),
            ("radius |a3/a4| over k = 11..19, min", min(hi), 62, 1, [(P, "62–71 node spacings for k = 11…19")]),
            ("radius |a3/a4| over k = 11..19, max", max(hi), 71, 1, [(P, "62–71 node spacings for k = 11…19")]),
            ("sqrt|a2/a4| at k = 19", r19["R24"], 54.3, 0.1, [(P, "54.3 against 61.8 at k = 19")]),
            ("inconsistent orders (sum of 10^k)", sum(10 ** k for k in incons), 11001, 1,
             [(P, "inconsistent only at k = 0, where σ₀ is constant"), (P, "at k = 3, 4). Under that picture")]),
            ("nodes for the radius-1e-4 box at k = 19", r19["nbox"], 3, 1, [(P, "needs **3 nodes** at every k ≤ 19"), (Q, "a box of radius 10⁻⁴ needs 3 nodes")]),
            ("nodes for the radius-1e-4 box, every consistent k", max(r["nbox"] for r in rows if r["consistent"]), 3, 1,
             [(P, "needs **3 nodes** at every k ≤ 19")])]


@check
def first_order():
    """The 1/n coefficients from the O(1/n) inner problem (notes/FIRST_ORDER.md): omega_1, Delta_1 = -2 pi omega_1 /
    omega*^2 and lambda_1 in the M-harmonic truncations (M = 5: the mean of the three runs), and the fitted values."""
    d = os.path.join(ROOT, "solver/inner_pert/data")
    def run(name):
        return json.load(open(os.path.join(d, name + ".json")))
    r1, r3 = run("first_order_hier_M1_xi0.2"), run("first_order_hier_M3_xi0.2")
    r5 = [run(n) for n in ("first_order_hier_M5_xi0.2", "first_order_hier_M5_xi0.19", "first_order_hier_M5_xi0.2_rtol1e-13")]
    mean = lambda k: sum(r[k] for r in r5) / len(r5)
    D1 = lambda r: -2 * math.pi * r["omega1"] / r["omega"] ** 2
    fit = open(os.path.join(ROOT, "solver/inner_pert/logs/first_order_fits.log")).read()
    m = re.search(r"Delta_1 in \[([0-9.]+), ([0-9.]+)\], omega_1 in \[(-[0-9.]+), (-[0-9.]+)\]", fit)
    t = [("README.md", "| ω₁ | −21.05 | −23.71 | −24.1485 | −24.123 |"), ("README.md", "| Δ₁ | 15.00 | 9.886 | 10.402 | 10.375 |"),
         ("README.md", "| λ₁ | 7.76 | 3.080 | 3.023 | 3.0324 |")]
    return [("omega_1, M = 1", r1["omega1"], -21.05, 1e-2, t[:1]), ("omega_1, M = 3", r3["omega1"], -23.71, 1e-2, t[:1]),
            ("omega_1, M = 5", mean("omega1"), -24.1485, 1e-4, t[:1]),
            ("Delta_1, M = 1", D1(r1), 15.00, 1e-2, t[1:2]), ("Delta_1, M = 3", D1(r3), 9.886, 1e-3, t[1:2]),
            ("Delta_1, M = 5", sum(D1(r) for r in r5) / 3, 10.402, 1e-3, t[1:2]),
            ("lambda_1, M = 1", r1["lam1"], 7.76, 1e-2, t[2:]), ("lambda_1, M = 3", r3["lam1"], 3.080, 1e-3, t[2:]),
            ("lambda_1, M = 5", mean("lam1"), 3.023, 1e-3, t[2:]),
            ("fitted Delta_1", (float(m.group(1)) + float(m.group(2))) / 2, 10.375, 1e-3, t[1:2]),
            ("fitted omega_1", (float(m.group(3)) + float(m.group(4))) / 2, -24.123, 1e-3, t[:1])]


@check
def gamma_slope():
    line = next(l for l in open(os.path.join(ROOT, "hp/data/local/gamma_compare.txt"))
                if l.startswith("D >= 30") and "fixed and 9 powers" in l)
    a1 = float(re.search(r"a1 = ([0-9.]+)", line).group(1))
    return [("lambda = 2 + a1/n (D >= 30, 9 powers, lambda_inf = 2 fixed)", a1, 3.0324, 1e-4,
             [("README.md", "3.0324"), ("GAMMA.md", "3.0324")]),
            ("gamma = 1/2 - (a1/4)/n", a1 / 4, 0.758, 1e-3, [("README.md", "0.758/n")])]


@check
def five_harmonics():
    log = open(os.path.join(ROOT, "solver/inner_stokes/logs/hier_M5_full_schur.log")).read()
    x = re.findall(r"x \[([0-9.]+), ([0-9.]+),", log)[-1]
    A, wf = float(x[0]), float(x[1])
    return [("M = 5 omega/omega_inf", wf, 0.999228, 1e-6, [("README.md", "0.999228")]),
            ("M = 5 amplitude", A, 2.71858, 1e-5, [("notes/INNER_STOKES.md", "A = 2.71858")]),
            ("M = 5 Delta = Delta_inf / wf", DINF / wf, 1.64516, 1e-5, [("README.md", "Δ = 1.64516")]),
            ("M = 5 offset in Delta, percent", (1 / wf - 1) * 100, 0.077, 1e-3, [("README.md", "0.077% above")])]


@check
def certificate():
    sys.path.insert(0, os.path.join(ROOT, "solver", "proof"))
    import mutation_controls as MC
    from flint import arb
    import krawczyk as KR
    d = json.load(open(MC.RUNS))
    ok, K, margins = MC.assemble(d)
    items = [("Krawczyk margin in A (1e-9 box)", margins[0], 7.925e-10, 1e-13, [("proof/A_certify.md", "7.925×10⁻¹⁰")]),
             ("Krawczyk margin in omega", margins[1], 9.247e-10, 1e-13, [("proof/A_certify.md", "9.247×10⁻¹⁰")]),
             ("test passes on the 1e-9 box", int(ok), 1, 1, [])]
    # the quoted zero enclosure: the 1.3e-10 sub-box test (J over the 1e-9 box encloses J over any sub-box)
    lo = ("2.54214804760473", "0.776994013601194")
    hi = ("2.54214804762839", "0.776994013618254")
    txt = [("proof/A_certify.md", f"A \\in [{lo[0]},\\ {hi[0]}]")]
    Kp = sub_box_K(d, "1.3e-10", MC, KR, arb)
    for i, nm in enumerate(("A", "omega/omega_inf")):
        inside = bool(Kp[i].lower() >= arb(lo[i]) and Kp[i].upper() <= arb(hi[i]))
        items.append((f"quoted {nm} enclosure contains K(X') (outward rounding)", int(inside), 1, 1, txt))
    return items


@check
def lemma6b1():
    """proof/A_certify.md section 12: the disc covering of lemma 6b_1 (saved check of lemma6b1_cover.py)"""
    d = json.load(open(os.path.join(ROOT, "solver/proof/lemma6b1_check_1e-9.json")))
    discs = sum(r["discs"] for r in d["rows"])
    lam = max(r["Lam"] for r in d["rows"])
    steps = sum(r["steps"] for r in d["rows"]) + sum(leg["steps"] for leg in d["ray"])
    txt = [("proof/A_certify.md", "178 discs in 12 rows; contraction ≤ 0.439")]
    # the contraction is quoted as an upper bound: it may be rounded up, never down
    q = 0.439
    lam_ok = lam <= q and q - lam < 1e-3
    return [("lemma 6b1 discs", discs, 178, 1, txt),
            ("lemma 6b1 rows", len(d["rows"]), 12, 1, txt),
            (f"lemma 6b1 max contraction {lam:.6f} (quoted rounded up)", q if lam_ok else float("nan"), q, 1e-3, txt),
            ("lemma 6b1 Lohner steps (rows and ray)", steps, 6073, 1, [("proof/A_certify.md", "6073 Lohner steps")]),
            ("lemma 6b1 check passes", int(d["ok"]), 1, 1, [])]


def sub_box_K(d, h, MC, KR, arb):
    from flint import ctx
    ctx.prec = 320
    xhat = [arb(d["A"]), arb(d["wf"])]
    Dm1, Dp1 = MC.parse_acb(d["point"]["Dm1"]), MC.parse_acb(d["point"]["Dp1"])
    dDm1 = [MC.parse_acb(p) for p in d["box"]["dDm1"]]
    dDp1 = [MC.parse_acb(p) for p in d["box"]["dDp1"]]
    N, xs = KR.normalization(xhat, d["xi1"], prec=320)
    g_hat, JX = KR.g_from_states(Dm1, Dp1, dDm1, dDp1, N, xs)
    ok, K, _ = KR.krawczyk(xhat, g_hat, JX, [arb(h), arb(h)])
    assert ok, "the sub-box test must pass"
    return K


@check
def short_forms():
    """README "The formula", the shorter-forms table: the 40-61 three-term series and its margin, recomputed (a fast
    row: one range, one fit; the whole table takes about 15 minutes with short_forms.py)."""
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import dd_fits_mp as F
    import short_forms as S
    pts, _ = F.load_points()
    Ds = F.clean(pts, sorted(d for d in pts if 10 <= d <= 61.03), verbose=False)
    Dl = [d for d in Ds if 40 <= d <= 61]
    ev, _, _ = F.fit(pts, Dl, "1/D", 2)
    c = [float(v) for v in S.series_coeffs(ev, 3)]
    err = float(S.error_in_range(pts, Ds, ev, 40, 61))
    txt = [("notes/FORMULA_FORMS.md", "| 40–61 | 10⁻⁴ | 3 | 1.6450284 + 10.19239/D − 18.04099/D² | 5.7×10⁻⁶ |")]
    return [("40-61 c0", c[0], 1.6450284, 1e-7, txt), ("40-61 c1", c[1], 10.19239, 1e-5, txt),
            ("40-61 c2", c[2], -18.04099, 1e-5, txt), ("40-61 margin", err, 5.7e-6, 1e-7, txt)]


@check
def gamma_35():
    """README "Below D = 4": gamma(3.5) at M = 128 and its change from M = 96, from hp/data/delta_dd_lowD.csv (#628)"""
    import csv
    g = {int(r["M"]): float(r["gamma"]) for r in csv.DictReader(l for l in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv"))
                                                                  if not l.startswith("#"))
         if abs(float(r["D"]) - 3.5) < 1e-9 and r["gamma"]}
    txt = [("README.md", "At D = 3.5 the double-double solver gives γ = 0.3223856, which moves by 3.8×10⁻⁹ from M = 96 to 128")]
    return [("gamma(3.5) at M 128", g[128], 0.3223856, 1e-7, txt),
            ("gamma(3.5), M 96 to 128", g[128] - g[96], 3.8e-9, 1e-10, txt)]


@check
def chain_reach():
    """the README's reach of the large-D chain: a chain_fit row of hp/data/delta_all.csv at that D (#603)"""
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv"))) if r["kind"] == "chain_fit"]
    D = min((float(r["D"]) for r in rows), key=lambda d: abs(d - 245.6366))
    txt = [("README.md", "harmonics 5–13 of the centre field on a fit, to D = 245.6."),
           ("README.md", "held on a fit of the full solutions, to D = 245.6.")]
    return [("the chain's reach in the README", D, 245.6, 0.1, txt)]


@check
def jet_634():
    """notes/LARGE_D_WALL.md, "The chain's series past D 236" (#634): the stored predictions against the converged
    states, the series' changes under injected noise, a moved base and quad-double, the attempts' first moves, and the
    singularity's place (hp/data/jet_634.json, written by hp/jet_singularity.py summary from the raw-data repository's
    I634-mac and I603-win2 folders). Each value is compared as a mantissa at its quoted power of ten."""
    J = json.load(open(os.path.join(ROOT, "hp/data/jet_634.json")))
    W = "notes/LARGE_D_WALL.md"
    p, v = J["preds"], J["variants"]["245.1366"]
    m = lambda x, e: x / 10.0 ** e  # noqa: E731

    def rel(tag, k):
        return v[tag]["rel"][str(k)]
    acc = [a for a in J["attempts"] if a["accepted"] and a["a"] > 0.9]
    ref = {round(a["D0"], 2): a for a in J["attempts"] if not a["accepted"]}
    r13 = [r["radius_12"] for r in J["records"] if r["D0"] < 236.7]
    r14 = [r["radius_12"] for r in J["records"] if r["D0"] > 236.7]
    L = J["locate"]["236.6312"]
    B = J["records_beyond"]
    Bd = {round(r["D0"], 2): r for r in B}
    t_pred = [(W, "All 27 predictions err by 1.1×10⁻³¹ to 1.7×10⁻²⁹ at a ≥ 0.93")]
    t_tab = [(W, "| 5 | 1.876×10⁻¹² | 3.3×10⁻¹¹ | 5.3×10⁻²⁶ | 1.7×10⁻²⁶ | 5.3×10² | 3.9×10¹³⁹ |")]
    items = [
        ("predictions", p["n"], 27, 1, t_pred),
        ("prediction error at a >= 0.93, least", m(p["err_a1_min"], -31), 1.1, 0.1, t_pred),
        ("prediction error at a >= 0.93, most", m(p["err_a1_max"], -29), 1.7, 0.1, t_pred),
        ("prediction error at a <= 0.61, least", m(p["err_small_min"], -36), 2.5, 0.1, [(W, "by 2.5×10⁻³⁶ to 2.0×10⁻³⁵ at a ≤ 0.61")]),
        ("prediction error at a <= 0.61, most", m(p["err_small_max"], -35), 2.0, 0.1, [(W, "by 2.5×10⁻³⁶ to 2.0×10⁻³⁵ at a ≤ 0.61")]),
        ("middle orders' terms at a >= 0.93, least", m(p["mid_a1_min"], -14), 5.9, 0.1, [(W, "are 5.9×10⁻¹⁴ to\n4.35×10⁻¹³ at a ≥ 0.93")]),
        ("middle orders' terms at a >= 0.93, most", m(p["mid_a1_max"], -13), 4.35, 0.01, [(W, "are 5.9×10⁻¹⁴ to\n4.35×10⁻¹³ at a ≥ 0.93")]),
        ("least factor below the middle orders", m(p["factor_min"], 16), 2.6, 0.1, [(W, "within 1/(2.6×10¹⁶) of them")]),
        ("reproduced norms at D 236.63", J["repro"]["236.6312_N24"]["equal"], 25, 1, [(W, "equals the record at all 25 orders")]),
        ("injected noise, order 1", m(rel("inject_1e-80_1", 1), -19), 4.4, 0.1, [(W, "| 1 | −1.69×10⁻⁴ | 4.4×10⁻¹⁹ | 9.9×10⁻³⁴ |")]),
        ("injected noise, order 5", m(rel("inject_1e-80_1", 5), -11), 3.3, 0.1, t_tab),
        ("injected noise, order 8", m(rel("inject_1e-80_1", 8), -11), 3.6, 0.1, [(W, "| 8 | 6.73×10⁻¹⁵ | 3.6×10⁻¹¹ |")]),
        ("base at 1.6e-60, order 5", m(rel("pert_1e-74_1", 5), -26), 5.3, 0.1, t_tab),
        ("base at 1.6e-60, order 8", m(rel("pert_1e-74_1", 8), -25), 5.7, 0.1, [(W, "| 8 | 6.73×10⁻¹⁵ | 3.6×10⁻¹¹ | 5.7×10⁻²⁵ |")]),
        ("base at 5.0e-61, order 5", m(rel("pert_1e-74_2", 5), -26), 1.7, 0.1, t_tab),
        ("base at 1.6e-32, order 5", m(rel("pert_1e-60_1", 5), 2), 5.3, 0.1, t_tab),
        ("quad-double, order 1", m(rel("qd", 1), 17), 6.5, 0.1, [(W, "It is wrong by 6.5×10¹⁷ at order 1")]),
        ("quad-double, order 5", m(rel("qd", 5), 139), 3.9, 0.1, t_tab),
        ("residual of the base moved at 1e-74, seed 1", m(v["pert_1e-74_1"]["info"]["dF_max"], -60), 1.6, 0.1, [(W, "base at max\\|F\\| 1.6×10⁻⁶⁰")]),
        ("move at 1e-74, seed 1", m(v["pert_1e-74_1"]["info"]["dz_max"], -48), 5.6, 0.1, [(W, "the move is 5.6×10⁻⁴⁸ (seed 1)")]),
        ("move at 1e-74, seed 2", m(v["pert_1e-74_2"]["info"]["dz_max"], -48), 1.8, 0.1, [(W, "or 1.8×10⁻⁴⁸\n  (seed 2)")]),
        ("move at 1e-60", m(v["pert_1e-60_1"]["info"]["dz_max"], -34), 5.6, 0.1, [(W, "10⁻⁶⁰ the move is 5.6×10⁻³⁴ and the residual 1.6×10⁻³²")]),
        ("residual at 1e-60", m(v["pert_1e-60_1"]["info"]["dF_max"], -32), 1.6, 0.1, [(W, "10⁻⁶⁰ the move is 5.6×10⁻³⁴ and the residual 1.6×10⁻³²")]),
        ("accepted first moves, least", m(min(a["dz"] for a in acc), -31), 1.1, 0.1, [(W, "| accepted at a > 0.9 (13) | 1.1×10⁻³¹ to 1.7×10⁻²⁹ |")]),
        ("accepted first moves, most", m(max(a["dz"] for a in acc), -29), 1.7, 0.1, [(W, "| accepted at a > 0.9 (13) | 1.1×10⁻³¹ to 1.7×10⁻²⁹ |")]),
        ("accepted attempts at a > 0.9", len(acc), 13, 1, [(W, "| accepted at a > 0.9 (13) |")]),
        ("refused at D 236.13, first move", m(ref[236.13]["dz"], -31), 7.7, 0.1, [(W, "| refused, base D 236.13, a = 1 | 7.7×10⁻³¹ |")]),
        ("refused at D 240.12, first move", m(ref[240.12]["dz"], -33), 4.9, 0.1, [(W, "| refused, base D 240.12, a = 0.763 | 4.9×10⁻³³ |")]),
        ("refused at D 244.64, after the first move", m(ref[244.64]["F1"], -24), 1.4, 0.1, [(W, "| refused, base D 244.64, a = 1 | 6.2×10⁻³⁰ | 2.6×10⁻³³ | 1.4×10⁻²⁴ |")]),
        ("root test to D 236.63, least", min(r13), 7.1, 0.1, [(W, "7.1 to 7.9 at the 13 bases from D 225.13 to 236.63")]),
        ("root test to D 236.63, most", max(r13), 7.9, 0.1, [(W, "7.1 to 7.9 at the 13 bases from D 225.13 to 236.63")]),
        ("root test from D 237.13, least", min(r14), 7.3, 0.1, [(W, "7.3 to 9.7 at the 14 bases from D 237.13 to\n245.14")]),
        ("root test from D 237.13, most", max(r14), 9.7, 0.1, [(W, "7.3 to 9.7 at the 14 bases from D 237.13 to\n245.14")]),
        ("rows with a stable pole at D 236.63", L["rows_found"], 17, 1, [(W, "Of the 24 rows with the largest |z_8|, 17 have a stable Padé pole")]),
        ("their |s|, least", L["rows_abs"][0], 8.3, 0.1, [(W, "at |s| = 8.3 to 9.7 (median 8.75)")]),
        ("their |s|, median", L["rows_abs"][1], 8.75, 0.01, [(W, "at |s| = 8.3 to 9.7 (median 8.75)")]),
        ("their |s|, most", L["rows_abs"][2], 9.7, 0.1, [(W, "at |s| = 8.3 to 9.7 (median 8.75)")]),
        ("their argument, median", L["rows_arg"][1], 86.6, 0.1, [(W, "59° to 113° (median 86.6°)")]),
        ("their argument, least", L["rows_arg"][0], 59, 1, [(W, "59° to 113° (median 86.6°)")]),
        ("their argument, most", L["rows_arg"][2], 113, 1, [(W, "59° to 113° (median 86.6°)")]),
        ("accepted steps' convergence, least", m(min(a["F_conv"] for a in J["attempts"] if a["accepted"]), -96), 2.85, 0.01,
         [(W, "#603's accepted steps converged to\n  2.85×10⁻⁹⁶ to 2.6×10⁻⁶²")]),
        ("accepted steps' convergence, most", m(max(a["F_conv"] for a in J["attempts"] if a["accepted"]), -62), 2.6, 0.1,
         [(W, "#603's accepted steps converged to\n  2.85×10⁻⁹⁶ to 2.6×10⁻⁶²")]),
        ("quad-double's rounding as an injection", m(1e-80 * rel("qd", 1) / rel("inject_1e-80_1", 1), -44), 1.5, 0.1,
         [(W, "Quad-double rounds like an injection of\n  1.5×10⁻⁴⁴")]),
        ("octo-double's rounding as an injection", m(2.0 ** -212 * 1e-80 * rel("qd", 1) / rel("inject_1e-80_1", 1), -108), 2, 1,
         [(W, "rounds like one of about 2×10⁻¹⁰⁸")]),
        ("octo-double's rounding in z_5..z_8", m(max(rel("inject_1e-80_1", k) for k in (5, 6, 7, 8)) * 2.0 ** -212 * rel("qd", 1)
                                                 / rel("inject_1e-80_1", 1), -39), 8, 1, [(W, "That would move z_5 to z_8 by\n  8×10⁻³⁹ at most")]),
        ("od series at D 245.14, norms equal to the record", J["od_vs_record"]["245.1366"]["equal"]
         + J["od_vs_record"]["245.1366"]["pivots_equal"], 10, 1, [(W, "Its od series at D 245.14 equals #603's record at orders 0 to 8 bit for bit")]),
        ("od series at D 243.64, norms equal to the record", J["od_vs_record"]["243.6366"]["equal"]
         + J["od_vs_record"]["243.6366"]["pivots_equal"], 10, 1, [(W, "At D 243.64 the od series equals\n#603's record at orders 0 to 8 bit for bit")]),
        ("driver against anm_run.py at D 236.63", int(J["driver"]["236.6312"]), 1, 1,
         [(W, "At D 236.63 the driver's od series equals `anm_run.py`'s 24-order series in every part of every entry")]),
        ("move at D 236.63", m(J["variants"]["236.6312"]["pert_1e-74_1"]["info"]["dz_max"], -51), 1.2, 0.1,
         [(W, "moves that base by 1.2×10⁻⁵¹ and leaves max|F| at 9.4×10⁻⁶⁹")]),
        ("residual at D 236.63", m(J["variants"]["236.6312"]["pert_1e-74_1"]["info"]["dF_max"], -69), 9.4, 0.1,
         [(W, "moves that base by 1.2×10⁻⁵¹ and leaves max|F| at 9.4×10⁻⁶⁹")]),
        ("change at D 236.63", m(J["variants"]["236.6312"]["pert_1e-74_1"]["rel_max_5_8"], -35), 1.1, 0.1,
         [(W, "z_8 by 1.1×10⁻³⁵ at most")]),
        ("quad-double at D 236.63, order 1", m(J["variants"]["236.6312"]["qd"]["rel"]["1"], 15), 1.6, 0.1,
         [(W, "the quad-double series is wrong by 1.6×10¹⁵ at order 1")]),
        ("quad-double at D 243.64, order 1", m(J["variants"]["243.6366"]["qd"]["rel"]["1"], 16), 1.4, 0.1,
         [(W, "is wrong by 1.4×10¹⁶ at order 1")]),
        ("change per unit residual at D 236.63", m(J["variants"]["236.6312"]["pert_1e-74_1"]["rel_max_5_8"]
                                                  / J["variants"]["236.6312"]["pert_1e-74_1"]["info"]["dF_max"], 33), 1.2, 0.1,
         [(W, "1.2×10³³ at D 236.63 to 3.7×10³⁵ at D 245.14")]),
        ("change per unit residual at D 245.14", m(v["pert_1e-74_1"]["rel_max_5_8"] / v["pert_1e-74_1"]["info"]["dF_max"], 35), 3.7, 0.1,
         [(W, "1.2×10³³ at D 236.63 to 3.7×10³⁵ at D 245.14")]),
        ("D 245.14, rows with a stable pole (16 orders)", J["locate"]["245.1366"]["rows_found"], 24, 1,
         [(W, "For the same 24 rows, they give stable Padé poles")]),
        ("D 245.14, |s| least (16 orders)", J["locate"]["245.1366"]["rows_abs"][0], 5.6, 0.1, [(W, "at |s| = 5.6 to 7.0 (median 6.7)")]),
        ("D 245.14, |s| median (16 orders)", J["locate"]["245.1366"]["rows_abs"][1], 6.7, 0.1, [(W, "at |s| = 5.6 to 7.0 (median 6.7)")]),
        ("D 245.14, |s| most (16 orders)", J["locate"]["245.1366"]["rows_abs"][2], 7.0, 0.1, [(W, "at |s| = 5.6 to 7.0 (median 6.7)")]),
        ("D 245.14, argument least", J["locate"]["245.1366"]["rows_arg"][0], 9, 1, [(W, "arguments 9° to 107°\n(median 42°)")]),
        ("D 245.14, argument median", J["locate"]["245.1366"]["rows_arg"][1], 42, 1, [(W, "arguments 9° to 107°\n(median 42°)")]),
        ("D 245.14, argument most", J["locate"]["245.1366"]["rows_arg"][2], 107, 1, [(W, "arguments 9° to 107°\n(median 42°)")]),
        ("D 245.14, root tests least", min(J["locate"]["245.1366"]["root_rows"]), 6.1, 0.1, [(W, "root tests of 6.1 to 8.0")]),
        ("D 245.14, root tests most", max(J["locate"]["245.1366"]["root_rows"]), 8.0, 0.1, [(W, "root tests of 6.1 to 8.0")]),
        ("D 245.14, 16 orders equal to the record", J["repro"]["245.1366_N16"]["equal"], 17, 1, [(W, "they equal the record's 17\nnorms")]),
        ("D 236.63 at 16 orders, |s| least", J["locate"]["236.6312"]["rows_abs_16"][0], 5.3, 0.1, [(W, "D 236.63's series gives 5.3 to 6.3 (median\n5.7)")]),
        ("D 236.63 at 16 orders, |s| median", J["locate"]["236.6312"]["rows_abs_16"][1], 5.7, 0.1, [(W, "D 236.63's series gives 5.3 to 6.3 (median\n5.7)")]),
        ("D 236.63 at 16 orders, |s| most", J["locate"]["236.6312"]["rows_abs_16"][2], 6.3, 0.1, [(W, "D 236.63's series gives 5.3 to 6.3 (median\n5.7)")]),
        ("D 236.63 at 16 orders, argument median", J["locate"]["236.6312"]["rows_arg_16"][1], 88, 1, [(W, "at a median of 88°")]),
        ("records past D 245.64, root test to D 251.89, least", min(r["radius_12"] for r in B if r["D0"] < 251.9), 7.2, 0.1,
         [(W, "The root test over orders 12 to 24 stays 7.2 to 8.5 to D 251.89")]),
        ("records past D 245.64, root test to D 251.89, most", max(r["radius_12"] for r in B if r["D0"] < 251.9), 8.5, 0.1,
         [(W, "The root test over orders 12 to 24 stays 7.2 to 8.5 to D 251.89")]),
        ("root test at D 252.25", Bd[252.25]["radius_12"], 6.5, 0.1, [(W, "to 6.5 at D 252.25")]),
        ("root test at D 252.61", Bd[252.61]["radius_12"], 6.0, 0.1, [(W, "6.0 at D 252.61 and 5.3 to 5.6 from D 253.07 to 253.53")]),
        ("root test from D 253.07, least", min(r["radius_12"] for r in B if r["D0"] > 253), 5.3, 0.1,
         [(W, "6.0 at D 252.61 and 5.3 to 5.6 from D 253.07 to 253.53")]),
        ("root test from D 253.07, most", max(r["radius_12"] for r in B if r["D0"] > 253), 5.6, 0.1,
         [(W, "6.0 at D 252.61 and 5.3 to 5.6 from D 253.07 to 253.53")]),
        ("|z_4| at D 245.64", m(Bd[245.64]["z"][3], -11), 1.9, 0.1, [(W, "|z_4| grows from 1.9×10⁻¹¹ to 4.8×10⁻⁷")]),
        ("|z_4| at D 253.53", m(Bd[253.53]["z"][3], -7), 4.8, 0.1, [(W, "|z_4| grows from 1.9×10⁻¹¹ to 4.8×10⁻⁷")]),
        ("|z_3| at D 245.64", m(Bd[245.64]["z"][2], -9), 4.0, 0.1, [(W, "|z_3| from 4.0×10⁻⁹ to 2.1×10⁻⁶")]),
        ("|z_3| at D 253.53", m(Bd[253.53]["z"][2], -6), 2.1, 0.1, [(W, "|z_3| from 4.0×10⁻⁹ to 2.1×10⁻⁶")]),
        ("|z_2| at D 253.53", m(Bd[253.53]["z"][1], -6), 4.9, 0.1, [(W, "|z_2| from 1.1×10⁻⁶ to\n4.9×10⁻⁶")]),
        ("accepted steps from D 253.07, least", min(x for r in B if r["D0"] > 253 for x in r["a_accepted"]), 0.11, 0.01,
         [(W, "the accepted steps fall to 0.11 to 0.18")]),
        ("accepted steps from D 253.07, most", max(x for r in B if r["D0"] > 253 for x in r["a_accepted"]), 0.18, 0.01,
         [(W, "the accepted steps fall to 0.11 to 0.18")]),
        ("polish at D 245.14, start", m(J["polish"]["245.1366"]["F_start"], -67), 2.0, 0.1,
         [(W, "from max|F| 2.0×10⁻⁶⁷ to\n  7.9×10⁻⁹³ in two Newton steps, a move of 9.3×10⁻⁵³")]),
        ("polish at D 245.14, end", m(J["polish"]["245.1366"]["F_final"], -93), 7.9, 0.1,
         [(W, "from max|F| 2.0×10⁻⁶⁷ to\n  7.9×10⁻⁹³ in two Newton steps, a move of 9.3×10⁻⁵³")]),
        ("polish at D 245.14, Newton steps", J["polish"]["245.1366"]["n_steps"], 2, 1, [(W, "7.9×10⁻⁹³ in two Newton steps")]),
        ("polish at D 245.14, first move", m(J["polish"]["245.1366"]["moves"][0], -53), 9.3, 0.1, [(W, "a move of 9.3×10⁻⁵³")]),
        ("polish at D 245.14, change of z_5..z_8", m(v["polish"]["rel_max_5_8"], -31), 1.0, 0.1,
         [(W, "Its z_5 to z_8 differ from the stored base's by 1.0×10⁻³¹ at\n  most")]),
        ("Delta's root test at D 236.63", L["delta"]["root_radius"], 7.4, 0.1, [(W, "Δ's own series has no stable pole within 50, and its root test gives 7.4")]),
    ]
    return items


@check
def lam_gauge_table():
    """notes/MODE_COUNT.md section 10: lam_g(D) from hp/data/lam_gauge.csv (hp/lam_gauge.py table), the D = 4 check
    against Reiterer and Trubowitz's constants (as solver/rt/mu_check.py holds them), the identity's three D (the CSV's
    lam_RT), the same-D spreads, the mu minimum, and the reference fit's residuals and the superseded I461 states'
    offsets (hp/lam_gauge_fit.py; the slow fit selection is full_lam_gauge)."""
    import csv
    import mpmath as mp
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    dps = mp.mp.dps
    import lam_gauge_fit as LF
    mp.mp.dps = dps
    N = "notes/MODE_COUNT.md"
    with mp.workdps(60):
        rows = list(csv.DictReader(x for x in open(os.path.join(ROOT, "hp/data/lam_gauge.csv")) if not x.startswith("#")))
        bb = [r for r in rows if r["role"] == "backbone"]
        at = lambda D: [r for r in bb if abs(float(r["D"]) - D) < 1e-6][0]
        src = open(os.path.join(ROOT, "solver/rt/mu_check.py")).read()
        mu_rt = mp.mpf(re.search(r'MU_RT = mp.mpf\("([0-9.]+)"\)', src).group(1))
        k_rt = mp.mpf(re.search(r'K_RT = mp.mpf\("([0-9.]+)"\)', src).group(1))
        l4 = mp.mpf(at(4.0)["lam_g"])
        d4 = float(l4 - 4 * mp.pi * mu_rt / (2 * k_rt))
        ident = []
        for D in (15.0, 20.043653, 40.020065):
            r = [x for x in rows if x["lam_RT"] and abs(float(x["D"]) - D) < 1e-6][0]
            ident.append(float(abs(mp.mpf(r["lam_g"]) - mp.mpf(r["lam_RT"])) / mp.mpf(r["lam_g"])))
        top = bb[-1]
        ltop = mp.mpf(top["lam_g"])
        ntop = mp.mpf(top["D_str"]) - 2
        mumin = min(bb, key=lambda r: mp.mpf(r["mu"]))
        groups = {}
        for r in rows:
            groups.setdefault(round(float(r["D"]), 6), []).append(r)
        spread = lambda g: float(max(mp.mpf(r["lam_g"]) for r in g) - min(mp.mpf(r["lam_g"]) for r in g))
        multi = [g for g in groups.values() if len(g) > 1]
        chain = max(spread(g) for g in multi if {r["kind"] for r in g} == {"chain"})
        pinned = max(spread(g) for g in multi if {r["kind"] for r in g} == {"pinned"})
        m4 = {r["M"]: mp.mpf(r["lam_g"]) for r in rows if "d4hm15_D4.0000" in r["state"]}
    with mp.workdps(40):
        pts = LF.backbone()
        ts = [1 / (D - 2) for D, _ in pts]
        c, _ = LF.lsq(ts, [y for _, y in pts], LF.REF_DEG, fixed0=mp.mpf(1))
        res = [(float(D), float(y - mp.polyval(c[::-1], t))) for (D, y), t in zip(pts, ts)]
        sup = [(float(D), float(e)) for D, e, _ in LF.superseded(c)]
    rms = lambda v: (sum(x * x for x in v) / len(v)) ** 0.5
    low = [rms([e for D, e in res if lo <= D < hi]) for lo, hi in LF.BANDS if hi <= 226]
    high = rms([e for D, e in res if D >= 226])
    worst = max(res, key=lambda x: abs(x[1]))
    early = max(abs(e) for D, e in sup if D <= 238.76)
    late = [e for D, e in sup if D > 238.76]
    t_res = [(N, "have an rms of 1.7×10⁻¹⁴ to"), (N, "6.8×10⁻¹⁴ in every band of D below 226, and 2.1×10⁻¹³ above it")]
    t_sup = [(N, "are off by −5.1×10⁻¹² to 2.5×10⁻¹⁰"), (N, "those up to D = 238.76 are within 1.2×10⁻¹²")]
    return [("lam_g at D 4", float(l4), 0.6138552748, 1e-10, [(N, "rises from 0.6138552748 at"), ("README.md", "Martín-García give[^9], 0.6138552748")]),
            ("lam_g(4) - 4 pi mu_RT/(2 K_RT)", d4, 2.7e-27, 1e-28,
             [(N, "against 4πμ_RT/(2K_RT) | 2.7×10⁻²⁷ |"), (N, "constants to 2.7×10⁻²⁷")]),
            ("identity D 15", ident[0], 2.5e-23, 1e-24, [(N, "| 2.5×10⁻²³ relative |")]),
            ("identity D 20.04", ident[1], 1.7e-21, 1e-22, [(N, "| 1.7×10⁻²¹ relative |")]),
            ("identity D 40.02", ident[2], 3.8e-16, 1e-17, [(N, "| 3.8×10⁻¹⁶ relative |")]),
            ("lam_g at the frontier", float(ltop), 0.97115850, 1e-8, [(N, "D = 4 to 0.97115850 at D = 244.64"), ("README.md", "It rises to 0.97115850 at D = 244.64")]),
            ("the frontier's D", float(top["D"]), 244.64, 0.01, [(N, "D = 4 to 0.97115850 at D = 244.64")]),
            ("n(1 - lam_g) at the frontier", float(ntop * (1 - ltop)), 6.998004, 1e-6, [(N, "| 6.998004 |")]),
            ("states in the table", len(rows), 1510, 1, [(N, "holds 1510 stored states")]),
            ("backbone D", len(bb), 1274, 1, [(N, "at each of 1274 D")]),
            ("D of the mu minimum", float(mumin["D"]), 38.02, 0.01, [(N, "minimum near D = 38.02 (0.12910162")]),
            ("mu at its minimum", float(mp.mpf(mumin["mu"])), 0.12910162, 1e-8, [(N, "minimum near D = 38.02 (0.12910162")]),
            ("chain spread, #603 against I461", chain, 3.8e-13, 1e-14, [(N, "| 14 | 3.8×10⁻¹³ at D = 237.13 |")]),
            ("pinned spread, 5 against 7", pinned, 1.6e-14, 1e-15, [(N, "| 160 | 1.6×10⁻¹⁴ |")]),
            ("D 4, M 160 against 192", float(m4["160"] - m4["192"]), 1.6e-23, 1e-24,
             [(N, "M = 160 against 192 there: 1.6×10⁻²³")]),
            ("reference fit, smallest band rms below 226", min(low), 1.7e-14, 1e-15, t_res),
            ("reference fit, largest band rms below 226", max(low), 6.8e-14, 1e-15, t_res),
            ("reference fit, rms above 226", high, 2.1e-13, 1e-14, t_res),
            ("reference fit, largest residual", abs(worst[1]), 7.5e-13, 1e-14, [(N, "the largest, 7.5×10⁻¹³, at the last state")]),
            ("superseded, up to 238.76", early, 1.2e-12, 1e-13, t_sup),
            ("superseded, first offset past 238.76", late[0], -5.1e-12, 1e-13, t_sup),
            ("superseded, largest offset", max(late), 2.5e-10, 1e-11, t_sup)]


@check
def chain_dM_table():
    """notes/QUAD_DOUBLE.md 13.31: Delta and lam_g of the chain's pinned system at M = 48 against M = 32
    (hp/data/chain_dM.csv, hp/chain_dM.py table; #636), and its plants (hp/chain_dM.py PLANTED)"""
    import csv
    import mpmath as mp
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    N = "notes/QUAD_DOUBLE.md"
    rows = list(csv.DictReader(x for x in open(os.path.join(ROOT, "hp/data/chain_dM.csv")) if not x.startswith("#")))

    def at(role, rule, D, mutate=""):
        return [r for r in rows if r["role"] == role and r["rule"] == rule and r["mutate"] == mutate
                and abs(float(r["D"]) - D) < 1e-3][0]
    with mp.workdps(50):
        p20 = at("planted", "polish", 20.043653)
        # gamma_dd.py --M48's record at D 20.043653 (raw I271-macm3/gamma_dd.jsonl): Delta, Delta_lo, M48 Delta_diff
        known20 = sum(mp.mpf(x) for x in (2.1111402926144796, -2.625407535122099e-17, 1.6728161489881694e-20))  # floats
        off20 = float(mp.mpf(p20["Delta48"]) - known20)
    t80, p80 = at("planted", "tol", 80.0), at("planted", "polish", 80.0)
    src = open(os.path.join(ROOT, "hp/chain_dM.py")).read()
    known80 = float(re.search(r'name="chain80".*?dM=([-0-9.e]+)', src, re.S).group(1))
    t_pl = [(N, "| dd20: `up2_D20.0437`, no cut")]
    items = [("dd20: Delta(M 48) - Delta(M 32)", float(p20["dDelta_stored"]), 1.673e-20, 1e-23, t_pl + [(N, "| Δ(M 48) − Δ(M 32) = +1.673×10⁻²⁰ |")]),
             ("dd20: our Delta(M 48) below the record's", -off20, 4.4e-27, 1e-28, [(N, "| 4.4×10⁻²⁷ below the record's |")]),
             ("chain80, stop at tol: dDelta", float(t80["dDelta_stored"]), -5.647e-25, 1e-28, [(N, "| −2.376×10⁻²⁵ | −5.647×10⁻²⁵ | 10⁻²⁷ | fail |")]),
             ("chain80, stop at tol: max|F| at M 48", float(t80["F48"]), 6.0e-31, 1e-32, [(N, "The corrector stopped at max |F| 6.0×10⁻³¹")]),
             ("chain80, stop at tol: est", float(t80["est48"]), 3.3e-25, 1e-26, [(N, "One more Newton step moved Δ by\n  +3.3×10⁻²⁵ there")]),
             ("chain80, polish: dDelta", float(p80["dDelta_stored"]), -2.308e-25, 1e-28, [(N, "| −2.376×10⁻²⁵ | −2.308×10⁻²⁵ | 10⁻²⁷ | fail |"), (N, "| −2.309×10⁻²⁵ | −2.308×10⁻²⁵ | 10⁻²⁷ | pass |")]),
             ("chain80, polish: est", float(p80["est48"]), -2.6e-58, 1e-59, [(N, "it is −2.6×10⁻⁵⁸ at D 80")]),
             ("chain80: the plant's value on this build", known80, -2.309e-25, 1e-28, [(N, "Its M 48 difference is −2.309×10⁻²⁵ here")])]
    # the chain table: (role, D, the row's first cell, max|F| at M 32 and M 48, est, dDelta, dlam, their units)
    table = [("planted", 80.0, "| 80 (the plant) | `T138-win2/chain` |", 4.6e-52, 8.1e-52, -2.6e-58, -2.340e-25, 3.925e-26, 1e-29),
             ("chain", 100.0, "| 100 | `T138-win2/chain` |", 3.2e-49, 3.1e-49, 6.8e-58, 2.874e-25, 2.337e-26, 1e-29),
             ("chain", 150.0, "| 150 | `T138-win2/chain` |", 2.8e-41, 2.1e-41, 5.2e-51, 3.106e-25, 3.426e-27, 1e-30),
             ("chain", 200.0, "| 200, at the floor | `T138-win2/chain_o` |", 9.8e-29, 9.8e-29, -5.4e-21, -2.235e-19, 2.748e-22, 1e-25)]
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

    def u(x, d):
        m, e = f"{x:.{d}e}".split("e")
        return f"{m.replace('-', '−')}×10{str(int(e)).translate(sup)}"
    for role, D, head, f32, f48, est, dd, dl, ul in table:
        r = at(role, "polish", D)
        cells = (f"{head} {u(f32, 1)} / {u(f48, 1)} | {'+' if est > 0 else ''}{u(est, 1)} | {'+' if dd > 0 else ''}{u(dd, 3)} | "
                 f"{'+' if dl > 0 else ''}{u(dl, 3)} |")
        txt = [(N, cells)]
        items += [(f"D {D}: max|F| at M 32", float(r["F32"]), f32, abs(f32) / 10, txt),
                  (f"D {D}: max|F| at M 48", float(r["F48"]), f48, abs(f48) / 10, txt),
                  (f"D {D}: est at M 48", float(r["est48"]), est, abs(est) / 10, txt),
                  (f"D {D}: Delta(M 48) - Delta(M 32)", float(r["dDelta"]), dd, abs(dd) / 1e3, txt),
                  (f"D {D}: lam_g(M 48) - lam_g(M 32)", float(r["dlam"]), dl, ul, txt)]
    da = [r for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv"))) if r["kind"] == "chain_fit"]
    de = [float(r["err"]) for r in da if r["err"]]
    t_err = [(N, "all 190 chain_fit rows (1.4×10⁻¹⁵ to 4.4×10⁻¹¹), so every row\nkeeps its err")]
    items += [("chain_fit rows with an err", len(de), 190, 1, t_err), ("smallest chain_fit err", min(de), 1.4e-15, 1e-16, t_err),
              ("largest chain_fit err", max(de), 4.4e-11, 1e-12, t_err),
              ("chain_fit rows without an err", len(da) - len(de), 0, 1, t_err)]
    with mp.workdps(50):
        offD = float(mp.mpf(at("controls", "polish", 20.043653, "D")["Delta48"]) - known20)
    items.append(("dd20 under --mutate=D: Delta(M 48) - the record's", offD, -4.307e-15, 1e-18,
                  [(N, "lies −4.307×10⁻¹⁵ from the record's, the plant fails")]))
    ramp = {r["eps"]: r for r in csv.DictReader(x for x in open(os.path.join(ROOT, "hp/data/chain_dM_ramp.csv"))
                                                if not x.startswith("#"))}
    move = float(mp.mpf(ramp["1e-12"]["c0_highest"]) - mp.mpf(ramp["0"]["c0_highest"]))
    t_ramp = [(N, "moves the top of their c₀ range by\n3.0×10⁻¹²")]
    items += [("ramp of 1e-12: the free c0's top moves", move, 3.0e-12, 1e-13, t_ramp),
              ("ramp 0: 1 - the free c0's top", 1 - float(ramp["0"]["c0_highest"]), 8.6e-11, 1e-12,
               [(N, "put c₀ below 1 by 8.6×10⁻¹¹ (notes/MODE_COUNT.md 10.5)")])]
    r240 = at("chain", "floor", 240.1244140625)
    t240 = [(N, "| 240.12, at the floor, against the stored state | `I603-win2/chain` | octo-double / 8.9×10⁻²⁸ | +7.1×10⁻²¹ | "
                "−1.602×10⁻²¹ | −4.697×10⁻²⁵ |")]
    items += [("D 240: max|F| at M 48", float(r240["F48"]), 8.9e-28, 1e-29, t240 + [(N, "max |F| 8.9×10⁻²⁸, with the LU's")]),
              ("D 240: est at M 48", float(r240["est48"]), 7.1e-21, 1e-22, t240),
              ("D 240: Delta(M 48) - Delta(stored)", float(r240["dDelta_stored"]), -1.602e-21, 1e-24, t240),
              ("D 240: lam_g(M 48) - lam_g(stored)", float(r240["dlam_stored"]), -4.697e-25, 1e-28, t240),
              ("D 240: M 48 start", float(r240["F48_start"]), 1.4e24, 1e23, [(N, "start from 1.4×10²⁴ to 5.0×10⁻¹⁹")]),
              ("D 240: M 48 start re-seeded", float(r240["F48_reseeded"]), 5.0e-19, 1e-20, [(N, "start from 1.4×10²⁴ to 5.0×10⁻¹⁹")])]
    r200 = at("chain", "polish", 200.0)
    items += [("D 200: the quad-double path's move at M 32", float(r200["M32_moved"]), 2.61e-19, 1e-21,
               [(N, "+2.61×10⁻¹⁹ from the stored state")]),
              ("D 200: M 48 against the stored state", float(r200["dDelta_stored"]), 3.74e-20, 1e-22,
               [(N, "solution lies +3.74×10⁻²⁰ from the stored state")]),
              ("D 200: M 48 start", float(r200["F48_start"]), 2.3e15, 1e14, [(N, "start from 2.3×10¹⁵ to 1.0×10⁻¹⁸")]),
              ("D 200: M 48 start re-seeded", float(r200["F48_reseeded"]), 1.0e-18, 1e-19, [(N, "start from 2.3×10¹⁵ to 1.0×10⁻¹⁸")])]
    t_tail = [(N, "the M = 32 states is 4.4×10⁻¹⁴ at D 80, 1.4×10⁻¹⁴ at D 100 and 2.1×10⁻¹⁵ at D 150")]
    items += [("tau tail k >= 28, D 80", float(p80["tail32_k28"]), 4.4e-14, 1e-15, t_tail),
              ("tau tail k >= 28, D 100", float(at("chain", "polish", 100.0)["tail32_k28"]), 1.4e-14, 1e-15, t_tail),
              ("tau tail k >= 28, D 150", float(at("chain", "polish", 150.0)["tail32_k28"]), 2.1e-15, 1e-16, t_tail),
              ("tau tail k >= 28, D 20", float(p20["tail32_k28"]), 7.1e-10, 1e-11, [(N, "dd20 plant) the tail is 7.1×10⁻¹⁰, and Δ moves only 1.7×10⁻²⁰")])]
    # D 61.026449: lam_g of the kfreeze-23 states at M 48 and M 64 (I252-mac), hp/data/lam_gauge.csv
    lg = list(csv.DictReader(x for x in open(os.path.join(ROOT, "hp/data/lam_gauge.csv")) if not x.startswith("#")))
    with mp.workdps(50):
        k23 = {r["M"]: mp.mpf(r["lam_g"]) for r in lg if "I252-mac/states/modecount_state_D61.0264_M" in r["state"]
               and r["state"].endswith(("M48_kf23_from_I212.npz", "M64_kf23.npz"))}
        d23 = float(abs(k23["48"] - k23["64"]))
    items.append(("D 61.03: kfreeze-23 lam_g, M 48 against M 64", d23, 6.5e-18, 1e-19, [(N, "λ_g moves 6.5×10⁻¹⁸")]))
    return items


def full_lam_gauge():
    """notes/MODE_COUNT.md section 10.4: the fit selection on the real backbone and the planted test's T1, T6 and its
    control (hp/lam_gauge_fit.py; about a minute)."""
    import mpmath as mp
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    dps = mp.mp.dps
    import lam_gauge_fit as LF
    mp.mp.dps = dps
    N = "notes/MODE_COUNT.md"
    with mp.workdps(40):
        pts = LF.backbone()
        grid = [D for D, _ in pts]
        free, con = LF.fits(pts), LF.fits(pts, fixed0=mp.mpf(1))
        fs = {k: [float(x) for x in LF.summary(free, k)] for k in (0, 1, 2)}
        cs = {k: [float(x) for x in LF.summary(con, k)] for k in (1, 2, 3)}
        nf, nc = sum(f["sel"] for f in free), sum(f["sel"] for f in con)
        t1 = LF.planted_points("T1", grid)
        p_free, p_con = LF.fits(t1), LF.fits(t1, fixed0=mp.mpf(1))
        p0, p1 = LF.summary(p_free, 0), LF.summary(p_con, 1)
        q1 = LF.summary(LF.fits(LF.planted_points("T1", grid, sigma=5e-14), fixed0=mp.mpf(1)), 1)
        t6 = LF.summary(LF.fits(LF.planted_points("T6", grid)), 0)
        mu = LF.summary(LF.fits(LF.planted_points("T1", grid, c0=1 + mp.mpf("1e-5"))), 0)
    half = lambda v: float((v[2] - v[1]) / 2)
    t_c0 = [(N, "| c₀ | 0.99999999941 ± 9.2×10⁻⁸ | 1 |")]
    t_rng = [(N, "give c₀ from 0.99999981663 to 0.999999999914"), (N, "so 1 lies 8.6×10⁻¹¹ above the range")]
    t_con = [(N, "| c₁ | −7.41656 ± 1.6×10⁻⁴ | −7.4165644 ± 4.4×10⁻⁶ |"), (N, "| c₂ | 107.51 ± 0.12 | 107.5086 ± 0.0069 |"),
             (N, "| c₃ | | −1530 ± 5 |"), (N, "\\lambda_g = 1 - \\frac{7.4165644}{n} + \\frac{107.5086}{n^2} - \\frac{1530}{n^3}"),
             ("README.md", "tends to 1 as 1 − 7.4165644/n + 107.5086/n² − 1530/n³")]
    return [("free c0, median", fs[0][0], 0.99999999941, 1e-11, t_c0),
            ("free c0, half-width", (fs[0][2] - fs[0][1]) / 2, 9.2e-8, 1e-9, t_c0 + [(N, "c₀ = 1 to 9.2×10⁻⁸")]),
            ("free c0, lowest", fs[0][1], 0.99999981663, 1e-11, t_rng),
            ("free c0, highest", fs[0][2], 0.999999999914, 1e-12, t_rng),
            ("1 - highest free c0", 1 - fs[0][2], 8.6e-11, 1e-12, t_rng),
            ("free fits kept", nf, 36, 1, [(N, "| c₀ free (36 fits) | c₀ = 1 (31 fits) |")]),
            ("c0 = 1 fits kept", nc, 31, 1, [(N, "| c₀ free (36 fits) | c₀ = 1 (31 fits) |")]),
            ("free c1", fs[1][0], -7.41656, 1e-5, t_con), ("free c1 half-width", (fs[1][2] - fs[1][1]) / 2, 1.6e-4, 1e-5, t_con),
            ("free c2", fs[2][0], 107.51, 0.01, t_con), ("free c2 half-width", (fs[2][2] - fs[2][1]) / 2, 0.12, 0.01, t_con),
            ("c1 at c0 = 1", cs[1][0], -7.4165644, 1e-7, t_con),
            ("c1 half-width", (cs[1][2] - cs[1][1]) / 2, 4.4e-6, 1e-7, t_con + [(N, "fit choices of 4.4×10⁻⁶, 0.0069 and 5")]),
            ("c2 at c0 = 1", cs[2][0], 107.5086, 1e-4, t_con),
            ("c2 half-width", (cs[2][2] - cs[2][1]) / 2, 0.0069, 1e-4, t_con),
            ("c3 at c0 = 1", cs[3][0], -1530, 10, t_con),
            ("c3 half-width", (cs[3][2] - cs[3][1]) / 2, 5, 1, t_con),
            ("T1 free c0 half-width", half(p0), 2.0e-7, 1e-8, [(N, "| 1 ± 2.0×10⁻⁷, holds 1 |")]),
            ("T1 c1 half-width", half(p1), 6.8e-6, 1e-7, [(N, "| −7.4 ± 6.8×10⁻⁶, holds −7.4 |")]),
            ("T1 c1 half-width, sigma 5e-14", half(q1), 3.4e-7, 1e-8, [(N, "| −7.4 ± 3.4×10⁻⁷, holds −7.4 |")]),
            ("T6 free c0", float(t6[0]), 1.000021, 1e-6, [(N, "| 1.000021 ± 1.6×10⁻⁵, 1 outside |")]),
            ("T6 free c0 half-width", half(t6), 1.6e-5, 1e-6, [(N, "| 1.000021 ± 1.6×10⁻⁵, 1 outside |")]),
            ("control: lowest c0", float(mu[1]), 1.0000100, 1e-7, [(N, "| 1.0000100 to 1.0000104, 1 outside")]),
            ("control: highest c0", float(mu[2]), 1.0000104, 1e-7, [(N, "| 1.0000100 to 1.0000104, 1 outside")])]


@check
def sigma_ms():
    """matter/SIGMA_DSS.md section 11: the sigma-model's Delta(eta) at D = 4 on the multishoot kernel (#639), from
    matter/sigma_ms/data/delta_eta.csv (tables.py), fit_min.json (fit_min.py) and planted.csv (the planted check)"""
    S = "matter/SIGMA_DSS.md"
    d = os.path.join(ROOT, "matter/sigma_ms/data")
    import csv as _csv
    rows = {r["eta"]: r for r in _csv.DictReader(open(os.path.join(d, "delta_eta.csv")))}
    pl = {r["mutate"]: r for r in _csv.DictReader(open(os.path.join(d, "planted.csv"))) if r["M"] == "8"}
    fit = json.load(open(os.path.join(d, "fit_min.json")))
    sup = lambda x: x.replace("-", "−").translate(str.maketrans("0123456789−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))  # noqa: E731
    sci = lambda v, p: (lambda m, e: f"{'−' if v < 0 else ''}{abs(float(m)):.{p}f}×10{sup(str(int(e)))}")(*f"{v:.{p}e}".split("e"))  # noqa: E731
    out = []
    table = {"5": "5", "6": "6", "7": "7", "8.5": "8.5", "10": "10", "14": "14", "20": "20", "inf": "∞"}
    for e, lab in table.items():
        r = rows[e]
        d32, d48, k39 = float(r["Delta_M32"]), float(r["Delta_M48"]), float(r["Delta_kc39"])
        dm, dk, kk = float(r["M48_minus_M32"]), float(r["best_minus_kc39"]), float(r["kc39_minus_kc23"])
        line = (f"| {lab} | {d32:.11f} | {d48:.11f} | {sci(dm, 2)} | {k39:.10f} | {sci(dk, 1)} | {sci(kk, 2)} |")
        t = [(S, line)]
        out += [(f"Delta({lab}), M 32", d32, float(f"{d32:.11f}"), 1e-11, t),
                (f"Delta({lab}), M 48", d48, float(f"{d48:.11f}"), 1e-11, t),
                (f"Delta({lab}), M 48 - 32", dm, float(f"{dm:.2e}"), 10 ** (math.floor(math.log10(abs(dm))) - 2), t),
                (f"Delta({lab}), sigdss k_c 39", k39, float(f"{k39:.10f}"), 1e-10, t),
                (f"Delta({lab}), M 48 - k_c 39", dk, float(f"{dk:.1e}"), 10 ** (math.floor(math.log10(abs(dk))) - 1), t),
                (f"Delta({lab}), k_c 39 - 23", kk, float(f"{kk:.2e}"), 10 ** (math.floor(math.log10(abs(kk))) - 2), t)]
    m64 = {e: float(rows[e]["M64_minus_M48"]) for e in rows if rows[e].get("M64_minus_M48")}
    t64 = [(S, f"M = 48 → 64 changes it by {sci(m64['inf'], 1)} at η = ∞ and {sci(m64['7'], 1)} at η = 7")]
    out += [("M 48 -> 64 at inf", m64["inf"], 2.5e-11, 0.1e-11, t64), ("M 48 -> 64 at 7", m64["7"], 9.5e-12, 0.1e-12, t64)]
    # the ratio of the M changes at eta = 7 and inf, and the fit's shift scaled by it (an estimate, as the notes say)
    rat = {e: m64[e] / float(rows[e]["M48_minus_M32"]) for e in ("7", "inf")}
    tr = [(S, "is −4.3×10⁻⁴ times the M = 32 → 48 one at both η = 7 and"),
          (S, "−7×10⁻⁷ in η_min and +9×10⁻¹² in Δ_min")]
    out += [("M change ratio at 7", rat["7"], -4.3e-4, 0.1e-4, tr), ("M change ratio at inf", rat["inf"], -4.3e-4, 0.1e-4, tr),
            ("projected eta_min shift", rat["7"] * fit["result"]["resolution_eta"], -7e-7, 1e-7, tr),
            ("projected Delta_min shift", rat["7"] * fit["result"]["resolution_Delta"], 9e-12, 1e-12, tr)]
    n, g, sh, dc, qs = pl["none"], pl["gamma"], pl["sigma_h"], pl["dropcol"], pl["qd_state"]
    tp = [(S, "to max |F| = 6.8×10⁻³²"), (S, "Newton reaches |F| = 6.5×10⁻³² in 7 steps, with max |z − z*| = 3.2×10⁻³⁰"),
          (S, "Δ − Δ* = 1.6×10⁻³⁰. Each of four controls")]
    out += [("planted max |F(z*)|", float(n["F_star"]), 6.8e-32, 0.1e-32, tp),
            ("planted Newton |F|", float(n["dd_F_last"]), 6.5e-32, 0.1e-32, tp),
            ("planted Newton steps", int(n["dd_newton_steps"]), 7, 1, tp),
            ("planted max |z - z*|", float(n["dd_err"]), 3.2e-30, 0.1e-30, tp),
            ("planted Delta - Delta*", float(n["dd_dDelta"]), 1.6e-30, 0.1e-30, tp),
            ("control gamma", float(g["dd_dDelta"]), -6.2e-12, 0.1e-12, [(S, "| Δ − Δ* = −6.2×10⁻¹² |")]),
            ("control sigma_h", float(sh["dd_dDelta"]), 1.3e-10, 0.1e-10, [(S, "| Δ − Δ* = 1.3×10⁻¹⁰ |")]),
            ("control dropcol", float(dc["dd_F_last"]), 1.65e-28, 0.01e-28, [(S, "|F| = 1.65×10⁻²⁸ after 10 steps")]),
            ("control qd_state", float(qs["qd_F_last"]), 5.8e-33, 0.1e-33, [(S, "|F| stops at 5.8×10⁻³³")]),
            ("planted quad-double |F|", float(n["qd_F_last"]), 1.2e-59, 0.1e-59, [(S, "reaches |F| = 1.2×10⁻⁵⁹")]),
            ("planted quad-double minus double-double", abs(float(n["qd_dDelta_dd"])), 4.1e-30, 0.1e-30,
             [(S, "within 4.1×10⁻³⁰ of the")])]
    p16 = [r for r in _csv.DictReader(open(os.path.join(d, "planted.csv"))) if r["M"] == "16" and r["mutate"] == "none"][0]
    t16 = [(S, "7 Newton steps, max |z − z*| = 2.3×10⁻²⁹, Δ − Δ* = −1.1×10⁻²⁹")]
    out += [("planted M 16 steps", int(p16["dd_newton_steps"]), 7, 1, t16),
            ("planted M 16 max |z - z*|", float(p16["dd_err"]), 2.3e-29, 0.1e-29, t16),
            ("planted M 16 Delta - Delta*", float(p16["dd_dDelta"]), -1.1e-29, 0.1e-29, t16)]
    f32, f48, res = fit["32"]["summary"], fit["48"]["summary"], fit["result"]
    tf = [(S, "| η_min | 6.97851 | 6.98019 | 6.9801938 to 6.9801973 |"),
          (S, "| Δ_min | 0.45980938351 | 0.45980936174 | within 8×10⁻¹² |"),
          ("README.md", "minimum at η = 6.98019, with Δ = 0.45980936174"),
          ("notes/REVIEW_GHMG25.md", "minimum at η = 6.98019, Δ_min = 0.45980936174")]
    out += [("eta_min, M 32", f32["eta_min"], 6.97851, 1e-5, tf), ("eta_min, M 48", f48["eta_min"], 6.98019, 1e-5, tf),
            ("eta_min spread low", f48["spread_eta"][0], 6.9801938, 1e-7, tf),
            ("eta_min spread high", f48["spread_eta"][1], 6.9801973, 1e-7, tf),
            ("Delta_min, M 32", float(f32["Delta_min"]), 0.45980938351, 1e-11, tf),
            ("Delta_min, M 48", float(f48["Delta_min"]), 0.45980936174, 1e-11, tf),
            ("Delta_min spread", max(abs(x) for x in f48["spread_Delta"]), 8e-12, 1e-12, tf),
            ("fit residual, M 48", f48["residual"], 1.9e-16, 0.1e-16, [(S, "largest residual is 1.9×10⁻¹⁶")]),
            ("eta_min, M 48 - 32", res["resolution_eta"], 1.7e-3, 0.1e-3, [(S, "1.7×10⁻³ in η_min and −2.2×10⁻⁸ in Δ_min")]),
            ("Delta_min, M 48 - 32", res["resolution_Delta"], -2.2e-8, 0.1e-8, [(S, "1.7×10⁻³ in η_min and −2.2×10⁻⁸ in Δ_min")])]
    return out


@check
def chain_frontier():
    """the README's reach of the large-D chain is the highest chain_fit D of hp/data/delta_all.csv (#653: chain_reach
    takes the row nearest 245.6, so it would still pass after the chain moved on)"""
    import csv
    top = max(float(r["D"]) for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv")))
              if r["kind"] == "chain_fit")
    txt = [("README.md", "harmonics 5–13 of the centre field on a fit, to D = 245.6.")]
    return [("the chain's highest D", top, 245.6, 0.1, txt)]


@check
def pinned_frontier():
    """the pinned rows of hp/data/delta_all.csv from D 73.1 to the frontier 73.265722 (#756, notes/CONSISTENCY_AUDIT.md
    P2-2): each has an err; the quad-double Delta_full1 is the row at 73.1 and at the frontier (rule (a) of
    hp/export_delta_all.py), whose pins differ by 5.0e-16 there; the double-double rows between carry the dd-qd gap
    (rule (b)), which notes/QUAD_DOUBLE.md 12.4 quotes at the frontier"""
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv")))
            if r["kind"] == "pinned" and float(r["D"]) >= 73.1 - 1e-9]
    qd = [r for r in rows if r["Delta_q2"]]
    gap = max((float(r["err"]) for r in rows if r["err_kind"] == "dd-qd gap"), default=0.0)
    front = max(rows, key=lambda r: float(r["D"]))
    return [("pinned rows at D >= 73.1 without an err", sum(not r["err"] for r in rows), 0, 1, []),
            ("quad-double pinned rows at D >= 73.1", len(qd), 2, 1, []),
            ("the frontier row's quad-double pins' spread", float(front["err"] or 0) if front["Delta_q2"] else 0.0,
             5.0e-16, 0.1e-16, []),
            ("the dd-qd gap of the rows below the frontier", gap, 1.1e-14, 0.1e-14,
             [("notes/QUAD_DOUBLE.md", "1.06e-14 and 1.11e-14 above them")])]


@check
def table_errs():
    """every pinned and chain_fit row of hp/data/delta_all.csv has an err or a reason named in its err_kind (#759).
    The pinned rows that one pin reached take the larger of the dd-qd gap and the neighbouring waypoints' pins spread
    (rules (b) and (c) of hp/export_delta_all.py). Where T75's pin 7 meets T73's pin 5 (70.65 to 70.95), the row is
    pin 7's with the pins' spread. The backbone's half steps 173.5 and 190.5 take the comparison fits interpolated
    there"""
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv"))))
    bare = lambda kind: sum(r["kind"] == kind and not r["err"] and not r["err_kind"] for r in rows)
    cross = [r for r in rows if r["source"].startswith("T75/pin7.jsonl:") and r["err_kind"] == "pins"
             and float(r["D"]) < 71.0]
    half = [r for r in rows if r["kind"] == "chain_fit" and float(r["D"]) in (173.5, 190.5)
            and r["err_kind"].endswith(" interp")]
    return [("pinned rows without an err or a named reason", bare("pinned"), 0, 1, []),
            ("chain_fit rows without an err or a named reason", bare("chain_fit"), 0, 1, []),
            ("pin 7 rows with the pins' spread where T75's pin 7 meets T73's pin 5", len(cross), 7, 1, []),
            ("half-step chain_fit rows with the fits interpolated", len(half), 2, 1, [])]


@check
def table_errs_all():
    """every row of hp/data/delta_all.csv has an err or a reason named in its err_kind (#764). The full-system
    quad-double rows with one start (71.0, 72.8) take the larger two-start spread of the nearest qd rows below and
    above. The low-D rows solved at one M (3.72-3.8 at M 80, 4.25-8.2 at M 48) take that M's departure from the best
    value at the nearest D below and above with a larger M, below only from 8.059375 on (hp/export_lowD.py).
    Delta(7)'s err stays below half the last digit the README quotes"""
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv"))))
    at = {(r["kind"], float(r["D"])): r for r in rows}
    err = lambda kind, D: float(at[(kind, D)]["err"] or "nan")  # noqa: E731
    kinds = lambda ek: [r for r in rows if r["err_kind"] == ek]  # noqa: E731
    txt7 = [("README.md", "| 7 | 2.84275320647 | 0.439991242 |"),
            ("notes/LOW_D.md", "unchecked, to 10⁻¹¹ by its neighbours")]
    return [("rows of any kind without an err or a named reason", sum(not r["err"] and not r["err_kind"] for r in rows),
             0, 1, []),
            ("qd rows with the neighbouring starts' spread", len(kinds("starts neighbours")), 2, 1, []),
            ("low-D rows with the neighbouring dM", len(kinds("dM neighbours")), 21, 1, []),
            ("low-D rows with the dM below only", len(kinds("dM neighbours one-sided")), 3, 1, []),
            ("qd err at 71.0 (starts neighbours)", err("qd", 71.0), 3.8e-28, 0.1e-28, []),
            ("qd err at 72.8 (starts neighbours)", err("qd", 72.8), 8.4e-28, 0.1e-28, []),
            ("M 80 err at 3.72 (dM neighbours)", err("dd", 3.72), 1.3e-9, 0.1e-9, []),
            ("M 48 err at 4.25 (dM neighbours)", err("dd", 4.25), 2.6e-7, 0.1e-7, []),
            ("M 48 err at 7 (dM neighbours)", err("dd", 7.0), 3.0e-14, 0.1e-14, txt7),
            ("M 48 err at 8.2 (dM neighbours one-sided)", err("dd", 8.2), 9.5e-21, 0.1e-21, []),
            ("Delta(7)'s err below half the README's last digit", int(err("dd", 7.0) < 0.5e-11), 1, 1, txt7)]


@check
def pinned_qd_table():
    """notes/QUAD_DOUBLE.md 12.4: the quad-double Delta_full1 of the pinned solutions against
    hp/data/delta_pinned_qd.csv (hp/export_pinned_qd.py, the exact sums of T84-mac's *_fullcorr.jsonl records; #758).
    The section's two tables are read from the note: each Delta_full1 and T75's double-double value to its printed
    digits, each difference, |F|, correction and |dz| to its printed precision. Then the figures of the text"""
    import csv
    import mpmath as mp
    N, A = "notes/QUAD_DOUBLE.md", "notes/CONSISTENCY_AUDIT.md"
    nan = float("nan")
    rows = {(round(float(r["D"]), 4), int(r["pin"])): r
            for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_pinned_qd.csv")))}
    sec = DOCS[N].split("\n### 12.4 ", 1)[1].split("\n### 12.5 ", 1)[0].splitlines()

    def table(head):
        """the cells of each body row of the table whose header starts with head"""
        i = next((k for k, x in enumerate(sec) if x.startswith(head)), len(sec))
        out = []
        for x in sec[i + 2:]:
            if not x.startswith("|"):
                break
            out.append([c.strip() for c in x.strip().strip("|").split("|")])
        return out

    def sci(label, have, cell):
        """a printed number (1.2e-32, -5.3e-29, 2e-10) against the file's, to its last printed digit"""
        m = re.fullmatch(r"[+-]?\d+(?:\.(\d+))?e([+-]?\d+)", cell)
        if not m:
            return (label, float(have or nan), nan, 1.0, [])
        return (label, float(have or nan), float(cell), 10.0 ** (int(m.group(2)) - len(m.group(1) or "")), [])

    def decimal(label, have, cell):
        """a printed decimal of up to 22 digits against the file's, as its offset from its first 12 decimals"""
        if not re.fullmatch(r"\d\.\d{12,}", cell) or not have:
            return (label, nan, nan, 1.0, [])
        with mp.workdps(60):
            base = mp.mpf(cell[:14])
            return (label, float(mp.mpf(have) - base), float(mp.mpf(cell) - base), 10.0 ** (2 - len(cell)), [])
    items, full1, seen = [], 0, set()
    for c in table("| D | pin | full-system |F|"):
        D, pin = float(c[0]), int(c[1].split()[0])
        r = rows.get((D, pin), {})
        seen.add((D, pin))
        tag = f"D {c[0]} pin {pin}"
        items += [sci(f"{tag}: |F| at the pinned solution", r.get("F_full"), c[2]),
                  sci(f"{tag}: correction", r.get("correction"), c[3]),
                  decimal(f"{tag}: Delta_full1", r.get("Delta"), c[4])]
        full1 += 1
        m = re.fullmatch(r"(\d\.\d+) \(pin 7\)(?:, \.\.\.(\d+) \(pin 5\))?", c[5])
        if c[5] and not m:
            items.append((f"{tag}: T75's double-double Delta_full1", nan, nan, 1.0, []))
        elif m:
            items.append(decimal(f"{tag}: T75's pin 7 Delta_full1", r.get("dd_pin7"), m.group(1)))
            if m.group(2):
                p5 = m.group(1)[:-len(m.group(2))] + m.group(2)
                items.append(decimal(f"{tag}: T75's pin 5 Delta_full1", r.get("dd_pin5"), p5))
    for c in table("| D | Delta_full1 (pin 7 and pin 5)"):
        D = float(c[0])
        r7, r5 = rows.get((D, 7), {}), rows.get((D, 5), {})
        seen |= {(D, 7), (D, 5)}
        items += [decimal(f"D {c[0]} pin 7: Delta_full1", r7.get("Delta"), c[1]),
                  decimal(f"D {c[0]} pin 5: Delta_full1", r5.get("Delta"), c[1]),
                  sci(f"D {c[0]}: pin 7 - pin 5", r7.get("minus_other_pin"), c[2])]
        full1 += 2
        q, dz = [x.strip() for x in c[3].split(",")] if c[3] else ["", ""], [x.strip() for x in c[4].split(",")]
        for pin, r, x, z in ((7, r7, q[0], dz[0]), (5, r5, q[-1], dz[-1])):
            items += [sci(f"D {c[0]} pin {pin}: the full step's |dz|", r.get("dz"), z),
                      sci(f"D {c[0]} pin {pin}: minus the full system's", r.get("minus_qd"), x) if x else
                      (f"D {c[0]} pin {pin}: a full-system value the table leaves out", float(bool(r.get("minus_qd"))),
                       0.0, 1.0, [])]
    items += [("Delta_full1 values in the two tables", full1, 18, 1, []),
              ("records of delta_pinned_qd.csv in the tables", len(seen & set(rows)), 16, 1, [])]
    with mp.workdps(60):
        def at(D, pin, col):
            return mp.mpf(rows[(D, pin)][col])
        F = 73.2657
        leg = [r for (D, p), r in rows.items() if 71.0 <= D <= 73.1]
        qd = [r for r in leg if r["minus_qd"]]
        items += [("the frontier's pinned Delta, pin 3 - pin 1", float(at(F, 3, "Delta_pinned") - at(F, 1, "Delta_pinned")),
                   2.2e-15, 0.1e-15, [(N, "The two converged Delta differ by 2.2e-15")]),
                  ("T75's double-double pins at 73.1, 7 - 5", float(abs(at(73.1, 7, "dd_pin7") - at(73.1, 7, "dd_pin5"))),
                   8.6e-15, 0.1e-15, [(N, "2e-16 typically and 8.6e-15 at 73.1")]),
                  ("the leg's smallest pins' spread", min(abs(float(r["minus_other_pin"])) for r in leg), 7.1e-33, 0.1e-33,
                   [(N, "to between 7.1e-33 (71.5)")]),
                  ("the leg's largest pins' spread", max(abs(float(r["minus_other_pin"])) for r in leg), 9.0e-25, 0.1e-25,
                   [(N, "9.0e-25 (73.1) here"), (N, "spread, 9.0e-25 or better")]),
                  ("both pins against the full system", max(abs(float(r["minus_qd"])) for r in qd), 1.8e-26, 0.1e-26,
                   [(N, "both pins lie within 1.8e-26 of"), (N, "to within 1.8e-26 where the full system converged")]),
                  ("pin 7 against the full system", max(abs(float(r["minus_qd"])) for r in qd if r["pin"] == "7"), 8.1e-28,
                   0.1e-28, [(N, "lies within 8.1e-28 of it")]),
                  ("D with a full-system value where pin 7's full step is the smaller",
                   sum(float(r["dz"]) < float(rows[(round(float(r["D"]), 4), 5)]["dz"]) for r in qd if r["pin"] == "7"), 4,
                   1, [(N, "Pin 7, whose full step is the smaller there")])]
        sp = {round(float(r["D"]), 4): r["spread"] for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_qd.csv")))}
        items += [(f"delta_qd.csv's two-start spread at {D}", float(sp[D] or nan), q, 0.1e-28,
                   [(N, "(3.8e-28 at 71.5 and 8.4e-28 at 73.0, a spread between Taylor plans")])
                  for D, q in ((71.5, 3.8e-28), (73.0, 8.4e-28))]
        items += [("71.0: the pins against the full system", max(abs(float(rows[(71.0, p)]["minus_qd"])) for p in (7, 5)),
                   2.0e-28, 0.1e-28, [(N, "lie within 2.0e-28 (71.0) and 3.8e-28 (71.5) of it")]),
                  ("71.5: the pins against the full system", max(abs(float(rows[(71.5, p)]["minus_qd"])) for p in (7, 5)),
                   3.8e-28, 0.1e-28, [(N, "lie within 2.0e-28 (71.0) and 3.8e-28 (71.5) of it")]),
                  ("the frontier's pins' spread", abs(float(rows[(F, 3)]["minus_other_pin"])), 5.0e-16, 0.1e-16,
                   [(N, "5.0e-16 (a spread between two systems, not a bound)")]),
                  ("the frontier's full step |dz|", float(rows[(F, 3)]["dz"]), 7.9e-6, 0.1e-6,
                   [(N, "|dz| is 7.9e-6 against 1.6e-9 at 72.4")]),
                  ("72.4's full step |dz|, pin 7", float(rows[(72.4, 7)]["dz"]), 1.6e-9, 0.1e-9,
                   [(N, "|dz| is 7.9e-6 against 1.6e-9 at 72.4")])]
        t = [(N, "1.06e-14 and 1.11e-14 above them"), (A, "1.06×10⁻¹⁴ and 1.11×10⁻¹⁴ above the two quad-double ones")]
        items += [(f"the frontier's double-double Delta_full1 above pin {p}'s",
                   float(at(F, p, "dd_pin7") - at(F, p, "Delta")), q, 0.01e-14, t) for p, q in ((1, 1.06e-14), (3, 1.11e-14))]
    return items


@check
def profile_table():
    """README "How the solution scales with D" and the physics paper's Table tab:scaling: the profile F and the centre
    lapse on the double-double states nearest D = 20, 30, 40, 50, 60 (hp/data/largeD_F_all.csv, hp/largeD_F_all.py)"""
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "hp/data/largeD_F_all.csv"))))
    out = []
    quoted = {20: ("3.480", "0.130", "0.024", "3.021", "6.162"), 30: ("3.212", "0.105", "0.016", "3.201", "5.218"),
              40: ("3.086", "0.093", "0.012", "3.304", "4.804"), 50: ("3.011", "0.085", "0.0097", "3.372", "4.568"),
              60: ("2.962", "0.080", "0.0085", "3.421", "4.416")}
    cols = ("F1", "F3_over_F1", "F5_over_F1", "n_one_minus_c0", "F2_mean")
    for D, q in quoted.items():
        r = min(rows, key=lambda r: abs(float(r["D"]) - D))
        txt = [("README.md", f"| {D} | " + " | ".join(q) + " |"),
               ("paper/physics/main.tex", f"{D} & " + " & ".join(q) + " \\\\")]
        for c, s in zip(cols, q):
            unit = 10.0 ** -len(s.split(".")[1])
            out.append((f"{c} at D {float(r['D']):.2f}", float(r[c]), float(s), unit, txt))
    return out


@check
def ell_table():
    """README "Scalar multiplet with ℓ = 1" and the physics paper's table: Δ and γ of the ℓ = 1 multiplet at D = 4-9
    (solver/matter/data/l1_kc23r.jsonl for D = 5-9 and l1_kc23_gamma.jsonl for γ(4), k_c = 23; l1_D4_kc39.jsonl for
    Δ(4), k_c = 39; matter/ELL.md's table, solver/matter/ell_table.py)"""
    d = os.path.join(ROOT, "solver/matter/data")
    rows = {}
    for line in open(os.path.join(d, "l1_kc23r.jsonl")):
        r = json.loads(line)
        rows[r["D"]] = r
    g4 = json.loads(open(os.path.join(d, "l1_kc23_gamma.jsonl")).readline())
    d4 = json.loads(open(os.path.join(d, "l1_D4_kc39.jsonl")).readline())
    dq = ("0.4598611", "0.7833439", "0.9115281", "0.9639878", "0.9837905", "0.988205")
    gq = ("0.118900", "0.215509", "0.265565", "0.295733", "0.316203", "0.331286")
    txt = [("README.md", "| Δ | " + " | ".join(dq) + " |"), ("README.md", "| γ | " + " | ".join(gq) + " |"),
           ("paper/physics/main.tex", "$\\Delta$ & " + " & ".join(dq) + " \\\\"),
           ("paper/physics/main.tex", "$\\gamma$ & " + " & ".join(gq) + " \\\\")]
    delta = [d4["Delta"]] + [rows[D]["Delta"] for D in (5.0, 6.0, 7.0, 8.0, 9.0)]
    gamma = [g4["gamma"]] + [rows[D]["gamma"] for D in (5.0, 6.0, 7.0, 8.0, 9.0)]
    out = []
    for D, v, s in zip(range(4, 10), delta, dq):
        out.append((f"l = 1 Delta({D})", v, float(s), 10.0 ** -len(s.split(".")[1]), txt))
    for D, v, s in zip(range(4, 10), gamma, gq):
        out.append((f"l = 1 gamma({D})", v, float(s), 10.0 ** -len(s.split(".")[1]), txt))
    return out


@check
def d4_ladder():
    """README "Relation to published work", D = 4: the double-double value at M = 192 to the 22 decimals quoted (exact
    sum of its two parts) and the changes from M = 80 to 96, 128, 160 and 192 (hp/data/delta_dd_lowD.csv, whose
    records are hp/data/lowD_highM/lift_D4_points.jsonl); the Values table's Δ(4) and γ(4) (γ at M = 80, the largest M
    with a growth rate)"""
    import csv
    from decimal import Decimal, getcontext
    from fractions import Fraction
    getcontext().prec = 40
    rows = {int(r["M"]): r for r in csv.DictReader(l for l in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv"))
                                                   if not l.startswith("#")) if abs(float(r["D"]) - 4.0) < 1e-9}
    dec = lambda x: Decimal(Fraction(float(x)).numerator) / Decimal(Fraction(float(x)).denominator)  # noqa: E731
    v192 = dec(rows[192]["Delta_hi"]) + dec(rows[192]["Delta_lo"])
    q = "3.4454524022278213499715"
    txt = [("README.md", f"Δ =\n  {q} at M = 192 τ points"),
           ("README.md", "are\n  −1.3×10⁻¹¹, −9.0×10⁻¹⁴, −2.8×10⁻¹⁸ and +1.1×10⁻²²")]
    tv = [("README.md", "| 4 | 3.445452402 | 0.37396072 |"), ("paper/physics/main.tex", "4 & 3.445452402 & 0.37396072")]
    ch = {M: float(rows[M]["change_from_lower_M"]) for M in (96, 128, 160, 192)}
    return [("Delta(4) at M 192 minus the quoted 22 decimals", float(v192 - Decimal(q)), 0.0, 1e-22, txt),
            ("change M 80 -> 96", ch[96], -1.3e-11, 0.1e-11, txt),
            ("change M 96 -> 128", ch[128], -9.0e-14, 0.1e-14, txt),
            ("change M 128 -> 160", ch[160], -2.8e-18, 0.1e-18, txt),
            ("change M 160 -> 192", ch[192], 1.1e-22, 0.1e-22, txt),
            ("Values table Delta(4)", float(v192), 3.445452402, 1e-9, tv),
            ("Values table gamma(4), M 80", float(rows[80]["gamma"]), 0.37396072, 1e-8, tv)]


@check
def gamma_61():
    """GAMMA.md and the physics paper: γ at the highest double-double D of hp/data/gamma_dd.jsonl and
    hp/data/local/gamma_dd.jsonl (61.03)"""
    pts = {}
    for fn in ("hp/data/gamma_dd.jsonl", "hp/data/local/gamma_dd.jsonl"):
        for line in open(os.path.join(ROOT, fn)):
            r = json.loads(line)
            if r.get("gamma"):
                pts[r["D"]] = r["gamma"]
    top = max(pts)
    txt = [("GAMMA.md", "γ(61) = 0.48935 and the points keep rising"),
           ("paper/physics/main.tex", "At $D = 61$, $\\gamma = 0.48935$")]
    return [("gamma at the highest D", pts[top], 0.48935, 1e-5, txt), ("that D", top, 61, 1, txt)]


@check
def saved_states():
    """README "Reproduce": the number of saved solutions in hp/data/states and their D range"""
    names = sorted(glob.glob(os.path.join(ROOT, "hp/data/states", "*.npz")))
    D = [float(re.search(r"_D([0-9.]+)\.npz$", n).group(1)) for n in names]
    txt = [("README.md", "(53 states, about one per unit of D from 8.25 to 60, about 1 MB)")]
    return [("saved states", len(names), 53, 1, txt), ("lowest D", min(D), 8.25, 0.01, txt),
            ("highest D", max(D), 60, 1, txt)]


@check
def held_release():
    """notes/HELD_HARMONICS.md 4.4: the `fits` spread column against the err of hp/data/delta_all.csv at D 74-79, and
    the range quoted for D 80, 100, 150 and 200"""
    import csv
    err = {float(r["D"]): float(r["err"]) for r in csv.DictReader(open(os.path.join(ROOT, "hp/data/delta_all.csv")))
           if r["kind"] == "chain_fit" and r["err"]}
    H = "notes/HELD_HARMONICS.md"
    out = []
    for D, q, s in ((74.0, 1.4e-15, "| 1.4e-15 |"), (75.0, 3.9e-15, "| 3.9e-15 |"), (76.0, 7.6e-15, "| 7.6e-15 |"),
                    (77.0, 1.3e-14, "| 1.3e-14 |")):
        out.append((f"fits spread at D {D:g}", err[D], q, 10.0 ** (math.floor(math.log10(q)) - 1), [(H, s)]))
    t78 = [(H, "| 2.0e-14, 3.0e-14 |")]
    out += [("fits spread at D 78", err[78.0], 2.0e-14, 0.1e-14, t78), ("fits spread at D 79", err[79.0], 3.0e-14, 0.1e-14, t78)]
    t80 = [(H, "| 4.2e-14 to 2.8e-11 |")]
    out += [("fits spread, lowest of D 80-200", min(err[D] for D in (80.0, 100.0, 150.0, 200.0)), 4.2e-14, 0.1e-14, t80),
            ("fits spread, highest of D 80-200", max(err[D] for D in (80.0, 100.0, 150.0, 200.0)), 2.8e-11, 0.1e-11, t80)]
    return out


def _box_lines(path, after=None):
    """The `name in [lo, hi]` lines of a Krawczyk log, as decimal strings, from the first line containing `after` on."""
    lines = open(os.path.join(ROOT, path)).read().splitlines()
    if after is not None:
        lines = lines[next(i for i, l in enumerate(lines) if after in l):]
    out = {}
    for l in lines:
        m = re.match(r"\s+(\S+)\s+in \[([0-9.]+), ([0-9.]+)\]", l)
        if m and m.group(1) not in out:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def _outward(lo, hi, places):
    """[lo, hi] rounded outward to `places` decimals, exactly (decimal strings in, floats out)."""
    from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
    q = Decimal(1).scaleb(-places)
    return (float(Decimal(lo).quantize(q, rounding=ROUND_FLOOR)), float(Decimal(hi).quantize(q, rounding=ROUND_CEILING)))


@check
def m5_enclosure():
    """Theorem A(M = 5)'s zero (#658): the certificate's enclosure, rounded outward, in solver/proof/logs/
    A_M2_krawczyk_M5_1e-10.log (proof/A_M2.md section 6.4), against the shorter outward roundings quoted in the README
    and PROOF.md"""
    b = _box_lines("solver/proof/logs/A_M2_krawczyk_M5_1e-10.log", "the zero (endpoints rounded outward)")
    wl, wh = _outward(*b["w/w_inf"], 14)
    al, ah = _outward(*b["A"], 13)
    tw = [("README.md", "ω/ω∞ ∈ [0.99922840320656, 0.99922840320657]"),
          ("PROOF.md", "ω/ω∞ ∈ [0.99922840320656, 0.99922840320657]")]
    ta = [("PROOF.md", "A ∈ [2.7185791892117, 2.7185791892118]")]
    return [("A(5) omega/omega_inf, lower end", wl, 0.99922840320656, 1e-14, tw),
            ("A(5) omega/omega_inf, upper end", wh, 0.99922840320657, 1e-14, tw),
            ("A(5) amplitude, lower end", al, 2.7185791892117, 1e-13, ta),
            ("A(5) amplitude, upper end", ah, 2.7185791892118, 1e-13, ta)]


@check
def m3_enclosure():
    """Theorem A(M = 3)'s zero (#658): the point certificate's enclosure in solver/proof/logs/A_M_krawczyk_M3_1e-10.log,
    and the same box from the independent mpmath-interval recheck (A_M_recheck_M3_1e-10.log) rounded outward, against
    the README, the math paper and proof/A_M.md"""
    k = _box_lines("solver/proof/logs/A_M_krawczyk_M3_1e-10.log")
    r = _box_lines("solver/proof/logs/A_M_recheck_M3_1e-10.log")
    q = {"A": (2.72647588563032, 2.72647588563144), "w/w_inf": (1.01560309485743, 1.01560309485804)}
    t = [("README.md", "A ∈ [2.72647588563032, 2.72647588563144], ω/ω∞ ∈ [1.01560309485743, 1.01560309485804]."),
         ("paper/math/main.tex",
          "$A \\in [2.72647588563032, 2.72647588563144]$, $\\omega/\\winf \\in [1.01560309485743, 1.01560309485804]$")]
    out = []
    for nm in ("A", "w/w_inf"):
        rl, rh = _outward(*r[nm], 14)
        out += [(f"A(3) {nm}, lower end (Krawczyk log)", float(k[nm][0]), q[nm][0], 1e-14, t),
                (f"A(3) {nm}, upper end (Krawczyk log)", float(k[nm][1]), q[nm][1], 1e-14, t),
                (f"A(3) {nm}, lower end (recheck, outward)", rl, q[nm][0], 1e-14, t),
                (f"A(3) {nm}, upper end (recheck, outward)", rh, q[nm][1], 1e-14, t)]
    return out


@check
def first_order_lam():
    """notes/FIRST_ORDER.md: the truncated models' own growth rate lambda* near 2 at their zeros (M = 5: the mean of the
    three runs), from solver/inner_pert/data/first_order_hier_M*.json (#658: the one-harmonic model's root, quoted
    elsewhere as 1.765)"""
    d = os.path.join(ROOT, "solver/inner_pert/data")
    lam = lambda n: json.load(open(os.path.join(d, n + ".json")))["lam"]  # noqa: E731
    m5 = [lam(n) for n in ("first_order_hier_M5_xi0.2", "first_order_hier_M5_xi0.19", "first_order_hier_M5_xi0.2_rtol1e-13")]
    F = "notes/FIRST_ORDER.md"
    return [("lambda*, M = 1", lam("first_order_hier_M1_xi0.2"), 1.769122008, 1e-9, [(F, "| 14.9977 | 1.769122008 |")]),
            ("lambda*, M = 3", lam("first_order_hier_M3_xi0.2"), 2.006804459, 1e-9, [(F, "| 9.88625 | 2.006804459 |")]),
            ("lambda*, M = 5", sum(m5) / 3, 1.9996689, 1e-7, [(F, "| 10.4023 | 1.9996689 |")]),
            ("lambda*, M = 1, short", lam("first_order_hier_M1_xi0.2"), 1.76912, 1e-5,
             [(F, "| λ* (n = ∞) | 1.76912 | 2.00680 | 1.99967 | 2 |")])]


@check
def sigma_fold():
    """matter/SIGMA.md section 5 and the physics paper: the fold of the sigma model's CSS branch at eta = 0.14, the
    maximum of D along solver/matter/data/sigma_fold_eta0.14.jsonl (a parabola in the path parameter q through the
    point of largest D and its two neighbours), and lambda, gamma at that point"""
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "solver/matter/data/sigma_fold_eta0.14.jsonl"))]
    i = max(range(len(rows)), key=lambda j: rows[j]["D"])
    f = rows[i]
    (q0, d0), (q1, d1), (q2, d2) = ((rows[j]["q"], rows[j]["D"]) for j in (i - 1, i, i + 1))
    a = ((d2 - d1) / (q2 - q1) - (d1 - d0) / (q1 - q0)) / (q2 - q0)
    b = (d1 - d0) / (q1 - q0) - a * (q1 + q0)
    qm = -b / (2 * a)
    dmax = d1 + a * (qm * qm - q1 * q1) + b * (qm - q1)
    t = [("matter/SIGMA.md", "D_f = 8.5444**, where γ = 0.5135"), ("paper/physics/main.tex", "$0.5135$ at a fold at"),
         ("paper/physics/main.tex", "$D = 8.5444$")]
    tab = [("matter/SIGMA.md", "| **8.5444 (fold)** | 6.2607 | 1.51227 | **1.94724** | **0.51355** |")]
    return [("fold D", dmax, 8.5444, 1e-4, t), ("gamma at the fold", f["gamma"], 0.5135, 1e-4, t),
            ("fold D, README", dmax, 8.544, 1e-3, [("README.md", "at a fold at D = 8.544,")]),
            ("lambda at the fold (table)", f["lam"], 1.94724, 1e-5, tab),
            ("gamma at the fold (table)", f["gamma"], 0.51355, 1e-5, tab)]


@check
def css_omega():
    """matter/CSS.md section 7, the README and the papers: omega_inf of the axion-dilaton (kappa = 1) and the free
    complex scalar (kappa = 0) from the D = inf inner Stokes zero (solver/matter/data/css_inner_kpath.jsonl) against
    the median of css_fit.py's converged fits (data/css_fit_k*.json), and their relative difference"""
    d = os.path.join(ROOT, "solver/matter/data")
    inner = {r["kappa"]: r["omega"] for r in (json.loads(l) for l in open(os.path.join(d, "css_inner_kpath.jsonl")))}
    C = "matter/CSS.md"
    tag = [("README.md", "(agreement 1.6×10⁻⁴ and 1.3×10⁻⁴, relative)"),
           (C, "agree to 1.6×10⁻⁴ (κ = 1) and 1.3×10⁻⁴ (κ = 0) in ω"),
           ("paper/math/main.tex", "$1.6\\times10^{-4}$ and $1.3\\times10^{-4}$ (relative)")]
    out = []
    for k, w_in, w_x, rng, ag in ((1.0, 3.1998939020, 3.1994, (3.1985, 3.1999), 1.6e-4),
                                  (0.0, 3.9789695074, 3.9784, (3.9776, 3.9787), 1.3e-4)):
        med, lo, hi = json.load(open(os.path.join(d, f"css_fit_k{k:g}.json")))["omega"]
        t_in = [(C, f"**{w_in:.10f}**"), ("README.md", f"{w_in:.7f}")]
        t_x = [(C, f"{w_x:.4f} (" + ("fits: " if k else "") + f"{rng[0]:.4f}–{rng[1]:.4f})"), ("README.md", f"{w_x:.4f}")]
        out += [(f"kappa {k:g}: inner omega_inf", inner[k], w_in, 1e-10, t_in),
                (f"kappa {k:g}: extrapolated omega_inf (median)", med, w_x, 1e-4, t_x),
                (f"kappa {k:g}: lowest fit", lo, rng[0], 1e-4, t_x), (f"kappa {k:g}: highest fit", hi, rng[1], 1e-4, t_x),
                (f"kappa {k:g}: relative agreement", (inner[k] - med) / med, ag, 1e-5, tag)]
    return out


@check
def css_lambda_table():
    """matter/CSS.md section 7: lambda_inf - 2 of the free fits of lambda(D) in powers of 1/n, per D_min and number of
    powers K (solver/matter/data/css_fit_k1.json, css_fit.py)"""
    tab = json.load(open(os.path.join(ROOT, "solver/matter/data/css_fit_k1.json")))["lam_minus_2_table"]
    sup = lambda e: str(e).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))  # noqa: E731
    fmt = lambda v: (lambda m, e: f"{m}×10{sup(int(e))}")(*f"{v:.2e}".split("e"))  # noqa: E731
    out = []
    for dmin, row in tab.items():
        cells = [fmt(v) if (dmin, K) not in (("24", 7), ("30", 7)) else f"({fmt(v)})" for v, K in zip(row, range(3, 8))]
        txt = [("matter/CSS.md", f"| {dmin} | " + " | ".join(cells) + " |")]
        for v, K in zip(row, range(3, 8)):
            out.append((f"lambda_inf - 2, D >= {dmin}, K = {K}", v, float(f"{v:.2e}"),
                        10.0 ** (math.floor(math.log10(abs(v))) - 2), txt))
    return out


def full_dinf():
    sys.path.insert(0, os.path.join(ROOT, "hp"))
    import dd_fits_mp as F
    import dinf_budget as B
    pts, _ = F.load_points()
    rows = B.grid(pts, sorted(pts), verbose=False, dmaxes=(60.1, 61.03))
    lo, hi, n = B.quoted(rows)
    return [("fits in the quoted selection", n, 48, 1, []),
            ("lowest limit", float(lo), RANGE[0], 1e-9, []),
            ("highest limit", float(hi), RANGE[1], 1e-9, [])]


def evaluate(items, shift=False):
    bad = []
    for label, val, q, unit, txt in items:
        v = val + unit if shift else val
        num_ok = abs(v - q) <= unit / 2 * (1 + 1e-9)
        missing = [f"{doc}: '{t}'" for doc, t in txt if not quoted_in(doc, t)]
        if not num_ok or missing:
            bad.append((label, v, q, unit, missing))
    return bad


def main():
    selftest, full = "--selftest" in sys.argv, "--full" in sys.argv
    groups = [(fn.__name__, fn()) for fn in CHECKS]
    if full:
        groups.append(("full_dinf", full_dinf()))
        groups.append(("full_lam_gauge", full_lam_gauge()))
    n_items = sum(len(items) for _, items in groups)
    n_bad = 0
    for name, items in groups:
        bad = evaluate(items)
        n_bad += len(bad)
        print(f"{name:16s} {len(items) - len(bad)} of {len(items)} pass")
        for label, v, q, unit, missing in bad:
            print(f"   FAIL {label}: computed {v!r}, quoted {q!r} (unit {unit:g})" +
                  (f"; text missing: {missing}" if missing else ""))
    print(f"{n_items - n_bad} of {n_items} checks pass")
    if selftest:
        caught = 0
        for name, items in groups:
            for it in items:
                if evaluate([it], shift=True):
                    caught += 1
                else:
                    print(f"   selftest: a one-unit shift is NOT caught for {name}: {it[0]}")
        print(f"selftest: {caught} of {n_items} checks reject a value shifted by one unit")
        n_bad += n_items - caught
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
