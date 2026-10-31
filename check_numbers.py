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
                     "notes/PLANTED_SELECTION.md", "notes/PROOF_QUEUE.md", "proof/P5_scope.md")}


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
    txt = [("README.md", "and Δ to 9.6×10⁻⁸ up to D = 67.77"),
           ("README.md", "above D = 30.2 it matches them to 9.6×10⁻⁸"),
           ("README.md", "| 30.2 – 67.77, never seen by the fit | ≤ 9.6×10⁻⁸ |")]
    t2 = [("README.md", "3.0×10⁻⁹ or better on 8.25 ≤ D ≤ 21, 8.0×10⁻⁹ up to D = 30, and 9.6×10⁻⁸ from there to")]
    return [("formula against the double-double points, D 30.2-67.77", err, 9.6e-8, 1e-9, txt),
            ("formula against the double-double points, D 8.25-21", bands[0], 3.0e-9, 1e-10, t2),
            ("formula against the double-double points, D 21-30", bands[1], 8.0e-9, 1e-10, t2),
            ("formula against the double-double points, D 30-61.03", bands[2], 9.6e-8, 1e-9, t2)]


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
def qd_values():
    """README, the quad-double Newton: Delta at D = 68.5, 69.5, 70.5 (hp/data/delta_qd.csv, hp/export_qd.py) to the
    double-double digits quoted, and their two-start spreads at most 1e-32"""
    rows = {float(c[0]): c for c in (r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/delta_qd.csv"))) if c[0][:1].isdigit()}
    txt = [("README.md", "Δ(68.5) =\n  1.7899636306159916 − 1.016822×10⁻¹⁶, Δ(69.5) = 1.7879302979617628 + 7.083175×10⁻¹⁷ and Δ(70.5) = 1.7859529321182939 +\n  1.087438×10⁻¹⁶"),
           ("README.md", "to 10⁻³²")]
    out = []
    for D, hi, lo in ((68.5, 1.7899636306159916, -1.016822e-16), (69.5, 1.7879302979617628, 7.083175e-17), (70.5, 1.7859529321182939, 1.087438e-16)):
        c = rows[D]
        out += [(f"Delta({D}) hi", float(c[1]), hi, 4e-16, txt), (f"Delta({D}) lo", float(c[2]), lo, 1e-22, txt)]
    out.append(("largest two-start spread at 68.5-70.5, log10", math.log10(max(float(rows[D][6]) for D in (68.5, 69.5, 70.5))), -32, 1, txt))
    return out


@check
def lowD_305():
    """README "Below D = 4": T45's lowest point, D = 3.05 at M = 384, its step from M = 320, and T82's D = 3.045
    (hp/data/delta_dd_lowD.csv, hp/export_lowD.py reading the raw-data folder T82-win)"""
    rows = [r.rstrip("\n").split(",") for r in open(os.path.join(ROOT, "hp/data/delta_dd_lowD.csv")) if r[:1].isdigit()]
    r = [c for c in rows if float(c[0]) == 3.05 and c[1] == "384"][0]
    txt = [("README.md", "run reached D = 3.05 at M = 320/384, Δ = 2.4849271637881, where the step from M = 320 to 384 is 2.1×10⁻¹¹"),
           ("README.md", "and the step from 320 to 384 is 2.1×10⁻¹¹, 56 times smaller")]
    step = float(r[7])
    prev = [c for c in rows if float(c[0]) == 3.05 and c[1] == "320"][0]
    r2 = [c for c in rows if float(c[0]) == 3.045 and c[1] == "384"][0]
    t2 = [("README.md", "reached D = 3.045 at M = 320/384, Δ = 2.4464747274662, with the step from M = 320 to 384 6.7×10⁻¹¹, 3.2 times the step at 3.05")]
    return [("D 3.05 Delta", float(r[2]) + float(r[3]), 2.4849271637881, 1e-13, txt),
            ("D 3.05 step 320 -> 384", abs(step), 2.1e-11, 1e-12, txt),
            ("ratio of the steps 256/320 and 320/384", abs(float(prev[7]) / step), 56, 1, txt),
            ("D 3.045 Delta (T82)", float(r2[2]) + float(r2[3]), 2.4464747274662, 1e-13, t2),
            ("D 3.045 step 320 -> 384", abs(float(r2[7])), 6.7e-11, 1e-12, t2),
            ("ratio of the 3.045 and 3.05 steps 320 -> 384", abs(float(r2[7]) / step), 3.2, 0.1, t2)]


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
    """The selection rule's estimator range from the D = infinity inner problem (notes/PLANTED_SELECTION.md section 12.9,
    the headline of solver/inner_amp/planted_checks_k20.json, the union of the K = 19 headline and the K = 20 range),
    the K = 20, 19, 18, 17 and 16 ranges alone, the unions of each pair of neighbouring orders, and the Delta_inf range the
    headline gives with the finite-D value 1.643893746 (dinf_quotes): Delta = 1.643893746 / (omega* / omega_inf)."""
    c20 = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k20.json")))["K20 combine"]
    lo, hi = c20["headline"]
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
    t20 = [("README.md", "the twentieth order alone gives 0.999997–1.000002"),
           ("README.md", "the ring-free sums) is 0.999997–1.000002"),
           ("notes/PLANTED_SELECTION.md", "**The hull is 0.999997–1.000002.**"),
           ("notes/PROOF_QUEUE.md", "K = 20 range 0.999997–1.000002"),
           ("paper/math/main.tex", "the twenty-order estimator range is $0.999997$--$1.000002$")]
    t19 = [("README.md", "the ring-free sums) is 0.999996–1.000004"),
           ("notes/PLANTED_SELECTION.md", "**The hull is 0.999996–1.000004.**"),
           ("notes/PROOF_QUEUE.md", "K = 19 range 0.999996–1.000004"),
           ("paper/math/main.tex", "the nineteen-order estimator range is $0.999996$--$1.000004$")]
    t18 = [("README.md", "the ring-free sums) is 0.999992–1.000004"),
           ("notes/PLANTED_SELECTION.md", "**The hull is 0.999992–1.000004.**"),
           ("paper/math/main.tex", "the eighteen-order estimator range is $0.999992$--$1.000004$")]
    t17 = [("README.md", "the ring-free sums) is 0.999990–1.000009"),
           ("notes/PLANTED_SELECTION.md", "**The hull is 0.999990–1.000009.**"),
           ("paper/math/main.tex", "the seventeen-order estimator range is $0.999990$--$1.000009$")]
    t16 = [("README.md", "the ring-free sums) is 0.999980–1.000014"),
           ("notes/PLANTED_SELECTION.md", "**The hull is 0.999980–1.000014.**"),
           ("paper/math/main.tex", "the sixteen-order estimator range is $0.999980$--$1.000014$")]
    t2 = [("README.md", "the last two orders alone give 0.999996–1.000004"),
          ("notes/PLANTED_SELECTION.md", "the last two orders alone, reported as context by item 7, is 0.999996–1.000004")]
    t2_19 = [("notes/PLANTED_SELECTION.md", "the range from the last two orders alone, reported as context by item 7, is 0.999992–1.000004")]
    t2_18 = [("notes/PLANTED_SELECTION.md", "the range from the last two orders alone, reported as context by item 7, is 0.999990–1.000009")]
    t2_17 = [("notes/PLANTED_SELECTION.md", "the range from the last two orders alone, reported as context by item 7, is 0.999980–1.000014")]
    return [("selection range, lower end", lo, 0.999975, 1e-6, txt),
            ("selection range, upper end", hi, 1.00005, 1e-5, txt),
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
    out += [("ring-free C18-BP sums, best sigma_19 prediction", min(min(f) for f in free), 4e-9, 1e-9, [(P, "predicted σ₁₉ to 4×10⁻⁹–6×10⁻⁸"), ("README.md", "predicted it to 4×10⁻⁹–6×10⁻⁸. The ring-free"), ("paper/math/main.tex", "predicted $\\sigma_{19}$ to $4\\times10^{-9}$--$6\\times10^{-8}$")]),
            ("ring-free C18-BP sums, worst sigma_19 prediction", max(max(f) for f in free), 6e-8, 1e-8, [(P, "predicted σ₁₉ to 4×10⁻⁹–6×10⁻⁸"), ("README.md", "predicted it to 4×10⁻⁹–6×10⁻⁸. The ring-free"), ("paper/math/main.tex", "predicted $\\sigma_{19}$ to $4\\times10^{-9}$--$6\\times10^{-8}$")]),
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
            ("ring-free completions, low", min(rf), 0.99999998, 1e-8, [(P, "the ring-free completions, 0.99999998–1.00000009"), ("README.md", "put the zero at 0.99999998–1.00000009"), ("paper/math/main.tex", "put the zero at $0.99999998$--$1.00000009$")]),
            ("ring-free completions, high", max(rf), 1.00000009, 1e-8, [(P, "the ring-free completions, 0.99999998–1.00000009"), ("README.md", "put the zero at 0.99999998–1.00000009"), ("paper/math/main.tex", "put the zero at $0.99999998$--$1.00000009$")]),
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
    out += [("err19 relative at wf 0.99 (verdict)", cv["0.99"]["err19_rel"], 4.5e-13, 0.1e-13, [(P, "σ₁₉ is converged to 4.5–5.6×10⁻¹³ relative"), ("README.md", "the K = 41 runs; 4.5–5.6×10⁻¹³")]),
            ("err19 relative at wf 1.01 (verdict)", cv["1.01"]["err19_rel"], 5.6e-13, 0.1e-13, [(P, "σ₁₉ is converged to 4.5–5.6×10⁻¹³ relative"), ("README.md", "the K = 41 runs; 4.5–5.6×10⁻¹³")])]
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
            ("ring-free completions at K = 20, low", min(rf), 0.999999997, 1e-9, [(P, "the ring-free completions, 0.999999997–1.000000028"), ("README.md", "put the zero at 0.999999997–1.000000028"), ("paper/math/main.tex", "put the zero at $0.999999997$--$1.000000028$")]),
            ("ring-free completions at K = 20, high", max(rf), 1.000000028, 1e-9, [(P, "the ring-free completions, 0.999999997–1.000000028"), ("README.md", "put the zero at 0.999999997–1.000000028"), ("paper/math/main.tex", "put the zero at $0.999999997$--$1.000000028$")]),
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
            ("relative err20 at wf 0.99 (verdict)", cv["nodes"]["0.99"]["err20_rel"], 6.5e-13, 0.1e-13, [(P, "σ₂₀ is converged to 6.5–8.1×10⁻¹³ relative"), ("README.md", "6.5–8.1×10⁻¹³ relative")]),
            ("relative err20 at wf 1.01 (verdict)", cv["nodes"]["1.01"]["err20_rel"], 8.1e-13, 0.1e-13, [(P, "σ₂₀ is converged to 6.5–8.1×10⁻¹³ relative"), ("README.md", "6.5–8.1×10⁻¹³ relative")]),
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
            ("M = 5 amplitude", A, 2.71858, 1e-5, [("README.md", "A = 2.71858")]),
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
    txt = [("README.md", "| 40–61 | 10⁻⁴ | 3 | 1.6450284 + 10.19239/D − 18.04099/D² | 5.7×10⁻⁶ |")]
    return [("40-61 c0", c[0], 1.6450284, 1e-7, txt), ("40-61 c1", c[1], 10.19239, 1e-5, txt),
            ("40-61 c2", c[2], -18.04099, 1e-5, txt), ("40-61 margin", err, 5.7e-6, 1e-7, txt)]


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
