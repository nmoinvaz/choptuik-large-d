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
        for name in ("README.md", "paper/physics/main.tex", "paper/math/main.tex", "proof/A_certify.md", "GAMMA.md")}


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
    """The selection rule's estimator range from the D = infinity inner problem (notes/PLANTED_SELECTION.md section 7.8,
    the headline of solver/inner_amp/planted_checks_k15.json), and the Delta_inf range it gives with the finite-D
    value 1.643893746 (dinf_quotes): Delta = 1.643893746 / (omega* / omega_inf)."""
    c = json.load(open(os.path.join(ROOT, "solver/inner_amp/planted_checks_k15.json")))["K15 combine"]
    lo, hi = c["headline"]
    dlo, dhi = DINF / hi, DINF / lo
    txt = [("README.md", "ω*/ω∞ = 0.999975–1.00005, Δ∞ = 1.64387 ± 0.00006 (1.64381–1.64393)"),
           ("README.md", "puts the selected ω at 0.999975–1.00005 ω∞ (Δ∞ = 1.64387 ± 0.00006"),
           ("paper/math/main.tex", "\\Dinf = 1.64387 \\pm 0.00006 \\qquad (1.64381\\text{--}1.64393)"),
           ("paper/math/main.tex", "$\\omega^*/\\winf = 0.999975$--$1.00005$"),
           ("paper/physics/main.tex", "$\\omega/\\omega_\\infty = 0.999975$--$1.00005$"),
           ("paper/physics/main.tex", "\\Dinf = 1.64387 \\pm 0.00006")]
    return [("selection range, lower end", lo, 0.999975, 1e-6, txt),
            ("selection range, upper end", hi, 1.00005, 1e-5, txt),
            ("Delta_inf range, lower end", dlo, 1.64381, 1e-5, txt),
            ("Delta_inf range, upper end", dhi, 1.64393, 1e-5, txt),
            ("Delta_inf range, centre", (dlo + dhi) / 2, 1.64387, 1e-5, txt),
            ("Delta_inf range, half-width", (dhi - dlo) / 2, 0.00006, 1e-5, txt)]


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
