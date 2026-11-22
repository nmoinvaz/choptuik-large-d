#!/usr/bin/env python3
"""Clean-room pass over the commands README.md quotes (#349).

For each command the README quotes it asks three things: does it run from its stated directory, do the files it
names exist, and does it print the number the text quotes, to the digits quoted. The numbers are read from the
README at run time (one locator per row), so a changed number in a copy of the README (--readme) changes the
expectation, and the row must flag.

  reproduce_readme.py --list                  every row with README line, directory, command, files, quoted numbers
  reproduce_readme.py --coverage              every code span of the README is a row or a listed non-command
  reproduce_readme.py --control               the controls: a changed number, a hidden file and a failing command
                                              each flag exactly their own row (fast rows only)
  reproduce_readme.py --tree DIR --out DIR    run all rows, one at a time, in DIR (a throwaway checkout: after each
                                              row `git checkout` and `git clean` restore it) and write the table
  options: --only ID,ID  --skip ID,ID  --python PATH (interpreter for python3)  --needs build (run only the rows
  the first pass deferred for a missing hpkernel or rustkernel)  --limit SECONDS (default 1800)  --cap HOURS (10)

Verdicts: OK, FAIL, NOT RUN (with the reason). Numbers: a quoted number agrees with a printed one when the printed
one rounds, or truncates, to it at the quoted digits. Install nothing and build nothing: the rows that need hpkernel
or rustkernel are run again with --python of an existing build.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−", "0123456789-+-")
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------------------------------------------
# numbers
def dec(s):
    """A quoted token as a Decimal: 4.1×10⁻¹⁵, 10⁻¹⁸⁰, 1e-9, −0.52, 0.7%."""
    s = s.strip().replace("−", "-").replace("%", "").replace(",", "").replace(" ", "")
    s = re.sub(r"\\times10\^\{(-?\d+)\}", lambda m: "e" + m.group(1), s)
    m = re.fullmatch(r"([-+]?[\d.]*)(?:×|x)?10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", s)
    if m:
        mant = m.group(1) if m.group(1) not in ("", "+", "-") else m.group(1) + "1"
        return Decimal(mant) * Decimal(10) ** int(m.group(2).translate(SUP))
    return Decimal(s)


def agree_digits(q, v):
    """Leading significant digits of the quoted q that the printed v shares, at equal magnitude (0 if the order differs)."""
    if q == 0 or v == 0 or q.adjusted() != v.adjusted() or (q < 0) != (v < 0):
        return 0
    qd = q.as_tuple().digits
    vd = v.as_tuple().digits
    n = 0
    for a, b in zip(qd, vd):
        if a != b:
            break
        n += 1
    return n


def matches(q, v):
    """v rounds or truncates to q at q's last quoted digit."""
    ulp = Decimal(1).scaleb(q.as_tuple().exponent)
    d = v - q
    if abs(d) <= ulp / 2 * Decimal("1.000000001"):
        return True
    return (d >= 0) == (q >= 0) and abs(d) < ulp * Decimal("1.000000001") and abs(v) >= abs(q)


def output_numbers(text):
    out = []
    for m in NUM.finditer(text.replace("−", "-")):
        try:
            out.append(Decimal(m.group(0)))
        except InvalidOperation:
            pass
    return out


def check_quote(tokens, mode, text):
    """-> (ok, [(token, found, best digits agreeing)])."""
    pool = output_numbers(text)
    res = []
    ok = True
    for t in tokens:
        q = dec(t)
        if mode == "eq":
            hit = any(matches(q, v) for v in pool)
            best = max([agree_digits(q, v) for v in pool] or [0])
        elif mode == "le":  # a printed value below the quoted bound and within three decades of it
            hit = any(q / 1000 <= v <= q * Decimal("1.000000001") for v in pool if q > 0)
            best = 0
        elif mode == "info":
            hit, best = True, 0
        else:
            raise ValueError(mode)
        ok = ok and hit
        res.append((t, hit, best))
    return ok, res


# --------------------------------------------------------------------------------------------------------------
# README locators
def table_cells(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return [re.sub(r"\$`|`\$|`", "", c) for c in cells]


class Q:
    """Quoted numbers of one row. rx: regex over the README (re.S) whose groups are the tokens.
    row: (after-regex, first-cell-regex, [column indices]): the table row after that anchor."""

    def __init__(self, label, rx=None, row=None, mode="eq", tok=None):
        self.label, self.rx, self.row, self.mode, self.tok = label, rx, row, mode, tok

    def tokens(self, text):
        if self.rx:
            m = re.search(self.rx, text, re.S)
            if not m:
                return None
            g = list(m.groups())
        else:
            after, first, cols = self.row
            a = re.search(after, text, re.S)
            if not a:
                return None
            g = None
            pos = a.end()
            for ln in text[pos:].split("\n")[:60]:
                ln = ln.strip()
                if ln.startswith("|"):
                    c = table_cells(ln)
                    if re.fullmatch(first, c[0]):
                        g = [c[i] for i in cols]
                        break
            if g is None:
                return None
        return self.tok(g) if self.tok else g


class Row:
    def __init__(self, id, span, argv, cwd=".", files=(), q=(), est=1, note="", skip=None, pre=None, regen=(),
                 post=None, need=None, alias=(), expect_rc=0):
        self.id, self.span, self.argv, self.cwd = id, span, argv, cwd
        self.files, self.q, self.est, self.note = list(files), list(q), est, note
        self.skip, self.pre, self.regen, self.post, self.expect_rc = skip, pre, list(regen), post, expect_rc
        self.need, self.alias = need, list(alias)  # need: argv of the import-only check run with an existing build


ROWS = []


def row(*a, **k):
    ROWS.append(Row(*a, **k))


def py(path, *args):
    return ["python3", path, *args]


# --------------------------------------------------------------------------------------------------------------
# the rows. span: the text the README quotes, found in the README or the row is STALE. argv: what runs ("python3"
# is the interpreter chosen by --python). cwd: the directory the README states ("." with a repo-relative path:
# the repository root, and if that fails once from the script's own directory). est: 1 fast, 2 minutes, 3 long.
def cx(cell):
    """'0.291 - 1.5704i' -> ['0.291', '-1.5704']; a plain number stays one token."""
    m = re.fullmatch(r"\s*([-−+]?[\d.]+)\s*([-−+])\s*([\d.]+)i\s*", cell.replace("−", "-"))
    return [m.group(1), m.group(2) + m.group(3)] if m else re.findall(r"[-+]?\d[\d.]*", cell)


def trow(after, first, cols, label=None, tok=None):
    return Q(label or f"{first}", row=(after, first, cols), tok=tok)


def cells_tok(g):
    out = []
    for c in g:
        out += cx(c)
    return out


# --- build, test, reproduce
row("git_clone_raw", "git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw",
    ["git", "clone", "https://github.com/nmoinvaz/choptuik-large-d-raw", "../choptuik-large-d-raw"],
    files=["../choptuik-large-d-raw/INDEX.md"],
    pre=dict(rc=0, wall=281, note="run once in the setup from _i349/ by HTTPS (the repository is public): 4 min 41 s, 36879 files"))
row("pip_install", "pip install .", ["pip", "install", "."], skip="build: no builds on this pass (coordinator)",
    need=["python3", "-c", "import rustkernel; print('rustkernel', rustkernel.available_backends())"])
row("cargo_test", "cargo test --release", ["cargo", "test", "--release"], skip="cargo build: no builds on this pass (coordinator)")
row("test_bindings", "python tests/test_bindings.py", py("tests/test_bindings.py"), files=["tests/test_bindings.py"],
    need=["python3", "tests/test_bindings.py"])
row("build_sh_main", "./build.sh main", ["./build.sh", "main"], skip="build: no builds on this pass (coordinator)")
row("hp_build_sh", "hp/build.sh", ["hp/build.sh"], skip="build: no builds on this pass (coordinator)",
    need=["python3", "-c", "import hpkernel; print('hpkernel', getattr(hpkernel, '__file__', ''))"])


def count_checks(n):
    def post(out):
        m = re.search(r"(\d+) passed, (\d+) failed, (\d+) skipped", out)
        if not m:
            return False, "no summary line"
        t = sum(int(x) for x in m.groups())
        return t == n, f"{t} checks ({m.group(1)} passed, {m.group(2)} failed, {m.group(3)} skipped), README says {n}"
    return post


row("reproduce_sh", "./reproduce.sh", ["./reproduce.sh"], est=3, post=count_checks(14),
    note="README: about 6 minutes on a busy 2-core machine, 2-3 on an idle laptop")
row("reproduce_sh_full", "--full", ["./reproduce.sh", "--full"], est=3, post=count_checks(15),
    note="README: --full adds the Proposition 4 check")
row("check_numbers", "check_numbers.py", py("check_numbers.py"), est=3,
    note="README: ties more than 800 quoted numbers to their source files")
row("check_numbers_selftest", "check_numbers.py", py("check_numbers.py", "--selftest"), est=2,
    note="README: a self-test shows each check would catch a one-digit change (the flag is SETUP.md's)")
row("mutation_controls", "solver/proof/mutation_controls.py", py("solver/proof/mutation_controls.py"), est=2,
    q=[Q("eight versions", rx=r"(Eight) deliberately broken versions", tok=lambda g: ["8"])])

# --- results and evidence
row("plot_delta", "cd solver/data && python3 ../plot_delta.py --raw ../../../choptuik-large-d-raw",
    py("../plot_delta.py", "--raw", "../../../choptuik-large-d-raw"), cwd="solver/data", est=2,
    files=["solver/delta.svg"], regen=["solver/delta.svg"])
row("formula_refit", "python3 ../formula_all.py --refit", py("../formula_all.py", "--refit"), cwd="solver/data", est=2,
    note="README: the coefficients move by up to 0.7% and the values by 1e-9 (fitted range), 1e-7 (D = 80) in another environment")
row("formula_all", "solver/formula_all.py", py("../formula_all.py"), cwd="solver/data", est=2, files=["solver/data/formula_coefficients.json"],
    q=[Q("D = 4", rx=r"formula gives Δ = ([\d.]+) and γ = ([\d.]+),")],
    note="README gives the path from the root; the script reads data/ and runs from solver/data")
row("rational_fit", "solver/rational_fit.py", py("solver/rational_fit.py"), est=2)
row("formula_cut_fit", "hp/formula_cut_fit.py", py("hp/formula_cut_fit.py"), est=3, files=["hp/data/formula_cut.json"],
    q=[Q("limit", rx=r"give Δ∞ = ([\d.]+)\. It is a", )])
row("formula_cut", "hp/formula_cut.py", py("hp/formula_cut.py"), est=2)
row("lowD_max", "cd hp && python3 lowD_max.py", py("lowD_max.py"), cwd="hp", files=["hp/data/lowD_dd_t79.jsonl"],
    q=[Q("maximum", rx=r"python3 lowD_max\.py`\) \| ([\d.]+) \| ([\d.]+) \|")])
row("export_lowD", "hp/export_lowD.py", py("hp/export_lowD.py"), est=2)
row("export_dd", "hp/export_dd.py", py("hp/export_dd.py"), est=2)
row("export_delta_all", "cd hp && python3 export_delta_all.py ../../choptuik-large-d-raw",
    py("export_delta_all.py", "../../choptuik-large-d-raw"), cwd="hp", est=3, files=["hp/data/delta_all.csv"],
    regen=["hp/data/delta_all.csv"], alias=["hp/export_delta_all.py"])
row("export_delta_all_verify", "--verify", py("export_delta_all.py", "../../choptuik-large-d-raw", "--verify"), cwd="hp", est=3,
    note="README: --verify checks every row against its source")
row("plot_lowD_values", "hp/plot_lowD_values.py", py("hp/plot_lowD_values.py"), est=2, files=["hp/lowD_values.svg"], regen=["hp/lowD_values.svg"])
row("lowD_formula", "hp/lowD_formula.py", py("hp/lowD_formula.py"), est=2,
    q=[Q("delta poly", rx=r"\\Delta \\approx ([\d.]+) \+ ([\d.]+)\\,s \+ ([\d.]+)\\,s\^2 - ([\d.]+)\\,s\^3 \+ ([\d.]+)\\,s\^4 - ([\d.]+)\\,s\^5 \+ ([\d.]+)\\,s\^6 - ([\d.]+)\\,s\^7 \+ ([\d.]+)\\,s\^8",
         tok=lambda g: [g[0], g[1], g[2], "-" + g[3], g[4], "-" + g[5], g[6], "-" + g[7], g[8]]),
       Q("gamma poly", rx=r"\\gamma \\approx ([\d.]+) \+ ([\d.]+)\\,s \+ ([\d.]+)\\,s\^2 - ([\d.]+)\\,s\^3 \+ ([\d.]+)\\,s\^4 - ([\d.]+)\\,s\^5 \+ ([\d.]+)\\,s\^6",
         tok=lambda g: [g[0], g[1], g[2], "-" + g[3], g[4], "-" + g[5], g[6]])])
row("plot_lowD_limits", "hp/plot_lowD_limits.py", py("hp/plot_lowD_limits.py"), est=2, files=["hp/lowD_limits.svg", "hp/data/published_lowD_limits.json"], regen=["hp/lowD_limits.svg"])
row("plot_lowD_logslope", "hp/plot_lowD_logslope.py", py("hp/plot_lowD_logslope.py"), est=2, files=["hp/lowD_logslope.svg"], regen=["hp/lowD_logslope.svg"])
row("lowD_logslope", "hp/lowD_logslope.py", py("hp/lowD_logslope.py"), est=2,
    q=[Q("prediction", rx=r"predicts ([\d.]+) at D = 3\.0425 and ([\d.]+) at 3\.04\.")])
row("gamma_dd", "hp/gamma_dd.py", py("hp/gamma_dd.py"), est=3, files=["hp/data/gamma_dd.jsonl", "hp/data/local/gamma_dd.jsonl"])
row("gamma_m48_gap", "hp/gamma_m48_gap.py", py("hp/gamma_m48_gap.py"), est=3)
row("gamma_compare", "cd hp && python3 gamma_compare.py data/gamma_dd.jsonl data/local/gamma_dd.jsonl",
    py("gamma_compare.py", "data/gamma_dd.jsonl", "data/local/gamma_dd.jsonl"), cwd="hp", est=1,
    files=["hp/data/gamma_dd.jsonl", "hp/data/local/gamma_dd.jsonl"],
    q=[Q("lambda-2 by powers", rx=r"On D ≥ 30 it is\s+([\d.]+×10⁻⁴), ([\d.]+×10⁻⁵), ([\d.]+×10⁻⁵), ([\d.]+×10⁻⁶) and \(([\d.]+) ± ([\d.]+)\)×10⁻⁶",
         tok=lambda g: g[:4] + [g[4] + "×10⁻⁶", g[5] + "×10⁻⁶"]),
       Q("lambda series", rx=r"λ = 2 \+ ([\d.]+)/n − ([\d.]+)/n² \+ … \(γ = 1/2 −\s*([\d.]+)/n")])
row("plot_extrapolation", "solver/plot_extrapolation.py", py("solver/plot_extrapolation.py"), est=2, files=["solver/extrapolation.svg"], regen=["solver/extrapolation.svg"])
row("values_table", "cd hp && python3 values_table.py", py("values_table.py"), cwd="hp", est=2,
    q=[trow(r"### Values", r"4", [1, 2], "D = 4"), trow(r"### Values", r"70", [2], "gamma(70)"),
       trow(r"### Values", r"100", [2], "gamma(100)"), trow(r"### Values", r"1000", [2], "gamma(1000)")])
row("track_c", "hp/track_c.py", py("hp/track_c.py"), est=3,
    q=[Q("series", rx=r"([\d.]+)\s+± ([\d.]+)×10⁻¹¹ at D = 250, ([\d.]+) ± ([\d.]+)×10⁻¹⁰ at 300 and ([\d.]+) ± ([\d.]+)×10⁻¹⁰ at 400",
         tok=lambda g: [g[0], g[1] + "×10⁻¹¹", g[2], g[3] + "×10⁻¹⁰", g[4], g[5] + "×10⁻¹⁰"]),
       Q("limit", rx=r"Its limit,\s+([\d.]+) ± ([\d.]+)×10⁻⁹", tok=lambda g: [g[0], g[1] + "×10⁻⁹"])])
row("track_c_fold", "hp/track_c_fold.py", py("hp/track_c_fold.py"), est=3, files=["hp/data/fold_259.csv"])
row("plot_physics", "solver/plot_physics.py", py("solver/plot_physics.py"), est=1, files=["solver/physics.svg"], regen=["solver/physics.svg"])
row("modecount_test", "hp/modecount_test.py", py("hp/modecount_test.py"), est=3)
row("dd_D5", "hp/compare/dd_D5.py", py("hp/compare/dd_D5.py"), est=3, files=["hp/compare/dd_D5.log"],
    q=[Q("D = 5", rx=r"\(([\d.]+),\s+([\d.]+)\), which agree with each other", tok=lambda g: g)])
row("data_all", "solver/data_all.py", py("solver/data_all.py"), est=2,
    q=[Q("D = 5 f64", rx=r"\(([\d.]+) ± 7×10⁻¹¹, the size of that extrapolation step")])
row("cmp_ecker", "hp/compare/cmp_ecker.py", py("hp/compare/cmp_ecker.py"), est=1,
    note="needs a clone of github.com/EckerChristian/CriticalSpacetimeCrystal as data (not run as code)",
    q=[Q("moves", rx=r"moving by −([\d.]+×10⁻⁴) between 256 and 512 modes")])
row("identity_check", "I200-mac/identity_check.py", py("../choptuik-large-d-raw/I200-mac/identity_check.py"), est=2,
    files=["../choptuik-large-d-raw/I200-mac/identity_check.py"])
row("plot_solver_limits", "hp/plot_solver_limits.py", py("hp/plot_solver_limits.py"), est=2, files=["hp/solver_limits.svg", "hp/data/delta_all.csv"], regen=["hp/solver_limits.svg"])
row("rv2_noise", "hp/review/rv2_noise.py", py("hp/review/rv2_noise.py"), est=2,
    q=[Q("rms", rx=r"\| ([\d.]+×10⁻¹⁸) rms from the cloud points' interpolant")])

# --- validation: the fits
row("dd_fits_clean", "hp/dd_fits_mp.py --clean", py("hp/dd_fits_mp.py", "--clean", "--dmax=60.1"), est=3,
    note="the README gives --dmax=60.1 in the same sentence; its number is for that pair",
    q=[Q("dropped", rx=r"drops them \((\d+) on today's file", )])
MODELS = [r"1/D, degree 16", r"1/D, degree 18", r"1/\(D-2\), degree 18", r"\(D-2\)\^\{-1/2\}, degree 20", r"D\^\{-1/2\}, degree 20"]
A_TAB = r"\| Model \| max residual on 10 ≤ D ≤ 45"
row("dd_fits_10_45", "hp/dd_fits_mp.py 10 45 --clean --dmax=60.1", py("hp/dd_fits_mp.py", "10", "45", "--clean", "--dmax=60.1"), est=3,
    q=[trow(A_TAB, m, [1, 2, 4], m.replace("\\", "")) for m in MODELS])
row("dd_fits_10_375", "10 37.5 --clean --dmax=60.1", py("hp/dd_fits_mp.py", "10", "37.5", "--clean", "--dmax=60.1"), est=3,
    q=[trow(A_TAB, m, [3], m.replace("\\", "")) for m in MODELS])
row("dd_fits_mixed", "hp/dd_fits_mp.py 10 45 --clean --mixed --dmax=60.1", py("hp/dd_fits_mp.py", "10", "45", "--clean", "--mixed", "--dmax=60.1"), est=3,
    q=[Q("coefficient", rx=r"gets (3×10⁻⁵) at degree 14\s+and (6×10⁻⁷) at degree 16"),
       Q("limits", rx=r"quoted bar\s+\(([\d.]+) to ([\d.]+)\)")])
row("dd_fits_1e9", "hp/dd_fits_mp.py 10 1e9 --clean --dmax=60.1", py("hp/dd_fits_mp.py", "10", "1e9", "--clean", "--dmax=60.1"), est=3,
    q=[Q("limits", rx=r"Their limits run from ([\d.]+) to ([\d.]+) on 10 ≤ D ≤\s+60\.1")])
row("dinf_budget", "hp/dinf_budget.py", py("hp/dinf_budget.py"), est=3,
    q=[Q("range", rx=r"ranges over ([\d.]+)–([\d.]+) \(`hp/dinf_budget\.py`\), centred on ([\d.]+)\.")])
row("planted_dinf", "hp/planted_dinf.py", py("hp/planted_dinf.py"), est=3, files=["hp/data/planted_dinf.log"])
row("rv2_planted", "hp/review/rv2_planted.py", py("hp/review/rv2_planted.py"), est=3,
    q=[Q("offset", rx=r"finds the same offset, ([\d.]+)×10⁻⁸", )])

# --- large D: the checks of the propositions
row("largeD_scaling", "hp/largeD_scaling.py", py("hp/largeD_scaling.py"), est=2)
row("largeD_F", "hp/largeD_F.py", py("hp/largeD_F.py"), est=2,
    q=[trow(r"\| D \| first harmonic of F", str(d), [1, 2, 3, 4, 5], f"D = {d}") for d in (20, 30, 40, 50, 60)])
row("plot_echo", "hp/plot_echo.py", py("hp/plot_echo.py"), est=2, files=["hp/data/echo_shapes.csv", "hp/echo_shapes.svg"], regen=["hp/echo_shapes.svg", "hp/data/echo_shapes.csv"])
row("states_module", "hp/states.py", py("hp/states.py"), skip="a module the profile scripts import (where the states are read from), not a command; the README quotes no invocation",
    files=["hp/data/states"])
row("plot_wave_family", "hp/plot_wave_family.py", py("hp/plot_wave_family.py"), est=2, files=["hp/wave_family.svg"], regen=["hp/wave_family.svg"])
row("largeD_expand", "largeD_expand.py 4", py("solver/largeD_expand.py", "4"), est=2,
    alias=["solver/largeD_expand.py"])
row("largeD_formal_check", "solver/largeD_formal_check.py", py("solver/largeD_formal_check.py"), est=2)
row("largeD_fast_transport", "solver/largeD_fast_transport.py", py("solver/largeD_fast_transport.py"), est=2,
    q=[trow(r"\| n \| 100 \| 200 \| 400 \| 800 \| 1600 \|", r"relative error, no gravity \(F = 0 in c\)", [1, 2, 3, 4, 5], "no gravity"),
       trow(r"\| n \| 100 \| 200 \| 400 \| 800 \| 1600 \|", r"relative error, with gravity", [1, 2, 3, 4, 5], "with gravity")])
row("largeD_fast_transport_full", "solver/largeD_fast_transport_full.py", py("solver/largeD_fast_transport_full.py"), est=2,
    q=[trow(r"\| n \| 100 \| 200 \| 400 \| 800 \| 1600 \|", r"relative error, full background.*", [1, 2, 3, 4, 5], "full background")])
row("largeD_transport_sympy", "solver/largeD_transport_sympy.py", py("solver/largeD_transport_sympy.py"), est=2)
row("largeD_exponent", "hp/largeD_exponent.py", py("hp/largeD_exponent.py"), est=2,
    q=[trow(r"\| D \| 20 \| 30 \| 40 \| 50 \| 60 \| ∞ \(fit in 1/n\) \|", r"n\\langle G\(1\)\\rangle/2", [1, 2, 3, 4, 5], "n<G(1)>/2"),
       trow(r"\| D \| 20 \| 30 \| 40 \| 50 \| 60 \| ∞ \(fit in 1/n\) \|", r"\\langle F\^2\\rangle", [1, 2, 3, 4, 5], "<F^2>"),
       trow(r"\| D \| 20 \| 30 \| 40 \| 50 \| 60 \| ∞ \(fit in 1/n\) \|", r"exponent shift, real part", [1, 2, 3, 4, 5], "shift")])
row("plot_light_cone", "hp/plot_light_cone.py", py("hp/plot_light_cone.py"), est=1, files=["hp/light_cone.svg"], regen=["hp/light_cone.svg"])
row("largeD_linear_late", "solver/largeD_linear_late.py", py("solver/largeD_linear_late.py"), est=2,
    q=[trow(r"\| x \| χ measured \(orders 100–120\)", f, [1, 3], f, tok=cells_tok) for f in (r"1/\\sqrt2", r"0\.8", r"0\.9")])
row("largeD_dingle_K", "solver/largeD_dingle_K.py", py("solver/largeD_dingle_K.py"), est=2,
    q=[Q("K ratio", rx=r"\| \$`K_\{\\text\{measured\}\}/K`\$ \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", tok=cells_tok)])
row("largeD_linear_poles", "solver/largeD_linear_poles.py", py("solver/largeD_linear_poles.py"), est=2)
row("largeD_linear_poisson", "solver/largeD_linear_poisson.py", py("solver/largeD_linear_poisson.py"), est=2)
row("euler_late", "solver/largeD_prop4/", py("solver/largeD_prop4/euler_late.py"), est=2,
    alias=["euler_late.py"], q=[Q("deviation", rx=r"extrapolates to \(−([\d.]+) − ([\d.]+)i\)×10⁻³ over k = 100–122", tok=lambda g: ["-" + g[0] + "×10⁻³", "-" + g[1] + "×10⁻³"])])
row("kcheck", "kcheck.py", py("solver/largeD_prop4/kcheck.py"), est=2, q=[Q("1 + c/k", rx=r"equals K to 1 \+ ([\d.]+)/k")])
row("switch2", "switch2.py", py("solver/largeD_prop4/switch2.py"), est=2, q=[Q("singulant", rx=r"singulant is (−[\d.]+)i → −iπ", tok=lambda g: [g[0]])])
row("largeD_linear_truncation", "solver/largeD_linear_truncation.py", py("solver/largeD_linear_truncation.py"), est=2,
    q=[Q("x = 0.9", rx=r"the smallest difference is (4×10⁻⁴⁰) against \$`\\lvert C\\rvert\\times\\text\{branch\} = (\d+)\\times10\^\{(-\d+)\}`\$, and at x = 0\.8,\s+n = 40 it is (10⁻²³) against (2×10⁻⁷)",
         tok=lambda g: [g[0], g[1] + "e" + g[2], g[3], g[4]])])
row("largeD_stokes_C", "solver/largeD_stokes_C.py", py("solver/largeD_stokes_C.py"), est=2,
    q=[Q("ratios", rx=r"ratio of the two sides is ([\d.]+) at n = 10⁵ for μ = 1 \(([\d.]+) for μ = 3\)"),
       Q("n(ratio-1)", rx=r"\\to ([\d.]+) - ([\d.]+)i`\$, of the size of \$`\\lvert s\^2\\rvert = ([\d.]+)`", tok=lambda g: [g[0], "-" + g[1], g[2]])])
row("largeD_inner", "solver/largeD_inner.py", py("solver/largeD_inner.py"), est=2)
row("hier_schur", "hier.py --schur", py("solver/inner_stokes/hier.py", "--schur"), est=3,
    note="the README quotes the flag, not an M")
row("largeD_Fratios", "hp/largeD_Fratios.py", py("hp/largeD_Fratios.py"), est=2,
    q=[Q("ratios", rx=r"F_3/F_1\\rvert \\to ([\d.]+)`\$–([\d.]+) and \$`\\lvert F_5/F_1\\rvert \\to ([\d.]+)`\$–([\d.]+) \(degree 1: ([\d.]+),\s+([\d.]+)\)", tok=lambda g: g)])
row("amp_series", "amp_series.py", py("solver/inner_amp/amp_series.py"), est=2, q=[Q("agreement", rx=r"to (2)–(4)×10⁻¹⁶ at orders 0–3", tok=lambda g: [g[0] + "×10⁻¹⁶", g[1] + "×10⁻¹⁶"], mode="le")])
row("amp_flint_par", "solver/inner_amp/amp_flint_par.py --fit phys", py("solver/inner_amp/amp_flint_par.py", "--fit", "phys"), est=1,
    files=["solver/inner_amp/data/sigF_w1.0000_K15_J750_phys.json", "solver/inner_amp/data/sigF_w1.0000_K15_J1000_phys.json"],
    note="python-flint at 640 bits; AGENTS.md: hours per omega, with positional omega, K, J, bits")
row("zero_selection", "zero_selection.py", py("solver/inner_amp/zero_selection.py"), est=2)
row("arg_crossing", "arg_crossing.py", py("solver/inner_amp/arg_crossing.py"), est=2)
row("amp_flint", "amp_flint.py", py("solver/inner_amp/amp_flint.py"), est=3, skip="named as the legacy basis; no invocation or number quoted, and it is the serial reference (AGENTS.md: hours)")
row("harm_pred", "harm_pred.py", py("solver/inner_amp/harm_pred.py"), est=2,
    q=[Q("tracks", rx=r"\(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+) against ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+); `harm_pred\.py`")])
row("plot_selection", "solver/plot_selection.py", py("solver/plot_selection.py"), est=2, files=["solver/selection.svg"], regen=["solver/selection.svg"])
row("plot_spectrum", "solver/plot_spectrum.py", py("solver/plot_spectrum.py"), est=2, files=["solver/spectrum.svg"], regen=["solver/spectrum.svg"])
row("expand_pert", "solver/inner_pert/expand_pert.py", py("solver/inner_pert/expand_pert.py"), est=2, alias=["expand_pert.py"])
row("heat_form", "solver/inner_pert/heat_form.py", py("solver/inner_pert/heat_form.py"), est=1)
row("rw_map", "solver/inner_pert/rw_map.py", py("solver/inner_pert/rw_map.py"), est=1)
row("pert_check", "pert_check.py", py("solver/inner_pert/pert_check.py"), est=2, q=[Q("residual", rx=r"\(`pert_check\.py`, to (6×10⁻³⁰)\)", mode="le")])
row("pert_check2", "pert_check2.py", py("solver/inner_pert/pert_check2.py"), est=2, q=[Q("residual", rx=r"\(`pert_check2\.py`, with gravity, to (10⁻¹⁷⁷)\)", mode="le")])
row("pert_check3", "solver/inner_pert/pert_check3.py 45 11", py("solver/inner_pert/pert_check3.py", "45", "11"), est=2,
    q=[Q("relative", rx=r"coefficients to (10⁻¹⁸⁰) relative", mode="le"), Q("misses", rx=r"misses by ([\d.]+)% already at\s+the third order")])
row("plot_first_order", "solver/plot_first_order.py", py("solver/plot_first_order.py"), est=2, files=["solver/first_order.svg"], regen=["solver/first_order.svg"])
row("eeg20", "solver/eeg20/", py("solver/eeg20/eeg20.py"), est=2,
    q=[Q("ratio", rx=r"\(ω²F''\+3ω F'\)/\(3ω F'\) is ([\d.]+)–([\d.]+) at D = 12–60", tok=lambda g: g)])
row("tilt", "solver/eeg20/", py("solver/eeg20/tilt.py"), est=2,
    q=[Q("tilt", rx=r"Measured over predicted at x = 1 is ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+) at D = 20,")])
row("maxc", "solver/eeg20/", py("solver/eeg20/maxc.py"), est=2, q=[Q("ratio", rx=r"The ratio rises from ([\d.]+)\s+to ([\d.]+) from D = 20 to 60")])
for f in ["struct_common", "struct_css", "struct_eqs", "struct_inner", "struct_outer", "struct_symmetry"]:
    p = f"solver/matter/{f}.py"
    if (HERE / p).exists() and f != "struct_common":
        row(f, "solver/matter/struct_*.py", py(p), est=2, alias=[])
row("plot_matter", "solver/plot_matter.py", py("solver/plot_matter.py"), est=2, files=["solver/matter.svg"], regen=["solver/matter.svg"])
row("singularity_type", "hp/singularity_type.py planted", py("hp/singularity_type.py", "planted"), est=2,
    q=[Q("sqrt exponent", rx=r"(returns an isolated planted square root)", tok=lambda g: ["0.5"])])
row("horizon_toy", "hp/horizon_toy.py", py("hp/horizon_toy.py"), est=2)
row("plot_dropoffs", "hp/plot_dropoffs.py", py("hp/plot_dropoffs.py"), est=2, files=["hp/large_d_dropoffs.svg"], regen=["hp/large_d_dropoffs.svg"])
row("plot_gamma", "solver/plot_gamma.py", py("solver/plot_gamma.py"), est=2, files=["solver/gamma.svg"], regen=["solver/gamma.svg"])
row("plot_modecount", "hp/plot_modecount.py", py("hp/plot_modecount.py"), est=2, files=["hp/modecount.svg"], regen=["hp/modecount.svg"])
row("fast_eigenvalues", "solver/proof/fast_eigenvalues.py", py("solver/proof/fast_eigenvalues.py"), est=1)
for nm in ["fixed_point", "fixed_point_M", "validated_int", "krawczyk"]:
    row(nm, f"{nm}.py", py(f"solver/proof/{nm}.py"), skip="named as the code of certified lemmas; the README quotes no invocation and no number (PROOF.md and proof/A_certify.md hold the commands: out of scope)")


# --------------------------------------------------------------------------------------------------------------
# README code spans, and what is not a command
SPAN = re.compile(r"(?<![`$])`([^`\n]+)`(?!`)")
FILE_EXT = re.compile(r"\.(py|sh|svg|csv|json|jsonl|log|npz|md|rs|toml|txt|tex|pdf)$")


def readme_spans(text):
    """[(line, span)] of the inline code spans and the lines of the fenced blocks other than math."""
    out, fence, lang = [], False, ""
    for i, ln in enumerate(text.split("\n"), 1):
        if ln.startswith("```"):
            fence = not fence
            lang = ln[3:].strip() if fence else ""
            continue
        if fence:
            if lang != "math" and ln.strip():
                out.append((i, ln.split("#")[0].strip() if lang == "sh" else ln.strip()))
            continue
        for m in SPAN.finditer(ln):
            out.append((i, m.group(1)))
    return out


def is_math(s):
    return s.startswith("$") or s.endswith("$") or s.strip() in ("$", "")


def covered(span, rows):
    for r in rows:
        for t in [r.span] + r.alias:
            if span == t or span in t or (len(t) > 6 and t in span):
                return r.id
    return None


def named_files(text):
    """Paths the README names in code spans: with a file extension, or a directory."""
    out = []
    for i, s in readme_spans(text):
        if is_math(s) or " " in s.strip() and not s.startswith("["):
            continue
        for t in re.findall(r"[\w./*<>-]+", s):
            if FILE_EXT.search(t) or t.endswith("/"):
                out.append((i, t))
    return out


_TRACKED = None


def find_file(root, t, raw):
    """exists | where it is | None."""
    t0 = t.rstrip("/")
    if "*" in t0 or "<" in t0:
        return "pattern"
    for base, tag in ((root, ""), (raw, "raw clone: ")):
        if (base / t0).exists():
            return "ok" if not tag else tag + t0
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True).stdout.split("\n")
    hits = sorted({f for f in _TRACKED if f.endswith("/" + t0) or ("/" + t0 + "/") in f or f.startswith(t0 + "/")})
    return ("elsewhere: " + ", ".join(hits[:3])) if hits else None


# --------------------------------------------------------------------------------------------------------------
# running
class Ctx:
    pass


def tree_clean(ctx):
    if ctx.marker:
        subprocess.run(["git", "-C", str(ctx.tree), "checkout", "-q", "--", "."], check=False)
        subprocess.run(["git", "-C", str(ctx.tree), "clean", "-fdq", "-e", ".i349_run_tree"], check=False)


def threads_now():
    load = os.getloadavg()[0]
    return (4, 10, load) if load < 4 else (2, 19, load)


def run_command(ctx, argv, cwd, log, limit):
    nthreads, nice, load = threads_now()
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(nthreads)
    env["PATH"] = str(Path(ctx.python).parent) + os.pathsep + env["PATH"]
    run = [ctx.python if a in ("python3", "python") else a for a in argv]
    t0 = time.time()
    with open(log, "ab") as lf:
        lf.write(f"$ cd {cwd} && {' '.join(argv)}   [threads {nthreads}, nice {nice}, load {load:.1f}, {time.strftime('%F %T', time.gmtime())} UTC]\n".encode())
        lf.flush()
        try:
            p = subprocess.Popen(run, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True, preexec_fn=lambda: os.nice(nice))
        except OSError as e:
            lf.write(f"[cannot start: {e}]\n".encode())
            return dict(rc=127, wall=0.0, cpu=0.0, rss=0.0, timeout=False, threads=nthreads, nice=nice, load=load, pid=0)
        timeout = False
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if time.time() - t0 > limit:
                timeout = True
                lf.write(f"[stopped by pid {p.pid} after {limit} s]\n".encode())
                lf.flush()
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(p.pid, sig)
                    except ProcessLookupError:
                        break
                    time.sleep(5)
                pid, status, ru = os.wait4(p.pid, 0)
                break
            time.sleep(0.5)
    rc = os.waitstatus_to_exitcode(status)
    return dict(rc=rc, wall=time.time() - t0, cpu=ru.ru_utime + ru.ru_stime, rss=ru.ru_maxrss / 1024, timeout=timeout,
                threads=nthreads, nice=nice, load=load, pid=p.pid)


def git_changes(ctx):
    out = subprocess.run(["git", "-C", str(ctx.tree), "status", "--porcelain"], capture_output=True, text=True).stdout
    return [ln for ln in out.split("\n") if ln.strip() and ".i349_run_tree" not in ln]


def execute(ctx, r, readme):
    """One row -> a result dict."""
    res = dict(id=r.id, span=r.span, est=r.est, explicit=bool(re.search(r"\s", r.span)))
    res["line"] = (readme[: readme.index(r.span)].count("\n") + 1) if r.span in readme else None
    if r.span not in readme:
        res.update(verdict="FAIL", reason="STALE: the span is not in the README")
        return res
    argv, cwd = r.argv, r.cwd
    if ctx.pass2:
        if r.need:
            argv, cwd = r.need, "."
        elif r.skip:
            return {**res, "verdict": "NOT RUN", "reason": r.skip}
        res["existing_build"] = ctx.python
    elif r.pre:
        return {**res, "verdict": "OK", "rc": r.pre["rc"], "wall": r.pre["wall"], "reason": r.pre["note"],
                "files": [(f, (ctx.tree / f).exists()) for f in r.files]}
    elif r.skip:
        return {**res, "verdict": "NOT RUN", "reason": r.skip}
    if time.time() - ctx.t0 > ctx.cap * 3600:
        return {**res, "verdict": "NOT RUN", "reason": f"{ctx.cap} h cap of the pass reached"}
    log = ctx.out / "logs" / f"{r.id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b"")
    tree_clean(ctx)
    adapted = None
    wd = ctx.tree / cwd
    m = run_command(ctx, argv, wd, log, ctx.limit)
    out = log.read_text(errors="replace")
    # a path-written command that fails once from the root runs from the script's own directory
    if (m["rc"] != r.expect_rc and not m["timeout"] and cwd == "." and len(argv) > 1 and "/" in argv[1]
            and argv[1].endswith((".py", ".sh")) and not argv[1].startswith("..") and not ctx.pass2):
        d, b = os.path.split(argv[1])
        if (ctx.tree / d / b).exists():
            adapted = f"from the root: exit {m['rc']}; rerun from {d}/"
            tree_clean(ctx)
            m2 = run_command(ctx, [argv[0], b, *argv[2:]], ctx.tree / d, log, ctx.limit)
            m = {**m2, "wall": m["wall"] + m2["wall"], "cpu": m["cpu"] + m2["cpu"], "rss": max(m["rss"], m2["rss"])}
            out = log.read_text(errors="replace")
            res["cwd_run"] = d
    res.update(m, adapted=adapted, argv=argv, cwd=res.get("cwd_run", cwd))
    changes = git_changes(ctx)
    res["regen"] = []
    for f in r.regen:
        d = subprocess.run(["git", "-C", str(ctx.tree), "diff", "--numstat", "--", f], capture_output=True, text=True).stdout.strip()
        res["regen"].append((f, "identical" if not d else "differs: " + " ".join(d.split()[:2]) + " lines +/-"))
    res["written"] = [c for c in changes][:12]
    res["files"] = [(f, (ctx.tree / f).exists()) for f in r.files]
    tree_clean(ctx)
    reasons = []
    if m["timeout"]:
        return {**res, "verdict": "NOT RUN", "reason": f"not finished in {ctx.limit // 60} min (stopped by pid {m['pid']}); {r.note}".strip("; "),
                "tail": out.strip().split("\n")[-3:]}
    if not ctx.pass2 and re.search(r"No module named '?(hpkernel|rustkernel)", out) and m["rc"] != 0:
        return {**res, "verdict": "NOT RUN", "reason": "needs a build (hpkernel or rustkernel): second pass", "deferred": True,
                "tail": out.strip().split("\n")[-3:]}
    if m["rc"] != r.expect_rc:
        reasons.append(f"exit {m['rc']}")
    miss = [f for f, ok in res["files"] if not ok]
    if miss:
        reasons.append("missing: " + ", ".join(miss))
    q_res = []
    for q in r.q:
        toks = q.tokens(readme)
        if toks is None:
            q_res.append((q.label, None, [], False))
            reasons.append(f"quote locator lost: {q.label}")
            continue
        ok, det = check_quote(toks, q.mode, out)
        q_res.append((q.label, q.mode, det, ok))
        if not ok:
            bad = [f"{t} ({b} digits agree)" for t, h, b in det if not h]
            reasons.append(f"{q.label}: not printed: " + "; ".join(bad))
    res["quotes"] = q_res
    if r.post:
        ok, msg = r.post(out)
        res["post"] = msg
        if not ok:
            reasons.append(msg)
    res["tail"] = out.strip().split("\n")[-3:]
    res["verdict"] = "FAIL" if reasons else "OK"
    res["reason"] = "; ".join(reasons)
    return res


def select(rows, args):
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    return [r for r in rows if (only is None or r.id in only) and r.id not in skip]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--readme", default=str(HERE / "README.md"))
    ap.add_argument("--tree", default=str(HERE))
    ap.add_argument("--out", default="reproduce_out")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--pass2", metavar="RESULTS.jsonl", help="run the rows the first pass deferred for a missing build, and the import-only checks, with --python of an existing build")
    ap.add_argument("--limit", type=int, default=1800)
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--fast", action="store_true", help="only rows with est 1")
    args = ap.parse_args()
    readme = Path(args.readme).read_text(encoding="utf-8")
    rows = select(ROWS, args)
    if args.fast:
        rows = [r for r in rows if r.est == 1]
    if args.list:
        return cmd_list(rows, readme, args)
    if args.coverage:
        return cmd_coverage(readme, args)
    ctx = Ctx()
    ctx.tree, ctx.out, ctx.python = Path(args.tree).resolve(), Path(args.out).resolve(), os.path.abspath(args.python)
    ctx.marker = (ctx.tree / ".i349_run_tree").exists()
    ctx.limit, ctx.cap, ctx.pass2, ctx.t0 = args.limit, args.cap, bool(args.pass2), time.time()
    ctx.out.mkdir(parents=True, exist_ok=True)
    if args.control:
        return cmd_control(ctx, readme, args)
    return cmd_run(ctx, rows, readme, args)


# --------------------------------------------------------------------------------------------------------------
def cmd_list(rows, readme, args):
    out = []
    for r in rows:
        line = (readme[: readme.index(r.span)].count("\n") + 1) if r.span in readme else None
        qs = []
        for q in r.q:
            t = q.tokens(readme)
            qs.append({"label": q.label, "mode": q.mode, "tokens": t})
        out.append(dict(id=r.id, line=line, span=r.span, explicit=bool(re.search(r"\s", r.span)), cwd=r.cwd, argv=r.argv, files=r.files,
                        quotes=qs, est=r.est, skip=r.skip, regen=r.regen, note=r.note))
    explicit = sum(1 for o in out if o["explicit"])
    ntok = sum(len(q["tokens"] or []) for o in out for q in o["quotes"])
    lost = [(o["id"], q["label"]) for o in out for q in o["quotes"] if q["tokens"] is None]
    stale = [o["id"] for o in out if o["line"] is None]
    print(f"{len(out)} rows, {explicit} with an invocation quoted (arguments or a directory), {len(out) - explicit} named scripts, "
          f"{sum(1 for o in out if o['quotes'])} with quoted numbers ({ntok} numbers), {sum(1 for o in out if o['skip'])} marked not run before the pass")
    print("stale spans:", stale, "| lost locators:", lost)
    for o in out:
        print(f"\n{o['id']}  [README line {o['line']}, est {o['est']}]  {'(skip: ' + o['skip'] + ')' if o['skip'] else ''}")
        print(f"   quoted:  {o['span']}")
        print(f"   run:     cd {o['cwd']} && {' '.join(o['argv'])}")
        if o["files"]:
            print(f"   files:   {', '.join(o['files'])}")
        for q in o["quotes"]:
            print(f"   number:  {q['label']} [{q['mode']}] = {q['tokens']}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "commands.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def cmd_coverage(readme, args):
    bad = []
    nfile = 0
    root, raw = HERE, HERE.parent / "choptuik-large-d-raw"
    for i, s in readme_spans(readme):
        if is_math(s):
            continue
        cmdlike = re.match(r"(cd |python3? |\./|cargo |pip |git |hp/build|[\w./-]+\.(py|sh)\b)", s) or re.search(r"\.(py|sh)(\s|$)", s)
        if cmdlike and not covered(s, ROWS):
            bad.append((i, s))
    print("code spans that look like a command and are no row:", len(bad))
    for i, s in bad:
        print(f"  line {i}: {s}")
    missing = []
    for i, t in named_files(readme):
        nfile += 1
        st = find_file(root, t, raw)
        if st is None:
            missing.append((i, t))
    print(f"file names in code spans: {nfile}; not found anywhere: {len(missing)}")
    for i, t in missing:
        print(f"  line {i}: {t}")


def fmt_quotes(res):
    qs = res.get("quotes") or []
    if not qs:
        return "none quoted"
    parts = []
    for label, mode, det, ok in qs:
        if mode is None:
            parts.append(f"{label}: locator lost")
            continue
        n = sum(1 for _, h, _ in det if h)
        parts.append(f"{label}: {n}/{len(det)}" + ("" if ok else " MISMATCH"))
    return "; ".join(parts)


def table(results):
    lines = ["| id | README | command (dir) | exit | files | number printed vs quoted | wall | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        fl = r.get("files") or []
        files = "all %d exist" % len(fl) if fl and all(ok for _, ok in fl) else ("none named" if not fl else "MISSING " + ",".join(f for f, ok in fl if not ok))
        cmd = f"`{r['span']}`" + (f" ({r['cwd_run']}/)" if r.get("cwd_run") else "")
        wall = f"{r['wall']:.0f} s" if r.get("wall") is not None else ""
        v = r["verdict"] + (f": {r['reason']}" if r.get("reason") and r["verdict"] != "OK" else "")
        if r["verdict"] == "OK" and r.get("adapted"):
            v += f" ({r['adapted']})"
        lines.append(f"| {r['id']} | {r.get('line')} | {cmd} | {r.get('rc', '')} | {files} | {fmt_quotes(r)} | {wall} | {v} |")
    return "\n".join(lines)


def files_named_result(ctx, readme):
    raw = ctx.tree.parent / "choptuik-large-d-raw"
    names = sorted({t for _, t in named_files(readme)})
    ok, elsewhere, missing, pattern = [], [], [], []
    for t in names:
        st = find_file(ctx.tree, t, raw)
        (ok if st == "ok" else pattern if st == "pattern" else missing if st is None else elsewhere).append((t, st))
    reason = f"{len(names)} file names in code spans: {len(ok)} exist as written, {len(elsewhere)} found elsewhere in the repository or the raw clone, {len(pattern)} patterns, {len(missing)} not found"
    detail = "; ".join(f"{t} -> {st}" for t, st in elsewhere) + ("; NOT FOUND: " + ", ".join(t for t, _ in missing) if missing else "")
    return dict(id="files_named", span="every file name in a code span", line=None, est=1, explicit=False, rc=0, wall=0.0,
                verdict="FAIL" if missing else "OK", reason=reason + (" | " + detail if detail else ""), files=[], quotes=[])


def cmd_run(ctx, rows, readme, args):
    rows = sorted(rows, key=lambda r: r.est)
    resf = ctx.out / "results.jsonl"
    done = set()
    if resf.exists():
        done = {json.loads(l)["id"] for l in resf.read_text().split("\n") if l.strip()}
    if args.pass2:
        prev = [json.loads(l) for l in Path(args.pass2).read_text().split("\n") if l.strip()]
        ids = {p["id"] for p in prev if p.get("deferred")} | {r.id for r in ROWS if r.need}
        rows = [r for r in rows if r.id in ids]
    elif "files_named" not in done:
        res = files_named_result(ctx, readme)
        with open(resf, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print("files_named:", res["verdict"], res["reason"][:200], flush=True)
    for r in rows:
        if r.id in done:
            continue
        print(f"[{time.strftime('%F %T', time.gmtime())}] {r.id} ...", flush=True)
        res = execute(ctx, r, readme)
        with open(resf, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
        print(f"    {res['verdict']}  {res.get('reason', '')[:160]}  ({res.get('wall', 0):.0f} s)", flush=True)
    results = [json.loads(l) for l in resf.read_text().split("\n") if l.strip()]
    (ctx.out / "table.md").write_text(table(results) + "\n")
    cnt = {}
    for r in results:
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    print("verdicts:", cnt)


def cmd_control(ctx, readme, args):
    """Controls. A changed README number, a changed README command, a failing command and a named file that is
    missing each flag their own row and no other: the rows run on the real README, then on copies of it."""
    fast = ["lowD_max", "values_table", "heat_form", "rw_map", "plot_light_cone"]
    base_rows = [r for r in ROWS if r.id in fast]
    synthetic = [Row("ctl_exit", "ctl", ["false"]), Row("ctl_file", "ctl", ["true"], files=["hp/no_such_file.csv"]),
                 Row("ctl_ok", "ctl", ["true"])]
    plants = {
        "number planted": ("| 3.755725 | 3.46677169 |", "| 3.755725 | 3.46677179 |", ["lowD_max"]),
        "command changed": ("python3 lowD_max.py`)", "python3 lowD_maxx.py`)", ["lowD_max"]),
        "number planted in a second row": ("| 1000 | 1.65424086 | 0.499250 |", "| 1000 | 1.65424086 | 0.499251 |", ["values_table"]),
    }
    texts = {"real README": readme}
    lines = []
    for name, (a, b, _) in plants.items():
        assert readme.count(a) == 1, f"the plant target of '{name}' is not unique in this README"
        texts[name] = readme.replace(a, b)
        (ctx.out / f"README.{name.replace(' ', '_')}.md").write_text(texts[name], encoding="utf-8")
        lines.append(f"plant '{name}': `{a}` -> `{b}`")
    out = {}
    for name, text in texts.items():
        out[name] = {}
        for r in base_rows + synthetic:
            res = execute(ctx, r, text + "\nctl" if r.id.startswith("ctl") else text)
            out[name][r.id] = res["verdict"] + (": " + res.get("reason", "") if res.get("reason") else "")
    names = list(texts)
    lines += ["", "| row | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for rid in out["real README"]:
        lines.append(f"| {rid} | " + " | ".join(out[n][rid] for n in names) + " |")
    ok = out["real README"]["ctl_exit"].startswith("FAIL") and out["real README"]["ctl_file"].startswith("FAIL") \
        and out["real README"]["ctl_ok"].startswith("OK") and out["real README"]["lowD_max"].startswith("OK")
    lines.append("")
    lines.append(f"synthetic rows: ctl_exit and ctl_file flag, ctl_ok passes: {ok}")
    for name, (_, _, expect) in plants.items():
        flagged = [rid for rid in out["real README"] if out[name][rid] != out["real README"][rid]]
        lines.append(f"plant '{name}': flagged {flagged}, expected {expect}: {flagged == expect}")
        ok = ok and flagged == expect
    lines.append(f"CONTROL {'PASSED' if ok else 'FAILED'}")
    (ctx.out / "control.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
