#!/usr/bin/env python3
"""Do the stored runs of the start-value lemma FP(M) cover the start data of a point certificate? (#484, #862)

Lemma FP(M) (proof/A_lemmas_M.md 2.1) holds on a parameter set when fixed_point_M.run_M succeeds on it. Its stored
runs are centred on the hier.py zero x^ = fixed_point_M.ZEROS[M]:
    table rows  lemmas_M_runs/fixed_point_M_table_M{M}.json  fixed_point_M.table: the point x^, the real box
                                                             x^ +- 1e-6 and the complex box x^ +- 1e-3 (1 + i)
    data rows   lemmas_M_runs/fixed_point_M_data_M{M}.json   fixed_point_M.data_table -> start_data_M over x^ +- h,
                                                             h = 1e-9 and 1e-6. That call runs the fixed point over
                                                             the real box and over its Cauchy box (real parts
                                                             x^ +- (h + c), imaginary parts +- c, c = 1e-3) and raises
                                                             unless both hold (fixed_point_M.py:462), so a stored row
                                                             is a run that held on both.
A point certificate calls start_data_M twice, at its own centre p^ and over the hull of its box, so it uses four
parameter sets: p^, the hull, and the Cauchy box of each. Two kinds of stored run are read:
    box2  certify_M2.py, the second-order chart (M = 5: data/A_M2_box_M5_1e-10.json). start_data2 makes the centre
          run at p^ and passes it to both calls (certify_M2.py:164, 166, 193); the hull is that of p(X).
    box   certify_M.py, the first-order chart (M = 3: data/A_M_box_M3_1e-10.json). start_data calls start_data_M at
          the midpoints of the hull and over the hull of p^ + T X (certify_M.py:152, 154, 298), and start_data_M
          makes its own centre run at those midpoints. The run does not store its truncation N, so N is taken from
          that centre run, made here.

Every set is rebuilt here in arb by the expressions of the code path that ran it. data_table makes its real box
before anything sets the precision, so the first row of an invocation has a 53-bit box and the later ones 320-bit
boxes, which differ by 1e-17. A data row is therefore rebuilt both ways and must contain the certificate's sets both
ways. The tests are
    (a) the configuration: start ray, start radius, splitting, truncation N, Cauchy radius, and the SHA-256 of
        fixed_point_M.py against the prefix the certificate recorded;
    (b) containment, ball in ball (arb.contains, acb.contains): the certificate's two real sets in a lemma run's real
        box and its two Cauchy boxes in the same run's complex box;
    (c) the side condition of the diag splitting on a complex box, M W0 sup|Im(w/w_inf)| <= 3/2 (A_M_review.md F3).
The verdict is COVERED (exit 0) when one stored lemma run with the certificate's configuration contains all four sets
and (c) holds, and NOT COVERED (exit 1) otherwise. When the sets lie inside a run whose truncation alone differs, the
verdict line says so: that run proves the fixed point for another polynomial Y_N, on a set that contains the
certificate's.

    python3 start_cover_M2.py [--box=data/A_M2_box_M5_1e-10.json] [--rows=all|data|table]
    python3 start_cover_M2.py --mutate=widen|edge|centre|cauchy|ray|trunc
        widen   the certificate's outer half-width h = 2e-6, past the lemma's largest real box
        edge    h just past the edge of x^ +- 1e-6 (EDGE: by 9e-11 in the parameter with the largest offset)
        centre  p^ moved by 2e-6 in w/w_inf
        cauchy  the certificate's Cauchy radius 2e-3
        ray     the certificate's start ray 3pi/8 (r0 unchanged)
        trunc   the certificate's truncation N - 1
      The first five must put the sets outside every run with the certificate's configuration, and trunc must
      change the verdict (exit status as the verdict).
    python3 start_cover_M2.py --mutate=edge0   null mutation, h 2e-10 inside that edge: the sets must stay inside
    python3 start_cover_M2.py --selftest       the clean check, edge0 and the six mutations; exit 0 only if each
                                               gives what it must
    python3 start_cover_M2.py --rerun          also reruns the fixed point on the certificate's sets by the
                                               certificate's own calls (one process, about a minute): the contraction
                                               constants, and the stored eta bounds reproduced. Then, if a stored row
                                               covers, that row is rerun and compared. If only the truncation
                                               differs, the lemma is run with the certificate's N on the widest row
                                               that contains the sets, and the start values of the two truncations at
                                               p^ must overlap, since both enclose y+. Exit 0 when the rerun passes
                                               and a stored row or that run covers.
    python3 start_cover_M2.py --rerun --mutate=overlap   the other truncation's start values taken 1e-30 away in A:
                                               they must not overlap (RERUN: FAIL, exit 1)
"""
import hashlib
import json
import os
import sys
import time
from decimal import Decimal

import flint
from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixed_point_M as FPM  # noqa: E402
import certify_M as CM  # noqa: E402
import certify_M2 as C2M  # noqa: E402

RUNS = os.path.join(HERE, "lemmas_M_runs")
PREC = 320                    # the precision of start_data_M, of start_data2 and of certify_M.start_data
DATA_CAUCHY = "1e-3"          # fixed_point_M.data_table's Cauchy radius (its literal argument; diag=True likewise)
BOX_CAUCHY = "1e-3"           # certify_M.start_data's Cauchy radius (its literal argument; diag=True likewise)
MUTATIONS = ("widen", "edge", "centre", "cauchy", "ray", "trunc")
# outer half-widths that put the hull 9e-11 past the edge of x^ +- 1e-6, and 2e-10 inside it (edge, edge0), for the
# stored centres: the largest offset p^ - x^ is 7.93e-10 in w/w_inf at M = 5 and 3.69e-10 in A at M = 3
EDGE = {3: ("9.9972e-7", "9.9943e-7"), 5: ("9.993e-7", "9.99e-7")}
_centres = {}


def pnames(M):
    return ["A", "w/w_inf"] + [f"{t}{nu}" for nu in range(3, M + 1, 2) for t in ("B", "C")]


def centre_run(p, M, r0, arg, diag):
    """the centre run of start_data_M at the point p (exact arb): run_M at the midpoints with kcap = 40"""
    key = (M, r0, arg, diag, tuple(v.mid().str(60, radius=False) for v in p))
    if key not in _centres:
        ctx.prec = PREC
        _centres[key] = FPM.run_M([acb(v.mid()) for v in p], M, r0, arg, None, verbose=False, kcap=40, diag=diag)
    return _centres[key]


# ---------------------------------------------------------------- the certificate's sets

def cert_data(bx, mutate=None):
    """the configuration and the four parameter sets of the start-data calls of a stored box run"""
    M, kind = bx["M"], bx["kind"]
    m = M + 1
    h, ws = bx["h"], list(bx["w"])
    if mutate == "widen":
        h = "2e-6"
    elif mutate in ("edge", "edge0"):
        h = EDGE[M][mutate == "edge0"]
    st = bx["start"]
    words = st["start"]
    sha = words.split("sha256 ")[1].split(")")[0] if "sha256 " in words else None
    # box2_run and box_run: the tilt (and Gamma) and the widths at KPREC
    ctx.prec = CM.KPREC
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    widths = [arb(h), arb(h)] + [arb(ws[(i // 2) % len(ws)]) for i in range(M - 1)]
    if kind == "box2":
        G = [[arb(x) for x in r] for r in bx["Gamma"]]
        ctx.prec = PREC                                # start_data2 works at 320 bits
        p = CM.exact_p(bx["p"])
        if mutate == "centre":
            p[1] = p[1] + arb("2e-6")
        ctx.prec = PREC
        T = CM.T_matrix(Cm, M)
        X = [arb(0, wd) for wd in widths]
        hull = C2M.p_of_x(p, T, G, X, M)
        cstr, n_stored = C2M.CAUCHY, st["N"]
        calls = "certify_M2.start_data2 (certify_M2.py:164, 166, 193): the centre run at p^, passed to both calls"
    else:
        T = CM.T_matrix(Cm, M)                         # box_run builds the hull at KPREC
        p = CM.exact_p(bx["p"])
        if mutate == "centre":
            p[1] = p[1] + arb("2e-6")
        hull = [p[i] + arb(0, widths[i]) if i < 2 else
                p[i] + sum((T[i][j] * arb(0, widths[j]) for j in range(2)), arb(0)) + arb(0, widths[i])
                for i in range(m)]
        ctx.prec = PREC                                # start_data works at 320 bits
        p = [arb(x.mid()) for x in hull]               # its pc
        cstr, n_stored = BOX_CAUCHY, None
        calls = ("certify_M.start_data (certify_M.py:152, 154; the hull at :298): start_data_M at the midpoints of "
                 "the hull and over the hull, each with its own centre run")
    if mutate == "cauchy":
        cstr = "2e-3"
    ctx.prec = PREC
    c = arb(cstr)

    def cbox(x):                                       # start_data_M's xc
        return [acb(arb(v) + arb(0, c), arb(0, c)) for v in x]
    arg, r0, diag = bx["path"][1], bx["path"][0], "diag splitting" in words
    N = n_stored if n_stored is not None else centre_run(p, M, r0, arg, diag)["N"]
    return {"M": M, "kind": kind, "arg": "3pi/8" if mutate == "ray" else arg, "r0": r0, "diag": diag,
            "N": N - (1 if mutate == "trunc" else 0), "n_stored": n_stored is not None, "c": cstr, "sha": sha,
            "p": [arb(v) for v in p], "hull": hull, "cb_p": cbox(p), "cb_hull": cbox(hull),
            "hw": max(float(abs(hull[i] - p[i]).upper()) for i in range(m)), "hw_stored": st.get("hull_halfwidth"),
            "h": h, "w": ws, "calls": calls}


# ---------------------------------------------------------------- the lemma's stored runs

def lemma_runs(M, rows="all"):
    """the stored runs of lemma FP(M) with their sets: (list of runs, the table's point rows)"""
    ctx.prec = PREC
    xs = FPM.ZEROS[M]
    with open(os.path.join(RUNS, f"fixed_point_M_table_M{M}.json")) as fh:
        tab = [r for r in json.load(fh) if r["kind"] == "lemma" and r["ok"] and r.get("sector") is None]
    points = {(r["arg"], r["r0"], r["diag"]): r for r in tab if r["tag"] == "point"}
    runs = []
    if rows in ("all", "table"):
        for key, pt in points.items():
            rb = [r for r in tab if (r["arg"], r["r0"], r["diag"]) == key and r["box"]]
            cb = [r for r in tab if (r["arg"], r["r0"], r["diag"]) == key and r["cbox"]]
            if not (rb and cb) or {r["N"] for r in rb + cb} != {pt["N"]}:
                continue
            # table.row: real box [acb(arb(v, arb(box)))], complex box [acb(arb(v, arb(cbox)), arb(0, arb(cbox)))]
            runs.append({"name": f"table {key[0]}", "arg": key[0], "r0": key[1], "diag": key[2], "N": pt["N"],
                         "k": pt["k"], "c": cb[0]["cbox"], "lip": (rb[0]["lip"], cb[0]["lip"]),
                         "what": f"real x^ +- {rb[0]['box']}, complex x^ +- {cb[0]['cbox']}(1 + i)",
                         "sets": [{"real": [arb(v, arb(rb[0]["box"])) for v in xs],
                                   "cauchy": [acb(arb(v, arb(cb[0]["cbox"])), arb(0, arb(cb[0]["cbox"])))
                                              for v in xs]}]})
    if rows in ("all", "data"):
        with open(os.path.join(RUNS, f"fixed_point_M_data_M{M}.json")) as fh:
            dat = json.load(fh)
        c = arb(DATA_CAUCHY)
        for r in dat:
            pt = points.get((r["arg"], r["r0"], True))      # start_data_M's centre run is the table's point row
            if pt is None:
                continue
            # data_table: xb = [arb(v, arb(h))] at 53 or 320 bits; start_data_M: xc = [acb(v + arb(0, c), arb(0, c))]
            sets = []
            for prec in (PREC, 53):
                ctx.prec = prec
                real = [arb(v, arb(r["h"])) for v in xs]
                ctx.prec = PREC
                sets.append({"real": real, "cauchy": [acb(v + arb(0, c), arb(0, c)) for v in real]})
            runs.append({"name": f"data {r['arg']} h={r['h']}", "arg": r["arg"], "r0": r["r0"], "diag": True,
                         "N": pt["N"], "k": pt["k"], "c": DATA_CAUCHY, "h": r["h"], "eta": r["eta"], "deta": r["deta"],
                         "what": f"real x^ +- {r['h']}, Cauchy box x^ +- ({r['h']} + {DATA_CAUCHY}), imaginary +- "
                                 f"{DATA_CAUCHY}", "sets": sets})
    return runs, points


# ---------------------------------------------------------------- the tests

def slack(S, L):
    """lemma radius - |centre offset| - certificate radius (a float, for the report; the verdict is .contains)"""
    return float((L.rad() - abs(S.mid() - L.mid()) - S.rad()).mid())


def inside(S, Ls):
    """(every S_i lies in L_i for every construction L of the lemma's set, the smallest slack of the real parts, its
    index, the smallest slack of the imaginary parts or None); S and each L are lists of arb or of acb"""
    ok, worst, at, worst_im = True, None, None, None
    for L in Ls:
        for i, (s, lm) in enumerate(zip(S, L)):
            ok = ok and bool(lm.contains(s))
            if isinstance(lm, acb):
                sl = slack(acb(s).real, lm.real)
                si = slack(acb(s).imag, lm.imag)
                worst_im = si if worst_im is None else min(worst_im, si)
            else:
                sl = slack(s, lm)
            if worst is None or sl < worst:
                worst, at = sl, i
    return ok, worst, at, worst_im


def same_config(ce, run):
    return {"ray": ce["arg"] == run["arg"], "r0": Decimal(ce["r0"]) == Decimal(run["r0"]),
            "splitting": ce["diag"] == run["diag"], "N": ce["N"] == run["N"],
            "Cauchy radius": Decimal(ce["c"]) == Decimal(run["c"])}


def names(runs):
    return ", ".join("[" + r["name"] + "]" for r in runs)


def check(bx, rows="all", mutate=None, quiet=False, name=""):
    """the tests (a), (b), (c). Returns a dict: covered; inside (the sets lie in a run that has the certificate's
    configuration, the truncation aside); ce (the certificate's data); covering (the runs that cover); around (the
    runs that contain the sets and differ in the truncation alone); why (per lemma run, the configuration items that
    differ and the smallest slack of the four sets)"""
    def say(*a):
        if not quiet:
            print(*a)
    ce = cert_data(bx, mutate)
    M = ce["M"]
    runs, points = lemma_runs(M, rows)
    pn = pnames(M)
    ctx.prec = PREC
    xh = [arb(v) for v in FPM.ZEROS[M]]
    with open(FPM.__file__, "rb") as fh:
        sha_now = hashlib.sha256(fh.read()).hexdigest()[:16]
    sha_ok = ce["sha"] == sha_now
    hw_ok = ce["hw_stored"] is None or mutate is not None or abs(ce["hw"] - ce["hw_stored"]) <= 1e-12 * ce["hw_stored"]
    ray_ok = bool(FPM.parse_arg(ce["arg"]).overlaps(arb(CM.path_for(M)[1]))) or mutate == "ray"
    say(f"start_cover_M2: {name} against the stored runs of lemma FP({M}), rows {rows}, mutation {mutate}  "
        f"[python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}]")
    say(f"certificate: ray {ce['arg']}, r0 {ce['r0']}, {'diag' if ce['diag'] else 'plain'} splitting, N {ce['N']}"
        + ("" if ce["n_stored"] else " (from the centre run at p^; the run does not store it)")
        + f", Cauchy radius {ce['c']}, box h {ce['h']}, w {ce['w']}")
    say(f"  start data:        {ce['calls']}")
    say("  p^ - x^:           " + "  ".join(f"{pn[i]} {float((ce['p'][i] - xh[i]).mid()):+.3e}" for i in range(M + 1)))
    say("  hull half-widths:  " + "  ".join(f"{pn[i]} {float(ce['hull'][i].rad()):.3e}" for i in range(M + 1)))
    say(f"  largest hull half-width about p^ {ce['hw']:.16e}, "
        + (f"the run's stored value {ce['hw_stored']}: reproduced {hw_ok}" if ce["hw_stored"] is not None
           else "of which the run stores no value"))
    say(f"(a) fixed_point_M.py sha256 {sha_now}, the certificate recorded {ce['sha']}: same file {sha_ok}; the "
        f"lemma's ray is the path's start ray: {ray_ok}")
    cert_sets = (("p^", "real", ce["p"]), ("hull", "real", ce["hull"]), ("Cauchy(p^)", "cauchy", ce["cb_p"]),
                 ("Cauchy(hull)", "cauchy", ce["cb_hull"]))
    covering, around, why = [], [], {}
    for run in runs:
        cfg = same_config(ce, run)
        res = [(nm, *inside(S, [v[kind] for v in run["sets"]])) for nm, kind, S in cert_sets]
        allin = all(r[1] for r in res)
        low = min(res, key=lambda r: r[2])
        why[run["name"]] = ("configuration differs in " + ", ".join(k_ for k_, v in cfg.items() if not v) + "; "
                            if not all(cfg.values()) else "") + f"smallest slack {low[2]:+.3e} ({low[0]}, {pn[low[3]]})"
        if all(cfg.values()) and allin:
            covering.append(run)
        elif allin and all(v for k_, v in cfg.items() if k_ != "N"):
            around.append(run)
        say(f"lemma run [{run['name']}]: ray {run['arg']}, r0 {run['r0']}, {'diag' if run['diag'] else 'plain'}, N "
            f"{run['N']}, k {run['k']}; {run['what']}")
        say("  (a) same " + ", ".join(f"{k_} {v}" for k_, v in cfg.items()))
        say("  (b) " + "; ".join(f"{nm} inside {ok} (slack {sl:+.3e} at {pn[at]}"
                                 + (f", imaginary parts {si:+.1e}" if si is not None else "") + ")"
                                 for nm, ok, sl, at, si in res)
            + f" -> {'all inside' if allin else 'NOT all inside'}")
    # (c) the diag splitting on the certificate's complex boxes
    im = max(abs(acb(ce[k_][1]).imag).upper() for k_ in ("cb_p", "cb_hull"))
    lhs = FPM.W0() * M * arb(im)
    diag_ok = (not ce["diag"]) or bool(lhs <= arb(3) / 2)
    say(f"(c) diag splitting on the Cauchy boxes: M W0 sup|Im w/w_inf| = {float(lhs.upper()):.5f} <= 3/2: {diag_ok}")
    base = sha_ok and hw_ok and ray_ok and diag_ok
    covered = bool(covering) and base
    if not base:
        around = []
    if covered:
        verdict = f"COVERED by {names(covering)}"
    elif around:
        verdict = (f"NOT COVERED: the sets lie inside {names(around)}, whose truncation N = {around[0]['N']} is not "
                   f"the certificate's N = {ce['N']}")
    else:
        verdict = "NOT COVERED"
    say(f"START COVER M2 ({name}, rows {rows}, mutation {mutate}): {verdict}")
    return {"covered": covered, "inside": base and bool(covering or around), "ce": ce, "covering": covering,
            "around": around, "why": why}


# ---------------------------------------------------------------- the rerun

def bounds(eta):
    return float(eta[0].upper()), float(eta[1].upper())


def rerun(bx, res, mutate=None):
    """the fixed point on the certificate's own sets, by the certificate's own calls; then the widest stored row
    that covers, or, when only the truncation differs, the lemma with the certificate's N on the widest row that
    contains the sets, and the overlap of the two truncations' start values at p^"""
    ce, covering, around = res["ce"], res["covering"], res["around"]
    M, r0, arg, diag = ce["M"], ce["r0"], ce["arg"], ce["diag"]
    ok = True
    t0 = time.time()
    p = ce["p"]
    centre = centre_run(p, M, r0, arg, diag)
    eta_p = max(float(v.upper()) for v in FPM.eta_bounds(centre).values())
    print(f"rerun, centre run at p^: N {centre['N']}, k {centre['k']}, contraction {float(centre['lip'].upper()):.6f}, "
          f"holds {centre['ok']}, |eta| <= {eta_p:.3e}  [{time.time() - t0:.0f} s]", flush=True)
    ok = ok and centre["ok"] and centre["N"] == ce["N"]
    # box2 passes its centre run to both calls; box lets start_data_M make it (the same run at the same midpoints)
    cen = centre if ce["kind"] == "box2" else None
    ctx.prec = PREC
    print("rerun, start_data_M at p^ (its own line follows):", flush=True)
    _, vals_c, _, eta_c = FPM.start_data_M([arb(v) for v in p], M, r0, arg, +1, ce["c"], centre=cen, diag=diag,
                                           verbose=True)
    e0, e1 = bounds(eta_c)
    print(f"  eta {e0!r}, Cauchy part {e1!r}  [{time.time() - t0:.0f} s]", flush=True)
    print("rerun, start_data_M over the hull (its own line follows):", flush=True)
    _, _, _, eta_b = FPM.start_data_M([arb(v) for v in ce["hull"]], M, r0, arg, +1, ce["c"], centre=cen, diag=diag,
                                      verbose=True)
    b0, b1 = bounds(eta_b)
    print(f"  eta {b0!r}, Cauchy part {b1!r}  [{time.time() - t0:.0f} s]", flush=True)
    # what the run stored: eta at p^; the Cauchy part at p^ (box2) or over the hull (box)
    st = bx["start"]
    der, where = (e1, "at p^") if ce["kind"] == "box2" else (b1, "over the hull")
    same = abs(e0 - st["eta"]) <= 1e-9 * st["eta"] and abs(der - st["eta_der"]) <= 1e-9 * st["eta_der"]
    print(f"  the run stored eta {st['eta']!r} at p^ and Cauchy part {st['eta_der']!r} {where}: reproduced {same}",
          flush=True)
    ok = ok and same
    by_rerun = False
    wide = max((r for r in covering if "h" in r), key=lambda r: float(r["h"]), default=None)
    if wide is not None:
        if mutate == "overlap":
            sys.exit("mutation overlap has no effect here: a stored row has the certificate's truncation")
        print(f"rerun, the lemma's row [{wide['name']}] (data_table's call; its own line follows):", flush=True)
        ctx.prec = PREC
        xb = [arb(v, arb(wide["h"])) for v in FPM.ZEROS[M]]
        _, _, _, eta_l = FPM.start_data_M(xb, M, wide["r0"], wide["arg"], +1, DATA_CAUCHY, diag=True, verbose=True)
        l0, l1 = bounds(eta_l)
        same_l = abs(l0 - wide["eta"]) <= 1e-9 * wide["eta"] and abs(l1 - wide["deta"]) <= 1e-9 * wide["deta"]
        below = max(e0, b0) <= l0 and max(e1, b1) <= l1
        print(f"  eta {l0!r}, Cauchy part {l1!r}; the row stored {wide['eta']!r}, {wide['deta']!r}: reproduced "
              f"{same_l}; the certificate's bounds are below the row's: {below}  [{time.time() - t0:.0f} s]",
              flush=True)
        ok = ok and same_l and below
    wide = max((r for r in around if "h" in r), key=lambda r: float(r["h"]), default=None)
    if covering == [] and wide is not None:
        # the lemma with the certificate's truncation (N, k and vanishing orders of the centre run at p^) on the row
        print(f"rerun, lemma FP({M}) with the certificate's N = {ce['N']} on the sets of [{wide['name']}] (its own "
              f"line follows):", flush=True)
        ctx.prec = PREC
        xb = [arb(v, arb(wide["h"])) for v in FPM.ZEROS[M]]
        try:
            _, _, _, eta_l = FPM.start_data_M(xb, M, wide["r0"], wide["arg"], +1, DATA_CAUCHY, centre=centre,
                                              diag=True, verbose=True)
            l0, l1 = bounds(eta_l)
            by_rerun = True
            print(f"  holds True: eta {l0!r}, Cauchy part {l1!r}; the certificate's bounds are below these: "
                  f"{max(e0, b0) <= l0 and max(e1, b1) <= l1}  [{time.time() - t0:.0f} s]", flush=True)
        except (RuntimeError, ValueError) as e:
            print(f"  holds False ({e})", flush=True)
        ok = ok and by_rerun
        # both truncations enclose y+(xi0; p^), so their start values must overlap
        ctx.prec = PREC
        xs = [arb(v) for v in FPM.ZEROS[M]]
        lem = centre_run(xs, M, wide["r0"], wide["arg"], True)
        q = [arb(v) for v in p]
        if mutate == "overlap":
            q[0] = q[0] + arb("1e-30")                 # MUTATION: the other truncation at a point 1e-30 away in A
        _, vals_l, _, _ = FPM.start_data_M(q, M, r0, arg, +1, ce["c"], centre=lem, diag=diag)
        over = all(a.overlaps(b) for a, b in zip(vals_c, vals_l))
        dist = max(float(abs(acb(a.mid()) - acb(b.mid())).upper()) for a, b in zip(vals_c, vals_l))

        def rad(vs):
            return max(max(float(v.real.rad()), float(v.imag.rad())) for v in vs)
        print(f"rerun, the start values at p^ with N = {centre['N']} (radius {rad(vals_c):.1e}) and with the row's N "
              f"= {lem['N']} (radius {rad(vals_l):.1e}): midpoints {dist:.1e} apart, all {len(vals_c)} overlap: {over}"
              f"  [{time.time() - t0:.0f} s]", flush=True)
        ok = ok and over and lem["N"] == wide["N"]
    print(f"RERUN: {'PASS' if ok else 'FAIL'}"
          + (f"; lemma FP({M}) with N = {ce['N']} holds on the sets of [{wide['name']}], which contain the "
             f"certificate's" if ok and by_rerun else ""))
    return ok, by_rerun


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    box = opts.get("box", os.path.join("data", "A_M2_box_M5_1e-10.json"))
    with open(box if os.path.isabs(box) or os.path.exists(box) else os.path.join(HERE, box)) as fh:
        bx = json.load(fh)
    if bx.get("kind") not in ("box2", "box"):
        sys.exit(f"{box} is not a box run of certify_M2.py or of certify_M.py")
    if "fixed_point_M.start_data_M" not in bx["start"]["start"]:
        sys.exit(f"{box}: its start data are not fixed_point_M.start_data_M's ({bx['start']['start']})")
    if not os.path.exists(os.path.join(RUNS, f"fixed_point_M_table_M{bx['M']}.json")):
        sys.exit(f"{box}: no stored runs of lemma FP({bx['M']}) in lemmas_M_runs/")
    rows, mutate = opts.get("rows", "all"), opts.get("mutate")
    if "selftest" in opts:
        ref = "data 5pi/16 h=1e-6"
        print(f"start_cover_M2 selftest on {box}, rows {rows}; the slack is against the lemma run [{ref}]  "
              f"[python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}]")
        clean = check(bx, rows, None, quiet=True, name=box)
        good, fired = clean["inside"], 0

        def line(tag, r, must):
            print(f"{tag}: {'COVERED' if r['covered'] else 'NOT COVERED'}, sets inside {r['inside']} ({must}); "
                  f"{r['why'].get(ref, 'no such lemma run')}")
        line("clean", clean, "must be inside")
        r = check(bx, rows, "edge0", quiet=True, name=box)
        good = good and r["inside"]
        line("edge0 (null)", r, "must be inside")
        for mu in MUTATIONS:
            r = check(bx, rows, mu, quiet=True, name=box)
            hit = (r["covered"] != clean["covered"]) if mu == "trunc" else (not r["inside"])
            fired += hit
            line(mu, r, "must change the verdict" if mu == "trunc" else "must not be inside")
        good = good and fired == len(MUTATIONS)
        print(f"SELFTEST: {'PASS' if good else 'FAIL'} (clean and edge0 inside, {fired} of {len(MUTATIONS)} mutations "
              f"give what they must)")
        sys.exit(0 if good else 1)
    res = check(bx, rows, None if mutate == "overlap" else mutate, name=box)
    covered = res["covered"]
    if "rerun" in opts and (covered or res["around"]):
        ok, by_rerun = rerun(bx, res, mutate)
        covered = ok and (covered or by_rerun)
    sys.exit(0 if covered else 1)


if __name__ == "__main__":
    main()
