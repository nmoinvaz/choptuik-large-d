#!/usr/bin/env python3
"""Do the stored runs of the start-value lemma FP(5) cover the start data of the M = 5 point certificate? (#484)

Lemma FP(M) (proof/A_lemmas_M.md 2.1) holds on a parameter set when fixed_point_M.run_M succeeds on it. Its stored
M = 5 runs are centred on the hier.py zero x^ = fixed_point_M.ZEROS[5]:
    table rows  lemmas_M_runs/fixed_point_M_table_M5.json  fixed_point_M.table: the point x^, the real box x^ +- 1e-6
                                                           and the complex box x^ +- 1e-3 (1 + i)
    data rows   lemmas_M_runs/fixed_point_M_data_M5.json   fixed_point_M.data_table -> start_data_M over x^ +- h,
                                                           h = 1e-9 and 1e-6. That call runs the fixed point over the
                                                           real box and over its Cauchy box (real parts x^ +- (h + c),
                                                           imaginary parts +- c, c = 1e-3) and raises unless both hold,
                                                           so a stored row is a run that held on both.
The point certificate (certify_M2.start_data2) calls start_data_M twice, at its own centre p^ and over the hull of its
box p(X). So it uses four parameter sets: p^, the hull, and the Cauchy box of each.

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
and (c) holds, and NOT COVERED (exit 1) otherwise.

    python3 start_cover_M2.py [--box=data/A_M2_box_M5_1e-10.json] [--rows=all|data|table]
    python3 start_cover_M2.py --mutate=widen|edge|centre|cauchy|ray|trunc    each must give NOT COVERED (exit 1)
        widen   the certificate's outer half-width h = 2e-6, past the lemma's largest real box
        edge    h = 9.993e-7: the hull passes the edge of x^ +- 1e-6 by 9e-11 in w/w_inf
        centre  p^ moved by 2e-6 in w/w_inf
        cauchy  the certificate's Cauchy radius 2e-3
        ray     the certificate's start ray 3pi/8 (r0 unchanged)
        trunc   the certificate's truncation N - 1
    python3 start_cover_M2.py --mutate=edge0   null mutation, h = 9.99e-7: must stay COVERED (exit 0)
    python3 start_cover_M2.py --selftest       the clean check, edge0 and the six mutations; exit 0 only if the first
                                               two are COVERED and the six are NOT COVERED
    python3 start_cover_M2.py --rerun          also reruns the fixed point on the certificate's sets (start_data2's
                                               calls) and on the covering data row (one process, a few minutes): the
                                               contraction constants, and the stored eta bounds reproduced
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
PREC = 320                    # the precision of start_data_M and of start_data2
DATA_CAUCHY = "1e-3"          # fixed_point_M.data_table's Cauchy radius (its literal argument; diag=True likewise)
MUTATIONS = ("widen", "edge", "centre", "cauchy", "ray", "trunc")


def pnames(M):
    return ["A", "w/w_inf"] + [f"{t}{nu}" for nu in range(3, M + 1, 2) for t in ("B", "C")]


# ---------------------------------------------------------------- the certificate's sets

def cert_data(bx, mutate=None):
    """the configuration and the four parameter sets of certify_M2.start_data2 for a stored C^2 run"""
    M = bx["M"]
    m = M + 1
    h, ws = bx["h"], list(bx["w"])
    if mutate == "widen":
        h = "2e-6"
    elif mutate == "edge":
        h = "9.993e-7"
    elif mutate == "edge0":
        h = "9.99e-7"
    # box2_run: load_chart and the widths at KPREC, then start_data2 at 320 bits
    ctx.prec = CM.KPREC
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    G = [[arb(x) for x in r] for r in bx["Gamma"]]
    widths = [arb(h), arb(h)] + [arb(ws[(i // 2) % len(ws)]) for i in range(M - 1)]
    ctx.prec = PREC
    p = CM.exact_p(bx["p"])
    if mutate == "centre":
        p[1] = p[1] + arb("2e-6")
    ctx.prec = PREC
    T = CM.T_matrix(Cm, M)
    X = [arb(0, wd) for wd in widths]
    hull = C2M.p_of_x(p, T, G, X, M)
    cstr = "2e-3" if mutate == "cauchy" else C2M.CAUCHY
    c = arb(cstr)

    def cbox(x):                                  # start_data_M's xc
        return [acb(arb(v) + arb(0, c), arb(0, c)) for v in x]
    st = bx["start"]
    words = st["start"]
    sha = words.split("sha256 ")[1].split(")")[0] if "sha256 " in words else None
    return {"M": M, "arg": "3pi/8" if mutate == "ray" else bx["path"][1], "r0": bx["path"][0],
            "diag": "diag splitting" in words, "N": st["N"] - (1 if mutate == "trunc" else 0), "c": cstr, "sha": sha,
            "p": [arb(v) for v in p], "hull": hull, "cb_p": cbox(p), "cb_hull": cbox(hull),
            "hw": max(float(abs(hull[i] - p[i]).upper()) for i in range(m)), "hw_stored": st.get("hull_halfwidth"),
            "h": h, "w": ws}


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


def check(bx, rows="all", mutate=None, quiet=False, name=""):
    """the tests (a), (b), (c); returns (covered, the certificate's data, the covering runs)"""
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
    say(f"certificate: ray {ce['arg']}, r0 {ce['r0']}, {'diag' if ce['diag'] else 'plain'} splitting, N {ce['N']}, "
        f"Cauchy radius {ce['c']}, box h {ce['h']}, w {ce['w']}")
    say("  p^ - x^:           " + "  ".join(f"{pn[i]} {float((ce['p'][i] - xh[i]).mid()):+.3e}" for i in range(M + 1)))
    say("  hull half-widths:  " + "  ".join(f"{pn[i]} {float(ce['hull'][i].rad()):.3e}" for i in range(M + 1)))
    say(f"  largest hull half-width about p^ {ce['hw']:.16e}, the run's stored value {ce['hw_stored']}: reproduced "
        f"{hw_ok}")
    say(f"(a) fixed_point_M.py sha256 {sha_now}, the certificate recorded {ce['sha']}: same file {sha_ok}; the "
        f"lemma's ray is the path's start ray: {ray_ok}")
    cert_sets = (("p^", "real", ce["p"]), ("hull", "real", ce["hull"]), ("Cauchy(p^)", "cauchy", ce["cb_p"]),
                 ("Cauchy(hull)", "cauchy", ce["cb_hull"]))
    covering = []
    for run in runs:
        cfg = same_config(ce, run)
        res = [(nm, *inside(S, [v[kind] for v in run["sets"]])) for nm, kind, S in cert_sets]
        allin = all(r[1] for r in res)
        if all(cfg.values()) and allin:
            covering.append(run)
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
    covered = bool(covering) and sha_ok and hw_ok and ray_ok and diag_ok
    say(f"START COVER M2 ({name}, rows {rows}, mutation {mutate}): "
        + (f"COVERED by {', '.join('[' + r['name'] + ']' for r in covering)}" if covered else "NOT COVERED"))
    return covered, ce, covering


# ---------------------------------------------------------------- the rerun

def rerun(bx, ce, covering):
    """the fixed point on the certificate's own sets, by start_data2's calls, and on the widest covering data row"""
    M, r0, arg, diag = ce["M"], ce["r0"], ce["arg"], ce["diag"]
    ok = True
    t0 = time.time()
    ctx.prec = PREC
    p = ce["p"]
    centre = FPM.run_M([acb(v.mid()) for v in p], M, r0, arg, None, verbose=False, kcap=40, diag=diag)
    eta_p = max(float(v.upper()) for v in FPM.eta_bounds(centre).values())
    print(f"rerun, centre run at p^: N {centre['N']}, k {centre['k']}, contraction {float(centre['lip'].upper()):.6f}, "
          f"holds {centre['ok']}, |eta| <= {eta_p:.3e}  [{time.time() - t0:.0f} s]", flush=True)
    ok = ok and centre["ok"] and centre["N"] == ce["N"]
    print("rerun, start_data_M at p^ (its own line follows):", flush=True)
    _, _, _, eta_c = FPM.start_data_M([arb(v) for v in p], M, r0, arg, +1, ce["c"], centre=centre, diag=diag,
                                      verbose=True)
    e0, e1 = float(eta_c[0].upper()), float(eta_c[1].upper())
    st = bx["start"]
    same = abs(e0 - st["eta"]) <= 1e-9 * st["eta"] and abs(e1 - st["eta_der"]) <= 1e-9 * st["eta_der"]
    print(f"  eta {e0!r}, Cauchy part {e1!r}; the run stored {st['eta']!r}, {st['eta_der']!r}: reproduced {same}  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    ok = ok and same
    print("rerun, start_data_M over the hull of p(X) (its own line follows):", flush=True)
    _, _, _, eta_b = FPM.start_data_M([arb(v) for v in ce["hull"]], M, r0, arg, +1, ce["c"], centre=centre, diag=diag,
                                      verbose=True)
    b0, b1 = float(eta_b[0].upper()), float(eta_b[1].upper())
    print(f"  eta {b0!r}, Cauchy part {b1!r}  [{time.time() - t0:.0f} s]", flush=True)
    wide = max((r for r in covering if "h" in r), key=lambda r: float(r["h"]), default=None)
    if wide is not None:
        print(f"rerun, the lemma's row [{wide['name']}] (data_table's call; its own line follows):", flush=True)
        ctx.prec = PREC
        xb = [arb(v, arb(wide["h"])) for v in FPM.ZEROS[M]]
        _, _, _, eta_l = FPM.start_data_M(xb, M, wide["r0"], wide["arg"], +1, DATA_CAUCHY, diag=True, verbose=True)
        l0, l1 = float(eta_l[0].upper()), float(eta_l[1].upper())
        same_l = abs(l0 - wide["eta"]) <= 1e-9 * wide["eta"] and abs(l1 - wide["deta"]) <= 1e-9 * wide["deta"]
        below = max(e0, b0) <= l0 and max(e1, b1) <= l1
        print(f"  eta {l0!r}, Cauchy part {l1!r}; the row stored {wide['eta']!r}, {wide['deta']!r}: reproduced "
              f"{same_l}; the certificate's bounds are below the row's: {below}  [{time.time() - t0:.0f} s]", flush=True)
        ok = ok and same_l and below
    print(f"RERUN: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    box = opts.get("box", os.path.join("data", "A_M2_box_M5_1e-10.json"))
    with open(box if os.path.isabs(box) or os.path.exists(box) else os.path.join(HERE, box)) as fh:
        bx = json.load(fh)
    if bx.get("kind") != "box2":
        sys.exit(f"{box} is not a C^2 box run of certify_M2.py")
    rows = opts.get("rows", "all")
    if "selftest" in opts:
        clean, _, _ = check(bx, rows, None, quiet=True, name=box)
        null, _, _ = check(bx, rows, "edge0", quiet=True, name=box)
        print(f"clean: {'COVERED' if clean else 'NOT COVERED'} (must be COVERED)")
        print(f"edge0 (null): {'COVERED' if null else 'NOT COVERED'} (must be COVERED)")
        fired = 0
        for mu in MUTATIONS:
            cov, _, _ = check(bx, rows, mu, quiet=True, name=box)
            fired += not cov
            print(f"{mu}: {'COVERED' if cov else 'NOT COVERED'} (must be NOT COVERED)")
        good = clean and null and fired == len(MUTATIONS)
        print(f"SELFTEST: {'PASS' if good else 'FAIL'} (clean and edge0 covered, {fired} of {len(MUTATIONS)} mutations "
              f"not covered)")
        sys.exit(0 if good else 1)
    covered, ce, covering = check(bx, rows, opts.get("mutate"), name=box)
    if "rerun" in opts and covered:
        covered = rerun(bx, ce, covering)
    sys.exit(0 if covered else 1)


if __name__ == "__main__":
    main()
