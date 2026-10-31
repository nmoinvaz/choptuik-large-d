#!/usr/bin/env python3
"""Post-freeze follow-up of the planted-truth test at K = 20 (notes/PLANTED_SELECTION.md, section 12): sigma_20
(scan41J1250: K 41, J 1250, 1280 bits at wf 0.99, 1.00, 1.01, the node data; pilot (b) in scan41J1500: J 1500, 1280
bits at wf 1.00; both copied from the raw-data repository's T89-mac). The procedure and its pass criteria are section
12.0 with its amendment of items 1(b) and 3 (2026-10-31); each stage exits non-zero when one of its checks fails. The
earlier orders' scripts provide the shared pieces: planted_k14.py (Borel-Pade sums, zeros, the projection model, the
kbar minima of G), planted_k15.py ... planted_k19.py (their records and stored checks); the K = 19 node data and records
are planted_k19.py's under its --nodes mode (section 11.0's second amendment), which this script selects at import.

  converge    sigma_20's convergence: (a) err20 = the plateau error sigma_err_fit.py recomputes from fit_info (a1 from
              the 960-bit scan39J1000 file at the same wf) plus the wf 1.00 pilot difference |sigma_20(a) - sigma_20(b)|,
              at every node; (b) sigma_0..19 at wf 1.00 bitwise scan39J1250's pilot (b), the sigma_0..18 differences from
              the 960-bit values at 0.99 and 1.01 recorded (amended item 1(b)); (c) pilot (b)'s sigma_17..19 within (a)'s
              plateau errors, its sigma_20 difference recorded; (d) the ratio law at k = 18; the spread in wf.
              --mutate=err|low|pilot|ratiolaw|amend breaks one input on purpose, and the matching check must fire
  plants      N1, N2, the 28 arc plants, C-EXP and the 13 REFIT plants at K = 18, 19, 20 (planted_results_k20.jsonl,
              resumable); their K = 18, 19 estimates must equal the K = 19 records; N1 recovered at K = 20, N2 nothing
  real        the real sigma_0..20 ("N0 scan41J1250"); its K = 18, 19 estimates must equal the K = 19 record of the same
              name (the same sigma_0..19 through the same code, amended item 3); the sigma_20 sign flip (and a 1e-3
              scale) at wf 1.00 as mutation controls
  ring        the K = 19 completions of section 11.6 (C19-*), regenerated: their sigma_20, tau_20 against the measured
              ones and the detectors at K = 20 beyond the real data's, read with and without D-bracket (section 12.0
              item 4; no ring survived sigma_19, so none is carried over); also the BP calibration family of section
              11.5 (sixteen sums) and section 11.6's ring-free C19-BP sums
  calib       K = 20 calibration on the arc plants, C-EXP, REFIT and the BP family (twenty sums), and the zero each
              estimator implies; the K = 19 calibration from the new records must reproduce section 11.5
  recomplete  the recipes glued at sigma_20 (tail from k = 21) and the Borel-Pade sums [m/(20 - m)], m = 3..10
  project     the order-20 projection (est_k = w* + c r^k through k = 18, 19) against the real arg_20, back-tests, noise
  combine     section 12.0 item 7: the K = 20 estimator range, G over the minima of its objective, the headline
  table       the stored checks, as JSON

    nice -n 10 python3 planted_k20.py STAGE [--scratch DIR] [--mutate=...] [--data DIR]

--data DIR reads the scans from DIR/scan41J1250 and DIR/scan41J1500 and writes the records and checks there (the dry
run of section 12.0 item 0, planted_k20_dry.py); the default is this directory. A file whose stored sigma_err is finite
(the dry run's) is read with those errors; the real 1280-bit files store inf and are recomputed.
"""
import argparse
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted_k13 as K13  # noqa: E402
import planted_k14 as K14  # noqa: E402
import planted_k15 as K15  # noqa: E402
import planted_k16 as K16  # noqa: E402
import planted_k17 as K17  # noqa: E402
import planted_k18 as K18  # noqa: E402
import planted_k19 as K19  # noqa: E402
import planted_plants as PP  # noqa: E402
import planted_post as PO  # noqa: E402
import planted_run as R  # noqa: E402
import sigma_err_fit as SF  # noqa: E402
from arg_crossing import taus  # noqa: E402

# the K = 19 data and records are those of section 11.0's second amendment: the 1280-bit node data
K19.NODES = os.path.join(K19.HERE, "scan41J1250")
K19.REAL19 = "N0 scan41J1250"
SCAN20 = os.path.join(HERE, "scan41J1250")
PILOT20 = os.path.join(HERE, "scan41J1500")
A1SCAN = os.path.join(HERE, "scan39J1000")                              # a1 for the recomputed errors (960 bits)
LOW1280 = os.path.join(HERE, "scan39J1250")                             # T88's pilot (b), K 39: item 1(b)'s bitwise reference
REAL19 = K19.REAL19
REAL20 = "N0 scan41J1250"
OUT19 = K19.OUT19
OUT20 = os.path.join(HERE, "planted_results_k20.jsonl")
CHECKS20 = os.path.join(HERE, "planted_checks_k20.json")
KS20 = (18, 19, 20)
EST = K13.EST
ARC_FAMS = K14.ARC_FAMS
SHAPES = K14.SHAPES
HYPOTHESES = (1.0, 0.999996, 0.999975)      # zero at 1, at the lower end of the K = 19 range, of the headline
REPRO_TOL = 1e-9
BP_CAL_OLD = K19.BP_CAL                                                  # section 11.5's BP family (sixteen sums)
BP_CAL = BP_CAL_OLD + ("C19-BP 6-13", "C19-BP 7-12", "C19-BP 8-11", "C19-BP 9-10")  # plus section 11.6's ring-free sums
RINGS18 = ()                                                            # no ring survived sigma_19 (section 11.4)
K19_RANGE = (0.999975, 1.000050)                                        # the K = 19 headline (section 11.8)
BRACKET = "D-bracket"                                                   # left out in the second reading (section 12.0 item 4)
FAILED = []


def configure(data):
    global SCAN20, PILOT20, OUT20, CHECKS20
    SCAN20 = os.path.join(data, "scan41J1250")
    PILOT20 = os.path.join(data, "scan41J1500")
    OUT20 = os.path.join(data, "planted_results_k20.jsonl")
    CHECKS20 = os.path.join(data, "planted_checks_k20.json")


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILED.append(msg)
    return ok


def use20():
    R.OUT, R.KS = OUT20, KS20
    R.DONE.clear()
    R.DONE.update(R.done_names())


def load_checks():
    return json.load(open(CHECKS20)) if os.path.exists(CHECKS20) else {}


def save_checks(key, val):
    c = load_checks()
    c[key] = val
    with open(CHECKS20, "w") as f:
        json.dump(c, f, default=R.jdefault, indent=1)


def finite_errs(r):
    return all(isinstance(e, (int, float)) and np.isfinite(e) for e in r["sigma_err"][:21])


def errs_of(r):
    """the plateau errors of a scan file at k = 0..20: the stored ones when finite (the dry run), else recomputed from
    fit_info by sigma_err_fit.py with a1 from the 960-bit scan39J1000 file at the same wf (the real 1280-bit files)"""
    if finite_errs(r):
        return [float(e) for e in r["sigma_err"][:21]]
    p960 = K14.raw_scan(A1SCAN, "*_p960.json")
    if r["wf"] not in p960:
        raise SystemExit(f"{A1SCAN}: no 960-bit file at wf {r['wf']} for a1")
    rec = SF.recompute(r, SF.a1_of(p960[r["wf"]]))
    return [float(rec[k][2]) for k in range(21)]


def node_files():
    am = K14.raw_scan(SCAN20, "*.json")
    for w in PP.NODES:
        if w not in am:
            raise SystemExit(f"{SCAN20}: no file at wf {w}")
        if (am[w]["J"], am[w]["prec"]) != (1250, 1280) or len(am[w]["sigma_str"]) < 21:
            raise SystemExit(f"{SCAN20}: the file at wf {w} is not a J 1250 / 1280-bit file with sigma_0..20")
    return am


def pilot_file():
    pil = K14.raw_scan(PILOT20, "*.json")
    if 1.0 not in pil or (pil[1.0]["J"], pil[1.0]["prec"]) != (1500, 1280) or len(pil[1.0]["sigma_str"]) < 21:
        raise SystemExit(f"{PILOT20}: no J 1500 / 1280-bit file at wf 1.0 with sigma_0..20")
    return pil


def pilot_d20(am, pil):
    return float(abs(K14.sigmas(am[1.0])[20] - K14.sigmas(pil[1.0])[20]))


def node_errs(am, pil):
    """({wf: [err_0..err_20]}, the pilot difference): item 1(a)'s rule, err20 = plateau error + |sigma_20(a) - sigma_20(b)|
    at wf 1.00, added at every node"""
    pd = pilot_d20(am, pil)
    errs = {}
    for w in PP.NODES:
        errs[w] = errs_of(am[w])
        errs[w][20] += pd
    return errs, pd


def load_real20():
    mp.mp.dps = PP.DPS
    am, pil = node_files(), pilot_file()
    errs, _ = node_errs(am, pil)
    return {float(w): (K14.sigmas(am[w])[:21], errs[w]) for w in PP.NODES}


def load_real19():
    return K19.load_real19()                  # the 1280-bit node data, sigma_0..19 (planted_k19.py --nodes)


def k19_records():
    return {r["name"]: r for r in (json.loads(line) for line in open(OUT19) if line.strip())}


def new_records():
    use20()
    return {r["name"]: r for r in R.records()}


# ---------------------------------------------------------------- converge
def stage_converge(mutate=None):
    mp.mp.dps = 50
    am, pil = node_files(), pilot_file()
    low = K14.raw_scan(LOW1280, "*.json")
    p960 = K14.raw_scan(A1SCAN, "*_p960.json")
    print(f"node data: J 1250 / 1280-bit files at wf {sorted(am)} from {SCAN20}: "
          + ", ".join(f"K {r['K']} at wf {w}, {len(r['sigma_str'])} coefficients, stored errors "
                      f"{'finite' if finite_errs(r) else 'inf (recomputed)'}" for w, r in sorted(am.items())))
    print(f"  pilot (b), J 1500 at wf {sorted(pil)} from {PILOT20}; scan39J1250 (K 39, J 1250, 1280 bits) at wf {sorted(low)}; "
          f"scan39J1000 (960 bits) at wf {sorted(p960)}")
    if mutate == "low":
        x = mp.mpf(am[1.0]["sigma_str"][18][0]) * (1 + mp.mpf("1e-15"))
        am[1.0]["sigma_str"][18][0] = mp.nstr(x, 20)
    elif mutate == "pilot":
        x = mp.mpf(pil[1.0]["sigma_str"][17][1]) * (1 + mp.mpf("1e-10"))
        pil[1.0]["sigma_str"][17][1] = mp.nstr(x, 20)
    elif mutate == "amend":
        x = mp.mpf(pil[1.0]["sigma_str"][20][0]) * (1 + mp.mpf("1e-9"))
        pil[1.0]["sigma_str"][20][0] = mp.nstr(x, 20)
    elif mutate == "ratiolaw":
        am[1.01]["sigma_str"][20] = [mp.nstr(mp.mpf(x) * mp.mpf("1.1"), 20) for x in am[1.01]["sigma_str"][20]]
    elif mutate not in (None, "err"):
        raise SystemExit(f"unknown mutation {mutate}")
    errs, pd = node_errs(am, pil)
    if mutate == "err":
        errs[1.0][20] *= 1e14
    out = {"nodes": {}, "pilot_d20": pd}
    print(f"(a) err20 = plateau error + the wf 1.00 pilot difference |sigma_20(a) - sigma_20(b)| = {pd:.3e}, against the "
          "1e-10 relative requirement:")
    for w in sorted(am):
        s = K14.sigmas(am[w])
        e20 = errs[w][20]
        rel = e20 / float(abs(s[20]))
        out["nodes"][str(w)] = {"sigma20": am[w]["sigma_str"][20], "err20": e20, "err20_rel": rel, "err20_plateau": e20 - pd,
                                "err19": errs[w][19], "err18": errs[w][18], "err17": errs[w][17]}
        check(rel <= 1e-10, f"wf {w}: sigma_20 = {mp.nstr(s[20], 20)}, err20 {e20:.3g} = plateau {e20 - pd:.3g} + pilot "
                            f"difference {pd:.3g} ({rel:.2e} relative, margin {1e-10 / rel:.2f})")
    print("(b) sigma_0..19 at wf 1.00 against scan39J1250's pilot (b) (K 39, J 1250, 1280 bits), bitwise:")
    same = [am[1.0]["sigma_str"][k] == low[1.0]["sigma_str"][k] for k in range(20)]
    out["nodes"]["1.0"]["low_bitwise"] = all(same)
    check(all(same), f"wf 1.0: sigma_0..19 bitwise equal ({sum(same)}/20)")
    print("  at wf 0.99 and 1.01: sigma_0..18 against the 960-bit scan39J1000 values, |difference| / that file's plateau "
          "error, with the S_1 family's Mf picks (960, 1280); recorded, not a check (the amendment of item 1(b)):")
    for w in sorted(am):
        if w == 1.0:
            continue
        s, s0 = K14.sigmas(am[w]), K14.sigmas(p960[w])
        rows = []
        for k in range(19):
            e0, kk = p960[w]["sigma_err"][k], 2 * k + 1
            d = float(abs(s[k] - s0[k]))
            rows.append({"k": k, "diff": d, "err960": e0, "ratio": d / e0 if e0 else None,
                         "mf": [p960[w]["fit_info"].get(f"{kk}_1", [None])[0], am[w]["fit_info"].get(f"{kk}_1", [None])[0]]})
        out["nodes"][str(w)]["low_vs_960"] = rows
        over = [x for x in rows if x["ratio"] is not None and x["ratio"] > 1]
        print(f"  wf {w}: " + " ".join(f"k{x['k']}:{x['ratio']:.2f}" for x in rows if x["ratio"] is not None) + "; over 1: "
              + (", ".join(f"k {x['k']} ({x['ratio']:.2f}, Mf {x['mf'][0]} and {x['mf'][1]})" for x in over) or "none"))
    print("(c) pilot (b) at wf 1.00 (J 1500, 1280 bits) against (a): sigma_17..19 within (a)'s plateau errors; the sigma_20 "
          "difference is the term of (a), recorded:")
    sa, sb = K14.sigmas(am[1.0]), K14.sigmas(pil[1.0])
    ea = errs_of(am[1.0])
    rows = {k: (float(abs(sb[k] - sa[k])), ea[k]) for k in (17, 18, 19)}
    out["pilot"] = {"d20": pd, "d20_over_err20a": pd / ea[20] if ea[20] else None, "low": rows}
    check(all(d <= e for d, e in rows.values()), "sigma_17..19 of (b) against (a): " + ", ".join(
        f"k {k}: {d:.1e} (err {e:.1e}, ratio {d / e:.2f})" for k, (d, e) in rows.items()))
    print(f"  sigma_20: |a - b| = {pd:.3e}, {pd / ea[20]:.3f} of (a)'s plateau err20 {ea[20]:.3e} (recorded; ruling 2)")
    print("(d) the ratio law through sigma_20: second difference of sigma_k/sigma_{k+1}, k = 7..18 (D-ratiolaw <= 0.05):")
    out["ratiolaw_d2"] = {}
    for w in sorted(am):
        s = K14.sigmas(am[w])
        rat = [complex(s[k] / s[k + 1]) for k in range(20)]
        d2 = [abs(rat[k + 1] - 2 * rat[k] + rat[k - 1]) for k in range(7, 19)]
        out["ratiolaw_d2"][str(w)] = d2
        check(d2[-1] <= 0.05, f"wf {w}: |d2| k = 7..18: " + " ".join(f"{x:.4f}" for x in d2)
              + f"; sigma_19/sigma_20 = {rat[19]:.4f}")
    print("spread in wf (context): the change of sigma_k across the nodes, against its error")
    out["spread"] = {}
    for k in (18, 19, 20):
        s = {w: K14.sigmas(am[w])[k] for w in PP.NODES}
        d1 = abs(s[1.01] - s[0.99]) / 2
        d2 = abs(s[1.01] - 2 * s[1.0] + s[0.99])
        e = max(errs[w][k] for w in PP.NODES)
        out["spread"][str(k)] = {"half_range_rel": float(d1 / abs(s[1.0])), "d2_over_d1": float(d2 / d1),
                                 "err_over_d1": float(e / d1)}
        print(f"  k = {k}: |sigma_k(1.01) - sigma_k(0.99)|/2 = {float(d1 / abs(s[1.0])):.4f} |sigma_k(1.00)|; "
              f"second difference / first = {float(d2 / d1):.4f}; max err_k / first difference = {float(e / d1):.1e}")
    if not mutate:
        save_checks("K20 converge", out)
    return out


# ---------------------------------------------------------------- plants
def sigma_dev(nodes, real20):
    return {str(w): [float(abs(nodes[w][k] - real20[w][0][k]) / abs(real20[w][0][k])) for k in (19, 20)]
            for w in PP.NODES}


def stage_plants(scratch):
    use20()
    real20 = load_real20()
    devs = load_checks().get("K20 plant sigma19-20 deviation", {})
    for wt in PP.OMEGA_T:
        R.run_plant(f"N1 lone zero wt={wt}", "N1", PP.lone_zero(wt), float(wt), scratch)
    R.run_plant("N2 S=1", "N2", PP.no_zero("one"), None, scratch)
    R.run_plant("N2 S=exp(bx)", "N2", PP.no_zero("exp"), None, scratch)
    for tag in ARC_FAMS:
        if all(f"{tag} wt={wt}" in R.DONE for wt in PP.OMEGA_T):
            continue
        fam = PP.ArcFamily(tag.split()[-1] if tag != "P-ARC" else "arc")
        for wt in PP.OMEGA_T:
            name = f"{tag} wt={wt}"
            if name in R.DONE:
                continue
            nodes = fam.nodes(wt)
            devs[name] = sigma_dev(nodes, real20)
            save_checks("K20 plant sigma19-20 deviation", devs)
            R.run_plant(name, tag.split()[0], nodes, float(wt), scratch,
                        info={"fit_resid_max": {str(w): fam.resid[w] and max(fam.resid[w]) for w in PP.NODES}})
    if "C-EXP" not in R.DONE:
        nodes, _, info, truth = PP.completion("C-EXP")
        devs["C-EXP"] = sigma_dev(nodes, real20)
        save_checks("K20 plant sigma19-20 deviation", devs)
        R.run_plant("C-EXP", "C", nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    for wt in R.REFIT_WT:
        name = f"REFIT P-ARC wt={wt}"
        if name in R.DONE:
            continue
        nodes, resid = K14.refit_nodes(wt)
        devs[name] = sigma_dev(nodes, real20)
        save_checks("K20 plant sigma19-20 deviation", devs)
        R.run_plant(name, "REFIT", nodes, float(wt), scratch, info={"fit_resid_max": resid}, linear=False)
    plants_check()


def plants_check():
    old, new = k19_records(), new_records()
    names = [n for n in new if not n.startswith(("N0", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"))]
    print(f"\n{len(names)} plants with K = 20 records")
    worst = (0.0, None)
    for n in names:
        for K in (18, 19):
            d, k = K14.max_diff(K14.est_vector(new[n], K), K14.est_vector(old[n], K))
            if d > worst[0]:
                worst = (d, f"{n} K{K} {k}")
    check(worst[0] <= REPRO_TOL, f"regenerated plants reproduce the stored K = 18, 19 estimates: largest difference "
                                 f"{worst[0]:.1e} ({worst[1]})")
    n1 = [new[n] for n in names if n.startswith("N1")]
    errs = [abs(R.grade(r, 20)["est"][e] - r["truth"]) for r in n1 for e in ("arg", "ratio", "pade", "G")
            if R.grade(r, 20)["est"][e] is not None]
    nnone = sum(R.grade(r, 20)["est"][e] is None for r in n1 for e in ("arg", "ratio", "pade", "G"))
    check(len(n1) == 7 and nnone == 0 and max(errs) <= 1e-6,
          f"N1 at K = 20: {len(n1)} lone zeros, worst error {max(errs):.1e} over arg, ratio, Pade, G ({nnone} missing)")
    ctl = [abs(R.grade(r, 20)["est"]["arg"] - (r["truth"] + 0.002)) <= 1e-6 for r in n1]
    check(not any(ctl), "control: N1 graded against w_t + 0.002 fails (as it must)")
    n2 = [new[n] for n in names if n.startswith("N2")]
    printed = [(r["name"], e) for r in n2 for e, v in R.grade(r, 20)["est"].items() if v is not None]
    check(len(n2) == 2 and not printed, f"N2 at K = 20: no estimator prints a number ({printed or 'none'})")
    fams = {}
    for n in names:
        fams.setdefault(new[n]["family"], []).append(n)
    print("  records per family:", {f: len(v) for f, v in fams.items()})
    save_checks("K20 plants check", {"repro_worst": worst, "n1_worst": max(errs), "n2_printed": printed,
                                     "families": {f: len(v) for f, v in fams.items()}})


# ---------------------------------------------------------------- real data
def stage_real(scratch):
    use20()
    real = load_real20()
    R.run_plant(REAL20, "N0", {wf: real[wf][0] for wf in PP.NODES}, None, scratch)
    flip = {wf: list(real[wf][0]) for wf in PP.NODES}
    flip[1.0][20] = -flip[1.0][20]
    R.run_plant(f"{REAL20} X1 flip sigma_20", "N0 mutation", flip, None, scratch)
    sc = {wf: list(real[wf][0]) for wf in PP.NODES}
    sc[1.0][20] = sc[1.0][20] * (1 + mp.mpf("1e-3"))
    R.run_plant(f"{REAL20} X2 scale sigma_20", "N0 mutation", sc, None, scratch)
    old, new = k19_records(), new_records()
    rec = new[REAL20]
    for K in (18, 19):
        d, k = K14.max_diff(K14.est_vector(rec, K), K14.est_vector(old[REAL19], K))
        check(d <= REPRO_TOL, f"K = {K}: the K = 20 record's estimates equal the K = 19 record's ({REAL19}, the same "
                              f"sigma_0..19; largest difference {d:.1e}{', ' + k if k else ''})")
    for K in KS20:
        g = R.grade(rec, K)["est"]
        print(f"  K = {K}: " + ", ".join(f"{e} {v:.7f}" if v is not None else f"{e} None" for e, v in g.items())
              + f"; detectors {R.detectors(rec, K)}")
    e = rec["est"]["20"]
    print("  arg crossings k = 4..20:", {int(k): round(v, 7) for k, v in e["arg"].items()})
    print("  ratio estimates k = 9..20:", {int(k): round(v, 6) for k, v in e["ratio"].items()})
    print("  Pade central", e["pade_central"], "values", [round(v, 6) if v else v for v in e["pade_central_vals"]],
          "range", e["pade_central_range"], "mid", e["pade_central_mid"])
    print("  all Pade crossings K = 20:", {k: round(v, 6) if v else v for k, v in e["pade"].items()})
    print("  G K = 19, 20 via G_prev/G:", e.get("G_prev"), e.get("G"), "kbar", e.get("G_kbar"))
    det = set(R.detectors(rec, 20))
    for n in (f"{REAL20} X1 flip sigma_20", f"{REAL20} X2 scale sigma_20"):
        m = new[n]
        extra = sorted(set(R.detectors(m, 20)) - det)
        g, g0 = R.grade(m, 20)["est"], R.grade(rec, 20)["est"]
        shift = max((abs(g[x] - g0[x]) if None not in (g[x], g0[x]) else float("inf")) for x in EST)
        check(bool(extra) if "flip" in n else True, f"mutation {n.split()[-2]} sigma_20: new detectors {extra or 'none'}, "
                                                     f"largest shift {shift:.1e}")
    mp.mp.dps = 50
    rt = taus(list(real[1.0][0][:21]))
    ce = {}
    for k in (18, 19, 20):
        x0 = (k - 1) * rt[k - 1] / (k * rt[k])
        ce[k] = complex(-k * rt[k] * x0 ** k)
    print("  effective exponent c at wf 1.00 (two orders): " + ", ".join(f"k {k}: {v.real:.3f}{v.imag:+.3f}i"
                                                                          for k, v in ce.items()))
    save_checks("K20 real", {"est": {str(K): R.grade(rec, K)["est"] for K in KS20},
                             "detectors": {str(K): R.detectors(rec, K) for K in KS20},
                             "c_eff_wf1": {str(k): [v.real, v.imag] for k, v in ce.items()}})


# ---------------------------------------------------------------- ring test
def c19_completions():
    """section 11.6's K = 19 completions (tail from k = 20), regenerated from the K = 19 node data (no ring survived
    sigma_19, so nothing is carried over from section 11.4); and the BP calibration sums of K = 13..18"""
    real19 = load_real19()
    stored19 = json.load(open(K19.CHECKS19))["K19 recomplete"]
    stored18 = json.load(open(K18.CHECKS18))["K18 recomplete"]
    stored17 = json.load(open(K17.CHECKS17))["K17 recomplete"]
    stored16 = json.load(open(K16.CHECKS16))["K16 recomplete"]
    stored15 = json.load(open(K15.CHECKS15))["K15 recomplete"]
    stored14 = json.load(open(K14.CHECKS14))["K14 recomplete"]
    out = {}
    for kind in ("C-W1", "C-W2", "C-W3"):
        nodes, funcs, info, truth = PP.completion(kind, real=real19, kc=19)
        out[kind.replace("C-", "C19-")] = (nodes, funcs, truth, stored19[kind.replace("C-", "C19-")])
    for kind in SHAPES:
        name = f"C19-ARC {kind}"
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{stored19[name]['best_wt']:.5f}", kc=19, real=real19, funcs=funcs)
        out[name] = (nodes, funcs, truth, stored19[name])
    for m in (3, 4, 5, 6, 7, 8, 9):
        name = f"C19-BP [{m}/{19 - m}]"
        nodes, funcs, info = K14.borel_pade(real19, m, 19 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored19[name])
    real18 = K19.load_real18()
    for m in (6, 7, 8, 9):                       # the calibration family of section 11.5
        name = f"C18-BP [{m}/{18 - m}]"
        nodes, funcs, info = K14.borel_pade(real18, m, 18 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored18[name])
    real17 = K18.load_real17()
    for m in (6, 7, 8):
        name = f"C17-BP [{m}/{17 - m}]"
        nodes, funcs, info = K14.borel_pade(real17, m, 17 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored17[name])
    real16 = K17.load_real16()
    for m in (6, 7, 8):
        name = f"C16-BP [{m}/{16 - m}]"
        nodes, funcs, info = K14.borel_pade(real16, m, 16 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored16[name])
    real15 = K16.load_real15()
    for m in (6, 7):
        name = f"C15-BP [{m}/{15 - m}]"
        nodes, funcs, info = K14.borel_pade(real15, m, 15 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored15[name])
    real14 = K15.load_real14()
    for m in (5, 6):
        name = f"C14-BP [{m}/{14 - m}]"
        nodes, funcs, info = K14.borel_pade(real14, m, 14 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored14[name])
    real13 = PP.load_real(K13.SCAN13)
    stored13 = json.load(open(K14.CHECKS14))["K14 recomplete"]
    for m in (5, 6):
        name = f"C13-BP [{m}/{13 - m}]"
        nodes, funcs, info = K14.borel_pade(real13, m, 13 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored13[name])
    for name, (_, _, truth, st) in out.items():
        check(abs(truth["im_x"] - st["truth"]) <= REPRO_TOL,
              f"{name} regenerated: zero at {truth['im_x']:.9f} (stored {st['truth']:.9f})")
    return out


def rec_name(name):
    """the record name (no brackets or slashes: it becomes a scratch directory, read by glob)"""
    return name.replace("[", "").replace("]", "").replace("/", "-")


def family_of(name):
    if name.startswith("C19"):
        return "C19 (ring test)"
    if name in RINGS18:
        return "C18-BP (ring test)"
    return name[:6] + " (calibration)"


def stage_ring(scratch):
    use20()
    real = load_real20()
    mp.mp.dps = 50
    rt = {wf: taus(list(real[wf][0][:21])) for wf in PP.NODES}
    comps = c19_completions()
    for name, (nodes, _, truth, _) in comps.items():
        R.run_plant(rec_name(name), family_of(name), nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    new = new_records()
    real_rec = new[REAL20]
    real_det = set(R.detectors(real_rec, 20))
    a20 = R.grade(real_rec, 20)["est"]["arg"]
    print("real: sigma_20 / tau_20 at the nodes: " + "; ".join(
        f"{wf}: {mp.nstr(real[wf][0][20], 8)} / {mp.nstr(rt[wf][20], 8)}" for wf in PP.NODES))
    print(f"real arg_20 = {a20:.7f}; real K = 20 detectors: {sorted(real_det)}\n")
    out = {"simple-zero law from tau_18, tau_19": {"tau20_rel": K14.simple_zero_rel(rt, 20)},
           "simple-zero law from tau_17, tau_18 (tau_19, for comparison)": {"tau19_rel": K14.simple_zero_rel(rt, 19)}}
    print("| completion | its zero | zeros in |x| < 7.3 / 12 / 16 | sigma_20 rel. error (0.99 / 1.00 / 1.01) | "
          "tau_20 rel. error | its arg_20 | K = 20 detectors beyond the real data's | survives | without D-bracket |")
    print("|---|---|---|---|---|---|---|---|---|")
    sz = " / ".join(f"{x:.1e}" for x in K14.simple_zero_rel(rt, 20))
    print(f"| simple-zero law from tau_18, tau_19 | - | - | - | {sz} | - | - | - | - |")
    old19 = k19_records()
    for name, (nodes, _, truth, st) in comps.items():
        mp.mp.dps = 50
        se = [float(abs(nodes[wf][20] - real[wf][0][20]) / abs(real[wf][0][20])) for wf in PP.NODES]
        te = [float(abs(taus(list(nodes[wf][:21]))[20] - rt[wf][20]) / abs(rt[wf][20])) for wf in PP.NODES]
        rec = new[rec_name(name)]
        arg = R.grade(rec, 20)["est"]["arg"]
        extra = sorted(set(R.detectors(rec, 20)) - real_det)
        extra_nb = [d for d in extra if d != BRACKET]                   # the second reading of section 12.0 item 4
        kq = int(name[1:3])                       # the order whose real coefficients the completion carries
        if kq == 19:                              # the K = 19 node data: its K = 19 estimates are the real record's
            same = K14.max_diff(K14.est_vector(rec, 19), K14.est_vector(real_rec, 19))[0]
            label = "its K = 19 estimates are the real ones"
        else:                                     # glued below K = 19: its K = 18, 19 estimates are the stored K = 19 record's
            old = old19[rec_name(name)]
            same = max(K14.max_diff(K14.est_vector(rec, K), K14.est_vector(old, K))[0] for K in (18, 19))
            label = "its K = 18, 19 estimates reproduce the stored K = 19 record"
        zc = " / ".join(str(st.get(f"zeros_lt_{r}", "-")) for r in (7.3, 12, 16))
        ring = (st.get("zeros_lt_12") or 0) > 1
        out[name] = {"sigma20_rel": se, "tau20_rel": te, "arg20": arg, "extra_detectors": extra, "survives": not extra,
                     "survives_without_bracket": not extra_nb, "truth": truth["im_x"], "zeros": zc, "ring": ring,
                     ("K19_equals_real" if kq == 19 else "K18_19_equal_stored"): same,
                     "est20": R.grade(rec, 20)["est"]}
        print(f"| {name} | {truth['im_x']:.6f} | {zc} | {' / '.join(f'{x:.1e}' for x in se)} | "
              f"{' / '.join(f'{x:.1e}' for x in te)} | {'none' if arg is None else f'{arg:.6f}'} | "
              f"{', '.join(extra) or '-'} | {'yes' if not extra else 'no'} | {'yes' if not extra_nb else 'no'} |")
        check(same <= REPRO_TOL, f"{name}: {label} (difference {same:.1e})")
        if ring and not extra:
            print(f"  FINDING: {name} carries a ring ({zc} zeros) and survives at K = 20")
        elif ring and not extra_nb:
            print(f"  NOTE: {name} carries a ring ({zc} zeros), is rejected by D-bracket alone and survives without it")
    for r in RINGS18:
        o = out[r]
        print(f"\n{r}, a ring that survived sigma_19: survives sigma_20: {o['survives']} (without D-bracket: "
              f"{o['survives_without_bracket']}; extra detectors {o['extra_detectors'] or 'none'}; its sigma_20 is off by "
              f"{max(o['sigma20_rel']):.1e} relative)")
    save_checks("K20 ring test", out)


# ---------------------------------------------------------------- calibration
def calib_rows(by, real_est, K):
    rows = K14.calib_rows(by, real_est, K)
    for label, names in (("BP family (16)", BP_CAL_OLD), ("BP family (20)", BP_CAL)):
        bp = [by[n] for n in names if n in by]
        cells = {}
        for e in EST:
            v = [R.grade(r, K)["err"][e] for r in bp if R.grade(r, K)["err"][e] is not None]
            cells[e] = {"lo": min(v), "hi": max(v), "n": len(v), "zero": (real_est[e] - max(v), real_est[e] - min(v))} \
                if v and real_est[e] is not None else {"n": len(v)}
        rows[label] = {"n": len(bp), "cells": cells}
    return rows


def stage_calib():
    use20()
    by = new_records()
    real = by[REAL20]
    out = {}
    for K in (19, 20):
        e = R.grade(real, K)["est"]
        print(f"Real data, K = {K}: " + ", ".join(f"{x} {e[x]:.7f}" for x in EST) + f"; detectors {R.detectors(real, K)}")
        rows = calib_rows(by, e, K)
        out[str(K)] = rows
        print(f"\nK = {K}: | plants | n | " + " | ".join(f"{x}: error x1e4 -> zero" for x in EST) + " |")
        for fam, row in rows.items():
            cells = []
            for x in EST:
                c = row["cells"][x]
                none = f" ({row['n'] - c['n']} none)" if c["n"] < row["n"] else ""
                cells.append("-" if "lo" not in c else f"{c['lo'] * 1e4:+.2f} ... {c['hi'] * 1e4:+.2f} -> "
                             f"{c['zero'][0]:.6f}-{c['zero'][1]:.6f}" + none)
            print(f"| {fam} | {row['n']} | " + " | ".join(cells) + " |")
        print()
    a = out["19"]["all 28 arc plants"]["cells"]["arg"]["zero"]
    b = out["19"]["BP family (16)"]["cells"]["arg"]["zero"]
    s19 = json.load(open(K19.CHECKS19))["K19 combine"]
    sa, sb = s19["arc"], s19["bp"]
    check(all(abs(x - y) < 6e-7 for x, y in zip(a + b, sa + sb)),
          f"K = 19 calibration from the new records reproduces section 11.5 (arc {a[0]:.6f}-{a[1]:.6f}, BP family (16) "
          f"{b[0]:.6f}-{b[1]:.6f}; stored {sa[0]:.6f}-{sa[1]:.6f}, {sb[0]:.6f}-{sb[1]:.6f})")
    tr = {}
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B") or r["name"] in BP_CAL:
            tr[r["name"]] = {x: [R.grade(r, K)["err"][x] for K in KS20] for x in ("arg", "G", "pade")}
    out["per_plant_err_18_19_20"] = tr
    devs = load_checks().get("K20 plant sigma19-20 deviation", {})
    out["sigma_dev_summary"] = {}
    for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "C-EXP", "REFIT"):
        ns = [n for n in devs if n.startswith(f)]
        if ns:
            d19 = [max(devs[n][w][0] for w in devs[n]) for n in ns]
            d20 = [max(devs[n][w][1] for w in devs[n]) for n in ns]
            out["sigma_dev_summary"][f] = {"sigma19": [min(d19), max(d19)], "sigma20": [min(d20), max(d20)]}
            print(f"  {f}: plant sigma_19 vs measured {min(d19):.1e}-{max(d19):.1e}, sigma_20 {min(d20):.1e}-{max(d20):.1e}"
                  " (relative, worst node)")
    save_checks("K20 calib", out)


# ---------------------------------------------------------------- next generation and ring-free completions
def stage_recomplete():
    real = load_real20()
    by = new_records()
    e20 = R.grade(by[REAL20], 20)["est"]
    res = load_checks().get("K20 recomplete", {})
    wts = [f"{0.9990 + 0.0001 * i:.4f}" for i in range(21)]

    def finish(name, funcs, truth, extra, radii=(7.3, 12, 16)):
        f = funcs[1.0]
        counts = {f"zeros_lt_{r}": K13.count(f, mp.mpf(str(r))) for r in radii}
        rec = {"truth": truth["im_x"], "truth_im_sqrt": truth["im_sqrt"], **counts,
               "err_of_real_estimates": {e: e20[e] - truth["im_x"] for e in EST if e20[e] is not None}, **extra}
        res[name] = rec
        save_checks("K20 recomplete", res)
        print(f"{name}: zero at {truth['im_x']:.7f} (Im sqrt x: {truth['im_sqrt']:.7f}); {counts}; "
              f"{ {k: v for k, v in extra.items() if k not in ('x_R', 'poles_1.0', 'fit_resid_max', 'min_residue')} }",
              flush=True)

    for kind in ("C-W1", "C-W2", "C-W3"):
        name = kind.replace("C-", "C20-")
        if name in res:
            continue
        nodes, funcs, info, truth = PP.completion(kind, real=real, kc=20)
        finish(name, funcs, truth, {"kappa_beta": {str(w): info[w] for w in PP.NODES}}, radii=(7.3, 12))
    for kind in SHAPES:
        name = f"C20-ARC {kind}"
        if name in res:
            continue
        scan = PO.refit_scan(kind, wts, kc=20, real=real)
        bw = PO.best_wt(scan)
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{bw:.5f}", kc=20, real=real, funcs=funcs)
        finish(name, funcs, truth, {"best_wt": bw, "L2_floor": min(v[0] for v in scan.values()),
                                    "fit_resid_max": {w: info[w]["fit_resid_max"] for w in info}}, radii=(7.3, 12))
    for m in (3, 4, 5, 6, 7, 8, 9, 10):                          # up to [10/10], as [9/10] was at K = 19
        n = 20 - m
        name = f"C20-BP [{m}/{n}]"
        if name in res:
            continue
        try:
            nodes, funcs, info = K14.borel_pade(real, m, n)
            truth, xr = K14.truth_of(funcs)
        except Exception as ex:                                  # noqa: BLE001  (recorded, not swallowed)
            res[name] = {"error": f"{type(ex).__name__}: {ex}"}
            save_checks("K20 recomplete", res)
            print(f"{name}: {res[name]['error']}")
            continue
        maxp = {w: info[w]["max_pole"] for w in info}
        finish(name, funcs, truth, {"max_pole": maxp, "type_rule": all(v < 2 for v in maxp.values()),
                                    "reproduce_max_rel": max(info[w]["reproduce_max_rel"] for w in info),
                                    "min_residue": {w: info[w]["min_residue"] for w in info},
                                    "x_R": {str(w): [float(z.real), float(z.imag)] for w, z in xr.items()},
                                    "poles_1.0": info["1.0"]["poles"]})


# ---------------------------------------------------------------- order-20 projection
def stage_project():
    use20()
    by = new_records()
    real = by[REAL20]
    a = {int(k): v for k, v in real["est"]["20"]["arg"].items()}
    a20 = a[20]
    print("real arg crossings k = 10..20: " + ", ".join(f"{a[k]:.7f}" for k in range(10, 21)))
    out = {"arg": {k: a[k] for k in range(4, 21)}, "hyp": {}}
    print(f"real arg_20 - 1 = {a20 - 1:+.3e}")
    for w in HYPOTHESES:
        p = K14.project(a, 20, w)
        back = {k: a[k] - K14.project(a, k, w) for k in range(8, 20)}
        out["hyp"][str(w)] = {"pred20": p, "miss": a20 - p, "back": back}
        print(f"  zero at {w}: predicted arg_20 - 1 = {p - 1:+.3e}; real - predicted = {a20 - p:+.2e}; back-test (real - "
              f"predicted) at k = 17, 18, 19: " + ", ".join(f"{back[k]:+.1e}" for k in (17, 18, 19)))
    for lo, hi in ((4, 19), (4, 20)):
        ks = list(range(lo, hi + 1))
        sl, ic = np.polyfit(ks, [np.log10(a[k] - 1) for k in ks], 1)
        out[f"trend_{lo}_{hi}"] = {"per_order": 10 ** sl, "at20": 10 ** (ic + 20 * sl)}
        print(f"  log-linear trend of arg_k - 1 (zero at 1) over k = {lo}-{hi}: x{10 ** sl:.3f} per order; "
              f"at k = 20: {10 ** (ic + 20 * sl):.3e}")
    ait = {k: K14.aitken(a, k) for k in range(6, 21)}
    out["aitken"] = ait
    print("  Aitken on k-2..k, k = 6..20: " + ", ".join(f"{ait[k]:.6f}" for k in range(6, 21)))
    pl = []
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            b = {int(k): v for k, v in r["est"]["20"]["arg"].items()}
            if None in (b.get(18), b.get(19), b.get(20)):
                continue
            pl.append({"name": r["name"], "miss": b[20] - K14.project(b, 20, r["truth"]),
                       "aitken_err": K14.aitken(b, 20) - r["truth"]})
    out["plants"] = pl
    miss = [p["miss"] for p in pl]
    print(f"  arc plants ({len(pl)}): model through k = 18, 19 with the true w_t misses arg_20 by "
          f"{min(miss):+.1e} ... {max(miss):+.1e}")
    p1, pb = out["hyp"]["1.0"]["pred20"], [out["hyp"][str(w)]["pred20"] for w in HYPOTHESES[1:]]
    d1 = abs(a20 - p1)
    db = min(abs(a20 - p) for p in pb) if not (min(pb) <= a20 <= max(pb)) else 0.0
    fav = "zero at 1" if d1 < db else "zero in 0.999975-0.999996"
    back_real = {str(w): max(abs(out["hyp"][str(w)]["back"][k]) for k in (17, 18, 19)) for w in HYPOTHESES}
    model_err = max(max(back_real.values()), max(abs(x) for x in miss))
    other = db if fav == "zero at 1" else d1
    out.update({"favoured": fav, "dist_to_1": d1, "dist_to_lower": db, "model_err": model_err,
                "back_real_max": back_real, "plant_miss_max": max(abs(x) for x in miss), "decisive": other > 3 * model_err})
    print(f"  favoured: {fav} (|arg_20 - prediction|: zero at 1 {d1:.1e}, lower {db:.1e}); model error {model_err:.1e}; "
          f"decisive: {out['decisive']}")
    rl = load_real20()
    rng = np.random.default_rng(20)
    sh = []
    mp.mp.dps = 50
    for seed in range(20):
        rows = []
        for wf in PP.NODES:
            sig = list(rl[wf][0])
            for k in range(1, 21):
                nz = rng.standard_normal(2) / np.sqrt(2)
                sig[k] = sig[k] + mp.mpf(rl[wf][1][k]) * mp.mpc(nz[0], nz[1])
            rows.append((wf, taus(sig)))
        vals = [float(mp.arg(t[20]) / mp.pi) % 2 - 1 for _, t in rows]
        c = np.polyfit(PP.NODES, vals, 2)
        rts = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.96 < x.real < 1.04]
        sh.append(min(rts, key=lambda x: abs(x - 1)) - a20)
    out["noise_shift"] = [min(sh), max(sh)]
    print(f"  arg_20 under sigma_err noise (20 seeds): shift {min(sh):+.1e} ... {max(sh):+.1e}")
    save_checks("K20 project", out)


# ---------------------------------------------------------------- combination (section 12.0, item 7)
def stage_combine():
    use20()
    by = new_records()
    c = load_checks()
    cal = c["K20 calib"]["20"]
    arc = cal["all 28 arc plants"]["cells"]["arg"]["zero"]
    bp = cal["BP family (20)"]["cells"]["arg"]["zero"]
    rf = {n: v["truth"] for n, v in c["K20 recomplete"].items()
          if n.startswith("C20-BP") and v.get("type_rule") and v.get("zeros_lt_12", 99) <= 1}
    parts = [tuple(arc), tuple(bp)] + [(z, z) for z in rf.values()]
    lo, hi = min(p[0] for p in parts), max(p[1] for p in parts)
    print(f"calibrated arg, 28 arc plants: {arc[0]:.6f}-{arc[1]:.6f}; BP family (20): {bp[0]:.6f}-{bp[1]:.6f}; "
          f"ring-free completions passing the type rule: {rf}")
    print(f"K = 20 estimator range (hull): {lo:.6f}-{hi:.6f}")
    overlap = lo <= K19_RANGE[1] and K19_RANGE[0] <= hi
    head = (min(lo, K19_RANGE[0]), max(hi, K19_RANGE[1])) if overlap else None
    print(f"K = 19 headline {K19_RANGE[0]}-{K19_RANGE[1]}: overlap {overlap}; headline (union) "
          + (f"{head[0]:.6f}-{head[1]:.6f}" if head else "none: disjoint, diagnose first"))
    real = load_real20()
    rn = {w: real[w][0] for w in PP.NODES}
    gm = {}
    for K in (19, 20):
        mins = K14.kbar_minima(rn[1.0], K)
        ms = [{"kbar": [m[0], m[1]], "obj": m[2], "G": K14.G_at(rn, mp.mpc(m[0], m[1]), K)} for m in mins[:6]]
        g = [m["G"] for m in ms if m["obj"] <= 5 * ms[0]["obj"] and m["G"] is not None]
        prod = by[REAL20]["est"][str(K)]
        gm[str(K)] = {"minima": ms, "span": [min(g), max(g)], "global": ms[0]["G"], "production": prod["G"]}
        print(f"G at K = {K}: production {prod['G']:.6f}; over the minima within x5 of the best {min(g):.6f}-{max(g):.6f};"
              f" global minimum {ms[0]['G']:.6f}", flush=True)
    save_checks("K20 combine", {"arc": arc, "bp": bp, "ring_free": rf, "range20": [lo, hi], "overlap_k19": overlap,
                                "headline": head, "G_minima": gm})


def stage_table():
    c = load_checks()
    print(json.dumps({k: (v if "deviation" not in k else f"{len(v)} plants") for k, v in c.items()},
                     indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch20"))
    ap.add_argument("--mutate", default=None)
    ap.add_argument("--data", default=HERE)
    a = ap.parse_args()
    configure(a.data)
    {"converge": lambda: stage_converge(a.mutate), "plants": lambda: stage_plants(a.scratch),
     "real": lambda: stage_real(a.scratch), "ring": lambda: stage_ring(a.scratch), "calib": stage_calib,
     "recomplete": stage_recomplete, "project": stage_project, "combine": stage_combine, "table": stage_table,
     "plants-check": plants_check}[a.stage]()
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
