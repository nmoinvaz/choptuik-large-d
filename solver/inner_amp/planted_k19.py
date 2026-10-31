#!/usr/bin/env python3
"""Post-freeze follow-up of the planted-truth test at K = 19 (notes/PLANTED_SELECTION.md, section 11): sigma_19
(scan39J1000: K 39, J 1000, 960 bits; pilot (b) in scan39J1250: J 1250, 1280 bits; both copied from the raw-data
repository's T88-mac). The procedure and its pass criteria are section 11.0; each stage exits non-zero when one of its
checks fails. The earlier orders' scripts provide the shared pieces: planted_k14.py (Borel-Pade sums, zeros, the
projection model, the kbar minima of G), planted_k15.py, planted_k16.py, planted_k17.py and planted_k18.py (their
records and stored checks).

  converge    sigma_19's convergence (err19, pilot (b), sigma_0..18 unchanged, ratio law, spread in wf);
              --mutate=err|low|pilot|ratiolaw|amend breaks one input on purpose, and the matching check must fire
  plants      N1, N2, the 28 arc plants, C-EXP and the 13 REFIT plants at K = 17, 18, 19 (planted_results_k19.jsonl,
              resumable); their K = 17, 18 estimates must equal the K = 18 records; N1 recovered at K = 19, N2 nothing
  real        the real sigma_0..19 ("N0 scan39J1000"); its K = 17, 18 estimates must equal "N0 scan37J1000"'s; the
              sigma_19 sign flip (and a 1e-3 scale) at wf 1.00 as mutation controls
  ring        the K = 18 completions of section 10.6 (C18-*), regenerated: their sigma_19, tau_19 against the measured
              ones and the detectors at K = 19 beyond the real data's, read with and without D-bracket (section 11.0
              item 4; no ring survived sigma_18, so none is carried over); also the BP calibration family of section
              10.5 (C13-BP [5/8], [6/7], C14-BP [5/9], [6/8], C15-BP [6/9], [7/8], C16-BP [6/10], [7/9], [8/8],
              C17-BP [6/11], [7/10], [8/9])
  calib       K = 19 calibration on the arc plants, C-EXP, REFIT and the BP family (sixteen sums), and the zero each
              estimator implies; the K = 18 calibration from the new records must reproduce section 10.5
  recomplete  the recipes glued at sigma_19 (tail from k = 20) and the Borel-Pade sums [m/(19 - m)], m = 3..9
  project     the order-19 projection (est_k = w* + c r^k through k = 17, 18) against the real arg_19, back-tests, noise
  combine     section 11.0 item 7: the K = 19 estimator range, G over the minima of its objective, the headline
  table       the stored checks, as JSON

    nice -n 10 python3 planted_k19.py STAGE [--scratch DIR] [--mutate=...] [--data DIR]

--data DIR reads the scans from DIR/scan39J1000 and DIR/scan39J1250 and writes the records and checks there (the dry
run of section 11.0 item 0, planted_k19_dry.py); the default is this directory.
--amend DIR applies section 11.0's amendment of item 1 (2026-10-29): err19 at each node becomes the node's plateau error
(the file's sigma_err[19]) plus |sigma_19 - sigma_19 of DIR's J 1250 / 1280-bit file at the same wf| (T88's pilot (b)
at wf 1.00, T89's node runs at 0.99 and 1.01, copied into scan41J1250/), in every stage that reads sigma_19's error;
item 1(c) is then printed as written but not evaluated (its failure of 2026-10-28 is recorded in section 11.1), and
--mutate=amend scales DIR's sigma_19 at wf 0.99 by 1 + 1e-9, which must make (a') fire.
--nodes DIR applies section 11.0's second amendment of item 1 (2026-10-31, the fallback named in the first): the node data
are DIR's J 1250 / 1280-bit files (T89's K 41 runs, scan41J1250/), sigma_0..19, with every error the plateau error
sigma_err_fit.py recomputes from fit_info (a1 from the 960-bit scan39J1000 file at the same wf) and err19 that plus the
J 1500 pilot difference at wf 1.00 (scan41J1500/), (a''); the real record is "N0 scan41J1250". Its converge stage checks
(a''), (b'') bitwise equality with scan39J1250's pilot (b) at wf 1.00 (the differences from the 960-bit values at 0.99 and
1.01 are recorded, not checked) and (d); --mutate=err|low|pilot|ratiolaw, pilot scaling the J 1500 file's sigma_19 by
1 + 1e-9. --amend and --nodes exclude each other. Under --nodes the real stage reads item 3 as the third amendment says
(2026-10-31): the K = 17, 18 estimates of "N0 scan41J1250" agree with "N0 scan37J1000"'s within 3 rms of the plateau-error
noise reference (the 960-bit sigma_0..18 with Gaussian noise of their own sigma_err, 20 seeds, planted_noise_k19.jsonl,
the unperturbed data as the control); --mutate=real scales sigma_18 at wf 1.00 of the node data by 1 + 1e-9 and must fire.
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
import planted_plants as PP  # noqa: E402
import planted_post as PO  # noqa: E402
import planted_run as R  # noqa: E402
from arg_crossing import taus  # noqa: E402

SCAN19 = os.path.join(HERE, "scan39J1000")
PILOT19 = os.path.join(HERE, "scan39J1250")
SCAN18 = K18.SCAN18
REAL18 = K18.REAL18
REAL19 = "N0 scan39J1000"
OUT18 = K18.OUT18
OUT19 = os.path.join(HERE, "planted_results_k19.jsonl")
CHECKS19 = os.path.join(HERE, "planted_checks_k19.json")
KS19 = (17, 18, 19)
EST = K13.EST
ARC_FAMS = K14.ARC_FAMS
SHAPES = K14.SHAPES
HYPOTHESES = (1.0, 0.999992, 0.999975)      # zero at 1, at the lower end of the K = 18 range, of the headline
REPRO_TOL = 1e-9
BP_CAL_OLD = K18.BP_CAL                                                  # section 10.5's BP family (twelve sums)
BP_CAL = BP_CAL_OLD + ("C18-BP 6-12", "C18-BP 7-11", "C18-BP 8-10", "C18-BP 9-9")  # plus section 10.6's ring-free sums
RINGS17 = ()                                                            # no ring survived sigma_18 (section 10.4)
K18_RANGE = (0.999975, 1.000050)                                        # the K = 18 headline (section 10.8)
BRACKET = "D-bracket"                                                   # left out in the second reading (section 11.0 item 4)
AMEND = None                                                            # --amend DIR (section 11.0's amendment of item 1)
NODES = None                                                            # --nodes DIR (its second amendment, 2026-10-31)
J1500 = os.path.join(HERE, "scan41J1500")                               # the J 1500 pilot of (a'') at wf 1.00
NOISE19 = os.path.join(HERE, "planted_noise_k19.jsonl")                  # item 3's noise reference (third amendment)
NOISE_SEEDS = 20
FAILED = []


def configure(data):
    global SCAN19, PILOT19, OUT19, CHECKS19
    SCAN19 = os.path.join(data, "scan39J1000")
    PILOT19 = os.path.join(data, "scan39J1250")
    global J1500, NOISE19
    J1500 = os.path.join(data, "scan41J1500")
    NOISE19 = os.path.join(data, "planted_noise_k19.jsonl")
    OUT19 = os.path.join(data, "planted_results_k19.jsonl")
    CHECKS19 = os.path.join(data, "planted_checks_k19.json")


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILED.append(msg)
    return ok


def use19():
    R.OUT, R.KS = OUT19, KS19
    R.DONE.clear()
    R.DONE.update(R.done_names())


def load_checks():
    return json.load(open(CHECKS19)) if os.path.exists(CHECKS19) else {}


def save_checks(key, val):
    c = load_checks()
    c[key] = val
    with open(CHECKS19, "w") as f:
        json.dump(c, f, default=R.jdefault, indent=1)


def load_real(scandir, pattern):
    mp.mp.dps = PP.DPS
    return {float(w): (K14.sigmas(r), list(r["sigma_err"])) for w, r in K14.raw_scan(scandir, pattern).items()}


def amend_errs(p960, am=None):
    """section 11.0's amendment of item 1: sigma_err[19] of each 960-bit node file becomes its plateau error plus the
    node's pilot difference against the J 1250 / 1280-bit file at the same wf in AMEND (loaded here unless given)."""
    if am is None:
        am = K14.raw_scan(AMEND, "*.json")
    for w, r in p960.items():
        if w not in am:
            raise SystemExit(f"--amend {AMEND}: no file at wf {w}; the amendment needs a J 1250 / 1280-bit file there")
        if (am[w]["J"], am[w]["prec"]) != (1250, 1280):
            raise SystemExit(f"--amend {AMEND}: the file at wf {w} is J {am[w]['J']} / {am[w]['prec']} bits, not 1250 / 1280")
        d19 = float(abs(K14.sigmas(r)[19] - K14.sigmas(am[w])[19]))
        r["err19_plateau"], r["pilot_d19"], r["amend_K"] = r["sigma_err"][19], d19, am[w]["K"]
        r["sigma_err"][19] = r["err19_plateau"] + d19
    return p960


def node_files():
    """the J 1250 / 1280-bit node files of NODES by wf, one at every wf of the 960-bit scan (section 11.0, second amendment)"""
    am = K14.raw_scan(NODES, "*.json")
    for w in K14.raw_scan(SCAN19, "*_p960.json"):
        if w not in am:
            raise SystemExit(f"--nodes {NODES}: no file at wf {w}")
        if (am[w]["J"], am[w]["prec"]) != (1250, 1280):
            raise SystemExit(f"--nodes {NODES}: the file at wf {w} is J {am[w]['J']} / {am[w]['prec']} bits, not 1250 / 1280")
    return am


def node_errs(am, p960, j1500=None):
    """({wf: [err_0..err_19]}, the J 1500 pilot difference): the plateau errors sigma_err_fit.py recomputes from fit_info
    (a1 from the 960-bit file at the same wf), err19 plus |sigma_19(J 1250) - sigma_19(J 1500)| at wf 1.00, (a'')"""
    import sigma_err_fit as SF  # noqa: E402
    if j1500 is None:
        j1500 = K14.raw_scan(J1500, "*.json")
    if 1.0 not in j1500 or (j1500[1.0]["J"], j1500[1.0]["prec"]) != (1500, 1280):
        raise SystemExit(f"{J1500}: no J 1500 / 1280-bit file at wf 1.0 for the pilot difference of (a'')")
    pd = float(abs(K14.sigmas(am[1.0])[19] - K14.sigmas(j1500[1.0])[19]))
    errs = {}
    for w, n in am.items():
        if w not in p960:
            continue
        rec = SF.recompute(n, SF.a1_of(p960[w]))
        errs[w] = [float(rec[k][2]) for k in range(20)]
        errs[w][19] += pd
    return errs, pd


def load_nodes():
    """{wf: (sigma_0..19, sigma_err)} from the J 1250 / 1280-bit node files, errors as node_errs gives them"""
    mp.mp.dps = PP.DPS
    p960 = K14.raw_scan(SCAN19, "*_p960.json")
    am = node_files()
    errs, _ = node_errs(am, p960)
    return {float(w): (K14.sigmas(am[w])[:20], errs[w]) for w in p960}


def load_real19():
    if NODES:
        return load_nodes()
    if not AMEND:
        return load_real(SCAN19, "*_p960.json")
    mp.mp.dps = PP.DPS
    raw = amend_errs(K14.raw_scan(SCAN19, "*_p960.json"))
    return {float(w): (K14.sigmas(r), list(r["sigma_err"])) for w, r in raw.items()}


def load_real18():
    return load_real(SCAN18, "*_p960.json")


def k18_records():
    return {r["name"]: r for r in (json.loads(line) for line in open(OUT18) if line.strip())}


def new_records():
    use19()
    return {r["name"]: r for r in R.records()}


# ---------------------------------------------------------------- converge
def stage_converge_nodes(mutate=None):
    """item 1 under section 11.0's second amendment (2026-10-31): the node data are NODES' J 1250 / 1280-bit files.
    (a'') err19 = recomputed plateau error + the J 1500 pilot difference at wf 1.00; (b'') sigma_0..19 at wf 1.00
    bitwise scan39J1250's pilot (b), the differences from the 960-bit values at 0.99 and 1.01 recorded; (c) not
    evaluated; (d) the ratio law. Mutations err|low|pilot|ratiolaw, each making its check fire."""
    mp.mp.dps = 50
    p960 = K14.raw_scan(SCAN19, "*_p960.json")
    am = node_files()
    pil = K14.raw_scan(PILOT19, "*.json")
    j15 = K14.raw_scan(J1500, "*.json")
    print(f"node data: J 1250 / 1280-bit files at wf {sorted(am)} from {NODES}: "
          + ", ".join(f"K {r['K']} at wf {w}" for w, r in sorted(am.items())))
    print(f"  960-bit scan39J1000 at wf {sorted(p960)} (a1 and the sigma_0..18 comparison); scan39J1250's pilot (b) at wf "
          f"{sorted(pil)}; J 1500 at wf {sorted(j15)}")
    if mutate == "low":
        x = mp.mpf(am[1.0]["sigma_str"][18][0]) * (1 + mp.mpf("1e-15"))
        am[1.0]["sigma_str"][18][0] = mp.nstr(x, 20)
    elif mutate == "pilot":
        x = mp.mpf(j15[1.0]["sigma_str"][19][1]) * (1 + mp.mpf("1e-9"))
        j15[1.0]["sigma_str"][19][1] = mp.nstr(x, 20)
    elif mutate == "ratiolaw":
        am[1.01]["sigma_str"][19] = [mp.nstr(mp.mpf(x) * mp.mpf("1.1"), 20) for x in am[1.01]["sigma_str"][19]]
    elif mutate not in (None, "err"):
        raise SystemExit(f"unknown mutation {mutate} (under --nodes: err|low|pilot|ratiolaw)")
    errs, pd = node_errs(am, p960, j15)
    if mutate == "err":
        errs[1.0][19] *= 1e14
    out = {"nodes": {}, "nodes_dir": NODES, "pilot_j1500_d19": pd}
    print(f"(a'') err19 = the recomputed plateau error + the J 1500 pilot difference at wf 1.00, {pd:.3e}, against the 1e-10 "
          "relative requirement:")
    for w in sorted(p960):
        s = K14.sigmas(am[w])
        e19 = errs[w][19]
        rel = e19 / float(abs(s[19]))
        out["nodes"][str(w)] = {"sigma19": am[w]["sigma_str"][19], "err19": e19, "err19_rel": rel, "err19_plateau": e19 - pd,
                                "err18": errs[w][18], "err17": errs[w][17]}
        check(rel <= 1e-10, f"wf {w}: sigma_19 = {mp.nstr(s[19], 20)}, err19 {e19:.3g} = plateau {e19 - pd:.3g} + pilot "
                            f"difference {pd:.3g} ({rel:.2e} relative, margin {1e-10 / rel:.2f})")
    print("(b'') at wf 1.00, sigma_0..19 against scan39J1250's pilot (b) (K 39, J 1250, 1280 bits), bitwise:")
    same = [am[1.0]["sigma_str"][k] == pil[1.0]["sigma_str"][k] for k in range(20)]
    out["nodes"]["1.0"]["low_bitwise"] = all(same)
    check(all(same), f"wf 1.0: sigma_0..19 bitwise equal ({sum(same)}/20)")
    print("  at wf 0.99 and 1.01 (no second file at J 1250 / 1280 bits): sigma_0..18 against the 960-bit scan39J1000 values, "
          "|difference| / that file's plateau error, with the S_1 family's Mf picks (960, 1280); recorded, not a check:")
    for w in sorted(p960):
        s, s0 = K14.sigmas(am[w]), K14.sigmas(p960[w])
        rows = []
        for k in range(19):
            e0, kk = p960[w]["sigma_err"][k], 2 * k + 1
            d = float(abs(s[k] - s0[k]))
            r = d / e0 if e0 else None
            rows.append({"k": k, "diff": d, "err960": e0, "ratio": r,
                         "mf": [p960[w]["fit_info"].get(f"{kk}_1", [None])[0], am[w]["fit_info"].get(f"{kk}_1", [None])[0]]})
        out["nodes"][str(w)]["low_vs_960"] = rows
        over = [x for x in rows if x["ratio"] is not None and x["ratio"] > 1]
        print(f"  wf {w}: " + " ".join(f"k{x['k']}:{x['ratio']:.2f}" for x in rows if x["ratio"] is not None) + "; over 1: "
              + (", ".join(f"k {x['k']} ({x['ratio']:.2f}, Mf {x['mf'][0]} and {x['mf'][1]})" for x in over) or "none"))
    print("(c) pilot (b): not evaluated (failed 2026-10-28, section 11.1); under (b'') it is the node data's bitwise copy at "
          "sigma_0..19, wf 1.00")
    print("(d) the ratio law through sigma_19: second difference of sigma_k/sigma_{k+1}, k = 7..17 (D-ratiolaw <= 0.05):")
    out["ratiolaw_d2"] = {}
    for w in sorted(p960):
        s = K14.sigmas(am[w])
        rat = [complex(s[k] / s[k + 1]) for k in range(19)]
        d2 = [abs(rat[k + 1] - 2 * rat[k] + rat[k - 1]) for k in range(7, 18)]
        out["ratiolaw_d2"][str(w)] = d2
        check(d2[-1] <= 0.05, f"wf {w}: |d2| k = 7..17: " + " ".join(f"{x:.4f}" for x in d2)
              + f"; sigma_18/sigma_19 = {rat[18]:.4f}")
    print("spread in wf (context): the change of sigma_k across the nodes, against its error")
    out["spread"] = {}
    for k in (17, 18, 19):
        s = {w: K14.sigmas(am[w])[k] for w in p960}
        d1 = abs(s[1.01] - s[0.99]) / 2
        d2 = abs(s[1.01] - 2 * s[1.0] + s[0.99])
        e = max(errs[w][k] for w in p960)
        out["spread"][str(k)] = {"half_range_rel": float(d1 / abs(s[1.0])), "d2_over_d1": float(d2 / d1),
                                 "err_over_d1": float(e / d1)}
        print(f"  k = {k}: |sigma_k(1.01) - sigma_k(0.99)|/2 = {float(d1 / abs(s[1.0])):.4f} |sigma_k(1.00)|; "
              f"second difference / first = {float(d2 / d1):.4f}; max err_k / first difference = {float(e / d1):.1e}")
    if not mutate:
        c = load_checks()
        if c.get("K19 converge", {}).get("amend") and "K19 converge under (a')" not in c:
            save_checks("K19 converge under (a')", c["K19 converge"])  # the run of 2026-10-31 04:23 UTC, (a') failed at 0.99
        save_checks("K19 converge", out)
    return out


def stage_converge(mutate=None):
    if NODES:
        return stage_converge_nodes(mutate)
    mp.mp.dps = 50
    p960 = K14.raw_scan(SCAN19, "*_p960.json")
    pil = K14.raw_scan(PILOT19, "*.json")
    old = K14.raw_scan(SCAN18, "*_p960.json")
    print(f"files: 960 bits at wf {sorted(p960)}; pilot (b) at wf {sorted(pil)}; scan37J1000 at wf {sorted(old)}")
    for w, r in sorted(p960.items()):
        print(f"  wf {w}: K {r['K']} J {r['J']} prec {r['prec']} fit {r.get('fit')} mf {r.get('mf')}, "
              f"{len(r['sigma_str'])} coefficients")
    if mutate == "err":
        p960[1.0]["sigma_err"][19] *= 1e14
    elif mutate == "low":
        a = mp.mpf(p960[0.99]["sigma_str"][18][0]) * (1 + mp.mpf("1e-15"))
        p960[0.99]["sigma_str"][18][0] = mp.nstr(a, 20)
    elif mutate == "pilot":
        b = mp.mpf(pil[1.0]["sigma_str"][19][1]) * (1 + mp.mpf("1e-10"))
        pil[1.0]["sigma_str"][19][1] = mp.nstr(b, 20)
    elif mutate == "ratiolaw":
        p960[1.01]["sigma_str"][19] = [mp.nstr(mp.mpf(x) * mp.mpf("1.1"), 20) for x in p960[1.01]["sigma_str"][19]]
    elif mutate == "amend":
        if not AMEND:
            raise SystemExit("--mutate=amend needs --amend DIR")
    elif mutate:
        raise SystemExit(f"unknown mutation {mutate}")
    out = {"nodes": {}}
    if AMEND:
        am = K14.raw_scan(AMEND, "*.json")
        print(f"amendment of item 1 (section 11.0, 2026-10-29): J 1250 / 1280-bit files at wf {sorted(am)} from {AMEND}, "
              + ", ".join(f"K {r['K']} at wf {w}" for w, r in sorted(am.items())))
        if mutate == "amend":
            x = mp.mpf(am[0.99]["sigma_str"][19][0]) * (1 + mp.mpf("1e-9"))
            am[0.99]["sigma_str"][19][0] = mp.nstr(x, 20)
        amend_errs(p960, am)
        print("(a') err19 = plateau error + pilot difference at the node, against the 1e-10 relative requirement:")
    else:
        print("(a) err19 against the 1e-10 relative requirement (section 3.6):")
    for w, r in sorted(p960.items()):
        s = K14.sigmas(r)
        rel = r["sigma_err"][19] / float(abs(s[19]))
        out["nodes"][str(w)] = {"sigma19": r["sigma_str"][19], "err19": r["sigma_err"][19], "err19_rel": rel,
                                "err18": r["sigma_err"][18]}
        parts = ""
        if AMEND:
            out["nodes"][str(w)].update({"err19_plateau": r["err19_plateau"], "pilot_d19": r["pilot_d19"]})
            parts = f" = plateau {r['err19_plateau']:.3g} + pilot difference {r['pilot_d19']:.3g}"
        check(rel <= 1e-10, f"wf {w}: sigma_19 = {mp.nstr(s[19], 20)}, err19 {r['sigma_err'][19]:.3g}{parts} "
              f"({rel:.2e} relative, margin {1e-10 / rel:.2f})")
    print("(b) sigma_0..18 against scan37J1000 (960 bits):")
    for w, r in sorted(p960.items()):
        same = [r["sigma_str"][k] == old[w]["sigma_str"][k] for k in range(19)]
        erre = [r["sigma_err"][k] == old[w]["sigma_err"][k] for k in range(19)]
        out["nodes"][str(w)]["low_bitwise"] = all(same)
        check(all(same), f"wf {w}: sigma_0..18 bitwise equal ({sum(same)}/19); sigma_err_0..18 equal ({sum(erre)}/19)")
    print("(c) pilot (b) at wf 1.00 (J 1250, 1280 bits) against (a):")
    a, b = p960[1.0], pil[1.0]
    sa, sb, so = K14.sigmas(a), K14.sigmas(b), K14.sigmas(old[1.0])
    d19 = float(abs(sa[19] - sb[19]))
    low = {k: (float(abs(sb[k] - so[k])), old[1.0]["sigma_err"][k]) for k in (16, 17, 18)}
    err_a = a["err19_plateau"] if AMEND else a["sigma_err"][19]
    out["pilot"] = {"d19": d19, "err19": err_a, "low": low, "pilot_sigma_err": b["sigma_err"][:3]}
    if AMEND:
        print(f"  as written, not evaluated (failed 2026-10-28, section 11.1): |a - b| = {d19:.2e} against the plateau err19 "
              f"{err_a:.2e}, ratio {d19 / err_a:.2f}; the difference is part of err19 under (a'). sigma_16..18 of (b) "
              "against scan37J1000: " + ", ".join(f"k {k}: {d:.1e} (err {e:.1e})" for k, (d, e) in low.items()))
    else:
        check(d19 <= err_a, f"sigma_19: |a - b| = {d19:.2e} against err19 {err_a:.2e}")
        check(all(d <= e for d, e in low.values()), "sigma_16..18 of (b) against scan37J1000: " + ", ".join(
            f"k {k}: {d:.1e} (err {e:.1e})" for k, (d, e) in low.items()))
    print(f"  note: (b)'s own sigma_err is {b['sigma_err'][19]} at every k (its order-1 fit has residual 0), so (b) checks "
          f"(a)'s values, not their error")
    print("(d) the ratio law through sigma_19: second difference of sigma_k/sigma_{k+1}, k = 7..17 (D-ratiolaw <= 0.05):")
    out["ratiolaw_d2"] = {}
    for w, r in sorted(p960.items()):
        s = K14.sigmas(r)
        rat = [complex(s[k] / s[k + 1]) for k in range(19)]
        d2 = [abs(rat[k + 1] - 2 * rat[k] + rat[k - 1]) for k in range(7, 18)]
        out["ratiolaw_d2"][str(w)] = d2
        check(d2[-1] <= 0.05, f"wf {w}: |d2| k = 7..17: " + " ".join(f"{x:.4f}" for x in d2)
              + f"; sigma_18/sigma_19 = {rat[18]:.4f}")
    print("spread in wf (context): the change of sigma_k across the nodes, against its error")
    out["spread"] = {}
    for k in (17, 18, 19):
        s = {w: K14.sigmas(r)[k] for w, r in p960.items()}
        d1 = abs(s[1.01] - s[0.99]) / 2
        d2 = abs(s[1.01] - 2 * s[1.0] + s[0.99])
        e = max(r["sigma_err"][k] for r in p960.values())
        out["spread"][str(k)] = {"half_range_rel": float(d1 / abs(s[1.0])), "d2_over_d1": float(d2 / d1),
                                 "err_over_d1": float(e / d1)}
        print(f"  k = {k}: |sigma_k(1.01) - sigma_k(0.99)|/2 = {float(d1 / abs(s[1.0])):.4f} |sigma_k(1.00)|; "
              f"second difference / first = {float(d2 / d1):.4f}; max err_k / first difference = {float(e / d1):.1e}")
    if not mutate:
        c = load_checks()
        if AMEND and "K19 converge" in c and "K19 converge as written" not in c:
            save_checks("K19 converge as written", c["K19 converge"])     # the run of 2026-10-28 23:16 UTC, item 1(c) failed
        out["amend"] = AMEND
        save_checks("K19 converge", out)
    return out


# ---------------------------------------------------------------- plants
def sigma_dev(nodes, real19):
    return {str(w): [float(abs(nodes[w][k] - real19[w][0][k]) / abs(real19[w][0][k])) for k in (18, 19)]
            for w in PP.NODES}


def stage_plants(scratch):
    use19()
    real19 = load_real19()
    devs = load_checks().get("K19 plant sigma18-19 deviation", {})
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
            devs[name] = sigma_dev(nodes, real19)
            save_checks("K19 plant sigma18-19 deviation", devs)
            R.run_plant(name, tag.split()[0], nodes, float(wt), scratch,
                        info={"fit_resid_max": {str(w): fam.resid[w] and max(fam.resid[w]) for w in PP.NODES}})
    if "C-EXP" not in R.DONE:
        nodes, _, info, truth = PP.completion("C-EXP")
        devs["C-EXP"] = sigma_dev(nodes, real19)
        save_checks("K19 plant sigma18-19 deviation", devs)
        R.run_plant("C-EXP", "C", nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    for wt in R.REFIT_WT:
        name = f"REFIT P-ARC wt={wt}"
        if name in R.DONE:
            continue
        nodes, resid = K14.refit_nodes(wt)
        devs[name] = sigma_dev(nodes, real19)
        save_checks("K19 plant sigma18-19 deviation", devs)
        R.run_plant(name, "REFIT", nodes, float(wt), scratch, info={"fit_resid_max": resid}, linear=False)
    plants_check()


def plants_check():
    old, new = k18_records(), new_records()
    names = [n for n in new if not n.startswith(("N0", "C13", "C14", "C15", "C16", "C17", "C18", "C19"))]
    print(f"\n{len(names)} plants with K = 19 records")
    worst = (0.0, None)
    for n in names:
        for K in (17, 18):
            d, k = K14.max_diff(K14.est_vector(new[n], K), K14.est_vector(old[n], K))
            if d > worst[0]:
                worst = (d, f"{n} K{K} {k}")
    check(worst[0] <= REPRO_TOL, f"regenerated plants reproduce the stored K = 17, 18 estimates: largest difference "
                                 f"{worst[0]:.1e} ({worst[1]})")
    n1 = [new[n] for n in names if n.startswith("N1")]
    errs = [abs(R.grade(r, 19)["est"][e] - r["truth"]) for r in n1 for e in ("arg", "ratio", "pade", "G")
            if R.grade(r, 19)["est"][e] is not None]
    nnone = sum(R.grade(r, 19)["est"][e] is None for r in n1 for e in ("arg", "ratio", "pade", "G"))
    check(len(n1) == 7 and nnone == 0 and max(errs) <= 1e-6,
          f"N1 at K = 19: {len(n1)} lone zeros, worst error {max(errs):.1e} over arg, ratio, Pade, G ({nnone} missing)")
    ctl = [abs(R.grade(r, 19)["est"]["arg"] - (r["truth"] + 0.002)) <= 1e-6 for r in n1]
    check(not any(ctl), "control: N1 graded against w_t + 0.002 fails (as it must)")
    n2 = [new[n] for n in names if n.startswith("N2")]
    printed = [(r["name"], e) for r in n2 for e, v in R.grade(r, 19)["est"].items() if v is not None]
    check(len(n2) == 2 and not printed, f"N2 at K = 19: no estimator prints a number ({printed or 'none'})")
    fams = {}
    for n in names:
        fams.setdefault(new[n]["family"], []).append(n)
    print("  records per family:", {f: len(v) for f, v in fams.items()})
    save_checks("K19 plants check", {"repro_worst": worst, "n1_worst": max(errs), "n2_printed": printed,
                                     "families": {f: len(v) for f, v in fams.items()}})


# ---------------------------------------------------------------- real data
def noise_reference(scratch):
    """section 11.0's third amendment of item 3: the 960-bit node data (sigma_0..18) with Gaussian noise of each sigma_k's
    own plateau error, k = 1..18, NOISE_SEEDS seeds, through the harness at K = 17, 18, the unperturbed data first as the
    control; records in NOISE19 (resumable). Returns them by name."""
    real = load_real(SCAN19, "*_p960.json")
    base = {wf: list(real[wf][0][:19]) for wf in PP.NODES}
    out, ks, done = R.OUT, R.KS, set(R.DONE)
    R.OUT, R.KS = NOISE19, (17, 18)
    R.DONE.clear()
    R.DONE.update(R.done_names())
    try:
        R.run_plant("N0 scan39J1000 sigma_0..18", "N0 control", base, None, scratch)
        for seed in range(NOISE_SEEDS):
            rng = np.random.default_rng(1000 + seed)
            nodes = {}
            for wf in PP.NODES:
                sig = list(base[wf])
                for k in range(1, 19):
                    n = rng.standard_normal(2) / np.sqrt(2)
                    sig[k] = sig[k] + mp.mpf(real[wf][1][k]) * mp.mpc(n[0], n[1])
                nodes[wf] = sig
            R.run_plant(f"N0 noise seed={seed}", "N0 noise", nodes, None, scratch)
        recs = {r["name"]: r for r in R.records()}
    finally:
        R.OUT, R.KS = out, ks
        R.DONE.clear()
        R.DONE.update(done)
    return recs


def stage_real(scratch, mutate=None):
    use19()
    real = load_real19()
    if mutate == "real":                      # the third amendment's mutation: records go to a scratch file
        if not NODES:
            raise SystemExit("--mutate=real needs --nodes")
        real[1.0][0][18] = real[1.0][0][18] * (1 + mp.mpf("1e-9"))
        R.OUT = os.path.join(scratch, "mutate_real_records.jsonl")
        if os.path.exists(R.OUT):
            os.remove(R.OUT)
        R.DONE.clear()
    elif mutate:
        raise SystemExit(f"unknown mutation {mutate} (real stage: real)")
    R.run_plant(REAL19, "N0", {wf: real[wf][0] for wf in PP.NODES}, None, scratch)
    flip = {wf: list(real[wf][0]) for wf in PP.NODES}
    flip[1.0][19] = -flip[1.0][19]
    R.run_plant(f"{REAL19} X1 flip sigma_19", "N0 mutation", flip, None, scratch)
    sc = {wf: list(real[wf][0]) for wf in PP.NODES}
    sc[1.0][19] = sc[1.0][19] * (1 + mp.mpf("1e-3"))
    R.run_plant(f"{REAL19} X2 scale sigma_19", "N0 mutation", sc, None, scratch)
    old, new = k18_records(), {r["name"]: r for r in R.records()}
    rec = new[REAL19]
    noise = {}
    if NODES:                                 # the third amendment: agreement within 3 rms of the plateau-error noise
        ref = noise_reference(scratch)
        ctl = ref["N0 scan39J1000 sigma_0..18"]
        for K in (17, 18):
            d, k = K14.max_diff(K14.est_vector(ctl, K), K14.est_vector(old[REAL18], K))
            check(d <= REPRO_TOL, f"K = {K}: control, the 960-bit sigma_0..18 through the harness reproduce N0 scan37J1000 "
                                  f"(largest difference {d:.1e})")
            vn, vo, vc = K14.est_vector(rec, K), K14.est_vector(old[REAL18], K), K14.est_vector(ctl, K)
            rows, worst = {}, (0.0, None)
            for key in vn:
                vals = [K14.est_vector(ref[f"N0 noise seed={s}"], K).get(key) for s in range(NOISE_SEEDS)]
                if None in (vn[key], vo.get(key), vc.get(key)) or any(v is None for v in vals):
                    continue
                rms = float(np.sqrt(np.mean([(v - vc[key]) ** 2 for v in vals])))
                obs = abs(vn[key] - vo[key])
                ratio = obs / rms if rms else (0.0 if obs == 0 else float("inf"))
                rows[key] = {"observed": obs, "noise_rms": rms, "ratio": ratio}
                if ratio > worst[0]:
                    worst = (ratio, key)
            big = max(rows.items(), key=lambda kv: kv[1]["observed"])
            noise[str(K)] = {"rows": rows, "worst_ratio": worst, "largest_difference": [big[0], big[1]["observed"]]}
            check(worst[0] <= 3, f"K = {K}: the {REAL19[3:]} estimates agree with N0 scan37J1000's within 3 rms of the "
                                 f"plateau-error noise ({NOISE_SEEDS} seeds): largest ratio {worst[0]:.2f} ({worst[1]}); "
                                 f"largest difference {big[1]['observed']:.1e} ({big[0]}, noise rms {big[1]['noise_rms']:.1e})")
    else:
        for K in (17, 18):
            d, k = K14.max_diff(K14.est_vector(rec, K), K14.est_vector(old[REAL18], K))
            check(d <= REPRO_TOL, f"K = {K}: the {REAL19[3:]} estimates equal those of scan37J1000 (largest difference "
                                  f"{d:.1e}{', ' + k if k else ''})")
    for K in KS19:
        g = R.grade(rec, K)["est"]
        print(f"  K = {K}: " + ", ".join(f"{e} {v:.7f}" if v is not None else f"{e} None" for e, v in g.items())
              + f"; detectors {R.detectors(rec, K)}")
    e = rec["est"]["19"]
    print("  arg crossings k = 4..19:", {int(k): round(v, 7) for k, v in e["arg"].items()})
    print("  ratio estimates k = 9..19:", {int(k): round(v, 6) for k, v in e["ratio"].items()})
    print("  Pade central", e["pade_central"], "values", [round(v, 6) if v else v for v in e["pade_central_vals"]],
          "range", e["pade_central_range"], "mid", e["pade_central_mid"])
    print("  all Pade crossings K = 19:", {k: round(v, 6) if v else v for k, v in e["pade"].items()})
    print("  G K = 18, 19 via G_prev/G:", e.get("G_prev"), e.get("G"), "kbar", e.get("G_kbar"))
    det = set(R.detectors(rec, 19))
    for n in (f"{REAL19} X1 flip sigma_19", f"{REAL19} X2 scale sigma_19"):
        m = new[n]
        extra = sorted(set(R.detectors(m, 19)) - det)
        g, g0 = R.grade(m, 19)["est"], R.grade(rec, 19)["est"]
        shift = max((abs(g[x] - g0[x]) if None not in (g[x], g0[x]) else float("inf")) for x in EST)
        check(bool(extra) if "flip" in n else True, f"mutation {n.split()[-2]} sigma_19: new detectors {extra or 'none'}, "
                                                     f"largest shift {shift:.1e}")
    # the effective exponent at w_inf from k - 1, k (the README's column)
    mp.mp.dps = 50
    rt = taus(list(real[1.0][0][:20]))
    ce = {}
    for k in (17, 18, 19):
        x0 = (k - 1) * rt[k - 1] / (k * rt[k])
        ce[k] = complex(-k * rt[k] * x0 ** k)
    print("  effective exponent c at wf 1.00 (two orders): " + ", ".join(f"k {k}: {v.real:.3f}{v.imag:+.3f}i"
                                                                          for k, v in ce.items()))
    if not mutate:
        save_checks("K19 real", {"est": {str(K): R.grade(rec, K)["est"] for K in KS19},
                                 "detectors": {str(K): R.detectors(rec, K) for K in KS19},
                                 "c_eff_wf1": {str(k): [v.real, v.imag] for k, v in ce.items()}, "noise_comparison": noise})


# ---------------------------------------------------------------- ring test
def c18_completions():
    """section 10.6's K = 18 completions (tail from k = 19), regenerated (no ring survived sigma_18, so nothing is
    carried over from section 10.4); and the BP calibration sums of K = 13, 14, 15, 16 and 17"""
    real18 = load_real18()
    stored18 = json.load(open(K18.CHECKS18))["K18 recomplete"]
    stored17 = json.load(open(K17.CHECKS17))["K17 recomplete"]
    stored16 = json.load(open(K16.CHECKS16))["K16 recomplete"]
    stored15 = json.load(open(K15.CHECKS15))["K15 recomplete"]
    stored14 = json.load(open(K14.CHECKS14))["K14 recomplete"]
    out = {}
    for kind in ("C-W1", "C-W2", "C-W3"):
        nodes, funcs, info, truth = PP.completion(kind, real=real18, kc=18)
        out[kind.replace("C-", "C18-")] = (nodes, funcs, truth, stored18[kind.replace("C-", "C18-")])
    for kind in SHAPES:
        name = f"C18-ARC {kind}"
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{stored18[name]['best_wt']:.5f}", kc=18, real=real18, funcs=funcs)
        out[name] = (nodes, funcs, truth, stored18[name])
    for m in (3, 4, 5, 6, 7, 8, 9):
        name = f"C18-BP [{m}/{18 - m}]"
        nodes, funcs, info = K14.borel_pade(real18, m, 18 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored18[name])
    real17 = K18.load_real17()
    for m in (6, 7, 8):                          # the calibration family of section 10.5
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
    if name.startswith("C18"):
        return "C18 (ring test)"
    if name in RINGS17:
        return "C17-BP (ring test)"
    return name[:6] + " (calibration)"


def stage_ring(scratch):
    use19()
    real = load_real19()
    mp.mp.dps = 50
    rt = {wf: taus(list(real[wf][0][:20])) for wf in PP.NODES}
    comps = c18_completions()
    for name, (nodes, _, truth, _) in comps.items():
        R.run_plant(rec_name(name), family_of(name), nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    new = new_records()
    real_rec = new[REAL19]
    real_det = set(R.detectors(real_rec, 19))
    a19 = R.grade(real_rec, 19)["est"]["arg"]
    print("real: sigma_19 / tau_19 at the nodes: " + "; ".join(
        f"{wf}: {mp.nstr(real[wf][0][19], 8)} / {mp.nstr(rt[wf][19], 8)}" for wf in PP.NODES))
    print(f"real arg_19 = {a19:.7f}; real K = 19 detectors: {sorted(real_det)}\n")
    out = {"simple-zero law from tau_17, tau_18": {"tau19_rel": K14.simple_zero_rel(rt, 19)},
           "simple-zero law from tau_16, tau_17 (tau_18, for comparison)": {"tau18_rel": K14.simple_zero_rel(rt, 18)}}
    print("| completion | its zero | zeros in |x| < 7.3 / 12 / 16 | sigma_19 rel. error (0.99 / 1.00 / 1.01) | "
          "tau_19 rel. error | its arg_19 | K = 19 detectors beyond the real data's | survives | without D-bracket |")
    print("|---|---|---|---|---|---|---|---|---|")
    sz = " / ".join(f"{x:.1e}" for x in K14.simple_zero_rel(rt, 19))
    print(f"| simple-zero law from tau_17, tau_18 | - | - | - | {sz} | - | - | - | - |")
    for name, (nodes, _, truth, st) in comps.items():
        mp.mp.dps = 50
        se = [float(abs(nodes[wf][19] - real[wf][0][19]) / abs(real[wf][0][19])) for wf in PP.NODES]
        te = [float(abs(taus(list(nodes[wf][:20]))[19] - rt[wf][19]) / abs(rt[wf][19])) for wf in PP.NODES]
        rec = new[rec_name(name)]
        arg = R.grade(rec, 19)["est"]["arg"]
        extra = sorted(set(R.detectors(rec, 19)) - real_det)
        extra_nb = [d for d in extra if d != BRACKET]                   # the second reading of section 11.0 item 4
        kq = int(name[1:3])                       # the order whose real coefficients the completion carries
        if kq in KS19:
            # the K = 17, 18 completions carry the 960-bit sigma_0..18; under --nodes the real record is the 1280-bit
            # data, so their reproduction reference is the stored 960-bit record (fourth amendment, 2026-10-31)
            if NODES and kq < 19:
                ref_rec, ref_name = k18_records()[REAL18], f"{REAL18}'s (960 bits, the data it carries)"
            else:
                ref_rec, ref_name = real_rec, "the real ones"
            same = K14.max_diff(K14.est_vector(rec, kq), K14.est_vector(ref_rec, kq))[0]
            label = f"its K = {kq} estimates are {ref_name}"
        else:                                     # glued below K = 17: its K = 17, 18 estimates are the stored record's
            old = k18_records()[rec_name(name)]
            same = max(K14.max_diff(K14.est_vector(rec, K), K14.est_vector(old, K))[0] for K in (17, 18))
            label = "its K = 17, 18 estimates reproduce the stored K = 18 record"
        zc = " / ".join(str(st.get(f"zeros_lt_{r}", "-")) for r in (7.3, 12, 16))
        ring = (st.get("zeros_lt_12") or 0) > 1
        out[name] = {"sigma19_rel": se, "tau19_rel": te, "arg19": arg, "extra_detectors": extra, "survives": not extra,
                     "survives_without_bracket": not extra_nb, "truth": truth["im_x"], "zeros": zc, "ring": ring,
                     (f"K{kq}_equals_real" if kq in KS19 else "K17_18_equal_stored"): same,
                     "est19": R.grade(rec, 19)["est"]}
        print(f"| {name} | {truth['im_x']:.6f} | {zc} | {' / '.join(f'{x:.1e}' for x in se)} | "
              f"{' / '.join(f'{x:.1e}' for x in te)} | {'none' if arg is None else f'{arg:.6f}'} | "
              f"{', '.join(extra) or '-'} | {'yes' if not extra else 'no'} | {'yes' if not extra_nb else 'no'} |")
        check(same <= REPRO_TOL, f"{name}: {label} (difference {same:.1e})")
        if ring and not extra:
            print(f"  FINDING: {name} carries a ring ({zc} zeros) and survives at K = 19")
        elif ring and not extra_nb:
            print(f"  NOTE: {name} carries a ring ({zc} zeros), is rejected by D-bracket alone and survives without it")
    for r in RINGS17:
        o = out[r]
        print(f"\n{r}, a ring that survived sigma_18: survives sigma_19: {o['survives']} (without D-bracket: "
              f"{o['survives_without_bracket']}; extra detectors {o['extra_detectors'] or 'none'}; its sigma_19 is off by "
              f"{max(o['sigma19_rel']):.1e} relative)")
    save_checks("K19 ring test", out)


# ---------------------------------------------------------------- calibration
def calib_rows(by, real_est, K):
    rows = K14.calib_rows(by, real_est, K)
    for label, names in (("BP family (12)", BP_CAL_OLD), ("BP family (16)", BP_CAL)):
        bp = [by[n] for n in names if n in by]
        cells = {}
        for e in EST:
            v = [R.grade(r, K)["err"][e] for r in bp if R.grade(r, K)["err"][e] is not None]
            cells[e] = {"lo": min(v), "hi": max(v), "n": len(v), "zero": (real_est[e] - max(v), real_est[e] - min(v))} \
                if v and real_est[e] is not None else {"n": len(v)}
        rows[label] = {"n": len(bp), "cells": cells}
    return rows


def stage_calib():
    use19()
    by = new_records()
    real = by[REAL19]
    out = {}
    for K in (18, 19):
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
    a = out["18"]["all 28 arc plants"]["cells"]["arg"]["zero"]
    b = out["18"]["BP family (12)"]["cells"]["arg"]["zero"]
    s18 = json.load(open(K18.CHECKS18))["K18 combine"]
    sa, sb = s18["arc"], s18["bp"]
    check(all(abs(x - y) < 6e-7 for x, y in zip(a + b, sa + sb)),
          f"K = 18 calibration from the new records reproduces section 10.5 (arc {a[0]:.6f}-{a[1]:.6f}, BP family (12) "
          f"{b[0]:.6f}-{b[1]:.6f}; stored {sa[0]:.6f}-{sa[1]:.6f}, {sb[0]:.6f}-{sb[1]:.6f})")
    tr = {}
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B") or r["name"] in BP_CAL:
            tr[r["name"]] = {x: [R.grade(r, K)["err"][x] for K in KS19] for x in ("arg", "G", "pade")}
    out["per_plant_err_17_18_19"] = tr
    devs = load_checks().get("K19 plant sigma18-19 deviation", {})
    out["sigma_dev_summary"] = {}
    for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "C-EXP", "REFIT"):
        ns = [n for n in devs if n.startswith(f)]
        if ns:
            d18 = [max(devs[n][w][0] for w in devs[n]) for n in ns]
            d19 = [max(devs[n][w][1] for w in devs[n]) for n in ns]
            out["sigma_dev_summary"][f] = {"sigma18": [min(d18), max(d18)], "sigma19": [min(d19), max(d19)]}
            print(f"  {f}: plant sigma_18 vs measured {min(d18):.1e}-{max(d18):.1e}, sigma_19 {min(d19):.1e}-{max(d19):.1e}"
                  " (relative, worst node)")
    save_checks("K19 calib", out)


# ---------------------------------------------------------------- next generation and ring-free completions
def stage_recomplete():
    real = load_real19()
    by = new_records()
    e19 = R.grade(by[REAL19], 19)["est"]
    res = load_checks().get("K19 recomplete", {})
    wts = [f"{0.9990 + 0.0001 * i:.4f}" for i in range(21)]

    def finish(name, funcs, truth, extra, radii=(7.3, 12, 16)):
        f = funcs[1.0]
        counts = {f"zeros_lt_{r}": K13.count(f, mp.mpf(str(r))) for r in radii}
        rec = {"truth": truth["im_x"], "truth_im_sqrt": truth["im_sqrt"], **counts,
               "err_of_real_estimates": {e: e19[e] - truth["im_x"] for e in EST if e19[e] is not None}, **extra}
        res[name] = rec
        save_checks("K19 recomplete", res)
        print(f"{name}: zero at {truth['im_x']:.7f} (Im sqrt x: {truth['im_sqrt']:.7f}); {counts}; "
              f"{ {k: v for k, v in extra.items() if k not in ('x_R', 'poles_1.0', 'fit_resid_max', 'min_residue')} }",
              flush=True)

    for kind in ("C-W1", "C-W2", "C-W3"):
        name = kind.replace("C-", "C19-")
        if name in res:
            continue
        nodes, funcs, info, truth = PP.completion(kind, real=real, kc=19)
        finish(name, funcs, truth, {"kappa_beta": {str(w): info[w] for w in PP.NODES}}, radii=(7.3, 12))
    for kind in SHAPES:
        name = f"C19-ARC {kind}"
        if name in res:
            continue
        scan = PO.refit_scan(kind, wts, kc=19, real=real)
        bw = PO.best_wt(scan)
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{bw:.5f}", kc=19, real=real, funcs=funcs)
        finish(name, funcs, truth, {"best_wt": bw, "L2_floor": min(v[0] for v in scan.values()),
                                    "fit_resid_max": {w: info[w]["fit_resid_max"] for w in info}}, radii=(7.3, 12))
    for m in (3, 4, 5, 6, 7, 8, 9):                              # up to [9/10], as [8/9] was at K = 17
        n = 19 - m
        name = f"C19-BP [{m}/{n}]"
        if name in res:
            continue
        try:
            nodes, funcs, info = K14.borel_pade(real, m, n)
            truth, xr = K14.truth_of(funcs)
        except Exception as ex:                                  # noqa: BLE001  (recorded, not swallowed)
            res[name] = {"error": f"{type(ex).__name__}: {ex}"}
            save_checks("K19 recomplete", res)
            print(f"{name}: {res[name]['error']}")
            continue
        maxp = {w: info[w]["max_pole"] for w in info}
        finish(name, funcs, truth, {"max_pole": maxp, "type_rule": all(v < 2 for v in maxp.values()),
                                    "reproduce_max_rel": max(info[w]["reproduce_max_rel"] for w in info),
                                    "min_residue": {w: info[w]["min_residue"] for w in info},
                                    "x_R": {str(w): [float(z.real), float(z.imag)] for w, z in xr.items()},
                                    "poles_1.0": info["1.0"]["poles"]})


# ---------------------------------------------------------------- order-19 projection
def stage_project():
    use19()
    by = new_records()
    real = by[REAL19]
    a = {int(k): v for k, v in real["est"]["19"]["arg"].items()}
    a19 = a[19]
    print("real arg crossings k = 10..19: " + ", ".join(f"{a[k]:.7f}" for k in range(10, 20)))
    out = {"arg": {k: a[k] for k in range(4, 20)}, "hyp": {}}
    print(f"real arg_19 - 1 = {a19 - 1:+.3e}")
    for w in HYPOTHESES:
        p = K14.project(a, 19, w)
        back = {k: a[k] - K14.project(a, k, w) for k in range(8, 19)}
        out["hyp"][str(w)] = {"pred19": p, "miss": a19 - p, "back": back}
        print(f"  zero at {w}: predicted arg_19 - 1 = {p - 1:+.3e}; real - predicted = {a19 - p:+.2e}; back-test (real - "
              f"predicted) at k = 16, 17, 18: " + ", ".join(f"{back[k]:+.1e}" for k in (16, 17, 18)))
    for lo, hi in ((4, 18), (4, 19)):
        ks = list(range(lo, hi + 1))
        sl, ic = np.polyfit(ks, [np.log10(a[k] - 1) for k in ks], 1)
        out[f"trend_{lo}_{hi}"] = {"per_order": 10 ** sl, "at19": 10 ** (ic + 19 * sl)}
        print(f"  log-linear trend of arg_k - 1 (zero at 1) over k = {lo}-{hi}: x{10 ** sl:.3f} per order; "
              f"at k = 19: {10 ** (ic + 19 * sl):.3e}")
    ait = {k: K14.aitken(a, k) for k in range(6, 20)}
    out["aitken"] = ait
    print("  Aitken on k-2..k, k = 6..19: " + ", ".join(f"{ait[k]:.6f}" for k in range(6, 20)))
    pl = []
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            b = {int(k): v for k, v in r["est"]["19"]["arg"].items()}
            if None in (b.get(17), b.get(18), b.get(19)):
                continue
            pl.append({"name": r["name"], "miss": b[19] - K14.project(b, 19, r["truth"]),
                       "aitken_err": K14.aitken(b, 19) - r["truth"]})
    out["plants"] = pl
    miss = [p["miss"] for p in pl]
    print(f"  arc plants ({len(pl)}): model through k = 17, 18 with the true w_t misses arg_19 by "
          f"{min(miss):+.1e} ... {max(miss):+.1e}")
    p1, pb = out["hyp"]["1.0"]["pred19"], [out["hyp"][str(w)]["pred19"] for w in HYPOTHESES[1:]]
    d1 = abs(a19 - p1)
    db = min(abs(a19 - p) for p in pb) if not (min(pb) <= a19 <= max(pb)) else 0.0
    fav = "zero at 1" if d1 < db else "zero in 0.999975-0.999992"
    back_real = {str(w): max(abs(out["hyp"][str(w)]["back"][k]) for k in (16, 17, 18)) for w in HYPOTHESES}
    model_err = max(max(back_real.values()), max(abs(x) for x in miss))
    other = db if fav == "zero at 1" else d1
    out.update({"favoured": fav, "dist_to_1": d1, "dist_to_lower": db, "model_err": model_err,
                "back_real_max": back_real, "plant_miss_max": max(abs(x) for x in miss), "decisive": other > 3 * model_err})
    print(f"  favoured: {fav} (|arg_19 - prediction|: zero at 1 {d1:.1e}, lower {db:.1e}); model error {model_err:.1e}; "
          f"decisive: {out['decisive']}")
    rl = load_real19()
    rng = np.random.default_rng(19)
    sh = []
    mp.mp.dps = 50
    for seed in range(20):
        rows = []
        for wf in PP.NODES:
            sig = list(rl[wf][0])
            for k in range(1, 20):
                nz = rng.standard_normal(2) / np.sqrt(2)
                sig[k] = sig[k] + mp.mpf(rl[wf][1][k]) * mp.mpc(nz[0], nz[1])
            rows.append((wf, taus(sig)))
        vals = [float(mp.arg(t[19]) / mp.pi) % 2 - 1 for _, t in rows]
        c = np.polyfit(PP.NODES, vals, 2)
        rts = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.96 < x.real < 1.04]
        sh.append(min(rts, key=lambda x: abs(x - 1)) - a19)
    out["noise_shift"] = [min(sh), max(sh)]
    print(f"  arg_19 under sigma_err noise (20 seeds): shift {min(sh):+.1e} ... {max(sh):+.1e}")
    save_checks("K19 project", out)


# ---------------------------------------------------------------- combination (section 11.0, item 7)
def stage_combine():
    use19()
    by = new_records()
    c = load_checks()
    cal = c["K19 calib"]["19"]
    arc = cal["all 28 arc plants"]["cells"]["arg"]["zero"]
    bp = cal["BP family (16)"]["cells"]["arg"]["zero"]
    rf = {n: v["truth"] for n, v in c["K19 recomplete"].items()
          if n.startswith("C19-BP") and v.get("type_rule") and v.get("zeros_lt_12", 99) <= 1}
    parts = [tuple(arc), tuple(bp)] + [(z, z) for z in rf.values()]
    lo, hi = min(p[0] for p in parts), max(p[1] for p in parts)
    print(f"calibrated arg, 28 arc plants: {arc[0]:.6f}-{arc[1]:.6f}; BP family (16): {bp[0]:.6f}-{bp[1]:.6f}; "
          f"ring-free completions passing the type rule: {rf}")
    print(f"K = 19 estimator range (hull): {lo:.6f}-{hi:.6f}")
    overlap = lo <= K18_RANGE[1] and K18_RANGE[0] <= hi
    head = (min(lo, K18_RANGE[0]), max(hi, K18_RANGE[1])) if overlap else None
    print(f"K = 18 headline {K18_RANGE[0]}-{K18_RANGE[1]}: overlap {overlap}; headline (union) "
          + (f"{head[0]:.6f}-{head[1]:.6f}" if head else "none: disjoint, diagnose first"))
    # G over the minima of its objective (section 6.8)
    real = load_real19()
    rn = {w: real[w][0] for w in PP.NODES}
    gm = {}
    for K in (18, 19):
        mins = K14.kbar_minima(rn[1.0], K)
        ms = [{"kbar": [m[0], m[1]], "obj": m[2], "G": K14.G_at(rn, mp.mpc(m[0], m[1]), K)} for m in mins[:6]]
        g = [m["G"] for m in ms if m["obj"] <= 5 * ms[0]["obj"] and m["G"] is not None]
        prod = by[REAL19]["est"][str(K)]
        gm[str(K)] = {"minima": ms, "span": [min(g), max(g)], "global": ms[0]["G"], "production": prod["G"]}
        print(f"G at K = {K}: production {prod['G']:.6f}; over the minima within x5 of the best {min(g):.6f}-{max(g):.6f};"
              f" global minimum {ms[0]['G']:.6f}", flush=True)
    save_checks("K19 combine", {"arc": arc, "bp": bp, "ring_free": rf, "range19": [lo, hi], "overlap_k18": overlap,
                                "headline": head, "G_minima": gm})


def stage_table():
    c = load_checks()
    print(json.dumps({k: (v if "deviation" not in k else f"{len(v)} plants") for k, v in c.items()},
                     indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch19"))
    ap.add_argument("--mutate", default=None)
    ap.add_argument("--data", default=HERE)
    ap.add_argument("--amend", default=None, help="section 11.0's amendment of item 1: the J 1250 / 1280-bit files' directory")
    ap.add_argument("--nodes", default=None, help="section 11.0's second amendment of item 1: the J 1250 / 1280-bit files "
                                                  "are the node data (a directory, relative to this one unless absolute)")
    a = ap.parse_args()
    configure(a.data)
    global AMEND, NODES, REAL19
    AMEND = a.amend
    if a.nodes:
        if AMEND:
            raise SystemExit("--amend and --nodes exclude each other")
        NODES = a.nodes if os.path.isabs(a.nodes) else os.path.join(HERE, a.nodes)
        REAL19 = "N0 " + os.path.basename(os.path.normpath(NODES))
    {"converge": lambda: stage_converge(a.mutate), "plants": lambda: stage_plants(a.scratch),
     "real": lambda: stage_real(a.scratch, a.mutate), "ring": lambda: stage_ring(a.scratch), "calib": stage_calib,
     "recomplete": stage_recomplete, "project": stage_project, "combine": stage_combine, "table": stage_table,
     "plants-check": plants_check}[a.stage]()
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
