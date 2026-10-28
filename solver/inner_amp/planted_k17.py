#!/usr/bin/env python3
"""Post-freeze follow-up of the planted-truth test at K = 17 (notes/PLANTED_SELECTION.md, section 9): sigma_17
(scan35J1000: K 35, J 1000, 960 bits; pilot (b) in scan35J1250: J 1250, 1280 bits; both copied from the raw-data
repository's T86-mac). The procedure and its pass criteria are section 9.0; each stage exits non-zero when one of its
checks fails. The earlier orders' scripts provide the shared pieces: planted_k14.py (Borel-Pade sums, zeros, the
projection model, the kbar minima of G), planted_k15.py and planted_k16.py (their records and stored checks).

  converge    sigma_17's convergence (err17, pilot (b), sigma_0..16 unchanged, ratio law, spread in wf);
              --mutate=err|low|pilot|ratiolaw breaks one input on purpose, and the matching check must fire
  plants      N1, N2, the 28 arc plants, C-EXP and the 13 REFIT plants at K = 15, 16, 17 (planted_results_k17.jsonl,
              resumable); their K = 15, 16 estimates must equal the K = 16 records; N1 recovered at K = 17, N2 nothing
  real        the real sigma_0..17 ("N0 scan35J1000"); its K = 15, 16 estimates must equal "N0 scan33J1000"'s; the
              sigma_17 sign flip (and a 1e-3 scale) at wf 1.00 as mutation controls
  ring        the K = 16 completions of section 8.6 (C16-*), regenerated, and the two rings that survived sigma_16,
              C15-BP [4/11] and [5/10]: their sigma_17, tau_17 against the measured ones and the detectors at K = 17
              beyond the real data's, read with and without D-bracket (section 9.0 item 4); also the BP calibration
              family of section 8.5 (C13-BP [5/8], [6/7], C14-BP [5/9], [6/8], C15-BP [6/9], [7/8])
  calib       K = 17 calibration on the arc plants, C-EXP, REFIT and the BP family (nine sums), and the zero each
              estimator implies; the K = 16 calibration from the new records must reproduce section 8.5
  recomplete  the recipes glued at sigma_17 (tail from k = 18) and the Borel-Pade sums [m/(17 - m)], m = 3..8
  project     the order-17 projection (est_k = w* + c r^k through k = 15, 16) against the real arg_17, back-tests, noise
  combine     section 9.0 item 7: the K = 17 estimator range, G over the minima of its objective, the headline
  table       the stored checks, as JSON

    nice -n 10 python3 planted_k17.py STAGE [--scratch DIR] [--mutate=...] [--data DIR]

--data DIR reads the scans from DIR/scan35J1000 and DIR/scan35J1250 and writes the records and checks there (the dry
run of section 9.0 item 0, planted_k17_dry.py); the default is this directory.
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
import planted_plants as PP  # noqa: E402
import planted_post as PO  # noqa: E402
import planted_run as R  # noqa: E402
from arg_crossing import taus  # noqa: E402

SCAN17 = os.path.join(HERE, "scan35J1000")
PILOT17 = os.path.join(HERE, "scan35J1250")
SCAN16 = K16.SCAN16
REAL16 = K16.REAL16
REAL17 = "N0 scan35J1000"
OUT16 = K16.OUT16
OUT17 = os.path.join(HERE, "planted_results_k17.jsonl")
CHECKS17 = os.path.join(HERE, "planted_checks_k17.json")
KS17 = (15, 16, 17)
EST = K13.EST
ARC_FAMS = K14.ARC_FAMS
SHAPES = K14.SHAPES
HYPOTHESES = (1.0, 0.99998, 0.999975)       # zero at 1, at the lower end of the K = 16 range, of the headline
REPRO_TOL = 1e-9
BP_CAL_OLD = K16.BP_CAL                                                 # section 8.5's BP family (six sums)
BP_CAL = BP_CAL_OLD + ("C16-BP 6-10", "C16-BP 7-9", "C16-BP 8-8")      # plus section 8.6's ring-free, type-passing sums
RINGS15 = ("C15-BP [4/11]", "C15-BP [5/10]")                            # the rings that survived sigma_16 (section 8.4)
K16_RANGE = (0.999975, 1.000050)                                        # the K = 16 headline (section 8.8)
BRACKET = "D-bracket"                                                   # left out in the second reading (section 9.0 item 4)
FAILED = []


def configure(data):
    global SCAN17, PILOT17, OUT17, CHECKS17
    SCAN17 = os.path.join(data, "scan35J1000")
    PILOT17 = os.path.join(data, "scan35J1250")
    OUT17 = os.path.join(data, "planted_results_k17.jsonl")
    CHECKS17 = os.path.join(data, "planted_checks_k17.json")


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILED.append(msg)
    return ok


def use17():
    R.OUT, R.KS = OUT17, KS17
    R.DONE.clear()
    R.DONE.update(R.done_names())


def load_checks():
    return json.load(open(CHECKS17)) if os.path.exists(CHECKS17) else {}


def save_checks(key, val):
    c = load_checks()
    c[key] = val
    with open(CHECKS17, "w") as f:
        json.dump(c, f, default=R.jdefault, indent=1)


def load_real(scandir, pattern):
    mp.mp.dps = PP.DPS
    return {float(w): (K14.sigmas(r), list(r["sigma_err"])) for w, r in K14.raw_scan(scandir, pattern).items()}


def load_real17():
    return load_real(SCAN17, "*_p960.json")


def load_real16():
    return load_real(SCAN16, "*_p960.json")


def k16_records():
    return {r["name"]: r for r in (json.loads(line) for line in open(OUT16) if line.strip())}


def new_records():
    use17()
    return {r["name"]: r for r in R.records()}


# ---------------------------------------------------------------- converge
def stage_converge(mutate=None):
    mp.mp.dps = 50
    p960 = K14.raw_scan(SCAN17, "*_p960.json")
    pil = K14.raw_scan(PILOT17, "*.json")
    old = K14.raw_scan(SCAN16, "*_p960.json")
    print(f"files: 960 bits at wf {sorted(p960)}; pilot (b) at wf {sorted(pil)}; scan33J1000 at wf {sorted(old)}")
    for w, r in sorted(p960.items()):
        print(f"  wf {w}: K {r['K']} J {r['J']} prec {r['prec']} fit {r.get('fit')} mf {r.get('mf')}, "
              f"{len(r['sigma_str'])} coefficients")
    if mutate == "err":
        p960[1.0]["sigma_err"][17] *= 1e14
    elif mutate == "low":
        a = mp.mpf(p960[0.99]["sigma_str"][16][0]) * (1 + mp.mpf("1e-15"))
        p960[0.99]["sigma_str"][16][0] = mp.nstr(a, 20)
    elif mutate == "pilot":
        b = mp.mpf(pil[1.0]["sigma_str"][17][1]) * (1 + mp.mpf("1e-10"))
        pil[1.0]["sigma_str"][17][1] = mp.nstr(b, 20)
    elif mutate == "ratiolaw":
        p960[1.01]["sigma_str"][17] = [mp.nstr(mp.mpf(x) * mp.mpf("1.1"), 20) for x in p960[1.01]["sigma_str"][17]]
    elif mutate:
        raise SystemExit(f"unknown mutation {mutate}")
    out = {"nodes": {}}
    print("(a) err17 against the 1e-10 relative requirement (section 3.6):")
    for w, r in sorted(p960.items()):
        s = K14.sigmas(r)
        rel = r["sigma_err"][17] / float(abs(s[17]))
        out["nodes"][str(w)] = {"sigma17": r["sigma_str"][17], "err17": r["sigma_err"][17], "err17_rel": rel,
                                "err16": r["sigma_err"][16]}
        check(rel <= 1e-10, f"wf {w}: sigma_17 = {mp.nstr(s[17], 20)}, err17 {r['sigma_err'][17]:.3g} ({rel:.2e} relative)")
    print("(b) sigma_0..16 against scan33J1000 (960 bits):")
    for w, r in sorted(p960.items()):
        same = [r["sigma_str"][k] == old[w]["sigma_str"][k] for k in range(17)]
        erre = [r["sigma_err"][k] == old[w]["sigma_err"][k] for k in range(17)]
        out["nodes"][str(w)]["low_bitwise"] = all(same)
        check(all(same), f"wf {w}: sigma_0..16 bitwise equal ({sum(same)}/17); sigma_err_0..16 equal ({sum(erre)}/17)")
    print("(c) pilot (b) at wf 1.00 (J 1250, 1280 bits) against (a):")
    a, b = p960[1.0], pil[1.0]
    sa, sb, so = K14.sigmas(a), K14.sigmas(b), K14.sigmas(old[1.0])
    d17 = float(abs(sa[17] - sb[17]))
    low = {k: (float(abs(sb[k] - so[k])), old[1.0]["sigma_err"][k]) for k in (14, 15, 16)}
    out["pilot"] = {"d17": d17, "err17": a["sigma_err"][17], "low": low, "pilot_sigma_err": b["sigma_err"][:3]}
    check(d17 <= a["sigma_err"][17], f"sigma_17: |a - b| = {d17:.2e} against err17 {a['sigma_err'][17]:.2e}")
    check(all(d <= e for d, e in low.values()), "sigma_14..16 of (b) against scan33J1000: " + ", ".join(
        f"k {k}: {d:.1e} (err {e:.1e})" for k, (d, e) in low.items()))
    print(f"  note: (b)'s own sigma_err is {b['sigma_err'][17]} at every k (its order-1 fit has residual 0), so (b) checks "
          f"(a)'s values, not their error")
    print("(d) the ratio law through sigma_17: second difference of sigma_k/sigma_{k+1}, k = 7..15 (D-ratiolaw <= 0.05):")
    out["ratiolaw_d2"] = {}
    for w, r in sorted(p960.items()):
        s = K14.sigmas(r)
        rat = [complex(s[k] / s[k + 1]) for k in range(17)]
        d2 = [abs(rat[k + 1] - 2 * rat[k] + rat[k - 1]) for k in range(7, 16)]
        out["ratiolaw_d2"][str(w)] = d2
        check(d2[-1] <= 0.05, f"wf {w}: |d2| k = 7..15: " + " ".join(f"{x:.4f}" for x in d2)
              + f"; sigma_16/sigma_17 = {rat[16]:.4f}")
    print("spread in wf (context): the change of sigma_k across the nodes, against its error")
    out["spread"] = {}
    for k in (15, 16, 17):
        s = {w: K14.sigmas(r)[k] for w, r in p960.items()}
        d1 = abs(s[1.01] - s[0.99]) / 2
        d2 = abs(s[1.01] - 2 * s[1.0] + s[0.99])
        e = max(r["sigma_err"][k] for r in p960.values())
        out["spread"][str(k)] = {"half_range_rel": float(d1 / abs(s[1.0])), "d2_over_d1": float(d2 / d1),
                                 "err_over_d1": float(e / d1)}
        print(f"  k = {k}: |sigma_k(1.01) - sigma_k(0.99)|/2 = {float(d1 / abs(s[1.0])):.4f} |sigma_k(1.00)|; "
              f"second difference / first = {float(d2 / d1):.4f}; max err_k / first difference = {float(e / d1):.1e}")
    if not mutate:
        save_checks("K17 converge", out)
    return out


# ---------------------------------------------------------------- plants
def sigma_dev(nodes, real17):
    return {str(w): [float(abs(nodes[w][k] - real17[w][0][k]) / abs(real17[w][0][k])) for k in (16, 17)]
            for w in PP.NODES}


def stage_plants(scratch):
    use17()
    real17 = load_real17()
    devs = load_checks().get("K17 plant sigma16-17 deviation", {})
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
            devs[name] = sigma_dev(nodes, real17)
            save_checks("K17 plant sigma16-17 deviation", devs)
            R.run_plant(name, tag.split()[0], nodes, float(wt), scratch,
                        info={"fit_resid_max": {str(w): fam.resid[w] and max(fam.resid[w]) for w in PP.NODES}})
    if "C-EXP" not in R.DONE:
        nodes, _, info, truth = PP.completion("C-EXP")
        devs["C-EXP"] = sigma_dev(nodes, real17)
        save_checks("K17 plant sigma16-17 deviation", devs)
        R.run_plant("C-EXP", "C", nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    for wt in R.REFIT_WT:
        name = f"REFIT P-ARC wt={wt}"
        if name in R.DONE:
            continue
        nodes, resid = K14.refit_nodes(wt)
        devs[name] = sigma_dev(nodes, real17)
        save_checks("K17 plant sigma16-17 deviation", devs)
        R.run_plant(name, "REFIT", nodes, float(wt), scratch, info={"fit_resid_max": resid}, linear=False)
    plants_check()


def plants_check():
    old, new = k16_records(), new_records()
    names = [n for n in new if not n.startswith(("N0", "C13", "C14", "C15", "C16", "C17"))]
    print(f"\n{len(names)} plants with K = 17 records")
    worst = (0.0, None)
    for n in names:
        for K in (15, 16):
            d, k = K14.max_diff(K14.est_vector(new[n], K), K14.est_vector(old[n], K))
            if d > worst[0]:
                worst = (d, f"{n} K{K} {k}")
    check(worst[0] <= REPRO_TOL, f"regenerated plants reproduce the stored K = 15, 16 estimates: largest difference "
                                 f"{worst[0]:.1e} ({worst[1]})")
    n1 = [new[n] for n in names if n.startswith("N1")]
    errs = [abs(R.grade(r, 17)["est"][e] - r["truth"]) for r in n1 for e in ("arg", "ratio", "pade", "G")
            if R.grade(r, 17)["est"][e] is not None]
    nnone = sum(R.grade(r, 17)["est"][e] is None for r in n1 for e in ("arg", "ratio", "pade", "G"))
    check(len(n1) == 7 and nnone == 0 and max(errs) <= 1e-6,
          f"N1 at K = 17: {len(n1)} lone zeros, worst error {max(errs):.1e} over arg, ratio, Pade, G ({nnone} missing)")
    ctl = [abs(R.grade(r, 17)["est"]["arg"] - (r["truth"] + 0.002)) <= 1e-6 for r in n1]
    check(not any(ctl), "control: N1 graded against w_t + 0.002 fails (as it must)")
    n2 = [new[n] for n in names if n.startswith("N2")]
    printed = [(r["name"], e) for r in n2 for e, v in R.grade(r, 17)["est"].items() if v is not None]
    check(len(n2) == 2 and not printed, f"N2 at K = 17: no estimator prints a number ({printed or 'none'})")
    fams = {}
    for n in names:
        fams.setdefault(new[n]["family"], []).append(n)
    print("  records per family:", {f: len(v) for f, v in fams.items()})
    save_checks("K17 plants check", {"repro_worst": worst, "n1_worst": max(errs), "n2_printed": printed,
                                     "families": {f: len(v) for f, v in fams.items()}})


# ---------------------------------------------------------------- real data
def stage_real(scratch):
    use17()
    real = load_real17()
    R.run_plant(REAL17, "N0", {wf: real[wf][0] for wf in PP.NODES}, None, scratch)
    flip = {wf: list(real[wf][0]) for wf in PP.NODES}
    flip[1.0][17] = -flip[1.0][17]
    R.run_plant(f"{REAL17} X1 flip sigma_17", "N0 mutation", flip, None, scratch)
    sc = {wf: list(real[wf][0]) for wf in PP.NODES}
    sc[1.0][17] = sc[1.0][17] * (1 + mp.mpf("1e-3"))
    R.run_plant(f"{REAL17} X2 scale sigma_17", "N0 mutation", sc, None, scratch)
    old, new = k16_records(), new_records()
    rec = new[REAL17]
    for K in (15, 16):
        d, k = K14.max_diff(K14.est_vector(rec, K), K14.est_vector(old[REAL16], K))
        check(d <= REPRO_TOL, f"K = {K}: the scan35J1000 estimates equal those of scan33J1000 (largest difference "
                              f"{d:.1e}{', ' + k if k else ''})")
    for K in KS17:
        g = R.grade(rec, K)["est"]
        print(f"  K = {K}: " + ", ".join(f"{e} {v:.7f}" if v is not None else f"{e} None" for e, v in g.items())
              + f"; detectors {R.detectors(rec, K)}")
    e = rec["est"]["17"]
    print("  arg crossings k = 4..17:", {int(k): round(v, 7) for k, v in e["arg"].items()})
    print("  ratio estimates k = 9..17:", {int(k): round(v, 6) for k, v in e["ratio"].items()})
    print("  Pade central", e["pade_central"], "values", [round(v, 6) if v else v for v in e["pade_central_vals"]],
          "range", e["pade_central_range"], "mid", e["pade_central_mid"])
    print("  all Pade crossings K = 17:", {k: round(v, 6) if v else v for k, v in e["pade"].items()})
    print("  G K = 16, 17 via G_prev/G:", e.get("G_prev"), e.get("G"), "kbar", e.get("G_kbar"))
    det = set(R.detectors(rec, 17))
    for n in (f"{REAL17} X1 flip sigma_17", f"{REAL17} X2 scale sigma_17"):
        m = new[n]
        extra = sorted(set(R.detectors(m, 17)) - det)
        g, g0 = R.grade(m, 17)["est"], R.grade(rec, 17)["est"]
        shift = max((abs(g[x] - g0[x]) if None not in (g[x], g0[x]) else float("inf")) for x in EST)
        check(bool(extra) if "flip" in n else True, f"mutation {n.split()[-2]} sigma_17: new detectors {extra or 'none'}, "
                                                     f"largest shift {shift:.1e}")
    # the effective exponent at w_inf from k - 1, k (the README's column)
    mp.mp.dps = 50
    rt = taus(list(real[1.0][0][:18]))
    ce = {}
    for k in (15, 16, 17):
        x0 = (k - 1) * rt[k - 1] / (k * rt[k])
        ce[k] = complex(-k * rt[k] * x0 ** k)
    print("  effective exponent c at wf 1.00 (two orders): " + ", ".join(f"k {k}: {v.real:.3f}{v.imag:+.3f}i"
                                                                          for k, v in ce.items()))
    save_checks("K17 real", {"est": {str(K): R.grade(rec, K)["est"] for K in KS17},
                             "detectors": {str(K): R.detectors(rec, K) for K in KS17},
                             "c_eff_wf1": {str(k): [v.real, v.imag] for k, v in ce.items()}})


# ---------------------------------------------------------------- ring test
def c16_completions():
    """section 8.6's K = 16 completions (tail from k = 17), regenerated; the two rings that survived sigma_16
    (C15-BP [4/11], [5/10]); and the BP calibration sums of K = 13, 14 and 15"""
    real16 = load_real16()
    stored16 = json.load(open(K16.CHECKS16))["K16 recomplete"]
    stored15 = json.load(open(K15.CHECKS15))["K15 recomplete"]
    stored14 = json.load(open(K14.CHECKS14))["K14 recomplete"]
    out = {}
    for kind in ("C-W1", "C-W2", "C-W3"):
        nodes, funcs, info, truth = PP.completion(kind, real=real16, kc=16)
        out[kind.replace("C-", "C16-")] = (nodes, funcs, truth, stored16[kind.replace("C-", "C16-")])
    for kind in SHAPES:
        name = f"C16-ARC {kind}"
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{stored16[name]['best_wt']:.5f}", kc=16, real=real16, funcs=funcs)
        out[name] = (nodes, funcs, truth, stored16[name])
    for m in (3, 4, 5, 6, 7, 8):
        name = f"C16-BP [{m}/{16 - m}]"
        nodes, funcs, info = K14.borel_pade(real16, m, 16 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored16[name])
    real15 = K16.load_real15()
    for m in (4, 5, 6, 7):                       # [4/11], [5/10]: the surviving rings; [6/9], [7/8]: the calibration family
        name = f"C15-BP [{m}/{15 - m}]"
        nodes, funcs, info = K14.borel_pade(real15, m, 15 - m)
        truth, _ = K14.truth_of(funcs)
        out[name] = (nodes, funcs, truth, stored15[name])
    real14 = K15.load_real14()
    for m in (5, 6):                             # the calibration family
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
    if name.startswith("C16"):
        return "C16 (ring test)"
    if name in RINGS15:
        return "C15-BP (ring test)"
    return name[:6] + " (calibration)"


def stage_ring(scratch):
    use17()
    real = load_real17()
    mp.mp.dps = 50
    rt = {wf: taus(list(real[wf][0][:18])) for wf in PP.NODES}
    comps = c16_completions()
    for name, (nodes, _, truth, _) in comps.items():
        R.run_plant(rec_name(name), family_of(name), nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    new = new_records()
    real_rec = new[REAL17]
    real_det = set(R.detectors(real_rec, 17))
    a17 = R.grade(real_rec, 17)["est"]["arg"]
    print("real: sigma_17 / tau_17 at the nodes: " + "; ".join(
        f"{wf}: {mp.nstr(real[wf][0][17], 8)} / {mp.nstr(rt[wf][17], 8)}" for wf in PP.NODES))
    print(f"real arg_17 = {a17:.7f}; real K = 17 detectors: {sorted(real_det)}\n")
    out = {"simple-zero law from tau_15, tau_16": {"tau17_rel": K14.simple_zero_rel(rt, 17)},
           "simple-zero law from tau_14, tau_15 (tau_16, for comparison)": {"tau16_rel": K14.simple_zero_rel(rt, 16)}}
    print("| completion | its zero | zeros in |x| < 7.3 / 12 / 16 | sigma_17 rel. error (0.99 / 1.00 / 1.01) | "
          "tau_17 rel. error | its arg_17 | K = 17 detectors beyond the real data's | survives | without D-bracket |")
    print("|---|---|---|---|---|---|---|---|---|")
    sz = " / ".join(f"{x:.1e}" for x in K14.simple_zero_rel(rt, 17))
    print(f"| simple-zero law from tau_15, tau_16 | - | - | - | {sz} | - | - | - | - |")
    for name, (nodes, _, truth, st) in comps.items():
        mp.mp.dps = 50
        se = [float(abs(nodes[wf][17] - real[wf][0][17]) / abs(real[wf][0][17])) for wf in PP.NODES]
        te = [float(abs(taus(list(nodes[wf][:18]))[17] - rt[wf][17]) / abs(rt[wf][17])) for wf in PP.NODES]
        rec = new[rec_name(name)]
        arg = R.grade(rec, 17)["est"]["arg"]
        extra = sorted(set(R.detectors(rec, 17)) - real_det)
        extra_nb = [d for d in extra if d != BRACKET]                   # the second reading of section 9.0 item 4
        kq = int(name[1:3])                       # the order whose real coefficients the completion carries
        if kq in KS17:
            same = K14.max_diff(K14.est_vector(rec, kq), K14.est_vector(real_rec, kq))[0]
            label = f"its K = {kq} estimates are the real ones"
        else:                                     # glued below K = 15: its K = 15, 16 estimates are the stored record's
            old = k16_records()[rec_name(name)]
            same = max(K14.max_diff(K14.est_vector(rec, K), K14.est_vector(old, K))[0] for K in (15, 16))
            label = "its K = 15, 16 estimates reproduce the stored K = 16 record"
        zc = " / ".join(str(st.get(f"zeros_lt_{r}", "-")) for r in (7.3, 12, 16))
        ring = (st.get("zeros_lt_12") or 0) > 1
        out[name] = {"sigma17_rel": se, "tau17_rel": te, "arg17": arg, "extra_detectors": extra, "survives": not extra,
                     "survives_without_bracket": not extra_nb, "truth": truth["im_x"], "zeros": zc, "ring": ring,
                     (f"K{kq}_equals_real" if kq in KS17 else "K15_16_equal_stored"): same,
                     "est17": R.grade(rec, 17)["est"]}
        print(f"| {name} | {truth['im_x']:.6f} | {zc} | {' / '.join(f'{x:.1e}' for x in se)} | "
              f"{' / '.join(f'{x:.1e}' for x in te)} | {'none' if arg is None else f'{arg:.6f}'} | "
              f"{', '.join(extra) or '-'} | {'yes' if not extra else 'no'} | {'yes' if not extra_nb else 'no'} |")
        check(same <= REPRO_TOL, f"{name}: {label} (difference {same:.1e})")
        if ring and not extra:
            print(f"  FINDING: {name} carries a ring ({zc} zeros) and survives at K = 17")
        elif ring and not extra_nb:
            print(f"  NOTE: {name} carries a ring ({zc} zeros), is rejected by D-bracket alone and survives without it")
    for r in RINGS15:
        o = out[r]
        print(f"\n{r}, a ring that survived sigma_16: survives sigma_17: {o['survives']} (without D-bracket: "
              f"{o['survives_without_bracket']}; extra detectors {o['extra_detectors'] or 'none'}; its sigma_17 is off by "
              f"{max(o['sigma17_rel']):.1e} relative)")
    save_checks("K17 ring test", out)


# ---------------------------------------------------------------- calibration
def calib_rows(by, real_est, K):
    rows = K14.calib_rows(by, real_est, K)
    for label, names in (("BP family (6)", BP_CAL_OLD), ("BP family (9)", BP_CAL)):
        bp = [by[n] for n in names if n in by]
        cells = {}
        for e in EST:
            v = [R.grade(r, K)["err"][e] for r in bp if R.grade(r, K)["err"][e] is not None]
            cells[e] = {"lo": min(v), "hi": max(v), "n": len(v), "zero": (real_est[e] - max(v), real_est[e] - min(v))} \
                if v and real_est[e] is not None else {"n": len(v)}
        rows[label] = {"n": len(bp), "cells": cells}
    return rows


def stage_calib():
    use17()
    by = new_records()
    real = by[REAL17]
    out = {}
    for K in (16, 17):
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
    a = out["16"]["all 28 arc plants"]["cells"]["arg"]["zero"]
    b = out["16"]["BP family (6)"]["cells"]["arg"]["zero"]
    s16 = json.load(open(K16.CHECKS16))["K16 combine"]
    sa, sb = s16["arc"], s16["bp"]
    check(all(abs(x - y) < 6e-7 for x, y in zip(a + b, sa + sb)),
          f"K = 16 calibration from the new records reproduces section 8.5 (arc {a[0]:.6f}-{a[1]:.6f}, BP family (6) "
          f"{b[0]:.6f}-{b[1]:.6f}; stored {sa[0]:.6f}-{sa[1]:.6f}, {sb[0]:.6f}-{sb[1]:.6f})")
    tr = {}
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B") or r["name"] in BP_CAL:
            tr[r["name"]] = {x: [R.grade(r, K)["err"][x] for K in KS17] for x in ("arg", "G", "pade")}
    out["per_plant_err_15_16_17"] = tr
    devs = load_checks().get("K17 plant sigma16-17 deviation", {})
    out["sigma_dev_summary"] = {}
    for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "C-EXP", "REFIT"):
        ns = [n for n in devs if n.startswith(f)]
        if ns:
            d16 = [max(devs[n][w][0] for w in devs[n]) for n in ns]
            d17 = [max(devs[n][w][1] for w in devs[n]) for n in ns]
            out["sigma_dev_summary"][f] = {"sigma16": [min(d16), max(d16)], "sigma17": [min(d17), max(d17)]}
            print(f"  {f}: plant sigma_16 vs measured {min(d16):.1e}-{max(d16):.1e}, sigma_17 {min(d17):.1e}-{max(d17):.1e}"
                  " (relative, worst node)")
    save_checks("K17 calib", out)


# ---------------------------------------------------------------- next generation and ring-free completions
def stage_recomplete():
    real = load_real17()
    by = new_records()
    e17 = R.grade(by[REAL17], 17)["est"]
    res = load_checks().get("K17 recomplete", {})
    wts = [f"{0.9990 + 0.0001 * i:.4f}" for i in range(21)]

    def finish(name, funcs, truth, extra, radii=(7.3, 12, 16)):
        f = funcs[1.0]
        counts = {f"zeros_lt_{r}": K13.count(f, mp.mpf(str(r))) for r in radii}
        rec = {"truth": truth["im_x"], "truth_im_sqrt": truth["im_sqrt"], **counts,
               "err_of_real_estimates": {e: e17[e] - truth["im_x"] for e in EST if e17[e] is not None}, **extra}
        res[name] = rec
        save_checks("K17 recomplete", res)
        print(f"{name}: zero at {truth['im_x']:.7f} (Im sqrt x: {truth['im_sqrt']:.7f}); {counts}; "
              f"{ {k: v for k, v in extra.items() if k not in ('x_R', 'poles_1.0', 'fit_resid_max', 'min_residue')} }",
              flush=True)

    for kind in ("C-W1", "C-W2", "C-W3"):
        name = kind.replace("C-", "C17-")
        if name in res:
            continue
        nodes, funcs, info, truth = PP.completion(kind, real=real, kc=17)
        finish(name, funcs, truth, {"kappa_beta": {str(w): info[w] for w in PP.NODES}}, radii=(7.3, 12))
    for kind in SHAPES:
        name = f"C17-ARC {kind}"
        if name in res:
            continue
        scan = PO.refit_scan(kind, wts, kc=17, real=real)
        bw = PO.best_wt(scan)
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{bw:.5f}", kc=17, real=real, funcs=funcs)
        finish(name, funcs, truth, {"best_wt": bw, "L2_floor": min(v[0] for v in scan.values()),
                                    "fit_resid_max": {w: info[w]["fit_resid_max"] for w in info}}, radii=(7.3, 12))
    for m in (3, 4, 5, 6, 7, 8):
        n = 17 - m
        name = f"C17-BP [{m}/{n}]"
        if name in res:
            continue
        try:
            nodes, funcs, info = K14.borel_pade(real, m, n)
            truth, xr = K14.truth_of(funcs)
        except Exception as ex:                                  # noqa: BLE001  (recorded, not swallowed)
            res[name] = {"error": f"{type(ex).__name__}: {ex}"}
            save_checks("K17 recomplete", res)
            print(f"{name}: {res[name]['error']}")
            continue
        maxp = {w: info[w]["max_pole"] for w in info}
        finish(name, funcs, truth, {"max_pole": maxp, "type_rule": all(v < 2 for v in maxp.values()),
                                    "reproduce_max_rel": max(info[w]["reproduce_max_rel"] for w in info),
                                    "min_residue": {w: info[w]["min_residue"] for w in info},
                                    "x_R": {str(w): [float(z.real), float(z.imag)] for w, z in xr.items()},
                                    "poles_1.0": info["1.0"]["poles"]})


# ---------------------------------------------------------------- order-17 projection
def stage_project():
    use17()
    by = new_records()
    real = by[REAL17]
    a = {int(k): v for k, v in real["est"]["17"]["arg"].items()}
    a17 = a[17]
    print("real arg crossings k = 10..17: " + ", ".join(f"{a[k]:.7f}" for k in range(10, 18)))
    out = {"arg": {k: a[k] for k in range(4, 18)}, "hyp": {}}
    print(f"real arg_17 - 1 = {a17 - 1:+.3e}")
    for w in HYPOTHESES:
        p = K14.project(a, 17, w)
        back = {k: a[k] - K14.project(a, k, w) for k in range(8, 17)}
        out["hyp"][str(w)] = {"pred17": p, "miss": a17 - p, "back": back}
        print(f"  zero at {w}: predicted arg_17 - 1 = {p - 1:+.3e}; real - predicted = {a17 - p:+.2e}; back-test (real - "
              f"predicted) at k = 14, 15, 16: " + ", ".join(f"{back[k]:+.1e}" for k in (14, 15, 16)))
    for lo, hi in ((4, 16), (4, 17)):
        ks = list(range(lo, hi + 1))
        sl, ic = np.polyfit(ks, [np.log10(a[k] - 1) for k in ks], 1)
        out[f"trend_{lo}_{hi}"] = {"per_order": 10 ** sl, "at17": 10 ** (ic + 17 * sl)}
        print(f"  log-linear trend of arg_k - 1 (zero at 1) over k = {lo}-{hi}: x{10 ** sl:.3f} per order; "
              f"at k = 17: {10 ** (ic + 17 * sl):.3e}")
    ait = {k: K14.aitken(a, k) for k in range(6, 18)}
    out["aitken"] = ait
    print("  Aitken on k-2..k, k = 6..17: " + ", ".join(f"{ait[k]:.6f}" for k in range(6, 18)))
    pl = []
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            b = {int(k): v for k, v in r["est"]["17"]["arg"].items()}
            if None in (b.get(15), b.get(16), b.get(17)):
                continue
            pl.append({"name": r["name"], "miss": b[17] - K14.project(b, 17, r["truth"]),
                       "aitken_err": K14.aitken(b, 17) - r["truth"]})
    out["plants"] = pl
    miss = [p["miss"] for p in pl]
    print(f"  arc plants ({len(pl)}): model through k = 15, 16 with the true w_t misses arg_17 by "
          f"{min(miss):+.1e} ... {max(miss):+.1e}")
    p1, pb = out["hyp"]["1.0"]["pred17"], [out["hyp"][str(w)]["pred17"] for w in HYPOTHESES[1:]]
    d1 = abs(a17 - p1)
    db = min(abs(a17 - p) for p in pb) if not (min(pb) <= a17 <= max(pb)) else 0.0
    fav = "zero at 1" if d1 < db else "zero in 0.999975-0.99998"
    back_real = {str(w): max(abs(out["hyp"][str(w)]["back"][k]) for k in (14, 15, 16)) for w in HYPOTHESES}
    model_err = max(max(back_real.values()), max(abs(x) for x in miss))
    other = db if fav == "zero at 1" else d1
    out.update({"favoured": fav, "dist_to_1": d1, "dist_to_lower": db, "model_err": model_err,
                "back_real_max": back_real, "plant_miss_max": max(abs(x) for x in miss), "decisive": other > 3 * model_err})
    print(f"  favoured: {fav} (|arg_17 - prediction|: zero at 1 {d1:.1e}, lower {db:.1e}); model error {model_err:.1e}; "
          f"decisive: {out['decisive']}")
    rl = load_real17()
    rng = np.random.default_rng(17)
    sh = []
    mp.mp.dps = 50
    for seed in range(20):
        rows = []
        for wf in PP.NODES:
            sig = list(rl[wf][0])
            for k in range(1, 18):
                nz = rng.standard_normal(2) / np.sqrt(2)
                sig[k] = sig[k] + mp.mpf(rl[wf][1][k]) * mp.mpc(nz[0], nz[1])
            rows.append((wf, taus(sig)))
        vals = [float(mp.arg(t[17]) / mp.pi) % 2 - 1 for _, t in rows]
        c = np.polyfit(PP.NODES, vals, 2)
        rts = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.96 < x.real < 1.04]
        sh.append(min(rts, key=lambda x: abs(x - 1)) - a17)
    out["noise_shift"] = [min(sh), max(sh)]
    print(f"  arg_17 under sigma_err noise (20 seeds): shift {min(sh):+.1e} ... {max(sh):+.1e}")
    save_checks("K17 project", out)


# ---------------------------------------------------------------- combination (section 9.0, item 7)
def stage_combine():
    use17()
    by = new_records()
    c = load_checks()
    cal = c["K17 calib"]["17"]
    arc = cal["all 28 arc plants"]["cells"]["arg"]["zero"]
    bp = cal["BP family (9)"]["cells"]["arg"]["zero"]
    rf = {n: v["truth"] for n, v in c["K17 recomplete"].items()
          if n.startswith("C17-BP") and v.get("type_rule") and v.get("zeros_lt_12", 99) <= 1}
    parts = [tuple(arc), tuple(bp)] + [(z, z) for z in rf.values()]
    lo, hi = min(p[0] for p in parts), max(p[1] for p in parts)
    print(f"calibrated arg, 28 arc plants: {arc[0]:.6f}-{arc[1]:.6f}; BP family (9): {bp[0]:.6f}-{bp[1]:.6f}; "
          f"ring-free completions passing the type rule: {rf}")
    print(f"K = 17 estimator range (hull): {lo:.6f}-{hi:.6f}")
    overlap = lo <= K16_RANGE[1] and K16_RANGE[0] <= hi
    head = (min(lo, K16_RANGE[0]), max(hi, K16_RANGE[1])) if overlap else None
    print(f"K = 16 headline {K16_RANGE[0]}-{K16_RANGE[1]}: overlap {overlap}; headline (union) "
          + (f"{head[0]:.6f}-{head[1]:.6f}" if head else "none: disjoint, diagnose first"))
    # G over the minima of its objective (section 6.8)
    real = load_real17()
    rn = {w: real[w][0] for w in PP.NODES}
    gm = {}
    for K in (16, 17):
        mins = K14.kbar_minima(rn[1.0], K)
        ms = [{"kbar": [m[0], m[1]], "obj": m[2], "G": K14.G_at(rn, mp.mpc(m[0], m[1]), K)} for m in mins[:6]]
        g = [m["G"] for m in ms if m["obj"] <= 5 * ms[0]["obj"] and m["G"] is not None]
        prod = by[REAL17]["est"][str(K)]
        gm[str(K)] = {"minima": ms, "span": [min(g), max(g)], "global": ms[0]["G"], "production": prod["G"]}
        print(f"G at K = {K}: production {prod['G']:.6f}; over the minima within x5 of the best {min(g):.6f}-{max(g):.6f};"
              f" global minimum {ms[0]['G']:.6f}", flush=True)
    save_checks("K17 combine", {"arc": arc, "bp": bp, "ring_free": rf, "range17": [lo, hi], "overlap_k16": overlap,
                                "headline": head, "G_minima": gm})


def stage_table():
    c = load_checks()
    print(json.dumps({k: (v if "deviation" not in k else f"{len(v)} plants") for k, v in c.items()},
                     indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch17"))
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
