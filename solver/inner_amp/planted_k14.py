#!/usr/bin/env python3
"""Post-freeze follow-up of the planted-truth test at K = 14 (notes/PLANTED_SELECTION.md, section 6): the converged
sigma_14 (scan29J1000: K 29, J 1000, 960 bits, the _p960 files; a 1280-bit pilot at wf 1.00). The procedure and its
pass criteria are section 6.0; each stage exits non-zero when one of its checks fails.

  converge    sigma_14's convergence (err14, the 1280-bit pilot, sigma_0..13 unchanged, ratio law, spread in wf);
              --mutate=err|low|pilot|ratiolaw breaks one input on purpose, and the matching check must fire
  plants      N1, N2, the 28 arc plants, C-EXP and the 13 REFIT plants at K = 12, 13, 14 (records appended to
              planted_results_k14.jsonl, resumable); the regenerated plants' K = 12, 13 estimates must equal the stored
              records (planted_results.jsonl), N1 must be recovered at K = 14 and N2 give nothing
  real        the real sigma_0..14 through the harness ("N0 scan29J1000"); its K = 12, 13 estimates must equal
              "N0 scan27J1000p960"; the sigma_14 sign flip (and a 1e-3 scale) at wf 1.00 as mutation controls
  ring        the K = 13 re-glued completions of section 5.3 (C13-*): their predicted sigma_14 and tau_14 against the
              measured ones, and the detectors at K = 14 beyond the real data's (the section-5.3 survival rule)
  calib       K = 14 calibration on the ring-free plants and the zero each estimator implies (section 5.2's table)
  recomplete  the section-5.3 recipes glued at sigma_14 (tail from k = 15) and the Borel-Pade sums [m/(14-m)],
              m = 3..7: the zero, the type rule, the zero counts
  project     the README's order-14 projection (est_k = w* + c r^k through k = 12, 13) against the real arg_14, with
              back-tests on the real data and on the arc plants, and the spread of arg_14 under sigma_err noise
  posthoc     added after the runs above (section 6.7): the order-to-order change of arg and G, real against plants;
              the local minima of G's kbar objective and G at each; the K = 13 Borel-Pade sums [5/8], [6/7] against
              sigma_14; the poles and residues of the K = 14 sums [5/9], [6/8]
  table       the stored checks of section 6, as JSON

    nice -n 10 python3 planted_k14.py STAGE [--scratch DIR] [--mutate=...]
"""
import argparse
import glob
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted_k13 as K13  # noqa: E402
import planted_plants as PP  # noqa: E402
import planted_post as PO  # noqa: E402
import planted_run as R  # noqa: E402
from arg_crossing import taus  # noqa: E402

SCAN14 = os.path.join(HERE, "scan29J1000")
SCAN13 = K13.SCAN13
REAL13 = K13.REAL13
REAL14 = "N0 scan29J1000"
OUT_OLD = R.OUT
OUT14 = os.path.join(HERE, "planted_results_k14.jsonl")
CHECKS14 = os.path.join(HERE, "planted_checks_k14.json")
KS14 = (12, 13, 14)
EST = K13.EST
ARC_FAMS = ("P-ARC", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2")
SHAPES = ("arc", "chord", "mirror", "beta2")
HYPOTHESES = (1.0, 0.99998, 0.99995, 0.99992)       # README / plot_selection_zoom.py: zero at 1, or in the K = 13 band
REPRO_TOL = 1e-9
FAILED = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILED.append(msg)
    return ok


def use14():
    """point planted_run at the K = 14 record file"""
    R.OUT, R.KS = OUT14, KS14
    R.DONE.clear()
    R.DONE.update(R.done_names())


def load_checks():
    return json.load(open(CHECKS14)) if os.path.exists(CHECKS14) else {}


def save_checks(key, val):
    c = load_checks()
    c[key] = val
    with open(CHECKS14, "w") as f:
        json.dump(c, f, default=R.jdefault, indent=1)


def raw_scan(scandir, pattern="*.json"):
    return {r["wf"]: r for r in (json.load(open(fn)) for fn in sorted(glob.glob(os.path.join(scandir, pattern))))}


def sigmas(r):
    return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]]


def load_real14():
    """{wf: (sigma_0..14, sigma_err)} from the 960-bit files (the directory also holds the 1280-bit pilot)"""
    mp.mp.dps = PP.DPS
    return {float(w): (sigmas(r), list(r["sigma_err"])) for w, r in raw_scan(SCAN14, "*_p960.json").items()}


def old_records():
    return {r["name"]: r for r in (json.loads(line) for line in open(OUT_OLD) if line.strip())}


def new_records():
    use14()
    return {r["name"]: r for r in R.records()}


def est_vector(rec, K):
    """every number the grading and the detectors read at order K, flattened, for the reproduction check"""
    e = rec["est"][str(K)]
    out = {f"arg{k}": v for k, v in e["arg"].items()}
    out.update({f"ratio{k}": v for k, v in e["ratio"].items()})
    out.update({f"pade{c}": v for c, v in zip(e["pade_central"], e["pade_central_vals"])})
    out.update({"G": e.get("G"), "G_prev": e.get("G_prev"), "pade_mid": e["pade_central_mid"]})
    return out


def max_diff(a, b):
    keys = set(a) | set(b)
    worst = 0.0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if (x is None) != (y is None):
            return float("inf"), k
        if x is not None:
            d = abs(x - y)
            if d > worst:
                worst, wk = d, k
    return worst, (wk if worst > 0 else None)


# ---------------------------------------------------------------- converge
def stage_converge(mutate=None):
    mp.mp.dps = 50
    p960 = raw_scan(SCAN14, "*_p960.json")
    p1280 = raw_scan(SCAN14, "*_p1280.json")
    old = raw_scan(SCAN13)
    print(f"files: 960 bits at wf {sorted(p960)}; 1280-bit pilot at wf {sorted(p1280)}; scan27J1000p960 at wf {sorted(old)}")
    for w, r in sorted(p960.items()):
        print(f"  wf {w}: K {r['K']} J {r['J']} prec {r['prec']} fit {r['fit']} mf {r['mf']} phys_prec {r['phys_prec']}, "
              f"{len(r['sigma_str'])} coefficients")
    # mutations: each breaks the input of one check
    if mutate == "err":
        p960[1.0]["sigma_err"][14] *= 1e14
    elif mutate == "low":
        a = mp.mpf(p960[0.99]["sigma_str"][13][0]) * (1 + mp.mpf("1e-15"))
        p960[0.99]["sigma_str"][13][0] = mp.nstr(a, 20)
    elif mutate == "pilot":
        b = mp.mpf(p1280[1.0]["sigma_str"][14][1]) * (1 + mp.mpf("1e-15"))
        p1280[1.0]["sigma_str"][14][1] = mp.nstr(b, 20)
    elif mutate == "ratiolaw":
        p960[1.01]["sigma_str"][14] = [mp.nstr(mp.mpf(x) * mp.mpf("1.1"), 20) for x in p960[1.01]["sigma_str"][14]]
    elif mutate:
        raise SystemExit(f"unknown mutation {mutate}")
    out = {"nodes": {}}
    print("(a) err14 against the 1e-10 relative requirement (section 3.6):")
    for w, r in sorted(p960.items()):
        s = sigmas(r)
        rel = r["sigma_err"][14] / float(abs(s[14]))
        out["nodes"][str(w)] = {"sigma14": r["sigma_str"][14], "err14": r["sigma_err"][14], "err14_rel": rel,
                                "err13": r["sigma_err"][13], "err12": r["sigma_err"][12]}
        check(rel <= 1e-10, f"wf {w}: sigma_14 = {mp.nstr(s[14], 20)}, err14 {r['sigma_err'][14]:.3g} "
                            f"({rel:.2e} relative)")
    print("(b) sigma_0..13 against scan27J1000p960:")
    for w, r in sorted(p960.items()):
        same = [r["sigma_str"][k] == old[w]["sigma_str"][k] for k in range(14)]
        erre = [r["sigma_err"][k] == old[w]["sigma_err"][k] for k in range(14)]
        out["nodes"][str(w)]["low_bitwise"] = all(same)
        check(all(same), f"wf {w}: sigma_0..13 bitwise equal ({sum(same)}/14); sigma_err_0..13 equal "
                         f"({sum(erre)}/14)")
    print("(c) the 1280-bit pilot at wf 1.00 against 960 bits:")
    a, b = p960[1.0], p1280[1.0]
    eq = [a["sigma_str"][k] == b["sigma_str"][k] for k in range(1, 15)]
    s0 = a["sigma_str"][0][0] == b["sigma_str"][0][0] and abs(float(a["sigma_str"][0][1])) < 1e-280 \
        and abs(float(b["sigma_str"][0][1])) < 1e-280
    out["pilot"] = {"equal_1_14": all(eq), "sigma0_ok": s0, "pilot_sigma_err": b["sigma_err"][:3],
                    "pilot_fit_1_1": b["fit_info"]["1_1"], "p960_fit_1_1": a["fit_info"]["1_1"],
                    "S1_fits_equal_k3_29": all(a["fit_info"][k][:2] == b["fit_info"][k][:2] for k in a["fit_info"]
                                               if k.endswith("_1") and k != "1_1"),
                    "other_fits_differing_largest_err": max(max(a["fit_info"][k][1], b["fit_info"][k][1])
                                                            for k in a["fit_info"] if not k.endswith("_1")
                                                            and a["fit_info"][k][:2] != b["fit_info"][k][:2])}
    check(all(eq) and s0, f"sigma_1..14 equal in every printed digit ({sum(eq)}/14); sigma_0 = 1 + round-off "
                          f"({a['sigma_str'][0][1]} / {b['sigma_str'][0][1]})")
    print(f"  note: the pilot's own sigma_err is {b['sigma_err'][14]} at every k: its k = 1 fit has residual exactly 0 at "
          f"every Mf, so pick_plateau finds no falling residual and returns inf (fit_info 1_1 {b['fit_info']['1_1']}; "
          f"960 bits: {a['fit_info']['1_1']}). The multiplier fits S_1^(k), k = 3..29, choose the same Mf with the same "
          f"plateau step at both precisions: {out['pilot']['S1_fits_equal_k3_29']}; the higher-harmonic fits that differ "
          f"have plateau steps of at most {out['pilot']['other_fits_differing_largest_err']:.1e}")
    print("(d) the ratio law through sigma_14: second difference of sigma_k/sigma_{k+1}, k = 7..12 (D-ratiolaw <= 0.05):")
    out["ratiolaw_d2"] = {}
    for w, r in sorted(p960.items()):
        s = sigmas(r)
        rat = [complex(s[k] / s[k + 1]) for k in range(14)]
        d2 = [abs(rat[k + 1] - 2 * rat[k] + rat[k - 1]) for k in range(7, 13)]
        out["ratiolaw_d2"][str(w)] = d2
        check(d2[-1] <= 0.05, f"wf {w}: |d2| k = 7..12: " + " ".join(f"{x:.4f}" for x in d2)
              + f"; sigma_13/sigma_14 = {rat[13]:.4f}")
    print("spread in wf (context): the change of sigma_k across the nodes, against its error")
    out["spread"] = {}
    for k in (12, 13, 14):
        s = {w: sigmas(r)[k] for w, r in p960.items()}
        d1 = abs(s[1.01] - s[0.99]) / 2
        d2 = abs(s[1.01] - 2 * s[1.0] + s[0.99])
        e = max(r["sigma_err"][k] for r in p960.values())
        out["spread"][str(k)] = {"half_range_rel": float(d1 / abs(s[1.0])), "d2_over_d1": float(d2 / d1),
                                 "err_over_d1": float(e / d1)}
        print(f"  k = {k}: |sigma_k(1.01) - sigma_k(0.99)|/2 = {float(d1 / abs(s[1.0])):.4f} |sigma_k(1.00)|; "
              f"second difference / first = {float(d2 / d1):.4f}; max err_k / first difference = {float(e / d1):.1e}")
    if not mutate:
        save_checks("K14 converge", out)
    return out


# ---------------------------------------------------------------- plants
def refit_nodes(wt):
    """stage_refit of planted_run, without writing the checks file"""
    real = PP.load_real()
    mp.mp.dps = PP.DPS
    nodes, resid = {}, {}
    for wf in PP.NODES:
        arc = PP.Arc(wf, "arc")
        lap = PP.Laplace(arc, PP.mpc(PP.BETA[wf]) - 1, 6)
        xr = PP.x0_planted(wf, wt)
        r = lap.fit(real[wf][0], PP.factor([xr]))
        resid[str(wf)] = max(r)
        nodes[wf] = lap.coeffs(PP.factor([xr]), normalise=True)
    return nodes, resid


def sigma_dev(nodes, real14):
    """relative distance of a plant's sigma_13, sigma_14 from the measured ones, per node"""
    return {str(w): [float(abs(nodes[w][k] - real14[w][0][k]) / abs(real14[w][0][k])) for k in (13, 14)]
            for w in PP.NODES}


def stage_plants(scratch):
    use14()
    real14 = load_real14()
    devs = load_checks().get("K14 plant sigma13-14 deviation", {})
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
            devs[name] = sigma_dev(nodes, real14)
            save_checks("K14 plant sigma13-14 deviation", devs)
            R.run_plant(name, tag.split()[0], nodes, float(wt), scratch,
                        info={"fit_resid_max": {str(w): fam.resid[w] and max(fam.resid[w]) for w in PP.NODES}})
    if "C-EXP" not in R.DONE:
        nodes, _, info, truth = PP.completion("C-EXP")
        devs["C-EXP"] = sigma_dev(nodes, real14)
        save_checks("K14 plant sigma13-14 deviation", devs)
        R.run_plant("C-EXP", "C", nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    for wt in R.REFIT_WT:
        name = f"REFIT P-ARC wt={wt}"
        if name in R.DONE:
            continue
        nodes, resid = refit_nodes(wt)
        devs[name] = sigma_dev(nodes, real14)
        save_checks("K14 plant sigma13-14 deviation", devs)
        R.run_plant(name, "REFIT", nodes, float(wt), scratch, info={"fit_resid_max": resid}, linear=False)
    plants_check()


def plants_check():
    old, new = old_records(), new_records()
    names = [n for n in new if not n.startswith(("N0", "C13", "C14"))]
    print(f"\n{len(names)} plants with K = 14 records")
    worst = (0.0, None)
    for n in names:
        for K in (12, 13):
            d, k = max_diff(est_vector(new[n], K), est_vector(old[n], K))
            if d > worst[0]:
                worst = (d, f"{n} K{K} {k}")
    check(worst[0] <= REPRO_TOL, f"regenerated plants reproduce the stored K = 12, 13 estimates: largest difference "
                                 f"{worst[0]:.1e} ({worst[1]})")
    n1 = [new[n] for n in names if n.startswith("N1")]
    errs = [abs(R.grade(r, 14)["est"][e] - r["truth"]) for r in n1 for e in ("arg", "ratio", "pade", "G")
            if R.grade(r, 14)["est"][e] is not None]
    nnone = sum(R.grade(r, 14)["est"][e] is None for r in n1 for e in ("arg", "ratio", "pade", "G"))
    check(len(n1) == 7 and nnone == 0 and max(errs) <= 1e-6,
          f"N1 at K = 14: {len(n1)} lone zeros, worst error {max(errs):.1e} over arg, ratio, Pade, G ({nnone} missing)")
    ctl = [abs(R.grade(r, 14)["est"]["arg"] - (r["truth"] + 0.002)) <= 1e-6 for r in n1]
    check(not any(ctl), "control: N1 graded against w_t + 0.002 fails (as it must)")
    n2 = [new[n] for n in names if n.startswith("N2")]
    printed = [(r["name"], e) for r in n2 for e, v in R.grade(r, 14)["est"].items() if v is not None]
    check(len(n2) == 2 and not printed, f"N2 at K = 14: no estimator prints a number ({printed or 'none'})")
    fams = {}
    for n in names:
        fams.setdefault(new[n]["family"], []).append(n)
    print("  records per family:", {f: len(v) for f, v in fams.items()})
    save_checks("K14 plants check", {"repro_worst": worst, "n1_worst": max(errs), "n2_printed": printed,
                                     "families": {f: len(v) for f, v in fams.items()}})


# ---------------------------------------------------------------- real data
def stage_real(scratch):
    use14()
    real = load_real14()
    R.run_plant(REAL14, "N0", {wf: real[wf][0] for wf in PP.NODES}, None, scratch)
    flip = {wf: list(real[wf][0]) for wf in PP.NODES}
    flip[1.0][14] = -flip[1.0][14]
    R.run_plant(f"{REAL14} X1 flip sigma_14", "N0 mutation", flip, None, scratch)
    sc = {wf: list(real[wf][0]) for wf in PP.NODES}
    sc[1.0][14] = sc[1.0][14] * (1 + mp.mpf("1e-3"))
    R.run_plant(f"{REAL14} X2 scale sigma_14", "N0 mutation", sc, None, scratch)
    old, new = old_records(), new_records()
    rec = new[REAL14]
    for K in (12, 13):
        d, k = max_diff(est_vector(rec, K), est_vector(old[REAL13], K))
        check(d <= REPRO_TOL, f"K = {K}: the scan29J1000 estimates equal those of scan27J1000p960 (largest difference "
                              f"{d:.1e}{', ' + k if k else ''})")
    for K in KS14:
        g = R.grade(rec, K)["est"]
        print(f"  K = {K}: " + ", ".join(f"{e} {v:.6f}" if v is not None else f"{e} None" for e, v in g.items())
              + f"; detectors {R.detectors(rec, K)}")
    e = rec["est"]["14"]
    print("  arg crossings k = 4..14:", {int(k): round(v, 7) for k, v in e["arg"].items()})
    print("  ratio estimates k = 9..14:", {int(k): round(v, 6) for k, v in e["ratio"].items()})
    print("  Pade central", e["pade_central"], "values", [round(v, 6) if v else v for v in e["pade_central_vals"]],
          "range", e["pade_central_range"], "mid", e["pade_central_mid"])
    print("  all Pade crossings K = 14:", {k: round(v, 6) if v else v for k, v in e["pade"].items()})
    print("  G K = 9..14 via G_prev/G:", e.get("G_prev"), e.get("G"), "last terms", e.get("G_last_prev"), e.get("G_last"),
          "kbar", e.get("G_kbar"))
    print("  linear (arg_crossing.py):", e.get("linear"))
    det = set(R.detectors(rec, 14))
    for n in (f"{REAL14} X1 flip sigma_14", f"{REAL14} X2 scale sigma_14"):
        m = new[n]
        extra = sorted(set(R.detectors(m, 14)) - det)
        g, g0 = R.grade(m, 14)["est"], R.grade(rec, 14)["est"]
        shift = max((abs(g[x] - g0[x]) if None not in (g[x], g0[x]) else float("inf")) for x in EST)
        check(bool(extra) if "flip" in n else True, f"mutation {n.split()[-2]} sigma_14: new detectors {extra or 'none'}, "
                                                     f"largest shift {shift:.1e}")
    save_checks("K14 real", {"est": {str(K): R.grade(rec, K)["est"] for K in KS14},
                             "detectors": {str(K): R.detectors(rec, K) for K in KS14}})


# ---------------------------------------------------------------- ring test
def c13_completions():
    """the section-5.3 re-glued completions (tail from k = 14), regenerated: {name: (nodes, funcs, truth)}"""
    real13 = PP.load_real(SCAN13)
    stored = json.load(open(R.CHECKS))["K13 recomplete"]
    out = {}
    for kind in ("C-W1", "C-W2", "C-W3", "C-EXP"):
        nodes, funcs, info, truth = PP.completion(kind, real=real13, kc=13)
        out[kind.replace("C-", "C13-")] = (nodes, funcs, truth, info)
    for kind in SHAPES:
        name = f"C13-ARC {kind}"
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{stored[name]['best_wt']:.5f}", kc=13, real=real13, funcs=funcs)
        out[name] = (nodes, funcs, truth, info)
    for name, (_, _, truth, _) in out.items():
        check(abs(truth["im_x"] - stored[name]["truth"]) <= REPRO_TOL,
              f"{name} regenerated: zero at {truth['im_x']:.9f} (stored {stored[name]['truth']:.9f})")
    return out, stored


def simple_zero_rel(rt, k):
    """relative error of the simple-zero law's tau_k, from tau_{k-2}, tau_{k-1}, per node"""
    out = []
    for wf in PP.NODES:
        t = rt[wf]
        x0 = (k - 2) * t[k - 2] / ((k - 1) * t[k - 1])
        p = t[k - 1] * (k - 1) / (k * x0)
        out.append(float(abs(p - t[k]) / abs(t[k])))
    return out


def stage_ring(scratch):
    use14()
    real = load_real14()
    mp.mp.dps = 50
    rt = {wf: taus(list(real[wf][0][:15])) for wf in PP.NODES}
    comps, stored = c13_completions()
    for name, (nodes, _, truth, _) in comps.items():
        R.run_plant(name, "C13 (ring test)", nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})
    new = new_records()
    real_rec = new[REAL14]
    real_det = set(R.detectors(real_rec, 14))
    a14 = R.grade(real_rec, 14)["est"]["arg"]
    print(f"real: sigma_14 / tau_14 at the nodes: " + "; ".join(
        f"{wf}: {mp.nstr(real[wf][0][14], 8)} / {mp.nstr(rt[wf][14], 8)}" for wf in PP.NODES))
    print(f"real arg_14 = {a14:.7f}; real K = 14 detectors: {sorted(real_det)}\n")
    out = {"simple-zero law from tau_12, tau_13": {"tau14_rel": simple_zero_rel(rt, 14)},
           "simple-zero law from tau_11, tau_12 (tau_13, for comparison)": {"tau13_rel": simple_zero_rel(rt, 13)}}
    print(f"| completion (K = 13) | sigma_14 rel. error (0.99 / 1.00 / 1.01) | tau_14 rel. error | its arg_14 | "
          f"K = 14 detectors beyond the real data's | survives |")
    print("|---|---|---|---|---|---|")
    print(f"| simple-zero law from tau_12, tau_13 | - | {' / '.join(f'{x:.1e}' for x in simple_zero_rel(rt, 14))} | - | - | - |")
    for name, (nodes, _, truth, _) in comps.items():
        mp.mp.dps = 50
        se = [float(abs(nodes[wf][14] - real[wf][0][14]) / abs(real[wf][0][14])) for wf in PP.NODES]
        te = [float(abs(taus(list(nodes[wf][:15]))[14] - rt[wf][14]) / abs(rt[wf][14])) for wf in PP.NODES]
        rec = new[name]
        arg = R.grade(rec, 14)["est"]["arg"]
        extra = sorted(set(R.detectors(rec, 14)) - real_det)
        k13 = R.grade(rec, 13)["est"]
        same13 = max_diff(est_vector(rec, 13), est_vector(real_rec, 13))[0]
        out[name] = {"sigma14_rel": se, "tau14_rel": te, "arg14": arg, "extra_detectors": extra, "survives": not extra,
                     "truth": truth["im_x"], "zeros_lt_7.3": stored[name]["zeros_lt_7.3"],
                     "zeros_lt_12": stored[name]["zeros_lt_12"], "K13_equals_real": same13,
                     "est14": R.grade(rec, 14)["est"], "arg13": k13["arg"]}
        print(f"| {name} (zero {truth['im_x']:.6f}; {stored[name]['zeros_lt_7.3']} / {stored[name]['zeros_lt_12']}) | "
              f"{' / '.join(f'{x:.1e}' for x in se)} | {' / '.join(f'{x:.1e}' for x in te)} | "
              f"{'none' if arg is None else f'{arg:.6f}'} | {', '.join(extra) or '-'} | {'yes' if not extra else 'no'} |")
        check(same13 <= REPRO_TOL, f"{name}: its K = 13 estimates are the real ones (difference {same13:.1e})")
    save_checks("K14 ring test", out)


# ---------------------------------------------------------------- calibration
def ring_free14(by):
    out = {}
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            tag = "P-ARC" if r["family"] == "P-ARC" else r["name"].split(" wt=")[0].replace("P-ARC-B ", "")
            out.setdefault(tag, []).append(r)
    out["C-EXP"] = [by["C-EXP"]]
    return out


def calib_rows(by, real_est, K):
    fams = ring_free14(by)
    refit_res = json.load(open(R.CHECKS))["REFIT"]
    refit = [r for r in by.values() if r["family"] == "REFIT"
             and max(refit_res[r["name"].split("wt=")[1]].values()) <= 1e-4]
    fams["REFIT (resid <= 1e-4)"] = refit
    rows = {}
    for fam, rs in fams.items():
        cells = {}
        for e in EST:
            v = [R.grade(r, K)["err"][e] for r in rs]
            v = [x for x in v if x is not None]
            if v and real_est[e] is not None:
                cells[e] = {"lo": min(v), "hi": max(v), "n": len(v), "zero": (real_est[e] - max(v), real_est[e] - min(v))}
            else:
                cells[e] = {"n": len(v)}
        rows[fam] = {"n": len(rs), "cells": cells}
    arc = [r for f, rs in fams.items() if f in ("P-ARC", "chord", "mirror", "beta2") for r in rs]
    cells = {}
    for e in EST:
        v = [R.grade(r, K)["err"][e] for r in arc if R.grade(r, K)["err"][e] is not None]
        cells[e] = {"lo": min(v), "hi": max(v), "n": len(v), "zero": (real_est[e] - max(v), real_est[e] - min(v))} \
            if v and real_est[e] is not None else {"n": len(v)}
    rows["all 28 arc plants"] = {"n": len(arc), "cells": cells}
    return rows


def stage_calib():
    by = new_records()
    real = by[REAL14]
    for K in (13, 14):
        print(f"Real data, K = {K}: " + ", ".join(f"{e} {R.grade(real, K)['est'][e]:.7f}" for e in EST))
        print(f"  detectors {R.detectors(real, K)}")
    out = {}
    for K in (13, 14):
        e = R.grade(real, K)["est"]
        rows = calib_rows(by, e, K)
        out[str(K)] = rows
        print(f"\nK = {K}: | plants | n | " + " | ".join(f"{x}: error x1e4 -> zero" for x in EST) + " |")
        for fam, row in rows.items():
            cells = []
            for x in EST:
                c = row["cells"][x]
                cells.append("-" if "lo" not in c else f"{c['lo'] * 1e4:+.2f} ... {c['hi'] * 1e4:+.2f} -> "
                             f"{c['zero'][0]:.6f}-{c['zero'][1]:.6f}" + (f" ({row['n'] - c['n']} none)" if c['n'] < row['n'] else ""))
            print(f"| {fam} | {row['n']} | " + " | ".join(cells) + " |")
    # check against section 5.2 (K = 13, stored records): the same calibration from the K = 14 records
    a = out["13"]["all 28 arc plants"]["cells"]
    check(abs(a["arg"]["zero"][0] - 0.999916) < 6e-7 and abs(a["arg"]["zero"][1] - 0.999992) < 6e-7
          and abs(a["G"]["zero"][0] - 0.999920) < 6e-7 and abs(a["G"]["zero"][1] - 0.999964) < 6e-7,
          f"K = 13 calibration from the new records reproduces section 5.2 (arg {a['arg']['zero'][0]:.6f}-"
          f"{a['arg']['zero'][1]:.6f}, G {a['G']['zero'][0]:.6f}-{a['G']['zero'][1]:.6f})")
    # per-plant errors at K = 13 and 14, for the arc plants (the trend of each plant's error)
    tr = {}
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            tr[r["name"]] = {x: [R.grade(r, K)["err"][x] for K in (12, 13, 14)] for x in ("arg", "G", "pade")}
    out["per_plant_err_12_13_14"] = tr
    devs = load_checks().get("K14 plant sigma13-14 deviation", {})
    out["sigma_dev_summary"] = {}
    for f in ("P-ARC wt", "P-ARC-B chord", "P-ARC-B mirror", "P-ARC-B beta2", "C-EXP", "REFIT"):
        ns = [n for n in devs if n.startswith(f)]
        if ns:
            d13 = [max(devs[n][w][0] for w in devs[n]) for n in ns]
            d14 = [max(devs[n][w][1] for w in devs[n]) for n in ns]
            out["sigma_dev_summary"][f] = {"sigma13": [min(d13), max(d13)], "sigma14": [min(d14), max(d14)]}
            print(f"  {f}: plant sigma_13 vs measured {min(d13):.1e}-{max(d13):.1e}, sigma_14 {min(d14):.1e}-{max(d14):.1e}"
                  " (relative, worst node)")
    save_checks("K14 calib", out)


# ---------------------------------------------------------------- next generation and ring-free completions
def truth_of(funcs, start="7.40"):
    xr = {}
    for wf in PP.NODES:
        xr[wf] = mp.findroot(funcs[wf], mp.mpc(start, "0"), tol=mp.mpf(10) ** -40, maxsteps=200)
    truth = {}
    for lab, fn in (("im_x", lambda z: mp.im(z)), ("im_sqrt", lambda z: mp.im(mp.sqrt(z)))):
        c = np.polyfit(PP.NODES, [float(fn(xr[w])) for w in PP.NODES], 2)
        rts = [r.real for r in np.roots(c) if abs(r.imag) < 1e-12 and 0.95 < r.real < 1.05]
        truth[lab] = min(rts, key=lambda r: abs(r - 1)) if rts else None
    return truth, xr


def borel_pade(real, m, n):
    """the exponential sum of the [m/n] Pade approximant (m <= n) of a_k = k! sigma_k, k <= m + n (planted_plants
    .completion's C-EXP recipe with general m, n; a constant term when m = n)"""
    mp.mp.dps = PP.DPS
    funcs, info, nodes = {}, {}, {}
    for wf in PP.NODES:
        sig = list(real[wf][0][:m + n + 1])
        a = [sig[k] * mp.factorial(k) for k in range(m + n + 1)]
        p, q = mp.pade(a, m, n)
        us = mp.polyroots(q[::-1], maxsteps=400, extraprec=400)
        dq = [k * q[k] for k in range(1, len(q))]
        ts, rs = [], []
        for u in us:
            ts.append(1 / u)
            rs.append(-mp.polyval(p[::-1], u) / (u * mp.polyval(dq[::-1], u)))
        c0 = p[m] / q[n] if m == n else mp.mpc(0)
        funcs[wf] = (lambda ts, rs, c0: (lambda x: c0 + mp.fsum(r * mp.exp(t * x) for t, r in zip(ts, rs))))(ts, rs, c0)
        coeffs = [(c0 if k == 0 else 0) + mp.fsum(r * t ** k for t, r in zip(ts, rs)) / mp.factorial(k)
                  for k in range(PP.KMAX + 1)]
        nodes[wf] = coeffs
        info[str(wf)] = {"poles": [[float(mp.re(t)), float(mp.im(t))] for t in ts],
                         "max_pole": max(float(abs(t)) for t in ts),
                         "min_residue": min(float(abs(r)) for r in rs),
                         "reproduce_max_rel": max(float(abs(coeffs[k] - sig[k]) / abs(sig[k])) for k in range(m + n + 1))}
    return nodes, funcs, info


def stage_recomplete():
    real = load_real14()
    by = new_records()
    e14 = R.grade(by[REAL14], 14)["est"]
    res = load_checks().get("K14 recomplete", {})
    wts = [f"{0.9990 + 0.0001 * i:.4f}" for i in range(21)]

    def finish(name, funcs, truth, extra, radii=(7.3, 12)):
        f = funcs[1.0]
        counts = {f"zeros_lt_{r}": K13.count(f, mp.mpf(str(r))) for r in radii}
        rec = {"truth": truth["im_x"], "truth_im_sqrt": truth["im_sqrt"], **counts,
               "err_of_real_estimates": {e: e14[e] - truth["im_x"] for e in EST if e14[e] is not None}, **extra}
        res[name] = rec
        save_checks("K14 recomplete", res)
        print(f"{name}: zero at {truth['im_x']:.7f} (Im sqrt x: {truth['im_sqrt']:.7f}); {counts}; "
              f"{ {k: v for k, v in extra.items() if k not in ('x_R', 'poles', 'fit_resid_max')} }", flush=True)

    for kind in ("C-W1", "C-W2", "C-W3"):
        name = kind.replace("C-", "C14-")
        if name in res:
            continue
        nodes, funcs, info, truth = PP.completion(kind, real=real, kc=14)
        finish(name, funcs, truth, {"kappa_beta": {str(w): info[w] for w in PP.NODES}})
    for kind in SHAPES:
        name = f"C14-ARC {kind}"
        if name in res:
            continue
        scan = PO.refit_scan(kind, wts, kc=14, real=real)
        bw = PO.best_wt(scan)
        funcs = {}
        nodes, info, truth = PO.arc_completion(kind, f"{bw:.5f}", kc=14, real=real, funcs=funcs)
        finish(name, funcs, truth, {"best_wt": bw, "L2_floor": min(v[0] for v in scan.values()),
                                    "fit_resid_max": {w: info[w]["fit_resid_max"] for w in info}})
    for m in (3, 4, 5, 6, 7):
        n = 14 - m
        name = f"C14-BP [{m}/{n}]"
        if name in res:
            continue
        try:
            nodes, funcs, info = borel_pade(real, m, n)
            truth, xr = truth_of(funcs)
        except Exception as ex:                                  # noqa: BLE001  (recorded, not swallowed)
            res[name] = {"error": f"{type(ex).__name__}: {ex}"}
            save_checks("K14 recomplete", res)
            print(f"{name}: {res[name]['error']}")
            continue
        maxp = {w: info[w]["max_pole"] for w in info}
        finish(name, funcs, truth, {"max_pole": maxp, "type_rule": all(v < 2 for v in maxp.values()),
                                    "reproduce_max_rel": max(info[w]["reproduce_max_rel"] for w in info),
                                    "min_residue": {w: info[w]["min_residue"] for w in info},
                                    "x_R": {str(w): [float(z.real), float(z.imag)] for w, z in xr.items()},
                                    "poles_1.0": info["1.0"]["poles"]}, radii=(7.3, 12, 16))
    # the same Borel-Pade family at K = 13, to see whether [6/7] was the only one that passed the type rule
    real13 = PP.load_real(SCAN13)
    for m in (3, 4, 5, 6):
        n = 13 - m
        name = f"C13-BP [{m}/{n}]"
        if name in res:
            continue
        try:
            nodes, funcs, info = borel_pade(real13, m, n)
            truth, xr = truth_of(funcs)
        except Exception as ex:                                  # noqa: BLE001
            res[name] = {"error": f"{type(ex).__name__}: {ex}"}
            save_checks("K14 recomplete", res)
            continue
        maxp = {w: info[w]["max_pole"] for w in info}
        f = funcs[1.0]
        res[name] = {"truth": truth["im_x"], "max_pole": maxp, "type_rule": all(v < 2 for v in maxp.values()),
                     "zeros_lt_12": K13.count(f, mp.mpf(12))}
        save_checks("K14 recomplete", res)
        print(f"{name}: zero at {truth['im_x']:.7f}; type rule {res[name]['type_rule']} (max |t| "
              f"{max(maxp.values()):.3f}); zeros |x| < 12: {res[name]['zeros_lt_12']}", flush=True)
    st = json.load(open(R.CHECKS))["K13 recomplete"]["C13-EXP"]["truth"]
    check(abs(res["C13-BP [6/7]"]["truth"] - st) <= REPRO_TOL,
          f"borel_pade reproduces C13-EXP [6/7]: {res['C13-BP [6/7]']['truth']:.9f} (stored {st:.9f})")


# ---------------------------------------------------------------- order-14 projection
def project(a, k, w):
    """est_k from est_{k-2}, est_{k-1} if est_j = w + c r^j"""
    d1, d2 = a[k - 2] - w, a[k - 1] - w
    return w + d2 * d2 / d1


def aitken(a, k):
    x0, x1, x2 = a[k - 2], a[k - 1], a[k]
    return x2 - (x2 - x1) ** 2 / ((x2 - x1) - (x1 - x0))


def stage_project():
    by = new_records()
    real = by[REAL14]
    a = {int(k): v for k, v in real["est"]["14"]["arg"].items()}
    a14 = a[14]
    print(f"real arg crossings k = 10..14: " + ", ".join(f"{a[k]:.7f}" for k in range(10, 15)))
    out = {"arg": {k: a[k] for k in range(4, 15)}, "hyp": {}}
    print(f"real arg_14 - 1 = {a14 - 1:+.3e}")
    for w in HYPOTHESES:
        p = project(a, 14, w)
        back = {k: a[k] - project(a, k, w) for k in range(8, 14)}
        out["hyp"][str(w)] = {"pred14": p, "miss": a14 - p, "back": back}
        print(f"  zero at {w}: predicted arg_14 - 1 = {p - 1:+.3e}; real - predicted = {a14 - p:+.2e}; "
              f"back-test (real - predicted) at k = 11, 12, 13: "
              + ", ".join(f"{back[k]:+.1e}" for k in (11, 12, 13)))
    ait = {k: aitken(a, k) for k in range(6, 15)}
    out["aitken"] = ait
    print("  Aitken (3-point geometric limit) on k-2..k, k = 6..14: " + ", ".join(f"{ait[k]:.6f}" for k in range(6, 15)))
    # plants: the model through 12, 13 with the true w_t, predicting 14; and Aitken on 12..14
    pl = []
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            b = {int(k): v for k, v in r["est"]["14"]["arg"].items()}
            if None in (b.get(12), b.get(13), b.get(14)):
                continue
            pl.append({"name": r["name"], "miss": b[14] - project(b, 14, r["truth"]),
                       "aitken_err": aitken(b, 14) - r["truth"], "steps": [b[12] - r["truth"], b[13] - r["truth"],
                                                                         b[14] - r["truth"]]})
    out["plants"] = pl
    miss = [p["miss"] for p in pl]
    ae = [p["aitken_err"] for p in pl]
    print(f"  arc plants ({len(pl)}): model through k = 12, 13 with the true w_t misses arg_14 by "
          f"{min(miss):+.1e} ... {max(miss):+.1e}; Aitken(12, 13, 14) - w_t {min(ae):+.1e} ... {max(ae):+.1e}")
    # which hypothesis, and how decisive (section 6.0, item 8)
    p1, pb = out["hyp"]["1.0"]["pred14"], [out["hyp"][str(w)]["pred14"] for w in HYPOTHESES[1:]]
    d1 = abs(a14 - p1)
    db = min(abs(a14 - p) for p in pb) if not (min(pb) <= a14 <= max(pb)) else 0.0
    fav = "zero at 1" if d1 < db else "zero in the K = 13 band"
    back_real = {str(w): max(abs(out["hyp"][str(w)]["back"][k]) for k in (11, 12, 13)) for w in HYPOTHESES}
    model_err = max(max(back_real.values()), max(abs(x) for x in miss))
    out.update({"favoured": fav, "dist_to_1": d1, "dist_to_band": db, "model_err": model_err,
                "back_real_max": back_real, "plant_miss_max": max(abs(x) for x in miss)})
    other = db if fav == "zero at 1" else d1
    out["decisive"] = other > 3 * model_err
    print(f"  favoured: {fav} (|arg_14 - prediction|: zero at 1 {d1:.1e}, band {db:.1e}); model error (largest back-test "
          f"miss, real k = 11-13 and plants k = 14) {model_err:.1e}; decisive (other > 3x model error): {out['decisive']}")
    # arg_14 under the coefficients' own errors: sigma_err noise, 20 seeds
    rl = load_real14()
    rng = np.random.default_rng(14)
    sh = []
    mp.mp.dps = 50
    for seed in range(20):
        rows = []
        for wf in PP.NODES:
            sig = list(rl[wf][0])
            for k in range(1, 15):
                n = rng.standard_normal(2) / np.sqrt(2)
                sig[k] = sig[k] + mp.mpf(rl[wf][1][k]) * mp.mpc(n[0], n[1])
            rows.append((wf, taus(sig)))
        vals = [float(mp.arg(t[14]) / mp.pi) % 2 - 1 for _, t in rows]
        c = np.polyfit(PP.NODES, vals, 2)
        rts = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.96 < x.real < 1.04]
        sh.append(min(rts, key=lambda x: abs(x - 1)) - a14)
    out["noise_shift"] = [min(sh), max(sh)]
    print(f"  arg_14 under sigma_err noise (20 seeds): shift {min(sh):+.1e} ... {max(sh):+.1e}")
    save_checks("K14 project", out)


# ---------------------------------------------------------------- post-hoc diagnostics (after section 6.0's runs)
def kbar_minima(sig, K, starts=(np.linspace(0.6, 1.35, 6), np.linspace(0.2, 0.8, 5))):
    """local minima of entire_cert's kbar objective (the last two terms of G_K at X0) from a grid of Nelder-Mead starts"""
    import entire_cert as EC
    from scipy.optimize import minimize
    mp.mp.dps = 60

    def f(v):
        g = EC.gcoef(sig, mp.mpc(v[0], v[1]), K)
        return float(abs(g[K]) * EC.X0 ** K + abs(g[K - 1]) * EC.X0 ** (K - 1))
    mins = []
    for a in starts[0]:
        for b in starts[1]:
            r = minimize(f, [a, b], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-15, maxiter=3000))
            if r.success and not any(abs(r.x[0] - m[0]) < 1e-3 and abs(r.x[1] - m[1]) < 1e-3 for m in mins):
                mins.append((float(r.x[0]), float(r.x[1]), float(r.fun)))
    return sorted(mins, key=lambda m: m[2])


def G_at(nodes, kb, K):
    """entire_cert.resummed's crossing for a given kbar"""
    import entire_cert as EC
    mp.mp.dps = 60
    zs = {wf: mp.findroot(EC.poly(EC.gcoef(nodes[wf], kb, K)), mp.mpc(EC.X0, 0)) for wf in PP.NODES}
    c = np.polyfit(PP.NODES, [float(mp.im(mp.sqrt(zs[w]))) for w in PP.NODES], 2)
    r = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and 0.97 < x.real < 1.03]
    return min(r, key=lambda x: abs(x - 1)) if r else None


def stage_posthoc(scratch):
    use14()
    by = new_records()
    res = load_checks().get("K14 posthoc", {})
    real = load_real14()
    rn = {w: real[w][0] for w in PP.NODES}
    # (i) order-to-order change of arg and G, 12 -> 13 -> 14
    if "steps" not in res:
        st = {}
        for n, r in by.items():
            if r["family"] in ("P-ARC", "P-ARC-B") or n in (REAL14, "C-EXP", "C13-EXP"):
                g = {K: R.grade(r, K)["est"] for K in (12, 13, 14)}
                st[n] = {e: [None if None in (g[K][e], g[K + 1][e]) else g[K + 1][e] - g[K][e] for K in (12, 13)]
                         for e in ("arg", "G", "pade")}
        res["steps"] = st
        save_checks("K14 posthoc", res)
    st = res["steps"]
    for e in ("arg", "G"):
        arcs = [v[e][1] for n, v in st.items() if n.startswith("P-ARC") and v[e][1] is not None]
        arcs12 = [v[e][0] for n, v in st.items() if n.startswith("P-ARC") and v[e][0] is not None]
        print(f"(i) {e}: change 12->13 real {st[REAL14][e][0]:+.2e}, arc plants {min(arcs12):+.2e} ... {max(arcs12):+.2e}; "
              f"13->14 real {st[REAL14][e][1]:+.2e}, arc plants {min(arcs):+.2e} ... {max(arcs):+.2e}; "
              f"C-EXP {st['C-EXP'][e][1]:+.2e}; C13-EXP {st['C13-EXP'][e][1]:+.2e}")
    # (ii) the kbar objective's local minima and G at each: real data and the four arc shapes and C-EXP (wt = 1)
    km = res.get("kbar", {})
    fams = {"real": (rn, None)}
    for tag in ARC_FAMS:
        fams[tag] = (None, 1.0)
    fams["C-EXP"] = (None, by["C-EXP"]["truth"])
    for name, (nodes, truth) in fams.items():
        for K in (12, 13, 14):
            key = f"{name} K{K}"
            if key in km:
                continue
            if nodes is None:
                if name == "C-EXP":
                    nodes = PP.completion("C-EXP")[0]
                else:
                    fam = PP.ArcFamily(name.split()[-1] if name != "P-ARC" else "arc")
                    nodes = fam.nodes("1.0000")
                fams[name] = (nodes, truth)
            mins = kbar_minima(nodes[1.0], K)
            km[key] = [{"kbar": [m[0], m[1]], "obj": m[2], "G": G_at(nodes, mp.mpc(m[0], m[1]), K)} for m in mins[:6]]
            res["kbar"] = km
            save_checks("K14 posthoc", res)
            prod = by[REAL14 if name == "real" else (f"{name} wt=1.0000" if name != "C-EXP" else "C-EXP")]["est"][str(K)]
            print(f"(ii) {key}: production kbar {prod['G_kbar'][0]:.4f}{prod['G_kbar'][1]:+.4f}i -> G {prod['G']:.6f}; minima: "
                  + "; ".join(f"{m['kbar'][0]:.4f}{m['kbar'][1]:+.4f}i obj {m['obj']:.1e} G "
                              + ("none" if m["G"] is None else f"{m['G']:.6f}") for m in km[key]), flush=True)
    # (iii) the K = 13 Borel-Pade sums that pass the type rule and are ring-free, against sigma_14
    real13 = PP.load_real(SCAN13)
    mp.mp.dps = 50
    rt = {wf: taus(list(real[wf][0][:15])) for wf in PP.NODES}
    real_det = set(R.detectors(by[REAL14], 14))
    bp = res.get("C13-BP sigma14", {})
    for m in (5, 6):
        name = f"C13-BP [{m}/{13 - m}]"
        nodes, funcs, info = borel_pade(real13, m, 13 - m)
        truth, _ = truth_of(funcs)
        R.run_plant(name, "C13-BP (post-hoc)", nodes, truth["im_x"], scratch)
        rec = new_records()[name]
        mp.mp.dps = 50
        se = [float(abs(nodes[wf][14] - real[wf][0][14]) / abs(real[wf][0][14])) for wf in PP.NODES]
        te = [float(abs(taus(list(nodes[wf][:15]))[14] - rt[wf][14]) / abs(rt[wf][14])) for wf in PP.NODES]
        extra = sorted(set(R.detectors(rec, 14)) - real_det)
        bp[name] = {"truth": truth["im_x"], "sigma14_rel": se, "tau14_rel": te, "extra_detectors": extra,
                    "est14": R.grade(rec, 14)["est"]}
        print(f"(iii) {name}: zero {truth['im_x']:.7f}; sigma_14 rel. error {' / '.join(f'{x:.1e}' for x in se)}; tau_14 "
              f"{' / '.join(f'{x:.1e}' for x in te)}; K = 14 estimates {({k: round(v, 7) if v else v for k, v in R.grade(rec, 14)['est'].items()})}; "
              f"detectors beyond the real data's: {extra or 'none'}")
    res["C13-BP sigma14"] = bp
    save_checks("K14 posthoc", res)
    # (iv) the poles and residues of the K = 14 Borel-Pade sums that pass the type rule
    rp = {}
    for m in (5, 6):
        nodes, funcs, info = borel_pade(real, m, 14 - m)
        mp.mp.dps = PP.DPS
        sig = list(real[1.0][0][:15])
        a = [sig[k] * mp.factorial(k) for k in range(15)]
        p, q = mp.pade(a, m, 14 - m)
        us = mp.polyroots(q[::-1], maxsteps=400, extraprec=400)
        dq = [k * q[k] for k in range(1, len(q))]
        rows = sorted(([complex(1 / u), complex(-mp.polyval(p[::-1], u) / (u * mp.polyval(dq[::-1], u)))] for u in us),
                      key=lambda z: abs(z[0]))
        rp[f"[{m}/{14 - m}]"] = rows
        print(f"(iv) [{m}/{14 - m}] at wf 1.00, poles t and residues r: "
              + "; ".join(f"t {t.real:.4f}{t.imag:+.4f}i |r| {abs(r):.1e}" for t, r in rows))
    res["C14-BP poles"] = rp
    save_checks("K14 posthoc", res)


# ---------------------------------------------------------------- tables
def stage_table():
    c = load_checks()
    print(json.dumps({k: (v if k != "K14 plant sigma13-14 deviation" else f"{len(v)} plants") for k, v in c.items()},
                     indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch14"))
    ap.add_argument("--mutate", default=None)
    a = ap.parse_args()
    {"converge": lambda: stage_converge(a.mutate), "plants": lambda: stage_plants(a.scratch),
     "real": lambda: stage_real(a.scratch), "ring": lambda: stage_ring(a.scratch), "calib": stage_calib,
     "recomplete": stage_recomplete, "project": stage_project, "table": stage_table,
     "posthoc": lambda: stage_posthoc(a.scratch),
     "plants-check": plants_check}[a.stage]()
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
