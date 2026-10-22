#!/usr/bin/env python3
"""Driver for the planted-truth test (notes/PLANTED_SELECTION.md). Runs every plant of section 2 through the production
estimators (planted_estimators.py) at K = 11, 12, 13 and appends one JSON record per plant to planted_results.jsonl
(resumable: a plant already in the file is skipped). `report` grades the records against the frozen criteria.

    nice -n 10 python3 planted_run.py STAGE [--scratch DIR]
        STAGE: null | parc | parcb | z2 | comp | noise | corrupt | refit | all
    python3 planted_run.py report            the tables of notes/PLANTED_SELECTION.md section 3, from the records
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402

import planted_estimators as PE  # noqa: E402
import planted_plants as PP  # noqa: E402

OUT = os.path.join(HERE, "planted_results.jsonl")
CHECKS = os.path.join(HERE, "planted_checks.json")
KS = (11, 12, 13)
TOL_BEST = 1.2e-4          # Delta_inf = 1.6439 +- 0.0002  <=>  w/w_inf within ~ +-1.2e-4
TOL_AITKEN = 2e-4
NODES = PP.NODES


def jdefault(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (mp.mpf,)):
        return float(o)
    if isinstance(o, (mp.mpc,)):
        return [float(o.real), float(o.imag)]
    raise TypeError(type(o))


def done_names():
    if not os.path.exists(OUT):
        return set()
    return {json.loads(line)["name"] for line in open(OUT) if line.strip()}


def ratio_law_d2(nodes, K):
    """max over nodes of |second difference of sigma_k/sigma_{k+1}|, k = 6..K-2 (detector D-ratiolaw)"""
    worst = 0.0
    for sig in nodes.values():
        if any(sig[k] == 0 for k in range(1, K + 1)):
            return float("inf")
        r = [complex(sig[k] / sig[k + 1]) for k in range(K)]
        for k in range(6, K - 1):
            worst = max(worst, abs(r[k + 1] - 2 * r[k] + r[k - 1]))
    return worst


def run_plant(name, family, nodes, truth, scratch, labels=None, prod_labels=None, info=None, linear=True):
    """nodes: {wf: sigma list} at the nominal nodes; labels: {wf: label written to the JSON} (grid corruptions)."""
    if name in DONE:
        return
    t0 = time.time()
    labels = labels or {wf: wf for wf in NODES}
    lab = [labels[wf] for wf in NODES]
    dirs = {}
    for K in [K for K in KS if min(len(s) for s in nodes.values()) > K]:
        d = os.path.join(scratch, name.replace("/", "_").replace(" ", "_"), f"K{K}")
        PE.write_scan(d, {labels[wf]: nodes[wf] for wf in NODES}, K)
        dirs[K] = d
    mid_key = None if labels[1.0] == 1.0 else labels[1.0]
    est = PE.run_all(dirs, sorted(lab), mid_key=mid_key, linear=linear)
    if lab != list(NODES):             # what the production entire_cert does with these files: WFS hard-coded
        prod = PE.factored(dirs[max(dirs)], (0.99, 1.0, 1.01), max(dirs))
        for K in est:
            est[K]["G_production"] = prod if isinstance(prod, str) else "ran"
    for K in est:
        est[K]["ratiolaw_d2"] = ratio_law_d2({w: s for w, s in nodes.items()}, K)
        est[K].pop("linear_stdout", None) if not name.startswith("N0") else None
    rec = {"name": name, "family": family, "truth": truth, "labels": lab, "info": info or {},
           "est": {str(K): v for K, v in est.items()}, "seconds": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=jdefault) + "\n")
    DONE.add(name)
    e = est[min(12, max(est))]
    print(f"{name:40s} truth {truth}  arg12 {fmt(e['arg'].get(12))} ratio12 {fmt(e['ratio'].get(12))} "
          f"pade {fmt(e['pade_central_mid'])} G12 {fmt(e.get('G'))}  ({rec['seconds']} s)", flush=True)


def fmt(v):
    return "None" if v is None else f"{v:.6f}"


# ---------------------------------------------------------------- stages
def stage_null(scratch):
    # N0: the real data through the harness (records only; graded in report)
    for scan in ("scan25phys", "scan23phys"):
        name = f"N0 {scan}"
        if name in DONE:
            continue
        real = PP.load_real(os.path.join(HERE, scan))
        run_plant(name, "N0", {wf: real[wf][0] for wf in NODES}, None, scratch)
    for wt in PP.OMEGA_T:
        run_plant(f"N1 lone zero wt={wt}", "N1", PP.lone_zero(wt), float(wt), scratch)
    run_plant("N2 S=1", "N2", PP.no_zero("one"), None, scratch)
    run_plant("N2 S=exp(bx)", "N2", PP.no_zero("exp"), None, scratch)


def arc_stage(scratch, family, fam, tag):
    chk = load_checks()
    if tag not in chk:
        chk[tag] = {str(k): v for k, v in fam.checks().items()}
        save_checks(chk)
    for wt in PP.OMEGA_T:
        run_plant(f"{tag} wt={wt}", family, fam.nodes(wt), float(wt), scratch,
                  info={"fit_resid_max": {str(w): fam.resid[w] and max(fam.resid[w]) for w in NODES}})


def stage_parc(scratch):
    arc_stage(scratch, "P-ARC", PP.ArcFamily("arc"), "P-ARC")


def stage_parcb(scratch):
    for kind in ("chord", "mirror", "beta2"):
        arc_stage(scratch, "P-ARC-B", PP.ArcFamily(kind), f"P-ARC-B {kind}")


def stage_z2(scratch):
    for x2 in ("9.0-1.0j", "8.6+1.2j", "9.3+0j"):
        arc_stage(scratch, "P-Z2", PP.ArcFamily("arc", extra_zeros=(complex(x2),)), f"P-Z2 x2={x2}")


def stage_comp(scratch):
    for kind in ("C-W1", "C-W2", "C-W3", "C-EXP"):
        name = f"{kind}"
        if name in DONE:
            continue
        nodes, _, info, truth = PP.completion(kind)
        chk = load_checks()
        chk[kind] = {"info": {str(k): v for k, v in info.items()}, "truth": truth}
        save_checks(chk)
        run_plant(name, "C", nodes, truth["im_x"], scratch, info={"truth_im_sqrt": truth["im_sqrt"]})


def noisy(nodes, sd_fn, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for wf in NODES:
        sig = list(nodes[wf])
        for k in range(1, 14):
            sd = sd_fn(wf, k, sig)
            n = rng.standard_normal(2) / np.sqrt(2)
            sig[k] = sig[k] + mp.mpf(sd) * mp.mpc(n[0], n[1])
        out[wf] = sig
    return out


def stage_noise(scratch):
    fam = PP.ArcFamily("arc")
    real = fam.real

    def sd_err(wf, k, sig):
        se = real[wf][1]
        if k < len(se):
            return se[k]
        return se[12] / float(abs(real[wf][0][12])) * float(abs(sig[k]))   # k = 13: the relative error of k = 12
    for wt in ("1.0000", "0.9985"):
        base = fam.nodes(wt)
        for seed in range(10):
            run_plant(f"P-NOISE sigma_err wt={wt} seed={seed}", "P-NOISE-a", noisy(base, sd_err, seed), float(wt),
                      scratch, linear=False)
        for eps in ("1e-12", "1e-9", "1e-6", "1e-4"):
            for seed in range(5):
                run_plant(f"P-NOISE rel={eps} wt={wt} seed={seed}", f"P-NOISE-b {eps}",
                          noisy(base, lambda wf, k, sig, e=float(eps): e * float(abs(sig[k])), 100 + seed), float(wt),
                          scratch, linear=False)


def corrupt_set(base_name, nodes, truth, scratch, family):
    run_plant(f"{base_name} clean twin", family, nodes, truth, scratch)
    for k in (12, 9, 5):
        n2 = {wf: list(s) for wf, s in nodes.items()}
        n2[1.0][k] = -n2[1.0][k]
        run_plant(f"{base_name} X1 flip sigma_{k}", family, n2, truth, scratch)
    for k in (12, 9, 5, 2):
        n2 = {wf: list(s) for wf, s in nodes.items()}
        n2[1.0][k] = n2[1.0][k] * (1 + mp.mpf("1e-3"))
        run_plant(f"{base_name} X2 scale sigma_{k}", family, n2, truth, scratch)
    n2 = {wf: list(s) for wf, s in nodes.items()}
    for wf in NODES:
        n2[wf][12] = n2[wf][12] * (1 + mp.mpf("1e-3"))
    run_plant(f"{base_name} X2 scale sigma_12 all nodes", family, n2, truth, scratch)
    for tag, lab in (("X3a all +0.0005", {0.99: 0.9905, 1.0: 1.0005, 1.01: 1.0105}),
                     ("X3b 1.01->1.0105", {0.99: 0.99, 1.0: 1.0, 1.01: 1.0105}),
                     ("X3c 1.00->1.0005", {0.99: 0.99, 1.0: 1.0005, 1.01: 1.01})):
        run_plant(f"{base_name} {tag}", family, nodes, truth, scratch, labels=lab)


def stage_corrupt(scratch):
    fam = PP.ArcFamily("arc")
    corrupt_set("CORR P-ARC wt=1.0000", fam.nodes("1.0000"), 1.0, scratch, "CORR P-ARC")
    nodes, _, info, truth = PP.completion("C-W1")
    corrupt_set("CORR C-W1", nodes, truth["im_x"], scratch, "CORR C-W1")


REFIT_WT = ("0.9900", "0.9950", "0.9970", "0.9980", "0.9990", "0.99950", "1.0000", "1.00050", "1.0010", "1.0020",
            "1.0030", "1.0050", "1.0100")


def stage_refit(scratch):
    """Added after the freeze (section 3.6): refit L with the planted zero at each wt, i.e. ask which wt the real
    coefficients can be fitted with, at the P-ARC accuracy."""
    real = PP.load_real()
    chk = load_checks()
    res = chk.get("REFIT", {})
    for wt in REFIT_WT:
        name = f"REFIT P-ARC wt={wt}"
        if name in DONE:
            continue
        mp.mp.dps = PP.DPS
        nodes, resid = {}, {}
        for wf in NODES:
            arc = PP.Arc(wf, "arc")
            lap = PP.Laplace(arc, PP.mpc(PP.BETA[wf]) - 1, 6)
            xr = PP.x0_planted(wf, wt)
            r = lap.fit(real[wf][0], PP.factor([xr]))
            resid[str(wf)] = max(r)
            nodes[wf] = lap.coeffs(PP.factor([xr]), normalise=True)
        res[wt] = resid
        chk["REFIT"] = res
        save_checks(chk)
        run_plant(name, "REFIT", nodes, float(wt), scratch, info={"fit_resid_max": resid}, linear=False)


def load_checks():
    return json.load(open(CHECKS)) if os.path.exists(CHECKS) else {}


def save_checks(c):
    with open(CHECKS, "w") as f:
        json.dump(c, f, default=jdefault, indent=1)


# ---------------------------------------------------------------- grading
def grade(rec, K=12, truth=None):
    e = rec["est"][str(K)]
    wt = rec["truth"] if truth is None else truth
    arg, rat = {int(k): v for k, v in e["arg"].items()}, {int(k): v for k, v in e["ratio"].items()}
    a, r = arg.get(K), rat.get(K)
    g = {"arg": a, "ratio": r, "pade": e["pade_central_mid"], "G": e.get("G"),
         "aitken": {int(k): v for k, v in e["aitken_quad"].items()}.get(K)}
    out = {"est": g}
    if wt is None:
        return out
    out["err"] = {k: (v - wt if v is not None else None) for k, v in g.items()}
    out["P1"] = a is not None and r is not None and r <= wt <= a
    out["P2"] = all(arg.get(k) is not None and arg[k] >= wt for k in range(8, K + 1)) and \
        all(rat.get(k) is not None and rat[k] <= wt for k in range(9, K + 1))
    out["P3"] = g["pade"] is not None and g["G"] is not None and abs(g["pade"] - wt) <= TOL_BEST and \
        abs(g["G"] - wt) <= TOL_BEST
    out["P3_pade"] = g["pade"] is not None and abs(g["pade"] - wt) <= TOL_BEST
    out["P3_G"] = g["G"] is not None and abs(g["G"] - wt) <= TOL_BEST
    out["P4"] = g["aitken"] is not None and abs(g["aitken"] - wt) <= TOL_AITKEN
    return out


def detectors(rec, K=12):
    e = rec["est"][str(K)]
    arg, rat = {int(k): v for k, v in e["arg"].items()}, {int(k): v for k, v in e["ratio"].items()}
    a8 = [arg.get(k) for k in range(8, K + 1)]
    r9 = [rat.get(k) for k in range(9, K + 1)]
    fired = []
    if None in a8 or any(x <= y for x, y in zip(a8, a8[1:])):
        fired.append("D-arg")
    if None in r9 or any(x > y for x, y in zip(r9, r9[1:])):
        fired.append("D-ratio")
    a, r, p, G = arg.get(K), rat.get(K), e["pade_central_mid"], e.get("G")
    if None in (a, r) or r > a or p is None or G is None or not (r <= p <= a) or not (r <= G <= a):
        fired.append("D-bracket")
    cen = e["pade_central"]
    far = any(v is None or abs(complex(*v) - PE.X0) > 1 for c in cen for v in e["pade_poles_mid"].get(c, [None]))
    cv = e["pade_central_vals"]
    if far or None in cv or max(cv) - min(cv) > 1e-3:
        fired.append("D-pade")
    if K >= 12 and (e.get("G") is None or e.get("G_prev") is None or e.get("G_last") is None
                    or e.get("G_last_prev") is None or e["G_last"] > e["G_last_prev"] or abs(e["G"] - e["G_prev"]) > 5e-4):
        fired.append("D-G")
    if e["ratiolaw_d2"] > 0.05:
        fired.append("D-ratiolaw")
    return fired


def records():
    return [json.loads(line) for line in open(OUT) if line.strip()]


def report():
    recs = records()
    by = {r["name"]: r for r in recs}
    print(f"{len(recs)} records in {OUT}")
    fams = {}
    for r in recs:
        fams.setdefault(r["family"], []).append(r)
    for fam, rs in fams.items():
        print(f"\n== {fam}: {len(rs)} plants")
        for r in rs:
            for K in KS:
                g = grade(r, K)
                det = detectors(r, K)
                if "err" in g:
                    errs = " ".join(f"{k} {v * 1e4:+7.2f}" if v is not None else f"{k}   None" for k, v in g["err"].items())
                    print(f"  {r['name']:42s} K{K} truth {r['truth']:.6f} P1 {g['P1']:d} P2 {g['P2']:d} "
                          f"P3 {g['P3']:d} P4 {g['P4']:d} | err x1e4: {errs} | {','.join(det) or '-'}")
                else:
                    ests = " ".join(f"{k} {fmt(v)}" for k, v in g["est"].items())
                    print(f"  {r['name']:42s} K{K} {ests} | {','.join(det) or '-'}")
    return by


DONE = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch"))
    a = ap.parse_args()
    if a.stage == "report":
        report()
        return
    DONE.update(done_names())
    stages = {"null": stage_null, "parc": stage_parc, "parcb": stage_parcb, "z2": stage_z2, "comp": stage_comp,
              "noise": stage_noise, "corrupt": stage_corrupt, "refit": stage_refit}
    for s in (stages if a.stage == "all" else [a.stage]):
        stages[s](a.scratch)


if __name__ == "__main__":
    main()
