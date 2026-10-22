#!/usr/bin/env python3
"""Post-freeze follow-up of the planted-truth test (notes/PLANTED_SELECTION.md, section 5): the converged sigma_13
(scan27J1000p960, 960 bits, err13 ~1e-22).

  real      the real sigma_0..13 through the harness at K = 11, 12, 13 (one record, "N0 scan27J1000p960");
  calib     K = 13 calibration: each estimator's error range on the ring-free plants (the 28 arc plants and C-EXP),
            and the zero range it implies for the real K = 13 estimate;
  predict   each section-3.4 completion's predicted sigma_13 and tau_13 against the converged values, and the frozen
            detectors at K = 13 against those the real data trips;
  recomplete the same completion recipes glued at sigma_13 (tail from k = 14): where the zero can still be, and whether
            the re-glued functions carry a ring (zero counts at wf = 1.00).

    nice -n 10 python3 planted_k13.py STAGE [--scratch DIR]
"""
import argparse
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted_plants as PP  # noqa: E402
import planted_post as PO  # noqa: E402
import planted_run as R  # noqa: E402
from arg_crossing import taus  # noqa: E402

SCAN13 = os.path.join(HERE, "scan27J1000p960")
REAL13 = "N0 scan27J1000p960"
EST = ("arg", "ratio", "pade", "G", "aitken")
C_ARC = [("arc", "1.0000"), ("chord", "1.0000"), ("mirror", "1.0000"), ("beta2", "1.0000"), ("arc", "0.9980"),
         ("arc", "1.0020"), ("arc", "0.99998"), ("chord", "1.00040"), ("mirror", "1.00101"), ("beta2", "0.99926")]


def c_arc_name(kind, wt):
    best = {("arc", "0.99998"), ("chord", "1.00040"), ("mirror", "1.00101"), ("beta2", "0.99926")}
    return f"C-ARC {kind} fit-wt=best({wt})" if (kind, wt) in best else f"C-ARC {kind} fit-wt={wt}"


def stage_real(scratch):
    R.DONE.update(R.done_names())
    real = PP.load_real(SCAN13)
    R.run_plant(REAL13, "N0", {wf: real[wf][0] for wf in PP.NODES}, None, scratch)


def ring_free(by):
    out = {}
    for r in by.values():
        if r["family"] in ("P-ARC", "P-ARC-B"):
            tag = "P-ARC" if r["family"] == "P-ARC" else r["name"].split(" wt=")[0].replace("P-ARC-B ", "")
            out.setdefault(tag, []).append(r)
    out["C-EXP"] = [by["C-EXP"]]
    return out


def stage_calib():
    by = {r["name"]: r for r in R.records()}
    real = by[REAL13]
    print(f"Real data, K = 13: " + ", ".join(f"{e} {R.grade(real, 13)['est'][e]:.6f}" for e in EST))
    print(f"Real data, K = 12: " + ", ".join(f"{e} {R.grade(real, 12)['est'][e]:.6f}" for e in EST))
    print(f"Detectors on the real data: K = 12 {R.detectors(real, 12)}, K = 13 {R.detectors(real, 13)}")
    e13 = R.grade(real, 13)["est"]
    fams = ring_free(by)
    pooled = {e: [] for e in EST}
    rows = []
    for fam, rs in fams.items():
        cells = []
        for e in EST:
            v = [R.grade(r, 13)["err"][e] for r in rs]
            v = [x for x in v if x is not None]
            pooled[e] += v
            cells.append((min(v), max(v), len(v)) if v else None)
        rows.append((fam, len(rs), cells))
    rows.append(("pooled ring-free", sum(len(r) for r in fams.values()),
                 [(min(pooled[e]), max(pooled[e]), len(pooled[e])) for e in EST]))
    print("\n| plants | n | " + " | ".join(f"{e}: error ×10⁴ → zero" for e in EST) + " |")
    print("|---|---|" + "---|" * len(EST))
    for fam, n, cells in rows:
        out = []
        for e, c in zip(EST, cells):
            if c is None:
                out.append("—")
                continue
            lo, hi, m = c
            out.append(f"{lo * 1e4:+.2f} … {hi * 1e4:+.2f} → {e13[e] - hi:.6f}–{e13[e] - lo:.6f}"
                       + (f" ({n - m} none)" if m < n else ""))
        print(f"| {fam} | {n} | " + " | ".join(out) + " |")
    # arc plants only (the coordinator's set), for the check
    arc = {e: [R.grade(r, 13)["err"][e] for f, rs in fams.items() if f != "C-EXP" for r in rs] for e in ("arg", "G")}
    for e, v in arc.items():
        print(f"arc plants only, {e}: errors {min(v) * 1e4:+.3f} … {max(v) * 1e4:+.3f} ×10⁻⁴ -> zero "
              f"{e13[e] - max(v):.6f}–{e13[e] - min(v):.6f}")
    refit = [r for r in by.values() if r["family"] == "REFIT"]
    for e in ("arg", "G"):
        v = [R.grade(r, 13)["err"][e] for r in refit if R.grade(r, 13)["err"][e] is not None]
        print(f"REFIT plants (post-freeze, arc shape, 13): {e} errors {min(v) * 1e4:+.2f} … {max(v) * 1e4:+.2f} ×10⁻⁴")


def tau13(sig):
    mp.mp.dps = 50
    return taus(list(sig[:14]))


def stage_predict():
    by = {r["name"]: r for r in R.records()}
    real = PP.load_real(SCAN13)
    rt = {wf: tau13(real[wf][0]) for wf in PP.NODES}
    real_rec = by[REAL13]
    real_det = set(R.detectors(real_rec, 13))
    a13 = R.grade(real_rec, 13)["est"]["arg"]
    preds = {}
    # the simple-zero law from tau_11, tau_12: tau_k = -c/(k x0^k)
    preds["simple-zero law from τ₁₁, τ₁₂"] = {wf: None for wf in PP.NODES}
    for kind in ("C-W1", "C-W2", "C-W3", "C-EXP"):
        preds[kind] = PP.completion(kind)[0]
    for kind, wt in C_ARC:
        preds[c_arc_name(kind, wt)] = PO.arc_completion(kind, wt)[0]
    print(f"real: sigma_13 / tau_13 at the nodes: " + "; ".join(
        f"{wf}: {mp.nstr(real[wf][0][13], 8)} / {mp.nstr(rt[wf][13], 8)}" for wf in PP.NODES))
    print(f"real arg_13 = {a13:.6f}; real K = 13 detectors: {sorted(real_det)}\n")
    print("| completion | σ₁₃ rel. error (0.99 / 1.00 / 1.01) | τ₁₃ rel. error | its arg₁₃ | K = 13 detectors beyond the "
          "real data's | survives |")
    print("|---|---|---|---|---|---|")
    out = {}
    for name, nodes in preds.items():
        if name.startswith("simple"):
            te = []
            for wf in PP.NODES:
                t = rt[wf]
                x0 = 11 * t[11] / (12 * t[12])
                p13 = t[12] * 12 / (13 * x0)
                te.append(float(abs(p13 - t[13]) / abs(t[13])))
            print(f"| {name} | — | {' / '.join(f'{x:.1e}' for x in te)} | — | — | — |")
            out[name] = {"tau13_rel": te}
            continue
        se = [float(abs(nodes[wf][13] - real[wf][0][13]) / abs(real[wf][0][13])) for wf in PP.NODES]
        te = [float(abs(tau13(nodes[wf])[13] - rt[wf][13]) / abs(rt[wf][13])) for wf in PP.NODES]
        rec = by.get(name)
        arg = R.grade(rec, 13)["est"]["arg"] if rec else None
        extra = sorted(set(R.detectors(rec, 13)) - real_det) if rec else ["no record"]
        surv = not extra
        out[name] = {"sigma13_rel": se, "tau13_rel": te, "arg13": arg, "extra_detectors": extra, "survives": surv,
                     "truth": rec["truth"] if rec else None}
        print(f"| {name} | {' / '.join(f'{x:.1e}' for x in se)} | {' / '.join(f'{x:.1e}' for x in te)} | "
              f"{'none' if arg is None else f'{arg:.6f}'} | {', '.join(extra) or '—'} | {'yes' if surv else 'no'} |")
    chk = R.load_checks()
    chk["K13 predict"] = out
    R.save_checks(chk)


def count(f, rad, n=1600):
    tot, prev = 0.0, None
    for j in range(n + 1):
        a = float(mp.arg(f(rad * mp.expjpi(2 * mp.mpf(j) / n))))
        if prev is not None:
            d = a - prev
            d -= 2 * np.pi * round(d / (2 * np.pi))
            tot += d
        prev = a
    return round(tot / (2 * np.pi))


def stage_recomplete():
    by = {r["name"]: r for r in R.records()}
    real = PP.load_real(SCAN13)
    e13 = R.grade(by[REAL13], 13)["est"]
    chk = R.load_checks()
    res = chk.get("K13 recomplete", {})
    jobs = [("C13-W1", "C-W1"), ("C13-W2", "C-W2"), ("C13-W3", "C-W3"), ("C13-EXP", "C-EXP")]
    jobs += [(f"C13-ARC {k}", k) for k in ("arc", "chord", "mirror", "beta2")]
    wts = [f"{0.9990 + 0.0001 * i:.4f}" for i in range(21)]
    for name, kind in jobs:
        if name in res:
            continue
        mp.mp.dps = PP.DPS
        if name.startswith("C13-ARC"):
            scan = PO.refit_scan(kind, wts, kc=13, real=real)
            bw = PO.best_wt(scan)
            funcs = {}
            nodes, info, truth = PO.arc_completion(kind, f"{bw:.5f}", kc=13, real=real, funcs=funcs)
            extra = {"best_wt": bw, "L2_floor": min(v[0] for v in scan.values()),
                     "fit_resid_max": {w: info[w]["fit_resid_max"] for w in info}}
        else:
            nodes, funcs, info, truth = PP.completion(kind, real=real, kc=13)
            extra = {"poles_max": {str(w): info[w].get("max_pole") for w in PP.NODES}} if kind == "C-EXP" else {}
        f = funcs[1.0]
        z73, z12 = count(f, mp.mpf("7.3")), count(f, mp.mpf(12))
        rec = {"truth": truth["im_x"], "truth_im_sqrt": truth["im_sqrt"], "zeros_lt_7.3": z73, "zeros_lt_12": z12,
               "x_R": {str(w): v["x_R"] for w, v in info.items()},
               "err_of_real_estimates": {e: e13[e] - truth["im_x"] for e in EST if e13[e] is not None}, **extra}
        res[name] = rec
        chk = R.load_checks()
        chk["K13 recomplete"] = res
        R.save_checks(chk)
        print(f"{name}: zero at {truth['im_x']:.6f} (Im √x: {truth['im_sqrt']:.6f}); zeros |x| < 7.3: {z73}, "
              f"|x| < 12: {z12}; {extra}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "planted_scratch"))
    a = ap.parse_args()
    {"real": lambda: stage_real(a.scratch), "calib": stage_calib, "predict": stage_predict,
     "recomplete": stage_recomplete}[a.stage]()


if __name__ == "__main__":
    main()
