#!/usr/bin/env python3
"""The tables of matter/COMPLEX_CRITICAL.md from the raw records (choptuik-large-d-raw, folder I95-mac).

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw
    python3 solver/matter/complex_table.py ../choptuik-large-d-raw/I95-mac

Prints: the CSS's oscillatory mode and lambda_phys along the track (css/track_k0*.jsonl), the CSS box counts
(css/boxes_k0.jsonl), the imaginary-sector checks and counts (imag/checks.jsonl, imag/counts.jsonl) and the crate's
D = 60 count (imag/crate/modecount_test_results.jsonl).
"""
import glob
import json
import os
import sys


def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    R = sys.argv[1]
    # ---- CSS track
    tr = {0: {}, 1: {}}
    for f in ("css/track_k0.jsonl", "css/track_k0_b.jsonl"):
        for r in rows(os.path.join(R, f)):
            if r.get("kind") == "track":
                tr[r["mode"]][round(r["D"], 6)] = r
    print("CSS kappa = 0, along the track (complex_css.py track): D, omega, lambda_osc, Im lambda_osc / omega, "
          "radius spread, lambda_phys")
    Ds = sorted(tr[0])
    want = [4.1, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80]
    pick = []
    for w in want:
        d = min(Ds, key=lambda x: abs(x - w))
        if d not in pick:
            pick.append(d)
    print("| D | ω | λ_osc | Im λ_osc / ω | radius spread | λ_phys |")
    print("|---|---|---|---|---|---|")
    for d in pick:
        a, b = tr[0][d], tr[1].get(d)
        print("| %.4f | %.6f | %.8f + %.8fi | %.4f | %.0e | %s |" % (d, a["omega"], a["lam"][0], a["lam"][1],
              a["lam"][1] / a["omega"], a["radius_spread"], "%.10f" % b["lam"][0] if b else "-"))
    # ---- CSS boxes
    bx = rows(os.path.join(R, "css/boxes_k0.jsonl"))
    if bx:
        print("\nCSS box counts (complex_css.py boxes): D, box, R/x_h, winding, largest step, evaluations, plant, mutate")
        for r in bx:
            print("%5g %s R=%.2f wind %+.4f step %.2f ev %d plant %s mutate %s osc %.8f%+.8fi" % (
                r["D"], r["box_name"], r["R_over_xh"], r["winding"], r["largest_step"], r["evaluations"], r["plant"],
                r["mutate"], r["lam_osc"][0], r["lam_osc"][1]))
    for r in rows(os.path.join(R, "css/roots_k0.jsonl")):
        print("root D=%g R=%.2f lambda = %.10f %+.10fi converged %s" % (r["D"], r["R_over_xh"], r["lam"][0], r["lam"][1],
                                                                       r["converged"]))
    # ---- imaginary sector
    print("\nImaginary sector checks (complex_imag.py check):")
    for r in rows(os.path.join(R, "imag/checks.jsonl")):
        print("D %.4f M %d: background mismatch <= %.1e, series at last node %.1e, null residual of (p, m) %.1e, "
              "sv min/next/max %.1e %.1e %.1e, Cauchy-Riemann %.1e (control %.1f), conj symmetry %.1e" % (
                  r["D"], r["M"], max(m["maxdiff"] for m in r["background_mismatch"]), r["series_at_last_node"],
                  r["null_residual_bg"], r["sv_min"], r["sv_next"], r["sv_max"], r["cauchy_riemann"],
                  r["cauchy_riemann_control"], r["conj_symmetry"]))
    print("\nImaginary sector counts (complex_imag.py count): D, kf, box, plant, mutate, winding of g, largest step, "
          "evaluations, candidate poles inside")
    for r in rows(os.path.join(R, "imag/counts.jsonl")):
        print("%9.4f %4s %-4s plant %-12s mutate %-8s wind %+.4f step %.2f hits %d ev %d poles %d" % (
            r["D"], r["kf"], r["box_name"], r["plant"], r["mutate"], r["winding_g"], r["largest_step"],
            r["refinement_limit_hits"], r["evaluations"], r["candidate_poles_inside"]))
    for r in rows(os.path.join(R, "imag/roots.jsonl")):
        print("root D %.4f from %s: lambda = %.10f %+.10fi" % (r["D"], r["start"], r["lam"][0], r["lam"][1]))
    for f in sorted(glob.glob(os.path.join(R, "imag/crate/modecount_test_results.jsonl"))):
        for r in rows(f):
            b = r.get("boxes", {})
            print("crate D %.4f M %d kfreeze %s plant %s: %s" % (r["D"], r["M"], r.get("kfreeze"), r.get("plant"),
                  {k: (round(v["winding_g"], 4), v["zeros_upper_bound"], round(v["max_step"], 2)) for k, v in b.items()}))


if __name__ == "__main__":
    main()
