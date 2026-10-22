#!/usr/bin/env python3
"""Summarise hp/data/modecount_results.jsonl (the mode counts) as Markdown tables for notes/MODE_COUNT.md.

    python3 modecount_table.py [results.jsonl]

Table 1: per run, winding of g and of det M for every box, the expected zeros and candidate poles inside, and the
bound on the number of zeros. Table 2: sampling diagnostics per run.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "modecount_results.jsonl")
recs = [json.loads(l) for l in open(path)]
last = {}
for r in recs:
    last[r["tag"]] = r          # the latest run per tag
recs = sorted(last.values(), key=lambda r: (r["D"], -(r.get("M") or 0), r["tag"]))
BOXES = ["B0", "U0", "B1a", "B1b", "B1c", "B1", "U", "full"]


def sol(r):
    if r.get("kc"):
        return f"f64, kc {r['kc']}"
    return f"dd, M {r['M']}" + (", kf 13" if r.get("kfreeze") else "")


print("Windings: wind(g) [wind(det M) = wind(g) + expected - candidate poles]; zeros of det M <= wind(g) + expected.")
print()
print("| D | solver | λ_g | λ_phys | " + " | ".join(BOXES) + " |")
print("|---|---|---|---|" + "---|" * len(BOXES))
for r in recs:
    b = r["boxes"]
    cells = []
    for k in BOXES:
        x = b[k]
        cells.append(f"{x['winding_g']:+.3f} [{x['winding_det']:+d}]")
    print(f"| {r['D']:.4f} | {sol(r)} | {r['lam_g']:.6f} | {r['lam_f']:.6f} | " + " | ".join(cells) + " |")
print()
print("| D | solver | expected zeros / candidate poles inside, per box | zeros of det M in the full box (bound) |")
print("|---|---|---|---|")
for r in recs:
    b = r["boxes"]
    s = ", ".join(f"{k} {b[k]['expected_zeros_inside']}/{b[k]['candidate_poles_inside']}" for k in BOXES)
    print(f"| {r['D']:.4f} | {sol(r)} | {s} | ≤ {b['full']['zeros_upper_bound']} (+ B0: ≤ {b['B0']['zeros_upper_bound']}, U0: ≤ {b['U0']['zeros_upper_bound']}) |")
print()
print("| D | solver | aLL, aL, a1, a2, aR | y1, y2 | evaluations | s / eval | max phase step of g (rad) | max step of det M (rad) | ln\\|g\\| on the boundary | nearest candidate pole / expected zero to an edge | real-axis sign changes |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    b = r["boxes"]
    E = r["edges"].values()
    mx = max(e["max_step"] for e in E)
    mr = max(e["max_raw_step"] for e in E)
    lo = min(e["min_lng"] for e in E)
    hi = max(e["max_lng"] for e in E)
    dp = min(x["pole_min_dist_to_edge"] for x in b.values())
    dz = min(x["expected_zero_min_dist_to_edge"] for x in b.values())
    sc = ", ".join(f"({a:.2f}, {c:.2f})" for a, c in r["real_sign_changes"])
    spe = f"{r['sec_per_eval']:.0f}" if r.get("sec_per_eval") else "–"
    print(f"| {r['D']:.4f} | {sol(r)} | {r['aLL']:.3f}, {r['aL']:.3f}, {r['a1']:.3f}, {r['a2']:.3f}, {r['aR']:.3f} | "
          f"{r['y1']:.3f}, {r['y2']:.3f} | {r['evaluations']} | {spe} | {mx:.3f} | {mr:.2f} | {lo:.3f} … {hi:.3f} | "
          f"{dp:.3f} / {dz:.3f} | {sc} |")
