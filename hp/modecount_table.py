#!/usr/bin/env python3
"""Summarise hp/data/modecount_results.jsonl (the mode counts) as Markdown tables for notes/MODE_COUNT.md.

    python3 modecount_table.py [results.jsonl]
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
recs = sorted(last.values(), key=lambda r: (r["D"], -(r.get("M") or 0)))

print("| D | solver, M | omega | lam_g | B1a | B1b | B1c | B1 | U (poles) | full (poles) | zeros in full box (upper bound) | evaluations | max step (rad) | min ln|g| |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    b = r["boxes"]
    sol = "f64 kc %d" % r["kc"] if r.get("kc") else "dd M %d%s" % (r["M"], ", kf 13" if r.get("kfreeze") else "")
    mx = max(v["max_step"] for v in b.values())
    mn = min(v["min_lng"] for v in b.values())
    print(f"| {r['D']:.4f} | {sol} | {r['omega']:.4f} | {r['lam_g']:.6f} | {b['B1a']['winding']:+.4f} | {b['B1b']['winding']:+.4f} | "
          f"{b['B1c']['winding']:+.4f} | {b['B1']['winding']:+.4f} | {b['U']['winding']:+.4f} ({b['U']['poles_inside']}) | "
          f"{b['full']['winding']:+.4f} ({b['full']['poles_inside']}) | {b['full']['zeros_upper_bound']} | {r['evaluations']} | {mx:.3f} | {mn:.2f} |")
print()
print("| D | solver, M | aL | a1 | a2 | aR | y1 | y2 | real-axis sign changes | edges: points / max raw phase step of det M (rad) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    sol = "f64 kc %d" % r["kc"] if r.get("kc") else "dd M %d" % r["M"]
    sc = ", ".join(f"({a:.2f}, {b:.2f})" for a, b in r["real_sign_changes"])
    ed = "; ".join(f"{k} {e['points']}/{e['max_raw_step']:.2f}" for k, e in r["edges"].items())
    print(f"| {r['D']:.4f} | {sol} | {r['aL']:.3f} | {r['a1']:.3f} | {r['a2']:.3f} | {r['aR']:.3f} | {r['y1']:.3f} | {r['y2']:.3f} | {sc} | {ed} |")
