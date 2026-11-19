#!/usr/bin/env python3
"""The quad-double values of Delta at large D into hp/data/delta_qd.csv: the full system (kfreeze 13) converged in
quad-double from the double-double states (notes/QUAD_DOUBLE.md 12.1 and 12.4), one row per D with Delta in four
parts (Delta = q0 + q1 + q2 + q3, each an f64; the first two are the double-double value).

    cd hp && python3 export_qd.py ../../choptuik-large-d-raw      (the raw-data clone; folders I940-dh610, I940-mac,
                                                                   T84-mac, T81 and qd2start-mac)

Sources. #940's window (I940-dh610/qd_window.jsonl, hp/qd_window.py --mode full on T73's and T75's pin-5 and pin-7
states, today's kernel) where it converged; elsewhere T84-mac's full leg, T81's states and qd2start-mac's second starts
(the kernel of 2026-11-05). At a D that #940 covers only its values are kept, since the two kernels' solutions differ
(below).
Error terms, each an estimate, not a bound:
  spread  the difference between two starts' exact sums, where a second start converged (a spread between starts);
  plan    #940's pin-5 change from plan tolerance 1e-32 to 1e-34;
  spread_nov5  the Nov 5 kernel's two-start spread where it had two (T84-mac, T81, qd2start-mac), kept for the
          notes that quote it; not an error term of the row.
  kernel  the full system is near-singular here (QUAD_DOUBLE 12.4), so its converged Delta moves with the arithmetic's
          rounding: the Nov 5 kernel's solutions differ from today's by I940-mac's measured gaps (kernel_gap.jsonl),
          and each row takes the largest measured gap as this term.
err is the largest term and err_kind names it.
"""
import csv
import glob
import json
import os
import sys

import mpmath as mp

mp.mp.dps = 80

HERE = os.path.dirname(os.path.abspath(__file__))
raw = sys.argv[1]


def total(q):
    return sum(map(mp.mpf, q))


EXACT = {}   # the grouping key round(D, 4) -> the record's own D, which the row reports (a 4-decimal D is up to
             # 5e-5 off a state's D, e.g. 67.742203, and Delta moves by up to 1e-7 over that, #997's discrepancy)


def key_of(D):
    k = round(D, 4)
    EXACT.setdefault(k, D)
    return k


def window():
    """#940's converged full-system solutions by D, and each D's pin-5 plan change"""
    sols, plan = {}, {}
    base, fine = {}, {}
    for line in open(os.path.join(raw, "I940-dh610", "qd_window.jsonl")):
        r = json.loads(line)
        if r.get("mode") != "full" or not r["conv"] or r.get("mutate"):
            continue
        k = key_of(r["D"])
        if r.get("plan_tol", 1e-32) == 1e-32:
            sols.setdefault(k, []).append((r["Delta_q"], f"I940-dh610/qd_window.jsonl:{r['state']}"))
            if r["state"].startswith("pin5"):
                base[k] = r["Delta_q"]
        elif r["state"].startswith("pin5"):
            fine[k] = r["Delta_q"]
    for k in base:
        if k in fine:
            plan[k] = abs(total(fine[k]) - total(base[k]))
    return sols, plan


def kernel_gap():
    """the largest measured |Delta(Nov 5 kernel) - Delta(today's)| (I940-mac/kernel_gap.jsonl), or None"""
    f = os.path.join(raw, "I940-mac", "kernel_gap.jsonl")
    if not os.path.exists(f):
        return None
    return max(abs(mp.mpf(json.loads(line)["gap"])) for line in open(f) if line.strip())


old = {}        # D -> list of (Delta parts, source): the kernel of 2026-11-05
for f in sorted(glob.glob(os.path.join(raw, "T84-mac", "fullleg_pin*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["conv"]:
            old.setdefault(key_of(r["D"]), []).append((r["Delta_q"], f"T84-mac/{os.path.basename(f)}:{r['state']}"))
for f in sorted(glob.glob(os.path.join(raw, "T81", "conv_pin*_D7*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r.get("ok") and r.get("F", 1) < 1e-50:
            old.setdefault(key_of(r["D"]), []).append((r["Delta_q"], f"T81/{os.path.basename(f)}"))
for line in open(os.path.join(raw, "qd2start-mac", "second.jsonl")):
    r = json.loads(line)
    if r["conv"]:
        old.setdefault(key_of(r["D"]), []).append((r["Delta_q"], f"qd2start-mac/second.jsonl:{r['state']}"))
new, plan = window()
old_spread = {}
for D, sl in old.items():
    dist = []
    for parts, _ in sl:
        if not any(p == parts for p in dist):
            dist.append(parts)
    if len(dist) > 1:
        old_spread[D] = abs(total(dist[0]) - total(dist[1]))
kgap = kernel_gap()
sols = dict(old)
sols.update(new)        # #940 replaces the older kernel's values at its D
rows = []
for D in sorted(sols):
    # distinct solutions only (a re-convergence of a saved solution repeats its record)
    distinct = []
    for parts, src in sols[D]:
        if not any(p == parts for p, _ in distinct):
            distinct.append((parts, src))
    parts, src = distinct[0]
    terms = {}
    if len(distinct) > 1:
        terms["spread"] = abs(total(parts) - total(distinct[1][0]))
    if D in plan:
        terms["plan"] = plan[D]
    if kgap is not None:
        terms["kernel"] = kgap
    kind = max(terms, key=terms.get) if terms else ""
    rows.append(dict(D=repr(EXACT.get(D, D)), Delta_q0=repr(parts[0]), Delta_q1=repr(parts[1]), Delta_q2=repr(parts[2]),
                     Delta_q3=repr(parts[3]), n_solutions=len(distinct),
                     spread=f"{float(terms['spread']):.1e}" if "spread" in terms else "",
                     plan=f"{float(terms['plan']):.1e}" if "plan" in terms else "",
                     spread_nov5=f"{float(old_spread[D]):.1e}" if D in old_spread else "",
                     err=f"{float(terms[kind]):.1e}" if kind else "", err_kind=kind, source=src))
out = os.path.join(HERE, "data", "delta_qd.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows ({len(new)} from #940) -> {out}; kernel gap {'%.1e' % float(kgap) if kgap is not None else 'not measured'}")
for r in rows:
    print(f"  D {float(r['D']):8.4f} Delta {r['Delta_q0']} {float(r['Delta_q1']):+.6e} {float(r['Delta_q2']):+.3e}  solutions "
          f"{r['n_solutions']} spread {r['spread'] or '-'} plan {r['plan'] or '-'} err {r['err'] or '-'} ({r['err_kind'] or '-'})")
