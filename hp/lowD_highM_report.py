#!/usr/bin/env python3
"""Table of the high-M pilot (hp/lowD_highM.py; notes/HIGH_M.md): every point with its resolution pair, Delta at the
larger M, the M difference (the error estimate), |F|, iterations, seconds and series tails; the local machine's
values where they overlap; and the README's formula in s = sqrt(D - 3) (hp/lowD_formula.py, fitted to Ecker, Ecker,
Grumiller & Jechtl's values, good to about 1e-5).

    cd hp && python3 lowD_highM_report.py [data/lowD_highM/pilot.jsonl] [--lift=data/lowD_highM/lift_D3.7.jsonl]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# the README's formula (section "A formula for 3 < D <= 4"): Delta = sum c_k s^k, s = sqrt(D - 3)
FORMULA = [1.752403, 2.486429, 8.257623, -29.245836, 44.944706, -45.106964, 30.660868, -12.653060, 2.349289]
# the local machine's values quoted in comms.txt (2026-10-22 18:31 and the task of this run): (D, M) -> Delta
LOCAL = {(3.7, 96): 3.4652004823053, (3.6, 128): 3.4527209746080, (3.55, 96): 3.4404023464863085839}


def formula(D):
    s = (D - 3) ** 0.5
    return sum(c * s ** k for k, c in enumerate(FORMULA))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    path = args[0] if args else os.path.join(HERE, "data", "lowD_highM", "pilot.jsonl")
    lift = opts.get("lift", os.path.join(HERE, "data", "lowD_highM", "lift_D3.7.jsonl"))
    recs = [json.loads(ln) for ln in open(path) if ln.strip()]
    pts = [r for r in recs if "D" in r and "Mw" in r]
    stops = [r for r in recs if "stop" in r]
    vals = {}
    if os.path.exists(lift):
        for ln in open(lift):
            r = json.loads(ln)
            if r.get("converged"):
                vals[(r["D"], r["M"])] = r["Delta"] + r["Delta_lo"]
    print(f"{path}: {len(pts)} point records")
    print(f"{'D':>8} {'M_w/M_n':>9} {'Delta (M_n)':>22} {'M_w -> M_n':>11} {'|F| w, n':>17} {'its':>5} {'min w+n':>8} "
          f"{'tails (centre, horizon)':>23} {'formula - Delta':>15}  decision")
    for r in pts:
        D = r["D"]
        Dn = r["Delta"] + r["Delta_lo"]
        vals[(D, r["Mn"])] = Dn
        vals[(D, r["Mw"])] = r["Delta_w"] + r["Delta_w_lo"]
        print(f"{D:8.5f} {r['Mw']:>4}/{r['Mn']:<4} {r['Delta']:.16f} {r['diff']:+11.2e} {r['F_w']:8.1e} {r['F_n']:8.1e} "
              f"{r['its_w']:>2}+{r['its_n']:<2} {(r['seconds_w'] + r['seconds_n']) / 60:8.1f} {r['tail_c']:11.1e} {r['tail_h']:11.1e} "
              f"{formula(D) - Dn:+15.2e}  {r['decision'][:60]}")
    for s in stops:
        print(f"stop: {s['stop']} (last D {s['D_last']}, M {s['Mw']}/{s['Mn']}, {s['hours']} h)")
    print("\nagainst the local machine's values (comms.txt):")
    for (D, M), v in sorted(LOCAL.items(), reverse=True):
        mine = sorted((m, x) for (d, m), x in vals.items() if abs(d - D) < 1e-9)
        if not mine:
            print(f"  D {D} M {M}: {v:.13f}; no point here")
            continue
        same = [x for m, x in mine if m == M]
        big = mine[-1]
        line = f"  D {D}: local M {M} {v:.13f}"
        if same:
            line += f"; here M {M} {same[0]:.13f} ({same[0] - v:+.1e})"
        line += f"; here M {big[0]} {big[1]:.13f}, minus local M {M}: {big[1] - v:+.2e}"
        print(line)


if __name__ == "__main__":
    main()
