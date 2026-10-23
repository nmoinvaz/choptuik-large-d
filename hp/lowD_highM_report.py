#!/usr/bin/env python3
"""Table of the high-M pilot (hp/lowD_highM.py; notes/HIGH_M.md): every point with its resolution pair, Delta at the
larger M, the M difference (the error estimate), |F|, iterations, seconds and series tails; the local machine's
values where they overlap; and the README's formula in s = sqrt(D - 3) (hp/lowD_formula.py, fitted to Ecker, Ecker,
Grumiller & Jechtl's values, good to about 1e-5).

    cd hp && python3 lowD_highM_report.py [data/lowD_highM/pilot.jsonl] [--lift=data/lowD_highM/lift_D3.7.jsonl]
    cd hp && python3 lowD_highM_report.py --stage1 [--lift=...]

--stage1: the D = 3.7 lifts (hp/highm_check.py lift): per node set and M, Delta (hi + lo, 25 digits), the change from
the previous M, |F|, the Taylor steps of the two passes, the series tails, one Newton step's seconds by part, the LU's
MB, the peak RSS and the load average; and the change from the M = 80 state and from the local machine's M = 96 value.
"""
import mpmath as mp
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


def stage1(lift):
    mp.mp.dps = 40
    recs = [json.loads(ln) for ln in open(lift) if ln.strip()]
    m80 = mp.mpf(3.4652004835440313) + mp.mpf(1.572231757910144e-16)    # hp/data/lowD_dd.jsonl, D = 3.7, M = 80
    m96 = mp.mpf("3.4652004823053")                                       # the local machine (task of this run)
    for tag in sorted({r.get("tag", "highm") for r in recs}):
        rs = sorted((r for r in recs if r.get("tag", "highm") == tag), key=lambda r: r["M"])
        print(f"tag {tag}, nodes {rs[0]['nodes'][0]}..{rs[0]['nodes'][-1]} ({len(rs[0]['nodes'])})")
        print(f"  {'M':>4} {'N':>6} {'Delta':>27} {'change':>10} {'|F|':>8} {'steps':>9} {'tails c, h':>15} {'step s':>7} "
              f"{'jac s':>6} {'no d/dtau':>9} {'LU s':>5} {'LU MB':>6} {'RSS MB':>7} {'load':>5}")
        prev = None
        for r in rs:
            v = mp.mpf(r["Delta"]) + mp.mpf(r["Delta_lo"])
            ch = mp.nstr(v - prev, 3) if prev is not None else "-"
            f = lambda x, fmt: (fmt % x) if x is not None else "-"  # noqa: E731
            print(f"  {r['M']:>4} {r['N']:>6} {mp.nstr(v, 25):>27} {ch:>10} {r['F']:8.1e} {f(r.get('steps_pass1'), '%d'):>4}>{r['steps']:<4} "
                  f"{r['tail_c']:7.0e} {r['tail_h']:7.0e} {r['s_newton_step']:7.1f} {f(r['s_jacobian'], '%.1f'):>6} "
                  f"{f(r['s_jacobian_no_dtau'], '%.1f'):>9} {f(r['s_factor'], '%.2f'):>5} {r['lu_mb']:6.1f} {r['peak_rss_mb']:7.1f} {r['load']:5.1f}")
            prev = v
        r128 = [r for r in rs if r["M"] == 128]
        if r128:
            v = mp.mpf(r128[0]["Delta"]) + mp.mpf(r128[0]["Delta_lo"])
            print(f"  M = 128 minus the M = 80 state: {mp.nstr(v - m80, 4)}; minus the local machine's M = 96 value: {mp.nstr(v - m96, 4)}")


def main():
    if "--stage1" in sys.argv:
        opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        return stage1(opts.get("lift", os.path.join(HERE, "data", "lowD_highM", "lift_D3.7.jsonl")))
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
