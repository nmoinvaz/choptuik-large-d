#!/usr/bin/env python3
"""Table of the high-M pilot (hp/lowD_highM.py; notes/HIGH_M.md): every point with its resolution pair, Delta at the
larger M, the M difference (the error estimate), |F|, iterations, seconds and series tails; the local machine's
values where they overlap; and the README's formula in s = sqrt(D - 3) (hp/lowD_formula.py, fitted to Ecker, Ecker,
Grumiller & Jechtl's values, good to about 1e-5).

    cd hp && python3 lowD_highM_report.py [data/lowD_highM/pilot.jsonl] [--lift=data/lowD_highM/lift_D3.7.jsonl]
    cd hp && python3 lowD_highM_report.py --stage1 [--lift=...]
    cd hp && python3 lowD_highM_report.py --project [--ladder=80,96,128,160,192,256,320,384,512]

--project: an estimate (not a bound) of the M and the time a continuation to D = 3.2, 3.1 and 3.05 would need at the
pilot's criterion (the M difference below 1e-8): the modes K needed for 1e-8 from hp/lowD_cost.py's four model
variants, scaled by the pilot's own crossing (the D where the M_w difference first exceeded 1e-8, with K = M_w - 1 at
that D, against the model's lower value there); M_w the first rung of --ladder above K, M_n the next, M_c the one
below; seconds per Newton iteration c M^3, c fitted to the pilot's iterations (t_its of hp/data/lowD_highM/
pilot_points.jsonl, scaled to an idle 2-core machine by min(1, 2 / load) with the load from logs/load.log at that
time); per point the pilot's mean iterations per solve at each role; the points at --dD (0.05) from the pilot's last D.
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


def load_at(t, loads):
    """load average (1 min) logged closest to the epoch time t"""
    if not loads:
        return None
    i = min(range(len(loads)), key=lambda k: abs(loads[k][0] - t))
    return loads[i][1] if abs(loads[i][0] - t) < 600 else None


def project(opts):
    import datetime
    import math
    import lowD_cost
    ladder = [int(v) for v in opts.get("ladder", "80,96,128,160,192,256,320,384,512").split(",")]
    dD = float(opts.get("dD", 0.05))
    pts_path = os.path.join(HERE, "data", "lowD_highM", "pilot_points.jsonl")
    pairs = [json.loads(ln) for ln in open(os.path.join(HERE, "data", "lowD_highM", "pilot.jsonl")) if ln.strip()]
    pairs = [r for r in pairs if "Mw" in r]
    loads = []
    lp = os.path.join(HERE, "data", "lowD_highM", "logs", "load.log")
    if os.path.exists(lp):
        for ln in open(lp):
            f = ln.split()
            loads.append((datetime.datetime.fromisoformat(f[0]).timestamp(), float(f[1])))
    # seconds per Newton iteration, scaled to an idle machine, per M
    per = {}
    for ln in open(pts_path):
        r = json.loads(ln)
        if not r.get("converged") or not r.get("t_its"):
            continue
        # the record is written at the end of the solve: its iterations ended about 'seconds' before... use the
        # load logged at the record's end time when the solve's own time is not known; records carry no timestamp, so
        # take the pilot log's timestamps instead
        per.setdefault(r["M"], []).extend(r["t_its"])
    stamp = {}
    for ln in open(os.path.join(HERE, "data", "lowD_highM", "logs", "pilot.log")):
        f = ln.split()
        if len(f) > 4 and f[1] == "D" and f[3] == "M" and "Delta" in ln and ":" in f[0]:
            stamp.setdefault((float(f[2]), int(f[4].rstrip(":"))), f[0])
    day = datetime.date.fromisoformat(opts.get("day", "2026-10-23"))
    samples = []
    for ln in open(pts_path):
        r = json.loads(ln)
        if not r.get("converged") or not r.get("t_its"):
            continue
        hh = stamp.get((r["D"], r["M"]))
        L = None
        if hh and loads:
            t = datetime.datetime.combine(day if hh < "12" else day - datetime.timedelta(days=1),
                                          datetime.time.fromisoformat(hh)).timestamp()
            L = load_at(t, loads)
        for ti in r["t_its"][1:]:            # the first iteration of a resumed run includes the start-up
            samples.append((r["M"], ti, L))
    use = [(M, t * min(1.0, 2.0 / L)) for M, t, L in samples if L]
    if not use:
        use = [(M, t) for M, t, L in samples]
        print("no load log for the samples: unscaled times")
    # the timed Newton steps of the D = 3.7 lifts (block storage above M = 128), with the load recorded with them
    lift = os.path.join(HERE, "data", "lowD_highM", "lift_D3.7.jsonl")
    if os.path.exists(lift):
        for ln in open(lift):
            r = json.loads(ln)
            if r.get("lu_block") and r.get("s_newton_step"):
                use.append((r["M"], r["s_newton_step"] * min(1.0, 2.0 / max(r.get("load", 2.0), 1e-9))))
    # c from the block-storage sizes (M > 128) when there are any: the dense LU at M <= 128 is slower than c M^3
    big = [(M, t) for M, t in use if M > 128]
    c = sum(t / M ** 3 for M, t in (big or use)) / len(big or use)
    print(f"seconds per Newton iteration ~ c M^3 with c = {c:.2e} (an idle 2-core machine; fitted on "
          f"{'M > 128 (block storage)' if big else 'all M'}; {len(use)} iterations at M = {sorted({M for M, _ in use})})")
    byM = {}
    for M, t in use:
        byM.setdefault(M, []).append(t)
    for M in sorted(byM):
        print(f"  M {M}: measured (scaled) {sum(byM[M]) / len(byM[M]):7.1f} s per iteration over {len(byM[M])}; c M^3 = {c * M ** 3:7.1f} s")
    # iterations per solve by role
    its_c = [r["its_c"] for r in pairs if r.get("its_c")]
    its_w = [r["its_w"] for r in pairs if r["decision"] == "pair"]
    its_n = [r["its_n"] for r in pairs if r["decision"] == "pair"]
    ic = sum(its_c) / len(its_c) if its_c else 6.0
    iw = sum(its_w) / len(its_w)
    inn = sum(its_n) / len(its_n)
    print(f"Newton calls per solve (pilot means): helper {ic:.1f}, working {iw:.1f}, next {inn:.1f}")
    # the pilot's crossing: the first pair whose difference exceeded 1e-8
    target = 1e-8
    A4 = lowD_cost.calibrate(55, 2.6e-8, lowD_cost.BETA[0][1])

    def kmodel(D):
        bl, bp = lowD_cost.beta_of(D, "linear"), lowD_cost.beta_of(D, "power")
        f1 = lowD_cost.F1_of(D)
        ks = [lowD_cost.modes_needed(A, b, target) for b in (bl, bp) for A in (A4, A4 * (f1 / 16.23) ** 2)]
        return min(ks), max(ks)
    cross = [r for r in pairs if abs(r["diff"]) > target]
    scale = 1.0
    if cross:
        r0 = cross[0]
        Kc = r0["Mw"] - 1
        scale = Kc / kmodel(r0["D"])[0]
        print(f"pilot crossing: M {r0['Mw']} -> {r0['Mn']} = {r0['diff']:+.2e} at D {r0['D']}: K = {Kc} there against the "
              f"model's {kmodel(r0['D'])[0]:.0f}-{kmodel(r0['D'])[1]:.0f}; scale {scale:.2f}")
    D_last = pairs[-1]["D"]
    print(f"from the pilot's last D {D_last}, steps of {dD}:")
    print("  K: modes for 1e-8 (model range, and scaled by the pilot's crossing); M_c/M_w/M_n for the low and the high K;")
    print("  hours on an idle 2-core machine, per point and cumulative from the pilot's last D (low K - high K); move ups")
    print("  (two passes, 3-4 times a solve), node changes, step halvings and the growth of the Taylor steps toward D = 3")
    print("  are not included")
    print(f"  {'D':>5} {'K (model)':>11} {'K scaled':>9} {'low K':>13} | {'high K':>13} {'h/point':>11} {'h cumulative':>13}")
    tot = 0.0
    D = round(D_last - dD, 6)
    rows = []
    while D >= 3.05 - 1e-9:
        klo, khi = kmodel(D)
        ks = (klo * scale, khi * scale)
        out = []
        for K in ks:
            iw_ = next((i for i, M in enumerate(ladder) if M - 1 >= K), None)
            if iw_ is None or iw_ + 1 >= len(ladder):
                out.append(None)
                continue
            Mw, Mn = ladder[iw_], ladder[iw_ + 1]
            Mc = ladder[iw_ - 1] if iw_ > 0 else None
            tp = (ic * c * Mc ** 3 if Mc else 0) + iw * c * Mw ** 3 + inn * c * Mn ** 3
            out.append((Mc, Mw, Mn, tp))
        rows.append((D, klo, khi, ks, out))
        D = round(D - dD, 6)
    cum = [0.0, 0.0]
    for D, klo, khi, ks, out in rows:
        cells = []
        for j, o in enumerate(out):
            if o is None:
                cells.append(("beyond the ladder", None))
                cum[j] = float("inf")
            else:
                cum[j] += o[3]
                cells.append((f"{o[0]}/{o[1]}/{o[2]}", o[3]))
        print(f"  {D:5.2f} {klo:5.0f}-{khi:<5.0f} {ks[0]:4.0f}-{ks[1]:<4.0f} {cells[0][0]:>13} | {cells[1][0]:>13} "
              f"{(cells[0][1] or 0) / 3600:5.1f}-{(cells[1][1] or float('inf')) / 3600:<5.1f} {cum[0] / 3600:6.1f}-{cum[1] / 3600:<6.1f}")


def main():
    if "--project" in sys.argv:
        opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        return project(opts)
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
