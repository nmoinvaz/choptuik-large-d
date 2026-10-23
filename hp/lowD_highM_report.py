#!/usr/bin/env python3
"""Table of the high-M pilot (hp/lowD_highM.py; notes/HIGH_M.md): every point with its resolution pair, Delta at the
larger M, the M difference (the error estimate), |F|, iterations, seconds and series tails; the local machine's
values where they overlap; and the README's formula in s = sqrt(D - 3) (hp/lowD_formula.py, fitted to Ecker, Ecker,
Grumiller & Jechtl's values, good to about 1e-5).

    cd hp && python3 lowD_highM_report.py [data/lowD_highM/pilot.jsonl] [--lift=data/lowD_highM/lift_D3.7.jsonl]
    cd hp && python3 lowD_highM_report.py --stage1 [--lift=...]
    cd hp && python3 lowD_highM_report.py --project [--ladder=80,96,128,160,192,256,320,384,512] [--dD=0.05]

--project: an estimate (not a bound) of the M and the time a continuation from the pilot's last D to D = 3.05 would
need at the pilot's criterion (the M difference below 1e-8). Modes: hp/lowD_cost.py's model (four variants) checked
against every pair and helper difference of the pilot (K = M - 1 sufficed or not); the lowest variant, scaled to fit
them all, gives K. M_w is the first rung of --ladder with M - 1 >= K, M_n the next, M_c the one below. Time: the wall
time of the pilot's last complete point (from logs/pilot.log; all its solves, two-pass solves included) scaled by
sum(M^3) over its resolutions, plus a move up (the new M_n in two passes, 1.2 times its share) where the rungs change.
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
LOCAL = {(3.7, 96): 3.4652004823053, (3.6, 128): 3.4527209746080, (3.55, 96): 3.4404023464863085839,
         (3.4, 80): 3.3662519179569, (3.35, 80): 3.3234621776975}
# the local machine's M 80 -> 96 changes of Delta (the task of this run)
LOCAL_80_96 = {3.7: -1.2e-9, 3.65: -3.3e-9, 3.6: -4.3e-9, 3.55: -2.5e-9, 3.5: -1.27e-8, 3.45: -6.6e-8}


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
    """see the module docstring (--project)"""
    import datetime
    import lowD_cost
    ladder = [int(v) for v in opts.get("ladder", "80,96,128,160,192,256,320,384,512").split(",")]
    dD = float(opts.get("dD", 0.05))
    target = 1e-8
    pairs = [json.loads(ln) for ln in open(os.path.join(HERE, "data", "lowD_highM", "pilot.jsonl")) if ln.strip()]
    pairs = [r for r in pairs if "Mw" in r]
    A4 = lowD_cost.calibrate(55, 2.6e-8, lowD_cost.BETA[0][1])

    def kmodel(D):
        bl, bp = lowD_cost.beta_of(D, "linear"), lowD_cost.beta_of(D, "power")
        f1 = lowD_cost.F1_of(D)
        ks = [lowD_cost.modes_needed(A, b, target) for b in (bl, bp) for A in (A4, A4 * (f1 / 16.23) ** 2)]
        return min(ks), max(ks)
    # 1. the modes K = M_w - 1 the pilot found sufficient (|Delta(M_n) - Delta(M_w)| < 1e-8) or not, against the model
    print("modes for a 1e-8 Delta: the pilot's pairs against hp/lowD_cost.py's model (lowest - highest of its 4 variants)")
    lo_s, hi_s = 0.0, float("inf")
    seen = set()
    checks = []
    for r in pairs:       # the pair (M_w -> M_n) and the helper (M_c -> M_w)
        checks.append((r["D"], r["Mw"], r["diff"]))
        if r.get("Mc") and r.get("diff_c") is not None:
            checks.append((r["D"], r["Mc"], r["diff_c"]))
    for D_, Mw_, diff_ in sorted(checks, key=lambda c: (-c[0], c[1])):
        if (D_, Mw_) in seen:
            continue
        seen.add((D_, Mw_))
        klo, khi = kmodel(D_)
        ok = abs(diff_) < target
        K = Mw_ - 1
        if ok:
            hi_s = min(hi_s, K / klo)       # K suffices: the needed K (scale x klo) is at most K
        else:
            lo_s = max(lo_s, K / klo)       # K does not: the needed K is above it
        print(f"  D {D_:.3f}: K {K:3d} {'suffices    ' if ok else 'is too few  '} ({diff_:+.1e}); model {klo:5.0f}-{khi:<5.0f}"
              f" -> scale of the lowest variant {'<=' if ok else '>'} {K / klo:.3f}")
    s_a, s_b = sorted((lo_s if lo_s > 0 else 1.0, hi_s if hi_s < float("inf") else 1.0))
    print(f"the lowest variant needs a scale above {lo_s:.3f} (K too few) and at most {hi_s:.3f} (K sufficed)"
          + ("" if lo_s <= hi_s else ": no single scale fits all (the M differences do not grow smoothly with D, e.g."
             " 128 -> 160: -6.3e-9 at 3.2, -6.4e-9 at 3.175, -4.4e-8 at 3.15)")
          + f"; K is projected with {s_a:.3f}-{s_b:.3f} times the lowest variant (the higher variants need more than the"
          " K that sufficed)")
    # 2. measured wall times from logs/pilot.log: each complete point (from its first start after the previous point's
    # last record to its pair record; failed tries before a node change excluded) and each move up (from the decision
    # to its record); other resolutions scale as M^3
    log = [ln.rstrip("\n") for ln in open(os.path.join(HERE, "data", "lowD_highM", "logs", "pilot.log"))]
    sec = lambda h: sum(int(x) * m for x, m in zip(h.split(":"), (3600, 60, 1)))  # noqa: E731
    pts = []
    for r in pairs:
        if r["decision"] != "pair":
            continue
        D_ = r["D"]
        i1 = next((i for i, ln in enumerate(log) if ln[8:].startswith(f" D {D_:g}: Delta(M") and ln.endswith("; pair")), None)
        if i1 is None:
            continue
        ip = max((i for i, ln in enumerate(log[:i1]) if ": Delta(M" in ln and not ln[8:].startswith(f" D {D_:g}:")), default=0)
        t0 = next((ln[:8] for ln in log[ip:i1] if f" --- D {D_:g} (" in ln), None)
        if t0 is None:
            continue
        Ms_ = tuple(m for m in (r.get("Mc"), r["Mw"], r["Mn"]) if m)
        pts.append((D_, Ms_, (sec(log[i1][:8]) - sec(t0)) % 86400))
    mus = []
    for i, ln in enumerate(log):
        if "moving up to M" in ln:
            M2 = int(ln.split("moving up to M")[1].split("/")[1])
            j = next((k for k in range(i + 1, len(log)) if "; moved up" in log[k]), None)
            if j is not None:
                mus.append((float(ln.split(" D ")[1].split(":")[0]) if " D " in ln[:20] else None, M2, (sec(log[j][:8]) - sec(ln[:8])) % 86400))
    lastM = pts[-1][1]
    same = [t for D_, Ms_, t in pts if Ms_ == lastM]
    W0 = sum(m ** 3 for m in lastM)
    Ta, Tb = min(same), max(same)
    print(f"points at M {'/'.join(map(str, lastM))}: " + ", ".join(f"D {D_:g} {t / 60:.0f} min" for D_, Ms_, t in pts if Ms_ == lastM)
          + " (wall, load about 2: the pilot alone on the 2 cores; restarts and failed tries included)")
    if mus:
        print("moves up (the new M_n at that D, two passes): " + ", ".join(f"M {m} {t / 60:.0f} min" for _, m, t in mus))
    mu_ref = mus[-1] if mus else None
    print(f"from D {pairs[-1]['D']:g}, steps of {dD}: K from the scaled lowest variant; M_w the first rung with M - 1 >= K,")
    print(f"  M_n the next, M_c the one below; hours per point = ({Ta / 3600:.2f}-{Tb / 3600:.2f} h) x sum(M^3) / {W0:.3g}, plus a move")
    print("  up for each new M_n (the last measured one scaled as M^3). Not included: node changes (about 10% of a point")
    print("  each), step halvings, and the growth of the Taylor plans toward D = 3; so these are lower estimates.")
    print(f"  {'D':>5} {'K model':>11} {'K':>9} {'M_c/M_w/M_n':>27} {'h/point':>9} {'move up h':>9} {'h cumulative':>12}")
    D = round(pairs[-1]["D"] - dD, 6)
    cum = [0.0, 0.0]
    prev = [tuple(lastM), tuple(lastM)]
    if pairs[-1]["decision"].startswith("moved up"):
        prev = [(pairs[-1].get("Mc") or 0, pairs[-1]["Mw"], pairs[-1]["Mn"])] * 2
    while D >= 3.05 - 1e-9:
        klo, khi = kmodel(D)
        cells = []
        for j, (sc, T) in enumerate(((s_a, Ta), (s_b, Tb))):
            K = klo * sc
            iw_ = next((i for i, M in enumerate(ladder) if M - 1 >= K), None)
            if iw_ is None or iw_ + 1 >= len(ladder):
                cells.append(None)
                continue
            Mc, Mw, Mn = (ladder[iw_ - 1] if iw_ else None), ladder[iw_], ladder[iw_ + 1]
            tp = T * sum(m ** 3 for m in (Mc, Mw, Mn) if m) / W0
            mu = sum(mu_ref[2] * (m / mu_ref[1]) ** 3 for m in {Mn, Mw} - set(prev[j][1:])) if mu_ref else 0.0
            cum[j] += tp + mu
            prev[j] = (Mc, Mw, Mn)
            cells.append((K, f"{Mc}/{Mw}/{Mn}", tp, mu, cum[j]))
        if None in cells:
            print(f"  {D:5.2f} {klo:5.0f}-{khi:<5.0f}  beyond the ladder")
            break
        a_, b_ = cells
        print(f"  {D:5.2f} {klo:5.0f}-{khi:<5.0f} {a_[0]:4.0f}-{b_[0]:<4.0f} {a_[1]:>13}|{b_[1]:<13} {a_[2] / 3600:4.1f}-{b_[2] / 3600:<4.1f}"
              f" {a_[3] / 3600:4.1f}-{b_[3] / 3600:<4.1f} {a_[4] / 3600:5.1f}-{b_[4] / 3600:<5.1f}")
        D = round(D - dD, 6)


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
        if r.get("Mc"):
            vals[(D, r["Mc"])] = r["Delta_c"] + r["Delta_c_lo"]
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
    # every M solved at each D (working pair and helper), as hi + lo in mpmath, and the differences of consecutive M
    mp.mp.dps = 40
    allv = {}
    if os.path.exists(lift):
        for ln in open(lift):
            r = json.loads(ln)
            if r.get("converged") and r.get("tag", "highm") == "highm15":       # the pilot's 15 nodes
                allv.setdefault(r["D"], {})[r["M"]] = mp.mpf(r["Delta"]) + mp.mpf(r["Delta_lo"])
    for r in pts:
        d = allv.setdefault(r["D"], {})
        d[r["Mn"]] = mp.mpf(r["Delta"]) + mp.mpf(r["Delta_lo"])
        d[r["Mw"]] = mp.mpf(r["Delta_w"]) + mp.mpf(r["Delta_w_lo"])
        if r.get("Mc"):
            d[r["Mc"]] = mp.mpf(r["Delta_c"]) + mp.mpf(r["Delta_c_lo"])
    print("\nevery M solved at each D, and the change of Delta from the M before (the M the pilot needed: the smallest M "
          "whose change to the next is below 1e-8)")
    for D in sorted(allv, reverse=True):
        Ms = sorted(allv[D])
        ch = [f"{a}->{b} {mp.nstr(allv[D][b] - allv[D][a], 3)}" for a, b in zip(Ms, Ms[1:])]
        need = next((a for a, b in zip(Ms, Ms[1:]) if abs(allv[D][b] - allv[D][a]) < 1e-8), None)
        print(f"  {D:5.3f}: M {','.join(map(str, Ms))}; {'; '.join(ch)}; needed {need if need else '?'}")
    print("\nM 80 -> 96 against the local machine's (the task of this run):")
    for D, v in sorted(LOCAL_80_96.items(), reverse=True):
        d = allv.get(D, {})
        mine = d[96] - d[80] if 80 in d and 96 in d else None
        print(f"  D {D}: local {v:+.2e}, here {mp.nstr(mine, 3) if mine is not None else '-'}")


if __name__ == "__main__":
    main()
