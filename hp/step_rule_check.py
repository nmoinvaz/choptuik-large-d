#!/usr/bin/env python3
"""Checks of the ANM step tied to the re-seed (hp/step_rule.py, anm_run.py --a_rule reseed).

    python3 step_rule_check.py planted [--mutate none|tail10|noretry]
    python3 step_rule_check.py offline RAW [--leg I903-win2/ks_chain/leg07_D222.0000_ks240] [--mutate none|tail10]
        [--h_max 1] [--a_grow 1.25] [--a_hold 1] [--D_max D]
    python3 step_rule_check.py pilot RAW --jet JET --state STATE --as A1,A2,.. --scale S [--N N] [--refuse A,..]
        [--threads 4] [--out OUT.jsonl]

`planted` runs the rule on synthetic data with a known answer, no files: (1) series with |z_k| = C 3^k / k!, the law of
the chain's a_h tail at orders 12-24, whose estimated error e_N+1 a^(N+1) must lie within a factor 2 of the exact tail
sum for a <= 2 at N 24 and 31; (2) the solved a must meet its equation to 1e-12; (3) a chain against a planted edge
(the re-seed accepts while the error is at most s* delta_pole) must keep its mean step at least 0.85 of the edge's and
refuse at most once per two bases. --mutate tail10 multiplies the coefficients the estimate reads by 10 and must fail
(1); noretry keeps the level after a refusal and must fail (3).

`offline` reads one leg of the chain in the raw-data repository: anm_run.jsonl, the series saved with --save_jet 1, the
states and the predictions saved with --save_pred 1. Criteria, each must hold:
- A, the estimate: at every attempt whose target D has a converged state in the leg, the prediction's a_h error,
  summed exactly from the saved series against that state, is within a factor 2 of the rule's estimate wherever the
  estimate is at least 1e-38 (below that the series' rounding, about 1e-40 to 1e-39, dominates).
- B, the prediction: where a saved prediction was made from the leg's series (its meta a equal to an attempt's), the
  prediction summed here equals it to 1e-60 in every row.
- C, one level: the estimated levels err(a) / delta_pole(D + a) of the attempts whose re-seed succeeded all lie below
  those of the attempts whose re-seed failed, so one level s classifies every attempt of the leg as the chain's
  outcome. Its window and middle s_mid are printed.
- D, the replay: the leg's own step rule (a0 = h_max, --a_grow, --a_hold, halving; the defaults are #903's flags),
  run against the edge s_mid, takes exactly the leg's recorded steps.
It then prints the steps --a_rule reseed would take against the same edge at N 24 and at N 31 (the series' a_h norms
past order 24 extrapolated with the two-step ratios linear in the order, and bounded with them constant), the
refusals, and the wall per unit of D, from the leg's own costs. --mutate tail10 multiplies the rule's input
coefficients at orders N-4..N by 10 (the predictions and states are not touched) and must fail A. Exit 1 on failure.

`pilot` (needs hpkernel) re-seeds predictions summed from a saved series (anm_run.py --save_jet) at the steps --as, at
the series' first N orders, with the chain's re-seed (reseed.Reseeder, quad-double polish from the prediction's own
data). The rule's own step at level --scale is tried with them. Every a whose estimated level err(a) / delta_pole(D + a)
is at most --scale must re-seed, and every a in --refuse must fail, else exit 1. One record per a goes to --out.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from step_rule import ReseedRule, ah_norms, pole_delta, tail_next, TAIL_W  # noqa: E402

LEG = "I903-win2/ks_chain/leg07_D222.0000_ks240"
K_NODES = 5


def parse(argv, opt):
    argv = list(argv)
    pos = []
    while argv:
        x = argv.pop(0)
        if x.startswith("--"):
            k = x[2:]
            if "=" in k:
                k, v = k.split("=", 1)
            else:
                v = argv.pop(0)
            if k not in opt:
                raise SystemExit(f"step_rule_check.py: unknown option --{k}")
            opt[k] = type(opt[k])(v) if opt[k] is not None else v
        else:
            pos.append(x)
    return pos, opt


def mutated(e, mutate):
    """the rule's input norms, with --mutate tail10 the orders N-4..N multiplied by 10"""
    e = list(e)
    if mutate == "tail10":
        for k in range(len(e) - 1 - TAIL_W, len(e)):
            e[k] *= 10.0
    return e


def extend(e, N2, linear=True, lo=12):
    """e_0..e_N2 from e_0..e_N: past N by the two-step ratios r_k = (e_(k-2) / e_k)^(1/2), k = lo..N, fitted as a line in
    k (one-step ratio rho_k = r_k + slope / 2), or with rho constant at the last four orders' geometric mean"""
    N = len(e) - 1
    out = list(e)
    if linear:
        ks = np.arange(lo, N + 1)
        r = np.array([(e[k - 2] / e[k]) ** 0.5 for k in ks])
        b, a0 = np.polyfit(ks, r, 1)
        for k in range(N + 1, N2 + 1):
            out.append(out[-1] / (a0 + b * k + 0.5 * b))
    else:
        rho = tail_next(e)[1]
        for k in range(N + 1, N2 + 1):
            out.append(out[-1] / rho)
    return out


# ---------------------------------------------------------------------------------------------------------- planted
def planted(argv):
    _, opt = parse(argv, dict(mutate="none"))
    ok_all = True
    # (1) the estimate against the exact tail of |z_k| = C 3^k / k!
    worst = []
    for N in (24, 31):
        e = [3.0 ** k / math.factorial(k) for k in range(N + 1)]
        rule = ReseedRule(0.3)
        rule.start(mutated(e, opt["mutate"]), 225.0, 10.0)
        for a in (0.25, 0.5, 1.0, 1.5, 2.0):
            exact = sum(3.0 ** k / math.factorial(k) * a ** k for k in range(N + 1, N + 60))
            worst.append((N, a, rule.err(a) / exact))
    r_lo, r_hi = min(w[2] for w in worst), max(w[2] for w in worst)
    ok1 = 0.5 <= r_lo and r_hi <= 2.0
    print(f"(1) estimate / exact tail at N 24, 31 and a 0.25-2: {r_lo:.3f} .. {r_hi:.3f} (pass in [0.5, 2]): "
          f"{'pass' if ok1 else 'FAIL'}", flush=True)
    # (2) the solved step meets its equation
    rule = ReseedRule(0.3, scale=0.07)
    e = [1.0] + [1.43e-32 * 7.46 ** (24 - k) for k in range(1, 25)]
    a = rule.start(e, 229.05, 5.0)
    resid = abs(rule.err(a) / (0.07 * pole_delta(229.05 + a, 0.3)) - 1.0)
    ok2 = resid <= 1e-12 and not rule.capped
    print(f"(2) solved a {a:.6f}: |err(a) / (s delta_pole) - 1| = {resid:.1e} (pass <= 1e-12, uncapped): "
          f"{'pass' if ok2 else 'FAIL'}", flush=True)
    # (3) a chain against a planted edge s* = 0.07 over 10 units of D, at N 24 and 31
    ok3 = True
    for N in (24, 31):
        e = extend([1.0] + [1.43e-32 * 7.46 ** (24 - k) for k in range(1, 25)], N)
        sim = simulate_rule(lambda D: e, 222.0, 232.0, edge=0.07, u0=0.3, h_max=5.0, scale=1.0,
                            noretry=opt["mutate"] == "noretry")
        rule = ReseedRule(0.3, scale=0.07)
        a_star = rule.start(e, 227.0, 5.0)
        good = sim["stopped"] is None and sim["mean_a"] >= 0.85 * a_star and sim["refusals"] <= 0.5 * sim["bases"]
        ok3 = ok3 and good
        print(f"(3) N {N}: edge step {a_star:.3f}, mean step {sim['mean_a']:.3f} over {sim['bases']} bases, "
              f"{sim['refusals']} refusals{', stopped at D ' + format(sim['stopped'], '.3f') if sim['stopped'] else ''} "
              f"(pass: mean >= 0.85 edge, refusals <= bases / 2): {'pass' if good else 'FAIL'}", flush=True)
    ok_all = ok1 and ok2 and ok3
    print(f"planted ({opt['mutate']}): {'pass' if ok_all else 'FAIL'}", flush=True)
    return 0 if ok_all else 1


def simulate_rule(e_at, D0, D1, edge, u0, h_max, scale, noretry=False, a_max=math.inf, cost=None):
    """--a_rule reseed from D0 to D1 against a planted edge: an attempt is accepted iff its estimated level is at most
    edge. e_at(D): the series' a_h norms at a base. cost = (series, accepted, refused) seconds, for the wall."""
    rule = ReseedRule(u0, scale)
    D, bases, refusals, steps, wall, stopped = D0, 0, 0, [], 0.0, None
    while D < D1 - 1e-12:
        e = e_at(D)
        a = rule.start(e, D, min(a_max, h_max, D1 - D))
        bases += 1
        wall += cost[0] if cost else 0.0
        for attempt in range(8):
            if rule.level(a) <= edge * (1 + 1e-12):
                wall += cost[1] if cost else 0.0
                break
            refusals += 1
            wall += cost[2] if cost else 0.0
            if noretry:
                continue
            a = rule.refused(a)
        else:
            stopped = D
            break
        rule.accepted(attempt == 0)
        steps.append(a)
        D += a
    return dict(bases=bases, refusals=refusals, mean_a=(D - D0) / max(1, len(steps)) if steps else 0.0, steps=steps,
                stopped=stopped, wall=wall, D_end=D)


def simulate_step(level_at, D0, D1, h_max, a_grow, a_hold, a_max=math.inf, cost=None, n_max=10 ** 9):
    """anm_run.py's own rule (a0 = h_max, halving, --a_grow, --a_hold) against an acceptance test level_at(D, a), to D1
    or n_max attempts"""
    D, a, seq, wall, bases, refusals = D0, None, [], 0.0, 0, 0
    while D < D1 - 1e-12 and len(seq) < n_max:
        room = D1 - D
        a = min(a_max, h_max, room) if a is None else min(a, a_max, h_max, room)
        bases += 1
        wall += cost[0] if cost else 0.0
        for attempt in range(4):
            if level_at(D, a):
                wall += cost[1] if cost else 0.0
                seq.append((round(D, 9), a, True))
                break
            seq.append((round(D, 9), a, False))
            refusals += 1
            wall += cost[2] if cost else 0.0
            a *= 0.5
        else:
            break
        D += a
        if not a_hold or attempt == 0:
            a = min(a_grow * a, h_max)
    return dict(seq=seq, wall=wall, bases=bases, refusals=refusals, D_end=D)


# ---------------------------------------------------------------------------------------------------------- offline
def mp_rows(parts, sl):
    """the rows sl of a vector in parts, summed exactly (smallest part first) at mp precision"""
    P = [np.asarray(p, dtype=float)[sl] for p in parts]
    return [sum((mp.mpf(float(p[i])) for p in reversed(P)), mp.mpf(0)) for i in range(len(P[0]))]


def load_parts(path):
    d = np.load(path)
    keys = [k for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in d.files]
    return [np.asarray(d[k]) for k in keys], json.loads(str(d["meta"]))


def horner_q4(Z, a):
    """the series z_0..z_N (eight parts each) summed at a by Horner's rule in octo-double and rounded to quad-double, as
    anm_run.horner_o and octo_corrector.q4_of_o form an attempt's prediction"""
    from multidouble import od_axpy, qd_add_dd
    p = [np.ascontiguousarray(q) for q in Z[-1]]
    for k in range(len(Z) - 2, -1, -1):
        p = [np.ascontiguousarray(np.asarray(q, dtype=float)) for q in od_axpy(a, *p, *Z[k])]
    tail = p[4] + (p[5] + (p[6] + p[7]))
    return [np.asarray(q) for q in qd_add_dd(p[0], p[1], p[2], p[3], tail, np.zeros(len(tail)))]


def offline(argv):
    pos, opt = parse(argv, dict(leg=LEG, mutate="none", h_max=1.0, a_grow=1.25, a_hold=1, D_max=0.0))
    if len(pos) != 1:
        raise SystemExit("usage: step_rule_check.py offline RAW [--leg LEG] [--mutate none|tail10]")
    mp.mp.dps = 150
    leg = os.path.join(pos[0], opt["leg"])
    recs = [json.loads(x) for x in open(os.path.join(leg, "anm_run.jsonl"))]
    states, preds = {}, {}
    for f in sorted(os.listdir(leg)):
        if f.startswith("anm_pin") and f.endswith(".npz"):
            parts, meta = load_parts(os.path.join(leg, f))
            states[meta["D_str"]] = (parts, meta)
        elif f.startswith("pred_pin") and f.endswith(".npz"):
            parts, meta = load_parts(os.path.join(leg, f))
            preds[meta["D_str"]] = (parts, meta)
    any_state = next(iter(states.values()))[1]
    M, nodes = int(any_state["M"]), list(any_state["nodes"])
    K = len(nodes)
    u0 = 1.0 - nodes[-1]
    ah = slice(3 * M + 4 * M * K, 4 * M + 4 * M * K)
    rows, bases = [], []
    okB, nB = True, 0
    for r in recs:
        jp = os.path.join(leg, f"jet_pin{r['pin']}_D{r['D0']:.4f}.npz")
        if not os.path.exists(jp):
            print(f"  base D {r['D0_str']}: no saved series, skipped", flush=True)
            continue
        jd = np.load(jp)
        Z8 = [[np.asarray(jd[f"z{q}"][k]) for q in range(8)] for k in range(jd["z0"].shape[0])]
        N = len(Z8) - 1
        e_true = ah_norms([z[0] for z in Z8], M, K)
        e = mutated(e_true, opt["mutate"])
        rule = ReseedRule(u0)
        rule.start(e, r["D0"], 10.0)
        bases.append(dict(D0=r["D0"], e=e, t_coef=r["t_coef"], a_max=r["a_max"]))
        zs = None
        for at in r.get("attempts", []):
            a = at["a"]
            rs = at.get("reseed") or {}
            row = dict(D0=r["D0"], a=a, D=float(at["D_str"]), accepted=bool(at["accepted"]), reseed_ok=bool(rs.get("ok")),
                       err=rule.err(a), level=rule.level(a),
                       cost=(rs.get("seconds") or 0.0) + at["t_corr"], actual=None)
            if at["D_str"] in states:
                if zs is None:
                    zs = [mp_rows(z, ah) for z in Z8]
                pred = zs[N]
                for k in range(N - 1, -1, -1):
                    pred = [p * mp.mpf(a) + c for p, c in zip(pred, zs[k])]
                tru = mp_rows(states[at["D_str"]][0], ah)
                row["actual"] = float(max(abs(p - t) for p, t in zip(pred, tru)))
            pq = preds.get(at["D_str"])
            if pq is not None and pq[1].get("a") == a:
                q4 = horner_q4(Z8, a)
                d = max(float(np.max(np.abs((np.asarray(x, dtype=float) - y)))) for x, y in zip(pq[0], q4))
                same = all(np.array_equal(np.asarray(x, dtype=float), y) for x, y in zip(pq[0], q4))
                okB = okB and d <= 1e-60
                nB += 1
                row["pred_diff"], row["pred_bitwise"] = d, same
            rows.append(row)
    # A: the estimate against the truth
    judged = [x for x in rows if x["actual"] is not None and x["err"] >= 1e-38]
    ratios = [x["actual"] / x["err"] for x in judged]
    okA = bool(judged) and all(0.5 <= q <= 2.0 for q in ratios)
    print(f"leg {opt['leg']} (mutate {opt['mutate']}): {len(rows)} attempts at {len(bases)} bases, u0 {u0:.3f}", flush=True)
    print("   D0        a       D          outcome            est. a_h err  actual       act/est  level", flush=True)
    for x in rows:
        out = "accepted" if x["accepted"] else ("re-seed failed" if not x["reseed_ok"] else "corrector failed")
        act = "" if x["actual"] is None else f"{x['actual']:.2e}"
        q = "" if x["actual"] is None or x["err"] < 1e-38 else f"{x['actual'] / x['err']:.3f}"
        print(f"   {x['D0']:9.4f} {x['a']:.4f} {x['D']:9.4f}  {out:17s}  {x['err']:.2e}     {act:11s}  {q:7s}  "
              f"{x['level']:.3g}", flush=True)
    print(f"A, the estimate: actual / estimated a_h error at {len(judged)} attempts with estimate >= 1e-38: "
          f"{min(ratios) if ratios else float('nan'):.3f} .. {max(ratios) if ratios else float('nan'):.3f} (pass in [0.5, 2]): "
          f"{'pass' if okA else 'FAIL'}", flush=True)
    okB = okB and nB > 0
    nbit = sum(1 for x in rows if x.get("pred_bitwise"))
    print(f"B, the prediction: {nB} saved predictions summed again, max difference "
          f"{max((x['pred_diff'] for x in rows if 'pred_diff' in x), default=float('nan')):.1e}, {nbit} bit for bit "
          f"(pass <= 1e-60): {'pass' if okB else 'FAIL'}", flush=True)
    # C: one level separates the re-seed's outcomes
    acc = [x["level"] for x in rows if x["reseed_ok"]]
    ref = [x["level"] for x in rows if not x["reseed_ok"]]
    s_lo, s_hi = max(acc, default=0.0), min(ref, default=math.inf)
    okC = s_lo < s_hi
    s_mid = math.sqrt(s_lo * s_hi) if okC and s_lo > 0 and s_hi < math.inf else (s_hi / 10 if s_hi < math.inf else 10 * s_lo)
    print(f"C, one level: re-seeds ok up to level {s_lo:.3g}, failed from {s_hi:.3g}; middle s_mid {s_mid:.3g} "
          f"(pass: ok below failed): {'pass' if okC else 'FAIL'}", flush=True)
    # D: the leg's own rule replayed against the edge s_mid
    bD = {round(b["D0"], 9): b for b in bases}
    first = recs[0]
    D_end = max(x["D"] for x in rows if x["accepted"])      # the leg's last state
    D_max = opt["D_max"] or D_end                            # the leg's --D_max (its last state's D for a whole leg)
    h_max = opt["h_max"]

    def e_near(D):
        return min(bases, key=lambda b: abs(b["D0"] - D))["e"]

    def level_at(D, a):
        b = bD.get(round(D, 9))
        rr = ReseedRule(u0)
        rr.start(b["e"] if b else e_near(D), D, 10.0)
        return rr.level(a) <= s_mid

    t_series = float(np.mean([b["t_coef"] for b in bases]))
    c_acc = float(np.mean([x["cost"] for x in rows if x["accepted"]]))
    c_ref = float(np.mean([x["cost"] for x in rows if not x["accepted"]] or [0.0]))
    a_grow, a_hold = opt["a_grow"], opt["a_hold"]
    rec_seq = [(round(x["D0"], 9), x["a"], x["accepted"]) for x in rows]
    rep = simulate_step(level_at, first["D0"], D_max, h_max, a_grow, a_hold, cost=(t_series, c_acc, c_ref),
                        n_max=len(rec_seq))
    okD = len(rep["seq"]) == len(rec_seq) and all(p[0] == q[0] and abs(p[1] - q[1]) <= 1e-12 and p[2] == q[2]
                                                  for p, q in zip(rep["seq"], rec_seq))
    print(f"D, the replay: the leg's rule (a0 = h_max {h_max}, a_grow {a_grow}, a_hold {a_hold}) against s_mid takes "
          f"{len(rep['seq'])} attempts, the leg {len(rec_seq)}: {'the same steps' if okD else 'DIFFERENT steps'}: "
          f"{'pass' if okD else 'FAIL'}", flush=True)
    # the steps --a_rule reseed would take, at N 24 and 31, against the same edge
    sums = lambda n: sum(j + 1 for j in range(n + 1))  # noqa: E731
    N0 = len(bases[0]["e"]) - 1
    span = D_end - first["D0"]
    print(f"steps against the edge s_mid over D {first['D0']:.3f}-{D_end:.3f}; series {t_series:.0f} s at N {N0} (scaled by "
          f"sum(j+1), {sums(N0)} at N {N0}), accepted attempt {c_acc:.0f} s, refused {c_ref:.0f} s:", flush=True)
    print(f"   leg's rule (recorded): {rep['bases']} bases, {rep['refusals']} refusals, mean step {span / rep['bases']:.3f}, "
          f"{rep['wall'] / span:.0f} s per unit D", flush=True)
    base_wall = rep["wall"] / span
    summary = dict(leg=opt["leg"], mutate=opt["mutate"], A=okA, B=okB, C=okC, D=okD, s_lo=s_lo, s_hi=s_hi, s_mid=s_mid,
                   ratio_range=[min(ratios, default=None), max(ratios, default=None)], per_unit_D=dict(leg=base_wall))
    for N, model in ((N0, "saved"), (31, "linear"), (31, "constant")):
        def e_model(D, N=N, model=model):
            e = e_near(D)
            return e if N == len(e) - 1 else extend(e, N, linear=(model == "linear"))
        cost = (t_series * sums(N) / sums(N0), c_acc, c_ref)
        sim = simulate_rule(e_model, first["D0"], D_end, s_mid, u0, h_max=5.0, scale=1.0, cost=cost)
        edge_a = ReseedRule(u0, s_mid)
        a_edge = edge_a.start(e_model(first["D0"] + span / 2), first["D0"] + span / 2, 5.0)
        w = sim["wall"] / (sim["D_end"] - first["D0"])
        print(f"   --a_rule reseed, N {N} ({model}): edge step {a_edge:.3f} at mid-leg, {sim['bases']} bases, "
              f"{sim['refusals']} refusals, mean step {sim['mean_a']:.3f}, {w:.0f} s per unit D, {base_wall / w:.2f}x the leg's",
              flush=True)
        summary["per_unit_D"][f"N{N}_{model}"] = w
    ok = okA and okB and okC and okD
    print(json.dumps(summary), flush=True)
    print(f"offline ({opt['mutate']}): {'pass' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------- pilot
def pilot(argv):
    from raw_out import pop_allow, require_raw
    argv = list(argv)
    pop_allow(argv)                     # --allow_unraw is require_raw's, read from sys.argv
    pos, opt = parse(argv, dict(jet=None, state=None, N=0, scale=0.0, threads=4, out="", refuse="", **{"as": ""}))
    if len(pos) != 1 or not opt["jet"] or not opt["state"] or not opt["as"] or not opt["scale"] > 0:
        raise SystemExit("usage: step_rule_check.py pilot RAW --jet JET --state STATE --as A1,A2 --scale S [--N N] "
                         "[--refuse A,..] [--threads 4] [--out OUT.jsonl]")
    from reseed import Reseeder
    raw = pos[0]
    mp.mp.dps = 50
    jd = np.load(os.path.join(raw, opt["jet"]))
    jmeta = json.loads(str(jd["meta"]))
    _, smeta = load_parts(os.path.join(raw, opt["state"]))
    N = opt["N"] or jmeta["N"]
    if N > jmeta["N"]:
        raise SystemExit(f"--N {N}: the series holds orders up to {jmeta['N']}")
    Z8 = [[np.ascontiguousarray(jd[f"z{q}"][k]) for q in range(8)] for k in range(N + 1)]
    M, nodes = int(smeta["M"]), list(smeta["nodes"])
    u0 = 1.0 - nodes[-1]
    D = mp.mpf(jmeta["D_str"])
    if abs(float(D) - float(smeta["D"])) > 1e-9:
        raise SystemExit(f"the series' D {jmeta['D_str']} is not the state's {smeta['D_str']}")
    rule = ReseedRule(u0, opt["scale"])
    rule.start(ah_norms([z[0] for z in Z8], M, len(nodes)), D, 10.0)
    a_rule = rule.solve(opt["scale"], 10.0)
    out = opt["out"]
    if out:
        require_raw(out)
    refuse = [float(x) for x in opt["refuse"].split(",") if x]
    print(f"pilot: series {opt['jet']} at D {jmeta['D_str']}, N {N} of {jmeta['N']}, e_N+1 {rule.e_next:.2e} "
          f"(ratio {rule.rho:.2f}); the rule's step at level {opt['scale']:.3g}: a {a_rule:.4f}", flush=True)
    ok = True
    trials = sorted(set([float(x) for x in opt["as"].split(",") if x] + [a_rule])) + refuse
    for a in trials:
        Dn = D + mp.mpf(a)
        pred4 = horner_q4(Z8, a)
        rs = Reseeder(dict(M=M, nodes=nodes, kt=smeta.get("kt", 36), kc=smeta.get("kc", 146), ks=smeta.get("ks", 130)), Dn,
                      threads=opt["threads"], qd=True, od=0, qd_start="pred")
        _, diag = rs.reseed(tuple(pred4))
        level = rule.level(a)
        want = "refuse" if a in refuse else ("accept" if level <= opt["scale"] * (1 + 1e-9) else "either")
        good = (want == "either") or (want == "accept" and diag["ok"]) or (want == "refuse" and not diag["ok"])
        ok = ok and good
        r0 = diag["rungs"][0] if diag["rungs"] else {}
        rec = dict(D0=str(D), a=a, D=str(Dn), N=N, err=rule.err(a), level=level, scale=opt["scale"], want=want,
                   ok=bool(diag["ok"]), its=r0.get("its"), R=r0.get("R"), seconds=diag.get("seconds"), good=good)
        print(f"  a {a:.4f} -> D {mp.nstr(Dn, 12)}: est. a_h error {rule.err(a):.2e}, level {level:.3g}, re-seed "
              f"{'ok' if diag['ok'] else 'FAILED'} ({r0.get('its')} its, |R| {r0.get('R', float('nan')):.2e}, "
              f"{diag.get('seconds')} s); want {want}: {'pass' if good else 'FAIL'}", flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    print(f"pilot: {'pass' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    cmds = dict(planted=planted, offline=offline, pilot=pilot)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    sys.exit(cmds[sys.argv[1]](sys.argv[2:]))
