#!/usr/bin/env python3
"""Where a weak second singular set near P0 + kappa_* has to be: a lattice of positions, lengths and exponents with the
zero factor at the real zero (issue #718).

#703 (pade_second_zero.py; notes/PADE_FAR_END.md 15) found 35 plants that pass (a) and (c') at all three nodes, each with
a segment of length 0.3 and near-end exponent 4 ending 0.25 past P0 + kappa_* (|Q| = 2.48-2.72 at wf 1.00). Its grid
had one segment shape and nine end points. Here the end point Q of the second set moves over a lattice around
P0 + kappa_*, the segment's length L and near-end exponent P vary, and each plant is judged with #703's frozen judge,
unchanged (pade_second_zero.evaluate, judge, relocated, judge_reloc).

Plants. #703's (pade_second_zero.ZMain, pade_second_set.make_plant): pref (L_main + lam L_2) / S_0 with pref at ZLOC, L_2
the density s^P (1-s)^alpha on the segment from Q - L Q/|Q| to Q. Position. Q = P0 + kappa_* + 0.25 (a + ib) at each
node, a = -1..4, b = -2..4 (the lattice, 42 positions); the diagonal points Q + 0.25 e^(i (2k+1) pi/4) of stage 2.
Groups (kind, L, P): (chord2, 0.3, 4), (arc, 0.3, 4), (chord2, 0.15, 4), (chord2, 0.6, 4), (chord2, 0.3, 2), (chord2,
0.3, 6); alpha = 0 or beta - 1; phi = 0, 90, 180, 270 degrees.

Strength, from the (c') window. On #703's grid chi_22 / chi_22(real) in zk/chord is eta times a constant g per curve and
alpha, the same to 3 digits over phases and kinds, since the main density lies in zk/chord's span. So per position, L, P
and alpha, g is measured on one chord2 plant at eta = 1e-2, phi = 0 (wf 1.00), and the (c') window in eta is
W = [0.1/g, 10/g] cut to [1e-6, 0.2]. If W is not empty, eta takes the log-midpoints of 3 equal bins of W. If it is,
eta = 0.2 (the cap: the set is too weak for (c') at every allowed eta) or 1e-6 (the floor: too strong). Control R8 checks
on every plant at wf 1.00 that chi_22 / chi_22(real) = eta g within 1e-2, and that every plant inside a window passes (c').

Verdict. A plant passes if (a) and (c') hold at wf 0.99, 1.00 and 1.01. A position passes for a group if one of its plants
does, and passes if it passes for some group. Each plant runs at wf 1.00; per position and group the plants that pass (a)
and (c') there, smallest D first, run at 0.99 and then 1.01 until one passes at all three nodes. A lattice position is
located if it passes, its four lattice neighbours 0.25 away pass for no group, and its four diagonal points 0.25 away (run
in stage 2 for such candidates only) pass for no group. A passing position whose lattice neighbour lies outside the
lattice cannot be located: the region is open there. Apart, the same with (c'') in place of (c'), on the same plants;
(c'') is evaluated at a node only for a plant that passes (a) there.

    python3 pade_second_ring.py --selftest [--mutate=secmom|len|window]
    python3 pade_second_ring.py pilot OUTDIR --ref703=I703-mac/grid.jsonl [--mutate=len|window] [--allow_unraw]
    python3 pade_second_ring.py ring OUTDIR [--allow_unraw]     (reads OUTDIR/control.json, OUTDIR/pilot.json; resumes)
    python3 pade_second_ring.py report OUTDIR [--allow_unraw]   (post-freeze summary of the ring's files, not judged)

OUTDIR/control.json is pade_second_zero.py control's. Exit 1 when a frozen control fails or the gate forbids a stage.
Numerical evidence on plants, not a bound.
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_second_zero as Z                                           # noqa: E402

ss, te, tc, pg, pp = Z.ss, Z.te, Z.tc, Z.pg, Z.pp
N23, NODES, WF0 = Z.N23, Z.NODES, Z.WF0
JUDGED_SCAN, NEED_DA, TOL_POLE, CHI_FACTOR = Z.JUDGED_SCAN, Z.NEED_DA, Z.TOL_POLE, Z.CHI_FACTOR
STEP = ss.SEG_STEP                  # 0.25, #667's lattice step and the issue's distance
LAT_A = tuple(range(-1, 5))
LAT_B = tuple(range(-2, 5))
DIAG = (1, 3, 5, 7)                 # Q + 0.25 e^(i k pi/4)
GROUPS = (("chord2", 0.3, 4), ("arc", 0.3, 4), ("chord2", 0.15, 4), ("chord2", 0.6, 4), ("chord2", 0.3, 2),
          ("chord2", 0.3, 6))
ALPHAS, PHASES = ss.ALPHAS, ss.PHASES
ETA_MIN, ETA_MAX, ETA_PILOT, N_ETA = 1e-6, 0.2, 1e-2, 3
LIN_TOL = 1e-2
R6_TOL = 1e-12
FROZEN = {                          # written before the pilot and ring stages were first run (PADE_FAR_END.md 16.2)
    "plants": "#703's plants (pade_second_zero.ZMain, pade_second_set.make_plant), zero at ZLOC, main c_m refitted; the "
              "second set's density s^P (1-s)^alpha on the segment from Q - L Q/|Q| to Q; nothing else changed",
    "positions": "Q = P0 + kappa_* + 0.25 (a + ib) per node, a = -1..4, b = -2..4 (42 lattice positions); stage 2: "
                 "Q + 0.25 e^(i k pi/4), k = 1, 3, 5, 7, around each candidate",
    "groups": "(kind, L, P) = (chord2, 0.3, 4), (arc, 0.3, 4), (chord2, 0.15, 4), (chord2, 0.6, 4), (chord2, 0.3, 2), "
              "(chord2, 0.3, 6); alpha 0 or beta - 1; phi 0, 90, 180, 270",
    "eta": "g = (chi_22 / chi_22(real) in zk/chord) / eta of the chord2 plant at eta 1e-2, phi 0, wf 1.00, per position, "
           "L, P, alpha; W = [0.1/g, 10/g] cut to [1e-6, 0.2]; eta = the log-midpoints of 3 equal bins of W; if W is "
           "empty, eta = 0.2 when 0.1/g >= 0.2 (cap), else 1e-6 (floor); the same eta at every node",
    "judge": "#703's frozen judge, unchanged (pade_second_zero.evaluate: (a) m_DA <= 5, (b) d_P <= 0.03, (c') in "
             "zk/chord M_min = 23 and 0.1 <= chi_22 / chi_22(real) <= 10; (c'') by pade_second_zero.relocated and "
             "judge_reloc). (c'') is evaluated at a node only for a plant that passes (a) there, and only if control's "
             "R5 passed",
    "nodes": "every plant at wf 1.00; per position and group, the plants passing (a) and (c') at wf 1.00, smallest D "
             "first, run at 0.99 and then 1.01 until one passes (a) and (c') at all three nodes; apart, the same for "
             "(a) and (c''), smallest D'' first",
    "located": "a plant passes if (a) and (c') hold at all three nodes; a position passes for a group if one of its "
               "plants passes, and passes if it passes for some group; a lattice position is located if it passes, "
               "its 4 lattice neighbours (a +- 1, b), (a, b +- 1) exist in the lattice and pass for no group, and its 4 "
               "diagonal points (stage 2) pass for no group; otherwise no position is located and the region is not "
               "bounded by this rule, reported with the passing positions, |Q| and arg Q, and the lattice edges they "
               "touch. Apart, the same with (c'')",
    "stage2": "candidates: lattice positions that pass and whose 4 lattice neighbours exist and fail; their diagonal "
              "points run with every group, the eta rule and the node rule above",
    "R6": "pilot: 12 of #703's plants (seg(+1, +1), seg(0, 0), seg(-1, -1) x chord2, arc x alpha 0, beta - 1, eta 1e-1, "
          "phi 90) through this path at wf 1.00 against I703-mac/grid.jsonl: m_DA, M_min and the judge's (a), (b), (c') "
          "equal; d_P and chi_22 / chi_22(real) within 1e-12 relative; (c'')'s ratio within 1e-9 where both have it",
    "R7": "selftest: the second set's exact moments against mpmath.quad for (L, P) = (0.15, 4), (0.6, 4), (0.3, 2), "
          "(0.3, 6), alpha 0 and beta - 1, at a lattice and a diagonal position, k = 0, 7, 15, 22, 31: rel diff < 1e-40; "
          "the lattice curves with L = 0.3 equal pade_second_set.second_curve's bit for bit for #667's nine seg; the "
          "diagonal offsets 0.25 from their centre to 1e-50; the eta rule on planted g (7.79, 0.49, 1e8) gives a "
          "window whose eta g lie in [0.1, 10], the cap, and the floor",
    "R8": "pilot and ring: every plant at wf 1.00 has |chi_22 / chi_22(real) / (eta g) - 1| <= 1e-2, and every plant "
          "inside a window passes (c') at wf 1.00; the pilot checks it on Q = P0 + kappa_* + 0.25 (1 + i), (chord2, 0.3, "
          "4), alpha 0, phi 0",
    "J": "pilot: J+ the real series judged through this path passes (a), (b), (c') at every node; J- the eta = 0 chord2 "
         "and arc plants at wf 1.00 fail the judge, M_min = 6 in their own kappa_* basis. #703's control "
         "(pade_second_zero.py control and its mutations) is rerun into OUTDIR first",
    "gate": "pilot runs only if control.json's controls pass; ring runs only if the pilot passes (R6, R8, J+, J-)",
    "mutations": "--selftest --mutate=secmom (alpha + 1e-10 in the exact moments): R7's moments fail; --selftest "
                 "--mutate=len (L x (1 + 1e-6)): R7's curves fail; --selftest --mutate=window (g x 100 in the eta "
                 "rule): R7's eta rule fails; pilot --mutate=len: R6 fails; pilot --mutate=window: R8 fails. Each "
                 "exits 1",
    "C1": "#672's amended C1 on every plant scan; a failure is listed and the ring stage exits 1",
    "context": "per plant: the record of pade_second_zero.evaluate (the other 15 scans' chi_22 and M_min kept), the "
               "relocation; per position: |Q|, arg Q, g, the window; group-resolved maps; (b) and d_P. Reported, not "
               "judged",
}


# ---------------------------------------------------------------- positions and curves
def pos_id(pos):
    a, b, k = pos
    return f"L{a:+d}{b:+d}" + (f"d{k}" if k is not None else "")


def offset(pos):
    """Q - (P0 + kappa_*): 0.25 (a + ib), plus 0.25 e^(i k pi/4) for a diagonal point"""
    a, b, k = pos
    with mp.workdps(pg.DPS_PLANT):
        o = mp.mpf(STEP) * mp.mpc(a, b)
        if k is not None:
            o = o + mp.mpf(STEP) * mp.expjpi(mp.mpf(k) / 4)
        return o


def ring_curve(main_arc, pos, L, mut=None):
    """the segment of length L pointing radially inward and ending at Q; as pade_second_set.second_curve for L = 0.3"""
    with mp.workdps(pg.DPS_PLANT):
        P0, ks = mp.mpc(main_arc.P0), mp.mpc(main_arc.P1)
        Q = P0 + ks + offset(pos)
        ln = mp.mpf(L) * (1 + mp.mpf("1e-6")) if mut == "len" else mp.mpf(L)
        return SimpleNamespace(P0=Q - ln * Q / abs(Q), P1=Q, e0=mp.mpc(0), e1=mp.mpc(0), kind="chord")


def eta_rule(g, mut=None):
    """(kind, etas): the (c') window [0.1/g, 10/g] cut to [ETA_MIN, ETA_MAX], sampled at the log-midpoints of N_ETA bins"""
    if mut == "window":
        g = g * 100
    lo, hi = max(1 / (CHI_FACTOR * g), ETA_MIN), min(CHI_FACTOR / g, ETA_MAX)
    if lo < hi:
        return "window", [lo * (hi / lo) ** ((j + 0.5) / N_ETA) for j in range(N_ETA)]
    if 1 / (CHI_FACTOR * g) >= ETA_MAX:
        return "cap", [ETA_MAX]
    return "floor", [ETA_MIN]


def plant_key(kind, pos, L, P, a, eta, ph):
    return f"{kind}|{pos_id(pos)}|L{L}|P{P}|{a}|eta{eta:.6e}|ph{ph}"


def compact(r):
    """the record kept: the judged scan and nozero/chord in full, the other scans' M_min, chi_22 and C1"""
    tw = r["twin"]
    r["twin"] = {s: (v if s in (JUDGED_SCAN, "nozero/chord") else
                     {"M_min": v["M_min"], "chi22": v["chi"][22], "C1_pass": v["C1_pass"]}) for s, v in tw.items()}
    return r


# ---------------------------------------------------------------- one node's setup and one plant
class Node:
    def __init__(self, wf):
        self.wf = wf
        self.real = tc.real_sig()[wf]
        self.rel, _ = tc.err_model(wf)
        self.projs, _ = Z.projections(wf)
        self.ref, self.real_tw = Z.real_ref(self.projs, wf)
        self.mains = {k: Z.ZMain(wf, k, self.real, Z.zloc(wf)) for k in ss.KINDS}
        self.parts = {}
        self.zf = None

    def part(self, kind, pos, L, P, a, mut=None):
        key = (kind, pos, L, P, a, mut)
        if key not in self.parts:
            main = self.mains[kind]
            cur = ring_curve(main.base.arc, pos, L, mut)
            with mp.workdps(pg.DPS_PLANT):
                al = ss.alpha_of(self.wf, a)
            self.parts[key] = (ss.second_part(main, cur, al, p=P), complex(cur.P1))
        return self.parts[key]

    def g(self, pos, L, P, a, mut=None):
        """chi_22 / chi_22(real) in zk/chord per unit eta, from the chord2 plant at ETA_PILOT, phi 0"""
        w, _ = self.part("chord2", pos, L, P, a, mut)
        sig, _ = ss.make_plant(self.mains["chord2"], w, ETA_PILOT, 0)
        rows = self.projs[JUDGED_SCAN][0].rows(sig, Ms=(22,))
        return rows[22][0] / self.ref[JUDGED_SCAN][0] / ETA_PILOT


def evaluate_plant(node, kind, pos, L, P, a, eta, ph, reloc_ok, mut=None):
    """#703's evaluate and judge, unchanged; (c'') only when (a) passes"""
    t0 = time.time()
    w, Q = node.part(kind, pos, L, P, a, mut)
    sig, info = ss.make_plant(node.mains[kind], w, eta, ph)
    r = Z.evaluate(sig, node.wf, node.projs, node.ref)
    j = r["judge"]
    if reloc_ok and j["a"]:
        l2z = r["twin"][JUDGED_SCAN]["l2"][20]
        if l2z < 0.1 * node.ref[JUDGED_SCAN][0]:
            rl = {"skipped": True, "bound_ratio": l2z / node.ref[JUDGED_SCAN][0]}
        else:
            if node.zf is None:
                node.zf = Z.ZeroFit(node.wf)
            rl = Z.relocated(node.wf, node.zf, sig[:N23], node.rel, Z.zloc(node.wf))
        j["c2"], j["ratio2"] = Z.judge_reloc(j, rl, node.ref)
        if j["ratio2"] is not None:
            j["D2"] = max(j["m_DA"] / (NEED_DA - 1), j["d_P"] / TOL_POLE, abs(math.log10(j["ratio2"])))
        else:
            j["D2"] = max(j["m_DA"] / (NEED_DA - 1), j["d_P"] / TOL_POLE, abs(math.log10(rl["bound_ratio"])))
        r["reloc"] = rl
    else:
        j["c2"], j["ratio2"], j["D2"] = (False if reloc_ok else None), None, None
        r["reloc"] = {"not_evaluated": "(a) fails" if reloc_ok else "R5 failed"}
    j["all2"] = j["a"] and j["b"] and bool(j["c2"])
    r.update({"id": plant_key(kind, pos, L, P, a, eta, ph), "wf": node.wf, "kind": kind, "pos": pos_id(pos),
              "a_b_k": list(pos), "L": L, "P": P, "alpha": a, "eta": eta, "phase": ph, "Q": Q, "plant": info,
              "seconds": time.time() - t0})
    return json.loads(json.dumps(pg.jsonable(compact(r))))


def plant_line(r):
    j, rl = r["judge"], r["reloc"]
    rt = ""
    if "skipped" in rl:
        rt = f" | reloc skipped ({rl['bound_ratio']:.1e})"
    elif "zero" in rl:
        rt = f" | reloc dz {rl['dz_start']:.1e} ratio {j['ratio2']:.2e} c'' {j['c2']!s:5s}"
    return (f"wf {r['wf']} {r['id']:52s} M {r['plant']['M']} | m_DA {j['m_DA']:2d} d_P {float(j['d_P']):.3f} Mmin "
            f"{j['M_min']} ratio {j['ratio']:.2e} -> a {j['a']!s:5s} b {j['b']!s:5s} c' {j['c']!s:5s}{rt}"
            f"{'' if j['C1_pass'] else ' C1 BAD'} ({r['seconds']:.1f}s)")


# ---------------------------------------------------------------- selftest (R7)
def selftest(mut=None):
    ok = True
    wf = WF0
    real = tc.real_sig()
    main = ss.Main(wf, "arc", real[wf])
    # 1: exact moments of the new shapes against mpmath.quad
    if mut in (None, "secmom"):
        worst = 0
        for pos in ((2, 2, None), (0, 0, 1)):
            for L, P in ((0.15, 4), (0.6, 4), (0.3, 2), (0.3, 6)):
                cur = ring_curve(main.base.arc, pos, L)
                for a in ALPHAS:
                    with mp.workdps(pg.DPS_PLANT):
                        al = ss.alpha_of(wf, a)
                        ex = ss.sec_moments(cur, al, P, 32, mut)
                        e = max(abs(ex[k] - ss.sec_moments_quad(cur, al, P, k)) / abs(ex[k]) for k in (0, 7, 15, 22, 31))
                    worst = max(worst, float(e))
        print(f"selftest 1: exact moments of (L, P) = (0.15, 4), (0.6, 4), (0.3, 2), (0.3, 6), alpha 0 and beta - 1, at "
              f"{pos_id((2, 2, None))} and {pos_id((0, 0, 1))}, against mpmath.quad at k = 0, 7, 15, 22, 31: max rel diff "
              f"{worst:.1e} (pass < 1e-40)")
        ok &= worst < 1e-40
    # 2: lattice curves at L = 0.3 against pade_second_set.second_curve, and the diagonal offsets
    same = True
    arcs = {"chord2": ss.Main(wf, "chord2", real[wf]).base.arc, "arc": main.base.arc}
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for kind, arc in arcs.items():
                c1 = ring_curve(arc, (a, b, None), ss.SEG_LEN, mut)
                c2 = ss.second_curve(wf, kind, f"seg{a:+d}{b:+d}", arc)
                same &= c1.P0 == c2.P0 and c1.P1 == c2.P1
    with mp.workdps(pg.DPS_PLANT):
        dd = max(abs(abs(offset((1, 1, k)) - offset((1, 1, None))) - mp.mpf(STEP)) for k in DIAG)
    print(f"selftest 2: lattice curves with L = 0.3 equal pade_second_set.second_curve bit for bit (9 seg x 2 kinds): "
          f"{same}; diagonal points 0.25 from their centre to {float(dd):.1e} (pass < 1e-50)")
    ok &= same and dd < mp.mpf("1e-50")
    # 3: the eta rule on planted g
    good = True
    for g, want in ((7.79, "window"), (0.49, "cap"), (1e8, "floor")):
        kind, etas = eta_rule(g, mut)
        inside = all(0.1 <= e * g <= 10 and ETA_MIN <= e <= ETA_MAX for e in etas) if want == "window" else True
        print(f"selftest 3: eta rule at g = {g:g}: {kind} {['%.3e' % e for e in etas]}; eta g "
              f"{['%.3f' % (e * g) for e in etas]} (pass: {want}" + (", eta g in [0.1, 10])" if want == "window" else ")"))
        good &= kind == want and inside
    ok &= good
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- pilot (R6, R8, J+, J-)
def run_pilot(outdir, ref703, mut=None):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    if not ctl["controls_pass"]:
        print("the controls failed: no pilot (frozen rule)")
        return False
    reloc_ok = ctl["relocation_pass"]
    log = Z.tz.Log(outdir, "pilot", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "R6": [], "R8": [], "J": {}}
    ok = {"R6": True, "R8": True, "J+": True, "J-": True}
    nodes = {}
    for wf in NODES:
        t0 = time.time()
        nodes[wf] = Node(wf)
        log(f"node {wf}: set up in {time.time() - t0:.0f}s; the real chi_22 in zk/chord {nodes[wf].ref[JUDGED_SCAN][0]:.6f}")
    # J+ and J- through this path
    for wf in NODES:
        n = nodes[wf]
        rr = Z.evaluate(n.real, wf, n.projs, n.ref)
        ok["J+"] &= rr["judge"]["all"]
        rec["J"][f"J+ {wf}"] = rr["judge"]
        log(f"J+ wf {wf}: the real series: m_DA {rr['judge']['m_DA']}, d_P {rr['judge']['d_P']:.1e}, ratio "
            f"{rr['judge']['ratio']:.3g} -> (a) {rr['judge']['a']} (b) {rr['judge']['b']} (c') {rr['judge']['c']}")
    n = nodes[WF0]
    for kind, own in (("chord2", "kstar/chord"), ("arc", "kstar/smooth")):
        sig, _ = ss.make_plant(n.mains[kind], None, 0, 0)
        pr = Z.evaluate(sig, WF0, n.projs, n.ref)
        jm = (not pr["judge"]["all"]) and pr["twin"][own]["M_min"] == 6
        ok["J-"] &= jm
        rec["J"][f"J- {kind}"] = {"judge": pr["judge"], "M_min_own": pr["twin"][own]["M_min"]}
        log(f"J- wf {WF0} {kind} eta 0: m_DA {pr['judge']['m_DA']}, d_P {pr['judge']['d_P']:.3f}, M_min in {own} "
            f"{pr['twin'][own]['M_min']} -> {'OK' if jm else 'BAD'}")
    # R6: #703's plants through this path
    old = {}
    for line in open(ref703):
        r = json.loads(line)
        if r["wf"] == WF0:
            old[r["id"]] = r
    for (a, b) in ((1, 1), (0, 0), (-1, -1)):
        for kind in ss.KINDS:
            for al in ALPHAS:
                r = evaluate_plant(n, kind, (a, b, None), ss.SEG_LEN, ss.P_INNER, al, 1e-1, 90, reloc_ok, mut)
                o = old[ss.plant_id(kind, f"seg{a:+d}{b:+d}", al, 1e-1, 90)]
                j, jo = r["judge"], o["judge"]
                dv = {"d_P": abs(j["d_P"] / jo["d_P"] - 1), "ratio": abs(j["ratio"] / jo["ratio"] - 1)}
                same = (j["m_DA"] == jo["m_DA"] and j["M_min"] == jo["M_min"]
                        and all(j[x] == jo[x] for x in ("a", "b", "c")) and max(dv.values()) <= R6_TOL)
                if j.get("ratio2") is not None and jo.get("ratio2") is not None:
                    dv["ratio2"] = abs(j["ratio2"] / jo["ratio2"] - 1)
                    same &= dv["ratio2"] <= 1e-9
                ok["R6"] &= same
                rec["R6"].append({"id": r["id"], "dev": dv, "pass": same})
                log(f"R6 {r['id']}: m_DA {j['m_DA']} (#703 {jo['m_DA']}), ratio {j['ratio']:.6e} (#703 {jo['ratio']:.6e}); "
                    f"rel dev {max(dv.values()):.1e} (pass <= {R6_TOL:.0e}) -> {'OK' if same else 'BAD'}")
    # R8: the eta rule on one position
    pos, (kind, L, P), al = (1, 1, None), GROUPS[0], "log"
    g = n.g(pos, L, P, al)
    wk, etas = eta_rule(g, mut)
    log(f"R8 pilot {pos_id(pos)} {kind} L {L} P {P} {al}: g {g:.4g}, {wk} {['%.3e' % e for e in etas]}")
    for eta in etas:
        r = evaluate_plant(n, kind, pos, L, P, al, eta, 0, reloc_ok)
        j = r["judge"]
        lin = abs(j["ratio"] / (eta * g) - 1)
        good = lin <= LIN_TOL and (wk != "window" or j["c"])
        ok["R8"] &= good
        rec["R8"].append({"id": r["id"], "lin": lin, "c": j["c"], "pass": good})
        log(plant_line(r))
        log(f"R8 {r['id']}: ratio / (eta g) - 1 = {lin:.1e} (pass <= {LIN_TOL:.0e}), (c') {j['c']} (pass: True) -> "
            f"{'OK' if good else 'BAD'}")
    rec["checks"] = ok
    if mut == "len":
        rec["mutation_fired"] = not ok["R6"]
    elif mut == "window":
        rec["mutation_fired"] = not ok["R8"]
    if mut:
        log(f"mutation {mut}: fired as frozen: {rec['mutation_fired']} (must)")
    rec["pilot_pass"] = all(ok.values()) and not mut
    log(f"pilot {ok} -> {rec['pilot_pass']}")
    log.dump(rec)
    return rec["pilot_pass"]


# ---------------------------------------------------------------- the ring
def run_ring(outdir):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    pil = json.load(open(os.path.join(outdir, "pilot.json")))
    if not (ctl["controls_pass"] and pil["pilot_pass"]):
        print("the controls or the pilot failed: no ring run (frozen rule)")
        return False
    reloc_ok = ctl["relocation_pass"]
    log = Z.tz.Log(outdir, "ring", None)
    path = os.path.join(outdir, "ring.jsonl")
    done, gs = {}, {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("type") == "g":
                gs[(r["pos"], r["L"], r["P"], r["alpha"])] = r
            else:
                done[(r["id"], r["wf"])] = r
    jl = open(path, "a")
    nodes = {}

    def node(wf):
        if wf not in nodes:
            nodes[wf] = Node(wf)
        return nodes[wf]

    def write(r):
        jl.write(json.dumps(r) + "\n")
        jl.flush()

    def run(wf, kind, pos, L, P, a, eta, ph):
        key = (plant_key(kind, pos, L, P, a, eta, ph), wf)
        if key not in done:
            r = evaluate_plant(node(wf), kind, pos, L, P, a, eta, ph, reloc_ok)
            write(r)
            done[key] = r
            log(plant_line(r))
        return done[key]

    def windows(pos):
        out = {}
        for L, P in dict.fromkeys((g[1], g[2]) for g in GROUPS):
            for a in ALPHAS:
                k = (pos_id(pos), L, P, a)
                if k not in gs:
                    g = node(WF0).g(pos, L, P, a)
                    wk, etas = eta_rule(g)
                    _, Q = node(WF0).part("chord2", pos, L, P, a)
                    gs[k] = {"type": "g", "pos": pos_id(pos), "L": L, "P": P, "alpha": a, "g": g, "window": wk,
                             "etas": etas, "Q": [Q.real, Q.imag]}
                    write(gs[k])
                out[(L, P, a)] = gs[k]
        return out

    verdict = {}

    def do_position(pos):
        """every plant at wf 1.00, then the node rule per group; returns {group: {"c": plant or None, "c2": ...}}"""
        t0 = time.time()
        win = windows(pos)
        res = {}
        for grp in GROUPS:
            kind, L, P = grp
            plants = [(kind, pos, L, P, a, eta, ph) for a in ALPHAS for eta in win[(L, P, a)]["etas"] for ph in PHASES]
            recs = [run(WF0, *p) for p in plants]
            gres = {}
            for crit, dkey in (("c", "D"), ("c2", "D2")):
                if crit == "c2" and not reloc_ok:
                    continue
                cand = sorted([(float(r["judge"][dkey]), i, p) for i, (p, r) in enumerate(zip(plants, recs))
                               if r["judge"]["a"] and r["judge"][crit]])
                hit = None
                for _, _, p in cand:
                    if all((lambda r: r["judge"]["a"] and r["judge"][crit])(run(wf, *p)) for wf in (0.99, 1.01)):
                        hit = plant_key(*p)
                        break
                gres[crit] = {"pass": hit is not None, "plant": hit, "n_wf1": len(cand)}
            res["/".join(map(str, grp))] = gres
        union = {crit: any(v[crit]["pass"] for v in res.values() if crit in v) for crit in ("c", "c2")}
        q = win[(0.3, 4, "log")]["Q"]
        verdict[pos_id(pos)] = {"pos": list(pos), "Q": q, "absQ": abs(complex(*q)),
                                "argQ_deg": math.degrees(math.atan2(q[1], q[0])), "groups": res, "pass": union,
                                "windows": {f"{L}/{P}/{a}": (w["window"], w["g"]) for (L, P, a), w in win.items()}}
        log(f"position {pos_id(pos)} Q {complex(*q):.4f} |Q| {abs(complex(*q)):.3f}: (a)+(c') "
            f"{'PASS' if union['c'] else 'fail'}, (a)+(c'') {'PASS' if union['c2'] else 'fail'}; groups (c') "
            + ", ".join(f"{g} {'P' if v['c']['pass'] else '-'}{v['c']['n_wf1']}" for g, v in res.items())
            + f" ({time.time() - t0:.0f}s)")
        log.dump({"frozen": FROZEN, "positions": verdict})
        return union

    lattice = sorted(((a, b, None) for a in LAT_A for b in LAT_B), key=lambda p: (math.hypot(p[0] - 0.5, p[1] - 0.5),
                                                                                  p[0], p[1]))
    log(f"ring: {len(lattice)} lattice positions, {len(GROUPS)} groups, {len(done)} plant records and {len(gs)} g records "
        f"already in ring.jsonl; (c'') {'on' if reloc_ok else 'off (R5 failed)'}")
    for pos in lattice:
        do_position(pos)
    log("lattice done")
    lat = {(p[0], p[1]): verdict[pos_id(p)]["pass"] for p in lattice}
    out = {}
    for crit in ("c", "c2"):
        passing = [ab for ab, v in lat.items() if v[crit]]
        nb = {ab: [(ab[0] + 1, ab[1]), (ab[0] - 1, ab[1]), (ab[0], ab[1] + 1), (ab[0], ab[1] - 1)] for ab in passing}
        edge = [ab for ab in passing if any(n not in lat for n in nb[ab])]
        cands = [ab for ab in passing if ab not in edge and not any(lat[n][crit] for n in nb[ab])]
        out[crit] = {"passing": passing, "edge": edge, "candidates": cands}
    for crit in ("c", "c2"):
        loc = []
        for ab in out[crit]["candidates"]:
            diag = [do_position((ab[0], ab[1], k))[crit] for k in DIAG]
            if not any(diag):
                loc.append(ab)
        out[crit]["located"] = loc
    # R8 over every plant at wf 1.00, and C1
    lin_bad, win_bad = [], []
    gmap = {(v["pos"], v["L"], v["P"], v["alpha"]): v for v in gs.values()}
    for (pid, wf), r in done.items():
        if wf != WF0:
            continue
        gv = gmap[(r["pos"], r["L"], r["P"], r["alpha"])]
        if abs(r["judge"]["ratio"] / (r["eta"] * gv["g"]) - 1) > LIN_TOL:
            lin_bad.append(pid)
        if gv["window"] == "window" and not r["judge"]["c"]:
            win_bad.append(pid)
    c1_fail = [f"{k[0]} wf {k[1]}" for k, r in done.items() if not r["judge"]["C1_pass"]]
    r8 = not lin_bad and not win_bad
    for crit, lab in (("c", "(a)+(c')"), ("c2", "(a)+(c'')")):
        o = out[crit]
        log(f"{lab}: passing lattice positions {len(o['passing'])} of {len(lattice)}: "
            + ", ".join(pos_id((a, b, None)) for a, b in sorted(o["passing"])))
        log(f"{lab}: on the lattice edge {[pos_id((a, b, None)) for a, b in o['edge']]}; candidates "
            f"{[pos_id((a, b, None)) for a, b in o['candidates']]}; located {[pos_id((a, b, None)) for a, b in o['located']]}")
    log(f"R8: linearity failures {len(lin_bad)}, windowed plants failing (c') {len(win_bad)} -> {'OK' if r8 else 'BAD'}; "
        f"C1 failures: {c1_fail or 'none'}; plant records {len(done)}")
    log.dump({"frozen": FROZEN, "positions": verdict, "verdict": pg.jsonable({k: {kk: [list(x) for x in vv]
                                                                                for kk, vv in v.items()}
                                                                            for k, v in out.items()}),
              "R8": {"pass": r8, "lin_bad": lin_bad, "win_bad": win_bad}, "C1_fail": c1_fail,
              "n_records": len(done)})
    return r8 and not c1_fail


# ---------------------------------------------------------------- report (written after the freeze; judges nothing)
def run_report(outdir):
    """post-freeze summary of OUTDIR/ring.jsonl and ring.json: the position maps, per-position g and windows, counts at
    wf 1.00, and the ranges of the passing plants. Every verdict is the ring stage's; nothing here is judged"""
    log = Z.tz.Log(outdir, "report", None)
    recs, gs = {}, {}
    for line in open(os.path.join(outdir, "ring.jsonl")):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:            # a line still being written
            continue
        if r.get("type") == "g":
            gs[(r["pos"], r["L"], r["P"], r["alpha"])] = r
        else:
            recs[(r["id"], r["wf"])] = r
    ring = json.load(open(os.path.join(outdir, "ring.json")))
    posv = ring["positions"]
    at0 = [r for (_, wf), r in recs.items() if wf == WF0]
    groups = ["/".join(map(str, g)) for g in GROUPS]

    def code(p, crit, grp=None):
        v = posv.get(p)
        if v is None:
            return "  "
        gl = [grp] if grp else groups
        if any(v["groups"][g][crit]["pass"] for g in gl if crit in v["groups"][g]):
            return "P"
        if any(v["groups"][g][crit]["n_wf1"] for g in gl if crit in v["groups"][g]):
            return "1"
        ws = [w[0] for k, w in v["windows"].items()
              if grp is None or k.startswith(f"{grp.split('/')[1]}/{grp.split('/')[2]}/")]
        return "cap" if all(w == "cap" for w in ws) else "·"

    log("Codes: P passes (a)+(crit) at all three nodes; 1 only at wf 1.00; · some (c') window, no (a)+(crit) at wf 1.00; "
        "cap every window empty (too weak at eta 0.2). Rows b, columns a; Q = P0 + kappa_* + 0.25 (a + ib)")
    for crit, lab in (("c", "(a)+(c')"), ("c2", "(a)+(c'')")):
        for grp in [None] + groups:
            log(f"map {lab}, {'union' if grp is None else grp}:")
            log("| b \\ a | " + " | ".join(f"{a:+d}" for a in LAT_A) + " |")
            log("|---|" + "---|" * len(LAT_A))
            for b in sorted(LAT_B, reverse=True):
                log(f"| {b:+d} | " + " | ".join(code(pos_id((a, b, None)), crit, grp) for a in LAT_A) + " |")
    diag = sorted(p for p in posv if "d" in p)
    if diag:
        log("stage 2 points: " + ", ".join(f"{p} {code(p, 'c')}/{code(p, 'c2')}" for p in diag))
    log("per position at wf 1.00: |Q|, arg Q, g at (0.3, 4) for alpha 0 and beta - 1, windows (cap/window/floor over the "
        "10 shape-alpha pairs), plants, (a), (a)+(c'), (a)+(c''), m_DA range over the windowed plants")
    for p in sorted(posv, key=lambda q: (posv[q]["pos"][0], posv[q]["pos"][1], str(posv[q]["pos"][2]))):
        v = posv[p]
        rs = [r for r in at0 if r["pos"] == p]
        wk = [w[0] for w in v["windows"].values()]
        ws = [r for r in rs if gs[(p, r["L"], r["P"], r["alpha"])]["window"] == "window"]
        mda = [r["judge"]["m_DA"] for r in ws]
        log(f"{p:10s} Q {v['Q'][0]:.3f}{v['Q'][1]:+.3f}i |Q| {v['absQ']:.3f} arg {v['argQ_deg']:5.1f} | g "
            f"{v['windows']['0.3/4/log'][1]:.3g}, {v['windows']['0.3/4/ex'][1]:.3g} | cap {wk.count('cap')} window "
            f"{wk.count('window')} floor {wk.count('floor')} | plants {len(rs)}, (a) {sum(r['judge']['a'] for r in rs)}, "
            f"(a)+(c') {sum(r['judge']['a'] and r['judge']['c'] for r in rs)}, (a)+(c'') "
            f"{sum(r['judge']['a'] and bool(r['judge']['c2']) for r in rs)} | m_DA "
            + (f"{min(mda)}-{max(mda)}" if mda else "-") + f" | pass {v['pass']}")
    j0 = [r["judge"] for r in at0]
    log(f"wf 1.00, {len(at0)} plants: (a) {sum(j['a'] for j in j0)}, (b) {sum(j['b'] for j in j0)}, (c') "
        f"{sum(j['c'] for j in j0)}, (a)+(c') {sum(j['a'] and j['c'] for j in j0)}, (c'') evaluated "
        f"{sum(j['c2'] is not None and 'ratio2' in j and j['ratio2'] is not None for j in j0)}, (a)+(c'') "
        f"{sum(j['a'] and bool(j['c2']) for j in j0)}; smallest d_P {min(float(j['d_P']) for j in j0):.3f}")
    for grp in groups:
        kind, L, P = grp.split("/")
        rs = [r for r in at0 if r["kind"] == kind and str(r["L"]) == L and str(r["P"]) == P]
        log(f"group {grp:14s}: plants {len(rs)}, (a) {sum(r['judge']['a'] for r in rs)}, (a)+(c') "
            f"{sum(r['judge']['a'] and r['judge']['c'] for r in rs)}, positions passing (c') "
            f"{sum(v['groups'][grp]['c']['pass'] for v in posv.values())}, (c'') "
            f"{sum(v['groups'][grp].get('c2', {}).get('pass', False) for v in posv.values())}")
    # the plants that pass at all three nodes
    full = {}
    for (pid, wf), r in recs.items():
        full.setdefault(pid, {})[wf] = r
    win = [d for d in full.values() if len(d) == 3 and all(d[w]["judge"]["a"] and d[w]["judge"]["c"] for w in NODES)]
    if win:
        e = [d[WF0]["eta"] for d in win]
        s12 = [d[WF0]["plant"]["share"]["12"] for d in win]
        log(f"plants passing (a)+(c') at all three nodes: {len(win)}; eta {min(e):.2e}-{max(e):.2e}; share of sigma_12 "
            f"{min(s12):.1e}-{max(s12):.1e}; d_P {min(float(d[w]['judge']['d_P']) for d in win for w in NODES):.3f}-"
            f"{max(float(d[w]['judge']['d_P']) for d in win for w in NODES):.3f}; ratio at wf 1.00 "
            f"{min(d[WF0]['judge']['ratio'] for d in win):.2f}-{max(d[WF0]['judge']['ratio'] for d in win):.2f}")
    # (a) against eta g, the misfit ratio, on the windowed plants
    bins = ((0.1, 0.3), (0.3, 1), (1, 3), (3, 10))
    ws = [r for r in at0 if gs[(r["pos"], r["L"], r["P"], r["alpha"])]["window"] == "window"]
    log("(a) on windowed plants by chi_22 / chi_22(real): " + "; ".join(
        f"{lo}-{hi}: {sum(r['judge']['a'] for r in ws if lo <= r['judge']['ratio'] < hi)} of "
        f"{sum(1 for r in ws if lo <= r['judge']['ratio'] < hi)}" for lo, hi in bins))
    log(f"ring.json: verdict {ring.get('verdict')}; R8 {ring.get('R8', {}).get('pass')}; C1 failures "
        f"{ring.get('C1_fail')}")
    log.dump({"note": "post-freeze summary, judges nothing", "n_wf1": len(at0), "n_records": len(recs)})
    return True


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    ref703 = next((a.split("=", 1)[1] for a in argv if a.startswith("--ref703=")), None)
    if "--selftest" in argv:
        if mut not in (None, "secmom", "len", "window"):
            raise SystemExit(f"unknown selftest mutation {mut}")
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] not in ("pilot", "ring", "report"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    if mut not in {"pilot": (None, "len", "window"), "ring": (None,), "report": (None,)}[stage]:
        raise SystemExit(f"unknown {stage} mutation {mut}")
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    if stage == "pilot":
        if not ref703:
            raise SystemExit("pilot needs --ref703=PATH (I703-mac/grid.jsonl of the raw-data repository)")
        ok = run_pilot(outdir, ref703, mut)
        if mut:
            ok = False
    elif stage == "ring":
        ok = run_ring(outdir)
    else:
        ok = run_report(outdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
