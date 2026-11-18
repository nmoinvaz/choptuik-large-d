#!/usr/bin/env python3
"""A pre-registered scan of second sets for #541's far Pade poles alone, feature (b) (issue #747).

#703 and #718 judged (a) the DA2 scatter, (b) the far Pade poles and (c') the twin misfit together, and no plant had d_P
below 0.136. Here only (b) is judged, over end points and cut directions wider than #718's and strengths that (c')
excluded (eta up to 1). Plants are #718's (pade_second_ring, pade_second_zero, pade_second_set, called unchanged): the
main density (chord2 or arc, zero factor at ZLOC) refitted to the real sigma_0..sigma_12 minus the second set's part. The
second set is the density s^4 (1-s)^alpha on the segment from A = Q - L e^(i theta) to Q, Q = P0 + kappa_* + o at each
node, with the share eta e^(i phi) of sigma_22(real). Radial: theta = arg Q (#718's ring_curve). Horizontal: theta = 0.

(b): d_P = max over k = 8, 9, 10 of the distance from REF541[wf][k] to the nearest pole of the [k/k] approximant (120
digits, sigma_0..sigma_20), pass if d_P <= 0.03. The poles are computed as pade_geometry.pade does, without the
numerator's zeros and the residues (R1 checks it bit for bit against pade_second_set.pade_record). A plant is admissible
if its main fit reproduces sigma_0..sigma_12 within 1e-2, and passes if it is admissible with d_P <= 0.03 at wf 0.99,
1.00 and 1.01 with the same parameters.

    python3 pade_far_poles.py --selftest [--mutate=prec|dir|pole_w]
    python3 pade_far_poles.py control OUTDIR [--mutate=shift_poles|len] [--ring=I718-mac/ring.jsonl] [--allow_unraw]
    python3 pade_far_poles.py scan OUTDIR [--max_hours=4] [--allow_unraw]   (reads OUTDIR/control.json; resumes)
    python3 pade_far_poles.py report OUTDIR [--allow_unraw]    (post-freeze maps of plants.jsonl, judges nothing)

Exit 0 all pass, 1 a control failed or a mutation fired as it must, 2 a mutation did not fire, 3 stopped on swap.
Numerical evidence on plants, not a bound. notes/PADE_FAR_END.md 18.
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_second_ring as R                                               # noqa: E402

Z = R.Z
ss, te, tc, pg, pp, tz = Z.ss, Z.te, Z.tc, Z.pg, Z.pp, Z.tz
NODES, WF0, N23 = pg.NODES, Z.WF0, Z.N23
KS = (8, 9, 10)
TOL_POLE = ss.TOL_POLE                      # 0.03, #667's
ADM = 1e-2                                  # admissible: the main fit reproduces sigma_0..sigma_12 within this
LAT = tuple(range(-3, 5))                   # a, b of Q = P0 + kappa_* + 0.25 (a + ib); #718's is a -1..4, b -2..4
GROUPS = (("chord2", 0.3, "radial"), ("arc", 0.3, "radial"), ("chord2", 0.3, "horizontal"),
          ("chord2", 0.6, "horizontal"))
P_NEAR = 4
ALPHAS, PHASES = ss.ALPHAS, ss.PHASES
ETAS = tuple(10.0 ** (-6 + j / 2) for j in range(13))
N_A = len(LAT) ** 2 * len(GROUPS) * len(ALPHAS) * len(ETAS) * len(PHASES)
A_CAP_H = 4.0
N_START, MAXFEV = 8, 200
STEPS = (0.5, 45.0, 0.06, 0.06, 0.08, 20.0)   # log10 eta, phi deg, Re o, Im o, L, theta deg
BOX = ((-6.0, 0.0), (None, None), (-1.0, 1.25), (-1.0, 1.25), (0.05, 1.2), (None, None))
C_LIM, C_CAP = 0.06, 300
RUNGS = tuple(10.0 ** (-3 + j / 2) for j in range(7))      # P: the point masses' share of sigma_22
TOL_JP = 1e-9                               # J+: the real series' d_P
TOL_PLUS = 1e-6                             # J- and R4: d_P against #718's, relative (another machine's lstsq)
SWAP_MB = 150                               # stop if swap use grows by this over its value at the start
FROZEN = {                                  # written before any plant was judged (notes/PADE_FAR_END.md 18.2)
    "stage_a": "Q = P0 + kappa_* + 0.25 (a + ib), a, b = -3..4 (64); groups (kind, L, direction) (chord2, 0.3, radial), "
               "(arc, 0.3, radial), (chord2, 0.3, horizontal), (chord2, 0.6, horizontal), near-end exponent 4; alpha 0 or "
               "beta - 1; eta 1e-6..1 in half decades (13); phi 0, 90, 180, 270; 26624 plants at wf 1.00, in the order a, "
               "b, group, alpha, eta, phi; stops after 4 h and lists what it did not reach",
    "stage_b": "from the 8 best admissible stage-A starts by d_P at wf 1.00, one per (kind, a, b, group, alpha), "
               "Nelder-Mead on (log10 eta, phi, Re o, Im o, L, theta) with D3 = the largest d_P over the three nodes (1 + "
               "the largest fit_resid if a node is inadmissible, 10 if a Pade fails); simplex steps (0.5, 45, 0.06, "
               "0.06, 0.08, 20), bounds log10 eta [-6, 0], Re o and Im o [-1, 1.25], L [0.05, 1.2]; 200 evaluations, "
               "xatol 1e-3, fatol 1e-5; theta absolute (radial start: arg Q at wf 1.00)",
    "stage_c": "every admissible stage-A plant with d_P <= 0.06 at wf 1.00 (the 300 smallest) at wf 0.99 and 1.01",
    "passes": "admissible and d_P <= 0.03 at the three nodes, same parameters; the answer is yes if any plant of stages "
              "A to C passes, no otherwise; every plant with d_P <= 0.03 at a node is rechecked through "
              "pade_second_set.pade_record, and a disagreement exits 1",
    "located": "#718's rule on the 64 positions; a passing position with four lattice neighbours inside the lattice that "
               "fail is a candidate; diagonal points are not run",
    "R1": "the poles-only d_P and nearest poles equal pade_record's bit for bit, 6 plants at wf 1.00; --mutate=prec (12 digits) fails it",
    "R2": "radial curves equal ring_curve's bit for bit at 64 positions and L 0.3, 0.6; horizontal A = Q - L exactly; "
          "exact moments of a horizontal and a theta = 30 deg segment against mpmath.quad within 1e-40; --mutate=dir "
          "fails it",
    "R3": "point masses' moments equal the mean of the tiny-segment limits (L = 1e-30, P 0, alpha 0) at k = 0, 7, 15, 22, "
          "31 within 1e-20 relative; --mutate=pole_w fails it",
    "R4": "12 of #718's plants (first record at wf 1.00 of I718-mac/ring.jsonl for (chord2, arc) x L 0.3 radial "
          "positions (+1, +1), (0, 0), (-1, -1) x alpha 0, beta - 1) through this path: d_P within 1e-6 relative; "
          "control --mutate=len (L x 1.01) fails it",
    "J": "J+ the real series' d_P <= 1e-9 at every node; J- the eta = 0 chord2 and arc plants at wf 1.00 have d_P > 0.03 "
         "and within 1e-6 relative of I718-mac/pilot.json's",
    "P": "main chord2 and main arc with three point masses at REF541[wf][8], [9], [10], equal weights, phi 0, the share "
         "10^(-3 + j/2), j 0..6, of sigma_22: for each kind some rung has d_P <= 0.03 at the three nodes, admissible or "
         "not; the lowest admissible one is the sensitivity",
    "mutation": "control --mutate=shift_poles (the reference poles + 0.05): J+ and P must fail; --mutate=len: R4 must "
                "fail",
    "gate": "scan runs only if the selftest, J+, J-, P and R4 pass",
}


# ---------------------------------------------------------------- the second set's segment, and the point masses
def seg_curve(main_arc, o, L, theta, mut=None):
    """the segment from Q - L e^(i theta) to Q, Q = P0 + kappa_* + o; theta None is radial, ring_curve's bit for bit,
    theta 0 gives A = Q - L exactly. mut len: L x 1.01; mut dir: theta + 1e-6 rad where theta is not None"""
    with mp.workdps(pg.DPS_PLANT):
        P0, ks = mp.mpc(main_arc.P0), mp.mpc(main_arc.P1)
        Q = P0 + ks + o
        ln = mp.mpf(L) * mp.mpf("1.01") if mut == "len" else mp.mpf(L)
        if theta is None:
            A = Q - ln * Q / abs(Q)
        elif theta == 0 and mut != "dir":
            A = Q - ln
        else:
            th = mp.mpf(theta) + (mp.mpf("1e-6") * 180 / mp.pi if mut == "dir" else 0)
            A = Q - ln * mp.expjpi(th / 180)
        return SimpleNamespace(P0=A, P1=Q, e0=mp.mpc(0), e1=mp.mpc(0), kind="chord")


def pole_moments(pts, mut=None):
    """mom_k = (1/n) sum_j t_j^k / k!, k <= KM, the moments of equal point masses at pts (second set's convention)"""
    with mp.workdps(pg.DPS_PLANT):
        n = len(pts)
        w = [mp.mpf(1) / n for _ in pts]
        if mut == "pole_w":
            w[0] *= 1 + mp.mpf("1e-6")
        t = [mp.mpc(x) for x in pts]
        return [mp.fsum(wj * tj ** k for wj, tj in zip(w, t)) / mp.factorial(k) for k in range(pg.KM + 1)]


def pole_part(main, pts, mut=None):
    """(pref L_P)_k, k <= KM, unscaled, as pade_second_set.second_part"""
    mp.mp.dps = pg.DPS_PLANT
    return pp.polymul(main.base.pref, pole_moments(pts, mut), pg.KM + 1)


# ---------------------------------------------------------------- (b): the poles only
def poles(sig, wf, ref=None, dps=None):
    """pade_second_set.pade_record's {k: {dist, nearest}}, from mp.pade and the denominator's roots only"""
    ref = ref or ss.REF541[wf]
    out = {}
    with mp.workdps(dps or pg.DPS_PADE):
        s = [mp.mpc(z) for z in sig[:N23]]
        for k in KS:
            try:
                a = [mp.factorial(j) * s[j] for j in range(2 * k + 1)]
                _, q = mp.pade(a, k, k)
                us = mp.polyroots(q[::-1], maxsteps=500, extraprec=2 * mp.mp.prec)
                ts = [complex(1 / u) for u in us]
            except (ZeroDivisionError, ValueError, mp.libmp.NoConvergence) as e:
                out[k] = {"dist": math.inf, "nearest": None, "error": type(e).__name__}
                continue
            t = min(ts, key=lambda z: abs(z - ref[k]))
            out[k] = {"dist": abs(t - ref[k]), "nearest": t}
    return out


def d_P(pl):
    return max(pl[k]["dist"] for k in KS)


# ---------------------------------------------------------------- nodes, plants
class FNode:
    """the main plants and the second-set parts at one node: no twin bases, so a few MB"""

    def __init__(self, wf):
        self.wf = wf
        self.real = tc.real_sig()[wf]
        self.mains = {k: Z.ZMain(wf, k, self.real, Z.zloc(wf)) for k in ss.KINDS}
        self.parts = {}

    def part(self, spec, mut=None, cache=True):
        key = (spec["kind"], tuple(spec["o"]), spec["L"], spec["theta"], spec["P"], spec["alpha"], mut)
        if cache and key in self.parts:
            return self.parts[key]
        main = self.mains[spec["kind"]]
        with mp.workdps(pg.DPS_PLANT):
            o = mp.mpc(spec["o"][0], spec["o"][1])
            cur = seg_curve(main.base.arc, o, spec["L"], spec["theta"], mut)
            al = ss.alpha_of(self.wf, spec["alpha"])
        out = (ss.second_part(main, cur, al, p=spec["P"]), complex(cur.P1))
        if cache:
            self.parts[key] = out
        return out


def lat_o(a, b):
    """#718's offset 0.25 (a + ib) as a pair of floats, exact"""
    return (0.25 * a, 0.25 * b)


def plant_id(spec):
    if spec["ab"] is not None:
        a, b = spec["ab"]
        pos = f"a{a:+d}b{b:+d}"
    else:
        pos = f"o{spec['o'][0]!r},{spec['o'][1]!r}"
    th = "radial" if spec["theta"] is None else f"th{spec['theta']!r}"
    return (f"{spec['kind']}|{pos}|L{spec['L']!r}|{th}|P{spec['P']}|{spec['alpha']}|eta{spec['eta']!r}|"
            f"ph{spec['phase']!r}")


def evaluate(node, spec, stage, ref=None, mut=None, cache=True, dps=None):
    t0 = time.time()
    w, Q = node.part(spec, mut, cache)
    sig, info = ss.make_plant(node.mains[spec["kind"]], w, spec["eta"], spec["phase"])
    pl = poles(sig, node.wf, ref, dps)
    dp = d_P(pl)
    return {"type": "p", "id": plant_id(spec), "stage": stage, "wf": node.wf, "kind": spec["kind"], "ab": spec["ab"],
            "o": list(spec["o"]), "L": spec["L"], "theta": spec["theta"], "P": spec["P"], "alpha": spec["alpha"],
            "eta": spec["eta"], "phase": spec["phase"], "Q": [Q.real, Q.imag],
            "d": {str(k): pl[k]["dist"] for k in KS}, "d_P": dp,
            "nearest": {str(k): (None if pl[k]["nearest"] is None else [pl[k]["nearest"].real, pl[k]["nearest"].imag])
                        for k in KS},
            "adm": info["fit_resid"] <= ADM, "fit_resid": info["fit_resid"], "M": info["M"],
            "share": {str(k): v for k, v in info["share"].items()}, "dev_13_22": info["dev_13_22"],
            "sec": time.time() - t0}


def swap_mb():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0])
    return (m["SwapTotal"] - m["SwapFree"]) / 1024


# ---------------------------------------------------------------- selftest (R1, R2, R3)
def selftest(mut=None):
    ok = True
    wf = WF0
    node = FNode(wf)
    # R1: the poles-only path against pade_record, bit for bit
    cases = []
    for kind in ss.KINDS:
        cases += [(kind, None, 0.0), (kind, (1, 1, None), 1e-2), (kind, (-1, 0, "h"), 1e-1)]
    same, worst = True, 0.0
    for kind, pos, eta in cases:
        main = node.mains[kind]
        if pos is None:
            sig, _ = ss.make_plant(main, None, 0, 0)
        else:
            a, b, h = pos
            spec = {"kind": kind, "ab": (a, b), "o": lat_o(a, b), "L": 0.3, "theta": 0.0 if h else None,
                    "P": P_NEAR, "alpha": "log", "eta": eta, "phase": 90}
            w, _ = node.part(spec)
            sig, _ = ss.make_plant(main, w, eta, 90)
        mine = poles(sig, wf, dps=12 if mut == "prec" else None)
        full = ss.pade_record(sig, wf)
        for k in KS:
            same &= mine[k]["dist"] == full[k]["dist"] and mine[k]["nearest"] == full[k]["nearest"]
            worst = max(worst, abs(mine[k]["dist"] - full[k]["dist"]))
    print(f"selftest R1: the poles-only d_P and nearest poles equal pade_record's bit for bit on 6 plants: {same} "
          f"(largest difference {worst:.1e}; pass = 0)")
    ok &= same
    # R2: the curves and the second set's moments on them
    rad = hor = True
    for kind, main in node.mains.items():
        arc = main.base.arc
        for a in LAT:
            for b in LAT:
                for L in (0.3, 0.6):
                    with mp.workdps(pg.DPS_PLANT):
                        o = mp.mpc(*lat_o(a, b))
                        c0 = R.ring_curve(arc, (a, b, None), L)
                        c1 = seg_curve(arc, o, L, None, mut)
                        c2 = seg_curve(arc, o, L, 0.0, mut)
                        rad &= c0.P0 == c1.P0 and c0.P1 == c1.P1
                        hor &= c2.P1 == c0.P1 and c2.P0 == c0.P1 - mp.mpf(L)
    print(f"selftest R2: radial curves equal ring_curve's bit for bit (64 positions x L 0.3, 0.6 x 2 mains): {rad}; "
          f"horizontal curves have A = Q - L exactly: {hor}")
    ok &= rad and hor
    worst = 0.0
    arc = node.mains["chord2"].base.arc
    for theta in (0.0, 30.0):
        for L in (0.3, 0.6):
            for al in ALPHAS:
                with mp.workdps(pg.DPS_PLANT):
                    cur = seg_curve(arc, mp.mpc(*lat_o(2, 2)), L, theta, mut)
                    alm = ss.alpha_of(wf, al)
                    ex = ss.sec_moments(cur, alm, P_NEAR, 32)
                    e = max(abs(ex[k] - ss.sec_moments_quad(cur, alm, P_NEAR, k)) / abs(ex[k]) for k in (0, 7, 15, 22, 31))
                worst = max(worst, float(e))
    print(f"selftest R2: exact moments of horizontal and 30 deg segments against mpmath.quad at k = 0, 7, 15, 22, 31: "
          f"max rel diff {worst:.1e} (pass < 1e-40)")
    ok &= worst < 1e-40
    # R3: the point masses' moments against the limit of tiny segments
    pts = [ss.REF541[wf][k] for k in KS]
    mom = pole_moments(pts, mut)
    lim = []
    with mp.workdps(pg.DPS_PLANT):
        for t in pts:
            ln = mp.mpf("1e-30")
            P0m = mp.mpc(arc.P0)
            o = mp.mpc(t) - P0m - mp.mpc(arc.P1)          # puts Q at t
            cur = seg_curve(arc, o, ln, None)
            m = ss.sec_moments(cur, mp.mpc(0), 0, 32)
            lim.append([x / m[0] for x in m])
        worst = max(float(abs(mom[k] - mp.fsum(l[k] for l in lim) / 3) / abs(mom[k])) for k in (0, 7, 15, 22, 31))
    print(f"selftest R3: three point masses' moments against the mean of the tiny-segment limits at k = 0, 7, 15, 22, 31: "
          f"max rel diff {worst:.1e} (pass < 1e-20)")
    ok &= worst < 1e-20
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- control stage
def ring_plants(path):
    """12 of #718's plants: the first wf 1.00 record for each (chord2, arc) x L 0.3 radial (+1, +1), (0, 0), (-1, -1) x
    alpha"""
    want = {(k, p, a) for k in ss.KINDS for p in ("L+1+1", "L+0+0", "L-1-1") for a in ALPHAS}
    got = {}
    for line in open(path):
        if '"wf": 1.0' not in line:
            continue
        r = json.loads(line)
        if r.get("type") == "g":
            continue
        key = (r["kind"], r["pos"], r["alpha"])
        if key in want and key not in got and r["L"] == 0.3 and r["P"] == P_NEAR and r["wf"] == WF0:
            got[key] = r
        if len(got) == len(want):
            break
    return [got[k] for k in sorted(got)]


def run_control(outdir, mut, ring_path):
    log = tz.Log(outdir, "control", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "nodes": {}}
    ok = {k: True for k in ("selftest", "J+", "J-", "P", "R4")}
    ok["selftest"] = selftest(None)
    nodes = {}
    for wf in NODES:
        t0 = time.time()
        nodes[wf] = FNode(wf)
        log(f"node {wf}: set up in {time.time() - t0:.0f}s")
    shift = 0.05 if mut == "shift_poles" else 0
    refs = {wf: {k: z + shift for k, z in ss.REF541[wf].items()} for wf in NODES}
    # J+: the real series
    for wf in NODES:
        dp = d_P(poles(nodes[wf].real, wf, refs[wf]))
        good = dp <= TOL_JP
        ok["J+"] &= good
        rec["nodes"].setdefault(wf, {})["J+"] = {"d_P": dp, "pass": good}
        log(f"J+ wf {wf}: the real series' d_P {dp:.2e} (pass <= {TOL_JP:.0e}) -> {'OK' if good else 'BAD'}")
    # J-: the eta = 0 plants at wf 1.00 against #718's
    ref718 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(outdir)), "I718-mac", "pilot.json")))["J"]
    for kind in ss.KINDS:
        sig, _ = ss.make_plant(nodes[WF0].mains[kind], None, 0, 0)
        dp = d_P(poles(sig, WF0, refs[WF0]))
        old = ref718[f"J- {kind}"]["judge"]["d_P"]
        good = dp > TOL_POLE and abs(dp / old - 1) <= TOL_PLUS
        ok["J-"] &= good
        rec["nodes"][WF0][f"J- {kind}"] = {"d_P": dp, "d_P_718": old, "pass": good}
        log(f"J- wf {WF0} {kind} eta 0: d_P {dp:.6f} (#718 {old:.6f}; pass > 0.03 and within {TOL_PLUS:.0e}) -> "
            f"{'OK' if good else 'BAD'}")
    # P: the planted poles
    rec["P"] = {}
    for kind in ss.KINDS:
        rows = []
        for eta in RUNGS:
            per = {}
            for wf in NODES:
                main = nodes[wf].mains[kind]
                w = pole_part(main, [ss.REF541[wf][k] for k in KS])
                sig, info = ss.make_plant(main, w, eta, 0)
                per[wf] = {"d_P": d_P(poles(sig, wf, refs[wf])), "adm": info["fit_resid"] <= ADM,
                           "fit_resid": info["fit_resid"]}
            passes = all(v["d_P"] <= TOL_POLE for v in per.values())
            adm = all(v["adm"] for v in per.values())
            rows.append({"eta": eta, "nodes": per, "passes": passes, "admissible": adm})
            log(f"P {kind} eta_P {eta:.3g}: d_P " + ", ".join(f"{per[wf]['d_P']:.4f}" for wf in NODES)
                + f"; fit_resid " + ", ".join(f"{per[wf]['fit_resid']:.1e}" for wf in NODES)
                + f" -> passes {passes}, admissible {adm}")
        some = any(r["passes"] for r in rows)
        low = next((r["eta"] for r in rows if r["passes"] and r["admissible"]), None)
        ok["P"] &= some
        rec["P"][kind] = {"rows": rows, "some_rung_passes": some, "lowest_admissible_passing": low}
        log(f"P {kind}: some rung passes at the three nodes: {some} -> {'OK' if some else 'BAD'}; lowest admissible "
            f"passing rung {low}")
    # R4: #718's plants through this path
    rec["R4"] = []
    for r in ring_plants(ring_path):
        a, b, _ = r["a_b_k"]
        spec = {"kind": r["kind"], "ab": (a, b), "o": lat_o(a, b), "L": r["L"], "theta": None, "P": r["P"],
                "alpha": r["alpha"], "eta": r["eta"], "phase": r["phase"]}
        got = evaluate(nodes[WF0], spec, "R4", refs[WF0], mut)
        old = r["judge"]["d_P"]
        dev = abs(got["d_P"] / old - 1)
        good = dev <= TOL_PLUS
        ok["R4"] &= good
        rec["R4"].append({"id": r["id"], "d_P": got["d_P"], "d_P_718": old, "dev": dev, "pass": good})
        log(f"R4 {r['id']}: d_P {got['d_P']:.6f} (#718 {old:.6f}); rel dev {dev:.1e} (pass <= {TOL_PLUS:.0e}) -> "
            f"{'OK' if good else 'BAD'}")
    if len(rec["R4"]) != 12:
        ok["R4"] = False
        log(f"R4: found {len(rec['R4'])} of 12 plants in the ring file -> BAD")
    rec["checks"] = ok
    if mut == "shift_poles":
        fired = (not ok["J+"]) and (not ok["P"])
    elif mut == "len":
        fired = not ok["R4"]
    else:
        fired = None
    rec["mutation_fired"] = fired
    if mut:
        log(f"mutation {mut}: fired as frozen: {fired} (must)")
    rec["controls_pass"] = all(ok.values()) and not mut
    log(f"controls {ok} -> gate {rec['controls_pass']}")
    log.dump(rec)
    return rec


# ---------------------------------------------------------------- the scan
def stage_a_specs():
    for a in LAT:
        for b in LAT:
            for kind, L, dirn in GROUPS:
                for al in ALPHAS:
                    for eta in ETAS:
                        for ph in PHASES:
                            yield {"kind": kind, "ab": (a, b), "o": lat_o(a, b), "L": L,
                                   "theta": None if dirn == "radial" else 0.0, "P": P_NEAR, "alpha": al, "eta": eta,
                                   "phase": ph}


def load(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done[(r["id"], r["wf"])] = r
    return done


def spec_of(r):
    return {"kind": r["kind"], "ab": None if r["ab"] is None else tuple(r["ab"]), "o": tuple(r["o"]), "L": r["L"],
            "theta": r["theta"], "P": r["P"], "alpha": r["alpha"], "eta": r["eta"], "phase": r["phase"]}


def three_node(done):
    by = {}
    for (pid, wf), r in done.items():
        by.setdefault(pid, {})[wf] = r
    return by


def passes(rs):
    return len(rs) == 3 and all(r["adm"] and r["d_P"] <= TOL_POLE for r in rs.values())


def run_scan(outdir, max_hours):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    if not ctl["controls_pass"]:
        print("the controls failed: no scan (frozen rule)")
        return 1
    log = tz.Log(outdir, "scan", None)
    path = os.path.join(outdir, "plants.jsonl")
    done = load(path)
    jl = open(path, "a")
    swap0 = swap_mb()
    nodes = {wf: FNode(wf) for wf in NODES}
    t_start = time.time()
    state = {"n": 0}

    def guard():
        if swap_mb() > swap0 + SWAP_MB:
            log(f"swap use {swap_mb():.0f} MB, {swap_mb() - swap0:.0f} MB over its start: stopped, resumable")
            jl.flush()
            raise SystemExit(3)

    def run(wf, spec, stage):
        key = (plant_id(spec), wf)
        if key not in done:
            r = evaluate(nodes[wf], spec, stage, cache=stage == "A")
            jl.write(json.dumps(r) + "\n")
            jl.flush()
            done[key] = r
            state["n"] += 1
            if state["n"] % 200 == 0:
                guard()
        return done[key]

    log(f"scan: {len(done)} records already in plants.jsonl; swap in use {swap0:.0f} MB; stage A {N_A} plants")
    # stage A
    best = math.inf
    t_a = time.time()
    left, n_run = [], 0
    for spec in stage_a_specs():
        if (time.time() - t_a) / 3600 > max_hours:
            left.append(spec["ab"])
            continue
        r = run(WF0, spec, "A")
        n_run += 1
        if r["adm"] and r["d_P"] < best:
            best = r["d_P"]
            log(f"stage A: new smallest d_P {best:.5f} at {r['id']} ({n_run} plants, {(time.time() - t_a) / 60:.0f} min)")
        if n_run % 1000 == 0:
            log(f"stage A: {n_run} plants, {(time.time() - t_a) / 60:.0f} min, smallest d_P {best:.5f}, {r['sec']:.2f}s a plant")
    left_pos = sorted(set(left))
    stage_a = {"plants": n_run, "minutes": (time.time() - t_a) / 60, "smallest_d_P": best, "positions_left_out": left_pos}
    log(f"stage A done: {n_run} plants in {(time.time() - t_a) / 60:.0f} min; smallest d_P {best:.5f}; positions left "
        f"out {left_pos or 'none'}")
    # stage B
    from scipy.optimize import minimize
    a_rec = [r for (pid, wf), r in done.items() if r["stage"] == "A" and wf == WF0 and r["adm"]
             and math.isfinite(r["d_P"])]
    starts = {}
    for r in a_rec:
        key = (r["kind"], tuple(r["ab"]), r["L"], r["theta"], r["alpha"])
        if key not in starts or r["d_P"] < starts[key]["d_P"]:
            starts[key] = r
    starts = sorted(starts.values(), key=lambda r: r["d_P"])[:N_START]
    stage_b = []
    for i, r0 in enumerate(starts):
        theta0 = r0["theta"] if r0["theta"] is not None else math.degrees(math.atan2(r0["Q"][1], r0["Q"][0]))
        x0 = np.array([math.log10(r0["eta"]), float(r0["phase"]), r0["o"][0], r0["o"][1], r0["L"], theta0])
        sim = [x0]
        for j in range(6):
            x = x0.copy()
            x[j] += STEPS[j]
            lo, hi = BOX[j]
            if (hi is not None and x[j] > hi) or (lo is not None and x[j] < lo):
                x[j] = x0[j] - STEPS[j]
            sim.append(x)
        evals = {"n": 0}

        def f(x, i=i, r0=r0, evals=evals):
            spec = {"kind": r0["kind"], "ab": None, "o": (float(x[2]), float(x[3])), "L": float(x[4]),
                    "theta": float(x[5]), "P": r0["P"], "alpha": r0["alpha"], "eta": 10.0 ** float(x[0]),
                    "phase": float(x[1])}
            recs = []
            for wf in NODES:
                s = dict(spec)
                rr = run(wf, s, f"B{i}")
                recs.append(rr)
            evals["n"] += 1
            if any(not math.isfinite(r["d_P"]) for r in recs):
                return 10.0
            if any(not r["adm"] for r in recs):
                return 1.0 + max(r["fit_resid"] for r in recs)
            return max(r["d_P"] for r in recs)
        f0 = f(x0)
        res = minimize(f, x0, method="Nelder-Mead", bounds=BOX,
                       options={"maxfev": MAXFEV, "xatol": 1e-3, "fatol": 1e-5, "initial_simplex": np.array(sim)})
        stage_b.append({"start": i, "from": r0["id"], "d_P_wf1_start": r0["d_P"], "D3_start": f0, "D3_best": float(res.fun),
                        "x_best": [float(v) for v in res.x], "nfev": int(res.nfev), "message": str(res.message)})
        log(f"stage B start {i} ({r0['id']}, d_P {r0['d_P']:.4f}): D3 {f0:.4f} -> {res.fun:.4f} in {res.nfev} evaluations; "
            f"x = {[round(float(v), 4) for v in res.x]}")
    # stage C
    cands = sorted((r for r in a_rec if r["d_P"] <= C_LIM), key=lambda r: r["d_P"])[:C_CAP]
    for r in cands:
        for wf in NODES:
            if wf != WF0:
                run(wf, spec_of(r), "C")
    log(f"stage C: {len(cands)} stage-A plants with d_P <= {C_LIM} at wf 1.00 run at the other nodes")
    jl.close()
    # verdict
    done = load(path)
    by = three_node(done)
    full = {pid: rs for pid, rs in by.items() if len(rs) == 3}
    rechecked = True
    passing = []
    for pid, rs in full.items():
        if passes(rs):
            passing.append(pid)
            for wf, r in rs.items():
                node = nodes[wf]
                w, _ = node.part(spec_of(r), None, False)
                sig, _ = ss.make_plant(node.mains[r["kind"]], w, r["eta"], r["phase"])
                full_dp = d_P(ss.pade_record(sig, wf))
                if not (full_dp <= TOL_POLE):
                    rechecked = False
                    log(f"RECHECK FAILED {pid} wf {wf}: pade_record d_P {full_dp} against {r['d_P']}")
    d3 = {pid: max(r["d_P"] for r in rs.values()) for pid, rs in full.items() if all(r["adm"] for r in rs.values())}
    closest = min(d3, key=d3.get) if d3 else None
    allw = [r for (pid, wf), r in done.items() if wf == WF0 and r["adm"] and math.isfinite(r["d_P"])]
    r_min = min(allw, key=lambda r: r["d_P"])
    pos = {}
    for pid, rs in by.items():
        r = rs.get(WF0)
        if r and r["ab"] is not None and r["stage"] in ("A", "C"):
            p = tuple(r["ab"])
            st = pos.setdefault(p, {"min_d_P": math.inf, "pass": False, "unknown": False})
            if r["adm"]:
                st["min_d_P"] = min(st["min_d_P"], r["d_P"])
                if r["d_P"] <= TOL_POLE:
                    if len(rs) == 3:
                        st["pass"] |= passes(rs)
                    else:
                        st["unknown"] = True
    cand = []
    for (a, b), st in pos.items():
        if st["pass"]:
            nb = [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
            if all(n in pos for n in nb) and all(not pos[n]["pass"] and not pos[n]["unknown"] for n in nb):
                cand.append((a, b))
    verdict = {"answer": "yes" if passing else "no", "passing_plants": passing, "n_passing": len(passing),
               "passing_positions": sorted(p for p, st in pos.items() if st["pass"]),
               "unknown_positions": sorted(p for p, st in pos.items() if st["unknown"]),
               "candidates": cand, "recheck_ok": rechecked, "n_full_plants": len(full),
               "closest_D3": None if closest is None else {"id": closest, "D3": d3[closest],
                                                          "nodes": {str(wf): full[closest][wf]["d_P"] for wf in NODES}},
               "smallest_d_P_wf1": {"id": r_min["id"], "d_P": r_min["d_P"], "stage": r_min["stage"]},
               "stage_a": stage_a, "stage_b": stage_b, "n_records": len(done)}
    log(f"VERDICT: a second set of this family gives (b): {verdict['answer']}; {len(passing)} plants pass at the three "
        f"nodes; passing positions {verdict['passing_positions'] or 'none'}; candidates {cand or 'none'}")
    log(f"smallest d_P at wf 1.00 {r_min['d_P']:.5f} ({r_min['id']}, stage {r_min['stage']}); smallest D3 "
        + (f"{d3[closest]:.5f} ({closest})" if closest else "none") + f"; {len(full)} plants run at the three nodes; "
        f"recheck through pade_record {'ok' if rechecked else 'FAILED'}")
    log.dump({"frozen": FROZEN, "verdict": verdict})
    return 0 if rechecked else 1


# ---------------------------------------------------------------- report (post-freeze, judges nothing)
def run_report(outdir):
    log = tz.Log(outdir, "report", None)
    done = load(os.path.join(outdir, "plants.jsonl"))
    rs = [r for (pid, wf), r in done.items() if wf == WF0 and r["stage"] == "A" and math.isfinite(r["d_P"])]
    log(f"{len(done)} records, {len(rs)} stage-A plants at wf 1.00")
    letter = {("chord2", 0.3, None): "r", ("arc", 0.3, None): "a", ("chord2", 0.3, 0.0): "h", ("chord2", 0.6, 0.0): "H"}
    for lab, filt in (("admissible", lambda r: r["adm"]), ("all", lambda r: True)):
        log(f"smallest d_P at wf 1.00 per position, {lab} plants (row b from +4 to -3, column a from -3 to +4; "
            f"letter = group: r chord2 radial, a arc radial, h chord2 horizontal 0.3, H chord2 horizontal 0.6)")
        log("   b\\a " + "".join(f"{a:>8d}" for a in LAT))
        for b in reversed(LAT):
            row = []
            for a in LAT:
                c = [r for r in rs if tuple(r["ab"]) == (a, b) and filt(r)]
                if c:
                    m = min(c, key=lambda r: r["d_P"])
                    row.append(f"{m['d_P']:7.3f}{letter[(m['kind'], m['L'], m['theta'])]}")
                else:
                    row.append("       -")
            log(f"  {b:+d}   " + "".join(row))
    for k in KS:
        c = [r for r in rs if r["adm"]]
        m = min(c, key=lambda r: r["d"][str(k)])
        log(f"smallest d_{k} over admissible stage-A plants: {m['d'][str(k)]:.5f} at {m['id']} (its d_P {m['d_P']:.4f})")
    log("smallest d_P by eta (admissible, over positions, groups, alpha, phi):")
    for eta in ETAS:
        c = [r for r in rs if r["adm"] and r["eta"] == eta]
        n_adm = len(c)
        n = sum(1 for r in rs if r["eta"] == eta)
        log(f"  eta {eta:.3g}: {min(r['d_P'] for r in c):.4f} ({n_adm} of {n} admissible)" if c else
            f"  eta {eta:.3g}: none admissible of {n}")
    log("smallest d_P by phi (admissible): " + ", ".join(
        f"{ph}: {min(r['d_P'] for r in rs if r['adm'] and r['phase'] == ph):.4f}" for ph in PHASES))
    full = {pid: v for pid, v in three_node(done).items() if len(v) == 3}
    close = sorted(((max(r["d_P"] for r in v.values()), pid) for pid, v in full.items()
                    if all(r["adm"] for r in v.values())))[:10]
    log("ten smallest D3 over plants run at three nodes: " + "; ".join(f"{d:.4f} {pid}" for d, pid in close))
    for d, pid in close[:3]:
        v = full[pid]
        log(f"  {pid}: d_k at wf 0.99/1.00/1.01 " + " | ".join(
            ",".join(f"{v[wf]['d'][str(k)]:.3f}" for k in KS) for wf in NODES)
            + f"; share12/22 at wf 1.00 {v[WF0]['share']['12']:.1e}/{v[WF0]['share']['22']:.1e}; fit_resid "
            f"{v[WF0]['fit_resid']:.1e}")
    return 0


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    ring = next((a.split("=", 1)[1] for a in argv if a.startswith("--ring=")), None)
    hours = next((float(a.split("=", 1)[1]) for a in argv if a.startswith("--max_hours=")), A_CAP_H)
    if "--selftest" in argv:
        if mut not in (None, "prec", "dir", "pole_w"):
            raise SystemExit(f"unknown selftest mutation {mut}")
        ok = selftest(mut)
        if mut:
            print(f"mutation {mut} {'made the selftest fail, as it must' if not ok else 'did NOT fail the selftest'}")
            sys.exit(1 if not ok else 2)
        sys.exit(0 if ok else 1)
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) != 2 or pos[0] not in ("control", "scan", "report"):
        raise SystemExit(__doc__)
    stage, outdir = pos
    if stage == "control" and mut not in (None, "shift_poles", "len"):
        raise SystemExit(f"unknown control mutation {mut}")
    if stage != "control" and mut:
        raise SystemExit(f"{stage} has no mutation")
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    if stage == "control":
        ring = ring or os.path.join(os.path.dirname(os.path.abspath(outdir)), "I718-mac", "ring.jsonl")
        rec = run_control(outdir, mut, ring)
        if mut:
            fired = rec["mutation_fired"]
            print(f"mutation {mut} {'fired as frozen' if fired else 'did NOT fire'}")
            sys.exit(1 if fired else 2)
        sys.exit(0 if rec["controls_pass"] else 1)
    if stage == "scan":
        sys.exit(run_scan(outdir, hours))
    sys.exit(run_report(outdir))


if __name__ == "__main__":
    main()
