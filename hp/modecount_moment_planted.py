#!/usr/bin/env python3
"""Planted-truth test of the zero location from a count's samples (hp/modecount_moment.py) on a synthetic det M.

    python3 modecount_moment_planted.py [--mutate=nok|shift]     exit 1 if a criterion fails (seconds)
    python3 modecount_moment_planted.py --controls               the test and both mutations: exit 0 only if the test
                                                                 passes and each mutation fails

The synthetic determinant has the structure and the contour layout of the D 61.03, M 48, kf 23 count (#212): omega,
lam_g, lam_phys and n of that state, the rows of horizon poles n/2 - lam_g (kk + 1) + i (2j + 1) omega, the copies
2 i k omega of the expected zeros, growth exp(41.2 lam + 1.40 lam^2), and the count's aLL, aL, a1, a2, aR. Each case
plants extra zeros (with their conjugates). The count runs through modecount_wind.count with the real count's options
(h0 0.6, tol pi/4, --resolve=last), writing its record and evaluation cache to a temporary directory, and the zeros are
located from them by modecount_moment.locate, the code path of the real data.

Cases. The first three were seen while the estimator was chosen (development cases):
  row      B1b 1.55+0.70i, 2.10+1.30i, 2.45+0.40i; B1c 4.30+1.00i; U a row of 8 at Im 2.3, Re 1.0, 1.7, ..., 5.9
  edges    B1b 1.35+0.25i (0.08 from Re = a1), 2.20+1.60i (0.14 below Im = y1); B1c 3.60+0.12i (0.12 above the real
           axis, where the half contour has only the count's real-axis samples, 0.5 apart), 5.80+1.20i
  none     no extra zero
  random0 .. random7, drawn with seed 230: 1-3 zeros in the upper half of B1b, 1-2 in that of B1c, 0-9 in U, each at
           least 0.1 from its box's edges, 0.25 from the others and lam_phys, and 0.2 from the candidate poles (seen
           in the second version, below).
  held0 .. held11, drawn in the same way with seed 231, first run with the criteria below (held-out cases).
  left     B0 0.10+0.80i; U0 0.11+2.60i; U 2.00+2.60i (the boxes of the D 55 extras, #232)
  left0 .. left5, drawn with seed 232: 1-2 zeros in the upper half of B0 (at least 0.25 from lam = 0), 0-2 in U0,
           0-3 in U, 0-2 in the upper half of B1b, with the margins above; first run with criterion (e).

Pass criteria (third version):
  (a) the count: in the boxes of |Im| < y1 (B0, B1a, B1b, B1c, B1) wind(g) equals the number of planted zeros inside
      (conjugates included) and every edge is resolved, with every phase step at most pi/4. U, U0 and full are
      printed, not tested (their long edge Im = y2 can hide whole turns, see History);
  (b) for B1b and B1c the windings of the sym and half contours equal the numbers of planted zeros inside, to 1e-6;
  (c) B1b and B1c: the reported roots pair one to one with the planted zeros of the upper half (the pairing that
      minimises the largest distance). Every root with spread below 0.05 lies within 0.05 of its planted zero (the
      accuracy that centring circles of radius 0.1 or more needs), and at least half of all planted zeros of B1b and
      B1c, over the cases, have a root with spread below 0.05;
  (d) ln g recomputed by the moment code agrees with the samples stored in the count's record to 1e-4;
  (e) B0, and U0 where the count's U0 winding is right: (b) and the first part of (c) hold as for B1b and B1c (B0 has
      no half contour: the count has no real-axis samples left of aL). Added for the left column (#238) with the left
      cases, before their first run.
  U is printed, not tested; U's roots are not used on real data.
Controls, each must make the test fail: --mutate=nok drops the factor k of the integration by parts in the moment code;
--mutate=shift plants every extra zero 0.1 to the right of where the criteria expect it.

History. The first version (commit 38ab804) estimated a root's error by its distance from the trapezoid rule's root
and required estimates below 0.02 in mode sym. It failed on row and edges: the cubic roots were within 1.1e-2 of the
planted zeros, but the trapezoid rule is 10-100 times worse, so the estimates were 0.1-0.6. The second version
(72337bc) used the spread, required every root within 0.05 and within its spread, and added random0-7. It failed:
(i) the spread was below the error in 5 of 8 random cases, by up to 4 times (errors 1e-3 to 2.3e-2), because every
stencil and both contours share the same samples; (ii) with three zeros of B1b near the real axis or close together
(random3, random6) the error reached 0.1 and 0.45, with spreads as large; (iii) the count itself missed whole turns
on the edge Im = y2 in random0, random4 and random7 (U 5, -4 and 2 for 8, 3 and 4 planted). That edge's phase grows
by about 11 rad per unit of Re; an interval of the initial grid (0.57) then turns by about 2 pi, looks flat, and is
never bisected or verified. The count's boxes of |Im| < y1 do not use that edge. The third version keeps the spread
as a flag (a root with a small spread is accurate enough to centre a circle) and the circles of
hp/modecount_circle.py give the positions.
"""
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modecount_wind as W  # noqa: E402
import modecount_moment as MM  # noqa: E402

# D 61.0264, M 48, kf 23 (#212, raw I212-macm3/data/modecount_results.jsonl)
OM, LG, LP, N = 3.476754518412088, 0.8992305398413484, 2.043532530673195, 59.0264
LAYOUT = dict(aLL=-0.0805, aL=0.3, a1=1.2692241294287814, a2=2.643532530673195, aR=6.5845)
Y1, Y2 = 0.5 * OM, 1.5 * OM
BASE_ZEROS = [z + 2j * k * OM for z in (0.0, LG, LP) for k in range(-4, 5)]
POLES = [N / 2 - LG * (kk + 1) + s * 1j * OM for kk in range(101) for s in (1, -1, 3, -3, 5, -5)]
CASES = {
    "row": [1.55 + 0.70j, 2.10 + 1.30j, 2.45 + 0.40j, 4.30 + 1.00j] + [1.0 + 0.7 * j + 2.3j for j in range(8)],
    "edges": [1.35 + 0.25j, 2.20 + 1.60j, 3.60 + 0.12j, 5.80 + 1.20j],
    "none": [],
}
TOL = 0.05


def random_cases(seed=230, n=8, prefix="random"):
    rng = np.random.default_rng(seed)
    near = [p for p in POLES if abs(p.imag) < Y2 + 0.5 and -1 < p.real < 8]

    def draw(k, r0, r1, i0, i1, taken):
        out = []
        while len(out) < k:
            z = complex(rng.uniform(r0 + 0.1, r1 - 0.1), rng.uniform(i0 + 0.1, i1 - 0.1))
            if (all(abs(z - w) > 0.25 for w in taken + out) and abs(z - LP) > 0.25
                    and all(abs(z - p) > 0.2 for p in near)):
                out.append(z)
        return out
    cases = {}
    for i in range(n):
        nb, nc, nu = int(rng.integers(1, 4)), int(rng.integers(1, 3)), int(rng.integers(0, 10))
        z = draw(nb, LAYOUT["a1"], LAYOUT["a2"], 0.0, Y1, [])
        z += draw(nc, LAYOUT["a2"], LAYOUT["aR"], 0.0, Y1, z)
        z += draw(nu, LAYOUT["aL"], LAYOUT["aR"], Y1, Y2, z)
        cases[f"{prefix}{i}"] = z
    return cases


def left_cases(seed=232, n=6):
    rng = np.random.default_rng(seed)
    near = [p for p in POLES if abs(p.imag) < Y2 + 0.5 and -1 < p.real < 8]
    aLL, aL = LAYOUT["aLL"], LAYOUT["aL"]

    def draw(k, r0, r1, i0, i1, taken):
        out = []
        while len(out) < k:
            z = complex(rng.uniform(r0 + 0.1, r1 - 0.1), rng.uniform(i0 + 0.1, i1 - 0.1))
            if (all(abs(z - w) > 0.25 for w in taken + out) and abs(z - LP) > 0.25 and abs(z) > 0.25
                    and all(abs(z - p) > 0.2 for p in near)):
                out.append(z)
        return out
    cases = {}
    for i in range(n):
        n0, nu0, nu, nb = (int(rng.integers(1, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 4)),
                           int(rng.integers(0, 3)))
        z = draw(n0, aLL, aL, 0.0, Y1, [])
        z += draw(nu0, aLL, aL, Y1, Y2, z)
        z += draw(nu, aL, LAYOUT["aR"], Y1, Y2, z)
        z += draw(nb, LAYOUT["a1"], LAYOUT["a2"], 0.0, Y1, z)
        cases[f"left{i}"] = z
    return cases


CASES.update(random_cases())
CASES.update(random_cases(231, 12, "held"))
CASES["left"] = [0.10 + 0.80j, 0.11 + 2.60j, 2.00 + 2.60j]
CASES.update(left_cases())
GATED = ("B0", "B1a", "B1b", "B1c", "B1")
GATED_EDGES = ("L1", "H0", "V0", "V1", "H3", "V2", "H2", "R1", "H1")


def det_for(extra):
    zeros = np.array(BASE_ZEROS + extra + [z.conjugate() for z in extra], complex)
    poles = np.array(POLES, complex)

    def det(re, im):
        lam = complex(re, im)
        v = 41.2 * lam + 1.40 * lam * lam - 300 + np.sum(np.log(lam - zeros)) - np.sum(np.log(lam - poles))
        return v.real, math.remainder(v.imag, 2 * math.pi), 0.0, 0.0
    return det


def inside(rect, pts):
    r0, r1, i0, i1 = rect
    return [z for z in pts if r0 < z.real < r1 and i0 < z.imag < i1]


def run_case(name, extra, mutate, tmp):
    shift = 0.1 if mutate == "shift" else 0.0
    planted = list(extra) + [z.conjugate() for z in extra]
    info = dict(D=N + 2, M=0, omega=OM, lam_g=LG, lam_f=LP)
    opts = dict(LAYOUT, data=tmp, h0="0.6", resolve="last")
    tag = f"planted_{name}"
    rec = W.count(det_for([z + shift for z in extra]), info, POLES, opts, lambda m: None, tag,
                  os.path.join(tmp, "modecount_results.jsonl"))
    bad, ungated = [], []
    for b, x in rec["boxes"].items():
        want = len(inside(x["rect"], planted))
        if x["winding_g_int"] != want or abs(x["winding_g"] - want) > 1e-6 or not x["resolved"]:
            msg = f"{b}: wind(g) {x['winding_g']:+.3f}, planted {want}" + ("" if x["resolved"] else ", not resolved")
            (bad.append("(a) " + msg) if b in GATED else ungated.append(msg))
    mx = max(rec["edges"][e]["max_step"] for e in GATED_EDGES)
    if mx > math.pi / 4 + 1e-12:
        bad.append(f"(a) max phase step {mx:.3f} > pi/4")
    r, ev = MM.load_count(tmp, tag)
    Q = MM.factored(POLES, r)
    chk = MM.stored_check(r, MM.LnG(r, ev, Q), list(r["edges"]))
    if chk > 1e-4:
        bad.append(f"(d) ln g against the stored samples {chk:.1e}")
    lines, n_planted, n_conf = [], 0, 0
    u0_ok = rec["boxes"]["U0"]["winding_g_int"] == len(inside(rec["boxes"]["U0"]["rect"], planted))
    for box in ("B1b", "B1c", "U", "B0", "U0"):
        tested = box in ("B1b", "B1c", "B0") or (box == "U0" and u0_ok)
        x = MM.locate(r, ev, Q, box, mutate if mutate == "nok" else None)
        rect = rec["boxes"][box]["rect"]
        strip = box in ("U", "U0")
        upper = inside(rect if strip else (rect[0], rect[1], 0.0, rect[3]), planted)
        if box in ("B1b", "B1c"):
            n_planted += len(upper)
        if tested:
            for m, want in ((("strip", upper),) if strip else (("sym", inside(rect, planted)), ("half", upper))):
                if m in x and abs(x[m]["winding"] - len(want)) > 1e-6:
                    bad.append(f"(b) {box} {m}: winding {x[m]['winding']:+.6f}, planted {len(want)}")
        if not x["ok"]:
            lines.append(f"    {box}: not located, {x['why']}")
            if tested and not any(b.startswith(f"(b) {box}") for b in bad):
                bad.append(f"(b) {box}: {x['why']}")
            continue
        got = [complex(*z) for z in x["roots"]]
        if len(got) != len(upper):
            lines.append(f"    {box}: {len(got)} roots for {len(upper)} planted")
            continue
        paired = MM.match(got, upper)
        worst, worst_conf, conf = 0.0, 0.0, 0
        for z, sp, w in zip(got, x["spread"], paired):
            d = abs(z - w)
            worst = max(worst, d)
            if sp < TOL:
                conf += 1
                worst_conf = max(worst_conf, d)
                if d > TOL and tested:
                    bad.append(f"(c) {box}: root {z:.4f} with spread {sp:.1e} is {d:.3f} from the planted {w:.4f}")
        if box in ("B1b", "B1c"):
            n_conf += conf
        lines.append(f"    {box:3s}: {x['N']} roots, {conf} with spread < {TOL}; largest distance to the planted "
                     f"{worst:.1e} (of those {worst_conf:.1e}), largest spread {max(x['spread'] + [0.0]):.1e}"
                     + ("" if tested else "  (printed, not tested)"))
    if ungated:
        lines.append("    count, not tested: " + "; ".join(ungated))
    return bad, lines, rec, n_planted, n_conf


def test(mutate=None):
    ok, n_planted, n_conf = True, 0, 0
    for name, extra in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            bad, lines, rec, npl, nc = run_case(name, extra, mutate, tmp)
        ok &= not bad
        n_planted += npl
        n_conf += nc
        wind = ", ".join(f"{b} {x['winding_g_int']}" for b, x in rec["boxes"].items())
        print(f"{'PASS' if not bad else 'FAIL'} {name}: count {wind}; {rec['evaluations']} evaluations", flush=True)
        for l in lines:
            print(l)
        for b in bad:
            print("    <- " + b)
    frac = n_conf / max(n_planted, 1)
    print(f"B1b and B1c: {n_conf} of {n_planted} planted zeros located with spread < {TOL} ({frac:.0%}; at least half "
          f"required): {'pass' if frac >= 0.5 else 'FAIL'}")
    ok &= frac >= 0.5
    print(("all criteria pass" if ok else "SOME CRITERIA FAIL") + (f" (mutate={mutate})" if mutate else ""), flush=True)
    return ok


def main():
    args, opts = W.parse(sys.argv[1:])
    if "controls" in opts:
        res = {m: test(m) for m in (None, "nok", "shift")}
        good = res[None] and not res["nok"] and not res["shift"]
        print(f"controls: test {'passes' if res[None] else 'FAILS'}, nok {'fails' if not res['nok'] else 'PASSES'}, "
              f"shift {'fails' if not res['shift'] else 'PASSES'} -> {'ok' if good else 'NOT OK'}")
        sys.exit(0 if good else 1)
    sys.exit(0 if test(opts.get("mutate")) else 1)


if __name__ == "__main__":
    main()
