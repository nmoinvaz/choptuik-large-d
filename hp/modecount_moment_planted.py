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
  random0 .. random7, drawn with seed 230 and first run with the criteria below: 1-3 zeros in the upper half of B1b,
           1-2 in that of B1c, 0-9 in U, each at least 0.1 from its box's edges, 0.25 from the others and lam_phys, and
           0.2 from the candidate poles.

Pass criteria:
  (a) the count: in every box wind(g) equals the number of planted zeros inside (conjugates included), every edge is
      resolved, every phase step is at most pi/4;
  (b) the winding of each moment contour equals the number of planted zeros inside it, to 1e-6 (B1b and B1c: sym and
      half; U: strip);
  (c) B1b and B1c: each planted zero of the upper half pairs (the pairing that minimises the largest distance) with a
      reported root within 0.05, the accuracy that centring circles of radius 0.1 or more needs, and that distance is
      at most the root's spread (or 1e-3, whichever is larger);
  (d) ln g recomputed by the moment code agrees with the samples stored in the count's record to 1e-4.
  U is reported, not tested: its roots are used on real data only if every case meets (c) for U too (printed).
Controls, each must make the test fail: --mutate=nok drops the factor k of the integration by parts in the moment code;
--mutate=shift plants every extra zero 0.1 to the right of where the criteria expect it.

History: the first version (commit 38ab804) estimated a root's error as its distance from the trapezoid rule's root
and required, in (c), estimates below 0.02 in mode sym and every root with a smaller estimate within 0.02. It failed
on row and edges: the cubic roots were within 1.1e-2 of the planted zeros, but the trapezoid rule is 10-100 times
worse, so the estimates were 0.1-0.6. The estimator was replaced by the spread and the random cases were added.
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


def random_cases(seed=230, n=8):
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
        cases[f"random{i}"] = z
    return cases


CASES.update(random_cases())


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
    bad = []
    for b, x in rec["boxes"].items():
        want = len(inside(x["rect"], planted))
        if x["winding_g_int"] != want or abs(x["winding_g"] - want) > 1e-6:
            bad.append(f"(a) {b}: wind(g) {x['winding_g']:+.3f}, planted {want}")
        if not x["resolved"]:
            bad.append(f"(a) {b} not resolved")
    mx = max(e["max_step"] for e in rec["edges"].values())
    if mx > math.pi / 4 + 1e-12:
        bad.append(f"(a) max phase step {mx:.3f} > pi/4")
    r, ev = MM.load_count(tmp, tag)
    Q = MM.factored(POLES, r)
    chk = MM.stored_check(r, MM.LnG(r, ev, Q), list(r["edges"]))
    if chk > 1e-4:
        bad.append(f"(d) ln g against the stored samples {chk:.1e}")
    lines, u_ok = [], True
    for box in ("B1b", "B1c", "U"):
        x = MM.locate(r, ev, Q, box, mutate if mutate == "nok" else None)
        rect = rec["boxes"][box]["rect"]
        upper = inside(rect if box == "U" else (rect[0], rect[1], 0.0, rect[3]), planted)
        for m, want in ((("strip", upper),) if box == "U" else (("sym", inside(rect, planted)), ("half", upper))):
            if abs(x[m]["winding"] - len(want)) > 1e-6:
                bad.append(f"(b) {box} {m}: winding {x[m]['winding']:+.6f}, planted {len(want)}")
        if not x["ok"]:
            lines.append(f"    {box}: not located, {x['why']}")
            if box == "U":
                u_ok = False
            elif not any(b.startswith(f"(b) {box}") for b in bad):
                bad.append(f"(b) {box}: {x['why']}")
            continue
        got = [complex(*z) for z in x["roots"]]
        paired = MM.match(got, upper) if len(got) == len(upper) else None
        worst, cover = 0.0, True
        for i, z in enumerate(got):
            d = abs(z - paired[i]) if paired else math.inf
            worst = max(worst, d)
            ok = d <= TOL and d <= max(x["spread"][i], 1e-3)
            cover &= d <= max(x["spread"][i], 1e-3)
            if not ok:
                if box == "U":
                    u_ok = False
                else:
                    bad.append(f"(c) {box}: root {z:.4f}, spread {x['spread'][i]:.1e}, is {d:.3f} from the planted")
        sp = max(x["spread"] + [0.0])
        lines.append(f"    {box:3s}: {x['N']} roots, largest distance to the planted {worst:.1e}, largest spread "
                     f"{sp:.1e}, spread covers the error: {'yes' if cover else 'NO'}"
                     + ("  (U: reported, not a criterion)" if box == "U" else ""))
    return bad, lines, rec, u_ok


def test(mutate=None):
    ok, u_all = True, True
    for name, extra in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            bad, lines, rec, u_ok = run_case(name, extra, mutate, tmp)
        ok &= not bad
        u_all &= u_ok
        wind = ", ".join(f"{b} {x['winding_g_int']}" for b, x in rec["boxes"].items())
        print(f"{'PASS' if not bad else 'FAIL'} {name}: count {wind}; {rec['evaluations']} evaluations", flush=True)
        for l in lines:
            print(l)
        for b in bad:
            print("    <- " + b)
    print(f"U meets (c) in every case: {'yes' if u_all else 'no'} (U's roots are used on real data only if yes)")
    print(("all cases pass" if ok else "SOME CASES FAIL") + (f" (mutate={mutate})" if mutate else ""), flush=True)
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
