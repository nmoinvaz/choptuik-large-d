#!/usr/bin/env python3
"""Planted-truth test of the argument-principle count (hp/modecount_wind.py, function count) on a synthetic det M.

    python3 modecount_planted.py [--mutate=flat|nolng|nocorner]   exits 1 if any case fails (about 2 min)
    python3 modecount_planted.py --controls                        the test and the three mutations: exit 0 only if the
                                                                   test passes and each mutation fails as stated below

Group D30. The synthetic determinant has the structure found for the DSS problem at D = 30 (omega, lam_g, lam_phys,
the rows of horizon poles on Im = +-omega, +-3 omega, +-5 omega, exp(57 lam + 0.3 lam^2) growth, copies 2 i k omega of
the zeros), and each case plants extra zeros at known places. The count runs through the same code path as on the real
det M (same geometry, trend, factors, sampling and bookkeeping), with an evaluation cache in a temporary directory, and
the default options (h0 0.4).

Group D61 (#242). The synthetic determinant and layout of hp/modecount_moment_planted.py (D 61.03, M 48, kf 23 of #212:
growth exp(41.2 lam + 1.40 lam^2), the count's aLL, aL, a1, a2, aR), with the real count's options (h0 0.6, tol pi/4,
--resolve=last). On the edge Im = y2 its phase of g grows by about 11 rad per unit of Re, so an interval of the
initial grid (0.571) can turn by 2 pi and look flat. Cases:
  random0, random4, random7, held1, held5, held9   the moment test's configurations whose U and full the old rule
                                                   miscounts (#230: U 5, -4, 2, 1, 3, -4 for 8, 3, 4, 4, 9, 4 planted)
  pair <edge> f<f>    two zeros 0.008 apart, 0.01 inside the edge H2 (Im = y1, in B1b) or T (Im = y2, in U), at the
                      fraction f of an interval of the initial grid; they turn the phase by 2 pi between two samples
  row k<k> d<d>       a row of k zeros per initial interval of T, d below it in U, over the length of T
  long row k<k> d<d>  the same row from Re aLL - 2 to aR + 2: every interval of T and T0 holds whole turns
These were seen while the rule was chosen (development cases). Held-out cases, drawn with seed 242 and first run under
the criteria below: h242_0 .. h242_7 (the moment test's generator: 1-3 zeros in the upper half of B1b, 1-2 in that of
B1c, 0-9 in U), hpair0 .. hpair3 (a pair as above on a random edge of the network, at a random fraction, 0.005-0.03
inside, inside the box on the edge's left), hrow0 .. hrow2 (a long row below T with k in {2, 4}, d in 0.08-0.3 and a
random offset along Re).

Pass criteria, fixed before running: in every box (all eight, U, U0 and full included), wind(g) equals the number of
planted extra zeros inside it (counting the conjugates for the boxes symmetric about the real axis) to 1e-6, every box
is resolved, and every phase step is at most pi/4.
Controls, each must make the test fail: --mutate=flat restores the old rule (no alias rule) and must fail each of the
six moment-test configurations and at least one long row; --mutate=nolng drops ln|g| from the alias rule's scale and
must fail at least one pair; --mutate=nocorner drops the corner scales and must fail at least one long row.
"""
import cmath
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modecount_wind as W  # noqa: E402
import modecount_moment_planted as MP  # noqa: E402

OM, LG, LP, N = 3.1987, 0.8265, 2.0797, 28.07
BASE_ZEROS = [z + 2j * k * OM for z in (0.0, LP, LG) for k in range(-4, 5)]
POLES = [N / 2 - LG * (kk + 1) + s * 1j * OM for kk in range(101) for s in (1, -1, 3, -3, 5, -5)]
CASES = {
    "none": [],
    "B1c, complex pair": [4.0 + 0.4j],
    "B1c, real": [3.3 + 0j],
    "U, other-sector real mode": [1.9 + 1j * OM],
    "U, near the top edge": [5.0 + 4.6j],
    "B0, slowly growing pair": [0.07 + 0.5j],
    "B1b, second mode next to lam_phys": [2.4 + 0.2j],
    "B1a, near an edge (0.05 from Re = aL)": [0.35 + 1.0j],
}
SIX = ("random0", "random4", "random7", "held1", "held5", "held9")


def det_for(extra):
    zeros = BASE_ZEROS + extra + [z.conjugate() for z in extra if z.imag != 0]

    def det(re, im):
        lam = complex(re, im)
        v = 57 * lam - 700 + 0.3 * lam * lam
        v += sum(cmath.log(lam - z) for z in zeros)
        v -= sum(cmath.log(lam - p) for p in POLES)
        return v.real, math.remainder(v.imag, 2 * math.pi), 0.0, 0.0
    return det


def expected(rect, extra):
    r0, r1, i0, i1 = rect
    pts = extra + [z.conjugate() for z in extra if z.imag != 0]
    return sum(1 for z in pts if r0 < z.real < r1 and i0 < z.imag < i1)


def d61_cases():
    """the group D61 cases, development and held out (seed 242)"""
    L, Y1, Y2 = MP.LAYOUT, MP.Y1, MP.Y2
    hT = (L["aR"] - L["aL"]) / math.ceil((L["aR"] - L["aL"]) / 0.6)
    hH2 = (L["a2"] - L["a1"]) / math.ceil((L["a2"] - L["a1"]) / 0.6)
    cases = {name: MP.CASES[name] for name in SIX}
    for f in (0.5, 0.25):
        x = L["a2"] - (1 + f) * hH2
        cases[f"pair H2 f{f}"] = [complex(x - 0.004, Y1 - 0.01), complex(x + 0.004, Y1 - 0.01)]
    for f in (0.5, 0.25, 0.1):
        x = L["aR"] - (4 + f) * hT
        cases[f"pair T f{f}"] = [complex(x - 0.004, Y2 - 0.01), complex(x + 0.004, Y2 - 0.01)]
    nT = round((L["aR"] - L["aL"]) / hT)
    for k, d in ((2, 0.15), (4, 0.08)):
        cases[f"row k{k} d{d}"] = [complex(L["aR"] - (j + 0.5) * hT / k, Y2 - d) for j in range(k * nT)]
    for k, d in ((2, 0.15), (2, 0.3), (4, 0.1)):
        s = hT / k
        cases[f"long row k{k} d{d}"] = [complex(L["aR"] + 2.0 - (j + 0.5) * s, Y2 - d)
                                        for j in range(int((L["aR"] - L["aLL"] + 4.0) / s))]
    # held out
    cases.update(MP.random_cases(242, 8, "h242_"))
    rng = np.random.default_rng(242)
    # edges as (start, end); the box on the left of the direction of travel is inside the count's contours
    edges = [(complex(L["aR"], Y1), complex(L["a2"], Y1)), (complex(L["a2"], Y1), complex(L["a1"], Y1)),
             (complex(L["a1"], Y1), complex(L["aL"], Y1)), (complex(L["aR"], Y2), complex(L["aL"], Y2)),
             (complex(L["aR"], Y1), complex(L["aR"], Y2)), (complex(L["a2"], 0.0), complex(L["a2"], Y1))]
    for i in range(4):
        z0, z1 = edges[int(rng.integers(len(edges)))]
        n = math.ceil(abs(z1 - z0) / 0.6)
        j = int(rng.integers(n))
        f = float(rng.uniform(0.05, 0.95))
        d = float(rng.uniform(0.005, 0.03))
        u = (z1 - z0) / abs(z1 - z0)
        c = z0 + (j + f) * (z1 - z0) / n + 1j * u * d          # 1j * u points to the left of the direction of travel
        cases[f"hpair{i}"] = [c - 0.004 * u, c + 0.004 * u]
    for i in range(3):
        k = int(rng.choice([2, 4]))
        d = float(rng.uniform(0.08, 0.3))
        s = hT / k
        off = float(rng.uniform(0.0, s))
        cases[f"hrow{i}"] = [complex(L["aR"] + 2.0 - off - j * s, Y2 - d)
                             for j in range(int((L["aR"] - L["aLL"] + 4.0) / s))]
    return cases


def run(group, name, extra, mutate, tmp, i):
    if group == "D30":
        info = dict(D=N + 2, M=0, omega=OM, lam_g=LG, lam_f=LP)
        opts = {} if mutate is None else dict(mutate=mutate)
        rec = W.count(det_for(extra), info, POLES, opts, lambda m: None, f"planted{i}", os.path.join(tmp, "res.jsonl"))
        planted = extra
    else:
        info = dict(D=MP.N + 2, M=0, omega=MP.OM, lam_g=MP.LG, lam_f=MP.LP)
        opts = dict(MP.LAYOUT, data=tmp, h0="0.6", resolve="last", **({} if mutate is None else dict(mutate=mutate)))
        rec = W.count(MP.det_for(extra), info, MP.POLES, opts, lambda m: None, f"p61_{i}",
                      os.path.join(tmp, "res.jsonl"))
        planted = extra
    bad = []
    for b, x in rec["boxes"].items():
        want = expected(x["rect"], planted)
        if x["winding_g_int"] != want or abs(x["winding_g"] - want) > 1e-6 or not x["resolved"]:
            bad.append(f"{b}: wind(g) {x['winding_g']:+.3f}, planted {want}" + ("" if x["resolved"] else ", unresolved"))
    mx = max(e["max_step"] for e in rec["edges"].values())
    if mx > math.pi / 4 + 1e-12:
        bad.append(f"max phase step {mx:.3f} > pi/4")
    return bad, rec, mx


def test(mutate=None):
    groups = [("D30", CASES), ("D61", d61_cases())]
    failed, ok, nev = set(), True, 0
    for group, cases in groups:
        for i, (name, extra) in enumerate(cases.items()):
            with tempfile.TemporaryDirectory() as tmp:
                W.DATA = tmp
                bad, rec, mx = run(group, name, extra, mutate, tmp, i)
            ok &= not bad
            nev += rec["evaluations"]
            if bad:
                failed.add(name)
            got = ", ".join(f"{b} {x['winding_g_int']}" for b, x in rec["boxes"].items())
            print(f"{'PASS' if not bad else 'FAIL'} {group} {name}: {got}; {rec['evaluations']} evaluations, max step "
                  f"{mx:.3f}" + ("" if not bad else "  <- " + "; ".join(bad)), flush=True)
    print(("all cases pass" if ok else "SOME CASES FAIL") + f" ({nev} evaluations"
          + (f", mutate={mutate})" if mutate else ")"), flush=True)
    return ok, failed


def main():
    args, opts = W.parse(sys.argv[1:])
    if "controls" in opts:
        res = {m: test(m) for m in (None, "flat", "nolng", "nocorner")}
        longs = {n for n in d61_cases() if n.startswith("long row")}
        pairs = {n for n in d61_cases() if n.startswith("pair")}
        need = {"flat": all(n in res["flat"][1] for n in SIX) and bool(longs & res["flat"][1]),
                "nolng": bool(pairs & res["nolng"][1]), "nocorner": bool(longs & res["nocorner"][1])}
        for m in need:
            print(f"control {m}: fails {sorted(res[m][1])} -> {'as required' if need[m] else 'NOT AS REQUIRED'}")
        good = res[None][0] and all(need.values())
        print(f"controls: test {'passes' if res[None][0] else 'FAILS'} -> {'ok' if good else 'NOT OK'}")
        sys.exit(0 if good else 1)
    sys.exit(0 if test(opts.get("mutate"))[0] else 1)


if __name__ == "__main__":
    main()
