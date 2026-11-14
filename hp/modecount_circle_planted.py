#!/usr/bin/env python3
"""Planted-truth test of the small-circle winding and position (hp/modecount_circle.py, function circle) on the
synthetic det M of hp/modecount_moment_planted.py (the D 61.03, M 48, kf 23 structure).

    python3 modecount_circle_planted.py [--mutate=nodiv|offset|flat]   exit 1 if a criterion fails (seconds)
    python3 modecount_circle_planted.py --controls                      the test and the three mutations: exit 0 only
                                                                        if the test passes and each mutation fails

Planted extra zeros (with their conjugates): 1.80+0.60i, 2.20+0.90i and 3.02+0.05i. Circles with n0 = 20 and tol
pi/4, as on the real data (--n0=16 tests 16 points):
  one      centre 1.84+0.57i, R 0.2: the zero 1.80+0.60i, 0.05 from the centre    winding +1, position
  phys     centre lam_phys+0.03i, R 0.2: lam_phys                                   winding +1, position
  edge     centre 1.92+0.60i, R 0.2: the zero 0.12 from the centre, 0.08 inside     winding +1, position
  empty    centre 3.5+0.8i, R 0.2                                                    winding 0
  two      centre 2.0+0.75i, R 0.35: 1.80+0.60i and 2.20+0.90i                       winding +2, positions
  pair     centre 3.0, R 0.2: the conjugate pair 3.02 +- 0.05i                       winding +2, positions
  pole     centre 2.536+3.477i, R 0.2: the candidate pole n/2 - 30 lam_g + i omega   winding -1
  four     centre 3.12, R 0.36, with the extra pair 3.25 +- 0.12i planted for this case alone: 3.02 +- 0.05i and
           3.25 +- 0.12i, at 0.31 R and 0.49 R                                       winding +4, positions
Pass criteria: every winding as listed, to 1e-6; every position within 1e-4 of its planted zero in one, phys, two,
pair and four, with error estimate below 1e-3; in every case with positions each error is at most its estimate (the estimate
covers the error). History: the first version (af2f8db, positions for winding 1 only, without pair, n0 16 and 20)
passed, and its controls failed. The second (4a004e6) added positions in two and the case pair, and failed in two at
n0 = 16 only: errors 6.3e-5, estimates 1.3e-3 (the zeros lie at 0.71 R, where the 8-point sums of the estimate
converge slowly). At n0 = 20 every case passed, so the real circles use n0 = 20 and the test runs it by default.
The case four was added with positions for windings up to 4 (#708's circles at kf 35 wind +4).
Controls, each must make the test fail: --mutate=nodiv leaves out the division by (lam - c)^W in the position sums;
--mutate=offset plants the zero of case one 0.01 from where the criteria expect it.

Alias cases (#257, the alias rule of #242 on the circle's arcs). A pair of zeros 0.008 apart along the circle, 0.002
inside (in) or outside (out) the circle of case empty (3.5+0.8i, R 0.2), centred at the fraction f = 0.3, 0.5, 0.7 of
the fourth arc of the initial grid: the pair turns the phase by about 2 pi - 0.26 rad between two samples, so the step
test alone sees a small step. Pass: winding +2 (in) or 0 (out) to 1e-6, and every circle resolved (steps <= tol, no
mismatch, every arc within the alias rule). --mutate=flat restores the old rule (no alias rule, one verification
pass) and must fail at least one alias case; --controls runs it too.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modecount_circle as C  # noqa: E402
import modecount_moment_planted as P  # noqa: E402
from modecount_wind import parse  # noqa: E402

ZEROS = [1.80 + 0.60j, 2.20 + 0.90j, 3.02 + 0.05j]
POLE = P.N / 2 - P.LG * 30 + 1j * P.OM
N0 = 20
CASES = [("one", 1.84 + 0.57j, 0.2, 1, [ZEROS[0]]), ("phys", P.LP + 0.03j, 0.2, 1, [complex(P.LP)]),
         ("edge", 1.92 + 0.60j, 0.2, 1, [ZEROS[0]]), ("empty", 3.5 + 0.8j, 0.2, 0, None),
         ("two", 2.0 + 0.75j, 0.35, 2, ZEROS[:2]), ("pair", 3.0 + 0j, 0.2, 2, [ZEROS[2], ZEROS[2].conjugate()]),
         ("pole", POLE, 0.2, -1, None)]
# case four's own extra pair, planted only for it, so the other cases' zeros are unchanged
FOUR = 3.25 + 0.12j
CASES_FOUR = [("four", 3.12 + 0j, 0.36, 4, [ZEROS[2], ZEROS[2].conjugate(), FOUR, FOUR.conjugate()])]
ALIAS_C, ALIAS_R, ALIAS_ARC, ALIAS_D, ALIAS_SEP = 3.5 + 0.8j, 0.2, 3, 0.002, 0.008


def alias_pair(side, f, n0):
    """two zeros ALIAS_SEP apart along the circle of case empty, ALIAS_D inside (side -1) or outside (+1), centred at
    the fraction f of arc ALIAS_ARC of the n0-point grid"""
    t = 2 * math.pi * (ALIAS_ARC + f) / n0
    u = complex(math.cos(t), math.sin(t))
    mid = ALIAS_C + (ALIAS_R + side * ALIAS_D) * u
    return [mid - 0.5j * ALIAS_SEP * u, mid + 0.5j * ALIAS_SEP * u]


def alias_test(mutate=None):
    """the alias cases; True when every one passes"""
    ok = True
    for side, name, want in ((-1, "in", 2), (1, "out", 0)):
        for fr in (0.3, 0.5, 0.7):
            f = P.det_for(list(ZEROS) + alias_pair(side, fr, N0))

            def det(lam):
                la, arg, _, _ = f(lam.real, lam.imag)
                return la, arg
            rec = C.circle(det, ALIAS_C, ALIAS_R, math.pi / 4, N0, mutate)
            bad = []
            if abs(rec["winding"] - want) > 1e-6:
                bad.append(f"winding {rec['winding']:+.6f}, planted {want}")
            if not rec["resolved"]:
                bad.append("unresolved")
            ok &= not bad
            print(f"{'PASS' if not bad else 'FAIL'} n0 {N0} pair {name} f {fr}: winding {rec['winding']:+.6f}, "
                  f"{rec['points']} points, max step {rec['max_step']:.3f}, alias bisections {rec['alias_bisections']}, "
                  f"largest arc*scale {rec['max_scale_h']:.2f}" + ("" if not bad else "  <- " + "; ".join(bad)),
                  flush=True)
    return ok


def test(mutate=None):
    zeros = list(ZEROS)
    if mutate == "offset":
        zeros[0] += 0.01
    f, f4 = P.det_for(zeros), P.det_for(zeros + [FOUR])

    def det_of(g):
        def det(lam):
            la, arg, _, _ = g(lam.real, lam.imag)
            return la, arg
        return det
    ok = True
    for n0 in (N0,):
        for name, c, R, want, z in CASES + CASES_FOUR:
            det = det_of(f4 if name == "four" else f)
            rec = C.circle(det, c, R, math.pi / 4, n0, mutate if mutate in ("nodiv", "flat") else None)
            bad = []
            if abs(rec["winding"] - want) > 1e-6:
                bad.append(f"winding {rec['winding']:+.6f}, planted {want}")
            line = f"n0 {n0} {name:5s}: winding {rec['winding']:+.6f}, {rec['points']} points, max step {rec['max_step']:.3f}"
            if z is not None and "zeros" in rec and len(rec["zeros"]) == len(z):
                got = [complex(*x) for x in rec["zeros"]]
                want_z = __import__("modecount_moment").match(got, z)
                for g, w, e in zip(got, want_z, rec["zeros_est"]):
                    err = abs(g - w)
                    line += f", position error {err:.1e}, estimate {e:.1e}"
                    if err > max(e, 1e-12):
                        bad.append("the estimate does not cover the error")
                    if name in ("one", "phys", "two", "pair", "four") and (err > 1e-4 or e > 1e-3):
                        bad.append("position not within 1e-4 or estimate not below 1e-3")
            elif z is not None:
                bad.append("no positions")
            ok &= not bad
            print(f"{'PASS' if not bad else 'FAIL'} {line}" + ("" if not bad else "  <- " + "; ".join(bad)), flush=True)
    ok &= alias_test("flat" if mutate == "flat" else None)
    print(("all cases pass" if ok else "SOME CASES FAIL") + (f" (mutate={mutate})" if mutate else ""), flush=True)
    return ok


def main():
    global N0
    args, opts = parse(sys.argv[1:])
    N0 = int(opts.get("n0", 20))
    if "controls" in opts:
        res = {m: test(m) for m in (None, "nodiv", "offset", "flat")}
        good = res[None] and not res["nodiv"] and not res["offset"] and not res["flat"]
        print(f"controls: test {'passes' if res[None] else 'FAILS'}, nodiv {'fails' if not res['nodiv'] else 'PASSES'}, "
              f"offset {'fails' if not res['offset'] else 'PASSES'}, flat {'fails' if not res['flat'] else 'PASSES'}"
              f" -> {'ok' if good else 'NOT OK'}")
        sys.exit(0 if good else 1)
    sys.exit(0 if test(opts.get("mutate")) else 1)


if __name__ == "__main__":
    main()
