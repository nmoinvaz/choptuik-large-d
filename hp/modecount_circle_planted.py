#!/usr/bin/env python3
"""Planted-truth test of the small-circle winding and position (hp/modecount_circle.py, function circle) on the
synthetic det M of hp/modecount_moment_planted.py (the D 61.03, M 48, kf 23 structure).

    python3 modecount_circle_planted.py [--mutate=nodiv|offset]   exit 1 if a criterion fails (seconds)
    python3 modecount_circle_planted.py --controls                 the test and both mutations: exit 0 only if the
                                                                   test passes and each mutation fails

Planted extra zeros (with their conjugates): 1.80+0.60i, 2.20+0.90i and 3.02+0.05i. Circles, each with n0 = 16
and n0 = 20 (tol pi/4, as on the real data):
  one      centre 1.84+0.57i, R 0.2: the zero 1.80+0.60i, 0.05 from the centre    winding +1, position
  phys     centre lam_phys+0.03i, R 0.2: lam_phys                                   winding +1, position
  edge     centre 1.92+0.60i, R 0.2: the zero 0.12 from the centre, 0.08 inside     winding +1, position
  empty    centre 3.5+0.8i, R 0.2                                                    winding 0
  two      centre 2.0+0.75i, R 0.35: 1.80+0.60i and 2.20+0.90i                       winding +2, positions
  pair     centre 3.0, R 0.2: the conjugate pair 3.02 +- 0.05i                       winding +2, positions
  pole     centre 2.536+3.477i, R 0.2: the candidate pole n/2 - 30 lam_g + i omega   winding -1
Pass criteria: every winding as listed, to 1e-6; every position within 1e-4 of its planted zero in one, phys, two and
pair, with error estimate below 1e-3; in every case with positions each error is at most its estimate (the estimate
covers the error). The first version (af2f8db, positions for winding 1 only, without pair) passed, and its controls
failed; the cases two (positions) and pair were added with positions for windings 2 and 3, before their first run.
Controls, each must make the test fail: --mutate=nodiv leaves out the division by (lam - c)^W in the position sums;
--mutate=offset plants the zero of case one 0.01 from where the criteria expect it.
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
CASES = [("one", 1.84 + 0.57j, 0.2, 1, [ZEROS[0]]), ("phys", P.LP + 0.03j, 0.2, 1, [complex(P.LP)]),
         ("edge", 1.92 + 0.60j, 0.2, 1, [ZEROS[0]]), ("empty", 3.5 + 0.8j, 0.2, 0, None),
         ("two", 2.0 + 0.75j, 0.35, 2, ZEROS[:2]), ("pair", 3.0 + 0j, 0.2, 2, [ZEROS[2], ZEROS[2].conjugate()]),
         ("pole", POLE, 0.2, -1, None)]


def test(mutate=None):
    zeros = list(ZEROS)
    if mutate == "offset":
        zeros[0] += 0.01
    f = P.det_for(zeros)

    def det(lam):
        la, arg, _, _ = f(lam.real, lam.imag)
        return la, arg
    ok = True
    for n0 in (16, 20):
        for name, c, R, want, z in CASES:
            rec = C.circle(det, c, R, math.pi / 4, n0, "nodiv" if mutate == "nodiv" else None)
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
                    if name in ("one", "phys", "two", "pair") and (err > 1e-4 or e > 1e-3):
                        bad.append("position not within 1e-4 or estimate not below 1e-3")
            elif z is not None:
                bad.append("no positions")
            ok &= not bad
            print(f"{'PASS' if not bad else 'FAIL'} {line}" + ("" if not bad else "  <- " + "; ".join(bad)), flush=True)
    print(("all cases pass" if ok else "SOME CASES FAIL") + (f" (mutate={mutate})" if mutate else ""), flush=True)
    return ok


def main():
    args, opts = parse(sys.argv[1:])
    if "controls" in opts:
        res = {m: test(m) for m in (None, "nodiv", "offset")}
        good = res[None] and not res["nodiv"] and not res["offset"]
        print(f"controls: test {'passes' if res[None] else 'FAILS'}, nodiv {'fails' if not res['nodiv'] else 'PASSES'}, "
              f"offset {'fails' if not res['offset'] else 'PASSES'} -> {'ok' if good else 'NOT OK'}")
        sys.exit(0 if good else 1)
    sys.exit(0 if test(opts.get("mutate")) else 1)


if __name__ == "__main__":
    main()
