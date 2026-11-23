#!/usr/bin/env python3
"""A planted-truth control through lean/tools/export_krawczyk.py's `capd` path (#1077, notes/LEAN_REVIEW_1077.md).

The map is affine with a known zero, F(x) = A (x - x*), A = [[1, 100], [0, 1]] (far from symmetric, so that a
transposed Jacobian is not a near-inverse of the same Y), x^ = (0.5, 0.25), h = 1e-9. Its CAPD-format file
(F(x^) as outward hex-double intervals, DF as point intervals) goes through the exporter, unmodified, and the Lean
file it writes is compiled. Pass criteria, written before the first run:

  in    x* = x^ + (h/2, -h/4), inside the box     the exporter's `check = true` theorem compiles
  out   x* = x^ + (2h, 0), outside the box        it does not compile
  inT   as `in`, with DF transposed in the file   it does not compile (rows of Y F about 0.5, 2450 instead of 0.5, 0.25)
        (what an exporter that transposes J would hand to Lean)

    python3 planted.py WORKDIR      exit 0 iff the three verdicts are as stated
WORKDIR is a scratch directory for the JSON and Lean files (not inside the repository).
"""
import json
import math
import os
import subprocess
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
LEAN = os.path.dirname(HERE)
EXPORTER = os.path.join(LEAN, "tools", "export_krawczyk.py")
A = [[Fr(1), Fr(100)], [Fr(0), Fr(1)]]
XHAT = ["0.5", "0.25"]
H = Fr(1, 10**9)
HFILE = "0x1.12e0be826d695p-30"          # the double nearest 1e-9, as in capd_enclosures_1e-9.json


def outward(q):
    """the tightest interval of doubles that contains the rational q"""
    f = float(q)
    lo = f if Fr(f) <= q else math.nextafter(f, -math.inf)
    hi = f if Fr(f) >= q else math.nextafter(f, math.inf)
    assert Fr(lo) <= q <= Fr(hi)
    return [lo.hex(), hi.hex()]


def write_case(work, name, d, transpose):
    Fx = [-(A[i][0] * d[0] + A[i][1] * d[1]) for i in range(2)]             # F(x^) = A (x^ - x*) = -A d
    DF = [[A[j][i] if transpose else A[i][j] for j in range(2)] for i in range(2)]
    doc = {"producer": "lean/review/planted.py (a planted affine map, not a certificate)",
           "xhat": XHAT, "xhat_enclosure": [[float(Fr(x)).hex()] * 2 for x in XHAT], "h": HFILE,
           "F_at_xhat": [outward(v) for v in Fx],
           "DF_over_X": [[[float(DF[i][j]).hex()] * 2 for j in range(2)] for i in range(2)]}
    js, out = os.path.join(work, name + ".json"), os.path.join(work, name + ".lean")
    with open(js, "w") as fh:
        json.dump(doc, fh)
    subprocess.run([sys.executable, EXPORTER, "capd", js, name, out], check=True)
    r = subprocess.run(["lake", "env", "lean", out], cwd=LEAN, capture_output=True, text=True)
    return r.returncode == 0


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    work = os.path.abspath(sys.argv[1])
    assert not work.startswith(os.path.dirname(LEAN) + os.sep), "WORKDIR must be outside the repository"
    os.makedirs(work, exist_ok=True)
    cases = [("PlantedIn", (H / 2, -H / 4), False, True), ("PlantedOut", (2 * H, Fr(0)), False, False),
             ("PlantedInT", (H / 2, -H / 4), True, False)]
    allok = True
    for name, d, tr, expect in cases:
        got = write_case(work, name, d, tr)
        ok = got == expect
        allok &= ok
        print(f"  {name:10s} the exporter's `check = true` theorem compiles: {got!s:5s} expected {expect!s:5s} "
              f"{'as stated' if ok else 'NOT AS STATED'}")
    print("planted control:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
