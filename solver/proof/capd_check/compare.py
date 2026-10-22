#!/usr/bin/env python3
"""Compare saved capd_check point runs with each other and with section 9's saved run (validated_int.py).

    python3 compare.py results/point_standard.txt [results/point_rho0.2.txt]

For each jump functional (Re/Im of J_D, J_S, J_C2, J_C0 at xi1 = 0.3; see capd_check.cpp) it prints the CAPD
enclosure and, for J_D, the one from ../validated_int_krawczyk_1e-9.json; then whether the enclosures intersect, and
the distance between their midpoints. Exact arithmetic on the stored bounds (python-flint at 320 bits).
"""
import json
import os
import sys

from flint import arb, acb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
ctx.prec = 320
NAMES = ["Re J_D", "Im J_D", "Re J_S", "Im J_S", "Re J_C2", "Im J_C2", "Re J_C0", "Im J_C0"]


def read_point(fn):
    J = {}
    for ln in open(fn):
        p = ln.split("#")[0].split()
        if len(p) == 4 and p[0] == "J":
            lo, hi = arb(float.fromhex(p[2])), arb(float.fromhex(p[3]))
            J[int(p[1])] = arb((lo + hi) / 2, (hi - lo) / 2)      # exact: the ball [lo, hi]
    return J


def show(x):
    return f"{x.mid().str(16, radius=False):>24s} +- {float(x.rad()):.1e}"


def cmp(a, b):
    meet = bool(a.overlaps(b))
    return f"intersect: {'yes' if meet else 'NO'}, |mid difference| = {float(abs(a.mid() - b.mid())):.1e}"


if __name__ == "__main__":
    runs = [(os.path.basename(f), read_point(f)) for f in sys.argv[1:]]
    s9 = json.load(open(os.path.join(HERE, "..", "validated_int_krawczyk_1e-9.json")))["point"]
    Dm1 = acb(arb(s9["Dm1"][0]), arb(s9["Dm1"][1]))
    Dp1 = acb(arb(s9["Dp1"][0]), arb(s9["Dp1"][1]))
    JD9 = Dp1 - Dm1.conjugate()
    ref = {0: JD9.real, 1: JD9.imag}
    name0, J0 = runs[0]
    for q in range(8):
        print(f"{NAMES[q]:8s} {name0}: {show(J0[q])}")
        if q in ref:
            print(f"{'':8s} section 9 (validated_int.py): {show(ref[q])}   {cmp(J0[q], ref[q])}")
        for name, J in runs[1:]:
            print(f"{'':8s} {name}: {show(J[q])}   {cmp(J0[q], J[q])}")
