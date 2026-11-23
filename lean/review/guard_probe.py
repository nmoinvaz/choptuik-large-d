#!/usr/bin/env python3
"""How strong is the exporter's guard on Y for M = 3 (#1077, notes/LEAN_REVIEW_1077.md)?

lean/tools/export_krawczyk.py certM recomputes Y and asserts that Y l (c + Q q) reproduces krawczyk_test's Y g, by
`.equal` if arb has it and by the printed strings otherwise. python-flint 0.9.0's arb has no `equal`, so the strings
are compared. This script repeats that comparison with Y[0][0] multiplied by 1 + 2^-K and prints, for each K, whether
the string guard still passes (it should not) and whether an exact comparison of midpoints and radii sees the change.

    python3 guard_probe.py        exit 0; prints the table (a measurement, not a pass/fail check)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "solver", "proof")
sys.path.insert(0, PROOF)
import certify_M as CM  # noqa: E402
from flint import arb, arb_mat, ctx  # noqa: E402


def main():
    bx = json.load(open(os.path.join(PROOF, "data", "A_M_box_M3_1e-10.json")))
    res = CM.krawczyk_test(bx, verbose=False)
    m = bx["M"] + 1
    ctx.prec = CM.KPREC
    ell = CM.jump_rows(CM.CS.SysM(bx["M"]), None)
    c, Q, q, varb = CM.CL.load_lohner(bx["sets"])
    D0 = arb_mat(CM.apply_var(ell, varb))
    Yb = arb_mat([[arb(D0[i, j].mid()) for j in range(m)] for i in range(m)]).inv()
    print(f"arb has `equal`: {hasattr(arb, 'equal')}; printed Y g[0] of krawczyk_test: {res['Yg'][0]}")
    print("  K     string guard passes   exact (mid, rad) comparison passes")
    for K in (None, 700, 400, 200, 100, 60, 40, 30, 20, 10):
        Y = arb_mat([[arb(Yb[i, j].mid()) for j in range(m)] for i in range(m)])
        if K is not None:
            Y[0, 0] = arb((Y[0, 0] * (1 + arb(2) ** (-K))).mid())
        Yg2 = CM.apply_set(Y * ell, c, Q, q)
        s_ok = all(str(Yg2[i]) == str(res["Yg"][i]) for i in range(m))
        e_ok = all(Yg2[i].mid().man_exp() == res["Yg"][i].mid().man_exp() and
                   Yg2[i].rad().man_exp() == res["Yg"][i].rad().man_exp() for i in range(m))
        print(f"  {'none' if K is None else K:<5} {s_ok!s:21s} {e_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
