#!/usr/bin/env python3
"""Review of #1079, check 4: the verdicts of Exc1079.lean's mutated instances, predicted before Lean decides them.

    python3 review_1079/lean_predict.py BOX.json POINT.json          from solver/proof

Mirrors the Krawczyk check on certify_M.krawczyk_test's Y g(0) and Y Dg(X) (arb balls): row i passes iff
|Yg_i| + sum_j |delta_ij - YJ_ij| w_j < w_i. Prints, for the clean instance and for each mutation, the largest
|K_i| / w_i and the verdict. A ratio within 1e-6 of 1 would make the prediction unsafe; it is flagged.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flint import arb, ctx  # noqa: E402
import certify_M as CM  # noqa: E402


def ratios(Yg, YJ, w):
    m = len(w)
    out = []
    for i in range(m):
        t = abs(Yg[i])
        for j in range(m):
            t = t + abs((arb(1) if i == j else arb(0)) - YJ[i][j]) * w[j]
        out.append(float((t / w[i]).upper()))
    return out


def main():
    with open(sys.argv[1]) as fh:
        bx = json.load(fh)
    with open(sys.argv[2]) as fh:
        pt = json.load(fh)
    res = CM.krawczyk_test(bx, pt, verbose=False)
    ctx.prec = CM.KPREC
    Yg, YJ, w = list(res["Yg"]), [list(r) for r in res["YJ"]], list(res["widths"])
    m = len(w)
    cases = {
        "clean": (Yg, YJ, w),
        "YJ transposed": (Yg, [[YJ[j][i] for j in range(m)] for i in range(m)], w),
        "planted outside 0 (Yg_0 + 2 w_0)": ([Yg[0] + 2 * w[0]] + Yg[1:], YJ, w),
        "planted outside 2 (Yg_2 + 2 w_2)": (Yg[:2] + [Yg[2] + 2 * w[2]] + Yg[3:], YJ, w),
        "planted inside 0 (Yg_0 + w_0 / 10)": ([Yg[0] + w[0] / 10] + Yg[1:], YJ, w),
    }
    print(f"{os.path.basename(sys.argv[1])}: half-widths " + ", ".join(f"{float(x.mid()):.0e}" for x in w))
    for name, (a, b, c) in cases.items():
        r = ratios(a, b, c)
        worst = max(r)
        flag = "  (TOO CLOSE TO 1)" if abs(worst - 1) < 1e-6 else ""
        print(f"  {name}: largest |K_i| / w_i = {worst:.4g} -> check = {'true' if worst < 1 else 'false'}{flag}")


if __name__ == "__main__":
    main()
