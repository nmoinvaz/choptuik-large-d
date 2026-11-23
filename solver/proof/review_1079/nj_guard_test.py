#!/usr/bin/env python3
"""Review of #1079, check 1: can CERTIFY_M_NTAY / CERTIFY_M_NJAC give a wrong enclosure without an error?

    python3 review_1079/nj_guard_test.py            from solver/proof; seconds

certify_M_lohner.LohnerM._step forms the remainder of the step Jacobian from
    VR = taylor_V(s, ctrJ, NJ + 1),   ctrJ = truncate_ctx(ctr, NJ + 2) if NJ < N else ctr,
where ctr is the context of taylor_Y(.., N + 1): series of length N + 2. Its docstring states NJ <= N. Nothing checks
it, and the two environment options are read independently (CERTIFY_M_NTAY=10 alone leaves NJ at its default 12).
This test evaluates that line at one point of the start ray for several (N, NJ) and compares the coefficient
VR[NJ + 1] with the one from a context of the right length, taylor_Y(.., NJ + 1). Both are thin balls at a point, so a
correct line gives overlapping balls.

Expected, written before the run: (N, NJ) with NJ <= N + 1 agree; NJ >= N + 2 disagree (coefficients A_k with
k >= N + 2 are missing and are read as zero). Exit 0 iff that is what is found. The disagreement is the finding; the
agreeing rows are the control that the comparison can pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flint  # noqa: E402
from flint import acb, arb, ctx  # noqa: E402
import certify_M as CM  # noqa: E402
import certify_M_sys as CS  # noqa: E402
import certify_M_lohner as CL  # noqa: E402

P_X1 = ["3.4462182463667271490386733954779389341791395421154", "0.34760696521740738663621806885938050845968811925454",
        "0.80541758638529232157836211658860517241492322884730", "0.011780043848505832311374493364082518021457993309432"]


def main():
    M = 3
    ctx.prec = CM.PREC
    s = CS.SysM(M)
    p = CM.exact_p(P_X1)
    path = CM.path_for(M)
    _, vals, _, _ = CS.placeholder_start(p, M, path[0], path[1], dirs=[])
    leg = CS.legs_for(path)[0]
    Y0 = [acb(v.mid()) for v in vals] + [acb((CS.W0() * p[1]).mid())]
    T = acb(0)
    print(f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}; M = {M}, start of the ray, thin point")

    def vr_code(N, NJ):
        _, ctr = CL.taylor_Y(s, leg, T, Y0, N + 1)
        ctrJ = CL.truncate_ctx(ctr, NJ + 2) if NJ < N else ctr          # LohnerM._step, verbatim
        return CL.taylor_V(s, ctrJ, NJ + 1)[NJ + 1]

    def vr_true(NJ):
        _, ctr = CL.taylor_Y(s, leg, T, Y0, NJ + 1)
        return CL.taylor_V(s, ctr, NJ + 1)[NJ + 1]

    good = True
    for N, NJ in ((8, 6), (8, 8), (8, 9), (8, 10), (8, 12), (10, 12), (20, 12), (28, 16)):
        a, b = vr_code(N, NJ), vr_true(NJ)
        n = a.nrows()
        agree = all(a[i, j].overlaps(b[i, j]) for i in range(n) for j in range(n))
        dev = max(float(abs(a[i, j] - b[i, j]).lower()) for i in range(n) for j in range(n))
        size = max(float(abs(b[i, j]).upper()) for i in range(n) for j in range(n))
        want = NJ <= N + 1
        good &= (agree == want)
        print(f"N = {N:2d}, NJ = {NJ:2d}: remainder coefficient V_(NJ+1) agrees with the right-length one: {agree} "
              f"(largest certain difference {dev:.2e}, largest entry {size:.2e}); expected {want}: "
              f"{'ok' if agree == want else 'UNEXPECTED'}")
    print("NJ GUARD TEST: " + ("AS PREDICTED (NJ >= N + 2 gives a wrong remainder, silently)" if good else "NOT AS PREDICTED"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
