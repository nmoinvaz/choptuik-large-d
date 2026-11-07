#!/usr/bin/env python3
"""Three small checks of the review of proof/A_M2.md (proof/A_M2_review.md sections 2 and 3, issue 4).

  * the saved ball strings (certify_M_lohner.s_arb) contain the ball when read back at 256 and 768 bits (20000 random
    balls);
  * the run's parameter box (built at 256 bits from the decimal strings) equals the Krawczyk test's X (768 bits);
  * the Cauchy radius of certify_M2.start_data2: hw and nu are taken as float(upper bound), which rounds toward zero;
    for the given chart and box, is anything lost, and does the bidisc still lie in the lemma's complex box
    x + arb(0, c) + i arb(0, c)?

    python3 review_M2_misc.py data/A_M2_box_M5_1e-10.json [--mutate]
Exit status 1 if a check fails. --mutate (must fail): the ball strings printed without their radius, and hw taken
1e-11 below its upper bound.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb, acb, ctx  # noqa: E402
import certify_M_lohner as CL  # noqa: E402
import certify_M as CM  # noqa: E402
import certify_M2 as C  # noqa: E402

MUTATE = "--mutate" in sys.argv


def roundtrip(N=20000):
    ctx.prec = 256
    bad = 0
    random.seed(1)
    for _ in range(N):
        x = arb(random.uniform(-1, 1)) * arb(10) ** random.randint(-30, 30) / 3 + \
            arb(0, random.choice([0, 1e-80, 1e-77, 1e-60, 1e-20]) * random.random())
        s = x.mid().str(30, radius=False) if MUTATE else CL.s_arb(x)
        for p in (256, 768):
            ctx.prec = p
            if not arb(s).contains(x):
                bad += 1
            ctx.prec = 256
    print(f"saved ball strings read back: {bad} failures of {2 * N}")
    return bad == 0


def boxes(bx):
    ok = True
    for s in [bx["h"]] + list(bx["w"]):
        ctx.prec = 768
        X = arb(0, arb(s))
        ctx.prec = 256
        P = acb(arb(0, arb(s)))
        ctx.prec = 768
        eq = X.rad() == P.real.rad()
        ok &= eq
        print(f"half-width {s}: test X radius {X.rad().str(20)}, run box radius {P.real.rad().str(20)}, equal {eq}")
    return ok


def cauchy(bx):
    pstr, Cm, G, _ = C.load_chart(os.path.join(os.path.dirname(os.path.abspath(__file__)), bx["chart"]))
    M = bx["M"]
    widths = [arb(bx["h"])] * 2 + [arb(bx["w"][(i // 2) % len(bx["w"])]) for i in range(M - 1)]
    ctx.prec = 320
    p = CM.exact_p(pstr)
    ctx.prec = 320
    T = CM.T_matrix(Cm, M)
    X = [arb(0, wd) for wd in widths]
    pbox = C.p_of_x(p, T, G, X, M)
    ups = [abs(pbox[i] - p[i]).upper() for i in range(M + 1)]
    hw_true = max(ups, key=lambda u: float(u))
    hw_float = arb(max(float(u) for u in ups)) - (arb("1e-11") if MUTATE else 0)
    dirs = C.dp_dirs(T, G, X, M)
    nus = [abs(dirs[k][i]).upper() for k in range(M + 1) for i in range(M + 1)]
    nu_true = max(nus, key=lambda u: float(u))
    nu_float = arb(max(float(u) for u in nus))
    c = arb(C.CAUCHY)
    box_half = acb(arb(0, c)).real.rad()           # fixed_point_M.start_data_M: xc = x + arb(0, c) + i arb(0, c)
    lost = (hw_true - hw_float) + (nu_true - nu_float) * 2 * (c - hw_float) / (2 * nu_float)
    slack = box_half - c.lower()
    inside = bool(lost <= slack)
    print(f"hw: upper bound {hw_true.str(20)}, as float {hw_float.str(20)}; nu: {nu_true.str(20)}, as float "
          f"{nu_float.str(20)}; lost {lost.str(5)}; the lemma's box is wider than c's lower end by {slack.str(5)}: "
          f"bidisc inside the box {inside}")
    return inside


if __name__ == "__main__":
    bx = json.load(open(sys.argv[1]))
    ok = roundtrip() & boxes(bx) & cauchy(bx)
    print("MISC " + ("OK" if ok else "FAIL") + (" [mutated]" if MUTATE else ""))
    sys.exit(0 if ok else 1)
