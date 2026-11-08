#!/usr/bin/env python3
"""A hidden assumption of fixed_point.tail_maj (#189 review): at a parameter point, the first coefficient of each
residual numerator whose ball does not contain 0 is taken as its first nonzero order kmin, and every lower coefficient
is dropped from the majorant (it must contain 0 and be below 1e-60 on |t| <= R). The dropped coefficients are exact
zeros only if kmin is the structural order: N - 1 for the c rows (c_N is missing) and N for the p and v rows
(v_N, p_N are missing). A larger detected kmin would mean a nonzero coefficient dropped without a bound.

This runs fixed_point_M.run_M (the original code) at the point for the M = 5 configurations Theorem A(5) uses, at 320 and
at 640 bits, and checks that the detected orders are the structural ones at both precisions. Exit 0 only if they are.
    kmin_check.py [--x=<centre, comma-separated>] [--mutate=series]
--mutate=series (the control): one coefficient of the top harmonic's series off by a relative 1e-6 (run_M's own
mutation), so a residual numerator no longer vanishes below its structural order; the check must FAIL.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fixed_point as FP  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
import lemma6_sector as L6S  # noqa: E402
from flint import arb, acb, ctx  # noqa: E402


def configs():
    a_eff = None
    yield "old sector lemma, start of the C^1 ray (r0 0.07, 3pi/8 +- 0.2, diag)", dict(r0="0.07", arg="3pi/8",
                                                                                         sector="0.2", diag=True)
    yield "lemma FP, start of the 6a box run (r0 0.0622, 5pi/16, diag)", dict(r0="0.0622", arg="5pi/16", diag=True)
    ctx.prec = 320
    Rp = arb("0.08") ** 2
    a_eff = arb((arb(2).sqrt() / 4 - 20 * Rp / 2).lower())
    yield "lemma S (rp 0.08, k 20, strict kernel bound)", dict(r0="0.08", a_in=a_eff, intbound=L6S.intbound_strict,
                                                                k_in=20, sym=True)


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = 5
    xs = opts["x"].split(",") if "x" in opts else FPM.ZEROS[M]
    allok = True
    for name, kw in configs():
        res = {}
        for prec in (320, 640):
            ctx.prec = prec
            x = [acb(v) for v in xs]
            FP.Maj.BOX = False
            out = FPM.run_M(x, M, prec=prec, verbose=False, mutate=opts.get("mutate"), **kw)
            comps = out["comps"]
            res[prec] = (out["N"], out["kmins"], out["ok"], out["low_ok"], out["struct_ok"])
        N3, km3, ok3, low3, st3 = res[320]
        N6, km6, ok6, low6, st6 = res[640]
        # kmins are ordered as run_M builds them: (p, v) per odd mu, then c per even nu
        names = []
        for mu in range(-M, M + 1, 2):
            names += [("p", mu), ("v", mu)]
        names += [("c", nu) for nu in range(-(M + 1), M + 2, 2)]
        struct = [N3 - 1 if t == "c" else N3 for (t, _) in names]
        same = km3 == km6 and N3 == N6
        structural = km3 == struct
        ok = same and structural
        allok = allok and ok
        print(f"{name}: N = {N3}; kmins at 320 bits {sorted(set(km3))}, at 640 bits {sorted(set(km6))}; "
              f"equal {same}; structural (c: N - 1, p and v: N) {structural}; lemma ok {ok3}, low_ok {low3}")
        if not structural:
            for (t, mu), k, s in zip(names, km3, struct):
                if k != s:
                    print(f"   {t}{mu:+d}: detected {k}, structural {s}")
    print(f"KMIN CHECK: {'PASS' if allok else 'FAIL'}" + (f"   [MUTATION {opts['mutate']}: must FAIL]"
                                                          if opts.get("mutate") else ""))
    if opts.get("mutate"):
        return 0 if not allok else 3
    return 0 if allok else 3


if __name__ == "__main__":
    sys.exit(main())
