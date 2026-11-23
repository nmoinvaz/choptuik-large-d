#!/usr/bin/env python3
"""Review of #1079, hypothesis H0 at an excited zero: a second look at step (i) on a set that contains the
certificate's four start sets, by direct calls of fixed_point_M (not through certify_M.start_data).

    python3 review_1079/h0_recheck.py BOX.json [--h=1e-9] [--mutate=series|r0|cauchy]      from solver/proof

The certificate (certify_M.start_data) calls the lemma on: p^; p^'s Cauchy box; the hull of the run box; the hull's
Cauchy box (Cauchy radius 1e-3), on the ray 5pi/16 at r0 = certify_M.R0[M], diag splitting, with the truncation N
that is optimal at p^. Here the lemma is run, with that N, on
    X  = p^ +- h (real; default h = 1e-9, which contains the hull),
    Xc = real parts p^ +- (h + 1e-3), imaginary parts +- 1e-3 (contains both Cauchy boxes),
and the script checks that the hull lies in X. PASS (exit 0) iff both runs hold and the inclusion holds.
It also prints inner_np's own start radius at this centre, r0_inner_np(M, wf), which the certificate does not use:
the lemma is stated for any r0 at which the computation succeeds, and the path starts at the same r0.
Mutations (each must end FAIL, exit 1):
    series  one coefficient of the top harmonic's series off by 1e-6 (fixed_point_M's own mutation)
    r0      the start radius 0.3 in place of 0.0741
    cauchy  the hull test against X of half-width 1e-13 (smaller than the hull): the inclusion must fail
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flint  # noqa: E402
from flint import acb, arb, ctx  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
from fixed_point import Maj  # noqa: E402

R0 = {3: "0.0741", 5: "0.0622"}       # typed here, compared with the box file's path below
ARG = "5pi/16"
CAUCHY = "1e-3"
PREC = 320


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    mutate = opts.get("mutate")
    h = opts.get("h", "1e-9")
    with open(args[0]) as fh:
        bx = json.load(fh)
    M = bx["M"]
    m = M + 1
    print(f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}; box file {os.path.basename(args[0])}"
          + (f"; MUTATION {mutate}" if mutate else ""))
    assert bx["path"][0] == R0[M] and bx["path"][1] == ARG, f"the box file's path {bx['path']} is not the expected one"
    assert bx["start"]["rigorous"] is True and "fixed_point_M.start_data_M" in bx["start"]["start"]
    r0 = "0.3" if mutate == "r0" else R0[M]
    ctx.prec = 256                                        # certify_M.exact_p: midpoints of the decimals at 256 bits
    p = [arb(arb(x).mid()) for x in bx["p"]]
    ctx.prec = 768                                        # the hull as certify_M.box_run forms it
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    widths = [arb(bx["h"]), arb(bx["h"])] + [arb(bx["w"][(i // 2) % len(bx["w"])]) for i in range(M - 1)]
    hull = []
    for i in range(m):
        if i < 2:
            hull.append(p[i] + arb(0, widths[i]))
        else:
            hull.append(p[i] + sum((Cm[i - 2][j] * arb(0, widths[j]) for j in range(2)), arb(0)) + arb(0, widths[i]))
    ctx.prec = PREC
    hx = arb("1e-13") if mutate == "cauchy" else arb(h)
    inside = all(bool(abs(arb(hull[i].mid()) - p[i]) + arb(hull[i].rad()) <= hx) for i in range(m))
    print("hull half-widths: " + ", ".join(f"{float(hull[i].rad()):.3e}" for i in range(m))
          + f"; inside X = p^ +- {float(hx.mid()):.0e}: {inside}")
    wf = float(p[1].mid())
    print(f"start ray {ARG}, r0 = {r0} (the certificate's); inner_np's own r0 at this centre: "
          f"{FPM.r0_inner_np(M, wf):.4f} (at the fundamental zero: {FPM.r0_inner_np(M, FPM.ZEROS[M][1]):.4f})")
    c = arb(CAUCHY)
    xp = [acb(v) for v in p]
    xb = [acb(arb(v.mid(), arb(h))) for v in p]
    xc = [acb(arb(v.mid(), arb(h) + c), arb(0, c)) for v in p]
    mu = "series" if mutate == "series" else None
    try:
        Maj.BOX = False
        c0 = FPM.run_M(xp, M, r0, ARG, None, verbose=False, diag=True, mutate=mu)
        Maj.BOX = True
        oX = FPM.run_M(xb, M, r0, ARG, None, verbose=False, diag=True, kmins_in=c0["kmins"], N_in=c0["N"],
                       k_in=c0["k"], mutate=mu)
        oC = FPM.run_M(xc, M, r0, ARG, None, verbose=False, diag=True, kmins_in=c0["kmins"], N_in=c0["N"],
                       k_in=c0["k"], mutate=mu)
        Maj.BOX = False
    except ValueError as e:
        print(f"the lemma's harness refuses: {e}")
        print("H0 SECOND LOOK: FAIL")
        return 1
    side = 3 * float(FPM.W0().mid()) * float(c.mid())
    for nm, o in (("centre p^", c0), (f"X = p^ +- {h}", oX), (f"Cauchy box of X (+- {CAUCHY})", oC)):
        eb = max(float(v.upper()) for v in FPM.eta_bounds(o).values())
        print(f"  {nm}: N = {o['N']}, k = {o['k']}, contraction {float(o['lip'].upper()):.4f}, self-map "
              f"{bool(o['selfmap'] < o['rho'])}, lower coefficients contain 0 {o['low_ok']}, structural orders "
              f"{o['struct_ok']}, sup|eta| {eb:.3e}: holds {o['ok']}")
    # the centre's eta against the file's record (same quantity: start_data_M's largest eta bound at the centre)
    eta_c = max(float(v.upper()) for v in FPM.eta_bounds(c0).values())
    rel = abs(eta_c - bx["start"]["eta"]) / bx["start"]["eta"]
    print(f"  centre sup|eta| here {eta_c:.6e}, in the box file {bx['start']['eta']:.6e} (relative difference {rel:.1e})")
    print(f"  diag side condition |mu Im w| <= {side:.5f} <= 3/2: {side <= 1.5}")
    ok = bool(c0["ok"] and oX["ok"] and oC["ok"] and inside and side <= 1.5 and (mutate or rel < 1e-6))
    print(f"H0 SECOND LOOK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
