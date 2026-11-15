#!/usr/bin/env python3
"""Comparisons of fixed_point_M_b.py (the second implementation of the start-value contraction, issue #427) with the
first one, fixed_point.py and fixed_point_M.py. This file is a harness: it imports both. The proof code,
fixed_point_M_b.py, imports neither.

    python3 fixed_point_M_b_check.py m1
        planted: M = 1 at fixed_point.py's zero, r0 = 0.1. The second implementation's enclosures of
        (D, S, C)(xi0), xi0 = 0.1 e^{3 i pi/8}, must overlap fixed_point.py's start values in all 7 components.
    python3 fixed_point_M_b_check.py lip [--M=5] [--x=...] [--r0=0.07] [--sector=0.2]
        the two Lipschitz matrices entry by entry, at the first implementation's N, k and ball, with the kernel
        bound of A_certify.md 4.4 (--kernel=classic) in the second. Criteria, written before the first run:
          (a) every entry of the second is <= the first's (1 + 1e-9): its majorants keep each power of t and
              integrate term by term, the first collapses them to one power, so the second can only be smaller;
          (b) every entry of the first above 1e-6 is matched to within 20% (second >= 0.8 first): a term missing
              from either matrix would show as a larger gap;
          (c) the same for the forcing in the first's weighted norm.
    mutations (each must make its check fail, exit status 3):
        m1 --mutate=coef | sign | scale     as in fixed_point_M_b.py
        lip --mutate=drop                   the second implementation without the [ct W] term of Phi_V: (b) must fail
"""
import sys

import fixed_point as FP
import fixed_point_M as FPM
import fixed_point_M_b as B
from fixed_point import Maj
from flint import arb, acb, ctx


def ov(a, b):
    return a.real.overlaps(b.real) and a.imag.overlaps(b.imag)


def m1(mutate=None):
    ctx.prec = 320
    Maj.BOX = False
    o1 = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), "0.1", verbose=False)
    xi0, D1, S1, C1 = FP.start_values(o1, +1)
    ctx.prec = 512
    st = B.Setup([acb(arb(FP.A_STAR)), acb(arb(FP.WF_STAR))], 1, "0.1", "3pi/8", mutate=mutate)
    out = B.prove(st, True, "second implementation: ")
    if not out["ok"]:
        print("M = 1 PLANTED: FAIL (no contraction)")
        return False
    R = st.R
    phi = 3 * arb.pi() / 8
    xi = acb(st.r0 * phi.cos(), st.r0 * phi.sin())
    z0 = xi * xi
    ok = True
    print(f"first implementation: N = {o1['N']}, k = {o1['k']}, contraction {float(o1['lip'].upper()):.4f}")
    print("  component, overlap, |mid - first's mid|, radius, first's radius")
    for mu in st.odd:
        ep = B.eta_radius(out, ("p", mu), R)
        ev = B.eta_radius(out, ("v", mu), R)
        Dm = xi * (st.P[mu](z0) + B.disc(ep))
        # S = z Q, Q = V + (1 + s) P
        Sm = z0 * (st.Q[mu](z0) + B.disc(B.up(ev + st.s1[mu] * ep)))
        for nm, a, b in ((f"D{mu:+d}", Dm, D1[mu]), (f"S{mu:+d}", Sm, S1[mu])):
            o = ov(a, b)
            ok = ok and o
            print(f"  {nm} {o} {float(abs(a.mid() - b.mid()).upper()):.2e} {B.rad(a):.2e} {B.rad(b):.2e}")
    for nu in st.even:
        a = st.C[nu](z0) + B.disc(B.eta_radius(out, ("c", nu), R))
        o = ov(a, C1[nu])
        ok = ok and o
        print(f"  C{nu:+d} {o} {float(abs(a.mid() - C1[nu].mid()).upper()):.2e} {B.rad(a):.2e} {B.rad(C1[nu]):.2e}")
    print(f"M = 1 PLANTED (fixed_point.py's start values): {'PASS' if ok else 'FAIL'}")
    return ok


def lip(M, xs, r0, sector, mutate=None):
    ctx.prec = 320
    Maj.BOX = False
    o1 = FPM.run_M([acb(v) for v in xs], M, r0, "3pi/8", sector, verbose=False, diag=True)
    comps, wts, rho, k, N = o1["comps"], o1["wts"], o1["rho"], o1["k"], o1["N"]
    ctx.prec = 512
    st = B.Setup([acb(arb(v)) for v in xs], M, r0, "3pi/8", sector, N, k, True, "classic")
    assert st.comps == comps
    w = {cp: B.up(rho / arb(wts[i])) for i, cp in enumerate(comps)}
    if mutate == "drop":
        orig = B.mconv
        calls = {"n": 0}

        def mconv(A, Bd, left, right, target):
            calls["n"] += 1
            if calls["n"] == 2:          # the second product of lipschitz() is [ct W]
                return {key: B.MJ({}) for key in target}
            return orig(A, Bd, left, right, target)
        B.mconv = mconv
    L2 = st.lipschitz(w)
    if mutate == "drop":
        B.mconv = orig
    L1 = o1["L"]
    n = len(comps)
    worst_hi, worst_lo, bad_a, bad_b = 0.0, 1.0, [], []
    for i in range(n):
        for j in range(n):
            a, b = float(L1[i][j].upper()), float(L2[i][j])
            if b > a * (1 + 1e-9) + 1e-300:
                bad_a.append((comps[i], comps[j], a, b))
            if a > 1e-6:
                worst_hi = max(worst_hi, b / a)
                worst_lo = min(worst_lo, b / a)
                if b < 0.8 * a:
                    bad_b.append((comps[i], comps[j], a, b))
    lam1 = float(o1["lip"].upper())
    lam2 = max(sum(wts[i] * float(L2[i][j]) / wts[j] for j in range(n)) for i in range(n))
    F1 = float(o1["F"].upper())
    F2 = max(wts[i] * float(st.Fv[cp]) for i, cp in enumerate(comps))
    okc = F2 <= F1 * (1 + 1e-9) and F2 >= 0.8 * F1
    print(f"M = {M}, r0 = {r0}, sector {sector}: N = {N}, k = {k}, a = {float(o1['a']):.5f}")
    print(f"  entries of the first above 1e-6: ratio second / first between {worst_lo:.4f} and {worst_hi:.6f}")
    print(f"  (a) second <= first in every entry: {not bad_a}" + "".join(
        f"\n      {x[0]}<-{x[1]}: first {x[2]:.6g}, second {x[3]:.6g}" for x in bad_a[:8]))
    print(f"  (b) second >= 0.8 first: {not bad_b}" + "".join(
        f"\n      {x[0]}<-{x[1]}: first {x[2]:.6g}, second {x[3]:.6g}" for x in bad_b[:8]))
    print(f"  (c) forcing in the first's norm: first {F1:.6e}, second {F2:.6e}, ratio {F2 / F1:.6f}: {okc}")
    print(f"  contraction constant in the first's weights: first {lam1:.6f}, second {lam2:.6f}")
    ok = not bad_a and not bad_b and okc
    print(f"LIPSCHITZ COMPARISON: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    cmd = args[0] if args else "m1"
    if cmd == "m1":
        ok = m1(opts.get("mutate"))
    else:
        M = int(opts.get("M", "5"))
        xs = opts["x"].split(",") if "x" in opts else FPM.ZEROS[M]
        ok = lip(M, xs, opts.get("r0", "0.07" if M == 5 else "0.1"), opts.get("sector"), opts.get("mutate"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
