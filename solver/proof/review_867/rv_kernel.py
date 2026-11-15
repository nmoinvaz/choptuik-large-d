#!/usr/bin/env python3
"""Review of fixed_point_M_b.py (#867), check K2: the kernel bound of the fast rows, against quadrature.

The module bounds, for a majorant term t^m acting on a component of weight exponent kj, in a row of weight ki,
    sup_{0 < r <= R} r^-ki int_0^r e^{-a(1/rho - 1/r)} (rho/r)^e rho^(m + kj) d rho,    e >= emin,
by fast_norm: R^pw / (a + (mm + 1 + emin) R) for pw = mm + 2 - ki >= 1, and 1/a for pw = 0 (mm = m + kj). With
1/rho = 1/r + tau the integral is r^(mm + 2) I_n(r), I_n = int_0^inf e^{-a tau} (1 + r tau)^-n d tau, n = mm + 2 + e.

Here I_n is computed by mpmath's quadrature (40 digits), which shares nothing with the module. The cases are the
(mm, pw, a, emin, R) that the module itself asks for: its calls of fast_norm are recorded during the M = 5 runs at
the certificate's zero (the point, the complex box, a sector of rays, k = 40) and the M = 3 run. For each case the
true value at r = R, R/2, R/100 and R/10^4, with e = emin and e = emin + 0.05, must not exceed the module's bound.
The identity 1 = a I_(n-1) + (n - 1) r I_n behind the bound is checked on the way.

    python3 rv_kernel.py
    python3 rv_kernel.py --mutate=offbyone    THE REVIEWER'S MUTATION. fast_norm with a + (mm + 2 + emin) R in the
        denominator, one integration by parts too many. Written before the run: the quadrature must exceed this
        bound (exit 1). The author's own checks (planted, two truncations, start set) are then run with the mutated
        bound and their verdicts printed. Whether they notice is a finding, not a pass criterion.
Exit status 0 only if no case exceeds the bound.
"""
import json
import os
import sys

import mpmath as mp
from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import fixed_point_M_b as B  # noqa: E402  (the module under review)

ZERO5 = ["2.718579189", "0.999228404", "0.149203574672", "0.014719052460", "0.008405346879", "0.001470345176"]
ZERO3 = ["2.726475886", "1.015603095", "0.144419478488", "0.011087425887"]
ORIG = B.fast_norm
cases = {}


def to_mp(x):
    return mp.mpf(arb(x).mid().str(45, radius=False))


def recorder(sp, kj, ki, R, a, emin, kernel="sharp"):
    for m in sp:
        mm = m + kj
        cases[(mm, mm + 2 - ki, str(to_mp(a)), str(to_mp(emin)), str(to_mp(R)))] = (kj, ki)
    return B._kernel(sp, kj, ki, R, a, emin, kernel)


def offbyone(sp, kj, ki, R, a, emin, kernel="sharp"):
    """MUTATION: the module's fast_norm with n in place of n - 1 in the sharp bound"""
    tot = arb(0)
    for m, c in sp.items():
        mm = m + kj
        pw = mm + 2 - ki
        if mm < -1 or pw < 0:
            raise ValueError("fast row: exponent below the weight")
        b = 1 / a if pw == 0 else R ** pw / (a + (mm + 2 + emin) * R)
        tot = B.up(tot + c * b)
    return tot


def I_n(n, r, a):
    return mp.quad(lambda t: mp.exp(-a * t) * (1 + r * t) ** (-n), [0, 1 / a, 10 / a, 100 / a, mp.inf])


def collect():
    """run the module so that it records its own kernel calls"""
    B.fast_norm = recorder
    ctx.prec = 512
    x5 = [acb(arb(v)) for v in ZERO5]
    c, h = arb("1e-3"), arb("1e-18")
    x5c = [acb(arb(v) + arb(0, h + c), arb(0, c)) for v in ZERO5]
    runs = [("M = 5, the point", B.Setup(x5, 5, "0.07", "3pi/8")),
            ("M = 5, the complex box", B.Setup(x5c, 5, "0.07", "3pi/8", None, 105)),
            ("M = 5, sector 0.2", B.Setup(x5, 5, "0.07", "3pi/8", "0.2")),
            ("M = 5, sector 0.2, k = 40", B.Setup(x5, 5, "0.07", "3pi/8", "0.2", None, 40)),
            ("M = 3, r0 = 0.1", B.Setup([acb(arb(v)) for v in ZERO3], 3, "0.1", "3pi/8"))]
    for name, st in runs:
        out = B.prove(st, False)
        print(f"  {name}: N = {st.N}, k = {st.k}, a = {float(st.a):.5f}, e_min = {float(st.emin):.4f}, contraction "
              f"constant {float(out['lam']):.4f}, proved {out['ok']}")


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    mutate = opts.get("mutate")
    mp.mp.dps = 40
    B._kernel = offbyone if mutate == "offbyone" else ORIG
    print(f"rv_kernel: mutation {mutate}; the module's runs, recording its kernel calls")
    collect()
    B.fast_norm = B._kernel
    print(f"  {len(cases)} distinct cases (mm, pw, a, e_min, R); pw from {min(k[1] for k in cases)} to "
          f"{max(k[1] for k in cases)}, mm from {min(k[0] for k in cases)} to {max(k[0] for k in cases)}")
    worst, where, bad, ident = mp.mpf(0), None, 0, mp.mpf(0)
    for (mm, pw, a_s, e_s, R_s), (kj, ki) in sorted(cases.items()):
        a, emin, R = mp.mpf(a_s), mp.mpf(e_s), mp.mpf(R_s)
        ctx.prec = 512
        bound = to_mp(B._kernel({0: arb(1)}, mm, mm + 2 - pw, arb(R_s), arb(a_s), arb(e_s)))
        for e in (emin, emin + mp.mpf("0.05")):
            n = mm + 2 + e
            for fr in (1, mp.mpf("0.5"), mp.mpf("0.01"), mp.mpf("1e-4")):
                r = R * fr
                true = r ** pw * I_n(n, r, a)
                ratio = true / bound
                if ratio > worst:
                    worst, where = ratio, (mm, pw, float(a), float(e), float(r))
                bad += ratio > 1
        # the identity behind the bound, at r = R and e = emin
        n = mm + 2 + emin
        ident = max(ident, abs(1 - a * I_n(n - 1, R, a) - (n - 1) * R * I_n(n, R, a)))
    print(f"  the identity 1 = a I_(n-1) + (n - 1) r I_n holds to {float(ident):.1e} over the cases")
    print(f"  largest true value / bound: {float(worst):.6f} at (mm, pw, a, e, r) = {where}; cases above the bound: "
          f"{bad} of {8 * len(cases)}")
    ok = bad == 0
    print(f"RV_KERNEL: {'PASS' if ok else 'FAIL'} (the quadrature {'stays below' if ok else 'exceeds'} the module's "
          f"bound; its slack is {float(1 - worst):+.4%})")
    if mutate:
        ray = opts.get("ray")
        print("the author's checks with the mutated bound:")
        v = {"planted, M = 5, at the start point": B.planted(5, verbose=False),
             "planted, M = 5, at |z| = R": B.planted(5, rho0=204.1, verbose=False)}
        if ray:
            with open(ray) as fh:
                xs = json.load(fh)["x"]
            v["two truncations, N = 60 and 105"] = B.two_truncations(xs, 5, 60, 105, verbose=False)
            v["two truncations, N = 95 and 105"] = B.two_truncations(xs, 5, 95, 105, verbose=False)
            okc, vals, cols, _ = B.start_set(xs, 5, "1e-18", "1e-3", 405.0, verbose=False)
            v["start set against the stored one"] = okc and B.compare(5, vals, cols, ray, verbose=False)[0]
        for name, passed in v.items():
            print(f"  {name}: {'PASS (the mutation is not noticed)' if passed else 'FAIL (noticed)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
