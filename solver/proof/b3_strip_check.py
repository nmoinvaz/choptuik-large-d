"""Numerical check of hypothesis (H-strip) of proof/B3_borel.md on the ball-arithmetic outer series.

(H-sigma): for x in (0, 1] the Borel transform in eps = 1/n of the outer series is analytic in a neighbourhood
S_sigma = {dist(zeta, [0, oo)) < sigma} of the positive real axis (with exponential bounds). This replaces
C_selection.md's H-a1/H-a2 ("every singularity at |arg| >= 0.48 pi"), which the u-mode singulant Phi(x) - i pi/2 itself
violates for x > 1/sqrt 2 (B3_borel.md, Section 5).

Method: the [L/L] Pade approximant (L = (K-1)//2, mpmath at 120 digits) of the Borel transform of each series; poles
that are exact doublets (a zero within 1e-30 relative: numerically cancelled) are dropped, nothing else. The check
reads the poles in the disc |zeta| <= R = 1.6 x (modulus of the nearest pole) and reports the smallest |Im| among
those with Re zeta > -0.5. It FAILS if one of them has |Im zeta| < SIGMA = 1.0 (near R+ or near the origin). It also
prints the numerical sigma: the smallest dist(zeta, [0, oo)) over the stable poles in the disc (poles near the
negative axis, as at M = 3, lower sigma without obstructing the 1-sum along R+).

Stability: a pole with |Im zeta| < SIGMA counts only if the Pade approximant from the first K - 6 terms has a pole
within 5 % of it; otherwise it is reported as unstable and dropped (a pole drawn by the approximant outside the disc
of convergence moves when K changes; b3_pole_stability.py shows one such case, M = 3, B0 at x = 0.8).

Mutation control (--plant=r,w): add to the Borel transform w B(0) (1 - zeta/r)^(-0.7), a weak singularity on R+ at r;
the check must fail. Runs on the files written by b3_outer_m1.py (series D, S, G, B) or b3_outer_M.py (D1, D3, ...).

    python3 b3_strip_check.py file.json [--xs=0.5,0.8,0.95,1] [--names=D,B,G] [--plant=1.3,1e-3]
"""
import json
import sys

import mpmath as mp

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FN = [a for a in sys.argv[1:] if not a.startswith("--")][0]
mp.mp.dps = 120
SIGMA = float(ARGS.get("sigma", 1.0))
PLANT = [mp.mpf(v) for v in ARGS["plant"].split(",")] if "plant" in ARGS else None


def poles(cs, drop=0):
    cs = cs[:len(cs) - drop] if drop else cs
    while cs and cs[0] == 0:
        cs = cs[1:]
    b = [cs[k] / mp.factorial(k) for k in range(len(cs))]
    if PLANT:
        r, w = PLANT
        b0 = abs(b[0])
        b = [bk + w * b0 * mp.rf(mp.mpf("0.7"), k) / mp.factorial(k) / r ** k for k, bk in enumerate(b)]
    sc = abs(b[-1]) ** (-mp.mpf(1) / (len(b) - 1))
    bs = [bk * sc ** k for k, bk in enumerate(b)]
    L = (len(bs) - 1) // 2
    p, q = mp.pade(bs[:2 * L + 1], L, L)
    P = mp.polyroots(q[::-1], maxsteps=3000, extraprec=600)
    Z = mp.polyroots(p[::-1], maxsteps=3000, extraprec=600)
    out = []
    for z in P:
        dz = min(abs(z - w) for w in Z) / abs(z)
        if dz > mp.mpf("1e-30"):
            out.append(z * sc)
    return sorted(out, key=abs)


def main():
    d = json.load(open(FN))
    xs = ARGS.get("xs", ",".join(d["series"].keys())).split(",")
    names = ARGS.get("names", "D,B,G").split(",")
    fail = False
    for xs_ in xs:
        key = [k for k in d["series"] if abs(float(k) - float(xs_)) < 1e-9][0]
        for nm in names:
            if nm not in d["series"][key]:
                continue
            cs = [mp.mpc(re, im) for re, im, r in d["series"][key][nm]]
            if all(c == 0 for c in cs):
                continue
            P = poles(cs)
            R = mp.mpf("1.6") * abs(P[0])
            inreg = [z for z in P if abs(z) <= R and mp.re(z) > -0.5]
            # a candidate near R+ counts only if the approximant from K - 6 terms has a pole within 5 % of it
            cand = [z for z in inreg if abs(mp.im(z)) < SIGMA]
            unstable = []
            if cand:
                P6 = poles(cs, drop=6)
                for z in cand:
                    if not any(abs(z - w) < mp.mpf("0.05") * abs(z) for w in P6):
                        unstable.append(z)
                inreg = [z for z in inreg if z not in unstable]
            imin = min((abs(mp.im(z)) for z in inreg), default=mp.inf)
            zmin = min(inreg, key=lambda z: abs(mp.im(z))) if inreg else None
            bad = imin < SIGMA
            if unstable:
                print(f"   (unstable pole(s) dropped, absent from the K - 6 approximant: "
                      + ", ".join(mp.nstr(z, 6) for z in unstable) + ")")
            fail |= bad
            dist = lambda z: abs(mp.im(z)) if mp.re(z) >= 0 else abs(z)
            stable_all = [z for z in P if abs(z) <= R and z not in unstable]
            sig = min((dist(z) for z in stable_all), default=mp.inf)
            print(f"   sigma estimate (min dist to [0, oo) over poles in the disc): {mp.nstr(sig, 4)}")
            print(f"x = {float(key):.4f} {nm:3s}: nearest pole {mp.nstr(P[0], 6):>26s} (|a| {mp.nstr(abs(P[0]), 5)}); "
                  f"in |zeta| <= {mp.nstr(R, 4)}, Re > -0.5: min |Im| = {mp.nstr(imin, 5)}"
                  + (f" at {mp.nstr(zmin, 6)}" if zmin is not None else "") + ("   <-- FAIL" if bad else ""))
    print(("FAIL" if fail else "PASS") + f": strip check |Im zeta| >= {SIGMA}" + (f" (planted {ARGS['plant']})" if PLANT else ""))
    sys.exit(1 if fail else 0)


main()
