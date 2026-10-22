#!/usr/bin/env python3
"""Review check R4 (agent RV): the saved certificate runs, the JSON path, and the Krawczyk step, re-derived independently.

  1. JSON round trip. validated_int.py writes every enclosure as arb.str(600, radius=True, more=True) and reads it back
     with arb(str). For 20 000 random balls (256-bit midpoints, radii from 1e-80 to 1e-2, as validated_int produces
     them) the parsed ball must contain the original, at 320 bits and at 53 bits. The old lossy format (12 digits)
     is shown for comparison: still an enclosure, but wider.
  2. K(X) recomputed from validated_int_krawczyk_1e-9.json with code written here (not krawczyk.krawczyk): its own
     normalization (arb, a different evaluation order, checked against the double-precision lateral_np one), and
     two different preconditioners Y (the float inverse of mid J(X), and an exact rational inverse of a rounded
     mid J(X)); both must give K(X) in int X.
  3. The zero enclosure claimed in A_certify.md section 9 (A = 2.542148047617 +- 9.1e-11, w/w_inf =
     0.7769940136097 +- 6.6e-11) must contain K(X), and K is reported. A tighter box X' around x^ (J over X still
     encloses Dg over X' subset X) shows how far the uniqueness box can shrink with the same runs.
  4. Robustness: the factor by which the J(X) radii, or the g(x^) radius, can be inflated before the test fails, and a
     negative control (centre moved by 3h along A: the test must fail).

    python3 rv_krawczyk_recheck.py
"""
import json
import os
import random
import sys
from fractions import Fraction as Fr

from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(HERE, "..", "validated_int_krawczyk_1e-9.json")


def parse_acb(p):
    return acb(arb(p[0]), arb(p[1]))


def roundtrip(n=20000, seed=1):
    rng = random.Random(seed)
    bad320 = bad53 = 0
    grow_more, grow_lossy = [], []
    for _ in range(n):
        ctx.prec = 256
        m = arb(rng.uniform(-1, 1)) * arb(10) ** rng.randint(-20, 3) + arb(rng.getrandbits(200)) / arb(2) ** 260
        r = arb(10) ** rng.uniform(-80, -2)
        x = arb(m.mid(), r.upper())
        s = x.str(600, radius=True, more=True)
        sl = x.str(12, radius=True)
        for prec in (320, 53):
            ctx.prec = prec
            y = arb(s)
            if not y.contains(x):
                if prec == 320:
                    bad320 += 1
                else:
                    bad53 += 1
        ctx.prec = 320
        y, yl = arb(s), arb(sl)
        assert yl.contains(x)
        grow_more.append(float(y.rad()) / float(x.rad()))
        grow_lossy.append(float(yl.rad()) / float(x.rad()))
    return bad320, bad53, max(grow_more), max(grow_lossy)


def normalization_rv(A, wf, xi1):
    """N = 2 pi i K'(s) f(xi1) 2F_1 and xi1^(-A^2), written out here: K' = 2^(-s-1/2) / (Gamma(a) Gamma(b)),
    f = xi^(-2s) e^(-1/(2 xi^2)) 2F0(1-a, 1-b;; -2 xi^2) = xi^(-2s) e^(-X) X^(1-a) U(1-a, 3/2, X), X = 1/(2 xi^2)
    (2F0(a', b';; -1/X) = X^a' U(a', 1 + a' - b', X), and 1 + (1-a) - (1-b) = 1 + b - a = 3/2); powers via exp/log,
    1/Gamma via rgamma, so the evaluation order differs from krawczyk.normalization"""
    I = acb(0, 1)
    w = 2 * arb.pi() / arb("1.64389375") * wf
    s = I * w
    a, b = (1 + s) / 2, (2 + s) / 2
    X = acb(1 / (2 * xi1 * xi1))
    Kp = (-(s + acb(arb(1) / 2)) * acb(2).log()).exp() * (a.rgamma() * b.rgamma())
    f = (-2 * s * acb(xi1).log()).exp() * (-X).exp() * ((1 - a) * X.log()).exp() * X.hypgeom_u(1 - a, acb(3) / 2)
    N = 2 * arb.pi() * I * Kp * f * 2 * (-I * A / 2)
    return N, (-(A * A) * acb(xi1).log()).exp()


def kraw(xhat, g, J, h, Y):
    """K(X) = x^ - Y g + (I - Y J)(X - x^), X = x^ + [-h, h]^2, everything arb; Y a 2x2 list of exact arb"""
    dX = arb(0, h.upper())
    K = []
    for i in range(2):
        t = xhat[i] - (Y[i][0] * g[0] + Y[i][1] * g[1])
        for j in range(2):
            t += ((1 if i == j else 0) - (Y[i][0] * J[0][j] + Y[i][1] * J[1][j])) * dX
        K.append(t)
    ok = all(abs(K[i] - xhat[i]) < h for i in range(2))
    marg = [h - abs(K[i] - xhat[i]) for i in range(2)]
    return ok, K, marg


def main():
    b320, b53, gm, gl = roundtrip()
    print(f"1. JSON round trip, 20000 random balls: parsed ball fails to contain the original: {b320} at 320 bits, "
          f"{b53} at 53 bits; radius growth factor <= {gm:.6f} (format of validated_int) vs <= {gl:.3g} (12 digits)")
    ctx.prec = 320
    d = json.load(open(SAVED))
    assert (d["A"], d["wf"], d["h"], d["xi1"]) == ("2.5421480475", "0.7769940136", "1e-9", "0.3")
    xhat = [arb(d["A"]), arb(d["wf"])]
    h = arb(d["h"])
    xi1 = arb(d["xi1"])
    Dm1, Dp1 = parse_acb(d["point"]["Dm1"]), parse_acb(d["point"]["Dp1"])
    dDm1 = [parse_acb(p) for p in d["box"]["dDm1"]]
    dDp1 = [parse_acb(p) for p in d["box"]["dDp1"]]
    N, xs = normalization_rv(xhat[0], xhat[1], xi1)
    sys.path.insert(0, os.path.join(HERE, ".."))
    import krawczyk as KR
    N0, xs0 = KR.normalization(xhat, "0.3", prec=320)
    ctx.prec = 320
    print(f"2. normalization N(x^) = {N.str(15)}; overlaps krawczyk.normalization: {N.overlaps(N0)} "
          f"(radius {float(N.rad()):.1e}); xi1^(-A^2) overlaps: {xs.overlaps(xs0)}")
    try:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "inner_stokes"))
        import numpy as np
        from lateral_np import fast_lin, Kp
        W0f = 2 * np.pi / 1.64389375
        Nf = 2 * np.pi * 1j * Kp(1j * W0f * 0.7769940136) * fast_lin(1j * W0f * 0.7769940136, 0.3) * 2 * (-1j * 2.5421480475 / 2)
        print(f"   double-precision lateral_np normalization (Borel-summed f, integrated numerically): {Nf:.12g}; "
              f"relative difference {abs(Nf - complex(N.mid())) / abs(Nf):.1e}")
    except Exception as e:                         # noqa: BLE001
        print("   (lateral_np comparison skipped:", e, ")")
    kap = xs / N
    JD = Dp1 - Dm1.conjugate()
    dJ = [(dDp1[k] - dDm1[k].conjugate()) * kap for k in range(2)]
    gz = JD * kap
    g = [gz.real, gz.imag]
    J = [[dJ[0].real, dJ[1].real], [dJ[0].imag, dJ[1].imag]]
    print(f"   J_D(x^) = {JD.str(8, radius=True)}; g(x^) = [{g[0].str(6, radius=True)}, {g[1].str(6, radius=True)}]")
    print("   J(X) = [[" + ", ".join(J[0][j].str(5, radius=True) for j in range(2)) + "], [" +
          ", ".join(J[1][j].str(5, radius=True) for j in range(2)) + "]]")
    # preconditioner 1: float inverse of mid J (as krawczyk.py); preconditioner 2: exact rational inverse of mid J
    # rounded to 4 significant digits (a deliberately different, cruder Y)
    m = [[float(J[i][j].mid()) for j in range(2)] for i in range(2)]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    Y1 = [[arb(m[1][1] / det), arb(-m[0][1] / det)], [arb(-m[1][0] / det), arb(m[0][0] / det)]]
    mr = [[Fr(f"{m[i][j]:.4g}") for j in range(2)] for i in range(2)]
    dr = mr[0][0] * mr[1][1] - mr[0][1] * mr[1][0]
    Yr = [[mr[1][1] / dr, -mr[0][1] / dr], [-mr[1][0] / dr, mr[0][0] / dr]]
    Y2 = [[arb(Yr[i][j].numerator) / arb(Yr[i][j].denominator) for j in range(2)] for i in range(2)]
    allok = True
    for name, Y in (("float inverse of mid J(X)", Y1), ("4-digit rational inverse", Y2)):
        ok, K, mg = kraw(xhat, g, J, h, Y)
        allok &= ok
        print(f"   K(X) with Y = {name}: [{K[0].str(13, radius=True)}, {K[1].str(13, radius=True)}]; in int X: {ok}; "
              f"margins {float(mg[0].lower()):.4e}, {float(mg[1].lower()):.4e}")
    ok, K, mg = kraw(xhat, g, J, h, Y1)
    claimA = arb("2.542148047617", "9.1e-11")
    claimW = arb("0.7769940136097", "6.6e-11")
    print(f"3. claimed zero enclosure contains K(X): A {claimA.contains(K[0])}, w/w_inf {claimW.contains(K[1])}; "
          f"K(X) = A in [{K[0].lower().str(14)}, {K[0].upper().str(14)}], "
          f"w/w_inf in [{K[1].lower().str(15)}, {K[1].upper().str(15)}]")
    W0 = 2 * arb.pi() / arb("1.64389375")
    print(f"   omega itself (w_inf := 2 pi / 1.64389375 exactly): omega in {(W0 * K[1]).str(13, radius=True)}")
    best = None
    for k in range(1, 100):
        hh = h * k / 100
        okk, Kk, _ = kraw(xhat, g, J, hh, Y1)
        if okk:
            best = (hh, Kk)
            break
    if best:
        print(f"   smallest box x^ +- {best[0].str(3)} (J(X) over the 1e-9 box reused) also passes; "
              f"its K: [{best[1][0].str(12, radius=True)}, {best[1][1].str(12, radius=True)}]")
    # robustness: inflate radii until failure
    def inflate(x, f):
        return arb(x.mid(), x.rad() * f)
    fJ = 1.0
    while kraw(xhat, g, [[inflate(J[i][j], fJ * 2) for j in range(2)] for i in range(2)], h, Y1)[0] and fJ < 1e6:
        fJ *= 2
    fg = 1.0
    while kraw(xhat, [inflate(g[i], fg * 2) for i in range(2)], J, h, Y1)[0] and fg < 1e12:
        fg *= 2
    print(f"4. robustness: still passes with the J(X) radii x{fJ:g} (fails at x{2 * fJ:g}); g(x^) radius x{fg:g}")
    # negative control: centre moved by 3h along A (g moved by J (3h, 0), linearised; the zero is then outside X)
    gs = [g[i] + J[i][0] * 3 * h for i in range(2)]
    okn, _, _ = kraw([xhat[0] + 3 * h, xhat[1]], gs, J, h, Y1)
    print(f"   negative control, centre moved by 3h along A: K(X) in int X: {okn} (must be False)")
    allok &= not okn
    print("VERDICT:", "Krawczyk certificate reproduced independently from the saved runs" if allok else "PROBLEM")


if __name__ == "__main__":
    main()
