#!/usr/bin/env python3
"""P2 review: an independent implementation of B7 (proof/B_chain.md section 8.3), the implicit-function step of
Theorem B(1), written from the statement, in mpmath interval arithmetic (mpmath.iv), not python-flint.

What is re-implemented: the Krawczyk test for F = J_D + Q on Theorem A's box X, the degree (homotopy) condition, the
minimum of |J_D| on the boundary of X, the norm of J(X)^-1, the Cauchy bounds on DQ, the scan in ln n and the
constant C7 of |p_n - p*| <= C7/n.

Inputs (certified elsewhere; shared with b7_ift.py, recorded in proof/P2_review.md):
  - Theorem A: J_D(x^) (lemma6_path_point.json) and the interval Jacobian J(X) of J_D over X = x^ +- 1e-9
    (lemma6_path_box_1e-9.json), with J_D = D_{+1} - conj(D_{-1});
  - Lemma P + lemma L: sup over X_C of ||l~||_1 (b7_ift.json "lnorm"; its main part is re-derived at the centre by
    p2_ell.py);
  - lemma L: eps_L, ||V||, ||V^-1||, min/max Re(lambda) (b5_label.json);
  - B5(1)'s super-solution x_Phi for the (H8) data of d0 = 1e7 (b5_jump_d0_1e7.json "xPhi"), from which C_E is
    recomputed here.

The statement checked (for every n >= n0): with Q = l~ (I + R~)^-1 E~, |E~| <= C_E/n,
  (K) K(X) = x^ - Y F(x^) + (I - Y [J(X) + DQ]) (X - x^) lies in the interior of X (Y a fixed real matrix);
  (D) (1 - ||V|| ||V^-1|| eps_L) min_{bd X} |J_D| / (sup||l~||_1 ||V^-1||) > ||V|| C_E / n;
  (C7) |p_n - p*|_inf <= ||J(X)^-1||_inf sup||l~||_1 C_E / ((1 - eps_L) n).

    python3 p2_b7.py [--mu=hi|lo] [--nsub=400] [--mutate=JD|conj|CE5]
    --mu=hi (default) uses u*^{-Re lambda} <= u*^{-mu_hi} in C_E (the correct direction, see P2_review.md);
    --mu=lo reproduces b5_jump.py's u*^{-mu_lo}.
    Mutations (each must change the outcome): JD (J_D(x^) + 5e-11 (1 + i): no n may pass), conj (J_D without the
    conjugation of D_{-1}: no n may pass), CE5 (C_E multiplied by e^5: n0 must move up to about e^94.75).
"""
import sys

from mpmath import iv, mp

from p2_common import C, absup, hi, lo, load, sym

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")


def theorem_a():
    pt = load("lemma6_path_point.json")
    bx = load("lemma6_path_box_1e-9.json")
    assert (pt["A"], pt["wf"]) == (bx["A"], bx["wf"])
    xhat = [iv.mpf(pt["A"]), iv.mpf(pt["wf"])]
    Dm1, Dp1 = C.parse(pt["Dm1"]), C.parse(pt["Dp1"])
    JD = Dp1 - (Dm1 if MUT == "conj" else Dm1.conj())
    dDm1 = [C.parse(p) for p in bx["dDm1"]]
    dDp1 = [C.parse(p) for p in bx["dDp1"]]
    dJ = [dDp1[k] - (dDm1[k] if MUT == "conj" else dDm1[k].conj()) for k in range(2)]
    # real 2 x 2: rows (Re J_D, Im J_D), columns (A, w/w_inf)
    JX = [[dJ[0].re, dJ[1].re], [dJ[0].im, dJ[1].im]]
    h = iv.mpf(bx["h"])
    if MUT == "JD":
        JD = JD + C(iv.mpf("5e-11"), iv.mpf("5e-11"))
    return xhat, JD, JX, h


def inv2(M):
    """interval inverse of a 2 x 2 interval matrix (contains the inverse of every point matrix), or None"""
    a, b, c, d = M[0][0], M[0][1], M[1][0], M[1][1]
    det = a * d - b * c
    if lo(det) <= 0 <= hi(det):
        return None
    return [[d / det, -b / det], [-c / det, a / det]]


def norm_inf(M):
    return max(absup(M[i][0]) + absup(M[i][1]) for i in range(2))


def mid_inverse(M):
    """a point approximate inverse of the midpoint matrix (Krawczyk admits any fixed Y)"""
    a, b, c, d = ((lo(M[i][j]) + hi(M[i][j])) / 2 for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    det = a * d - b * c
    return [[iv.mpf(d / det), iv.mpf(-b / det)], [iv.mpf(-c / det), iv.mpf(a / det)]]


def krawczyk(xhat, Fhat, JF, h, Y):
    dX = sym(h)
    margins = []
    for i in range(2):
        t = xhat[i] - (Y[i][0] * Fhat[0] + Y[i][1] * Fhat[1])
        for j in range(2):
            Mij = (iv.mpf(1) if i == j else iv.mpf(0)) - (Y[i][0] * JF[0][j] + Y[i][1] * JF[1][j])
            t = t + Mij * dX
        dev = t - xhat[i]
        margins.append(lo(h) - absup(dev))          # K_i - x^_i strictly inside [-h, h]
    return all(m > 0 for m in margins), margins


def min_boundary(JD, JX, h, nsub):
    """lower bound of |J_D| on the boundary of X, from J_D(p) in J_D(x^) + J(X)(p - x^) on nsub segments per side"""
    best = None
    H = hi(h)
    for side in (0, 1):
        for sgn in (-1, 1):
            for k in range(nsub):
                seg = iv.mpf([-H + 2 * H * mp.mpf(k) / nsub, -H + 2 * H * mp.mpf(k + 1) / nsub])
                dp = [None, None]
                dp[side] = iv.mpf(sgn * H)
                dp[1 - side] = seg
                re = JD.re + JX[0][0] * dp[0] + JX[0][1] * dp[1]
                im = JD.im + JX[1][0] * dp[0] + JX[1][1] * dp[1]
                v = C(re, im).abslo()
                best = v if best is None or v < best else best
    return best


def main():
    xhat, JD, JX, h = theorem_a()
    lab = load("b5_label.json")
    jump = load("b5_jump_d0_1e7.json")
    ift = load("b7_ift.json")
    box = load("b7_box.json")
    infl = 1 + iv.mpf("1e-12")          # the JSON floats are rounded arb upper bounds
    epsL = iv.mpf(lab["epsL"]) * infl
    Vn = iv.mpf(lab["V_norm"]) * infl
    Vin = iv.mpf(lab["Vinv_norm"]) * infl
    mu_lo, mu_hi = iv.mpf(lab["mu_lo"]) - iv.mpf("1e-12"), iv.mpf(lab["mu_hi"]) + iv.mpf("1e-12")
    # B5(2): C_E = e^{1/(2u*)} u* max_i u*^{-Re lambda_i} ||V^-1|| (x_Phi,-1 + x_Phi,+1); u* < 1, so the max is at mu_hi
    u = iv.mpf(jump["ustar"])
    xs = (iv.mpf(jump["xPhi"][0]) + iv.mpf(jump["xPhi"][1])) * infl
    mu = mu_hi if OPTS.get("mu", "hi") == "hi" else mu_lo
    CE = iv.mpf(hi(iv.exp(1 / (2 * u)) * iv.exp((1 - mu) * iv.log(u)) * Vin * xs))
    if MUT == "CE5":
        CE = CE * iv.exp(5)
    Lsup = iv.mpf(ift["lnorm"]) * infl
    hr, hi_ = iv.mpf(box["hr"]), iv.mpf(box["hi"])
    rP = iv.mpf(min(lo(hr - h), lo(hi_)))          # polydisc radius of X_C around every point of X
    # B5(1)'s box x^ + [+-2e-6] + i[+-1e-6] contains the polydisc of radius rC around every point of X when
    # rC + h <= 2e-6 and rC <= 1e-6; rC is taken a hair below 1e-6 so that decimal rounding cannot matter
    rC = iv.mpf(mp.mpf("1e-6") * (1 - mp.mpf("1e-9")))
    assert hi(rC) + hi(h) < mp.mpf("2e-6") and hi(rC) < mp.mpf("1e-6")
    dL = Lsup / rP
    Rh = Vn * Vin * epsL
    Ji = inv2(JX)
    if Ji is None:
        print("J(X) contains a singular matrix")
        return 1
    invn = norm_inf(Ji)
    mbd = min_boundary(JD, JX, h, int(OPTS.get("nsub", "400")))
    Y = mid_inverse(JX)
    print(f"inputs: |J_D(x^)| <= {mp.nstr(C(JD.re, JD.im).absup(), 5)}; ||J(X)^-1||_inf <= {mp.nstr(invn, 6)}; "
          f"min over bd X of |J_D| >= {mp.nstr(mbd, 6)}")
    print(f"        sup||l~||_1 <= {mp.nstr(hi(Lsup), 6)}, Cauchy radius {mp.nstr(lo(rP), 3)} -> |D l~| <= "
          f"{mp.nstr(hi(dL), 6)}")
    print(f"        eps_L = {mp.nstr(hi(epsL), 8)}, ||V|| = {mp.nstr(hi(Vn), 8)}, ||V^-1|| = {mp.nstr(hi(Vin), 8)}, "
          f"||R|| <= {mp.nstr(hi(Rh), 6)}")
    print(f"        C_E = e^{mp.nstr(hi(iv.log(CE)), 8)} (mu = {OPTS.get('mu', 'hi')}; b5_jump.py has "
          f"e^{jump['lnCE']:.6f})" + (f"   [MUTATION {MUT}]" if MUT else ""))
    first = None
    rows = []
    for k in range(7400, 11001, 5):
        lnn = mp.mpf(k) / 100
        En = CE * iv.exp(-iv.mpf(k) / 100)
        omE = 1 - epsL
        Qb = Lsup * En / omE
        dQ = dL * En / omE + Lsup * (epsL / rC * En / (omE * omE) + En / (rC * omE))
        Fhat = [JD.re + sym(Qb), JD.im + sym(Qb)]
        JF = [[JX[i][j] + sym(dQ) for j in range(2)] for i in range(2)]
        okK, marg = krawczyk(xhat, Fhat, JF, h, Y)
        lhs = (1 - Rh) * mbd / (Lsup * Vin)
        okD = lo(lhs) > hi(Vn * En)
        rows.append((lnn, okK, okD, marg, hi(Qb), lo(lhs), hi(Vn * En)))
        if okK and okD and first is None:
            first = lnn
    for r in rows[::40]:
        print(f"   ln n = {float(r[0]):6.2f}: Krawczyk {r[1]} (margins {float(r[3][0]):.2e}, {float(r[3][1]):.2e}), "
              f"degree {r[2]} ({float(r[5]):.4e} vs {float(r[6]):.4e}), |Q| <= {float(r[4]):.2e}")
    if first is None:
        print("no ln n in [74, 110] passes both tests")
        if MUT in ("JD", "conj"):
            print(f"MUTATION {MUT}: caught (no n passes)")
            return 0
        return 1
    near = [r for r in rows if abs(r[0] - first) < mp.mpf("0.3")]
    for r in near:
        print(f"   ln n = {float(r[0]):6.2f}: Krawczyk {r[1]}, degree {r[2]} ({float(r[5]):.5e} vs {float(r[6]):.5e})")
    bad = [r for r in rows if r[0] >= first and not (r[1] and r[2])]
    if bad:
        print(f"a row beyond ln n = {float(first)} fails (monotonicity contradicted): {float(bad[0][0])}")
        return 1
    C7 = hi(iv.mpf(invn) * Lsup * CE / (1 - epsL))
    print(f"=> both tests pass for every ln n >= {mp.nstr(first, 6)} on the 0.05 grid up to 110 (monotone in C_E/n "
          f"with Y fixed, hence for every n >= e^{mp.nstr(first, 6)}); C7 = {mp.nstr(C7, 6)} = "
          f"e^{mp.nstr(mp.log(C7), 6)}")
    if MUT == "CE5":
        ok = abs(float(first) - 94.75) <= 0.3
        print(f"MUTATION CE5: n0 must move to about e^94.75: ln n0 = {float(first)}: {'caught' if ok else 'NOT caught'}")
        return 0 if ok else 3
    if MUT in ("JD", "conj"):
        print(f"MUTATION {MUT}: NOT caught")
        return 3
    claim_ok = first <= mp.mpf("89.75") and mp.log(C7) <= mp.mpf("67.845")
    print(f"claim (n0 = e^89.75, C7 = e^67.84) {'CONFIRMED' if claim_ok else 'NOT confirmed'} by this implementation")
    return 0 if claim_ok else 1


if __name__ == "__main__":
    sys.exit(main())
