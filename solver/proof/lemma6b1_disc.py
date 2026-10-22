"""Lemma 6b_1 (proof/A_certify.md section 12): the validated Taylor disc, rigorous (python-flint arb/acb).

Claim checked by `certify(c, Y0, w, r, n)`: for every exact centre c' in the ball c, every initial value y0 in the box
Y0 (7 components, the variables of lemma6_usys.SysU: [P_-1, P_+1, V_-1, V_+1, k C_-2, k (C_0 - 1), k C_+2], k = KAPPA)
and every w in the ball w, the solution of the truncated M = 1 system in u (lemma6_usys.SysU) with y(c') = y0 extends
analytically to the open disc |u - c'| < r (continuously to its closure).

Proof (section 12.3). Let tau = u - c' and let p be the Taylor polynomial of degree n of that solution at tau = 0
(its coefficients are enclosed by the Picard recursion on power series, run on the balls). The remainder R = y - p
solves
    R(tau) = T(R)(tau) = int_0^tau [ d(s) + F(c' + s, p(s) + R(s)) - F(c' + s, p(s)) ] ds,   d = F(c' + s, p) - p',
with the integral along the segment [0, tau]. d vanishes to order n at 0 (p is the exact Taylor polynomial), so
|d_i(s)| <= Delta_i |s|^n on |s| <= r with Delta_i = sum_{k >= n} |d_{i,k}| r^(k-n) (the coefficients d_{i,k}, k < n,
are exactly zero; the code checks that their balls contain 0). z = 1/u = 1/(c' + s) enters as the polynomial
z_J(s) = sum_{j <= J} (-s)^j / c^(j+1) plus s^(J+1) zeta(s), |zeta| <= 1/(|c|^(J+2) (1 - r/|c|)) (J >= n, so the
truncation touches only orders above n; at each fixed s it is a fixed number zeta(s) in that disc, so the coefficient
balls computed with zeta as a ball enclose the polynomial whose value at s is d(s)).
In the space of analytic R = O(tau^(n+1)) with ||R|| = max_i om_i sup_{|tau| <= r} |R_i(tau)| / |tau|^(n+1):
    ||T(0)|| <= F0 = max_i om_i Delta_i / (n + 1),
    ||T(R) - T(R')|| <= Lam ||R - R'||,   Lam = r / (n + 2) * max_i sum_j om_i L_ij / om_j,
where L_ij bounds |dF_i/dy_j| over the box B = {p(tau) + R : |tau| <= r, |R_j| <= b_j} (B_j = Y0_j + disc of radius
sum_{k >= 1} |p_{j,k}| r^k + b_j) and z in 1/c + disc(r / (|c| (|c| - r))) (mean value inequality on the convex box).
If Lam < 1, F0 + Lam rho <= rho and rho r^(n+1) / om_j <= b_j, then T maps the ball ||R|| <= rho into itself
(every R there has p + R in B) and contracts, so it has a fixed point: y = p + R is analytic on the open disc and
solves the system, with y(c') = y0.

    python3 lemma6b1_disc.py      self-test: the generic right-hand side against lemma6_usys.SysU.rhs, then one disc
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import acb, arb, acb_poly, acb_series, ctx  # noqa: E402
import lemma6_usys as U  # noqa: E402

ODD, EVEN = U.ODD, U.EVEN
I = acb(0, 1)
HALF = acb(0.5)


def pr(X, Y, out, zero):
    r = {}
    for k in out:
        acc = zero
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc = acc + xa * yb
        r[k] = acc
    return r


def rhs_gen(Y, z, w, zero):
    """lemma6_usys.SysU.rhs (first 7 components) for any ring type with scalar multiplication by acb on the right
    (acb, acb_series, acb_poly). Y: 7 ring elements; z = 1/u (ring element); w: acb."""
    kap = acb(U.KAPPA)
    ikap = acb(1) / kap
    P = {m: Y[i] for i, m in enumerate(ODD)}
    V = {m: Y[2 + i] for i, m in enumerate(ODD)}
    Cm1 = {n: Y[4 + i] * ikap for i, n in enumerate(EVEN)}
    C = {n: (Cm1[n] + 1 if n == 0 else Cm1[n]) for n in EVEN}
    s = {m: I * m * w for m in ODD}
    Q = {m: V[m] + P[m] * (1 + s[m]) for m in ODD}
    CQ = pr(Cm1, Q, ODD, zero)
    W = {m: V[m] + CQ[m] for m in ODD}
    CW = pr(Cm1, W, ODD, zero)
    Eh = {k: v * HALF for k, v in pr(P, P, EVEN, zero).items()}
    CE = pr(C, Eh, EVEN, zero)
    CEP = pr(CE, P, ODD, zero)
    dP = [(z * (V[m] + CQ[m])) * (-HALF) for m in ODD]
    dV = [V[m] * (-HALF) - (CQ[m] + CW[m]) * HALF - (z * (Q[m] * (-(2 + s[m])) - W[m] * (1 + s[m]) + CEP[m])) * HALF
          for m in ODD]
    dC = [(z * z * CE[n]) * (-HALF * kap) for n in EVEN]
    return dP + dV + dC


def taylor(c, Y0, w, n):
    """Taylor coefficients 0..n (as lists of acb, one list per component) of the solution through (c, Y0) in
    tau = u - c, by Picard iteration on power series (after n iterations the coefficients 0..n are final)"""
    L = n + 1
    ctx.cap = L
    zs = 1 / acb_series([c, 1], prec=L)
    zero = acb_series([0], prec=L)
    Ys = [acb_series([y], prec=L) for y in Y0]
    for _ in range(n):
        F = rhs_gen(Ys, zs, w, zero)
        Ys = [acb_series([Y0[i]], prec=L) + F[i].integral() for i in range(7)]
    out = []
    for s_ in Ys:
        cs = s_.coeffs()
        out.append(list(cs) + [acb(0)] * (L - len(cs)))
    return out


def box_add(z, rad):
    """the box z + [-rad, rad] + i [-rad, rad] (contains the closed disc of radius rad around every point of z)"""
    return acb(arb(z.real.mid(), z.real.rad() + rad), arb(z.imag.mid(), z.imag.rad() + rad))


def up(x):
    return arb(abs(x).upper())


def perron_weights(L):
    Lf = np.array([[float(x.upper()) for x in row] for row in L]) + 1e-12
    ev, evec = np.linalg.eig(Lf)
    i = int(np.argmax(ev.real))
    u = np.abs(evec[:, i].real) + 1e-6 * np.abs(evec[:, i].real).max()
    return list(1.0 / u)


def certify(c, Y0, w, r, n=40, J=None, verbose=False, mutate=None):
    """the disc certificate above. c: acb (a ball allowed), Y0: 7 acb (box), w: acb (ball), r: radius (str/arb).
    Returns a dict with ok, Lam, F0, rho, the sup bounds M_j of |y_j - Y0_j| on the disc, and diagnostics.
    mutate='sign': the fast term with the wrong sign in the defect (a wrong system): the certificate must fail."""
    r = arb(r)
    J = J or n + 8
    absc = arb(abs(c).lower())
    if not absc > r:
        raise ValueError("disc reaches u = 0")
    cf = taylor(c, Y0, w, n)
    p = [acb_poly(cf[i][:n + 1]) for i in range(7)]
    # z = 1/(c + s) = z_J(s) + s^(J+1) zeta(s)
    ctx.cap = J + 1
    zser = (1 / acb_series([c, 1], prec=J + 1)).coeffs()
    ez = arb(1) / (absc ** (J + 2) * (1 - r / absc))
    zpoly = acb_poly(list(zser) + [acb(0)] * (J + 1 - len(zser)) + [acb(arb(0, ez), arb(0, ez))])
    zero = acb_poly([0])
    F = rhs_gen(p, zpoly, w, zero)
    if mutate == "sign":
        F = [F[i] + (p[i] if i in (2, 3) else zero) for i in range(7)]       # V_u = +V/2 + ... instead of -V/2
    low_ok = True
    Delta = []
    for i in range(7):
        d = F[i] - p[i].derivative()
        ds = d.coeffs()
        for k in range(min(n, len(ds))):
            if not ds[k].contains(0):
                low_ok = False
        acc = arb(0)
        rp = arb(1)
        for k in range(n, len(ds)):
            acc = acc + up(ds[k]) * rp
            rp = rp * r
        Delta.append(acc)
    # sup of |p_j(tau) - Y0_j| on the disc
    Msup = []
    for i in range(7):
        acc, rp = arb(0), arb(1)
        for k in range(1, n + 1):
            rp = rp * r
            acc = acc + up(cf[i][k]) * rp
        Msup.append(acc)
    rz = r / (absc * (absc - r))
    Z = box_add(acb(1) / c, rz.upper())
    sysu = U.SysU()
    b = [arb("1e-30")] * 7
    res = None
    for it in range(6):
        B = [box_add(Y0[i], (Msup[i] + b[i]).upper()) for i in range(7)] + [w, acb(0)]
        _, cache = sysu.rhs(B, acb(1) / Z, Z, acb(0))
        L = [[None] * 7 for _ in range(7)]
        for j in range(7):
            e = [acb(1) if i == j else acb(0) for i in range(9)]
            col = sysu.lin(cache, e, acb(1) / Z, Z, acb(0))
            for i in range(7):
                L[i][j] = up(col[i])
        om = perron_weights(L)
        rowsum = [sum(arb(om[i]) * L[i][j] / arb(om[j]) for j in range(7)) for i in range(7)]
        Lam = r / (n + 2) * arb(max(x.upper() for x in rowsum))
        F0 = arb(max((arb(om[i]) * Delta[i] / (n + 1)).upper() for i in range(7)))
        if not Lam < 1:
            res = dict(ok=False, Lam=Lam, F0=F0, why="no contraction")
            break
        rho = arb((2 * F0 / (1 - Lam)).upper()) + arb("1e-300")
        selfmap = F0 + Lam * rho
        breq = [rho * r ** (n + 1) / arb(om[j]) for j in range(7)]
        consistent = all(breq[j] <= b[j] for j in range(7))
        res = dict(ok=bool(Lam < 1) and bool(selfmap <= rho) and consistent and low_ok, Lam=Lam, F0=F0, rho=rho,
                   b=b, breq=breq, low_ok=low_ok, Msup=[Msup[j] + b[j] for j in range(7)], Delta=Delta, om=om,
                   L=L, it=it)
        if consistent:
            break
        b = [arb((2 * breq[j]).upper()) for j in range(7)]
    if verbose:
        print(f"disc |u - c| < {r.str(4)} at c = {acb(c.mid()).str(6)}: n = {n}, Lam = {float(res['Lam'].upper()):.3f}, "
              f"F0 = {float(res['F0'].upper()):.2e}, ok = {res['ok']}"
              + (f", sup|y - y0| <= {max(float(m.upper()) for m in res['Msup']):.3g}, "
                 f"max |R| <= {max(float(x.upper()) for x in res['breq']):.1e}, low coefficients 0: {res['low_ok']}"
                 if 'Msup' in res else f" ({res.get('why')})"))
    return res


def selftest(trials=3, seed=2):
    import random
    ctx.prec = 200
    random.seed(seed)
    S = U.SysU()
    w = acb(U.VI.W0 * arb("0.7769940136"))
    worst = 0.0
    for _ in range(trials):
        Y = [acb(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(7)]
        u = acb(random.uniform(-30, 30), random.uniform(-30, 30))
        F1, _ = S.rhs(Y + [w, acb(2.5)], u, 1 / u, acb(0))
        F2 = rhs_gen(Y, 1 / u, w, acb(0))
        worst = max(worst, max(float(abs(F1[i] - F2[i]).upper()) for i in range(7)))
    print(f"generic rhs vs lemma6_usys.SysU.rhs: max mismatch {worst:.1e}")
    return worst


if __name__ == "__main__":
    selftest()
