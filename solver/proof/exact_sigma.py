#!/usr/bin/env python3
"""Rigorous sigma_1 at D = infinity from the exact Darboux telescoping of the order-3 Stokes multiplier.

What is computed (proof/D_entire.md, section 6):

    S^(3) = R_J + sum_{j >= J} DR_j,        DR_j = R_{j+1} - R_j,        sigma_1 = S^(3) / S^(1),  S^(1) = d0 K,

with K = 1/(Gamma(a) Gamma(b)), a = (1 + i w)/2, b = (2 + i w)/2, d0 = -i. R_J is evaluated in ball arithmetic from the
exact recursion (no fit). The tail is split as

    DR_j / K = F(j) + rho_j G(j) + r_j / K,

F and G explicit rational functions of j (the near-end convolution terms i = 1, 2, the far-end term i = j + 1, the
de-mixing and the psi-normalisation), rho_j = T^-_j / T^+_j with |rho_j| = 1 for real w. The F sum is closed by
partial fractions (digamma and Hurwitz zeta); the rho G sum by two summations by parts with an explicit remainder;
|r_j| <= B(j) = O(j^-3) by the log-convexity bound of section 6.3, summed as sup_{j >= J}(j^3 B(j)) / (2 (J - 1)^2).
Every constant is an arb ball; sup over j >= J is taken by interval evaluation in x = 1/j on [0, 1/J].

    python3 exact_sigma.py [J] [wf] [prec]        defaults J = 600, wf = 1 (w = wf * 2 pi / 1.64389375), prec = 256

wf may be given as an interval "lo,hi": the enclosure then holds for every w in it (the proof only needs w real).
--check prints the decomposition test j^3 |DR_j/K - F - rho G| <= j^3 B(j) at a few j and compares with fitted sigma_1.

    python3 exact_sigma.py --borel                 Borel-Pade poles t_p of B(t) = sum k! sigma_k t^(-k-1) from
                                                   solver/inner_amp/scan25phys (section 7), with exp(t_p)
"""
import sys
import time

import sympy as sp
from flint import acb, arb, ctx

I = acb(0, 1)

# ---------------------------------------------------------------- symbolic tail pieces
jj, ww, xx = sp.symbols("j w x", positive=True)


def _sym_tail():
    """F(j), G(j) with DR_j/K = F + rho_j G + O(j^-3); and phi(j) = rho_{j+1}/rho_j - 1."""
    Ii = sp.I
    ap_, bp_ = (1 + Ii * ww) / 2, (2 + Ii * ww) / 2
    am_, bm_ = (1 - Ii * ww) / 2, (2 - Ii * ww) / 2
    A = {1: lambda l: (l + ap_) * (l + bp_) / (l + 1), -1: lambda l: (l + am_) * (l + bm_) / (l + 1)}
    s = {1: Ii * ww, -1: -Ii * ww}
    d0 = {1: -Ii, -1: Ii}
    T1 = {1: A[1](0), -1: A[-1](0)}

    def ratio(mu, k):
        """T^mu_{j-k} / T^+_{j+1} = (coefficient, has rho_j)"""
        if mu == 1:
            return 1 / sp.Mul(*[A[1](jj - q) for q in range(0, k + 1)]), 0
        return 1 / (A[1](jj) * sp.Mul(*[A[-1](jj - q) for q in range(1, k + 1)])), 1

    pairs = {0: [(1, -1), (-1, 1)], 2: [(1, 1)]}
    g1 = {nu: sp.Rational(1, 8) * sum(d0[m1] * d0[m2] for m1, m2 in pairs[nu]) for nu in pairs}
    g2 = {nu: sp.Rational(1, 16) * sum(d0[m1] * d0[m2] * (T1[m2] + T1[m1]) for m1, m2 in pairs[nu]) for nu in pairs}
    c = (2 * jj + 2 + Ii * ww) / (4 * (jj + 1))
    F, G = sp.Integer(0), sp.Integer(0)

    def add(expr, mu, k):
        nonlocal F, G
        coef, rho = ratio(mu, k)
        if rho:
            G += expr * coef
        else:
            F += expr * coef

    for nu, mu in ((0, 1), (2, -1)):
        add(-c * g1[nu] * (2 * jj - 1 + s[mu]) * d0[mu], mu, 1)          # i = 1 in Ss(j)
        add(-g1[nu] * d0[mu], mu, 0)                                      # i = 1 in Sd(j+1)
        add(-c * g2[nu] * (2 * jj - 3 + s[mu]) * d0[mu], mu, 2)          # i = 2 in Ss(j)
        add(-g2[nu] * d0[mu], mu, 1)                                      # i = 2 in Sd(j+1)
        for m1, m2 in pairs[nu]:                                          # i = j+1 in Sd(j+1), l = 0 and l = j
            add(-d0[mu] * d0[m1] * d0[m2] / (8 * (jj + 1)), m2, 0)
            add(-d0[mu] * d0[m1] * d0[m2] / (8 * (jj + 1)), m1, 0)
    phi = A[-1](jj) / A[1](jj) - 1
    E2, h2 = 1 / (8 * Ii * ww), sp.Rational(1, 4)
    G += E2 * d0[-1] * phi                                                # de-mixing, rho_{j+1} - rho_j
    F += h2 * d0[1] * (1 / (jj + ap_) + 1 / (jj + bp_))                   # psi normalisation
    return sp.cancel(sp.together(F)), sp.cancel(sp.together(G)), sp.cancel(sp.together(phi))


# ---------------------------------------------------------------- sympy -> acb evaluation
def ev(e, env):
    if e.is_Symbol:
        return env[e]
    if e is sp.I:
        return I
    if e.is_Integer:
        return acb(int(e))
    if e.is_Rational:
        return acb(int(e.p)) / int(e.q)
    if e.is_Add:
        r = acb(0)
        for t in e.args:
            r += ev(t, env)
        return r
    if e.is_Mul:
        r = acb(1)
        for t in e.args:
            r *= ev(t, env)
        return r
    if e.is_Pow and e.exp.is_Integer:
        b, n = ev(e.base, env), int(e.exp)
        return b ** n if n >= 0 else 1 / b ** (-n)
    raise ValueError(f"cannot evaluate {e!r}")


def sup_in_x(expr_j, power, w, J):
    """rigorous bound of sup_{j >= J} j^power |expr_j(j)| for every w in the ball: x = 1/j over [0, 1/J]"""
    e = sp.cancel(sp.together(expr_j.subs(jj, 1 / xx) / xx ** power))
    n, d = sp.fraction(e)
    xb = acb(arb(1) / (2 * J) + arb(0, 0.5 / J * (1 + 1e-9)))     # contains [0, 1/J]
    return abs(ev(n, {xx: xb, ww: w}) / ev(d, {xx: xb, ww: w})).upper()


# ---------------------------------------------------------------- exact sequences, R_J
def sequences(w, J):
    """T^+_l (l <= J+2), order-2 g[nu]_m, order-3 d3_j (j <= J+1) for harmonic +1, exactly in balls"""
    a, b = (1 + I * w) / 2, (2 + I * w) / 2
    am, bm = (1 - I * w) / 2, (2 - I * w) / 2
    Tp, Tm = [acb(1)], [acb(1)]
    for l in range(J + 3):
        Tp.append(Tp[-1] * (l + a) * (l + b) / (l + 1))
        Tm.append(Tm[-1] * (l + am) * (l + bm) / (l + 1))
    d1 = {1: [-I * t for t in Tp], -1: [I * t for t in Tm]}
    s = {1: I * w, -1: -I * w}
    s1 = {mu: [(2 * l + 1 + s[mu]) * d1[mu][l] for l in range(J + 3)] for mu in (1, -1)}
    pairs = {0: [(1, -1), (-1, 1)], 2: [(1, 1)]}
    g = {}
    for nu in pairs:
        g[nu] = [acb(0)]
        for m in range(1, J + 3):
            acc = acb(0)
            for m1, m2 in pairs[nu]:
                A, B = d1[m1], d1[m2]
                for i in range(m):
                    acc += A[i] * B[m - 1 - i]
            g[nu].append(acc / (8 * m))
    chan = ((0, 1), (2, -1))
    d3 = [acb(0)]
    for j in range(J + 1):
        ss = acb(0)
        for nu, mu in chan:
            for i in range(1, j + 1):
                ss += g[nu][i] * s1[mu][j - i]
        sd = acb(0)
        for nu, mu in chan:
            for i in range(1, j + 2):
                sd += g[nu][i] * d1[mu][j + 1 - i]
        sj = (2 * j + 1 + s[1]) * d3[j] - ss
        d3.append((2 * j + 2 + s[1]) * sj / (4 * (j + 1)) - sd)
    return a, b, Tp, Tm, d3


def R_at(j, w, a, b, Tp, Tm, d3):
    """R_j / K = d3_j/T^+_j + E2 d0^- rho_j + h2 d0 (psi(a+j) + psi(b+j)), K = 1/(Gamma(a)Gamma(b)) (exact identity)"""
    E2 = 1 / (8 * I * w)
    return d3[j] / Tp[j] + E2 * I * Tm[j] / Tp[j] + (-I) * ((a + j).digamma() + (b + j).digamma()) / 4


# ---------------------------------------------------------------- the j^-3 remainder bound
def remainder_sup(w, J):
    """bound on sup_{j >= J} j^3 |r_j / K| (section 6.3); requires J >= 10"""
    xr = arb(1) / (2 * J) + arb(0, 0.5 / J * (1 + 1e-9))     # contains x = 1/j for all j >= J
    w = w.real
    W = (w * w / 4).upper()                                 # w^2/4 upper
    om = abs(w).upper()
    r = lambda m: ((arb(m) + 0.5) ** 2 + w * w / 4).sqrt() * ((arb(m) + 1) ** 2 + w * w / 4).sqrt() / (m + 1)
    # r_m = |a_m| increasing for m >= 1: checked here up to m0, and r_{m+1} >= m + 3/2 > m + 1 + W/(m+1) >= r_m beyond
    m0 = int(2 * float(W)) + 2
    for m in range(1, m0 + 1):
        assert (r(m + 1) - r(m)) > 0, "r_m not increasing"
    t = [arb(1)]
    for m in range(4):
        t.append(t[-1] * r(m))
    t = [abs(v).upper() for v in t]
    t1, t2, t3 = t[1], t[2], t[3]
    gam = arb(3) * (2 + t1) / 8                              # |g[0]_i| + |g[2]_i| <= gam t_{i-1} / i
    inv = lambda k: 1 / _prod([1 - (q - 0.5) * xr for q in range(k + 1)])
    cj = (2 + 2 * xr + om * xr) / (4 * (1 + xr))             # |c_j|
    a_ = (gam / 3 * t2 * (2 - 5 * xr + om * xr) * inv(3)                      # Ss, i = 3
          + gam / 4 * (2 + om * xr) * (1 - 6 * xr) * t3 * inv(4)               # Ss, 4 <= i <= j-3
          + gam * (5 + om) * t2 * xr * xr / (1 - 2 * xr) * inv(3)               # Ss, i = j-2
          + gam * (3 + om) * t1 * xr / (1 - xr) * inv(2)                        # Ss, i = j-1
          + gam * (1 + om) * inv(1)) * cj                                       # Ss, i = j
    b_ = (gam / 3 * t2 * inv(2)                                                 # Sd, i = 3
          + gam / 4 * t3 * (1 - 5 * xr) * inv(3)                                # Sd, 4 <= i <= j-2
          + gam * t2 * xr / (1 - xr) * inv(2)                                   # Sd, i = j-1
          + gam * t1 * inv(1))                                                  # Sd, i = j
    c_ = arb(3) / 8 / (1 + xr) * (2 * t1 * inv(1) + (1 - 3 * xr) * t2 * inv(2))  # Sd, i = j+1, 1 <= l <= j-1
    return (abs(a_) + abs(b_) + abs(c_)).upper()


def _prod(v):
    p = arb(1)
    for z in v:
        p *= z
    return p


# ---------------------------------------------------------------- tail sums
def sum_F(F, w, J):
    """sum_{j >= J} F(j) exactly: partial fractions in j, then digamma / Hurwitz zeta.
    F = O(j^-2), so the simple-pole residues sum to zero (checked symbolically) and
    sum_{j>=J} sum_p c_p/(j - p) = -sum_p c_p (psi(J - p) - psi(J))."""
    pf = sp.apart(F, jj, full=False)
    tot = acb(0)
    res1 = sp.Integer(0)
    for term in sp.Add.make_args(pf):
        n, d = sp.fraction(sp.factor(term))
        base = sp.factor_list(d, jj)
        facs = [(f, k) for f, k in base[1] if f.has(jj)]
        assert len(facs) == 1 and sp.Poly(facs[0][0], jj).degree() == 1 and not n.has(jj), term
        f, k = facs[0]
        cf = sp.Poly(f, jj).all_coeffs()                    # f = c1 j + c0
        coef_s = n / base[0] / cf[0] ** k                   # term = coef / (j - p)^k
        coef, p = ev(coef_s, {ww: w}), ev(-cf[1] / cf[0], {ww: w})
        if k == 1:
            res1 += coef_s
            tot += -coef * ((J - p).digamma() - acb(J).digamma())
        else:
            tot += coef * acb(k).zeta(J - p)
    assert sp.simplify(res1) == 0, "1/j terms must cancel"
    return tot


def sum_rhoG(G, phi, w, J, Tp, Tm, nsbp=2):
    """sum_{j >= J} rho_j G(j) by nsbp summations by parts; returns (value, error bound)"""
    tot = acb(0)
    g = G
    for n in range(nsbp):
        Q = sp.cancel(sp.together(g / phi))
        jn = J + n
        tot -= (Tm[jn] / Tp[jn]) * ev(Q, {jj: acb(jn), ww: w})
        g = sp.cancel(sp.together(Q.subs(jj, jj - 1) - Q))
    C = sup_in_x(g, 2, w, J + nsbp)
    return tot, C / (J + nsbp - 1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    J = int(args[0]) if args else 600
    wf = args[1] if len(args) > 1 else "1"
    ctx.prec = int(args[2]) if len(args) > 2 else 256
    if "," in wf:
        lo, hi = (arb(v) for v in wf.split(","))
        wfb = (lo + hi) / 2 + arb(0, (abs(hi - lo) / 2).upper())
    else:
        wfb = arb(wf)
    w = acb(wfb * 2 * arb.pi() / arb("1.64389375"))
    t0 = time.time()
    F, G, phi = _sym_tail()
    a, b, Tp, Tm, d3 = sequences(w, J + 4)
    RJ = R_at(J, w, a, b, Tp, Tm, d3)
    sF = sum_F(F, w, J)
    sG, eG = sum_rhoG(G, phi, w, J, Tp, Tm)
    C3 = remainder_sup(w, J)
    eB = C3 / (2 * (J - 1) ** 2)
    S3K = RJ + sF + sG                                       # S^(3)/K
    err = arb(eG + eB)
    sig1 = S3K / (-I)
    sig1 = acb(sig1.real + arb(0, err), sig1.imag + arb(0, err))
    print(f"J = {J}, w = {w.real.str(12)}, prec = {ctx.prec}, {time.time() - t0:.1f} s")
    print(f"  R_J/S1 (no tail)            = {(RJ / (-I)).str(12)}")
    print(f"  tail F (digamma/zeta)       = {(sF / (-I)).str(12)}")
    print(f"  tail rho G (2 x SBP)        = {(sG / (-I)).str(12)}   remainder <= {float(eG):.3e}")
    print(f"  sup j^3|r_j/K| <= {float(C3):.3f}  ->  sum_{{j>=J}} <= {float(eB):.3e}")
    print(f"  sigma_1 in ({sig1.real.mid().str(12, radius=False)} +- {float(sig1.real.rad()):.2e}) + ({sig1.imag.mid().str(12, radius=False)} +- {float(sig1.imag.rad()):.2e}) i")
    print(f"  half-widths: re {float(sig1.real.rad()):.2e}, im {float(sig1.imag.rad()):.2e}")
    if "--check" in sys.argv:
        x = {jj: None, ww: w}
        for j in (50, 100, 200, J - 1):
            dR = R_at(j + 1, w, a, b, Tp, Tm, d3) - R_at(j, w, a, b, Tp, Tm, d3)
            x[jj] = acb(j)
            r = dR - ev(F, x) - (Tm[j] / Tp[j]) * ev(G, x)
            print(f"  j = {j}: j^2|DR/K| = {float(abs(dR).mid()) * j * j:.4f},  j^3|r/K| = {float(abs(r).mid()) * j ** 3:.4f}")
        print("  fitted (amp_flint_par, wf = 1): sigma_1 = 0.743584579922 + 0.52447485276i")
    return sig1


def borel_poles(scan="../inner_amp/scan25phys"):
    """Pade poles of the Borel transform of S_1(x) = sum sigma_k x^k, and exp of each pole (section 7; floating point)"""
    import glob
    import json
    import os
    import mpmath as mp
    mp.mp.dps = 50
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(glob.glob(os.path.join(here, scan, "*.json"))):
        r = json.load(open(fn))
        sig = [mp.mpc(mp.mpf(u), mp.mpf(v)) for u, v in r["sigma_str"]]
        bk = [mp.factorial(k) * sig[k] for k in range(len(sig))]
        n = len(bk) - 1
        for L in (n // 2 - 1, n // 2, n // 2 + 1):
            M = n - L
            p, q = mp.pade(bk[:n + 1], L, M)
            ts = sorted((1 / u for u in mp.polyroots(q[::-1], maxsteps=200, extraprec=200)), key=lambda t: float(abs(t)))
            row = "  ".join(f"{mp.nstr(t, 4)} (e^t {mp.nstr(mp.exp(t), 3)})" for t in ts if 0.3 < abs(t) < 3)
            print(f"wf {float(r['wf']):.2f} [{L}/{M}]: {row}")


if __name__ == "__main__":
    if "--borel" in sys.argv:
        borel_poles()
    else:
        main()
