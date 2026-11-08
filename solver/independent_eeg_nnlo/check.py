#!/usr/bin/env python3
"""Order-by-order check of the large-D series of C. Ecker, F. Ecker, D. Grumiller, "Critical collapse remains
discretely self-similar at large D", arXiv:2611.07007v1 (4 Nov 2026), supplemental section "Simplified large-D
perturbation theory", Eqs. (S35)-(S51), against the paper's own field equations (S36)-(S39).

Written for issue #142 from the paper alone. Transcribed from PDF page 11 (rendered locally) and cross-checked
against the arXiv HTML rendering. Every coefficient agreed between the two.

Sets of NNLO terms tested (the NLO terms (S40)-(S43) are the printed ones in every set):
  printed   (S44)-(S51) as printed, reading the plain beta of (S43)-(S51) as beta-tilde (reading A).
  printedB  the same, reading plain beta literally as beta = sqrt(eps) beta-tilde (reading B, main-text Eq. (10)).
  claimed   the corrected Pi_3, Omega_3, f_3 stated in issue #142.
  derived   this script's own solution of (S36)-(S39), order by order, with the conventions (S29) f(tau,1) = 1 and
            (S34) Pi(tau,0) = sqrt(eps) beta-tilde exactly.

Usage (from the repository root):
  python3 solver/independent_eeg_nnlo/check.py                 # all sets, summary, exit 0 iff the claim holds
  python3 solver/independent_eeg_nnlo/check.py --set=claimed   # one set, exit 0 iff it passes every fixed order
  python3 solver/independent_eeg_nnlo/check.py --nlo           # NLO control only, exit 0 iff it passes
  python3 solver/independent_eeg_nnlo/check.py --numeric       # eps-scaling of the (S39) residual, mpmath;
                                                               # exit 0 iff claimed and derived (or --set) are
                                                               # O(eps^(7/2))
  ... --mutate=NAME                                            # perturb one coefficient (names in MUTATIONS)
  ... --json=PATH                                              # write the residual table as JSON

Exit codes. Summary mode: 0 claim confirmed (NLO passes, claimed passes, derived == claimed, printed fails first
at (S39) eps^(5/2)); otherwise a bitmask, 1 NLO control fails, 2 claimed set fails, 4 derived != claimed,
8 printed does not fail where the claim says, 16 a set's first failure is outside (S39) eps^(5/2). --set and --nlo
modes: 0 pass, 1 fail.
"""
import argparse
import json
import sys

import sympy as sp

# Mutations. Each perturbs one transcribed coefficient and names the residual order it enters.
MUTATIONS = {
    'S40': "f2: 4(1-x^2) -> 5(1-x^2); enters (S37) at eps^2",
    'S41': "Omega2: 3 b' -> 4 b'; enters (S36) at eps^1",
    'S42': "Pi2: 2 b -> 3 b; enters (S39) at eps^(3/2)",
    'S43': "Psi2: 11 b' -> 12 b'; enters (S38) at eps^(3/2)",
    'S45': "Omega3: -7 b^2 -> -6 b^2; enters (S36) at eps^2",
    'S49': "c2: 3/16 -> 1/16; enters (S37) at eps^3 (printed and claimed sets)",
    'S46': "Pi3 printed: 24 b^3 -> 12 b^3; enters (S39) at eps^(5/2)",
    'claimPi3': "claimed Pi3: 50 b1 -> 49 b1; enters (S39) at eps^(5/2)",
    'claimC4': "claimed c4: 5/4 b0^4 -> 9/4 b0^4 (the printed value); enters (S37) at eps^3",
}
MUT = set()


def m(name, val, mutated):
    """Coefficient val, or mutated when --mutate=name is active."""
    return mutated if name in MUT else val


R = sp.Rational
x, s = sp.symbols('x s', positive=True)   # s = sqrt(eps), eps = 1/(D-1)
NB = 10
b = sp.symbols('b0:%d' % NB)               # b[k] = k-th tau-derivative of beta-tilde


def Dt(e):
    """d/dtau on polynomials in b_k. Requires the top derivative absent from e."""
    e = sp.expand(e)
    assert sp.diff(e, b[NB - 1]) == 0
    return sp.expand(sum(sp.diff(e, b[k]) * b[k + 1] for k in range(NB - 1)))


# --------------------------------------------------------------------------------------------------------------
# Transcription of (S35)-(S51).

def lo():
    """(S35a), (S35b): leading terms. Tildes are printed on every beta here."""
    eps = s**2
    return dict(
        f1=-(1 - x**2) / 2 * b[0]**2,      # (S35a) f = 1 - eps (1-x^2)/2 bt^2 + sum_{n>=2} f_n eps^n
        O1=b[0]**2,                         # (S35a) Omega = eps bt^2 + sum_{n>=2} Omega_n eps^n
        P1=b[0],                            # (S35b) Pi = sqrt(eps)^(-1) (eps bt + sum_{n>=2} Pi_n eps^n)
        S1=b[1] + b[0],                     # (S35b) Psi = sqrt(eps) (eps (bt' + bt) + sum_{n>=2} Psi_n eps^n)
    )


def nlo(beta):
    """(S40)-(S43). (S40)-(S42) print beta-tilde; (S43) prints plain beta, read through beta(k)."""
    bt = b
    B = beta
    return dict(
        # (S40) f2 = 1/8 (1-x^2) bt (4(1-x^2) bt + (3+x^2) bt^3 - 2(1+x^2)(bt'' + 3 bt'))
        f2=R(1, 8) * (1 - x**2) * bt[0] * (m('S40', 4, 5) * (1 - x**2) * bt[0] + (3 + x**2) * bt[0]**3
                                            - 2 * (1 + x**2) * (bt[2] + 3 * bt[1])),
        # (S41) Omega2 = bt (x^2 (bt'' + 3 bt' - bt^3) - 2(1-x^2) bt)
        O2=bt[0] * (x**2 * (bt[2] + m('S41', 3, 4) * bt[1] - bt[0]**3) - 2 * (1 - x**2) * bt[0]),
        # (S42) Pi2 = x^2/2 (2 bt + bt'' + 3 bt' - bt^3)
        P2=x**2 / 2 * (m('S42', 2, 3) * bt[0] + bt[2] + 3 * bt[1] - bt[0]**3),
        # (S43) Psi2 = 1/2 (b^3 + b^2 b' + x^2 (-4 b^3 - 4 b^2 b' + 6 b + 11 b' + 6 b'' + b''')), plain beta printed
        S2=R(1, 2) * (B(0)**3 + B(0)**2 * B(1)
                      + x**2 * (-4 * B(0)**3 - 4 * B(0)**2 * B(1) + 6 * B(0) + m('S43', 11, 12) * B(1)
                                + 6 * B(2) + B(3))),
    )


def printed_pi3(B):
    """(S46) Pi3 = x^2/6 (24 b^3 + 15 b^2 b' + 3 b^2 b'') + x^4/6 (3 b^5 - 16 b^3 - 20 b^2 b' - 4 b^2 b''
    + 12 b + 22 b' + 12 b'' + 2 b'''). Plain beta printed."""
    return (x**2 / 6 * (m('S46', 24, 12) * B(0)**3 + 15 * B(0)**2 * B(1) + 3 * B(0)**2 * B(2))
            + x**4 / 6 * (3 * B(0)**5 - 16 * B(0)**3 - 20 * B(0)**2 * B(1) - 4 * B(0)**2 * B(2) + 12 * B(0)
                          + 22 * B(1) + 12 * B(2) + 2 * B(3)))


def omega3(B, Pi3):
    """(S45) Omega3 = 2 b Pi3 + x^4/4 (b^6 - 4 b^4 - 6 b^3 b' - 2 b^3 b'' + 4 b^2 + 12 b b' + 4 b b'' + 9 (b')^2
    + 6 b' b'' + (b'')^2) + x^2 (4 b^4 - 7 b^2 - 10 b b' - 4 b b'' + (b')^2). Plain beta printed. Pi3 is passed in,
    so the claim's Omega3 is this form with its own Pi3."""
    return (2 * B(0) * Pi3
            + x**4 / 4 * (B(0)**6 - 4 * B(0)**4 - 6 * B(0)**3 * B(1) - 2 * B(0)**3 * B(2) + 4 * B(0)**2
                          + 12 * B(0) * B(1) + 4 * B(0) * B(2) + 9 * B(1)**2 + 6 * B(1) * B(2) + B(2)**2)
            + x**2 * (4 * B(0)**4 + m('S45', -7, -6) * B(0)**2 - 10 * B(0) * B(1) - 4 * B(0) * B(2) + B(1)**2))


def printed_c(B):
    """(S48)-(S51). Plain beta printed."""
    c0 = (-R(17, 48) * B(0)**6 - R(79, 36) * B(0)**4 + R(35, 72) * B(0)**3 * B(1) + R(13, 72) * B(0)**3 * B(2)
          + R(11, 12) * B(0)**2 + R(7, 9) * B(0) * B(1) + R(1, 6) * B(0) * B(2) - R(1, 9) * B(0) * B(3)
          - R(5, 8) * B(1)**2 - R(1, 4) * B(1) * B(2) - R(1, 24) * B(2)**2)                          # (S48)
    c2 = (m('S49', R(3, 16), R(1, 16)) * B(0)**6 + R(3, 4) * B(0)**4 - R(3, 8) * B(0)**3 * B(1)
          - R(1, 8) * B(0)**3 * B(2))                                                                 # (S49)
    c4 = (R(1, 16) * B(0)**6 + R(9, 4) * B(0)**4 + R(7, 8) * B(0)**3 * B(1) + R(1, 8) * B(0)**3 * B(2)
          - R(7, 4) * B(0)**2 - R(5, 2) * B(0) * B(1) - B(0) * B(2) + R(1, 4) * B(1)**2)              # (S50)
    c6 = (R(5, 48) * B(0)**6 - R(29, 36) * B(0)**4 - R(71, 72) * B(0)**3 * B(1) - R(13, 72) * B(0)**3 * B(2)
          + R(5, 6) * B(0)**2 + R(31, 18) * B(0) * B(1) + R(5, 6) * B(0) * B(2) + R(1, 9) * B(0) * B(3)
          + R(3, 8) * B(1)**2 + R(1, 4) * B(1) * B(2) + R(1, 24) * B(2)**2)                           # (S51)
    return c0, c2, c4, c6


def f3_of(c):
    """(S44) f3 = c0 + x^2 c2 + x^4 c4 + x^6 c6."""
    c0, c2, c4, c6 = c
    return c0 + x**2 * c2 + x**4 * c4 + x**6 * c6


def claimed_terms():
    """The corrected terms stated in issue #142, with b_k = beta-tilde^(k)."""
    b0, b1, b2, b3, b4 = b[:5]
    Pi3 = (x**2 / 2 * b0 * (4 * b0**2 + 4 * b0 * b1 + b0 * b2 + b1**2)
           + x**4 / 8 * (3 * b0**5 - 24 * b0**3 - 40 * b0**2 * b1 - 8 * b0**2 * b2 - 8 * b0 * b1**2 + 24 * b0
                         + m('claimPi3', 50, 49) * b1 + 35 * b2 + 10 * b3 + b4))
    c0 = (-R(5, 16) * b0**6 - R(13, 12) * b0**4 + R(31, 24) * b0**3 * b1 + R(7, 24) * b0**3 * b2
          + R(1, 12) * b0**2 * b1**2 + R(7, 12) * b0**2 - R(1, 12) * b0 * b1 - R(5, 8) * b0 * b2
          - R(5, 12) * b0 * b3 - R(1, 24) * b0 * b4 - R(5, 8) * b1**2 - R(1, 4) * b1 * b2 - R(1, 24) * b2**2)
    c2 = printed_c(lambda k: b[k])[1]     # "c2 as printed in (S49)"
    c4 = (R(1, 16) * b0**6 + m('claimC4', R(5, 4), R(9, 4)) * b0**4 + R(5, 8) * b0**3 * b1
          + R(1, 8) * b0**3 * b2 + R(1, 4) * b0**2 * b1**2 - R(7, 4) * b0**2 - R(5, 2) * b0 * b1 - b0 * b2
          + R(1, 4) * b1**2)
    c6 = (R(1, 16) * b0**6 - R(11, 12) * b0**4 - R(37, 24) * b0**3 * b1 - R(7, 24) * b0**3 * b2
          - R(1, 3) * b0**2 * b1**2 + R(7, 6) * b0**2 + R(31, 12) * b0 * b1 + R(13, 8) * b0 * b2
          + R(5, 12) * b0 * b3 + R(1, 24) * b0 * b4 + R(3, 8) * b1**2 + R(1, 4) * b1 * b2 + R(1, 24) * b2**2)
    return Pi3, (c0, c2, c4, c6)


def reading(name):
    """beta(k) for the plain beta of (S43)-(S51). A: beta-tilde. B: sqrt(eps) beta-tilde, literally."""
    if name == 'A':
        return lambda k: b[k]
    return lambda k: s * b[k]


def build_set(name):
    """Coefficient functions of one set. Psi3 is 0 everywhere ((S47) "not needed")."""
    t = lo()
    if name == 'derived':
        t.update(derive())
        return t
    B = reading('B' if name == 'printedB' else 'A')
    t.update(nlo(B))
    if name in ('printed', 'printedB'):
        Pi3 = printed_pi3(B)
        t.update(P3=Pi3, O3=omega3(B, Pi3), f3=f3_of(printed_c(B)))
    elif name == 'claimed':
        Pi3, c = claimed_terms()
        t.update(P3=Pi3, O3=omega3(reading('A'), Pi3), f3=f3_of(c))
    elif name == 'nlo':
        t.update(P3=0, O3=0, f3=0)
    t['S3'] = 0
    return t


# --------------------------------------------------------------------------------------------------------------
# Residuals of (S36)-(S39) as Laurent polynomials in s = sqrt(eps).

def fields(t):
    eps = s**2
    f = 1 + eps * t['f1'] + eps**2 * t['f2'] + eps**3 * t['f3']
    Om = eps * t['O1'] + eps**2 * t['O2'] + eps**3 * t['O3']
    Pi = (eps * t['P1'] + eps**2 * t['P2'] + eps**3 * t['P3']) / s
    Psi = s * (eps * t['S1'] + eps**2 * t['S2'] + eps**3 * t['S3'])
    return [sp.expand(e) for e in (f, Om, Pi, Psi)]


def residuals(t):
    """LHS - RHS of (S36)-(S39). (S37) uses ln f = sum_{k<=4} (-1)^(k+1) u^k / k, u = f - 1 = O(eps), exact
    through eps^4."""
    f, Om, Pi, Psi = fields(t)
    eps = s**2
    r36 = (x * sp.diff(Om, x) + x**2 * Om**2
           - ((1 / eps) * ((1 - 2 * eps) * Pi**2 - Om) + Om * (Pi**2 * x**2 + x**4 * Psi**2)
              + (1 / eps - 2) * x**2 * Psi**2))
    u = f - 1
    L = sum((-1)**(k + 1) * u**k / k for k in range(1, 5))
    r37 = sp.diff(sp.expand(L), x) / x - Om
    r38 = (Dt(Pi) + x * sp.diff(Pi, x)
           - (f * x * sp.diff(Psi, x) + f * (1 / eps + x**2 * Om) * Psi - Pi))
    r39 = (Dt(Psi) + x * sp.diff(Psi, x)
           - (f * sp.diff(Pi, x) / x + f * Om * Pi - 2 * Psi))
    return {'S36': r36, 'S37': r37, 'S38': r38, 'S39': r39}


def coeffs(e, kmax):
    """{k: coefficient of s^k} for k <= kmax, from a Laurent polynomial in s."""
    out = {}
    for term in sp.Add.make_args(sp.expand(e)):
        c, k = term.as_coeff_exponent(s)
        if k <= kmax:
            out[k] = out.get(k, 0) + c
    return {k: sp.expand(v) for k, v in sorted(out.items())}


# Orders (powers of s) that the printed terms fix. Beyond them a term not given (Omega4, f4, Psi3, Pi4) enters.
FIXED = {'S36': [0, 2, 4], 'S37': [2, 4, 6], 'S38': [1, 3], 'S39': [1, 3, 5]}
NLO_ORDERS = {'S36': [0, 2], 'S37': [2, 4], 'S38': [1, 3], 'S39': [1, 3]}
KMAX = {'S36': 6, 'S37': 8, 'S38': 5, 'S39': 7}


def order_label(k):
    return 'eps^%s' % (sp.Rational(k, 2))


def table(t, orders=None):
    """Per equation: every power of s up to KMAX with its coefficient's zero status and term count. orders
    names the powers the given terms fix (FIXED by default)."""
    orders = orders or FIXED
    res = residuals(t)
    out = {}
    for eq, e in res.items():
        cs = coeffs(e, KMAX[eq])
        rows = []
        for k in range(min(list(cs) + [0]), KMAX[eq] + 1):
            c = cs.get(k, 0)
            rows.append(dict(k=k, order=order_label(k), zero=(c == 0),
                             nterms=len(sp.Add.make_args(c)) if c != 0 else 0, fixed=(k <= max(orders[eq])),
                             coeff=str(c) if c != 0 and len(str(c)) < 400 else ('' if c == 0 else '(long)')))
        out[eq] = rows
    return out, res


def verdict(tab, orders):
    """First power of s, among all powers up to the last fixed one, where a residual is nonzero."""
    fails = []
    for eq, rows in tab.items():
        top = max(orders[eq])
        for r in rows:
            if r['k'] <= top and not r['zero']:
                fails.append((r['k'], eq))
    return sorted(fails)


def print_table(name, tab):
    print('\n== set %s ==' % name)
    for eq, rows in tab.items():
        cells = []
        for r in rows:
            st = '0' if r['zero'] else ('FAIL(%d)' % r['nterms'] if r['fixed'] else 'open(%d)' % r['nterms'])
            cells.append('%s:%s' % (r['order'], st))
        print('  %s  %s' % (eq, '  '.join(cells)))
    print('  key: 0 = coefficient vanishes; FAIL(n) = nonzero at or below the last order the tested terms fix')
    print('       (n terms); open(n) = nonzero at an order that needs a term not given or not tested.')


# --------------------------------------------------------------------------------------------------------------
# Independent solution, order by order.

def solve_linear(rfun, unknown_key, t, mode):
    """Solve one order's residual for one unknown coefficient function. rfun(t) returns that order's residual.
    mode 'alg': residual = a*U + r0 with a independent of x. mode 'pi': residual = a*(1/x) dU/dx + r0,
    U(0) = 0 (S34). mode 'f': residual = a*(1/x) dU/dx + r0, U(1) = 0 (S29). a is measured, not assumed."""
    xi = sp.Symbol('xi', positive=True)
    t0 = dict(t)
    t0[unknown_key] = 0
    r0 = sp.expand(rfun(t0))
    t1 = dict(t)
    t1[unknown_key] = 1 if mode == 'alg' else x**2 / 2
    a = sp.expand(rfun(t1) - r0)
    assert a.free_symbols <= set(b) and a != 0, (unknown_key, a)
    g = sp.expand(-r0 / a)
    if mode == 'alg':
        return g
    if mode == 'pi':
        return sp.expand(sp.integrate((xi * g.subs(x, xi)), (xi, 0, x)))
    return sp.expand(-sp.integrate((xi * g.subs(x, xi)), (xi, x, 1)))


def derive():
    """Solve (S36)-(S39) through the orders fixed by Pi3, Omega3, f3. Unknowns not yet solved are 0."""
    t = lo()
    for k in ('f2', 'O2', 'P2', 'S2', 'f3', 'O3', 'P3', 'S3'):
        t[k] = 0

    def coef(eq, k):
        return lambda tt: coeffs(residuals(tt)[eq], k).get(k, 0)

    # Leading order is (S35) itself; verified below by the residual table of this set.
    t['P2'] = solve_linear(coef('S39', 3), 'P2', t, 'pi')     # (S39) at eps^(3/2)
    t['S2'] = solve_linear(coef('S38', 3), 'S2', t, 'alg')    # (S38) at eps^(3/2)
    t['O2'] = solve_linear(coef('S36', 2), 'O2', t, 'alg')    # (S36) at eps^1
    t['f2'] = solve_linear(coef('S37', 4), 'f2', t, 'f')      # (S37) at eps^2
    t['P3'] = solve_linear(coef('S39', 5), 'P3', t, 'pi')     # (S39) at eps^(5/2)
    t['O3'] = solve_linear(coef('S36', 4), 'O3', t, 'alg')    # (S36) at eps^2
    t['f3'] = solve_linear(coef('S37', 6), 'f3', t, 'f')      # (S37) at eps^3
    return {k: t[k] for k in ('f2', 'O2', 'P2', 'S2', 'f3', 'O3', 'P3', 'S3')}


def uniqueness(t):
    """The residual at each NNLO order is affine in its unknown: R(U + delta) - R(U) = a L[delta], with L the
    operator named. delta is a generic function of x (tau dependence enters only through b_k)."""
    d = sp.Function('delta')(x)
    out = {}
    for key, eq, k, op in (('P3', 'S39', 5, 'xinv_dx'), ('O3', 'S36', 4, 'id'), ('f3', 'S37', 6, 'xinv_dx')):
        t1 = dict(t)
        t1[key] = t[key] + d
        diff = sp.expand(coeffs(residuals(t1)[eq], k).get(k, 0) - coeffs(residuals(t)[eq], k).get(k, 0))
        L = sp.diff(d, x) / x if op == 'xinv_dx' else d
        a = sp.simplify(diff / L)
        out[key] = dict(eq=eq, order=order_label(k), operator=('(1/x) d/dx' if op == 'xinv_dx' else 'identity'),
                        factor=str(a), linear=(a.free_symbols <= set(b) and a != 0))
    return out


# --------------------------------------------------------------------------------------------------------------
# Numerical eps-scaling of the (S39) residual, mpmath. Shares the transcription and lambdify with the symbolic
# path, not the coefficient extraction.

def numeric(sets):
    import mpmath as mp
    mp.mp.dps = 60
    tau = sp.Symbol('tau')
    bt = sp.Rational(7, 10) * sp.sin(tau) + sp.Rational(2, 5) * sp.cos(2 * tau) + sp.Rational(1, 5)
    sub = {b[k]: sp.diff(bt, tau, k) for k in range(NB)}
    epsy = sp.Symbol('eps', positive=True)
    out = {}
    for name in sets:
        f, Om, Pi, Psi = fields(build_set(name))
        F = [sp.lambdify((tau, x, epsy), e.subs(sub).subs(s, sp.sqrt(epsy)), 'mpmath') for e in (f, Om, Pi, Psi)]

        def r39(t0, x0, e0):
            dPsi_t = mp.diff(lambda tt: F[3](tt, x0, e0), t0)
            dPsi_x = mp.diff(lambda xx: F[3](t0, xx, e0), x0)
            dPi_x = mp.diff(lambda xx: F[2](t0, xx, e0), x0)
            fv, Ov, Pv, Sv = (g(t0, x0, e0) for g in F)
            return dPsi_t + x0 * dPsi_x - (fv * dPi_x / x0 + fv * Ov * Pv - 2 * Sv)

        t0, x0 = mp.mpf('0.3'), mp.mpf('0.8')
        es = [mp.mpf(10)**(-k) for k in (3, 4, 5, 6)]
        vals = [abs(r39(t0, x0, e)) for e in es]
        slopes = [float(mp.log10(vals[i] / vals[i + 1])) for i in range(len(vals) - 1)]
        out[name] = dict(eps=[float(e) for e in es], absR39=[mp.nstr(v, 8) for v in vals], slopes=slopes)
        print('  %-8s |R39| at eps=1e-3..1e-6: %s   local exponents: %s'
              % (name, ', '.join(mp.nstr(v, 4) for v in vals), ', '.join('%.3f' % q for q in slopes)))
    return out


# --------------------------------------------------------------------------------------------------------------

def compare(t, u, keys):
    return {k: sp.expand(t[k] - u[k]) == 0 for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', choices=['printed', 'printedB', 'claimed', 'derived'])
    ap.add_argument('--nlo', action='store_true')
    ap.add_argument('--numeric', action='store_true')
    ap.add_argument('--mutate', default='', help=', '.join('%s (%s)' % kv for kv in MUTATIONS.items()))
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    for name in filter(None, a.mutate.split(',')):
        if name not in MUTATIONS:
            sys.exit('unknown mutation %s; choose from %s' % (name, ', '.join(MUTATIONS)))
        MUT.add(name)
        print('mutation active: %s: %s' % (name, MUTATIONS[name]))
    report = dict(mutations=sorted(MUT))

    if a.numeric:
        # Pass iff each tested set's (S39) residual is O(eps^(7/2)), i.e. its last local exponent exceeds 3.4.
        # A set wrong at eps^(5/2) gives 2.5. --set picks the tested set; printed is always shown for reference.
        tested = [a.set] if a.set else ['claimed', 'derived']
        print('numeric (S39) residual scaling at tau = 0.3, x = 0.8, bt = 0.7 sin tau + 0.4 cos 2tau + 0.2')
        report['numeric'] = numeric(list(dict.fromkeys(['printed'] + tested)))
        bad = [n for n in tested if report['numeric'][n]['slopes'][-1] < 3.4]
        print('\nverdict numeric %s: %s' % (','.join(tested), 'PASS' if not bad else
                                           'FAIL, residual not O(eps^(7/2)) for %s' % ','.join(bad)))
        rc = 1 if bad else 0
    elif a.nlo or a.set:
        name = 'nlo' if a.nlo else a.set
        orders = NLO_ORDERS if a.nlo else FIXED
        tab, _ = table(build_set(name), orders)
        print_table(name, tab)
        fails = verdict(tab, orders)
        print('\nverdict %s: %s' % (name, 'PASS' if not fails else
                                     'FAIL, first at %s of (%s)' % (order_label(fails[0][0]), fails[0][1])))
        report[name] = dict(table=tab, fails=[(order_label(k), e) for k, e in fails])
        rc = 0 if not fails else 1
    else:
        rc = 0
        tabs = {}
        for name in ('nlo', 'printed', 'printedB', 'claimed', 'derived'):
            tab, _ = table(build_set(name), NLO_ORDERS if name == 'nlo' else None)
            tabs[name] = tab
            print_table(name, tab)
        nlo_f = verdict(tabs['nlo'], NLO_ORDERS)
        res = {name: verdict(tabs[name], FIXED) for name in ('printed', 'printedB', 'claimed', 'derived')}
        print('\nNLO control (S35)-(S43): %s' % ('PASS' if not nlo_f else 'FAIL %s' % nlo_f))
        for name, fl in res.items():
            print('set %-8s through fixed orders: %s' % (
                name, 'PASS' if not fl else 'FAIL at ' + ', '.join('(%s) %s' % (e, order_label(k)) for k, e in fl)))
        der, cla = build_set('derived'), build_set('claimed')
        pri = build_set('printed')
        keys = ('P2', 'S2', 'O2', 'f2', 'P3', 'O3', 'f3')
        same_c = compare(der, cla, keys)
        same_p = compare(der, pri, keys)
        print('\nderived == claimed:  %s' % same_c)
        print('derived == printed:  %s' % same_p)
        for k in ('P3', 'O3', 'f3'):
            print('  printed - derived  %s = %s' % (k, sp.factor(sp.expand(pri[k] - der[k]))))
        # f3 by powers of x, for the c_k comparison.
        for nm, tt in (('printed', pri), ('claimed', cla), ('derived', der)):
            p = sp.Poly(tt['f3'], x)
            print('  %s f3 powers of x: %s' % (nm, sorted(m_[0] for m_ in p.monoms())))
        # Boundary conventions.
        for nm, tt in (('printed', pri), ('claimed', cla), ('derived', der)):
            print('  %-8s f3(x=1) = %s   Pi3(x=0) = %s   Omega3(x=0) = %s' % (
                nm, sp.expand(tt['f3'].subs(x, 1)), sp.expand(tt['P3'].subs(x, 0)),
                sp.expand(tt['O3'].subs(x, 0))))
        uq = uniqueness(der)
        print('\nuniqueness (residual linear in the unknown):')
        for k, v in uq.items():
            print('  %s at (%s) %s: R(U+delta)-R(U) = %s * %s[delta]  linear=%s'
                  % (k, v['eq'], v['order'], v['factor'], v['operator'], v['linear']))
        if nlo_f:
            rc |= 1
        if res['claimed']:
            rc |= 2
        if not all(same_c.values()):
            rc |= 4
        if not res['printed'] or res['printed'][0] != (5, 'S39'):
            rc |= 8
        if [fl for fl in res['printed'] if fl != (5, 'S39')]:
            rc |= 16
        print('\nclaim: printed fails first at (S39) eps^(5/2) and nowhere else below the fixed orders: %s'
              % (res['printed'] == [(5, 'S39')]))
        report.update(tables=tabs, nlo_fails=nlo_f,
                      fails={k: [(order_label(q), e) for q, e in v] for k, v in res.items()},
                      derived_eq_claimed={k: bool(v) for k, v in same_c.items()},
                      derived_eq_printed={k: bool(v) for k, v in same_p.items()},
                      uniqueness=uq,
                      derived={k: str(der[k]) for k in keys})
    report['exit'] = rc
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(report, fh, indent=1, default=str)
    print('\nexit %d' % rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
