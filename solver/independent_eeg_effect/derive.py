"""Symbolic check of the claimed NNLO corrections against the GG equations (S36)-(S39) of arXiv:2611.07007v1.

Inserts the expansions (S35), with the printed (S40)-(S51) plus the claimed dPi3, df3 of #176, into
(S36)-(S39) and requires every coefficient of sqrt(eps) that contains no undetermined higher order to
vanish. It then rederives Pi3, Omega3 and f3 from (S39), (S36) and (S37) with Pi3(tau, 0) = 0 and
f(tau, 1) = 1, and compares them with printed + delta.

    python3 derive.py                    # exit 0 when the corrected NNLO satisfies every determined order
    python3 derive.py --mutate=printed   # control, the printed Pi3 and f3 without delta, must exit 1
"""
import argparse
import sys

import sympy as sp

from fields import b, x, printed, corrections

s = sp.Symbol('s', positive=True)  # s = sqrt(eps)
t = sp.Symbol('t', real=True)
NB = len(b)
N = 10  # series kept through s^N

# Undetermined higher orders, functions of (x, t).
U = {k: sp.Function(k)(x, t) for k in ('f4', 'Om4', 'Pi4', 'Psi3', 'Psi4')}


def dtau(e):
    """tau-derivative: b_k -> b_{k+1}, plus explicit t-dependence of the unknowns."""
    out = sp.diff(e, t)
    for k in range(NB - 1):
        out += sp.diff(e, b[k]) * b[k + 1]
    if e.has(b[NB - 1]):
        raise RuntimeError('derivative order exceeds the symbol list')
    return out


def ser(terms):
    return {p: sp.sympify(c) for p, c in terms.items()}


def add(*ss):
    out = {}
    for a in ss:
        for p, c in a.items():
            out[p] = out.get(p, 0) + c
    return out


def scal(a, c, shift=0):
    return {p + shift: c * v for p, v in a.items() if p + shift <= N}


def mul(a, c):
    out = {}
    for p, u in a.items():
        for q, v in c.items():
            if p + q <= N:
                out[p + q] = out.get(p + q, 0) + u * v
    return out


def smap(a, fn):
    return {p: fn(v) for p, v in a.items()}


def residuals(F):
    one = {0: sp.Integer(1)}
    f = ser({0: 1, 2: F['f1'], 4: F['f2'], 6: F['f3'], 8: U['f4']})
    Om = ser({2: F['Om1'], 4: F['Om2'], 6: F['Om3'], 8: U['Om4']})
    Pi = ser({1: F['Pi1'], 3: F['Pi2'], 5: F['Pi3'], 7: U['Pi4']})
    Psi = ser({3: F['Psi1'], 5: F['Psi2'], 7: U['Psi3'], 9: U['Psi4']})
    dx = lambda a: smap(a, lambda v: sp.diff(v, x))
    xdx = lambda a: smap(a, lambda v: x * sp.diff(v, x))
    dt = lambda a: smap(a, dtau)
    Pi2_ = mul(Pi, Pi)
    Psi2_ = mul(Psi, Psi)
    one_m2e = add(one, {2: sp.Integer(-2)})
    # (S36) times eps.
    E1 = add(scal(add(xdx(Om), scal(mul(Om, Om), x**2), scal(mul(Om, add(scal(Pi2_, x**2),
                                                                          scal(Psi2_, x**4))), -1)), 1, 2),
             scal(mul(one_m2e, Pi2_), -1), Om, scal(mul(one_m2e, Psi2_), -x**2))
    # (S37) as d_x f = x f Omega.
    E2 = add(dx(f), scal(mul(f, Om), -x))
    # (S38) times eps.
    E3 = add(scal(add(dt(Pi), xdx(Pi), scal(mul(f, xdx(Psi)), -1), scal(mul(f, mul(Om, Psi)), -x**2), Pi),
                  1, 2),
             scal(mul(f, Psi), -1))
    # (S39).
    E4 = add(dt(Psi), xdx(Psi), scal(mul(f, smap(dx(Pi), lambda v: v / x)), -1),
             scal(mul(f, mul(Om, Pi)), -1), scal(Psi, 2))
    return dict(S36=E1, S37=E2, S38=E3, S39=E4)


def has_unknown(e):
    return any(e.has(u) for u in U.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mutate', default='', choices=['', 'printed'])
    a = ap.parse_args()
    P = printed()
    C = corrections()
    dPi3 = 0 if a.mutate == 'printed' else C['dPi3']
    df3 = 0 if a.mutate == 'printed' else C['df3']
    F = dict(P)
    F['Pi3'] = P['Pi3'] + dPi3
    F['f3'] = P['f3'] + df3
    F['Om3'] = 2 * b[0] * F['Pi3'] + P['Om3_rest']
    ok = True
    print('mutate =', a.mutate or 'none')

    # Boundary and centre conditions.
    for name, e in [('f2(x=1)', F['f2'].subs(x, 1)), ('f3(x=1)', F['f3'].subs(x, 1)),
                    ('df3(x=1)', C['df3'].subs(x, 1)), ('Pi2(x=0)', F['Pi2'].subs(x, 0)),
                    ('Pi3(x=0)', F['Pi3'].subs(x, 0)), ('dPi3(x=0)', C['dPi3'].subs(x, 0))]:
        z = sp.expand(e) == 0
        ok &= z
        print(f'{name:12s} = 0 : {z}')

    res = residuals(F)
    for eq, E in res.items():
        for p in sorted(E):
            c = sp.expand(E[p])
            if has_unknown(c):
                print(f'{eq} s^{p}: contains an undetermined order, not checked')
                continue
            z = c == 0
            ok &= z
            print(f'{eq} s^{p}: {"0" if z else "NONZERO, " + str(len(sp.Add.make_args(c))) + " terms"}')

    # Rederivation. Pi3 from (S39) at s^5 with Pi3(x=0) = 0.
    G = dict(F)
    q = sp.Function('q')(x)
    G['Pi3'] = q
    c5 = sp.expand(residuals(G)['S39'][5])
    sol = sp.solve(c5, sp.diff(q, x))
    assert len(sol) == 1
    dq = sp.expand(sol[0].subs(q, 0))
    assert not dq.has(q)
    Pi3_d = sp.integrate(dq, (x, 0, x))
    z = sp.expand(Pi3_d - (P['Pi3'] + C['dPi3'])) == 0
    ok &= z if a.mutate == '' else True
    print('Pi3 from (S39) == printed + dPi3 :', z)
    print('Pi3 from (S39) == printed        :', sp.expand(Pi3_d - P['Pi3']) == 0)
    # Omega3 from (S36) at s^6.
    G = dict(F)
    w = sp.Symbol('w')
    G['Pi3'] = Pi3_d
    G['Om3'] = w
    c6 = sp.expand(residuals(G)['S36'][6])
    Om3_d = sp.solve(c6, w)[0]
    z = sp.expand(Om3_d - (2 * b[0] * Pi3_d + P['Om3_rest'])) == 0
    ok &= z
    print('Om3 from (S36) == 2 b0 Pi3 + (S45) rest :', z)
    # f3 from (S37) at s^6 with f(x=1) = 1.
    G['Om3'] = Om3_d
    g = sp.Function('g')(x)
    G['f3'] = g
    c6 = sp.expand(residuals(G)['S37'][6])
    dg = sp.solve(c6, sp.diff(g, x))[0]
    assert not dg.has(g)
    f3_d = sp.integrate(sp.expand(dg), (x, 1, x))
    z = sp.expand(f3_d - (P['f3'] + C['df3'])) == 0
    ok &= z if a.mutate == '' else True
    print('f3 from (S37) == printed + df3 :', z)
    print('f3 from (S37) == printed       :', sp.expand(f3_d - P['f3']) == 0)
    # Psi2 from (S38) at eps^(3/2), which is s^5 after the factor eps.
    G = dict(F)
    G['Psi2'] = w
    Psi2_d = sp.solve(sp.expand(residuals(G)['S38'][5]), w)[0]
    z = sp.expand(Psi2_d - P['Psi2']) == 0
    ok &= z
    print('Psi2 from (S38) == (S43) :', z)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
