#!/usr/bin/env python3
"""One order past the NNLO of C. Ecker, F. Ecker, D. Grumiller, arXiv:2611.07007v1, supplement "Simplified large-D
perturbation theory": Psi_3 from (S38) at eps^(5/2), Pi_4 from (S39) at eps^(7/2), Omega_4 from (S36) at eps^3 and
f_4 from (S37) at eps^4, in the paper's ansatz (S35) and conventions (S29) f(tau,1) = 1 and (S34)
Pi(tau,0) = sqrt(eps) beta-tilde exactly.

The NLO and NNLO below these terms are check.py's 'claimed' set: (S40)-(S43) as printed (beta read as beta-tilde), and
the corrected Pi_3, Omega_3, f_3, which check.py derives again and finds equal (issue agent-comms#6). (S36)-(S39) are
typed again here from the paper, and the residuals are formed by truncated series arithmetic in s = sqrt(eps). They are
compared with check.py's fully expanded residuals on the NNLO set (--crosscheck).

Usage (from the repository root):
  python3 solver/independent_eeg_nnlo/psi3.py                 # closed forms: residuals, derivation, uniqueness
  python3 solver/independent_eeg_nnlo/psi3.py --numeric       # eps-scaling of all four residuals, mpmath
  python3 solver/independent_eeg_nnlo/psi3.py --crosscheck    # this file's residuals == check.py's on the NNLO set
  python3 solver/independent_eeg_nnlo/psi3.py --print         # the derived terms, as Python and LaTeX
  ... --mutate=NAME                                           # perturb one coefficient of a closed form
  ... --json=PATH                                             # write the results as JSON

Exit 0 iff every check of the mode passes. Default mode: every residual coefficient up to the last order the new terms
fix vanishes, the next order is nonzero in every equation, the derivation equals the closed forms, Pi_4(tau,0) = 0 and
f_4(tau,1) = 0, and each residual is affine in its unknown with a nonzero factor.
"""
import argparse
import json
import os
import sys
import time

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check as C  # noqa: E402  (the paper's NLO and the confirmed NNLO, issue agent-comms#6)

x, s, b, R = C.x, C.s, C.b, C.R
T0 = time.time()

MUTATIONS = {
    'Psi3': "Psi3: x^4 coefficient of b0^2 b2 -> +1; enters (S38) at eps^(5/2)",
    'Psi3c': "Psi3: x^0 coefficient of b0^3 b1 -> +1; enters (S38) at eps^(5/2)",
    'Pi4': "Pi4: x^6 coefficient of b6 -> +1/48; enters (S39) at eps^(7/2)",
    'Pi4b': "Pi4: x^2 coefficient of b0^5 -> +1; enters (S39) at eps^(7/2)",
    'Omega4': "Omega4: x^2 coefficient of b0^4 -> +1; enters (S36) at eps^3",
    'f4': "f4: x^0 coefficient of b0^8 -> +1; enters (S37) at eps^4",
}
MUT = set()


def bump(name, k=1):
    """k if --mutate=name is active, else 0. Added to one coefficient of a closed form."""
    return k if name in MUT else 0


# --------------------------------------------------------------------------------------------------------------
# Truncated series in s = sqrt(eps): dict {power of s: coefficient}.

def sclean(a):
    out = {}
    for k, v in a.items():
        v = sp.expand(v)
        if v != 0:
            out[k] = v
    return out


def sadd(*terms):
    """sum of (coefficient, series) pairs"""
    out = {}
    for c, a in terms:
        for k, v in a.items():
            out[k] = out.get(k, 0) + c * v
    return sclean(out)


def smul(a, bb, K):
    """product truncated at s^K"""
    out = {}
    for i, ai in a.items():
        for j, bj in bb.items():
            if i + j <= K:
                out[i + j] = out.get(i + j, 0) + ai * bj
    return sclean(out)


def sshift(a, k):
    return {i + k: v for i, v in a.items()}


def smap(fun, a):
    return sclean({k: fun(v) for k, v in a.items()})


def dx(a):
    return smap(lambda v: sp.diff(v, x), a)


def xdx(a):
    return smap(lambda v: x * sp.diff(v, x), a)


def dt(a):
    return smap(C.Dt, a)


def fields(t):
    """(S35): f = 1 + sum eps^k f_k, Omega = sum eps^k Omega_k, Pi = sum eps^(k-1/2) Pi_k,
    Psi = sum eps^(k+1/2) Psi_k, with f_1, Omega_1, Pi_1, Psi_1 the leading terms of (S35)."""
    f = {0: sp.Integer(1)}
    f.update({2 * k: t['f%d' % k] for k in range(1, 5) if 'f%d' % k in t})
    Om = {2 * k: t['O%d' % k] for k in range(1, 5) if 'O%d' % k in t}
    Pi = {2 * k - 1: t['P%d' % k] for k in range(1, 5) if 'P%d' % k in t}
    Psi = {2 * k + 1: t['S%d' % k] for k in range(1, 4) if 'S%d' % k in t}
    return [sclean(q) for q in (f, Om, Pi, Psi)]


# Last power of s each equation's residual is formed to. It is exact there: every product is truncated only above it.
KTOP = {'S36': 8, 'S37': 10, 'S38': 7, 'S39': 9}
# Last power of s the given terms (through Psi_3, Pi_4, Omega_4, f_4) fix. KTOP is the next order, which needs a term
# not given (Omega_5 and Pi_5, f_5, Psi_4, Pi_5).
FIXED = {'S36': 6, 'S37': 8, 'S38': 5, 'S39': 7}


def residuals(t):
    """LHS - RHS of (S36)-(S39), typed from the paper's HTML rendering (2026-11-10), as truncated series in s:
      (S36) x d_x Omega + x^2 Omega^2 = (1/eps)((1 - 2 eps) Pi^2 - Omega) + Omega (Pi^2 x^2 + x^4 Psi^2)
                                        + (1/eps - 2) x^2 Psi^2
      (S37) (1/x) d_x ln f = Omega
      (S38) (d_tau + x d_x) Pi = f x d_x Psi + f (1/eps + x^2 Omega) Psi - Pi
      (S39) (d_tau + x d_x) Psi = f (1/x) d_x Pi + f Omega Pi - 2 Psi"""
    f, Om, Pi, Psi = fields(t)
    K = KTOP['S36']
    Pi2 = smul(Pi, Pi, K + 2)
    Psi2 = smul(Psi, Psi, K + 2)
    r36 = sadd((1, xdx(Om)), (x**2, smul(Om, Om, K)),
               (-1, sshift(sadd((1, Pi2), (-2, sshift(Pi2, 2)), (-1, Om)), -2)),
               (-1, smul(Om, sadd((x**2, Pi2), (x**4, Psi2)), K)),
               (-x**2, sshift(Psi2, -2)), (2 * x**2, Psi2))
    r36 = {k: v for k, v in r36.items() if k <= K}
    # (S37): ln f = sum_{j<=5} (-1)^(j+1) u^j / j, u = f - 1 = O(s^2); exact through s^10.
    K = KTOP['S37']
    u = {k: v for k, v in f.items() if k > 0}
    L, up = {}, {0: sp.Integer(1)}
    for j in range(1, 6):
        up = smul(up, u, K)
        L = sadd((1, L), (sp.Rational((-1)**(j + 1), j), up))
    r37 = sadd((1, smap(lambda v: sp.diff(v, x) / x, L)), (-1, Om))
    r37 = {k: v for k, v in r37.items() if k <= K}
    K = KTOP['S38']
    fPsi = smul(f, Psi, K + 2)
    r38 = sadd((1, dt(Pi)), (1, xdx(Pi)), (-1, smul(f, xdx(Psi), K)), (-1, sshift(fPsi, -2)),
               (-x**2, smul(Om, fPsi, K)), (1, Pi))
    r38 = {k: v for k, v in r38.items() if k <= K}
    K = KTOP['S39']
    r39 = sadd((1, dt(Psi)), (1, xdx(Psi)), (-1, smul(f, smap(lambda v: sp.diff(v, x) / x, Pi), K)),
               (-1, smul(f, smul(Om, Pi, K), K)), (2, Psi))
    r39 = {k: v for k, v in r39.items() if k <= K}
    return {'S36': r36, 'S37': r37, 'S38': r38, 'S39': r39}


def order_label(k):
    return 'eps^%s' % sp.Rational(k, 2)


# --------------------------------------------------------------------------------------------------------------
# The closed forms (b_k = k-th tau-derivative of beta-tilde). Filled from --print, then checked against the
# derivation in the default mode.

def closed():
    b0, b1, b2, b3, b4, b5, b6 = b[:7]
    raise NotImplementedError


def nnlo():
    """check.py's 'claimed' set: LO (S35), NLO (S40)-(S43), corrected NNLO; Psi_3 = 0 there."""
    t = C.build_set('claimed')
    t.pop('S3')
    return t


def full_set():
    t = nnlo()
    t.update(closed())
    return t


# --------------------------------------------------------------------------------------------------------------
# Order-by-order solution, with the residual of this file.

def coef_fun(eq, k):
    return lambda tt: residuals(tt)[eq].get(k, 0)


def derive():
    t = nnlo()
    for key in ('S3', 'P4', 'O4', 'f4'):
        t[key] = 0
    t['S3'] = C.solve_linear(coef_fun('S38', 5), 'S3', t, 'alg')    # (S38) at eps^(5/2)
    t['P4'] = C.solve_linear(coef_fun('S39', 7), 'P4', t, 'pi')     # (S39) at eps^(7/2), Pi_4(0) = 0 (S34)
    t['O4'] = C.solve_linear(coef_fun('S36', 6), 'O4', t, 'alg')    # (S36) at eps^3
    t['f4'] = C.solve_linear(coef_fun('S37', 8), 'f4', t, 'f')      # (S37) at eps^4, f_4(1) = 0 (S29)
    return {k: t[k] for k in ('S3', 'P4', 'O4', 'f4')}


def uniqueness(t):
    """R(U + delta) - R(U) = a L[delta] at the order U fixes, with delta(x) generic and L the operator named."""
    d = sp.Function('delta')(x)
    out = {}
    for key, eq, k, op in (('S3', 'S38', 5, 'id'), ('P4', 'S39', 7, 'xinv_dx'), ('O4', 'S36', 6, 'id'),
                           ('f4', 'S37', 8, 'xinv_dx')):
        t1 = dict(t)
        t1[key] = t[key] + d
        diff = sp.expand(residuals(t1)[eq].get(k, 0) - residuals(t)[eq].get(k, 0))
        L = sp.diff(d, x) / x if op == 'xinv_dx' else d
        a = sp.simplify(diff / L)
        out[key] = dict(eq=eq, order=order_label(k), operator=('(1/x) d/dx' if op == 'xinv_dx' else 'identity'),
                        factor=str(a), linear=bool(a.free_symbols <= set(b) and a != 0))
    return out


def regular(t):
    """Each new term is a polynomial in x (no log, no negative power), with the parity of (S35)."""
    out = {}
    for key in ('S3', 'P4', 'O4', 'f4'):
        e = sp.expand(t[key])
        ok = e.is_polynomial(x) and not e.has(sp.log)
        degs = sorted({mm[0] for mm in sp.Poly(e, x).monoms()}) if ok else []
        out[key] = dict(polynomial=bool(ok), even=bool(ok and all(q % 2 == 0 for q in degs)), x_powers=degs)
    return out


# --------------------------------------------------------------------------------------------------------------
# Numerical eps-scaling of the four residuals, mpmath, with exact ln f and numerical derivatives. Shares the
# closed forms and the equations' reading with the symbolic path, not the series arithmetic.

def numeric(sets, dps=80):
    import mpmath as mp
    mp.mp.dps = dps
    tau = sp.Symbol('tau')
    bt = sp.Rational(7, 10) * sp.sin(tau) + sp.Rational(2, 5) * sp.cos(2 * tau) + sp.Rational(1, 5)
    sub = {b[k]: sp.diff(bt, tau, k) for k in range(C.NB)}
    epsy = sp.Symbol('eps', positive=True)
    out = {}
    for name, t in sets.items():
        f, Om, Pi, Psi = fields(t)
        tot = [sum(v * s**k for k, v in q.items()) for q in (f, Om, Pi, Psi)]
        F = [sp.lambdify((tau, x, epsy), sp.expand(e.subs(sub)).subs(s, sp.sqrt(epsy)), 'mpmath') for e in tot]

        def res(t0, x0, e0):
            fv, Ov, Pv, Sv = (g(t0, x0, e0) for g in F)
            dOx = mp.diff(lambda xx: F[1](t0, xx, e0), x0)
            dlnfx = mp.diff(lambda xx: mp.log(F[0](t0, xx, e0)), x0)
            dPt = mp.diff(lambda tt: F[2](tt, x0, e0), t0)
            dPx = mp.diff(lambda xx: F[2](t0, xx, e0), x0)
            dSt = mp.diff(lambda tt: F[3](tt, x0, e0), t0)
            dSx = mp.diff(lambda xx: F[3](t0, xx, e0), x0)
            r36 = (x0 * dOx + x0**2 * Ov**2 - ((1 / e0) * ((1 - 2 * e0) * Pv**2 - Ov)
                                               + Ov * (Pv**2 * x0**2 + x0**4 * Sv**2) + (1 / e0 - 2) * x0**2 * Sv**2))
            r37 = dlnfx / x0 - Ov
            r38 = dPt + x0 * dPx - (fv * x0 * dSx + fv * (1 / e0 + x0**2 * Ov) * Sv - Pv)
            r39 = dSt + x0 * dSx - (fv * dPx / x0 + fv * Ov * Pv - 2 * Sv)
            return dict(S36=r36, S37=r37, S38=r38, S39=r39)

        t0, x0 = mp.mpf('0.3'), mp.mpf('0.8')
        es = [mp.mpf(10)**(-k) for k in (3, 4, 5, 6)]
        vals = [res(t0, x0, e) for e in es]
        rows = {}
        for eq in ('S36', 'S37', 'S38', 'S39'):
            v = [abs(r[eq]) for r in vals]
            sl = [float(mp.log10(v[i] / v[i + 1])) for i in range(len(v) - 1)]
            rows[eq] = dict(absR=[mp.nstr(q, 6) for q in v], slopes=sl)
            print('  %-6s %s  |R| at eps = 1e-3..1e-6: %s   local exponents: %s'
                  % (name, eq, ', '.join(mp.nstr(q, 4) for q in v), ', '.join('%.3f' % q for q in sl)), flush=True)
        out[name] = rows
    return out


# --------------------------------------------------------------------------------------------------------------

def latex_of(e):
    return sp.latex(sp.collect(sp.expand(e), x, sp.factor))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--numeric', action='store_true')
    ap.add_argument('--crosscheck', action='store_true')
    ap.add_argument('--print', action='store_true', dest='prn')
    ap.add_argument('--mutate', default='', help=', '.join('%s (%s)' % kv for kv in MUTATIONS.items()))
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    for name in filter(None, a.mutate.split(',')):
        if name not in MUTATIONS:
            sys.exit('unknown mutation %s; choose from %s' % (name, ', '.join(MUTATIONS)))
        MUT.add(name)
        print('mutation active: %s: %s' % (name, MUTATIONS[name]))
    report = dict(mutations=sorted(MUT))
    rc = 0
    if a.prn:
        der = derive()
        for k, v in der.items():
            print('\n%s =\n%s' % (k, sp.collect(sp.expand(v), x)))
            print('%s latex: %s' % (k, latex_of(v)))
        report['derived'] = {k: str(v) for k, v in der.items()}
    elif a.crosscheck:
        # This file's truncated residuals equal check.py's fully expanded ones on the NNLO set, at every power of s
        # check.py forms exactly (its KMAX); there both have Psi_3 = Pi_4 = Omega_4 = f_4 = 0.
        t = nnlo()
        mine = residuals(t)
        t0 = dict(t)
        t0['S3'] = 0
        theirs = C.residuals(t0)
        bad = []
        for eq in mine:
            ct = C.coeffs(theirs[eq], C.KMAX[eq])
            for k in range(-4, C.KMAX[eq] + 1):
                if sp.expand(mine[eq].get(k, 0) - ct.get(k, 0)) != 0:
                    bad.append((eq, order_label(k)))
        print('crosscheck against check.py residuals (NNLO set, powers up to check.py KMAX): %s'
              % ('PASS' if not bad else 'FAIL at %s' % bad))
        report['crosscheck_fails'] = bad
        rc = 1 if bad else 0
    elif a.numeric:
        t = full_set()
        t_nnlo = dict(nnlo(), S3=0, P4=0, O4=0, f4=0)
        print('numeric residual scaling at tau = 0.3, x = 0.8, bt = 0.7 sin tau + 0.4 cos 2tau + 0.2')
        out = numeric({'new': t, 'nnlo': t_nnlo})
        # Pass: with the new terms, last local exponent above expected - 0.1; without them, below expected - 0.6.
        expect = {'S36': 4.0, 'S37': 5.0, 'S38': 3.5, 'S39': 4.5}
        bad = []
        for eq, p in expect.items():
            if out['new'][eq]['slopes'][-1] < p - 0.1:
                bad.append('new %s %.3f < %.1f' % (eq, out['new'][eq]['slopes'][-1], p - 0.1))
            if out['nnlo'][eq]['slopes'][-1] > p - 0.6:
                bad.append('nnlo %s %.3f > %.1f' % (eq, out['nnlo'][eq]['slopes'][-1], p - 0.6))
        print('\nverdict numeric: %s' % ('PASS' if not bad else 'FAIL: ' + '; '.join(bad)))
        report['numeric'] = out
        report['numeric_fails'] = bad
        rc = 1 if bad else 0
    else:
        t = full_set()
        res = residuals(t)
        fails, open_zero = [], []
        print('residuals with Psi_3, Pi_4, Omega_4, f_4 (closed forms) [%.0f s]' % (time.time() - T0), flush=True)
        for eq, r in res.items():
            cells = []
            for k in range(min(list(r) + [0]), KTOP[eq] + 1):
                c = r.get(k, 0)
                if k <= FIXED[eq]:
                    st = '0' if c == 0 else 'FAIL(%d)' % len(sp.Add.make_args(c))
                    if c != 0:
                        fails.append((order_label(k), eq))
                else:
                    st = 'open(%d)' % len(sp.Add.make_args(c)) if c != 0 else 'open-ZERO'
                    if c == 0:
                        open_zero.append((order_label(k), eq))
                cells.append('%s:%s' % (order_label(k), st))
            print('  %s  %s' % (eq, '  '.join(cells)))
        print('  key: 0 = coefficient vanishes; FAIL(n) = nonzero at a fixed order; open(n) = nonzero at the next')
        print('       order, which needs a term not given (it must be nonzero).')
        t_der = dict(nnlo())
        der = derive()
        t_der.update(der)
        cl = closed()
        same = {k: sp.expand(der[k] - cl[k]) == 0 for k in der}
        print('\nderived == closed form: %s [%.0f s]' % (same, time.time() - T0), flush=True)
        for k in der:
            if not same[k]:
                print('  closed - derived %s = %s' % (k, sp.factor(sp.expand(cl[k] - der[k]))))
        bc = dict(P4_at_0=sp.expand(cl['P4'].subs(x, 0)), f4_at_1=sp.expand(cl['f4'].subs(x, 1)))
        print('boundary: Pi4(tau,0) = %s   f4(tau,1) = %s' % (bc['P4_at_0'], bc['f4_at_1']))
        reg = regular(t_der)
        print('regularity of the derived terms: %s' % reg)
        uq = uniqueness(t_der)
        print('uniqueness (residual affine in the unknown):')
        for k, v in uq.items():
            print('  %s at (%s) %s: R(U+delta)-R(U) = %s * %s[delta]  linear=%s'
                  % (k, v['eq'], v['order'], v['factor'], v['operator'], v['linear']))
        bad = []
        if fails:
            bad.append('residual nonzero at %s' % fails)
        if open_zero:
            bad.append('next order vanishes at %s' % open_zero)
        if not all(same.values()):
            bad.append('closed form != derived for %s' % [k for k, v in same.items() if not v])
        if bc['P4_at_0'] != 0 or bc['f4_at_1'] != 0:
            bad.append('boundary condition fails')
        if not all(v['polynomial'] and v['even'] for v in reg.values()):
            bad.append('a derived term is not an even polynomial in x')
        if not all(v['linear'] for v in uq.values()):
            bad.append('a residual is not affine in its unknown')
        print('\nverdict: %s' % ('PASS' if not bad else 'FAIL: ' + '; '.join(bad)))
        report.update(fails=fails, open_zero=open_zero, same={k: bool(v) for k, v in same.items()},
                      boundary={k: str(v) for k, v in bc.items()}, regular=reg, uniqueness=uq)
        rc = 1 if bad else 0
    report['exit'] = rc
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(report, fh, indent=1, default=str)
    print('\nexit %d  [%.0f s]' % (rc, time.time() - T0))
    return rc


if __name__ == '__main__':
    sys.exit(main())
