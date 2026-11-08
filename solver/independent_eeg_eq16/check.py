#!/usr/bin/env python3
"""Independent check of a write-up on Eq. (16) of arXiv:2611.07007v1 (issue #165).

Eq. (16): bt(th) = A Re[e^{i th}/(1 + q e^{i(2 th + d)})], w = q e^{i d}. Sources: the paper (main text and its
Supplemental Material, (S52)-(S62)) and one shared input, the D = infinity harmonics file
solver/eeg_profile/data/tuned_profile_A2.72.txt, of which only the columns nu, |F_nu/F_1|, arg(F_nu/F_1), the
truncation estimate, and the header values A, <F^2>, <bt^2> are read. No other code of the repository is used.

Sections. I: forms A-E and the ellipse (sympy on rational functions of z = e^{i th}, and mpmath at random points).
P: poles, ratios and the time asymmetry of delta. N: the numbers recomputed from the data file.

Usage: check.py [--data PATH] [--mutate NAME] [--json OUT]. Exit 0 when every check has its expected verdict.
Mutations (each must flip at least one check): see MUTATIONS.
"""
import argparse
import json
import math
import random
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import least_squares, minimize

MUTATIONS = {
    'A_conj': 'form A with conj(w) in place of w',
    'ellipse_axes': 'ellipse semi-axes 1 +- q^2',
    'ellipse_ode': 'h = e^{-i th} + w e^{2 i th}, which breaks h\'\' + h = 0',
    'B_sign': 'form B with 1 + q^2 - 2q cos(2 th + d) in the denominator',
    'C_sign': 'form C with q^k in place of (-q)^k',
    'D_delta': 'form D with the weight cos(2 th - d)',
    'E_noi': 'form E without the factor -i',
    'E_sqrt': 'form E with sqrt(conj w) in place of sqrt(w)',
    'pole_sign': 'poles tested at (pi + d)/2 + k pi',
    'ratio_plus': 'harmonic ratio tested against +w',
    'delta0': 'asymmetry test run with d = 0',
    'treversal': 'time reversal tested as d -> d',
    'data_kink': 'data |F_13| multiplied by 1.05',
    'data_A': 'data A read as 2.70',
    'data_frame': 'data harmonics used in their own frame, no quarter-period map to the paper frame',
}

MP_DPS = 50
TOL = mp.mpf(10) ** (-40)  # identity tolerance, relative, at 50 digits

# Paper's printed numbers (arXiv:2611.07007v1): Eq. (7), (9), (16), (S61), (S62).
PAPER = dict(mu0=0.130818, mu1=-0.1449, D0=1.6439, D1=10.37, q=0.0551, delta=0.0985, Astar=3.8455)
# Rounding half-widths of the printed digits.
PAPER_HW = dict(mu0=5e-7, mu1=5e-5, D1=5e-3, q=5e-5, delta=5e-5)

DEFAULT_DATA = 'solver/eeg_profile/data/tuned_profile_A2.72.txt'


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail, expect=True, section=''):
        ok = bool(ok)
        self.rows.append(dict(name=name, ok=ok, expect=expect, good=(ok == expect), detail=detail,
                              section=section))
        tag = 'PASS' if ok else 'FAIL'
        if ok != expect:
            tag += ' (UNEXPECTED)'
        elif not expect:
            tag += ' (expected: documented discrepancy)'
        print(f'  [{tag}] {name}: {detail}')

    def all_good(self):
        return all(r['good'] for r in self.rows)


# ------------------------------------------------------------------------------------------------
# I. Forms A-E: mpmath definitions. th may be complex; for complex th the analytic continuation of
# Re[g] is (g(th) + conj(g(conj th)))/2, which equals Re[g] on the real axis.
# ------------------------------------------------------------------------------------------------

def eq16_real(th, q, d, A):
    return A * mp.re(mp.exp(1j * th) / (1 + q * mp.exp(1j * (2 * th + d))))


def eq16_c(th, q, d, A):
    w, wb = q * mp.exp(1j * d), q * mp.exp(-1j * d)
    return A / 2 * (mp.exp(1j * th) / (1 + w * mp.exp(2j * th)) + mp.exp(-1j * th) / (1 + wb * mp.exp(-2j * th)))


def formA_c(th, q, d, A, mut):
    w = q * mp.exp(1j * d)
    if mut == 'A_conj':
        w = mp.conj(w)
    wb = mp.conj(w)
    return A / 2 * (1 / (mp.exp(-1j * th) + w * mp.exp(1j * th)) + 1 / (mp.exp(1j * th) + wb * mp.exp(-1j * th)))


def formA_real(th, q, d, A, mut):
    w = q * mp.exp(1j * d)
    if mut == 'A_conj':
        w = mp.conj(w)
    return A * mp.re(1 / (mp.exp(-1j * th) + w * mp.exp(1j * th)))


def formB(th, q, d, A, mut):
    s = -1 if mut == 'B_sign' else 1
    return A * (mp.cos(th) + q * mp.cos(th + d)) / (1 + q ** 2 + s * 2 * q * mp.cos(2 * th + d))


def formC(th, q, d, A, mut, nterms):
    base = q if mut == 'C_sign' else -q
    return A * mp.fsum(base ** k * mp.cos((2 * k + 1) * th + k * d) for k in range(nterms))


def weightD(th, q, d, mut):
    dd = -d if mut == 'D_delta' else d
    return 1 + q ** 2 + 2 * q * mp.cos(2 * th + dd)


def formE_c(th, q, d, A, mut):
    # A Re[-(i/sqrt w) arctan(sqrt w e^{i th})], continued; principal branches, |sqrt w e^{i th}| < 1.
    w = q * mp.exp(1j * d)
    s = mp.sqrt(mp.conj(w)) if mut == 'E_sqrt' else mp.sqrt(w)
    fac = 1 if mut == 'E_noi' else -1j
    g = lambda t: fac / s * mp.atan(s * mp.exp(1j * t))
    return A / 2 * (g(th) + mp.conj(g(mp.conj(th))))


def rel(a, b):
    return abs(a - b) / (1 + abs(b))


def section_identities(C, mut, rng):
    print('I. forms A-E and the ellipse')
    mp.mp.dps = MP_DPS
    pts = []
    for _ in range(24):
        q = mp.mpf(rng.uniform(0.01, 0.6))
        d = mp.mpf(rng.uniform(-math.pi, math.pi))
        A = mp.mpf(rng.uniform(0.5, 5))
        strip = mp.log(1 / q) / 2
        th_r = mp.mpf(rng.uniform(-math.pi, math.pi))
        th_c = mp.mpc(rng.uniform(-math.pi, math.pi), float(strip) * rng.uniform(-0.9, 0.9))
        pts.append((q, d, A, th_r, th_c))
    # include the paper's values
    pts.append((mp.mpf(PAPER['q']), mp.mpf(PAPER['delta']), mp.mpf(PAPER['Astar']), mp.mpf('0.7'),
                mp.mpc('0.3', '0.9')))

    def worst(f):
        return max(f(*p) for p in pts)

    # Form A
    eA = worst(lambda q, d, A, tr, tc: max(rel(formA_real(tr, q, d, A, mut), eq16_real(tr, q, d, A)),
                                           rel(formA_c(tc, q, d, A, mut), eq16_c(tc, q, d, A))))
    C.add('I.A form A = Eq.(16) (mpmath)', eA < TOL, f'max rel diff {mp.nstr(eA, 3)} (tol 1e-40)', section='I')

    # Ellipse: e^{-i d/2} h = (1+q) cos(phi) - i (1-q) sin(phi), phi = th + d/2.
    ax_p, ax_m = ((lambda q: 1 + q ** 2), (lambda q: 1 - q ** 2)) if mut == 'ellipse_axes' else \
        ((lambda q: 1 + q), (lambda q: 1 - q))

    def h(t, q, d):
        w = q * mp.exp(1j * d)
        if mut == 'ellipse_ode':
            return mp.exp(-1j * t) + w * mp.exp(2j * t)
        return mp.exp(-1j * t) + w * mp.exp(1j * t)

    e_el = 0
    e_ode = 0
    e_ax = 0
    for q, d, A, tr, tc in pts:
        for t in [tr, tr + 1, tr + 2.5]:
            u = mp.exp(-0.5j * d) * h(t, q, d)
            e_el = max(e_el, abs((mp.re(u) / ax_p(q)) ** 2 + (mp.im(u) / ax_m(q)) ** 2 - 1))
            hpp = mp.diff(lambda s: h(s, q, d), t, 2)
            e_ode = max(e_ode, abs(hpp + h(t, q, d)))
        # extreme moduli over a fine grid of one period, refined
        gr = [mp.mpf(2) * mp.pi * j / 400 for j in range(400)]
        mods = [abs(h(t, q, d)) for t in gr]
        jmax, jmin = int(np.argmax([float(x) for x in mods])), int(np.argmin([float(x) for x in mods]))
        fmax = mp.findroot(lambda s: mp.diff(lambda r: abs(h(r, q, d)) ** 2, s), gr[jmax])
        fmin = mp.findroot(lambda s: mp.diff(lambda r: abs(h(r, q, d)) ** 2, s), gr[jmin])
        e_ax = max(e_ax, abs(abs(h(fmax, q, d)) - ax_p(q)), abs(abs(h(fmin, q, d)) - ax_m(q)))
    C.add('I.ell h on the ellipse x^2/(1+q)^2 + y^2/(1-q)^2 = 1 (rotated by d/2)', e_el < TOL,
          f'max |residual| {mp.nstr(e_el, 3)}', section='I')
    C.add('I.ell semi-axes = max, min |h| = 1 + q, 1 - q', e_ax < mp.mpf(10) ** -30,
          f'max deviation {mp.nstr(e_ax, 3)}', section='I')
    C.add("I.ell h'' + h = 0", e_ode < mp.mpf(10) ** -30, f"max |h''+h| {mp.nstr(e_ode, 3)}", section='I')

    # Form B
    eB = worst(lambda q, d, A, tr, tc: max(rel(formB(tr, q, d, A, mut), eq16_real(tr, q, d, A)),
                                           rel(formB(tc, q, d, A, mut), eq16_c(tc, q, d, A))))
    C.add('I.B form B = Eq.(16) (mpmath)', eB < TOL, f'max rel diff {mp.nstr(eB, 3)}', section='I')

    # Form C, truncated where the tail is below 1e-45 (|q e^{2|Im th|}| < q^{0.1})
    def eC(q, d, A, tr, tc):
        rate = q ** mp.mpf('0.1')
        n = int(mp.ceil(mp.log(mp.mpf(10) ** -45) / mp.log(rate))) + 5
        return max(rel(formC(tr, q, d, A, mut, n), eq16_real(tr, q, d, A)),
                   rel(formC(tc, q, d, A, mut, n), eq16_c(tc, q, d, A)))
    eCm = worst(eC)
    C.add('I.C form C = Eq.(16) (mpmath, |Im th| < 0.9 ln(1/q)/2)', eCm < TOL, f'max rel diff {mp.nstr(eCm, 3)}',
          section='I')

    # Form D
    def eD(q, d, A, tr, tc):
        out = 0
        for t in (tr, tc):
            P = lambda s: weightD(s, q, d, mut) * eq16_c(s, q, d, A)
            out = max(out, abs(mp.diff(P, t, 2) + P(t)) / A)
        return out
    eDm = worst(eD)
    C.add("I.D (d^2/dth^2 + 1)[(1 + q^2 + 2q cos(2th+d)) bt] = 0 (mpmath)", eDm < mp.mpf(10) ** -30,
          f'max |residual|/A {mp.nstr(eDm, 3)}', section='I')

    # Form E: derivative and a definite integral
    def eE(q, d, A, tr, tc):
        out = 0
        for t in (tr, tc):
            out = max(out, rel(mp.diff(lambda s: formE_c(s, q, d, A, mut), t), eq16_c(t, q, d, A)))
        integ = mp.quad(lambda s: eq16_real(s, q, d, A), [0, tr])
        out = max(out, rel(mp.re(formE_c(tr, q, d, A, mut) - formE_c(mp.mpf(0), q, d, A, mut)), integ))
        return out
    eEm = worst(eE)
    C.add('I.E d/dth of form E = Eq.(16), and its definite integral (mpmath)', eEm < mp.mpf(10) ** -30,
          f'max rel diff {mp.nstr(eEm, 3)}', section='I')

    # Symbolic route: rational functions of z = e^{i th}, u = e^{i d}; d/dth = i z d/dz.
    z, u = sp.symbols('z u', nonzero=True)
    q, A = sp.symbols('q A', positive=True)
    k, N = sp.symbols('k N', integer=True, nonnegative=True)
    cs = lambda e: (e + 1 / e) / 2
    E16 = A / 2 * (z / (1 + q * u * z ** 2) + (1 / z) / (1 + q / (u * z ** 2)))
    wA = q / u if mut == 'A_conj' else q * u
    FA = A / 2 * (1 / (1 / z + wA * z) + 1 / (z + (q ** 2 / wA) / z))
    sB = -1 if mut == 'B_sign' else 1
    FB = A * (cs(z) + q * cs(z * u)) / (1 + q ** 2 + sB * 2 * q * cs(z ** 2 * u))
    sym_A = sp.cancel(sp.together(FA - E16)) == 0
    sym_B = sp.cancel(sp.together(FB - E16)) == 0
    # C: the term T_k = (A/2)(x1^k z + x2^k / z), x1 = (-q) u z^2, x2 = (-q)/(u z^2), is A (-q)^k cos((2k+1)th + k d)
    # on the real axis. With P1 = x1^{N+1}, P2 = x2^{N+1}, the closed form S_N is proved by induction
    # (S_0 = T_0, S_{N+1} - S_N = T_{N+1}), and E16 = S_N + R_N exactly, where R_N -> 0 since |x1|, |x2| < 1.
    base = q if mut == 'C_sign' else -q
    x1, x2 = base * u * z ** 2, base / (u * z ** 2)
    P1, P2 = sp.symbols('P1 P2')
    S = lambda p1, p2: A / 2 * (z * (1 - p1) / (1 - x1) + (1 / z) * (1 - p2) / (1 - x2))
    T = lambda p1, p2: A / 2 * (p1 * z + p2 / z)
    c_base = sp.cancel(sp.together(S(x1, x2) - T(1, 1))) == 0
    c_step = sp.cancel(sp.together(S(P1 * x1, P2 * x2) - S(P1, P2) - T(P1, P2))) == 0
    R = A / 2 * (z * P1 / (1 + q * u * z ** 2) + (1 / z) * P2 / (1 + q / (u * z ** 2)))  # the tail k > N
    c_rem = sp.cancel(sp.together(E16 - S(P1, P2) - R)) == 0
    sym_C = c_base and c_step and c_rem
    sD = -1 if mut == 'D_delta' else 1
    W = 1 + q ** 2 + 2 * q * cs(z ** 2 * u ** sD)
    P = W * E16
    dth = lambda e: sp.I * z * sp.diff(e, z)
    sym_D = sp.cancel(sp.together(dth(dth(P)) + P)) == 0
    s, sb = sp.symbols('s sb', nonzero=True)  # s^2 = q u, sb^2 = q/u
    fac = 1 if mut == 'E_noi' else -sp.I
    s1, s2 = (sb, s) if mut == 'E_sqrt' else (s, sb)
    G = A / 2 * (fac / s1 * sp.atan(s1 * z) + sp.conjugate(fac) / s2 * sp.atan(s2 / z))
    dG = sp.cancel(sp.together(dth(G).subs({s: sp.sqrt(q * u), sb: sp.sqrt(q / u)})))
    sym_E = sp.simplify(sp.cancel(sp.together(dG - E16))) == 0
    C.add('I.sym forms A, B, C (with exact remainder), D, E by sympy on rational functions of z',
          sym_A and sym_B and sym_C and sym_D and sym_E,
          f'A {sym_A}, B {sym_B}, C {sym_C}, D {sym_D}, E {sym_E}', section='I')


# ------------------------------------------------------------------------------------------------
# P. Poles, the ratio -w and the time asymmetry
# ------------------------------------------------------------------------------------------------

def fourier_odd(f, nmax, npts=512):
    # complex amplitudes c_nu with f = Re sum c_nu e^{i nu th}, by the trapezoid rule
    ths = [2 * mp.pi * j / npts for j in range(npts)]
    vals = [f(t) for t in ths]
    return {nu: 2 * mp.fsum(v * mp.exp(-1j * nu * t) for v, t in zip(vals, ths)) / npts
            for nu in range(1, nmax + 1, 2)}


def section_poles(C, mut, rng):
    print('P. poles, ratios, time asymmetry')
    mp.mp.dps = MP_DPS
    q, d, A = mp.mpf(PAPER['q']), mp.mpf(PAPER['delta']), mp.mpf(PAPER['Astar'])
    ok_loc, ok_simple, worst_den = True, True, 0
    for qq, dd in [(q, d), (mp.mpf('0.3'), mp.mpf('-1.1')), (mp.mpf('0.05'), mp.mpf('2.5'))]:
        for kk in (-1, 0, 1, 2):
            for sgn in (+1, -1):
                re0 = (mp.pi + dd) / 2 if mut == 'pole_sign' else (mp.pi - dd) / 2
                th0 = re0 + kk * mp.pi + sgn * 0.5j * mp.log(1 / qq)
                den = (1 + qq * mp.exp(1j * (2 * th0 + dd))) * (1 + qq * mp.exp(-1j * (2 * th0 + dd)))
                worst_den = max(worst_den, abs(den))
                r1 = mp.mpf(10) ** -12 * eq16_c(th0 + mp.mpf(10) ** -12, qq, dd, A)
                r2 = mp.mpf(10) ** -24 * eq16_c(th0 + mp.mpf(10) ** -24, qq, dd, A)
                if abs(den) > TOL:
                    ok_loc = False
                if not (abs(r1) > 1e-6 and abs(r1 - r2) / abs(r2) < 1e-9):
                    ok_simple = False
    C.add('P.pole Eq.(16) is singular at (pi - d)/2 + k pi +- (i/2) ln(1/q)', ok_loc,
          f'max |denominator| {mp.nstr(worst_den, 3)}', section='P')
    C.add('P.pole these are simple poles with nonzero residue', ok_simple and ok_loc,
          'eps*f(th0+eps) constant to 1e-9 for eps = 1e-12, 1e-24', section='P')
    # Only singularities: in z = e^{i th}, Eq.(16) = (A/2) z [u z^2 (1 + q u) + q + u] / ((1 + q u z^2)(u z^2 + q))
    z, u = sp.symbols('z u', nonzero=True)
    qs, As = sp.symbols('q A', positive=True)
    E16 = As / 2 * (z / (1 + qs * u * z ** 2) + (1 / z) / (1 + qs / (u * z ** 2)))
    num, den = sp.fraction(sp.cancel(sp.together(E16)))
    den_f = sp.factor(den)
    roots = sp.solve(sp.Eq(den, 0), z)
    num_at = [sp.simplify(num.subs(z, r)) for r in roots]
    only = (sp.degree(den, z) == 4 and len(set(roots)) == 4 and all(sp.simplify(v) != 0 for v in num_at)
            and sp.simplify(den.subs(z, 0)) != 0)
    C.add('P.pole the only singularities (rational in z, 4 simple roots z^2 = -1/w, -conj(w), z = 0 regular)',
          only, f'denominator {den_f}', section='P')

    # Harmonic ratios of Eq.(16) are -w for every pair (complex amplitudes, paper frame c_1 > 0)
    w = q * mp.exp(1j * d)
    target = w if mut == 'ratio_plus' else -w
    cf = fourier_odd(lambda t: eq16_real(t, q, d, A), 21)
    e_rat = max(abs(cf[nu + 2] / cf[nu] - target) / q for nu in range(1, 19, 2))
    e_c1 = abs(cf[1] - A) / A
    C.add('P.ratio Eq.(16): c_{nu+2}/c_nu = -w for nu = 1..17, c_1 = A*', e_rat < 1e-35 and e_c1 < 1e-35,
          f'max |ratio + w|/q {mp.nstr(e_rat, 3)}, |c_1/A* - 1| {mp.nstr(e_c1, 3)}', section='P')
    # Converse: c_1 = A*, c_{nu+2} = -w c_nu fixes c_nu = A* (-w)^k, whose sum is Eq.(16) (form C, checked above).
    fC = lambda t: mp.re(mp.fsum(A * (-w) ** kk * mp.exp(1j * (2 * kk + 1) * t) for kk in range(60)))
    e_conv = max(abs(fC(t) - eq16_real(t, q, d, A)) for t in [mp.mpf(x) / 7 for x in range(-20, 21)])
    C.add('P.ratio converse: the series with c_1 = A*, ratios -w sums to Eq.(16)', e_conv < TOL,
          f'max diff {mp.nstr(e_conv, 3)}', section='P')

    # Time reversal: bt(-th; d) = bt(th; -d)
    e_tr = 0
    for _ in range(10):
        t = mp.mpf(rng.uniform(-3, 3))
        dd = d if mut == 'treversal' else -d
        e_tr = max(e_tr, abs(eq16_real(-t, q, d, A) - eq16_real(t, q, dd, A)))
    C.add('P.delta time reversal th -> -th maps d -> -d', e_tr < TOL, f'max diff {mp.nstr(e_tr, 3)}',
          section='P')

    # No time shift removes d: shift-invariant I = arg(c_3 conj(c_1)^3) = pi + d (mod 2 pi); a profile even or odd
    # about some time has I = 0 mod pi. Brute force over shifts as a second route.
    dA = mp.mpf(0) if mut == 'delta0' else d
    f = lambda t: eq16_real(t, q, dA, A)
    cfa = fourier_odd(f, 5, 256)
    inv = mp.arg(cfa[3] * mp.conj(cfa[1]) ** 3)
    inv_ok = abs(mp.sin(inv)) > 1e-3
    tgrid = [mp.pi * j / 64 for j in range(1, 64)]

    def asym(t0, sgn):
        return mp.sqrt(mp.fsum((f(t0 + t) - sgn * f(t0 - t)) ** 2 for t in tgrid) / len(tgrid)) / A

    best = []
    for sgn in (+1, -1):
        vals = [(asym(mp.pi * j / 200, sgn), mp.pi * j / 200) for j in range(200)]
        v0, t0 = min(vals)
        res = minimize(lambda x: float(asym(mp.mpf(x[0]), sgn)), [float(t0)], method='Nelder-Mead',
                       options=dict(xatol=1e-12, fatol=1e-16))
        best.append(res.fun)
    sym_dist = min(best)
    C.add('P.delta no time shift makes Eq.(16) even or odd (d = 0.0985)', inv_ok and sym_dist > 1e-4,
          f'arg(c3 conj(c1)^3) = {mp.nstr(inv, 6)} (pi + d = {mp.nstr(mp.pi + dA, 6)}); '
          f'min over shifts of rms asymmetry/A = {sym_dist:.3e}', section='P')


# ------------------------------------------------------------------------------------------------
# N. Numbers from the shared data file
# ------------------------------------------------------------------------------------------------

ROW = re.compile(r'^\s*(\d+)\s+([0-9.]+e[+-]\d+)\s+([+-]?[0-9.]+)\s+(\d+)\s+([0-9.]+e[+-]\d+)')


def load(path, mut=''):
    """Read the first block of the file: header values and the columns nu, |F/F1|, arg, nterms, last/|F|."""
    blocks = []
    cur = None
    for line in open(path):
        m = re.search(r'A = ([0-9.]+)\s*$', line)
        if line.startswith('==') and m:
            cur = dict(A=float(m.group(1)), rows=[], in_table=False)
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r'<F\^2> = ([0-9.]+)\s+<bt\^2> = N0 = ([0-9.]+)', line.strip())
        if m:
            cur['F2'], cur['bt2'] = float(m.group(1)), float(m.group(2))
            continue
        if line.strip().startswith('nu  |F_nu/F_1|'):
            cur['in_table'] = True
            continue
        if cur['in_table']:
            m = ROW.match(line)
            if m:
                cur['rows'].append((int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)),
                                    float(m.group(5))))
            else:
                cur['in_table'] = False
    for b in blocks:
        b.pop('in_table')
        b['F'] = {nu: mod * np.exp(1j * ar) for nu, mod, ar, _, _ in b['rows']}
        b['err'] = {nu: e for nu, _, _, _, e in b['rows']}
        if mut == 'data_kink':
            b['F'][13] *= 1.05
        if mut == 'data_A':
            b['A'] = 2.70
    return blocks


def paper_frame(F, mut):
    # The paper frame has c_0 > 0, s_0 = 0 (S53) and phi_n ~ n(pi + d) (S57). The data's ratios are ~ +w, so the
    # map tested here is a quarter-period shift, F_nu -> (-1)^k F_nu, nu = 2k+1.
    if mut == 'data_frame':
        return dict(F)
    return {nu: ((-1) ** ((nu - 1) // 2)) * v for nu, v in F.items()}


def s62_fit(c, N0):
    """Paper's (S61)-(S62): A*(q) = sqrt(2 N0 (1 - q^2)); fit (q, d) to complex c_n = c_{n,0} - i s_{n,0}, n = 0..3."""
    def resid(x):
        q, d = x
        As = math.sqrt(2 * N0 * (1 - q * q))
        out = []
        for n in range(4):
            m = As * (-q) ** n * np.exp(1j * n * d)
            out += [(c[2 * n + 1] - m).real, (c[2 * n + 1] - m).imag]
        return out
    r = least_squares(resid, [0.05, 0.1], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    q, d = r.x
    return q, d, math.sqrt(2 * N0 * (1 - q * q)), math.sqrt(sum(v * v for v in r.fun))


def eq16_devs(Fabs, As, w, nus):
    # c_nu of Eq.(16) in the paper frame: As (-w)^k
    return {nu: Fabs[nu] / (As * (-w) ** ((nu - 1) // 2)) for nu in nus}


def onezero_devs(Fabs, As, w, b, nus):
    # A* Re[e^{i th}(1 + b e^{2 i th})/(1 + w e^{2 i th})]: c_1 = A*, c_{2k+1} = A* (b - w)(-w)^{k-1}
    out = {}
    for nu in nus:
        kk = (nu - 1) // 2
        m = As if kk == 0 else As * (b - w) * (-w) ** (kk - 1)
        out[nu] = Fabs[nu] / m
    return out


def pade(a, L, M):
    """[L/M] Pade of sum a_k z^k: returns numerator p (L+1) and denominator qd (M+1, qd_0 = 1)."""
    a = np.asarray(a, dtype=complex)
    if M == 0:
        return a[:L + 1].copy(), np.array([1.0 + 0j])
    Amat = np.array([[a[L + i - j] if L + i - j >= 0 else 0 for j in range(1, M + 1)] for i in range(1, M + 1)])
    rhs = -np.array([a[L + i] for i in range(1, M + 1)])
    qd = np.concatenate([[1.0], np.linalg.solve(Amat, rhs)])
    p = np.array([sum(qd[j] * a[i - j] for j in range(0, min(i, M) + 1)) for i in range(L + 1)])
    return p, qd


def series_of(p, qd, n):
    # Taylor coefficients of p/qd to order n
    out = np.zeros(n + 1, dtype=complex)
    for i in range(n + 1):
        s = p[i] if i < len(p) else 0
        s -= sum(qd[j] * out[i - j] for j in range(1, min(i, len(qd) - 1) + 1))
        out[i] = s
    return out


def drift_fits(F, lo=5, hi=15):
    """Drift of the ratio moduli. D_nu = |r_{nu+2}| - |r_nu|, r_nu = F_{nu+2}/F_nu, nu = lo..hi (uses F to hi + 4).

    A second pole gives D_nu ~ lam^nu, log D linear in nu. A branch point at the pole, F_nu ~ rho^nu (1 + a nu^-s),
    gives D_nu ~ nu^-(s+2), log D linear in log nu. Each straight-line fit's rms in log D is returned. Moduli only,
    since the file's phases have 5 decimals (5e-6 rad), the size of the late drift.
    """
    nus = [nu for nu in range(lo, hi + 1, 2) if nu + 4 in F]
    m = {nu: abs(F[nu + 2] / F[nu]) for nu in range(lo, hi + 3, 2) if nu + 2 in F}
    D = np.array([m[nu + 2] - m[nu] for nu in nus])
    x = np.array(nus, float) + 1  # midpoint of the two ratios
    out = dict(nus=nus, D=D)
    if np.any(np.abs(D) < 1e-9) or len(set(np.sign(D))) > 1:
        out.update(cls='none', rms_pow=float('nan'), rms_exp=float('nan'), p1=float('nan'), lam=float('nan'))
        return out
    y = np.log(np.abs(D))
    Pp = np.polyfit(np.log(x), y, 1)
    Pe = np.polyfit(x, y, 1)
    rp = math.sqrt(np.mean((np.polyval(Pp, np.log(x)) - y) ** 2))
    re_ = math.sqrt(np.mean((np.polyval(Pe, x) - y) ** 2))
    cls = 'pow' if 3 * rp < re_ else ('exp' if 3 * re_ < rp else 'undecided')
    out.update(cls=cls, rms_pow=rp, rms_exp=re_, p1=-Pp[0], lam=math.exp(Pe[0]))
    return out


def ratio_limit(F, lo=5, hi=17, P=3):
    """r_inf and the 1/nu content: r_nu = r_inf + d1/nu + ... + dP/nu^P (linear least squares, nu = lo..hi)."""
    nus = [nu for nu in range(lo, hi + 1, 2) if nu + 2 in F]
    nu = np.array(nus, float)
    r = np.array([F[v + 2] / F[v] for v in nus])
    X = np.stack([nu ** (-j) for j in range(P + 1)], axis=1).astype(complex)
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    gamma = -(coef[1] / coef[0]).real / 2  # F_nu ~ nu^-gamma e^{-sigma nu} gives d1/r_inf = -2 gamma
    return coef[0], gamma


def pade_table(F, kmax, Ms=(1, 2, 3)):
    a = np.array([F[2 * kk + 1] / F[1] for kk in range(kmax + 1)])
    rows = {}
    for M in Ms:
        for n in range(M, kmax):
            L = n - M
            try:
                p, qd = pade(a, L, M)
            except np.linalg.LinAlgError:
                continue
            s = series_of(p, qd, kmax)
            errs = [abs(s[kk] / a[kk] - 1) for kk in range(n + 1, kmax + 1)]
            rows[(L, M)] = max(errs) if errs else float('nan')
    return rows


def analyze(blk, mut=''):
    """All derived numbers for one block. Shared by the data and the planted files (controls.py)."""
    F = blk['F']
    A = blk['A']
    nus = sorted(F)
    out = {}
    S = sum(abs(F[nu]) ** 2 for nu in nus)
    out['F2_from_A'] = A * A / 2 * S
    out['N0_from_A'] = A * A * S
    Fp = paper_frame(F, mut)
    Abs = {nu: math.sqrt(2) * A * Fp[nu] for nu in nus}  # bt harmonics: bt = sqrt2 F, A = |F_1|
    out['r1'] = F[3] / F[1]
    out['r3'] = F[5] / F[3]
    N0 = out['N0_from_A']
    out['s62_repoN0'] = s62_fit(Abs, N0)
    out['s62_paperN0'] = s62_fit(Abs, paper_N0())
    nus19 = [nu for nu in nus if nu <= 19]
    wp = PAPER['q'] * np.exp(1j * PAPER['delta'])
    out['dev16_paper'] = eq16_devs(Abs, PAPER['Astar'], wp, nus19)
    rho1 = Fp[3] / Fp[1]  # paper-frame ratio, = -w for Eq.(16)
    out['dev16_w1'] = eq16_devs(Abs, math.sqrt(2) * A, -rho1, nus19)
    # One-zero form, write-up's parameters w = r_3, b = r_3 - r_1 (data ratios)
    w, b = out['r3'], out['r3'] - out['r1']
    out['onezero_sqrt2A'] = onezero_devs(Abs, math.sqrt(2) * A, w, b, nus19)
    out['onezero_paperA'] = onezero_devs(Abs, PAPER['Astar'], w, b, nus19)
    out['ratios'] = {nu: F[nu + 2] / F[nu] for nu in nus if nu + 2 in F}
    out['drift'] = drift_fits(F)
    out['rinf'], out['gamma'] = ratio_limit(F)
    out['pade'] = pade_table(F, 9)
    return out


def paper_N0():
    return (PAPER['D1'] - 4 * math.pi * PAPER['mu1']) / (4 * math.pi * PAPER['mu0'])


def paper_Astar_band():
    # (S61) on the printed inputs, and the range over their rounding half-widths
    vals = []
    for sD in (-1, 1):
        for sm1 in (-1, 1):
            for sm0 in (-1, 1):
                for sq in (-1, 1):
                    D1 = PAPER['D1'] + sD * PAPER_HW['D1']
                    mu1 = PAPER['mu1'] + sm1 * PAPER_HW['mu1']
                    mu0 = PAPER['mu0'] + sm0 * PAPER_HW['mu0']
                    q = PAPER['q'] + sq * PAPER_HW['q']
                    N0 = (D1 - 4 * math.pi * mu1) / (4 * math.pi * mu0)
                    vals.append(math.sqrt(2 * N0 * (1 - q * q)))
    N0 = paper_N0()
    return math.sqrt(2 * N0 * (1 - PAPER['q'] ** 2)), min(vals), max(vals)


def fmt_c(z):
    return f'{abs(z):.6f} e^({np.angle(z):+.5f} i)'


def section_numbers(C, path, mut):
    print(f'N. numbers from {path}')
    blocks = load(path, mut)
    blk = blocks[0]
    out = analyze(blk, mut)
    A = blk['A']
    r1 = out['r1']
    print(f'    blocks read: {len(blocks)}, rows in block 1: {len(blk["rows"])}, A = {A}')
    # N1: F3/F1 as printed in the write-up, 0.05509 e^{0.0988 i}
    m4 = float(f'{abs(r1):.4g}')
    C.add('N.r1 |F3/F1| = 0.05509 at 4 significant digits (write-up)', m4 == 0.05509,
          f'|F3/F1| = {abs(r1):.8f} -> {m4}', expect=False, section='N')
    C.add('N.r1 arg(F3/F1) = 0.0988 at 4 decimals (write-up)', round(float(np.angle(r1)), 4) == 0.0988,
          f'arg = {np.angle(r1):.6f}', section='N')
    C.add('N.r1 |F3/F1| within the paper q = 0.0551 +- 5e-5', abs(abs(r1) - PAPER['q']) <= PAPER_HW['q'],
          f'|F3/F1| - q = {abs(r1) - PAPER["q"]:+.2e}', section='N')
    # The paper's frame: Eq.(16) has F3/F1 = -w. Read in the paper frame without a map, F3/F1 is +w.
    C.add('N.frame data F3/F1 equals -w (the paper frame, S57: phi_1 ~ pi + d) without a frame map',
          abs(r1 + PAPER['q'] * np.exp(1j * PAPER['delta'])) < 1e-3,
          f'F3/F1 + w = {abs(r1 + PAPER["q"] * np.exp(1j * PAPER["delta"])):.3e}; F3/F1 - w = '
          f'{abs(r1 - PAPER["q"] * np.exp(1j * PAPER["delta"])):.3e}', expect=False, section='N')
    # N2: A is |F_1|, bt = sqrt2 F (header <bt^2> = 2 <F^2>)
    eF2 = abs(out['F2_from_A'] / blk['F2'] - 1)
    C.add('N.A A = |F_1|: (A^2/2) sum |F_nu/F_1|^2 = <F^2> of the header', eF2 < 2e-6,
          f'{out["F2_from_A"]:.7f} vs {blk["F2"]} (rel {eF2:.1e})', section='N')
    C.add('N.A header <bt^2> = 2 <F^2>, so bt = sqrt2 F and the bt fundamental is sqrt2 A',
          abs(blk['bt2'] / (2 * blk['F2']) - 1) < 2e-7, f'{blk["bt2"]} / (2 x {blk["F2"]})', section='N')
    Astar_p, lo, hi = paper_Astar_band()
    s2A = math.sqrt(2) * A
    C.add('N.A sqrt2 A within 1e-3 of the paper A* = 3.8455', abs(s2A / PAPER['Astar'] - 1) < 1e-3,
          f'sqrt2 A = {s2A:.5f}, ratio - 1 = {s2A / PAPER["Astar"] - 1:+.2e}; (S61) on the printed inputs gives '
          f'{Astar_p:.5f}, range over their rounding [{lo:.5f}, {hi:.5f}]', section='N')
    C.add('N.A sqrt2 A inside the rounding range of the paper (S61) inputs', lo <= s2A <= hi,
          f'{s2A:.5f} vs [{lo:.5f}, {hi:.5f}]', expect=False, section='N')
    C.add('N.N0 data <bt^2> vs paper N0 = (D1 - 4 pi mu1)/(4 pi mu0)', abs(out['N0_from_A'] / paper_N0() - 1) < 1e-3,
          f'{out["N0_from_A"]:.6f} vs {paper_N0():.6f} (rel {out["N0_from_A"] / paper_N0() - 1:+.2e})', section='N')
    # N3: the paper's own fit (S62) applied to the data
    for key, lab in [('s62_repoN0', 'N0 = data <bt^2>'), ('s62_paperN0', 'N0 = paper')]:
        q, d, As, res = out[key]
        print(f'    (S62) fit, {lab}: q = {q:.6f}, d = {d:.6f}, A* = {As:.5f}, residual {res:.2e}')
    q, d, As, res = out['s62_repoN0']
    C.add('N.S62 the paper fit (S61)-(S62) on the data gives q, d within 2e-4 of 0.0551, 0.0985',
          abs(q - PAPER['q']) < 2e-4 and abs(d - PAPER['delta']) < 2e-4,
          f'q = {q:.6f}, d = {d:.6f}', section='N')
    # N4: Eq.(16) deviations
    print('    Eq.(16) deviation |C/C_16| - 1, arg(C/C_16):')
    print('     nu   paper (q, d, A*)        w = F3/F1, A* = sqrt2 A')
    for nu in sorted(out['dev16_paper']):
        a, b = out['dev16_paper'][nu], out['dev16_w1'][nu]
        print(f'     {nu:2d}   {abs(a) - 1:+.4e} {np.angle(a):+.5f}   {abs(b) - 1:+.4e} {np.angle(b):+.5f}')
    dv = {nu: abs(out['dev16_paper'][nu]) - 1 for nu in (5, 7, 13, 19)}
    C.add('N.dev16 Eq.(16) deviation grows with nu: 5, 7, 13, 19', dv[5] < dv[7] < dv[13] < dv[19] and dv[19] > 0.3,
          ', '.join(f'nu {nu}: {v:+.4f}' for nu, v in dv.items()), section='N')
    # N5: one-zero form within 3% up to nu = 19
    oz = out['onezero_sqrt2A']
    mx = max(abs(abs(v) - 1) for v in oz.values())
    mxp = max(abs(abs(v) - 1) for v in out['onezero_paperA'].values())
    print('    one-zero form |F_nu|/|C_nu| - 1 (A* = sqrt2 A):',
          ', '.join(f'{nu}: {abs(v) - 1:+.2e}' for nu, v in sorted(oz.items()) if nu >= 7))
    C.add('N.onezero within 3% of every |F_nu|, nu <= 19 (A* = sqrt2 A)', mx < 0.03, f'max {mx:.4f}', section='N')
    C.add('N.onezero within 3% of every |F_nu|, nu <= 19 (A* = 3.8455)', mxp < 0.03, f'max {mxp:.4f}', section='N')
    # N6: singularity of the profile and the drift of the ratios
    dr = out['drift']
    rinf, gamma = out['rinf'], out['gamma']
    print(f"    ratio-modulus drift D_nu, nu = {dr['nus']}: " + ', '.join(f'{v:.2e}' for v in dr['D']))
    print(f"    straight-line fits of log D: vs log nu rms {dr['rms_pow']:.3f} (slope p+1 = {dr['p1']:.3f}), "
          f"vs nu rms {dr['rms_exp']:.3f} (lam = {dr['lam']:.4f} per unit nu); class {dr['cls']}")
    print(f'    r_inf = {fmt_c(rinf)} (1/nu^3 fit, nu = 5..17); 1/nu content gamma = {gamma:+.4f} (0 for a pole)')
    sig_inf = 0.5 * math.log(1 / abs(rinf))
    re_inf = (math.pi - np.angle(rinf)) / 2
    print(f'    profile singularity (paper frame): th = {re_inf:.4f} +- {sig_inf:.4f} i; '
          f'model (paper q, d): {(math.pi - PAPER["delta"]) / 2:.4f} +- {0.5 * math.log(1 / PAPER["q"]):.4f} i')
    C.add('N.sing the leading singularity is a simple pole (|gamma| < 0.02)', abs(gamma) < 0.02,
          f'gamma = {gamma:+.4f}', section='N')
    C.add('N.sing the ratio drift is a power law (branch point at the pole), not one exponential (second pole)',
          dr['cls'] == 'pow', f"class {dr['cls']}, p+1 = {dr['p1']:.2f}", section='N')
    out['class'] = dr['cls']
    print('    Pade [L/M] of sum F_{2k+1}/F_1 z^k, k <= 9: max rel error on the unused coefficients')
    for (L, M), e in sorted(out['pade'].items(), key=lambda t: (t[0][1], t[0][0])):
        print(f'      [{L}/{M}] {e:.2e}')
    # J spread between the two blocks
    if len(blocks) > 1:
        sp_ = max(abs(blocks[0]['F'][nu] / blocks[1]['F'][nu] - 1) for nu in blocks[0]['F'] if nu <= 19)
        print(f'    spread between the two blocks (J 1250, J 1500), nu <= 19: {sp_:.1e}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--mutate', default='')
    ap.add_argument('--json', default='')
    ap.add_argument('--skip', default='', help='comma list of sections to skip: I,P,N')
    a = ap.parse_args()
    if a.mutate and a.mutate not in MUTATIONS:
        sys.exit(f'unknown mutation {a.mutate}; known: {", ".join(MUTATIONS)}')
    print(f'check.py mutate={a.mutate or "none"}  mpmath {mp.__version__}, sympy {sp.__version__}, '
          f'numpy {np.__version__}')
    rng = random.Random(165)
    C = Checks()
    skip = set(a.skip.split(',')) if a.skip else set()
    if 'I' not in skip:
        section_identities(C, a.mutate, rng)
    if 'P' not in skip:
        section_poles(C, a.mutate, rng)
    if 'N' not in skip:
        section_numbers(C, a.data, a.mutate)
    good = C.all_good()
    print(f'verdict: {"all checks as expected" if good else "UNEXPECTED verdicts"} '
          f'({sum(r["ok"] for r in C.rows)} pass, {sum(not r["ok"] for r in C.rows)} fail, '
          f'{sum(not r["good"] for r in C.rows)} unexpected)')
    if a.json:
        json.dump(dict(mutate=a.mutate, checks=C.rows, good=good), open(a.json, 'w'), indent=1)
    sys.exit(0 if good else 1)


if __name__ == '__main__':
    main()
