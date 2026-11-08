"""Size of the claimed NNLO correction (#176) with Eq. (16) of arXiv:2611.07007v1 as beta~.

beta~(theta) = A Re[e^{i theta} / (1 + q e^{i(2 theta + delta)})], theta = tau / (2 mu0), eps = 1/(D - 1).
Printed minus corrected is -eps^3 df3 in f, -2 eps^3 beta~ dPi3 in Omega, -eps^(5/2) dPi3 in Pi.
Measure: rms over one period and 0 <= x <= 1 of that change over the rms of the corrected NNLO's
1 - f, Omega and Pi, and max_tau |change| / max_tau |1 - f| at x = 0.

    python3 effect.py --out <dir>/effect.json     # exit 0 when every check passes
    python3 effect.py --mutate=pi_eps             # control, Pi's change with eps^(3/2), must fail scaling
    python3 effect.py --mutate=pi_offset          # control, dPi3 + const, must fail the x = 0 check
    python3 effect.py --mutate=xrange             # control, x quadrature on [0, 0.9], must fail planted
"""
import argparse
import json
import sys

import numpy as np
import sympy as sp

from fields import b, x as X, printed, corrections

Q, DEL, A, MU0 = 0.0551, 0.0985, 3.8455, 0.130818
CLAIM = {25: (22, 15, 9.6, 20), 50: (4.6, 3.2, 2.2, 4.1), 200: (0.22, 0.17, 0.12, 0.20)}
KEYS = ('f', 'Om', 'Pi', 'f0')
NMODE = 40


def beta_derivs(theta, kmax=4):
    """[beta~, d/dtau beta~, ...] from the geometric series of Eq. (16) in odd harmonics."""
    n = np.arange(NMODE)
    a = (-Q) ** n * np.exp(1j * n * DEL)
    w = 2 * n + 1
    ph = np.exp(1j * np.outer(theta, w))
    return [A * np.real(ph @ (a * (1j * w / (2 * MU0)) ** k)) for k in range(kmax + 1)]


def beta_check():
    """Fourier derivatives against sympy derivatives of the closed form at a few theta."""
    th = sp.Symbol('th', real=True)
    e = A * sp.re(sp.exp(sp.I * th) / (1 + Q * sp.exp(sp.I * (2 * th + DEL))))
    pts = np.array([0.0, 0.7, 2.1, 4.4])
    B = beta_derivs(pts)
    err = 0.0
    for k in range(5):
        g = sp.lambdify(th, sp.diff(e, th, k) / (2 * MU0) ** k, 'numpy')
        ref = np.array([float(g(p)) for p in pts])
        err = max(err, np.max(np.abs(B[k] - ref)) / np.max(np.abs(ref)))
    return err


def build(plant=None):
    P = printed()
    C = corrections()
    if plant is not None:
        # Planted dPi3 = c x^2, with the df3 it induces through (S37), c b0 (x^4 - 1)/2.
        C = dict(dPi3=plant * X**2, df3=plant * b[0] * (X**4 - 1) / 2, d0=-plant * b[0] / 2)
    ex = dict(f1=P['f1'], f2=P['f2'], f3p=P['f3'], df3=C['df3'], Om2=P['Om2'], Om3r=P['Om3_rest'],
              Pi2=P['Pi2'], Pi3p=P['Pi3'], dPi3=C['dPi3'])
    args = (X,) + tuple(b[:5])
    return {k: sp.lambdify(args, v, 'numpy') for k, v in ex.items()}


def fields_on(eps, fn, th, xq, mut):
    """Corrected base (1 - f, Omega, Pi), its alternatives, and the change, on the grid th x xq."""
    bb = beta_derivs(th)
    XX = xq[None, :]
    BB = [v[:, None] for v in bb]
    ev = lambda k: np.broadcast_to(fn[k](XX, *BB), (len(th), len(xq)))
    b0 = BB[0]
    dPi3 = ev('dPi3') + (1.0 if mut == 'pi_offset' else 0.0)
    df3 = ev('df3')
    Pi3c = ev('Pi3p') + dPi3
    f3c = ev('f3p') + df3
    Om3c = 2 * b0 * Pi3c + ev('Om3r')
    om1 = np.broadcast_to(b0**2, dPi3.shape)
    pi_pow = 1.5 if mut == 'pi_eps' else 2.5
    base = dict(f=-(eps * ev('f1') + eps**2 * ev('f2') + eps**3 * f3c),
                Om=eps * om1 + eps**2 * ev('Om2') + eps**3 * Om3c,
                Pi=np.sqrt(eps) * np.broadcast_to(b0, dPi3.shape) + eps**1.5 * ev('Pi2') + eps**2.5 * Pi3c)
    printed_ = dict(f=base['f'] + eps**3 * df3, Om=base['Om'] - 2 * eps**3 * b0 * dPi3,
                    Pi=base['Pi'] - eps**2.5 * dPi3)
    lo = dict(f=-eps * ev('f1'), Om=eps * om1, Pi=np.sqrt(eps) * np.broadcast_to(b0, dPi3.shape))
    nlo = dict(f=-(eps * ev('f1') + eps**2 * ev('f2')), Om=eps * om1 + eps**2 * ev('Om2'),
               Pi=np.sqrt(eps) * np.broadcast_to(b0, dPi3.shape) + eps**1.5 * ev('Pi2'))
    chg = dict(f=-eps**3 * df3, Om=-2 * eps**3 * b0 * dPi3, Pi=-eps**pi_pow * dPi3)
    # The correction applied with the opposite sign, printed - delta. chg is in f, the base in 1 - f.
    flip = dict(f=printed_['f'] - chg['f'], Om=printed_['Om'] + chg['Om'], Pi=printed_['Pi'] + chg['Pi'])
    return base, printed_, lo, nlo, flip, chg, pi_pow


def evaluate(D, fn, nth, xq, wq, mut):
    eps = 1.0 / (D - 1)
    th = 2 * np.pi * np.arange(nth) / nth
    base, printed_, lo, nlo, flip, chg, pi_pow = fields_on(eps, fn, th, xq, mut)
    # Mean over theta (uniform, one period) and x in [0, 1] (Gauss-Legendre weights wq, sum 1).
    rms = lambda g: np.sqrt(np.mean(g**2 @ wq) / np.sum(wq))
    # Maxima on a uniform grid with both x endpoints.
    thm = 2 * np.pi * np.arange(4096) / 4096
    bm, _, _, _, _, cm, _ = fields_on(eps, fn, thm, np.linspace(0, 1, 201), mut)
    out = {}
    for k in ('f', 'Om', 'Pi'):
        out[k] = dict(rel=rms(chg[k]) / rms(base[k]), rel_printed=rms(chg[k]) / rms(printed_[k]),
                      rel_lo=rms(chg[k]) / rms(lo[k]), rel_nlo=rms(chg[k]) / rms(nlo[k]),
                      rel_flip=rms(chg[k]) / rms(flip[k]),
                      maxmax=np.max(np.abs(cm[k])) / np.max(np.abs(bm[k])),
                      abs_rms=rms(chg[k]))
    # x = 0 on a fine theta grid, separately from the 2D quadrature.
    th0 = 2 * np.pi * np.arange(8192) / 8192
    b0s = beta_derivs(th0)
    z = np.zeros_like(th0)
    at0 = lambda k: np.broadcast_to(fn[k](z, *b0s), th0.shape)
    dPi3_0 = at0('dPi3') + (1.0 if mut == 'pi_offset' else 0.0)
    d0 = at0('df3')
    one_m_f0 = -(eps * at0('f1') + eps**2 * at0('f2') + eps**3 * (at0('f3p') + d0))
    one_m_f0p = one_m_f0 + eps**3 * d0
    c0 = eps**3 * np.abs(d0)
    out['f0'] = dict(rel=np.max(c0) / np.max(np.abs(one_m_f0)),
                     rel_printed=np.max(c0) / np.max(np.abs(one_m_f0p)),
                     rms_rms=np.sqrt(np.mean(c0**2)) / np.sqrt(np.mean(one_m_f0**2)),
                     max_pointwise=np.max(c0 / np.abs(one_m_f0)))
    out['x0'] = dict(Om=float(np.max(np.abs(2 * eps**3 * b0s[0] * dPi3_0))),
                     Pi=float(np.max(np.abs(eps**pi_pow * dPi3_0))),
                     Om_scale=float(np.max(np.abs(chg['Om']))), Pi_scale=float(np.max(np.abs(chg['Pi']))))
    return out


def half_unit(c):
    """Half a unit in the last printed digit of a claimed value (two significant figures)."""
    s = f'{c:g}'
    dec = len(s.split('.')[1]) if '.' in s else 0
    if c == 0.2:
        dec = 2  # printed as 0.20
    return 0.5 * 10.0**-dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mutate', default='', choices=['', 'pi_eps', 'pi_offset', 'xrange'])
    ap.add_argument('--check', default='all', choices=['all', 'beta', 'table', 'x0', 'scaling', 'planted'],
                    help='which check sets the exit code')
    ap.add_argument('--nth', type=int, default=1024)
    ap.add_argument('--nx', type=int, default=24)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    mut = a.mutate
    xg, wg = np.polynomial.legendre.leggauss(a.nx)
    xhi = 0.9 if mut == 'xrange' else 1.0
    xq = (xg + 1) * xhi / 2
    # Weights of the mean over [0, 1]. With xrange they sum to 1 over [0, 0.9], a wrong measure.
    wq = wg / 2
    verdict = {}
    rep = dict(params=dict(q=Q, delta=DEL, A=A, mu0=MU0, nth=a.nth, nx=a.nx, nmode=NMODE, mutate=mut))
    print('mutate =', mut or 'none')

    e = beta_check()
    rep['beta_deriv_relerr'] = e
    verdict['beta'] = e < 1e-10
    print(f'Fourier vs sympy derivatives of Eq. (16), max rel err {e:.2e}')

    fn = build()
    res = {D: evaluate(D, fn, a.nth, xq, wq, mut) for D in (25, 50, 100, 200, 400, 800, 1600)}
    rep['results'] = {str(D): r for D, r in res.items()}

    print('\nTable, rms(change)/rms(corrected NNLO), and max|change|/max|1-f| at x = 0, in %.')
    print('d is recomputed/claimed - 1. r is whether the recomputed value rounds to the claimed digits.')
    worst = 0.0
    rep['table'] = {}
    for D in (25, 50, 200):
        for i, k in enumerate(KEYS):
            v = 100 * res[D][k]['rel']
            c = CLAIM[D][i]
            d = v / c - 1
            rnd = abs(v - c) <= half_unit(c)
            worst = max(worst, abs(d))
            rep['table'][f'{k}_D{D}'] = dict(claimed=c, recomputed=v, reldiff=d, rounds_to_claim=bool(rnd))
            print(f'D {D:>4} {k:>3}: recomputed {v:8.4g}  claimed {c:<5g}  d {100 * d:+6.2f}%  '
                  f'{"within 2%" if abs(d) <= 0.02 else "OUTSIDE 2%":10s}  r {rnd}')
    rep['table_worst_reldiff'] = worst
    verdict['table'] = worst <= 0.02
    print(f'worst |d| {100 * worst:.2f}% (pass <= 2%)')

    print('\nAlternative readings, in %:')
    for D in (25, 50, 200):
        r = res[D]
        print(f'D {D}: over printed NNLO  f {100 * r["f"]["rel_printed"]:.4g}  Om {100 * r["Om"]["rel_printed"]:.4g}'
              f'  Pi {100 * r["Pi"]["rel_printed"]:.4g}  f0 {100 * r["f0"]["rel_printed"]:.4g}')
        for lab, key in (('over LO', 'rel_lo'), ('over NLO', 'rel_nlo'), ('over printed-delta', 'rel_flip')):
            print(f'       {lab:20s}f {100 * r["f"][key]:.4g}  Om {100 * r["Om"][key]:.4g}'
                  f'  Pi {100 * r["Pi"][key]:.4g}')
        print(f'       max/max           f {100 * r["f"]["maxmax"]:.4g}  Om {100 * r["Om"]["maxmax"]:.4g}'
              f'  Pi {100 * r["Pi"]["maxmax"]:.4g}')
        print(f'       x = 0, rms/rms {100 * r["f0"]["rms_rms"]:.4g}, max of pointwise ratio '
              f'{100 * r["f0"]["max_pointwise"]:.4g}')
    rep['alt_xweight'] = {}
    for m in (1, 2):
        wm = wq * (m + 1) * xq**m
        for D in (25, 50, 200):
            r = evaluate(D, fn, a.nth, xq, wm, mut)
            rep['alt_xweight'][f'm{m}_D{D}'] = {k: r[k]['rel'] for k in ('f', 'Om', 'Pi')}
            print(f'D {D}: x weight {m + 1} x^{m}  f {100 * r["f"]["rel"]:.4g}  Om {100 * r["Om"]["rel"]:.4g}'
                  f'  Pi {100 * r["Pi"]["rel"]:.4g}')

    print('\nx = 0, max|change| over max|change| anywhere:')
    x0_ok = True
    for D in (25, 200, 1600):
        z = res[D]['x0']
        ro, rp = z['Om'] / z['Om_scale'], z['Pi'] / z['Pi_scale']
        x0_ok &= ro < 1e-14 and rp < 1e-14
        print(f'D {D}: Om {ro:.2e}  Pi {rp:.2e}')
    verdict['x0'] = bool(x0_ok)
    print('pass < 1e-14')

    target = (1599 / 399) ** 2
    print(f'\nScaling, rel(D 400)/rel(D 1600), target ((1/399)/(1/1599))^2 = {target:.4f}:')
    sc_ok = True
    rep['scaling'] = {}
    for k in KEYS:
        r = res[400][k]['rel'] / res[1600][k]['rel']
        g = abs(r / target - 1) <= 0.10
        sc_ok &= g
        rep['scaling'][k] = r
        print(f'{k:>3}: {r:.4f}  ({100 * (r / target - 1):+.2f}%)  {"pass" if g else "FAIL"}')
    verdict['scaling'] = bool(sc_ok)

    c = 0.37
    fp = build(plant=c)
    rb = np.sqrt(A**2 / (2 * (1 - Q**2)))  # rms of Eq. (16) over a period, sum of |mode|^2 / 2
    print(f'\nPlanted dPi3 = {c} x^2 and its df3, predicted rms of the change against measured:')
    pl_ok = True
    rep['planted'] = {}
    for D in (25, 400):
        eps = 1 / (D - 1)
        r = evaluate(D, fp, a.nth, xq, wq, mut)
        pred = dict(Pi=eps**2.5 * c / np.sqrt(5), Om=2 * eps**3 * c * rb / np.sqrt(5),
                    f=eps**3 * c * rb * np.sqrt(32 / 45) / 2)
        for k in ('f', 'Om', 'Pi'):
            m = r[k]['abs_rms']
            d = abs(m / pred[k] - 1)
            pl_ok &= d < 1e-10
            rep['planted'][f'{k}_D{D}'] = dict(pred=pred[k], meas=m)
            print(f'D {D} {k:>2}: predicted {pred[k]:.10e}  measured {m:.10e}  rel diff {d:.1e}')
    verdict['planted'] = bool(pl_ok)
    print('pass rel diff < 1e-10')

    print('\nVerdicts:')
    for k, v in verdict.items():
        print(f'  {k:8s} {"pass" if v else "FAIL"}')
    rep['verdict'] = verdict
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(rep, fh, indent=1, default=float)
    ok = all(verdict.values()) if a.check == 'all' else verdict[a.check]
    print(f'exit code from check "{a.check}":', 0 if ok else 1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

