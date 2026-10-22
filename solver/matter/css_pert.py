#!/usr/bin/env python3
"""Linear perturbations of the CSS solution by complex-path shooting (independent of the collocation in css_model.py).

    python3 css_pert.py D KAPPA OMEGA F0 LAM_RE [LAM_IM] [RFRAC]     root of det M(lambda) from the given start
    python3 css_pert.py count D KAPPA OMEGA F0 RE0 RE1 IM1 [K]       zeros of det M in the box RE0 < Re lam < RE1, |Im lam| < IM1

Perturbations ~ e^{lambda tau} (tau = -ln(-t)): psi = f + eps e^{lambda tau} A(x), psibar = fbar + eps e^{lambda tau} B(x),
b = b0 + eps e^{lambda tau} db, u = u0 + eps e^{lambda tau} du (A and B independent: the real system complexified).
The linearized equations are generated with sympy from the (tau, x) form of the equations that css_equations.py checks
(css_model.py docstring, with x d_x = 2 w d_w):
    psi_tau = b P + i om psi - x psi_x,
    P_tau = (i om - 1) P - x P_x + b psi_xx + (n b/x + b_x) psi_x + b (2 kappa psibar/rho)(psi_x^2 - P^2),
    b_x = (n-1) b u / x,   u_x = (1+u)[(2x/n) h (P Pbar + psi_x psibar_x) - (n-1) u/x],
    u_tau + x u_x = (4 x (1+u)/n) b h (P psibar_x + Pbar psi_x)/2,
with the momentum and Hamiltonian constraints combined into an algebraic du, and the gauge db(0) = 0 (alpha = 1 on the
axis).  Regular data at the centre: A = A0 + a2 x^2, B = B0 + b2 x^2, db = beta x^2 (a2, b2, beta fixed by the equations).
Analyticity at the horizon: integrate background and perturbation along the upper and lower half circles
|x - x_h| = R and require equal values of A and B at x_h + R.  M(lambda) maps (A0, B0) to the two jumps; the growth
rates are the zeros of det M, an analytic function of lambda, so the number of modes in a box is a winding number.
Known zeros: lambda = 1 (time translation), 0 (phase), +-i omega (target isometries), all exact.
"""
import sys, os
import numpy as np
import sympy as sp
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_lateral as cl


def build():
    x, lam, om, n, kap, eps = sp.symbols('x lam om n kap eps')
    f, g, q, qg, f2, g2, b0, b1, u0, P0, Pg0, P0x, Pg0x = sp.symbols('f g q qg f2 g2 b0 b1 u0 P0 Pg0 P0x Pg0x')
    A, A1, B, B1, db, du = sp.symbols('A A1 B B1 db du')
    unk = sp.symbols('Pa Pb db1 du1 Pa1 Pb1 A2 B2')
    Pa, Pb, db1, du1, Pa1, Pb1, A2, B2 = unk
    I = sp.I
    psi, psib = f + eps * A, g + eps * B
    psx, psbx = q + eps * A1, qg + eps * B1
    psxx, psbxx = f2 + eps * A2, g2 + eps * B2
    P, Pbar = P0 + eps * Pa, Pg0 + eps * Pb
    Px, Pbx = P0x + eps * Pa1, Pg0x + eps * Pb1
    u, b, bx = u0 + eps * du, b0 + eps * db, b1 + eps * db1
    rho = 1 - kap * psi * psib
    h = 1 / rho ** 2
    lin = lambda e: sp.diff(e, eps).subs(eps, 0)
    eqs = []
    # definition of P and its x-derivative (psi_tau = lam eps A)
    e1 = b * P + I * om * psi - x * psx
    e1b = b * Pbar - I * om * psib - x * psbx
    eqs.append(lin(e1) - lam * A)
    eqs.append(lin(e1b) - lam * B)
    eqs.append(lin(bx * P + b * Px + I * om * psx - psx - x * psxx) - lam * A1)
    eqs.append(lin(bx * Pbar + b * Pbx - I * om * psbx - psbx - x * psbxx) - lam * B1)
    # wave equations
    e2 = (I * om - 1) * P - x * Px + b * psxx + (n * b / x + bx) * psx + b * (2 * kap * psib / rho) * (psx ** 2 - P ** 2)
    e2b = (-I * om - 1) * Pbar - x * Pbx + b * psbxx + (n * b / x + bx) * psbx + b * (2 * kap * psi / rho) * (psbx ** 2 - Pbar ** 2)
    eqs.append(lin(e2) - lam * Pa)
    eqs.append(lin(e2b) - lam * Pb)
    # slicing and Hamiltonian constraint (ODEs in x for db, du)
    eqs.append(lin((n - 1) * b * u / x) - db1)
    ham = (1 + u) * ((2 * x / n) * h * (P * Pbar + psx * psbx) - (n - 1) * u / x)
    eqs.append(lin(ham) - du1)
    Mx, rhs = sp.linear_eq_to_matrix(eqs, list(unk))
    # momentum constraint, as a check:  lam du + x du1 - lin(mom) = 0
    mom = (4 * x * (1 + u) / n) * b * h * (P * psbx + Pbar * psx) / 2
    momres = lam * du + x * du1 - lin(mom)
    args = (x, lam, om, n, kap, f, g, q, qg, f2, g2, b0, b1, u0, P0, Pg0, P0x, Pg0x, A, A1, B, B1, db, du)
    fM = sp.lambdify(args, Mx, 'numpy')
    fr = sp.lambdify(args, rhs, 'numpy')
    fmom = sp.lambdify(args + tuple(unk), momres, 'numpy')
    return fM, fr, fmom


_FM, _FR, _FMOM = None, None, None
NP = 6   # per perturbation: A, A1, B, B1, db, du


def funcs():
    global _FM, _FR, _FMOM
    if _FM is None:
        _FM, _FR, _FMOM = build()
    return _FM, _FR


def background_derived(x, Z, n, kap, om):
    """Background quantities at (complex) x from Z = (f, g, q, qg, b)."""
    f, g, q, qg, b = Z[:5]
    d = cl.rhs_c(x, Z[:5], n, kap, om)
    f2, g2, b1 = d[2], d[3], d[4]
    u0 = b1 * x / ((n - 1) * b)
    P0 = -(1j * om * f - x * q) / b
    Pg0 = -(-1j * om * g - x * qg) / b
    # derivative of b P = -(i om f - x q):  b1 P + b Px = -(i om q - q - x f2)
    P0x = (-(1j * om * q - q - x * f2) - b1 * P0) / b
    Pg0x = (-(-1j * om * qg - qg - x * g2) - b1 * Pg0) / b
    return f, g, q, qg, f2, g2, b, b1, u0, P0, Pg0, P0x, Pg0x


def pert_rhs(x, Z, n, kap, om, lam, npert):
    """Z = background (5) + npert perturbations of (A, A1, B, B1, db, du)."""
    fM, fr = funcs()
    bg = background_derived(x, Z, n, kap, om)
    out = np.empty_like(Z)
    out[:5] = cl.rhs_c(x, Z[:5], n, kap, om)
    for k in range(npert):
        o = 5 + NP * k
        st = Z[o:o + NP]
        Mx = np.array(fM(x, lam, om, n, kap, *bg, *st), dtype=complex)
        r = np.array(fr(x, lam, om, n, kap, *bg, *st), dtype=complex).ravel()
        s = np.linalg.solve(Mx, r)
        # unknown order: Pa Pb db1 du1 Pa1 Pb1 A2 B2
        out[o:o + NP] = [st[1], s[6], st[3], s[7], s[2], s[3]]
    return out


def momentum_residual(x, Z, n, kap, om, lam, k=0):
    """Linearized momentum constraint (not imposed) for perturbation k, relative to its largest term."""
    fM, fr = funcs()
    bg = background_derived(x, Z, n, kap, om)
    o = 5 + NP * k
    st = Z[o:o + NP]
    Mx = np.array(fM(x, lam, om, n, kap, *bg, *st), dtype=complex)
    r = np.array(fr(x, lam, om, n, kap, *bg, *st), dtype=complex).ravel()
    s = np.linalg.solve(Mx, r)
    return _FMOM(x, lam, om, n, kap, *bg, *st, *s)


def start_vector(n, kap, om, f0, lam, x0, datas):
    """Regular start: A = A0 + a2 x^2, B = B0 + b2 x^2, db = be x^2, du = ga x^2, with (a2, b2, be, ga) fixed by
    A'' = 2 a2, B'' = 2 b2, db' = 2 be x0, du' = 2 ga x0 at x0 (affine in them)."""
    import css_bg as cb
    st = cb.centre_start(n, kap, om, f0, x0)
    f = st[0] + 1j * st[1]; q = st[2] + 1j * st[3]; b = st[4]
    Zb = np.array([f, np.conj(f), q, np.conj(q), b + 0j])
    fM, fr = funcs()
    bg = background_derived(x0, Zb, n, kap, om)
    Z = [Zb]
    for (A0, B0) in datas:
        def state(p):
            a2, b2, be, ga = p
            return [A0 + a2 * x0 ** 2, 2 * a2 * x0, B0 + b2 * x0 ** 2, 2 * b2 * x0, be * x0 ** 2, ga * x0 ** 2]
        def resid(p):
            a2, b2, be, ga = p
            stt = state(p)
            Mx = np.array(fM(x0, lam, om, n, kap, *bg, *stt), dtype=complex)
            r = np.array(fr(x0, lam, om, n, kap, *bg, *stt), dtype=complex).ravel()
            s = np.linalg.solve(Mx, r)
            return np.array([s[6] - 2 * a2, s[7] - 2 * b2, s[2] - 2 * be * x0, s[3] - 2 * ga * x0])
        r0 = resid(np.zeros(4, complex))
        Mj = np.empty((4, 4), complex)
        for k in range(4):
            e = np.zeros(4, complex); e[k] = 1.0
            Mj[:, k] = resid(e) - r0
        p = np.linalg.solve(Mj, -r0)
        Z.append(np.array(state(p)))
    return np.concatenate(Z)


def detM(n, kap, om, f0, lam, xh, R, x0=1e-3, rtol=1e-11, atol=1e-13, return_M=False):
    cl._NEV[0] = 0
    datas = [(1.0, 0.0), (0.0, 1.0)]
    Z0 = start_vector(n, kap, om, f0, lam, x0, datas)
    npert = len(datas)
    F = lambda s, Z: pert_rhs(s, Z, n, kap, om, lam, npert)
    s1 = solve_ivp(F, (x0, xh - R), Z0, method='DOP853', rtol=rtol, atol=atol)
    Za = s1.y[:, -1]
    ends = []
    for sgn in (+1, -1):
        def arc(phi, Z):
            xx = xh + R * np.exp(sgn * 1j * (np.pi - phi))
            dx = -sgn * 1j * R * np.exp(sgn * 1j * (np.pi - phi))
            return pert_rhs(xx, Z, n, kap, om, lam, npert) * dx
        s2 = solve_ivp(arc, (0.0, np.pi), Za, method='DOP853', rtol=rtol, atol=atol)
        ends.append(s2.y[:, -1])
    up, lo = ends
    M = np.empty((2, 2), complex)
    for k in range(2):
        o = 5 + NP * k
        M[0, k] = up[o] - lo[o]          # jump of A
        M[1, k] = up[o + 2] - lo[o + 2]  # jump of B
    d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return (d, M) if return_M else d


def find_root(n, kap, om, f0, lam0, xh, R, tol=1e-11, maxit=30, verbose=False):
    """Secant iteration on det M(lambda)."""
    l0 = complex(lam0); l1 = l0 * (1 + 1e-4) + 1e-5
    d0 = detM(n, kap, om, f0, l0, xh, R); d1 = detM(n, kap, om, f0, l1, xh, R)
    for it in range(maxit):
        l2 = l1 - d1 * (l1 - l0) / (d1 - d0)
        if verbose:
            print("   secant %d: lam = %.12f %+.12fi  |det| = %.2e" % (it, l2.real, l2.imag, abs(d1)), flush=True)
        if abs(l2 - l1) < tol * max(1, abs(l2)):
            return l2
        l0, d0 = l1, d1
        l1, d1 = l2, detM(n, kap, om, f0, l2, xh, R)
    return l2


def winding(n, kap, om, f0, xh, R, re0, re1, im1, K=24):
    """Number of zeros of det M inside the box (argument principle), with the phase tracked along K points per side
    (refined adaptively where the phase jumps by more than 1 rad)."""
    corners = [complex(re0, -im1), complex(re1, -im1), complex(re1, im1), complex(re0, im1), complex(re0, -im1)]
    total = 0.0
    nev = 0
    for a, c in zip(corners[:-1], corners[1:]):
        vals = {}
        def val(t):
            nonlocal nev
            if t not in vals:
                vals[t] = detM(n, kap, om, f0, a + (c - a) * t, xh, R); nev += 1
            return vals[t]
        def incr(t0, t1, depth=0):
            # phase increment, bisected until the two halves agree with the whole and each step is < 0.5 rad
            d = np.angle(val(t1) / val(t0))
            tm = 0.5 * (t0 + t1)
            d1 = np.angle(val(tm) / val(t0)); d2 = np.angle(val(t1) / val(tm))
            if (abs(d1 + d2 - d) > 1e-6 or abs(d) > 0.5) and depth < 30:
                return incr(t0, tm, depth + 1) + incr(tm, t1, depth + 1)
            return d1 + d2
        ts = np.linspace(0, 1, K + 1)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            total += incr(t0, t1)
    return total / (2 * np.pi), nev


if __name__ == "__main__":
    if sys.argv[1] == "count":
        D, kap, om, f0, re0, re1, im1 = [float(s) for s in sys.argv[2:9]]
        K = int(sys.argv[9]) if len(sys.argv) > 9 else 24
        Rf = float(sys.argv[10]) if len(sys.argv) > 10 else 0.85
        n = D - 2
        xh = cl.horizon_estimate(n, kap, om, f0)
        w, nev = winding(n, kap, om, f0, xh, Rf * xh, re0, re1, im1, K)
        print("D=%g kappa=%g: zeros of det M in %.3f < Re lam < %.3f, |Im lam| < %.3f: %.4f  (%d evaluations)" % (D, kap, re0, re1, im1, w, nev))
    else:
        D, kap, om, f0, lre = [float(s) for s in sys.argv[1:6]]
        lim = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        Rf = float(sys.argv[7]) if len(sys.argv) > 7 else 0.85
        n = D - 2
        xh = cl.horizon_estimate(n, kap, om, f0)
        lam = find_root(n, kap, om, f0, complex(lre, lim), xh, Rf * xh, verbose=True)
        print("D=%g kappa=%g R=%.2f x_h: lambda = %.12f %+.12fi   gamma = 1/Re lambda = %.12f" % (D, kap, Rf, lam.real, lam.imag, 1 / lam.real))
