#!/usr/bin/env python3
"""Independent check of the CSS background: analyticity at the self-similarity horizon by complex-path shooting.

    python3 css_lateral.py D KAPPA OMEGA F0 [R]        Newton on the jump, started at (OMEGA, F0)
    python3 css_lateral.py D KAPPA OMEGA F0 R --eval   only evaluate the jump

Near the horizon x_h (b = x) the general solution is (analytic part) + c (x_h - x)^mu (1 + ...), with
Re mu = 1 + (n - 2 + 2(n-1)u_h)/(2(1 - (n-1)u_h)) > 1, so finiteness of f' (or the first-order condition at x_h)
selects nothing; the critical solution is the one with c = 0, i.e. analytic at x_h.  Shoot from the centre with
(omega, |f(0)|), treat f and g = fbar as independent analytic functions, integrate along the real axis to x_h - R and
then along the upper half circle |x - x_h| = R to x_h + R.  The lower path is the complex conjugate (f <-> g), so the
jump across the horizon is J = f_upper(x_h + R) - conj(g_upper(x_h + R)).  J = 0 (two real conditions) fixes omega and
|f(0)|.  This shares no discretization with the collocation code (css_model.py), only the ODE (css_equations.py, D).
"""
import sys, os, math
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_bg as cb


class Budget(Exception):
    pass


_NEV = [0, 100000]


def rhs_c(x, y, n, kap, om):
    _NEV[0] += 1
    if _NEV[0] > _NEV[1]:
        raise Budget("integration budget exceeded")
    f, g, q, qg, b = y
    rho = 1 - kap * f * g
    h = 1 / rho ** 2
    p = (1j * om * f - x * q) / b
    pg = (-1j * om * g - x * qg) / b
    u = 2 * x * h / (n * (n - 1)) * (x * (p * pg + q * qg) + b * (p * qg + pg * q))
    Dl = b * b - x * x
    A = 2 * x - (n - 1) * u * x - b * b / x * (n + (n - 1) * u)
    rf = q * (A - 2j * om * x - 4j * kap * om * x * f * g / rho) - f * (om ** 2 + 1j * om * (1 - (n - 1) * u)) \
        - 2 * kap * g / rho * (Dl * q * q + om ** 2 * f * f)
    rg = qg * (A + 2j * om * x + 4j * kap * om * x * f * g / rho) - g * (om ** 2 - 1j * om * (1 - (n - 1) * u)) \
        - 2 * kap * f / rho * (Dl * qg * qg + om ** 2 * g * g)
    return np.array([q, qg, rf / Dl, rg / Dl, (n - 1) * b * u / x])


def horizon_estimate(n, kap, om, f0, x0=1e-4, frac=0.02):
    """x_h to O(frac^2): shoot (real axis) until b - x = frac x, then extrapolate b - x linearly to zero."""
    y0 = cb.centre_start(n, kap, om, f0, x0)
    ev = lambda x, y, *a: y[4] - x * (1 + frac)
    ev.terminal = True; ev.direction = -1
    sol = solve_ivp(cb.ode_rhs, (x0, 20.0), y0, args=(n, kap, om), method='DOP853', rtol=1e-10, atol=1e-12, events=ev)
    if not len(sol.t_events[0]):
        raise RuntimeError("no horizon found")
    x = sol.t_events[0][0]; y = sol.y_events[0][0]
    bp = cb.ode_rhs(x, y, n, kap, om)[4]
    return x + (y[4] - x) / (1 - bp)


def jump(n, kap, om, f0, R, xh=None, x0=1e-4, rtol=1e-12, atol=1e-14):
    _NEV[0] = 0
    st = cb.centre_start(n, kap, om, f0, x0)
    f = st[0] + 1j * st[1]; q = st[2] + 1j * st[3]; b = st[4]
    y0 = np.array([f, np.conj(f), q, np.conj(q), b + 0j])
    if xh is None:
        xh = horizon_estimate(n, kap, om, f0, x0)
    xa = xh - R
    s1 = solve_ivp(lambda s, y: rhs_c(s, y, n, kap, om), (x0, xa), y0, method='DOP853', rtol=rtol, atol=atol)
    ya = s1.y[:, -1]
    # half circle x = xh + R e^{i(pi - phi)}, phi in [0, pi]  (upper half plane)
    def arc(phi, y):
        x = xh + R * np.exp(1j * (np.pi - phi))
        dx = -1j * R * np.exp(1j * (np.pi - phi))
        return rhs_c(x, y, n, kap, om) * dx
    s2 = solve_ivp(arc, (0.0, np.pi), ya.astype(complex), method='DOP853', rtol=rtol, atol=atol)
    yb = s2.y[:, -1]
    J = yb[0] - np.conj(yb[1])
    return J, xh, (s1.status, s2.status)


def horizon_data(n, kap, om, f0, R, xc=None, K=256, x0=1e-4, rtol=1e-12, atol=1e-14):
    """Values at the true horizon of the (analytic) solution, from Cauchy integrals on the circle |x - xc| = R:
    Taylor coefficients of f, g, b about xc, then b(x_h) = x_h by Newton.  Returns x_h, f(x_h), g(x_h), u_h, and the
    horizon exponent mu (irregular solution (x_h - x)^mu) and kappa_s = 1 - (n-1) u_h."""
    _NEV[0] = 0
    st = cb.centre_start(n, kap, om, f0, x0)
    f = st[0] + 1j * st[1]; q = st[2] + 1j * st[3]; b = st[4]
    y0 = np.array([f, np.conj(f), q, np.conj(q), b + 0j])
    if xc is None:
        xc = horizon_estimate(n, kap, om, f0, x0)
    s1 = solve_ivp(lambda s, y: rhs_c(s, y, n, kap, om), (x0, xc - R), y0, method='DOP853', rtol=rtol, atol=atol)
    def arc(phi, y):
        x = xc + R * np.exp(1j * (np.pi - phi))
        return rhs_c(x, y, n, kap, om) * (-1j * R * np.exp(1j * (np.pi - phi)))
    phis = np.linspace(0, np.pi, K // 2 + 1)
    s2 = solve_ivp(arc, (0.0, np.pi), s1.y[:, -1], method='DOP853', rtol=rtol, atol=atol, t_eval=phis)
    up = s2.y                                   # at x = xc + R e^{i(pi - phi)}, angle theta = pi - phi in [pi, 0]
    th_up = np.pi - phis
    # lower half: values at conj(x) are conj of (g, f, qg, q, b)
    th = np.concatenate([th_up[::-1], -th_up[1:-1]])
    F = np.concatenate([up[0][::-1], np.conj(up[1][1:-1])])
    G = np.concatenate([up[1][::-1], np.conj(up[0][1:-1])])
    Bv = np.concatenate([up[4][::-1], np.conj(up[4][1:-1])])
    order = np.argsort(th); th = th[order]; F = F[order]; G = G[order]; Bv = Bv[order]
    m = len(th)
    def coeffs(vals):
        return np.array([np.mean(vals * np.exp(-1j * k * th)) / R ** k for k in range(m // 2)])
    cF, cG, cB = coeffs(F), coeffs(G), coeffs(Bv)
    ev = lambda c, dx, d=0: sum(c[k] * (math.perm(k, d) if d else 1) * dx ** (k - d) for k in range(d, len(c)))
    dx = 0.0
    for _ in range(50):
        r_ = ev(cB, dx) - (xc + dx)
        dr = ev(cB, dx, 1) - 1
        step = r_ / dr
        dx -= step
        if abs(step) < 1e-15:
            break
    xh = (xc + dx).real
    fh, gh = ev(cF, dx), ev(cG, dx)
    uh = (ev(cB, dx, 1) / (n - 1)).real
    ks = 1 - (n - 1) * uh
    ff = (fh * gh).real
    rho = 1 - kap * ff
    mu = 1 + (n - 2 + 2 * (n - 1) * uh + 2j * om * (1 + 2 * kap * ff / rho)) / (2 * ks)
    return dict(xh=xh, fh=fh, gh=gh, absfh=np.sqrt(ff), uh=uh, kappa_s=ks, mu=mu)


def newton(n, kap, om, f0, R, tol=1e-12, maxit=30, verbose=True, xh=None):
    """Newton on the jump J(omega, |f(0)|) = 0 with backtracking (|J| must decrease); the circle centre xh is
    re-estimated from the current iterate unless given.  R is the circle radius."""
    p = np.array([om, f0], float)
    J, xh_, st = jump(n, kap, p[0], p[1], R, xh=xh)
    for it in range(maxit):
        F = np.array([J.real, J.imag])
        if verbose:
            print("  it %d  omega %.12f  f0 %.12f  x_h %.10f  |J| %.2e" % (it, p[0], p[1], xh_, abs(J)), flush=True)
        if abs(J) < tol:
            break
        Jm = np.empty((2, 2))
        for k in range(2):
            e = 1e-7 * max(1, abs(p[k]))
            pp = p.copy(); pp[k] += e
            J2, _, _ = jump(n, kap, pp[0], pp[1], R, xh=xh_)
            Jm[:, k] = [(J2 - J).real / e, (J2 - J).imag / e]
        dp = np.linalg.solve(Jm, -F)
        t = 1.0
        for _ in range(12):
            pt = p + t * dp
            try:
                Jt, xht, _ = jump(n, kap, pt[0], pt[1], R, xh=xh)
                if not np.isfinite(Jt):
                    Jt = np.inf
            except Exception:
                Jt = np.inf
            if abs(Jt) < abs(J) * (1 - 1e-3 * t) or (abs(J) < 1e-9 and abs(Jt) < 10 * abs(J)):
                break
            t *= 0.5
        else:
            break
        p, J, xh_ = pt, Jt, xht
    return p, abs(J), xh_


def sensitivity(n, kap, om, f0, R, xh, d=1e-6):
    """|dJ/d omega| and |dJ/d f0| by forward differences (the selection strength)."""
    J0, _, _ = jump(n, kap, om, f0, R, xh=xh)
    J1, _, _ = jump(n, kap, om + d, f0, R, xh=xh)
    J2, _, _ = jump(n, kap, om, f0 + d, R, xh=xh)
    return abs(J1 - J0) / d, abs(J2 - J0) / d


def continue_D(kap, Dlist, om, f0, D0, Rfrac=0.85, out=None, step=0.5, seeds=None):
    """Continue the analytic CSS solution in D with Newton on the jump (circle radius Rfrac x_h).  Guesses: the seed
    table {D: (omega, f0)} if given (e.g. collocation values), else quadratic extrapolation of the last three points;
    the step is halved when Newton fails."""
    import json
    xh = horizon_estimate(D0 - 2, kap, om, f0)
    p, aJ, xh = newton(D0 - 2, kap, om, f0, Rfrac * xh, verbose=False)
    hist = [(D0, p[0], p[1])]
    Dcur = D0
    for Dt in Dlist:
        h = step
        while Dt - Dcur > 1e-12:
            Dn = min(Dcur + h, Dt)
            if seeds and Dn in seeds:
                om_g, f_g = seeds[Dn]
            elif len(hist) >= 3:
                Ds = np.array([t[0] for t in hist[-3:]])
                om_g = np.polyval(np.polyfit(Ds, [t[1] for t in hist[-3:]], 2), Dn)
                f_g = np.polyval(np.polyfit(Ds, [t[2] for t in hist[-3:]], 2), Dn)
            elif len(hist) == 2:
                (Da, oa, fa), (Db, ob, fb) = hist
                om_g = ob + (ob - oa) * (Dn - Db) / (Db - Da); f_g = fb + (fb - fa) * (Dn - Db) / (Db - Da)
            else:
                om_g, f_g = hist[-1][1], hist[-1][2]
            try:
                xg = horizon_estimate(Dn - 2, kap, om_g, f_g)
                p, aJ, xh = newton(Dn - 2, kap, om_g, f_g, Rfrac * xg, verbose=False)
                ok = aJ < 1e-10 and abs(p[0] - om_g) < 0.05 * abs(om_g) + 0.02
            except Exception:
                ok = False
            if not ok:
                h *= 0.5
                if h < 1e-3:
                    raise RuntimeError("continuation failed near D = %g" % Dn)
                continue
            hist.append((Dn, p[0], p[1])); Dcur = Dn
            h = min(step, 1.5 * h)
        hd = horizon_data(Dt - 2, kap, p[0], p[1], Rfrac * xh, xc=xh)
        s_om, s_f0 = sensitivity(Dt - 2, kap, p[0], p[1], Rfrac * xh, xh)
        p2, aJ2, _ = newton(Dt - 2, kap, p[0], p[1], 0.7 * Rfrac * xh, verbose=False, xh=xh)
        rec = dict(D=Dt, kappa=kap, omega=p[0], f0=p[1], X=hd["xh"], absfh=hd["absfh"], uh=hd["uh"],
                   kappa_s=hd["kappa_s"], mu=[hd["mu"].real, hd["mu"].imag], R=Rfrac * xh, absJ=aJ,
                   omega_R2=p2[0], f0_R2=p2[1], dJdomega=s_om, dJdf0=s_f0)
        print("D=%6.2f omega=%.12f f0=%.12f x_h=%.10f |f_h|=%.8f mu=%.3f%+.3fi |J|=%.1e (R2: d omega %.1e) |dJ/dom|=%.1e"
              % (Dt, p[0], p[1], hd["xh"], hd["absfh"], hd["mu"].real, hd["mu"].imag, aJ, p2[0] - p[0], s_om), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
    return hist


if __name__ == "__main__":
    if sys.argv[1] == "cont":
        # css_lateral.py cont KAPPA D0 OMEGA0 F00 DLIST [RFRAC] [OUT]
        kap, D0, om, f0 = [float(s) for s in sys.argv[2:6]]
        Dl = [float(s) for s in sys.argv[6].split(",")]
        Rf = float(sys.argv[7]) if len(sys.argv) > 7 else 0.85
        out = sys.argv[8] if len(sys.argv) > 8 else None
        step = float(sys.argv[9]) if len(sys.argv) > 9 else 0.5
        continue_D(kap, Dl, om, f0, D0, Rfrac=Rf, out=out, step=step)
        raise SystemExit
    D, kap, om, f0 = [float(s) for s in sys.argv[1:5]]
    R = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    n = D - 2
    if "--eval" in sys.argv:
        J, xh, st = jump(n, kap, om, f0, R)
        print("D=%g kappa=%g omega=%.10f f0=%.10f  x_h=%.8f  R=%.3f  |J| = %.3e  (status %s)" % (D, kap, om, f0, xh, R, abs(J), st))
    else:
        p, aJ, xh = newton(n, kap, om, f0, R)
        print("D=%g kappa=%g R=%.3f:  omega = %.12f  |f(0)| = %.12f  x_h = %.10f  |J| = %.1e" % (D, kap, R, p[0], p[1], xh, aJ))
