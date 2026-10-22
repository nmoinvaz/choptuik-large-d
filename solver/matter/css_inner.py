#!/usr/bin/env python3
"""The D = infinity inner problem of the phase-rotating CSS solution of the HE kappa-model, and its Stokes selection.

    python3 css_inner.py check                      kappa = 0 against matter/STRUCTURE.md (A = 1.9970774009, omega = 3.9789695074)
    python3 css_inner.py solve KAPPA F0 OMEGA       zero of the Stokes jump from (|f(0)|, omega), at two (r0, r1)

Derivation (HE97 normalization, F = (-t)^{i omega} f; matter/CSS.md section 5).  Put x = sqrt(n) xi in the CSS ODE of
css_equations.py (part D) with f = O(1), b = O(1), U = (n-1) u = O(1), and let n -> infinity:
    xi^2 f'' = f' [ b^2/xi - xi (2 - 2 i omega - U - 4 i kappa omega |f|^2/rho) ] + f (omega^2 + i omega (1 - U))
               - (2 kappa fbar/rho) (xi^2 f'^2 - omega^2 f^2),
    xi b' = U b,   U = 2 xi^2 h |i omega f - xi f'|^2 / b^2,   rho = 1 - kappa |f|^2,  h = rho^-2,   ' = d/dxi.
(The check below re-derives it from css_bg.ode_rhs numerically: the residual of the n -> infinity limit falls like 1/n.)
xi = 0 is an irregular singular point with the fast mode f' ~ e^{-b^2/(2 xi^2)} (the README's heat kernel): the regular
formal solution f = sum_j a_j xi^{2j}, b = sum_j beta_j xi^{2j} (a_0 = |f(0)|, beta_0 = 1) is divergent, and the
selection rule (the README's Conjecture 1 for this solution) is that its Stokes jump across real xi vanishes.  The
jump is computed as in solver/matter/struct_css.py (lateral solutions): optimally truncated series at
xi0 = r0 e^{3 i pi/8}, ray to |xi| = r1, arc to the real axis; the lower solution is the mirror image (f <-> fbar).
With kappa = 0 this is STRUCTURE.md's one-harmonic inner problem in other variables: A = omega |f(0)| (b = c, g_1 = A^2).
"""
import sys, os, json
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import root

mp.mp.dps = 40


# --- truncated power series in z = xi^2 (lists of mpc) -----------------------------------------------------------
def smul(a, b, J):
    return [mp.fsum(a[k] * b[j - k] for k in range(j + 1)) for j in range(J)]


def sinv(a, J):
    out = [1 / a[0]]
    for j in range(1, J):
        out.append(-mp.fsum(a[k] * out[j - k] for k in range(1, j + 1)) / a[0])
    return out


def series(kap, f0, om, J):
    """Coefficients of f, g (= continuation of fbar, omega -> -omega), b in z = xi^2, up to z^(J-1).
    With theta = xi d/dxi (theta z^k = 2k z^k) the inner ODE reads
        b^2 theta f = z [ (theta^2 - theta) f + theta f c1 - f c0 + (2 kappa g/rho)((theta f)^2 - omega^2 f^2) ],
        c1 = 2 - 2 i omega - U - 4 i kappa omega f g/rho,   c0 = omega^2 + i omega (1 - U),   theta b = U b,
    U = 2 z h P Pg / b^2, P = i omega f - theta f; the z^j coefficient gives a_j from lower ones (incremental, O(J^2))."""
    I = mp.mpc(0, 1)
    kap, f0, om = mp.mpf(kap), mp.mpf(f0), mp.mpf(om)
    cv = lambda u, v, k: mp.fsum(u[i] * v[k - i] for i in range(k + 1))
    a, ag, be = [mp.mpc(f0)], [mp.mpc(f0)], [mp.mpc(1)]
    fg, rho, irho, h, P, Pg, PP, hPP, b2, ib2, Uz = ([] for _ in range(11))
    U = [mp.mpc(0)]
    fgr, gr, ar = [], [], []        # f g/rho, g/rho, f/rho
    t2A, t2G = [], []
    def push(k):                    # derived coefficients of index k (needs a, ag, be up to k)
        fg.append(cv(a, ag, k))
        rho.append((1 if k == 0 else 0) - kap * fg[k])
        irho.append(1 / rho[0] if k == 0 else -mp.fsum(rho[i] * irho[k - i] for i in range(1, k + 1)) / rho[0])
        h.append(cv(irho, irho, k))
        P.append(I * om * a[k] - 2 * k * a[k]); Pg.append(-I * om * ag[k] - 2 * k * ag[k])
        PP.append(cv(P, Pg, k)); hPP.append(cv(h, PP, k))
        b2.append(cv(be, be, k))
        ib2.append(1 / b2[0] if k == 0 else -mp.fsum(b2[i] * ib2[k - i] for i in range(1, k + 1)) / b2[0])
        Uz.append(cv(hPP, ib2, k))
        fgr.append(cv(fg, irho, k)); gr.append(cv(ag, irho, k)); ar.append(cv(a, irho, k))
    tA = lambda k: 2 * k * a[k]
    tG = lambda k: 2 * k * ag[k]
    TA, TG = [], []
    def bracket(F_, TF, G_r, k, sgn):
        # coefficient k of (theta^2-theta)F + theta F c1 - F c0 + 2 kappa (G/rho)((theta F)^2 - om^2 F^2)
        c1 = lambda i: (2 - 2 * sgn * I * om if i == 0 else 0) - U[i] - 4 * sgn * I * kap * om * fgr[i]
        c0 = lambda i: (om * om + sgn * I * om if i == 0 else 0) - sgn * I * om * U[i]
        t = (4 * k * k - 2 * k) * F_[k]
        x1 = mp.fsum(TF[i] * c1(k - i) for i in range(k + 1))
        x2 = mp.fsum(F_[i] * c0(k - i) for i in range(k + 1))
        q2 = [mp.fsum(TF[i] * TF[m - i] for i in range(m + 1)) - om * om * mp.fsum(F_[i] * F_[m - i] for i in range(m + 1)) for m in range(k + 1)]
        x3 = mp.fsum(G_r[i] * q2[k - i] for i in range(k + 1))
        return t + x1 - x2 + 2 * kap * x3
    for j in range(1, J):
        push(j - 1)
        TA.append(tA(j - 1)); TG.append(tG(j - 1))
        U.append(2 * Uz[j - 1])
        be.append(mp.fsum(U[i] * be[j - i] for i in range(1, j + 1)) / (2 * j))
        RA = bracket(a, TA, gr, j - 1, 1)
        RG = bracket(ag, TG, ar, j - 1, -1)
        sA = mp.fsum(b2[k] * 2 * (j - k) * a[j - k] for k in range(1, j))
        sG = mp.fsum(b2[k] * 2 * (j - k) * ag[j - k] for k in range(1, j))
        a.append((RA - sA) / (2 * j * b2[0]))
        ag.append((RG - sG) / (2 * j * b2[0]))
    return a, ag, be


def rhs(xi, Z, kap, om):
    f, g, q, qg, b = Z
    rho = 1 - kap * f * g
    h = 1 / rho ** 2
    P = 1j * om * f - xi * q
    Pg = -1j * om * g - xi * qg
    U = 2 * xi * xi * h * P * Pg / (b * b)
    Af = b * b / xi - xi * (2 - 2j * om - U - 4j * kap * om * f * g / rho)
    Ag = b * b / xi - xi * (2 + 2j * om - U + 4j * kap * om * f * g / rho)
    fpp = (q * Af + f * (om * om + 1j * om * (1 - U)) - 2 * kap * g / rho * (xi * xi * q * q - om * om * f * f)) / (xi * xi)
    gpp = (qg * Ag + g * (om * om - 1j * om * (1 - U)) - 2 * kap * f / rho * (xi * xi * qg * qg - om * om * g * g)) / (xi * xi)
    return np.array([q, qg, fpp, gpp, U * b / xi])


def series_value(coef, xi):
    a, ag, be = coef
    xi = mp.mpc(xi); z = xi * xi
    terms = [abs(a[j] * z ** j) for j in range(len(a))]
    jt = min(range(5, len(a)), key=lambda j: terms[j])
    f = mp.fsum(a[j] * z ** j for j in range(jt))
    g = mp.fsum(ag[j] * z ** j for j in range(jt))
    q = mp.fsum(2 * j * a[j] * z ** j for j in range(1, jt)) / xi
    qg = mp.fsum(2 * j * ag[j] * z ** j for j in range(1, jt)) / xi
    b = mp.fsum(be[j] * z ** j for j in range(jt))
    return np.array([complex(v) for v in (f, g, q, qg, b)]), jt, float(terms[jt])


def jump(kap, f0, om, r0=0.1, r1=0.3, rtol=1e-12, atol=1e-15):
    phi0 = 3 * np.pi / 8
    J = int(1 / (2 * r0 ** 2)) + 25
    coef = series(kap, f0, om, J)
    Z0, jt, err = series_value(coef, r0 * np.exp(1j * phi0))
    u = np.exp(1j * phi0)
    s = solve_ivp(lambda t, Z: rhs(t * u, Z, kap, om) * u, (r0, r1), Z0, method="DOP853", rtol=rtol, atol=atol)
    Z = s.y[:, -1]
    s = solve_ivp(lambda t, Z: rhs(r1 * np.exp(1j * t), Z, kap, om) * 1j * r1 * np.exp(1j * t), (phi0, 0.0), Z,
                  method="DOP853", rtol=rtol, atol=atol)
    Z = s.y[:, -1]
    return Z[0] - np.conj(Z[1]), Z, err


def solve(kap, f0, om, r0=0.1, r1=0.3, rtol=1e-12):
    def F(v):
        Jv = jump(kap, v[0], v[1], r0, r1, rtol)[0] * r1 ** (-2 * (v[1] * v[0]) ** 2 / (1 - kap * v[0] ** 2) ** 2)
        return [Jv.real, Jv.imag]
    sol = root(F, [f0, om], method="hybr", options={"xtol": 1e-13})
    return sol.x, np.abs(sol.fun).max(), sol.success


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        # 1. the inner ODE is the n -> infinity limit of the finite-n CSS ODE (css_bg.ode_rhs)
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import css_bg as cb
        kap, om, xi = 0.7, 2.3, 0.37
        f, q, b = 0.4 + 0.1j, -0.2 + 0.3j, 1.3
        Zi = rhs(xi, np.array([f, np.conj(f), q, np.conj(q), b]), kap, om)
        for n in (1e3, 1e4, 1e5):
            x = np.sqrt(n) * xi
            y = cb.ode_rhs(x, [f.real, f.imag, (q / np.sqrt(n)).real, (q / np.sqrt(n)).imag, b], n, kap, om)
            fpp_xi = (y[2] + 1j * y[3]) * n                      # d^2 f/dxi^2 = n d^2 f/dx^2
            bxi = y[4] * np.sqrt(n)
            print("n = %.0e: |f'' inner - f'' finite n| = %.2e   |b' inner - b' finite n| = %.2e" % (n, abs(Zi[2] - fpp_xi), abs(Zi[4].real - bxi)))
        # 2. kappa = 0 against STRUCTURE.md: A = 1.9970774009, omega = 3.9789695074  ->  f0 = A/omega
        A, om = 1.9970774009, 3.9789695074
        for r0, r1 in ((0.1, 0.3), (0.1, 0.25)):
            x, res, ok = solve(0.0, A / om, om, r0, r1)
            print("kappa = 0, r0 %.2f r1 %.2f: |f(0)| = %.10f  omega = %.10f  A = omega|f0| = %.10f  (res %.1e, %s)"
                  % (r0, r1, x[0], x[1], x[0] * x[1], res, ok))
    elif cmd == "kpath":
        # continue the zero in kappa from the kappa = 0 root: css_inner.py kpath K1 [steps] [out.jsonl]
        import json
        k1 = float(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        out = sys.argv[4] if len(sys.argv) > 4 else None
        x = np.array([1.9970774009 / 3.9789695074, 3.9789695074])
        hist = []
        kaps = list(np.linspace(0, k1, steps + 1))
        if out and os.path.exists(out):          # resume: seed the extrapolation with the saved points
            for l in open(out):
                r = json.loads(l)
                hist.append((r["kappa"], np.array([r["f0"], r["omega"]])))
            kaps = [k for k in kaps if k > hist[-1][0] + 1e-12]
            x = hist[-1][1]
        for kap in kaps:
            if len(hist) >= 2:
                (ka, xa), (kb, xb) = hist[-2], hist[-1]
                x = xb + (xb - xa) * (kap - kb) / (kb - ka)
            x, res, ok = solve(kap, x[0], x[1])
            x2, res2, ok2 = solve(kap, x[0], x[1], 0.1, 0.25)
            hist.append((kap, x.copy()))
            h0 = 1 / (1 - kap * x[0] ** 2) ** 2
            rec = dict(kappa=float(kap), f0=float(x[0]), omega=float(x[1]), Delta=float(2 * np.pi / x[1]),
                       A2=float((x[1] * x[0]) ** 2 * h0), kfh=float(kap * x[0] ** 2), res=float(res),
                       f0_r1=float(x2[0]), omega_r1=float(x2[1]))
            print("kappa %.4f: |f(0)| = %.10f  omega = %.10f  Delta = %.10f  A^2 = %.8f  kappa f0^2 = %.6f  (r1=0.25: d omega %.1e; res %.1e %s)"
                  % (kap, x[0], x[1], 2 * np.pi / x[1], rec["A2"], rec["kfh"], x2[1] - x[1], res, ok), flush=True)
            if out:
                with open(out, "a") as fo:
                    fo.write(json.dumps(rec) + "\n")
    elif cmd == "solve":
        kap, f0, om = map(float, sys.argv[2:5])
        for r0, r1 in ((0.1, 0.3), (0.1, 0.25), (0.09, 0.3)):
            x, res, ok = solve(kap, f0, om, r0, r1)
            h0 = 1 / (1 - kap * x[0] ** 2) ** 2
            print("kappa = %g, r0 %.2f r1 %.2f: |f(0)| = %.10f  omega = %.10f  Delta = %.10f  A^2 = om^2 f0^2 h0 = %.8f  (res %.1e, %s)"
                  % (kap, r0, r1, x[0], x[1], 2 * np.pi / x[1], (x[1] * x[0]) ** 2 * h0, res, ok), flush=True)
