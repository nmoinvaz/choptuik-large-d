#!/usr/bin/env python3
"""The type of the D = infinity profile's singularity at theta_0 from the inner equations (#397; pre-registered on
the issue).

Stage 1 (wtc): the local (Weiss-Tabor-Carnevale) expansion of the gauge-fixed inner equations (DINF_ANALYTIC.md 1.1,
form A), Y_s + (V Y)_u = Y_uu, V_u = Y^2/4, about a movable singular manifold psi = u - f(s) = 0, with ln psi and
ln^2 psi terms allowed at every order; and of the planted ODE (A0), Y' = V Y + k1, V' = Y^2/4 (the s-independent
reduction, whose resonance condition -k1/2 = 0 fails for k1 != 0). For each order: the coefficients fixed, the free
ones, and the log terms forced.

Stage 2 (planted): Q1, exact Taylor coefficients of R_w/(w0 - w) + B_w (w0 - w) ln(1 - w/w0) + a far pole, against
the finite-k curve of the pole-plus-log model; Q2, the planted ODE's Taylor series (recursion at 80 digits), whose
nearest singularity is located from the ratio limit and whose log coefficient b2 = k1/3 comes from stage 1; and Q2's
control, k1 = 0 (the logarithm removed), whose (k + 1.5)^4 |r_{k+1} - r_k| must fall by more than 10 from k = 10 to 30.
Stage 3 (profile): F_nu at A = 2.72 from the amplitude expansion (K 45, omega_inf, each A series with its geometric
tail), c_k = F_{2k+1}; the local exponent, the free residue fit and the fixed-residue consistency fit.

    cd solver/eeg_profile
    python3 profile_singularity.py wtc [--order=4]
    python3 profile_singularity.py planted [--k1=0.3]
    python3 profile_singularity.py profile [--A=2.72]
"""
import sys

import sympy as sp

args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
s, P, L = sp.symbols("s psi Lg")
k1 = sp.Symbol("k1")


def expansion(system, order, lead):
    """order-by-order solution. system: 'A' (s-dependent) or 'A0' (ODE). lead: (a0, v0). Returns the list of
    (order, solved dict, free unknowns, residual conditions) and the substituted series."""
    sfun = system == "A"
    mk = (lambda nm: sp.Function(nm)(s)) if sfun else (lambda nm: sp.Symbol(nm))
    f = sp.Function("f")(s) if sfun else sp.Integer(0)
    fp = sp.diff(f, s) if sfun else sp.Integer(0)
    co = {}
    Y = lead[0] / P
    V = lead[1] / P
    for j in range(1, order + 1):
        for nm in ("a", "b", "e", "v", "d", "g"):
            co[(nm, j)] = mk("%s%d" % (nm, j))
        Y += (co[("a", j)] + co[("b", j)] * L + co[("e", j)] * L ** 2) * P ** (j - 1)
        V += (co[("v", j)] + co[("d", j)] * L + co[("g", j)] * L ** 2) * P ** (j - 1)

    def Du(e):
        return sp.diff(e, P) + sp.diff(e, L) / P

    def Ds(e):
        if not sfun:
            return sp.Integer(0)
        return sp.diff(e, s) - fp * Du(e)
    if sfun:
        E1 = Ds(Y) + Du(V * Y) - Du(Du(Y))
        src = Y ** 2 / 4
    else:
        E1 = Du(Y) - V * Y - k1                       # Y' = V Y + k1 (u = x, psi = x - x0)
        src = Y ** 2 / 4
    E2 = Du(V) - src
    E1, E2 = sp.expand(E1), sp.expand(E2)
    q1 = -3 if sfun else -2                           # leading power of E1 (Y_uu ~ psi^-3; ODE: Y' ~ psi^-2)
    q2 = -2

    def coeff(E, pw, lg):
        out = sp.Integer(0)
        for term in sp.Add.make_args(E):
            c, ex = term.as_independent(P, as_Add=False)
            _, pex = ex.as_base_exp() if ex != 1 else (P, 0)
            if ex == 1:
                pex = 0
            elif ex.is_Pow:
                pex = ex.exp
            else:
                pex = 1
            cl = sp.Poly(c, L) if c.has(L) else None
            if cl is None:
                if lg == 0 and pex == pw:
                    out += c
            elif pex == pw:
                out += cl.coeff_monomial(L ** lg)
        return sp.expand(out)
    # the leading order must vanish
    lead_res = [sp.simplify(coeff(E1, q1, 0)), sp.simplify(coeff(E2, q2, 0))]
    sub = {}
    report = []
    for j in range(1, order + 1):
        unk = [co[(nm, j)] for nm in ("a", "b", "e", "v", "d", "g")]
        eqs = []
        for E, q in ((E1, q1), (E2, q2)):
            for lg in (0, 1, 2, 3):
                e = coeff(E, q + j, lg)
                e = sp.expand(e.subs(sub).doit())
                if sfun:
                    e = sp.expand(e.subs(sub).doit())
                if e != 0:
                    eqs.append(e)
        # solve the linear system for this order's unknowns; free ones are those left undetermined
        sol = sp.solve(eqs, unk, dict=True)
        if not sol:
            report.append((j, None, unk, eqs))
            break
        sol = sol[0]
        for u_, v_ in sol.items():
            sub[u_] = sp.simplify(v_)
        free = [u_ for u_ in unk if u_ not in sol]
        resid = [sp.simplify(e.subs(sub).doit()) for e in eqs]
        report.append((j, {k: sub[k] for k in unk if k in sub}, free, [r for r in resid if r != 0]))
    return lead_res, report, sub


def cmd_wtc():
    order = int(args.get("order", "4"))
    ok = True
    for name, system, lead in (("form (A), Y_s + (VY)_u = Y_uu, V_u = Y^2/4", "A", (2, -1)),
                               ("form (A), the other sign of the residue", "A", (-2, -1)),
                               ("planted ODE (A0), Y' = V Y + k1, V' = Y^2/4", "A0", (-2, -1)),
                               ("planted ODE (A0), the other sign", "A0", (2, -1))):
        lead_res, report, sub = expansion(system, order, lead)
        print("%s, leading Y = %s/psi, V = %s/psi: leading residuals %s" % (name, lead[0], lead[1], lead_res))
        ok &= all(r == 0 for r in lead_res)
        for j, solved, free, resid in report:
            if solved is None:
                print("  order %d: no solution; equations %s" % (j, resid))
                ok = False
                continue
            nz = {k: v for k, v in solved.items() if v != 0}
            zero = [k for k, v in solved.items() if v == 0]
            print("  order %d (psi^%d): fixed %s; zero %s; free %s%s" % (
                j, j - 1, nz if nz else "{}", zero, free, "; unresolved %s" % resid if resid else ""))
    return ok


# ---- coefficient statistics --------------------------------------------------------------------------------------
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import mpmath as mp  # noqa: E402

mp.mp.dps = 80


def stats(c):
    """r_k = c_{k+1}/c_k, D_k = |r_{k+1} - r_k|, C_k = (k + 1.5)^4 D_k, local exponents p_k"""
    r = [c[k + 1] / c[k] for k in range(len(c) - 1)]
    D = [abs(r[k + 1] - r[k]) for k in range(len(r) - 1)]
    C = [(k + mp.mpf("1.5")) ** 4 * D[k] for k in range(len(D))]
    p = [-mp.log(D[k + 1] / D[k]) / mp.log((k + mp.mpf("2.5")) / (k + mp.mpf("1.5"))) for k in range(len(D) - 1)]
    return r, D, C, p


def model_coeffs(x0, kappa, n):
    """pole plus a log in the coefficients' variable: c_k = x0^-k (1 + kappa/(k(k-1))), k >= 2"""
    return [x0 ** (-k) * (1 + (kappa / (k * (k - 1)) if k >= 2 else 0)) for k in range(n)]


def cmd_planted():
    ok = True
    # Q1: R_w/(w0 - w) + B_w (w0 - w) ln(1 - w/w0) + R2/(w1 - w)
    th0 = mp.mpc("3.1052", "-1.4284")
    w0 = mp.exp(2j * th0)
    kap = mp.mpf("0.0326") * mp.expj(0.5)
    Rw, R2 = mp.mpf(1), mp.mpf("0.3")
    Bw = kap * Rw / w0 ** 2
    w1 = 30 * w0 * mp.expj(0.4)
    n = 14
    tk = [mp.mpf(0), mp.mpf(-1)] + [w0 ** (1 - k) / (k * (k - 1)) for k in range(2, n)]
    c = [Rw / w0 ** (k + 1) + Bw * tk[k] + R2 / w1 ** (k + 1) for k in range(n)]
    _, _, C, _ = stats(c)
    _, _, Cm, _ = stats(model_coeffs(w0, kap, n))
    asym = 6 * abs(kap) / abs(w0)
    dev = [float(abs(C[k] / Cm[k] - 1)) for k in range(5, 9)]
    good = max(dev) <= 0.03
    ok &= good
    print("Q1 (planted function, log in w): C_k at k = 5..8 %s; predicted finite-k curve %s (asymptote %.5f); "
          "largest deviation %.1e: %s" % (", ".join("%.5f" % float(C[k]) for k in range(5, 9)),
                                           ", ".join("%.5f" % float(Cm[k]) for k in range(5, 9)), float(asym),
                                           max(dev), "PASS" if good else "FAIL"))
    # Q2: Y' = V Y + k1, V' = Y^2/4, Taylor series at x = 0
    for k1v, control in ((mp.mpf(args.get("k1", "0.3")), False), (mp.mpf(0), True)):
        N = 90
        Y = [mp.mpc("0.4", "0.9")]
        V = [mp.mpc("-0.3", "0.5")]
        for nn in range(N):
            Y.append((mp.fsum(V[j] * Y[nn - j] for j in range(nn + 1)) + (k1v if nn == 0 else 0)) / (nn + 1))
            V.append(mp.fsum(Y[j] * Y[nn - j] for j in range(nn + 1)) / 4 / (nn + 1))
        r, D, C, p = stats(Y)
        # the singularity: 1/x0 from the ratios, Richardson on r_k assuming O(k^-3) corrections
        K = N - 3
        inv = (r[K] * (K + 1) ** 3 - r[K - 1] * K ** 3) / ((K + 1) ** 3 - K ** 3)
        x0 = 1 / inv
        a = -Y[K] * x0 ** (K + 1)                    # residue: c_k ~ -a / x0^(k+1)
        if control:
            fall = float(C[10] / C[30])
            good = fall > 10
            ok &= good
            print("Q2 control (k1 = 0, the logarithm removed): x0 = %s, residue %s; C_10 = %.3e, C_30 = %.3e, "
                  "C_60 = %.3e: falls by %.1e from k = 10 to 30 (must be > 10): %s" % (
                      mp.nstr(x0, 10), mp.nstr(a, 8), float(C[10]), float(C[30]), float(C[60]), fall,
                      "the check fails, as it must" if good else "DOES NOT FAIL"))
            print("   at the settled range (where Q2 is compared): C_k at k = 60, 70, 80: %s; local exponents %s "
                  "(the k^-4 law gives 4)" % (", ".join("%.2e" % float(C[k]) for k in (60, 70, 80)),
                                              ", ".join("%.1f" % float(p[k]) for k in (60, 70, 80))))
            continue
        b2 = k1v / 3
        kap = b2 * x0 ** 2 / a
        a = a / (1 + kap / (K * (K - 1)))              # the residue without the log's 1/k^2 term
        kap = b2 * x0 ** 2 / a
        _, _, Cm, _ = stats(model_coeffs(x0, kap, N))
        print("   C_k / predicted along k: " + ", ".join("k %d: %.4f" % (k, float(C[k] / Cm[k]))
                                                         for k in range(10, N - 2, 10)))
        ks = list(range(60, 81, 5))                    # where the second singularity's geometric term has died
        dev = [float(abs(C[k] / Cm[k] - 1)) for k in ks]
        good = max(dev) <= 0.03 and abs(abs(a) - 2) < 1e-4    # residue: the psi^3 ln^2 terms leave ~1e-6 at K 87
        ok &= good
        print("Q2 (planted ODE, k1 = %s): nearest singularity x0 = %s, residue a = %s (stage 1: +-2); "
              "kappa = b2 x0^2/a = %s with b2 = k1/3" % (mp.nstr(k1v, 3), mp.nstr(x0, 12), mp.nstr(a, 10),
                                                       mp.nstr(kap, 8)))
        print("   C_k at k = %s: measured %s; predicted %s; largest deviation %.1e; local exponents p_k %s: %s" % (
            ks, ", ".join("%.5f" % float(C[k]) for k in ks), ", ".join("%.5f" % float(Cm[k]) for k in ks),
            max(dev), ", ".join("%.3f" % float(p[k]) for k in ks), "PASS" if good else "FAIL"))
    print("planted: %s" % ("PASS" if ok else "FAIL"))
    return ok


HERE = os.path.dirname(os.path.abspath(__file__))
F1500 = os.path.join(HERE, "..", "inner_amp", "scan45J1500", "sigF_w1.0000_K45_J1500_p1280.json")
OMEGA = 2 * mp.pi / mp.mpf("1.64389375")


def profile_harmonics(A, path=F1500, numax=21):
    """F_nu at A (e^{i nu theta} coefficient, repository gauge, F_1 = -iA/2), each A series with its geometric tail;
    returns {nu: (F_nu, |tail|)}"""
    d = json.load(open(path))
    f = {tuple(int(v) for v in k.split("_")): mp.mpc(*v) for k, v in d["f"].items()}
    out = {1: (-1j * A / 2, mp.mpf(0))}
    for nu in range(3, numax + 1, 2):
        t = [f[(k, nu)] * A ** k for k in range(nu, 46, 2) if (k, nu) in f]
        S = mp.fsum(t)
        q = t[-1] / t[-2]
        tail = t[-1] * q / (1 - q) if abs(q) < 1 else mp.mpc(0)
        out[nu] = (S + tail, abs(tail))
    return out


def cmd_profile():
    A = mp.mpf(args.get("A", "2.72"))
    H = profile_harmonics(A)
    c = [H[2 * k + 1][0] for k in range(11)]
    tails = [H[2 * k + 1][1] / abs(H[2 * k + 1][0]) for k in range(11)]
    r, D, C, p = stats(c)
    ok = True
    print("profile at A = %s (K 45, omega_inf, J 1500): c_k = F_{2k+1}; relative tail added per k %s" % (
        A, ", ".join("%.0e" % float(t) for t in tails)))
    print("  C_k = (k + 1.5)^4 |r_{k+1} - r_k|, k = 0..8: " + ", ".join("%.5f" % float(v) for v in C))
    print("  local exponents p_k, k = 0..7: " + ", ".join("%.3f" % float(v) for v in p))
    # test 1: exponent
    t1 = all(abs(p[k] - 4) <= 0.2 for k in (5, 6, 7))
    ok &= t1
    print("test 1 (exponent): p_5, p_6, p_7 = %s; 4 +- 0.2: %s" % (
        ", ".join("%.4f" % float(p[k]) for k in (5, 6, 7)), "PASS" if t1 else "FAIL"))
    # test 2: free fit c_k = w0^-k [rho + kp/(k+1/2)^2 + lp/(k+1/2)^3] on k = 4..9 (6 complex data, 4 unknowns)
    ks = list(range(4, 10))

    def resid(prm, fixed_rho=None):
        w0, rho, kp, lp = prm if fixed_rho is None else (prm[0], fixed_rho, prm[1], prm[2])
        return [c[k] * w0 ** k - (rho + kp / (k + mp.mpf(0.5)) ** 2 + lp / (k + mp.mpf(0.5)) ** 3) for k in ks]

    def lsq(fun, x0):
        x = list(x0)
        for _ in range(60):                         # Gauss-Newton with complex-analytic residuals
            F0 = fun(x)
            J = []
            for i in range(len(x)):
                h = mp.mpf("1e-25") * (1 + abs(x[i]))
                xp = list(x)
                xp[i] = x[i] + h
                J.append([(a - b) / h for a, b in zip(fun(xp), F0)])
            Jm = mp.matrix(len(F0), len(x))
            for i in range(len(x)):
                for j in range(len(F0)):
                    Jm[j, i] = J[i][j]
            JH = Jm.transpose_conj()
            dx = mp.lu_solve(JH * Jm, -(JH * mp.matrix(F0)))
            x = [x[i] + dx[i] for i in range(len(x))]
            if max(abs(v) for v in dx) < mp.mpf("1e-40"):
                break
        return x, max(abs(v) for v in fun(x))
    w0g = 1 / r[8]
    rhog = c[8] * w0g ** 8
    (w0, rho, kp, lp), res = lsq(resid, [w0g, rhog, mp.mpc(0), mp.mpc(0)])
    th0 = mp.log(w0) / 2j                            # e^{2 i theta0} = w0; Im theta0 < 0
    if mp.im(th0) > 0:
        th0 = mp.conj(th0)
    # F_nu ~ -2i R e^{-i nu theta0}: c_k = -2i R e^{-i theta0} w0^-k, so R = i rho e^{i theta0} / 2 (both roots of w0)
    Rs = [1j * rho * sg * mp.sqrt(w0) / 2 for sg in (1, -1)]
    R = min(Rs, key=lambda z: min(abs(mp.arg(z)), abs(abs(mp.arg(z)) - mp.pi)))
    Rpred = OMEGA / mp.sqrt(2)
    argdev = min(abs(mp.arg(R)), abs(abs(mp.arg(R)) - mp.pi))
    t2 = abs(abs(R) / Rpred - 1) <= 0.002 and argdev <= 0.01
    ok &= t2
    print("test 2 (residue, free fit on k = 4..9, max residual %.1e): theta0 = %s (repository gauge), |R| = %s "
          "against omega/sqrt2 = %s (%+.2e), arg R off 0 or pi by %.2e; kappa' = %s, lambda' = %s: %s" % (
              float(res), mp.nstr(th0, 8), mp.nstr(abs(R), 8), mp.nstr(Rpred, 8), float(abs(R) / Rpred - 1),
              float(argdev), mp.nstr(kp / rho, 6), mp.nstr(lp / rho, 6), "PASS" if t2 else "FAIL"))
    # test 3: R fixed at the prediction (sign from test 2), fit w0, kappa, lambda
    sgn = 1 if abs(mp.arg(R)) < 1 else -1
    # rho = -2i R e^{-i theta0}, with e^{i theta0} the root of w0 that gave R above
    root = mp.sqrt(w0) if R == Rs[0] else -mp.sqrt(w0)

    def resid3(prm):
        w0_, kp_, lp_ = prm
        rt = mp.sqrt(w0_) * (1 if mp.re(mp.sqrt(w0_) / root) > 0 else -1)
        rho_ = -2j * sgn * Rpred / rt
        return [c[k] * w0_ ** k - rho_ * (1 + kp_ / (k + mp.mpf(0.5)) ** 2 + lp_ / (k + mp.mpf(0.5)) ** 3) for k in ks]
    (w03, kp3, lp3), res3 = lsq(resid3, [w0, kp / rho, lp / rho])
    rt3 = mp.sqrt(w03) * (1 if mp.re(mp.sqrt(w03) / root) > 0 else -1)
    rho3 = -2j * sgn * Rpred / rt3
    cm = [rho3 * w03 ** (-k) * (1 + kp3 / (k + mp.mpf(0.5)) ** 2 + lp3 / (k + mp.mpf(0.5)) ** 3) for k in range(11)]
    _, _, Cm, _ = stats(cm)
    dev = [float(abs(Cm[k] / C[k] - 1)) for k in (5, 6, 7, 8)]
    t3 = max(dev) <= 0.03
    ok &= t3
    print("test 3 (constant, consistency; R fixed at %s omega/sqrt2, fit of w0, kappa, lambda on k = 4..9, max "
          "residual %.1e): model C_k at k = 5..8 %s against the data's %s; largest deviation %.1e: %s" % (
              "+" if sgn > 0 else "-", float(res3), ", ".join("%.5f" % float(Cm[k]) for k in (5, 6, 7, 8)),
              ", ".join("%.5f" % float(C[k]) for k in (5, 6, 7, 8)), max(dev), "PASS" if t3 else "FAIL"))
    kap_nu = 4 * kp3                                # kp3/(k+1/2)^2 = 4 kp3/nu^2 with nu = 2k + 1
    lam_nu = 8 * lp3                                # lp3/(k+1/2)^3 = 8 lp3/nu^3
    print("  in nu: F_nu ~ -2i R e^{-i nu theta0} (1 + kappa/nu^2 + kappa3/nu^3), kappa = %s, |kappa| = %.5f; "
          "kappa3/kappa = %s against the predicted 3i/omega = %s (the centring; b-free)" % (
              mp.nstr(kap_nu, 6), float(abs(kap_nu)), mp.nstr(lam_nu / kap_nu, 5), mp.nstr(3j / OMEGA, 5)))
    print("  B/R = kappa (log strength relative to the pole) and C_inf = 1.5 e^{2 Im theta0} |kappa| = %.5f" % (
        float(1.5 * mp.exp(2 * mp.im(th0)) * abs(kap_nu))))
    # post hoc (not part of the test): kappa3 fixed at the b-free prediction 3i kappa/omega, a free nu^-4 term
    def resid4(prm):
        w0_, kp_, mu_ = prm
        rt = mp.sqrt(w0_) * (1 if mp.re(mp.sqrt(w0_) / root) > 0 else -1)
        rho_ = -2j * sgn * Rpred / rt
        out = []
        for k in ks:
            nu = 2 * k + 1
            out.append(c[k] * w0_ ** k - rho_ * (1 + kp_ / nu ** 2 + 3j * kp_ / OMEGA / nu ** 3 + mu_ / nu ** 4))
        return out
    (w04, kp4, mu4), res4 = lsq(resid4, [w03, kap_nu, mp.mpc(0)])
    rt4 = mp.sqrt(w04) * (1 if mp.re(mp.sqrt(w04) / root) > 0 else -1)
    rho4 = -2j * sgn * Rpred / rt4
    cm4 = [rho4 * w04 ** (-k) * (1 + kp4 / (2 * k + 1) ** 2 + 3j * kp4 / OMEGA / (2 * k + 1) ** 3
                                 + mu4 / (2 * k + 1) ** 4) for k in range(11)]
    _, _, Cm4, _ = stats(cm4)
    th04 = mp.log(w04) / 2j
    print("post hoc: kappa3 fixed at 3i kappa/omega, free mu/nu^4 (max residual %.1e): kappa = %s, |kappa| = %.5f, "
          "mu = %s; model C_k at k = 5..8 %s; C_inf = 1.5 e^{2 Im theta0} |kappa| = %.5f" % (
              float(res4), mp.nstr(kp4, 6), float(abs(kp4)), mp.nstr(mu4, 4),
              ", ".join("%.5f" % float(Cm4[k]) for k in (5, 6, 7, 8)),
              float(1.5 * mp.exp(-2 * abs(mp.im(th04))) * abs(kp4))))
    print("profile: %s" % ("PASS" if ok else "FAIL"))
    if "out" in args:
        json.dump({"A": str(A), "C": [float(v) for v in C], "p": [float(v) for v in p], "theta0": str(th0),
                   "R": str(R), "kappa_nu": str(kap_nu), "kappa3_nu": str(lam_nu), "t1": t1, "t2": t2, "t3": t3},
                  open(args["out"], "w"), indent=1)
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "profile":
        sys.exit(0 if cmd_profile() else 1)
    if cmd == "wtc":
        sys.exit(0 if cmd_wtc() else 1)
    if cmd == "planted":
        sys.exit(0 if cmd_planted() else 1)
    raise SystemExit(__doc__)
