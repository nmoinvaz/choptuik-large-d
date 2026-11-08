"""Ecker, Ecker and Grumiller's large-D expansion to NNLO (arXiv:2611.07007v1, supplement "Simplified large-D
perturbation theory", Eqs. (S35)-(S51)) against this repository's outer 1/n expansion (README Proposition 1), in sympy
(notes/SECOND_ORDER.md, section 1).

The map (solver/supercrit/eqmap.py, checked there symbolically): EEG's GG gauge is ours, with
    f = c,   x^2 Omega = u = (n - 1)(b - 1),   y = p - m = sqrt(2) x Pi,   p + m = sqrt(2) x^2 Psi,
    eps_E = 1/(D - 1) = 1/(n + 1) = e/(1 + e),   e = 1/n,   tau the same (theta = omega tau, primes = d/dtau).
Their free function is the exact central value, beta~ = Pi(tau, 0)/sqrt(eps_E) (S34). In Proposition 1,
D = sum e^k D_k with D_0 = 2xF and D_k = x K_k + O(x^3), so Pi(tau, 0) = sqrt(e) (2F + sum e^k K_k)/sqrt(2) and
    beta~ = sqrt(2 (1 + e)) F_n,   F_n = F + sum_k e^k K_k/2   (F_n = sqrt(n) p_x(0), the exact profile).
EEG's period is implicit (primes are tau-derivatives); ours enters through d/dtau = omega d/dtheta with
omega = w0 + w1 e + w2 e^2 + ... Both are therefore the same family (Proposition 1's free data F, K_k, omega_k are
EEG's beta~(tau)), and the expansions must agree coefficient by coefficient.

Checks (exit status 1 if any fails):
  A  EEG's f, Omega to O(eps^3), Pi to O(eps^(5/2)), Psi to O(eps^(5/2)), re-expanded in e = 1/n with the map above,
     equal Proposition 1's c, u/x^2, y/(sqrt 2 x), (p + m)/(sqrt 2 x^2) to the same order, from this script's own
     recursion of the outer equations (as solver/largeD_expand.py; F, K_1, K_2, w_0, w_1, w_2 symbolic).
  B  EEG's series substituted into their own equations (S36)-(S39) leave residuals of the next order only.
  C  the gauge map Delta = 4 pi mu / (1 - <Omega(tau, 1)>) (notes/RT_FORMULATION.md; their Eq. (3) with
     <psi_{-1}^2> = <x^2 Omega> at x = 1) expanded in eps_E: mu_0 = Delta_0/(4 pi), their (S59) for mu_1, and mu_2.
  D  the printed NNLO violates their (S39) at eps^(5/2), and the correction delta_Pi3 below is exactly what (S39)
     requires ((1/x) d_x delta = the residual).
Result. The NLO (S40)-(S43) agrees with Proposition 1 as printed. The NNLO as printed does not: (S46) for Pi_3 fails
their own (S39) at eps^(5/2), and (S45), (S48)-(S51) inherit the error through Omega_3 = 2 beta~ Pi_3 + ... and
(S37). With Pi_3 + delta_Pi3 and the f_3 coefficients that follow (delta_f3), A and B pass. By default the script
checks this corrected NNLO; --printed checks the NNLO as printed, and exits 1.
Mutations (each must make the default run exit 1; they act on the printed coefficients): --mutate=c0 (S48's 17/48 ->
18/48), --mutate=Pi3 (S46's 24 -> 23), --mutate=Psi2 (S43's 11 -> 10), --mutate=Omega3 (S45's -7 -> -6),
--mutate=map (eps_E = e instead of e/(1 + e)).

    python3 second_order_eeg.py [--printed] [--mutate=NAME]
"""
import sys
import time

import sympy as sp

T0 = time.time()
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
FAILS = []

x, e = sp.symbols("x e", positive=True)          # e = 1/n
NDER = 9
Fd = sp.symbols(f"F0:{NDER}")                       # F and its theta derivatives
K1d = sp.symbols(f"Ka0:{NDER}")
K2d = sp.symbols(f"Kb0:{NDER}")
w0, w1, w2, w3 = sp.symbols("w0 w1 w2 w3")
CHAINS = [Fd, K1d, K2d]


def dth(expr):
    """d/dtheta on polynomials in the derivative symbols"""
    expr = sp.expand(expr)
    out = 0
    for ch in CHAINS:
        for j in range(NDER - 1):
            if expr.has(ch[j]):
                out += sp.diff(expr, ch[j]) * ch[j + 1]
    return sp.expand(out)


def trunc(expr, N, var=e):
    """drop powers of var above N"""
    expr = sp.expand(expr)
    return sp.expand(sum(expr.coeff(var, k) * var ** k for k in range(N + 1)))


# ------------------------------------------------------------------ our outer expansion (Proposition 1)
def outer(NO=3):
    """S_k, D_k (P = S/2 + D/2 scaled by e^(1/2)), G_k (c = 1 - sum e^k G_k), B_k (b = 1 + sum e^k B_k), with omega =
    w0 + w1 e + w2 e^2 and K_1, K_2 symbolic. Same recursion as solver/largeD_expand.py (written again here)."""
    N = NO + 2
    W = [w0, w1, w2, w3][:N] + [0] * max(0, N - 4)
    Ks = [0, K1d[0], K2d[0], 0, 0]
    S = [sp.Integer(0)] * N
    D = [2 * x * Fd[0]] + [sp.Integer(0)] * (N - 1)
    G = [sp.Integer(0)] * N
    B = [sp.Integer(0)] * N

    def mul(a, b):
        return [sp.expand(sum(a[i] * b[k - i] for i in range(k + 1))) for k in range(N)]

    def add(*ss):
        return [sp.expand(sum(s[k] for s in ss)) for k in range(N)]

    def scal(a, cf):
        return [sp.expand(cf * v) for v in a]

    def shift(a, k):
        return ([sp.Integer(0)] * k + list(a))[:N]

    def inv(a):
        out = [1 / a[0]]
        for k in range(1, N):
            out.append(sp.expand(-sum(a[i] * out[k - i] for i in range(1, k + 1)) / a[0]))
        return out
    ome = [sp.Integer(1), sp.Integer(-1)] + [sp.Integer(0)] * (N - 2)
    cser = lambda: [sp.Integer(1)] + [-G[k] for k in range(1, N)]
    for k in range(1, NO + 1):
        P = [(S[i] + D[i]) / 2 for i in range(N)]
        Mm = [(S[i] - D[i]) / 2 for i in range(N)]
        q2 = add(mul(P, P), mul(Mm, Mm))
        opB = add([sp.Integer(1)] + [0] * (N - 1), B)
        rest = add(shift([sp.expand(x * sp.diff(v, x)) for v in B], 1), scal(shift(mul(opB, q2), 2), -1),
                   mul(ome, mul(opB, B)))
        B[k + 1] = sp.expand(-rest[k + 1])
        c = cser()
        rhs = mul(ome, mul(c, B))[k + 1]
        G[k] = sp.expand(sp.integrate(sp.expand(-rhs / x), (x, 1, x)))
        if k == NO:
            break
        c = cser()
        opB = add([sp.Integer(1)] + [0] * (N - 1), B)
        num = add([sp.expand(x ** 2 * sp.diff(v, x)) for v in D],
                  scal(mul(c, [sp.diff(v, x) for v in S]), -x), scal(mul(W, [dth(v) for v in D]), x))
        Snew = shift(mul(num, inv(mul(c, mul(ome, opB)))), 1)
        S[k] = sp.expand(Snew[k])
        Dk = sp.Function("Dk")(x)
        Dt = D[:k] + [Dk] + [sp.Integer(0)] * (N - k - 1)
        lhs = add(scal(mul(c, [sp.diff(v, x) for v in Dt]), x), scal(mul(c, Dt), -1))
        cBD = mul(c, mul(B, Dt))
        rr = add([sp.expand(x ** 2 * sp.diff(v, x)) for v in S], scal(mul(W, [dth(v) for v in S]), x),
                 scal([cBD[i + 1] if i + 1 < N else 0 for i in range(N)], -1), cBD)
        Rk = sp.expand(rr[k] - (lhs[k] - (x * sp.diff(Dk, x) - Dk)))
        assert not Rk.has(Dk)
        r1 = sp.Poly(Rk, x).coeff_monomial(x)
        assert sp.expand(r1) == 0, "solvability"
        D[k] = sp.expand(x * sp.integrate(sp.expand(Rk / x ** 2), x) + x * Ks[k])
    return S, D, G, B


# ------------------------------------------------------------------ EEG's expansion, as printed (beta read as beta~)
def delta_Pi3(b):
    """the correction to EEG's (S46) that their own (S39) requires at eps^(5/2) (check D derives it again)"""
    B0, B1, B2, B3, B4 = b[0], b[1], b[2], b[3], b[4]
    return (-B0 * x ** 2 * (4 * B0 ** 2 + B0 * B1 - B1 ** 2) / 2
            - x ** 4 * (3 * B0 ** 5 + 8 * B0 ** 3 + 40 * B0 ** 2 * B1 + 8 * B0 ** 2 * B2 + 24 * B0 * B1 ** 2 - 24 * B0
                        - 62 * B1 - 57 * B2 - 22 * B3 - 3 * B4) / 24)


def delta_f3(b):
    """the corrections to (S48)-(S51) that follow from delta_Pi3 through (S45) and (S37) with f(1) = 1"""
    B0, B1, B2, B3, B4 = b[0], b[1], b[2], b[3], b[4]
    R = sp.Rational
    d0 = (B0 ** 6 / 24 + R(10, 9) * B0 ** 4 + R(29, 36) * B0 ** 3 * B1 + B0 ** 3 * B2 / 9 + B0 ** 2 * B1 ** 2 / 12
          - B0 ** 2 / 3 - R(31, 36) * B0 * B1 - R(19, 24) * B0 * B2 - R(11, 36) * B0 * B3 - B0 * B4 / 24)
    d4 = -B0 ** 4 - B0 ** 3 * B1 / 4 + B0 ** 2 * B1 ** 2 / 4
    d6 = (-B0 ** 6 / 24 - B0 ** 4 / 9 - R(5, 9) * B0 ** 3 * B1 - B0 ** 3 * B2 / 9 - B0 ** 2 * B1 ** 2 / 3 + B0 ** 2 / 3
          + R(31, 36) * B0 * B1 + R(19, 24) * B0 * B2 + R(11, 36) * B0 * B3 + B0 * B4 / 24)
    return d0 + x ** 4 * d4 + x ** 6 * d6


CORRECTED = "--printed" not in sys.argv


def eeg_coeffs(b, corrected=None):
    """b = [beta~, beta~', beta~'', beta~''', beta~'''', ...] (tau derivatives). Returns f2, f3, Om2, Om3, Pi2, Pi3,
    Psi2 as printed (corrected = False), or with Pi3 + delta_Pi3 and f3 + delta_f3 (Om3 = 2 beta~ Pi3 + ... then
    carries the correction itself). Mutations act on the printed coefficients, so they reach both."""
    if corrected is None:
        corrected = CORRECTED
    B0, B1, B2, B3 = b[0], b[1], b[2], b[3]
    f2 = sp.Rational(1, 8) * (1 - x ** 2) * B0 * (4 * (1 - x ** 2) * B0 + (3 + x ** 2) * B0 ** 3
                                                 - 2 * (1 + x ** 2) * (B2 + 3 * B1))                    # (S40)
    Om2 = B0 * (x ** 2 * (B2 + 3 * B1 - B0 ** 3) - 2 * (1 - x ** 2) * B0)                             # (S41)
    Pi2 = x ** 2 / 2 * (2 * B0 + B2 + 3 * B1 - B0 ** 3)                                                # (S42)
    c11 = 10 if MUT == "Psi2" else 11
    Psi2 = sp.Rational(1, 2) * (B0 ** 3 + B0 ** 2 * B1 + x ** 2 * (-4 * B0 ** 3 - 4 * B0 ** 2 * B1 + 6 * B0 + c11 * B1
                                                                  + 6 * B2 + B3))                    # (S43)
    c24 = 23 if MUT == "Pi3" else 24
    Pi3 = (x ** 2 / 6 * (c24 * B0 ** 3 + 15 * B0 ** 2 * B1 + 3 * B0 ** 2 * B2)
           + x ** 4 / 6 * (3 * B0 ** 5 - 16 * B0 ** 3 - 20 * B0 ** 2 * B1 - 4 * B0 ** 2 * B2 + 12 * B0 + 22 * B1
                           + 12 * B2 + 2 * B3))                                                      # (S46)
    if corrected:
        Pi3 = Pi3 + delta_Pi3(b)
    c7 = -6 if MUT == "Omega3" else -7
    Om3 = (2 * B0 * Pi3 + x ** 4 / 4 * (B0 ** 6 - 4 * B0 ** 4 - 6 * B0 ** 3 * B1 - 2 * B0 ** 3 * B2 + 4 * B0 ** 2
                                       + 12 * B0 * B1 + 4 * B0 * B2 + 9 * B1 ** 2 + 6 * B1 * B2 + B2 ** 2)
           + x ** 2 * (4 * B0 ** 4 + c7 * B0 ** 2 - 10 * B0 * B1 - 4 * B0 * B2 + B1 ** 2))           # (S45)
    R = sp.Rational
    c17 = R(18, 48) if MUT == "c0" else R(17, 48)
    c0 = (-c17 * B0 ** 6 - R(79, 36) * B0 ** 4 + R(35, 72) * B0 ** 3 * B1 + R(13, 72) * B0 ** 3 * B2
          + R(11, 12) * B0 ** 2 + R(7, 9) * B0 * B1 + R(1, 6) * B0 * B2 - R(1, 9) * B0 * B3 - R(5, 8) * B1 ** 2
          - R(1, 4) * B1 * B2 - R(1, 24) * B2 ** 2)                                                  # (S48)
    c2 = R(3, 16) * B0 ** 6 + R(3, 4) * B0 ** 4 - R(3, 8) * B0 ** 3 * B1 - R(1, 8) * B0 ** 3 * B2      # (S49)
    c4 = (R(1, 16) * B0 ** 6 + R(9, 4) * B0 ** 4 + R(7, 8) * B0 ** 3 * B1 + R(1, 8) * B0 ** 3 * B2
          - R(7, 4) * B0 ** 2 - R(5, 2) * B0 * B1 - B0 * B2 + R(1, 4) * B1 ** 2)                      # (S50)
    c6 = (R(5, 48) * B0 ** 6 - R(29, 36) * B0 ** 4 - R(71, 72) * B0 ** 3 * B1 - R(13, 72) * B0 ** 3 * B2
          + R(5, 6) * B0 ** 2 + R(31, 18) * B0 * B1 + R(5, 6) * B0 * B2 + R(1, 9) * B0 * B3 + R(3, 8) * B1 ** 2
          + R(1, 4) * B1 * B2 + R(1, 24) * B2 ** 2)                                                  # (S51)
    f3 = c0 + x ** 2 * c2 + x ** 4 * c4 + x ** 6 * c6                                                 # (S44)
    if corrected:
        f3 = f3 + delta_f3(b)
    return dict(f2=f2, f3=f3, Om2=Om2, Om3=Om3, Pi2=Pi2, Pi3=Pi3, Psi2=Psi2)


def verdict(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    print(f"   {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def check_A():
    print("A. EEG's NLO and NNLO against Proposition 1's outer expansion (e = 1/n)", flush=True)
    S, D, G, B = outer(3)
    print(f"   outer expansion done [{time.time() - T0:.0f} s]", flush=True)
    # beta~ as a series in e, to e^2 (EEG need beta~ to e^2 for f, Omega to e^3)
    epsE = e if MUT == "map" else e / (1 + e)
    sq = sp.series(sp.sqrt(2 * (1 + e)), e, 0, 4).removeO()          # sqrt(2 (1 + e))
    Fn = [Fd[j] + e * K1d[j] / 2 + e ** 2 * K2d[j] / 2 for j in range(NDER)]
    om = w0 + w1 * e + w2 * e ** 2
    # tau derivatives: d/dtau = omega d/dtheta on the derivative chains
    bt = [trunc(sq * Fn[0], 3)]
    for j in range(4):
        bt.append(trunc(om * dth(bt[-1]), 3))
    co = eeg_coeffs(bt)
    eE = sp.series(epsE, e, 0, 5).removeO()
    # EEG's fields, re-expanded in e
    f_eeg = trunc(1 - eE * (1 - x ** 2) / 2 * bt[0] ** 2 + eE ** 2 * co["f2"] + eE ** 3 * co["f3"], 3)
    Om_eeg = trunc(eE * bt[0] ** 2 + eE ** 2 * co["Om2"] + eE ** 3 * co["Om3"], 3)
    # Pi / sqrt(e) and Psi / sqrt(e): eps_E^(1/2) = e^(1/2) (1 + e)^(-1/2) (map: e^(1/2))
    rt = sp.Integer(1) if MUT == "map" else sp.series((1 + e) ** sp.Rational(-1, 2), e, 0, 4).removeO()
    Pi_eeg = trunc(rt * (bt[0] + eE * co["Pi2"] + eE ** 2 * co["Pi3"]), 2)
    Psi_eeg = trunc(rt * (eE * (bt[1] + bt[0]) + eE ** 2 * co["Psi2"]), 2)
    print(f"   EEG's series re-expanded [{time.time() - T0:.0f} s]", flush=True)
    # ours
    c_our = 1 - sum(e ** k * G[k] for k in range(1, 4))
    Om_our = trunc((1 / e - 1) * sum(e ** k * B[k] for k in range(2, 5)) / x ** 2, 3)
    Pi_our = trunc(sum(e ** k * D[k] for k in range(0, 3)) / (sp.sqrt(2) * x), 2)
    Psi_our = trunc(sum(e ** k * S[k] for k in range(1, 3)) / (sp.sqrt(2) * x ** 2), 2)
    for name, a, b, N in (("f = c", f_eeg, c_our, 3), ("Omega = u/x^2", Om_eeg, Om_our, 3),
                          ("Pi = y/(sqrt2 x), / sqrt(e)", Pi_eeg, Pi_our, 2),
                          ("Psi = (p+m)/(sqrt2 x^2), / sqrt(e)", Psi_eeg, Psi_our, 2)):
        for k in range(N + 1):
            d = sp.simplify(sp.expand(a.coeff(e, k) - b.coeff(e, k)))
            nterms = len(sp.Add.make_args(sp.expand(b.coeff(e, k))))
            verdict(f"{name}, order e^{k}", d == 0, f"({nterms} terms in ours)" if d == 0 else f"difference {sp.factor(d)}")


def check_B():
    """EEG's series in their own equations (S36)-(S39), eps symbolic, beta~ a generic function of tau"""
    print("B. EEG's series in their equations (S36)-(S39)", flush=True)
    ep = sp.symbols("epsilon", positive=True)
    tau = sp.symbols("tau")
    bfun = sp.Function("b")(tau)
    bt = [bfun] + [sp.diff(bfun, tau, k) for k in range(1, 5)]
    co = eeg_coeffs(bt)
    f = 1 - ep * (1 - x ** 2) / 2 * bt[0] ** 2 + ep ** 2 * co["f2"] + ep ** 3 * co["f3"]
    Om = ep * bt[0] ** 2 + ep ** 2 * co["Om2"] + ep ** 3 * co["Om3"]
    Pi = sp.sqrt(ep) * (bt[0] + ep * co["Pi2"] + ep ** 2 * co["Pi3"])
    Psi3 = sp.Function("Psi3")(tau, x)                       # "not needed": kept unknown
    Psi = sp.sqrt(ep) * (ep * (bt[1] + bt[0]) + ep ** 2 * co["Psi2"] + ep ** 3 * Psi3)
    Dt = lambda g: sp.diff(g, tau) + x * sp.diff(g, x)
    eqs = {
        "S36": x * sp.diff(Om, x) + x ** 2 * Om ** 2 - ((1 / ep) * ((1 - 2 * ep) * Pi ** 2 - Om)
                                                       + Om * (Pi ** 2 * x ** 2 + x ** 4 * Psi ** 2) + (1 / ep - 2) * x ** 2 * Psi ** 2),
        "S37": sp.diff(f, x) - x * f * Om,
        "S38": Dt(Pi) - (f * x * sp.diff(Psi, x) + f * (1 / ep + x ** 2 * Om) * Psi - Pi),
        "S39": Dt(Psi) - (f * sp.diff(Pi, x) / x + f * Om * Pi - 2 * Psi),
    }
    # the last order each equation fixes with the printed terms: (S36) eps^2 (eps^3 needs Pi_4, Omega_4), (S37)
    # eps^3, (S38) eps^(3/2) (eps^(5/2) defines Psi_3), (S39) eps^(5/2)
    hi = {"S36": 2, "S37": 3, "S38": sp.Rational(3, 2), "S39": sp.Rational(5, 2)}
    s = sp.symbols("s", positive=True)                     # eps = s^2
    for name, ex in eqs.items():
        exs = sp.expand(sp.expand(ex.subs(ep, s ** 2)) * s ** 4)
        ok, first = True, None
        for p in range(0, int(2 * hi[name]) + 5):
            cf = sp.simplify(exs.coeff(s, p))
            if cf != 0:
                ok, first = False, (sp.Rational(p - 4, 2), cf)
                break
        verdict(f"({name}) vanishes through eps^{hi[name]}", ok,
                f"first nonzero at eps^{first[0]}: {sp.factor(first[1])}" if not ok else "")


def check_C():
    print("C. The gauge map Delta = 4 pi mu/(1 - <Omega(1)>), in eps_E", flush=True)
    ep = sp.symbols("epsilon")
    D0, D1, D2 = sp.symbols("Delta0 Delta1 Delta2")
    N0, N01, W2 = sp.symbols("N0 N01 W2")   # <bt0^2>, <2 bt0 bt1>, <Omega_2(1)>
    mu = (D0 + D1 * ep + D2 * ep ** 2) * (1 - (ep * (N0 + ep * N01) + ep ** 2 * W2)) / (4 * sp.pi)
    mus = [sp.expand(sp.expand(mu).coeff(ep, k)) for k in range(3)]
    print(f"   4 pi mu_0 = {sp.factor(4 * sp.pi * mus[0])}")
    print(f"   4 pi mu_1 = {sp.expand(4 * sp.pi * mus[1])}   (EEG (S59): N0 = (Delta1 - 4 pi mu1)/(4 pi mu0))")
    print(f"   4 pi mu_2 = {sp.expand(4 * sp.pi * mus[2])}")
    ok = sp.simplify((D1 - 4 * sp.pi * mus[1]) / (4 * sp.pi * mus[0]) - N0) == 0
    verdict("EEG's (S59) N0 = (Delta1 - 4 pi mu1)/(4 pi mu0) = <beta~_0^2>", ok)
    # <Omega_2(1)> from (S41) at x = 1, for a periodic beta~: <b b''> = -<b'^2>, <b b'> = 0
    tau = sp.symbols("tau")
    b = sp.Function("b")(tau)
    Om2_1 = eeg_coeffs([b] + [b.diff(tau, k) for k in range(1, 5)])["Om2"].subs(x, 1)
    print(f"   Omega_2(tau, 1) = {sp.expand(Om2_1)}  ->  <Omega_2(1)> = -<beta~'^2> - <beta~^4>")
    # numbers: N0 = <beta~_0^2> = 2 <F^2> of the tuned D = infinity profile (solver/eeg_profile/tuned_profile.py,
    # data/tuned_profile_A2.71918.txt), Delta_0 = 1.643893746 (README), Delta_1 = 10.37528 (first_order_fits.py)
    import os
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    txt = open(os.path.join(here, "..", "eeg_profile", "data", "tuned_profile_A2.71918.txt")).read()
    N0v = float(re.search(r"<bt\^2> = N0 = ([0-9.]+)", txt).group(1))
    D0v, D1v = 1.643893746, 10.37528
    mu0 = D0v / (4 * float(sp.pi))
    mu1 = (D1v - D0v * N0v) / (4 * float(sp.pi))
    print(f"   numbers: N0 = {N0v} (tuned profile), mu_0 = Delta_0/(4 pi) = {mu0:.7f} (EEG fit 0.130818), "
          f"mu_1 = (Delta_1 - Delta_0 N0)/(4 pi) = {mu1:.5f} (EEG fit -0.1449)")


def check_D():
    """EEG's printed NNLO in their own (S39) at eps^(5/2): the residual R, and the correction it requires,
    (1/x) d_x delta = R (Pi_3 enters (S39) only through f (1/x) d_x Pi at that order, f = 1 + O(eps))"""
    print("D. The printed NNLO in (S39) at eps^(5/2), and the correction it requires", flush=True)
    b = sp.symbols("b0:7")
    co = eeg_coeffs(list(b), corrected=False)
    ep = sp.symbols("epsilon", positive=True)
    dtau = lambda ex: sp.expand(sum(sp.diff(ex, b[j]) * b[j + 1] for j in range(6)))
    B0 = b[0]
    f = 1 - ep * (1 - x ** 2) / 2 * B0 ** 2 + ep ** 2 * co["f2"] + ep ** 3 * co["f3"]
    Om = ep * B0 ** 2 + ep ** 2 * co["Om2"] + ep ** 3 * co["Om3"]
    P = B0 + ep * co["Pi2"] + ep ** 2 * co["Pi3"]                   # Pi / sqrt(eps)
    Q = ep * (b[1] + B0) + ep ** 2 * co["Psi2"]                       # Psi / sqrt(eps)
    R = sp.expand(dtau(Q) + x * sp.diff(Q, x) - (f * sp.diff(P, x) / x + f * Om * P - 2 * Q))
    R2 = sp.expand(R.coeff(ep, 2))
    print(f"   printed: residual at eps^(5/2) = {sp.collect(R2, x)}")
    verdict("the printed NNLO violates (S39) at eps^(5/2)", R2 != 0)
    dl = sp.expand(delta_Pi3(list(b)))
    verdict("(1/x) d_x delta_Pi3 equals that residual", sp.expand(sp.diff(dl, x) / x - R2) == 0)
    Pi3c = sp.expand(co["Pi3"] + dl)
    print(f"   corrected Pi_3 = {sp.collect(Pi3c, x, sp.factor)}")
    co_c = eeg_coeffs(list(b), corrected=True)
    for k in range(0, 8, 2):
        print(f"   corrected c_{k} = {sp.expand(co_c['f3'].coeff(x, k))}")


def report_printed():
    """with --printed: which NNLO coefficients differ from ours (the comparison of check A on the printed text)"""
    print("Printed NNLO (S44)-(S51) as printed; check A and B then fail where the printed Pi_3 enters", flush=True)


if __name__ == "__main__":
    print(f"mutation: {MUT}" if MUT else "no mutation", flush=True)
    print("NNLO:", "with the correction of check D (default)" if CORRECTED else "as printed (--printed)", flush=True)
    if not CORRECTED:
        report_printed()
    check_A()
    check_B()
    check_C()
    if CORRECTED:
        check_D()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  [{time.time() - T0:.0f} s]")
    sys.exit(1 if FAILS else 0)
