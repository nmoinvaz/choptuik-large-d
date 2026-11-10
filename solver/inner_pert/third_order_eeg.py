"""Ecker, Ecker and Grumiller's large-D expansion one order past their NNLO (arXiv:2611.07007v1, supplement
"Simplified large-D perturbation theory"): Psi_3, Pi_4, Omega_4 and f_4 against this repository's outer 1/n expansion
(README Proposition 1) carried one order further, in sympy. Issue agent-comms#40.

The new terms are typed below as closed forms in beta~ and its tau derivatives (b_k). They were derived from EEG's
(S36)-(S39) in solver/independent_eeg_nnlo/psi3.py, which this script does not import. The NLO and the corrected NNLO
come from second_order_eeg.py (its eeg_coeffs, corrected = True), the map and the comparison follow its check A:
    f = c,   x^2 Omega = (n - 1)(b - 1),   Pi = y/(sqrt 2 x),   Psi = (p + m)/(sqrt 2 x^2),
    eps_E = 1/(D - 1) = e/(1 + e),   e = 1/n,   beta~ = sqrt(2 (1 + e)) F_n,   F_n = F + sum_k e^k K_k/2,
    d/dtau = omega d/dtheta,   omega = w0 + w1 e + w2 e^2 + w3 e^3.
Proposition 1's recursion is second_order_eeg.outer written again with K_3 and w_3 symbolic, run one order further
(NO = 4: S_3, D_3, G_4, B_5).

Checks (exit status 1 if any fails):
  O  this script's outer recursion equals second_order_eeg.outer(3) at every order the latter computes.
  A  EEG's f and Omega to e^4 and Pi/sqrt(e), Psi/sqrt(e) to e^3, with the new terms, re-expanded in e with the map,
     equal Proposition 1's c, u/x^2, y/(sqrt 2 x) and (p + m)/(sqrt 2 x^2) coefficient by coefficient. The orders
     e^4 (f, Omega) and e^3 (Pi, Psi) are the ones the new terms enter; the lower ones repeat second_order_eeg.py.
  N  each new term enters its order: with it set to zero, the comparison at that order fails.
  I  the NLO and NNLO taken from second_order_eeg.py equal those of solver/independent_eeg_nnlo/check.py (its claimed
     set), the other transcription. check.py is read only for this comparison.
Mutations (each must make the run exit 1): --mutate=Psi3, Pi4, Omega4, f4 (one coefficient of a new term, as in
psi3.py), K3 (beta~ without K_3, a map error at the new order only), map (eps_E = e instead of e/(1 + e)), Psi2
(second_order_eeg's mutation of (S43), 11 -> 10, which check I must catch too).
--controls runs every mutation in a subprocess and compares its exit and failing (field, order) pairs with CONTROLS,
written before the first run; it exits 0 only if every row matches.

    python3 third_order_eeg.py [--mutate=NAME | --controls]
"""
import os
import sys
import time

import sympy as sp

T0 = time.time()
ARGS = [a for a in sys.argv[1:]]
CONTROLS_MODE = ARGS == ["--controls"]
MUTS = {'Psi3', 'Pi4', 'Omega4', 'f4', 'K3', 'map', 'Psi2'}
MUT = next((a.split("=", 1)[1] for a in ARGS if a.startswith("--mutate=")), None)
if MUT is not None and MUT not in MUTS:
    sys.exit(f"unknown mutation {MUT}; choose from {', '.join(sorted(MUTS))}")
if not CONTROLS_MODE and any(not a.startswith("--mutate=") for a in ARGS):
    sys.exit(__doc__)
sys.argv = sys.argv[:1]                  # second_order_eeg reads sys.argv at import; it must see no flags
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import second_order_eeg as soe  # noqa: E402

soe.MUT = "Psi2" if MUT == "Psi2" else None     # Psi2: second_order_eeg's own mutation of (S43), 11 -> 10
FAILS = []
FAILED_ORDERS = []      # (field, power of e) of every failed comparison in check A

# (mutation, expected exit, expected failed (field, order) pairs of check A), written before the first run.
CONTROLS = [
    (None, 0, set()),
    ("Psi3", 1, {("Psi", 3)}),
    ("Pi4", 1, {("Pi", 3)}),
    ("Omega4", 1, {("Om", 4)}),
    ("f4", 1, {("f", 4)}),
    # K_3 enters beta~ at e^3: Pi/sqrt(e) at e^3, and f, Omega through eps_E beta~^2 at e^4. Psi/sqrt(e) needs
    # beta~ only to e^2.
    ("K3", 1, {("Pi", 3), ("f", 4), ("Om", 4)}),
    ("map", 1, {("f", 2), ("f", 3), ("f", 4), ("Om", 2), ("Om", 3), ("Om", 4), ("Pi", 1), ("Pi", 2), ("Pi", 3),
                ("Psi", 2), ("Psi", 3)}),
    # A lower-order input changed: Psi_2 enters Psi/sqrt(e) at e^2 and, through eps_E^2, at e^3. Check I fails too.
    ("Psi2", 1, {("Psi", 2), ("Psi", 3)}),
]

x, e = soe.x, soe.e
w0, w1, w2, w3 = soe.w0, soe.w1, soe.w2, soe.w3
NDER = 14
Fd = sp.symbols(f"F0:{NDER}")
K1d = sp.symbols(f"Ka0:{NDER}")
K2d = sp.symbols(f"Kb0:{NDER}")
K3d = sp.symbols(f"Kc0:{NDER}")
CHAINS = [Fd, K1d, K2d, K3d]
b = sp.symbols("b0:8")                   # b_k = k-th tau derivative of beta~, for the closed forms
R = sp.Rational


def bump(name, k=1):
    return k if MUT == name else 0


def dth(expr):
    """d/dtheta on polynomials in the derivative symbols. The top symbol of each chain must be absent."""
    expr = sp.expand(expr)
    out = 0
    for ch in CHAINS:
        assert not expr.has(ch[NDER - 1]), "derivative chain too short"
        for j in range(NDER - 1):
            if expr.has(ch[j]):
                out += sp.diff(expr, ch[j]) * ch[j + 1]
    return sp.expand(out)


# ------------------------------------------------------------------ Proposition 1, one order further
def outer(NO=4):
    """second_order_eeg.outer with K_3, w_3 symbolic and longer derivative chains"""
    N = NO + 2
    W = [w0, w1, w2, w3][:N] + [0] * max(0, N - 4)
    Ks = [0, K1d[0], K2d[0], K3d[0]] + [0] * N
    S = [sp.Integer(0)] * N
    D = [2 * x * Fd[0]] + [sp.Integer(0)] * (N - 1)
    G = [sp.Integer(0)] * N
    B = [sp.Integer(0)] * N

    def mul(a, c):
        return [sp.expand(sum(a[i] * c[k - i] for i in range(k + 1))) for k in range(N)]

    def add(*ss):
        return [sp.expand(sum(q[k] for q in ss)) for k in range(N)]

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
        print(f"   outer order {k}: B_{k + 1}, G_{k} [{time.time() - T0:.0f} s]", flush=True)
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
        print(f"   outer order {k}: S_{k}, D_{k} [{time.time() - T0:.0f} s]", flush=True)
    return S, D, G, B


# ------------------------------------------------------------------ the new terms (closed forms)
def new_terms(bb):
    """Psi_3, Pi_4, Omega_4, f_4 as polynomials in bb = [beta~, beta~', ...] and x (the sympy output of
    psi3.py --print, grouped by powers of x). The bump terms are the mutations."""
    b0, b1, b2, b3, b4, b5, b6 = bb[:7]
    S3 = (
        (-b0**5/8 - b0**4*b1/8 - b0**3/2 + b0**2*b1/4 + b0**2*b2/4 + 3*b0*b1**2/4 + b0*b1*b2/4)
        + x**2 * (-b0**5 - b0**4*b1 + 23*b0**3/2 + 75*b0**2*b1/4 + 5*b0**2*b2 + 3*b0**2*b3/4 + 11*b0*b1**2/2 +
            2*b0*b1*b2 - 6*b0 + b1**3/2 - 11*b1 - 6*b2 - b3)
        + x**4 * (3*b0**5 + 3*b0**4*b1 - 17*b0**3 - 38*b0**2*b1 - 47*b0**2*b2/4 - 5*b0**2*b3/4 - 63*b0*b1**2/4 -
            17*b0*b1*b2/4 + 15*b0 - b1**3 + 137*b1/4 + 225*b2/8 + 85*b3/8 + 15*b4/8 + b5/8
            + bump('Psi3') * b0**2*b2))
    P4 = (
        x**2 * (-b0**3 - b0**2*b1/2 + 3*b0**2*b2/8 + b0**2*b3/8 + 17*b0*b1**2/8 + 2*b0*b1*b2 + b0*b1*b3/8 +
            b0*b2**2/4 + 3*b1**3/8 + b1**2*b2/8)
        + x**4 * (-5*b0**5 - 6*b0**4*b1 - b0**4*b2 - 2*b0**3*b1**2 + 63*b0**3/4 + 69*b0**2*b1/2 + 107*b0**2*b2/8
            + 21*b0**2*b3/8 + b0**2*b4/4 + 117*b0*b1**2/8 + 29*b0*b1*b2/4 + 7*b0*b1*b3/8 + b0*b2**2/2 - 6*b0 +
            15*b1**3/8 + 7*b1**2*b2/8 - 25*b1/2 - 35*b2/4 - 5*b3/2 - b4/4)
        + x**6 * (-5*b0**7/16 + 139*b0**5/24 + 485*b0**4*b1/48 + 71*b0**4*b2/48 + 17*b0**3*b1**2/6 - 121*b0**3/6
            - 1289*b0**2*b1/24 - 1075*b0**2*b2/48 - 17*b0**2*b3/4 - 5*b0**2*b4/16 - 359*b0*b1**2/12 -
            173*b0*b1*b2/12 - 9*b0*b1*b3/8 - 7*b0*b2**2/8 + 15*b0 - 29*b1**3/8 - 29*b1**2*b2/24 + 147*b1/4 +
            203*b2/6 + 245*b3/16 + 175*b4/48 + 7*b5/16 + b6/48 + bump('Pi4', R(1, 48)) * b6))
    O4 = (
        x**2 * (bump('Omega4') * b0**4
            - 23*b0**4 - 15*b0**3*b1 - 13*b0**3*b2/4 + b0**3*b3/4 + 5*b0**2*b1**2/4 + 4*b0**2*b1*b2 +
            b0**2*b1*b3/4 + b0**2*b2**2/2 + 12*b0**2 + 3*b0*b1**3/4 + b0*b1**2*b2/4 + 16*b0*b1 + 8*b0*b2 - 4*b1**2)
        + x**4 * (-18*b0**6 - 14*b0**5*b1 - 5*b0**5*b2/2 - 9*b0**4*b1**2/2 + 147*b0**4/2 + 140*b0**3*b1 +
            179*b0**3*b2/4 + 21*b0**3*b3/4 + b0**3*b4/2 + 177*b0**2*b1**2/4 + 18*b0**2*b1*b2 + 7*b0**2*b1*b3/4 +
            3*b0**2*b2**2/2 - 48*b0**2 + 21*b0*b1**3/4 + 9*b0*b1**2*b2/4 - 101*b0*b1 - 70*b0*b2 - 19*b0*b3 - 2*b0*b4
            - 5*b1**2/2 - 3*b1*b2 + b1*b3 - 3*b2**2/2)
        + x**6 * (-b0**8 + 46*b0**6/3 + 79*b0**5*b1/3 + 13*b0**5*b2/3 + 20*b0**4*b1**2/3 - 148*b0**4/3 -
            398*b0**3*b1/3 - 325*b0**3*b2/6 - 39*b0**3*b3/4 - 3*b0**3*b4/4 - 461*b0**2*b1**2/6 - 221*b0**2*b1*b2/6 -
            9*b0**2*b1*b3/4 - 11*b0**2*b2**2/4 + 36*b0**2 - 41*b0*b1**3/4 - 41*b0*b1**2*b2/12 + 95*b0*b1 +
            953*b0*b2/12 + 265*b0*b3/8 + 181*b0*b4/24 + 7*b0*b5/8 + b0*b6/24 + 75*b1**2/4 + 155*b1*b2/8 + 15*b1*b3/4
            + 3*b1*b4/8 + 35*b2**2/8 + 5*b2*b3/4 + b2*b4/8))
    f4 = (
        (35*b0**8/128 + 27*b0**6/16 - 163*b0**5*b1/96 - 29*b0**5*b2/96 - b0**4*b1**2/8 - b0**4/2 - 10*b0**3*b1/3
            + 5*b0**3*b2/16 + 47*b0**3*b3/96 + b0**3*b4/32 + 241*b0**2*b1**2/96 + 11*b0**2*b1*b2/12 -
            7*b0**2*b1*b3/96 + b0**2*b2**2/48 + b0**2/2 + 7*b0*b1**3/32 - b0*b1**2*b2/96 + 23*b0*b1/24 - 25*b0*b2/96
            - 187*b0*b3/192 - 39*b0*b4/64 - 7*b0*b5/64 - b0*b6/192 - 89*b1**2/96 - 123*b1*b2/64 - 61*b1*b3/96 -
            3*b1*b4/64 - 19*b2**2/64 - 5*b2*b3/32 - b2*b4/64)
        + x**2 * (bump('f4') * b0**8
            - 5*b0**8/32 - 11*b0**6/12 + 31*b0**5*b1/48 + 7*b0**5*b2/48 + b0**4*b1**2/24 - 5*b0**4/24 +
            17*b0**3*b1/24 - b0**3*b2/16 - 5*b0**3*b3/24 - b0**3*b4/48 - 5*b0**2*b1**2/16 - b0**2*b1*b2/8 -
            b0**2*b2**2/48)
        + x**4 * (-3*b0**8/64 - 5*b0**6/8 - b0**5*b1/8 - b0**4*b1**2/8 - 33*b0**4/8 - 5*b0**3*b1/2 -
            5*b0**3*b2/16 + b0**3*b3/16 - 3*b0**2*b1**2/8 + 5*b0**2*b1*b2/8 + b0**2*b1*b3/16 + b0**2*b2**2/16 +
            3*b0**2 + 3*b0*b1**3/16 + b0*b1**2*b2/16 + 4*b0*b1 + 2*b0*b2 - b1**2)
        + x**6 * (-b0**8/32 - 17*b0**6/12 - 17*b0**5*b1/16 - 7*b0**5*b2/48 - 11*b0**4*b1**2/24 + 247*b0**4/24 +
            481*b0**3*b1/24 + 283*b0**3*b2/48 + 2*b0**3*b3/3 + b0**3*b4/16 + 117*b0**2*b1**2/16 + 23*b0**2*b1*b2/8 +
            7*b0**2*b1*b3/24 + 11*b0**2*b2**2/48 - 8*b0**2 + 7*b0*b1**3/8 + 3*b0*b1**2*b2/8 - 101*b0*b1/6 -
            35*b0*b2/3 - 19*b0*b3/6 - b0*b4/3 - 5*b1**2/12 - b1*b2/2 + b1*b3/6 - b2**2/4)
        + x**8 * (-5*b0**8/128 + 61*b0**6/48 + 215*b0**5*b1/96 + 29*b0**5*b2/96 + 2*b0**4*b1**2/3 - 131*b0**4/24
            - 179*b0**3*b1/12 - 35*b0**3*b2/6 - 97*b0**3*b3/96 - 7*b0**3*b4/96 - 877*b0**2*b1**2/96 -
            103*b0**2*b1*b2/24 - 9*b0**2*b1*b3/32 - 7*b0**2*b2**2/24 + 9*b0**2/2 - 41*b0*b1**3/32 -
            41*b0*b1**2*b2/96 + 95*b0*b1/8 + 953*b0*b2/96 + 265*b0*b3/64 + 181*b0*b4/192 + 7*b0*b5/64 + b0*b6/192 +
            75*b1**2/32 + 155*b1*b2/64 + 15*b1*b3/32 + 3*b1*b4/64 + 35*b2**2/64 + 5*b2*b3/32 + b2*b4/64))

    return dict(S3=S3, P4=P4, O4=O4, f4=f4)


# ------------------------------------------------------------------ truncated series in e
def tmul(a, c, N):
    return [sp.expand(sum(a[i] * c[k - i] for i in range(k + 1))) for k in range(N + 1)]


class SerEval:
    """Evaluates polynomials in b_0..b_7 (coefficients in x) on the e-series of beta~'s tau derivatives, truncated
    at e^N, with cached powers."""

    def __init__(self, ser):
        self.ser = ser
        self.cache = {}

    def power(self, k, p, N):
        key = (k, p, N)
        if key not in self.cache:
            base = self.ser[k][:N + 1] + [0] * max(0, N + 1 - len(self.ser[k]))
            self.cache[key] = base if p == 1 else tmul(self.power(k, p - 1, N), base, N)
        return self.cache[key]

    def __call__(self, expr, N):
        expr = sp.expand(expr)
        out = [sp.Integer(0)] * (N + 1)
        if expr == 0:
            return out
        P = sp.Poly(expr, *b)
        for monom, coeff in P.terms():
            prod = [sp.Integer(1)] + [sp.Integer(0)] * N
            for k, p in enumerate(monom):
                if p:
                    prod = tmul(prod, self.power(k, p, N), N)
            for j in range(N + 1):
                out[j] += coeff * prod[j]
        return [sp.expand(v) for v in out]


def ser_of(expr, N):
    """e-coefficients of an expression analytic in e at 0"""
    s_ = sp.series(expr, e, 0, N + 1).removeO()
    return [sp.expand(s_.coeff(e, k)) for k in range(N + 1)]


def verdict(name, ok, detail=""):
    if not ok:
        FAILS.append(name)
    print(f"   {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def check_O(S, D, G, B):
    print("O. this recursion against second_order_eeg.outer(3)", flush=True)
    S3, D3, G3, B3 = soe.outer(3)
    sub = {soe.Fd[j]: Fd[j] for j in range(soe.NDER)}
    sub.update({soe.K1d[j]: K1d[j] for j in range(soe.NDER)})
    sub.update({soe.K2d[j]: K2d[j] for j in range(soe.NDER)})
    ok = True
    for name, mine, theirs, ks in (("S", S, S3, range(1, 3)), ("D", D, D3, range(0, 3)), ("G", G, G3, range(1, 4)),
                                   ("B", B, B3, range(2, 5))):
        for k in ks:
            ok &= sp.expand(mine[k] - theirs[k].xreplace(sub)) == 0
    verdict("S_1..2, D_0..2, G_1..3, B_2..4 equal second_order_eeg.outer(3)", ok)


def eeg_series(drop=None):
    """EEG's f, Omega (to e^4) and Pi/sqrt(e), Psi/sqrt(e) (to e^3) re-expanded in e. drop names a new term set to
    zero (check N)."""
    NE = 4
    epsE = e if MUT == "map" else e / (1 + e)
    eE = ser_of(epsE, NE)
    sq = ser_of(sp.sqrt(2 * (1 + e)), NE)
    K3on = 0 if MUT == "K3" else 1
    Fn = [[Fd[j], K1d[j] / 2, K2d[j] / 2, K3on * K3d[j] / 2, 0] for j in range(NDER)]
    om = [w0, w1, w2, w3, 0]
    bt = [tmul(sq, Fn[0], NE)]
    for j in range(7):
        bt.append(tmul(om, [dth(v) for v in bt[-1]], NE))
    ev = SerEval(bt)
    co = soe.eeg_coeffs(list(b), corrected=True)
    nw = new_terms(list(b))
    if drop:
        nw[drop] = sp.Integer(0)
    eEp = [[sp.Integer(1)] + [0] * NE]
    for k in range(4):
        eEp.append(tmul(eEp[-1], eE, NE))

    def series_sum(terms, N):
        """sum over (power of eps_E, polynomial in b) of eps_E^p * poly, to e^N"""
        out = [sp.Integer(0)] * (N + 1)
        for p, poly in terms:
            v = tmul(eEp[p][:N + 1], ev(poly, N - p) + [0] * p, N)
            out = [sp.expand(out[i] + v[i]) for i in range(N + 1)]
        return out
    f = series_sum([(0, 1), (1, -(1 - x ** 2) / 2 * b[0] ** 2), (2, co["f2"]), (3, co["f3"]), (4, nw["f4"])], NE)
    Om = series_sum([(1, b[0] ** 2), (2, co["Om2"]), (3, co["Om3"]), (4, nw["O4"])], NE)
    rt = [sp.Integer(1), 0, 0, 0] if MUT == "map" else ser_of((1 + e) ** R(-1, 2), 3)
    Pi = tmul(rt, series_sum([(0, b[0]), (1, co["Pi2"]), (2, co["Pi3"]), (3, nw["P4"])], 3), 3)
    Psi = tmul(rt, series_sum([(1, b[1] + b[0]), (2, co["Psi2"]), (3, nw["S3"])], 3), 3)
    return dict(f=f, Om=Om, Pi=Pi, Psi=Psi)


def ours(S, D, G, B):
    c = [sp.Integer(1)] + [sp.expand(-G[k]) for k in range(1, 5)]
    # x^2 Omega = (1/e - 1)(b - 1), b - 1 = sum_{k>=2} e^k B_k
    Om = [sp.Integer(0)] + [sp.expand((B[k + 1] - B[k]) / x ** 2) for k in range(1, 5)]
    Pi = [sp.expand(D[k] / (sp.sqrt(2) * x)) for k in range(4)]
    Psi = [sp.Integer(0)] + [sp.expand(S[k] / (sp.sqrt(2) * x ** 2)) for k in range(1, 4)]
    return dict(f=c, Om=Om, Pi=Pi, Psi=Psi)


NAMES = dict(f="f = c", Om="Omega = u/x^2", Pi="Pi = y/(sqrt2 x), / sqrt(e)", Psi="Psi = (p+m)/(sqrt2 x^2), / sqrt(e)")
TOP = dict(f=4, Om=4, Pi=3, Psi=3)


def check_A(our):
    print("A. EEG's series with the new terms against Proposition 1 (e = 1/n)", flush=True)
    eeg = eeg_series()
    print(f"   EEG's series re-expanded [{time.time() - T0:.0f} s]", flush=True)
    for key in ("f", "Om", "Pi", "Psi"):
        for k in range(TOP[key] + 1):
            d = sp.expand(eeg[key][k] - our[key][k])
            nterms = len(sp.Add.make_args(our[key][k]))
            new = " (new order)" if k == TOP[key] else ""
            verdict(f"{NAMES[key]}, order e^{k}{new}", d == 0,
                    f"({nterms} terms in ours)" if d == 0 else f"difference has {len(sp.Add.make_args(d))} terms")
            if d != 0:
                FAILED_ORDERS.append((key, k))


def check_I():
    print("I. the lower orders against check.py's claimed set", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "independent_eeg_nnlo"))
    import check as C
    t = C.build_set("claimed")
    co = soe.eeg_coeffs(list(C.b), corrected=True)
    pairs = dict(f2="f2", O2="Om2", P2="Pi2", S2="Psi2", f3="f3", O3="Om3", P3="Pi3")
    same = [k for k, v in pairs.items() if sp.expand(t[k] - co[v].subs(x, C.x)) == 0]
    verdict("f2, Omega2, Pi2, Psi2, f3, Omega3, Pi3 equal check.py's", len(same) == len(pairs),
            "" if len(same) == len(pairs) else f"equal only {same}")


def check_N(our):
    print("N. each new term is needed at its order", flush=True)
    for drop, key in (("S3", "Psi"), ("P4", "Pi"), ("O4", "Om"), ("f4", "f")):
        eeg = eeg_series(drop=drop)
        d = sp.expand(eeg[key][TOP[key]] - our[key][TOP[key]])
        verdict(f"{drop} = 0 breaks {NAMES[key]} at e^{TOP[key]}", d != 0)


def run_controls():
    """each CONTROLS row in a subprocess; its last 'failed orders:' line gives the failed pairs"""
    import ast
    import subprocess
    bad = 0
    for mut, rc_exp, exp in CONTROLS:
        cmd = [sys.executable, os.path.abspath(__file__)] + ([f"--mutate={mut}"] if mut else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [q for q in p.stdout.splitlines() if q.startswith("failed orders:")]
        got = set(ast.literal_eval(line[-1].split(":", 1)[1].strip())) if line else None
        ok = p.returncode == rc_exp and got == exp
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} --mutate={mut}  exit {p.returncode} (expected {rc_exp})  "
              f"failed {sorted(got) if got is not None else '?'}" + ("" if ok else f"  expected {sorted(exp)}"),
              flush=True)
    print(f"\n{len(CONTROLS) - bad} of {len(CONTROLS)} controls as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    if CONTROLS_MODE:
        sys.exit(run_controls())
    print(f"mutation: {MUT}" if MUT else "no mutation", flush=True)
    S, D, G, B = outer(4)
    print(f"   outer expansion done [{time.time() - T0:.0f} s]", flush=True)
    check_O(S, D, G, B)
    our = ours(S, D, G, B)
    check_A(our)
    if MUT is None:
        check_N(our)
    check_I()
    print(f"failed orders: {sorted(FAILED_ORDERS)}")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  [{time.time() - T0:.0f} s]")
    sys.exit(1 if FAILS else 0)
