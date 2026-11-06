#!/usr/bin/env python3
"""Painleve test (Weiss-Tabor-Carnevale, with Kruskal's reduced ansatz) of the D = infinity inner problem, with controls.

The test (Ablowitz, Ramani, Segur, J. Math. Phys. 21 (1980) 715; Weiss, Tabor, Carnevale, J. Math. Phys. 24 (1983) 522;
Kruskal's simplification psi = x - f(t), as in Jimbo, Kruskal, Miwa, Phys. Lett. A 92 (1982) 59):

    u_i = sum_{j >= 0} a_{i,j}(t) psi^(p_i + j),      psi = x - f(t),

substituted into the system. A system passes when (1) every dominant balance has integer exponents p_i, (2) the
resonances r (roots of det Q(r), Q the linearisation of the dominant terms about the leading term) are integers,
(3) at each resonance r >= 0 the number of free coefficients equals the multiplicity of r as a root of det Q(r)
(otherwise the general solution needs psi^r log psi terms), and (4) the compatibility condition at each resonance holds
identically, for arbitrary f(t) and arbitrary earlier free coefficients. Failing any of them means the general solution
has a movable branch point, so the system does not have the Painleve property in these variables.

Systems (x is the variable in psi = x - f(t); t the other one):

    controls that must pass:   Burgers, KdV, the cubic NLS as a system for (u, v = conj u), and the forced Burgers
                               system Y_s + (V Y)_u = Y_uu, V_u = Y (the inner system with Y^2/4 replaced by Y)
    controls that must fail:   Newell-Whitehead u_t = u_xx + u - u^3, Fisher-KPP u_t = u_xx + u(1 - u)
    the inner problem:         (A) gauge-fixed form in the proper length u = int c dt and V = u_s:
                                   Y_s + (V Y)_u = Y_uu,   V_u = Y^2/4
                               (A0) its s-independent reduction (an ODE): Y_u = V Y + k1, V_u = Y^2/4
                               (A1) its self-similar reduction (an ODE in z = u/sqrt(s)), Y = g/sqrt(s), V = h/sqrt(s):
                                   g' = g h - z g / 2 + k,  h' = g^2/4
                               (B) the form of Prop. 5(i) in (s, t) with the lapse c (dominant balance only):
                                   c^3 Y_s - c Y_tt + c_t Y_t + c^3 Y^3/4 = 0,  c_s = c Y^2/4
                               (C) the self-similar form of the math paper's eq. (inner) in (xi, theta) (dominant
                                   balance only): L y = c sigma, L sigma = c xi y_xi + c (y^2/2 - 1) y,
                                   xi c_xi = c y^2/2, L = xi^2 d_xi + w xi d_theta

    python3 painleve.py [--order N] [--mutate=NAME]

--mutate=forced_to_real replaces the forced-Burgers control's V_u = Y by V_u = Y^2/4 (it must then fail);
--mutate=real_to_forced replaces the inner system's V_u = Y^2/4 by V_u = Y (it must then pass). Either mutation makes
the script exit non-zero, because a control or the inner verdict no longer matches its expected value.
Exit status 0 only if every control gives its expected verdict.
"""
import sys
import time

import sympy as sp

P = sp.Symbol("P")                     # psi
T = sp.Symbol("t")                     # the "other" variable (time of the evolution)


class Run:
    def __init__(self, name, fields, p, lead, eqs, ode=False, xexp=False, nmax=6, expect=None):
        self.name, self.fields, self.p, self.eqs, self.ode, self.nmax = name, fields, p, eqs, ode, nmax
        self.expect = expect
        self.f = sp.Symbol("z0") if ode else sp.Function("f")(T)
        self.coef = {}
        for fl in fields:
            for j in range(nmax + 1):
                self.coef[(fl, j)] = sp.Symbol(f"{fl}{j}") if ode else sp.Function(f"{fl}{j}")(T)
        self.lead = lead                  # dict field -> leading coefficient expression (may contain free functions)
        self.xexp = xexp

    # derivative operators on Laurent expressions in P whose coefficients depend on t
    def Dx(self, e):
        return sp.diff(e, P)

    def Dt(self, e):
        if self.ode:
            return sp.Integer(0)
        return sp.diff(e, T) - sp.diff(self.f, T) * sp.diff(e, P)

    def X(self):
        return P + self.f

    def series(self, upto, sub):
        U = {}
        for fl in self.fields:
            U[fl] = sum(sub.get(self.coef[(fl, j)], self.coef[(fl, j)]) * P ** (self.p[fl] + j) for j in range(upto + 1))
        return U


def by_power(expr):
    """dict exponent -> coefficient of a Laurent polynomial in P"""
    out = {}
    if expr == 0:
        return out
    for term in sp.Add.make_args(sp.expand(expr)):
        c, pw = term.as_independent(P, as_Add=False)
        if pw == 1:
            e = sp.Integer(0)
        else:
            b, e = pw.as_base_exp()
            assert b == P, term
        out[e] = out.get(e, 0) + c
    return out


def resonances(R):
    """det Q(r) from the dominant terms linearised about the leading term, with symbolic r"""
    r, eps = sp.symbols("r eps")
    al = {fl: sp.Symbol(f"alpha_{fl}") for fl in R.fields}
    U = {fl: R.lead[fl] * P ** R.p[fl] + eps * al[fl] * P ** (R.p[fl] + r) for fl in R.fields}
    gen = {fl: sp.Symbol(f"L_{fl}") for fl in R.fields}      # generic leading coefficients fix the dominant power
    E0 = R.eqs(R, {fl: gen[fl] * P ** R.p[fl] for fl in R.fields})
    E = R.eqs(R, U)
    rows = []
    for e0, e in zip(E0, E):
        q = min(by_power(e0).keys())
        # power_exp=False keeps P**(r + k) whole, so each term's exponent can be read off
        lin = sp.expand(sp.diff(e, eps).subs(eps, 0), power_exp=False)
        dom = 0
        for term in sp.Add.make_args(lin):
            c, ex = sp.powsimp(term).as_coeff_exponent(P)
            if sp.simplify(ex - (q + r)) == 0:
                dom += c
        rows.append([sp.simplify(sp.diff(dom, al[fl])) for fl in R.fields])
    Q = sp.Matrix(rows)
    det = sp.factor(sp.simplify(Q.det()))
    return Q, det, r


def expand_run(R):
    """order-by-order solution; returns list of (j, status, detail)"""
    sub = {}
    for fl in R.fields:
        sub[R.coef[(fl, 0)]] = R.lead[fl]
    U = R.series(R.nmax, {})
    E = [sp.expand(e) for e in R.eqs(R, U)]
    Ep = [by_power(e) for e in E]
    q = [min(sp.Rational(k) for k in d.keys()) for d in Ep]
    log = []
    # check the leading order
    lead_res = [sp.simplify(d.get(qe, 0).subs(sub).doit()) for d, qe in zip(Ep, q)]
    log.append((0, "leading", "ok" if all(x == 0 for x in lead_res) else f"FAILS: {lead_res}"))
    free = {}
    for j in range(1, R.nmax + 1):
        unk = [R.coef[(fl, j)] for fl in R.fields]
        eqs = []
        for d, qe in zip(Ep, q):
            c = d.get(qe + j, 0)
            c = c.subs(sub).doit() if c != 0 else sp.Integer(0)
            eqs.append(sp.expand(c))
        M = sp.Matrix([[sp.diff(e, u) for u in unk] for e in eqs])
        M = M.applyfunc(sp.simplify)
        rank = M.rank(simplify=True)
        if rank == len(unk):
            sol = sp.solve(eqs, unk, dict=True)
            assert len(sol) == 1, (R.name, j, sol)
            for u in unk:
                sub[u] = sp.simplify(sol[0][u])
            log.append((j, "determined", ""))
            continue
        # resonance: solve a maximal non-singular square subsystem; the other rows give the compatibility conditions
        _, pc = M.rref(simplify=True)
        _, pr = M.T.rref(simplify=True)
        chosen = [unk[c] for c in pc]
        rows = [eqs[i] for i in pr]
        if chosen:
            sol = sp.solve(rows, chosen, dict=True)
            assert len(sol) == 1, (R.name, j, sol)
            for u in chosen:
                sub[u] = sp.simplify(sol[0][u])
        fr = [u for u in unk if u not in chosen]
        free[j] = fr
        resid = [sp.simplify(sp.together(sp.expand(e.subs(sub).doit()))) for e in eqs]
        ok = all(x == 0 for x in resid)
        log.append((j, "RESONANCE", f"rank {rank}/{len(unk)}, free {fr}, compatibility "
                                     f"{'holds identically' if ok else 'FAILS: ' + str([sp.factor(x) for x in resid if x != 0])}"))
        if not ok:
            return log, free, False
    return log, free, True


def _subsets(seq, k):
    from itertools import combinations
    return combinations(seq, k)


# ------------------------------------------------------------------------------------------------ the systems
def burgers(R, U):
    u = U["u"]
    return [R.Dt(u) + u * R.Dx(u) - R.Dx(R.Dx(u))]


def kdv(R, U):
    u = U["u"]
    return [R.Dt(u) + 6 * u * R.Dx(u) + R.Dx(R.Dx(R.Dx(u)))]


def nls(R, U):
    u, v = U["u"], U["v"]
    return [sp.I * R.Dt(u) + R.Dx(R.Dx(u)) + 2 * u ** 2 * v, -sp.I * R.Dt(v) + R.Dx(R.Dx(v)) + 2 * v ** 2 * u]


def newell_whitehead(R, U):
    u = U["u"]
    return [R.Dt(u) - R.Dx(R.Dx(u)) - u + u ** 3]


def fisher(R, U):
    u = U["u"]
    return [R.Dt(u) - R.Dx(R.Dx(u)) - u * (1 - u)]


def make_inner(kind):
    """kind 'real': V_u = Y^2/4 (the inner problem); 'forced': V_u = Y (the forced Burgers control)"""
    def sysA(R, U):
        Y, V = U["Y"], U["V"]
        src = Y ** 2 / 4 if kind == "real" else Y
        return [R.Dt(Y) + R.Dx(V * Y) - R.Dx(R.Dx(Y)), R.Dx(V) - src]
    return sysA


K1 = sp.Symbol("k1")
KK = sp.Symbol("k")


def sysA0(R, U):
    Y, V = U["Y"], U["V"]
    return [R.Dx(Y) - V * Y - K1, R.Dx(V) - Y ** 2 / 4]


def sysA1(R, U):
    g, h = U["g"], U["h"]
    z = R.X()
    return [R.Dx(g) - g * h + z * g / 2 - KK, R.Dx(h) - g ** 2 / 4]


def log_check(kind):
    """Add a psi^1 log(psi) term to the series of the Y-V system and solve through order psi^0 (resonance r = 2).

    kind 'real' (V_u = Y^2/4, leading Y = 2/psi, V = -1/psi): a free log coefficient means the general solution has a
    movable logarithmic branch point. kind 'forced' (V_u = Y, leading Y = 2/psi^2, V = -2/psi): the control, whose
    log coefficient must be forced to zero. Returns (number of free functions at r = 2, log coefficient expression)."""
    Lg = sp.Symbol("Lg")                             # log(psi)
    f = sp.Function("f")(T)
    c = {nm: sp.Function(nm)(T) for nm in ("y1", "y2", "b", "v1", "v2", "d", "y3", "b3", "v3", "d3")}

    def dP(e):
        return sp.diff(e, P) + sp.diff(e, Lg) / P

    def dT(e):
        return sp.diff(e, T) - sp.diff(f, T) * dP(e)

    # the log enters at the r = 2 position of each field: psi^(p + 2) log(psi)
    if kind == "real":
        Y = 2 / P + c["y1"] + (c["y2"] + c["b"] * Lg) * P
        V = -1 / P + c["v1"] + (c["v2"] + c["d"] * Lg) * P
        src = Y ** 2 / 4
        q1, q2 = -3, -2                               # dominant powers of E1 and E2
    else:
        Y = 2 / P ** 2 + c["y1"] / P + (c["y2"] + c["b"] * Lg)
        V = -2 / P + c["v1"] + (c["v2"] + c["d"] * Lg) * P
        src = Y
        q1, q2 = -4, -2
    E1 = sp.expand(dT(Y) + dP(V * Y) - dP(dP(Y)))
    E2 = sp.expand(dP(V) - src)
    LL = sp.Symbol("LL")

    def coeff(E, pw, lg):
        """coefficient of psi^pw log(psi)^lg"""
        out = 0
        for term in sp.Add.make_args(sp.expand(E.subs(Lg, LL))):
            cc, ex = sp.powsimp(term).as_coeff_exponent(P)
            cl, el = cc.as_coeff_exponent(LL)
            if ex == pw and el == lg:
                out += cl
        return sp.expand(out)

    sub = {}
    for j, unk in ((1, [c["y1"], c["v1"]]), (2, [c["y2"], c["v2"], c["b"], c["d"]])):
        eqs = [sp.expand(coeff(E, q + j, lg).subs(sub).doit()) for E, q in ((E1, q1), (E2, q2)) for lg in (0, 1)]
        eqs = [e for e in eqs if e != 0]
        sol = sp.solve(eqs, unk, dict=True)
        assert len(sol) == 1, (kind, j, sol)
        for u, v in sol[0].items():
            sub[u] = sp.simplify(v)
        resid = [sp.simplify(e.subs(sub).doit()) for e in eqs]
        assert all(x == 0 for x in resid), (kind, j, resid)
    freev = [u for u in (c["y2"], c["v2"], c["b"], c["d"]) if u not in sub]
    return freev, sp.simplify(sub.get(c["b"], c["b"]))


def dominant_balance_B():
    """(s, t) form with the lapse: exponents from the dominant balance, psi = t - f(s)"""
    p, rr, y0, c0, fp = sp.symbols("p r y0 c0 fp")
    # E2: c_s = c Y^2/4 -> -r fp c0 psi^(r-1) = c0 y0^2 psi^(r+2p)/4  =>  2p = -1
    p_val = sp.Rational(-1, 2)
    # E1 exponents: c^3 Y_s and c^3 Y^3: 3r + 3p ; c Y_tt and c_t Y_t: r + p - 2
    r_val = sp.solve(sp.Eq(3 * rr + 3 * p_val, rr + p_val - 2), rr)[0]
    E2 = -r_val * fp - y0 ** 2 / 4
    E1 = c0 ** 2 * (-p_val) * fp - p_val * (p_val - 1) + r_val * p_val + c0 ** 2 * y0 ** 2 / 4
    sol = sp.solve([E2, E1], [y0, c0], dict=True)
    return p_val, r_val, sol


def dominant_balance_C():
    """(xi, theta) form: xi c_xi = c y^2/2 with psi = xi - f(theta), xi = f + psi -> f r c0 psi^(r-1) = c0 y0^2 psi^(r+2p)/2"""
    return sp.Rational(-1, 2)


# ------------------------------------------------------------------------------------------------ driver
def main(argv):
    nmax = 6
    mutate = None
    for a in argv[1:]:
        if a.startswith("--order"):
            nmax = int(a.split("=")[1])
        elif a.startswith("--mutate="):
            mutate = a.split("=")[1]
    forced_kind = "real" if mutate == "forced_to_real" else "forced"
    real_kind = "forced" if mutate == "real_to_forced" else "real"
    one = sp.Integer(1)
    a_free = sp.Function("A0")(T)
    runs = [
        Run("Burgers u_t + u u_x = u_xx", ["u"], {"u": -1}, {"u": -2 * one}, burgers, nmax=4, expect=True),
        Run("KdV u_t + 6 u u_x + u_xxx = 0", ["u"], {"u": -2}, {"u": -2 * one}, kdv, nmax=6, expect=True),
        Run("cubic NLS (u, v)", ["u", "v"], {"u": -1, "v": -1}, {"u": a_free, "v": -1 / a_free}, nls, nmax=4,
            expect=True),
        Run(f"forced Burgers system Y_s + (VY)_u = Y_uu, V_u = {'Y' if forced_kind == 'forced' else 'Y^2/4 (mutated)'}",
            ["Y", "V"], {"Y": -2 if forced_kind == "forced" else -1, "V": -1},
            {"Y": 2 * one, "V": -2 * one} if forced_kind == "forced" else {"Y": 2 * one, "V": -one},
            make_inner(forced_kind), nmax=4, expect=True),
        Run("Newell-Whitehead u_t = u_xx + u - u^3", ["u"], {"u": -1}, {"u": sp.sqrt(2)}, newell_whitehead, nmax=5,
            expect=False),
        Run("Fisher-KPP u_t = u_xx + u(1-u)", ["u"], {"u": -2}, {"u": 6 * one}, fisher, nmax=7, expect=False),
        Run(f"(A) inner, gauge-fixed: Y_s + (VY)_u = Y_uu, V_u = {'Y^2/4' if real_kind == 'real' else 'Y (mutated)'}",
            ["Y", "V"], {"Y": -1 if real_kind == "real" else -2, "V": -1},
            {"Y": 2 * one, "V": -one} if real_kind == "real" else {"Y": 2 * one, "V": -2 * one},
            make_inner(real_kind), nmax=4, expect=False),
        Run("(A0) s-independent reduction: Y' = V Y + k1, V' = Y^2/4", ["Y", "V"], {"Y": -1, "V": -1},
            {"Y": 2 * one, "V": -one}, sysA0, ode=True, nmax=4, expect=False),
        Run("(A1) self-similar reduction: g' = g h - z g/2 + k, h' = g^2/4", ["g", "h"], {"g": -1, "h": -1},
            {"g": 2 * one, "h": -one}, sysA1, ode=True, xexp=True, nmax=4, expect=False),
    ]
    bad = 0
    print(f"Painleve test, Kruskal ansatz psi = x - f(t); sympy {sp.__version__}\n")
    for R in runs:
        t0 = time.time()
        Q, det, r = resonances(R)
        roots = sp.roots(sp.Poly(sp.expand(det), r)) if det != 0 else {}
        log, free, ok = expand_run(R)
        # multiplicity check: free coefficients at resonance r must equal its multiplicity
        mult_ok = True
        notes = []
        for root, m in roots.items():
            if root.is_integer and 1 <= root <= R.nmax and ok:
                nf = len(free.get(int(root), []))
                if nf < m:
                    mult_ok = False
                    notes.append(f"r = {root}: multiplicity {m} but {nf} free coefficient(s): log terms needed")
            if not root.is_integer:
                mult_ok = False
                notes.append(f"non-integer resonance {root}")
        passed = ok and mult_ok
        verdict = "PASSES" if passed else "FAILS"
        print(f"== {R.name}")
        print(f"   leading exponents {R.p}, leading coefficients {R.lead}")
        print(f"   det Q(r) = {det};  resonances {dict(roots)}")
        for j, st, d in log:
            if st != "determined":
                print(f"   order {j}: {st} {d}")
        for n in notes:
            print(f"   {n}")
        exp_s = "" if R.expect is None else ("(expected)" if passed == R.expect else "(UNEXPECTED)")
        print(f"   verdict: {verdict} {exp_s}   [{time.time() - t0:.1f}s]\n")
        if R.expect is not None and passed != R.expect:
            bad += 1
    # direct check of the log term at the double resonance of (A), with the forced Burgers system as its control
    for kind, want_log in ((real_kind, True), (forced_kind, False)):
        fr, lc = log_check(kind)
        has_log = lc != 0
        lab = "inner V_u = Y^2/4" if kind == "real" else "forced Burgers V_u = Y"
        okk = has_log == want_log
        print(f"== log check at r = 2 for the {lab}: free functions {fr}, log(psi) coefficient = {lc}  "
              f"{'(expected)' if okk else '(UNEXPECTED)'}")
        if not okk:
            bad += 1
    print()
    pB, rB, solB = dominant_balance_B()
    print("== (B) Prop. 5(i) form in (s, t) with the lapse c, psi = t - f(s): dominant balance")
    print(f"   Y ~ y0 psi^({pB}), c ~ c0 psi^({rB}), with {solB}")
    print("   verdict: FAILS (movable square-root branch point; the lapse gauge c -> T'(t) c can move it, so this form")
    print("   alone is not decisive; (A) is the gauge-fixed form)\n")
    pC = dominant_balance_C()
    print("== (C) self-similar form in (xi, theta), psi = xi - f(theta): xi c_xi = c y^2/2 forces y ~ psi^(-1/2)")
    print(f"   leading exponent of y: {pC}; verdict: FAILS (movable square-root branch point, same gauge caveat as B)\n")
    print(f"controls and inner verdicts matching expectation: {len(runs) - bad}/{len(runs)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
