#!/usr/bin/env python3
"""Is lambda = 2 an exact growth rate of the l-multiplet's D = infinity inner problem? (#994)

The system (STRUCTURE.md 3 (d)): psi_t = c Q, Q_t = c psi_s, c_s = 2 s^l c Q^2, self-similar scaling
Z = 2 s d_s + t d_t - l psi d_psi - (l + 1) Q d_Q. A symmetry X with [Z, X] = -lambda X gives the self-similar
solution a perturbation ~ e^{lambda tau}. At l = 0 the s-translation d_s gives lambda = 2 (GAMMA.md 4-5).

    python3 ell_lambda2.py point        every Lie point symmetry, for symbolic l, without an ansatz (seconds)
    python3 ell_lambda2.py mseq         lambda_inf(l = 1) from its M = 1, 3 values with l = 0's M sequence as the
                                        calibration
    python3 ell_lambda2.py point --mutate=weight    the weight s^l replaced by s^(l + 1) in the c equation only:
                                        the planted l = 0 s-translation must disappear (exit 1)
    python3 ell_lambda2.py gen          first-order generalized symmetries of weight 2 at l = 0, 1, 2 within a
                                        polynomial ansatz (gen's docstring): l = 0 must give d_s and the gauge mode
                                        (planted), l = 1 and 2 the gauge mode alone (minutes)
    python3 ell_lambda2.py gen --mutate=weight      the same with the weight s^(l + 1): l = 0 loses d_s (exit 1)

point. The infinitesimals sigma, tau, Phi, Theta, Gamma (of s, t, psi, Q, c) are arbitrary functions. The first
prolongation's invariance condition on solutions (psi_t, Q_t, c_s eliminated) is split by the parametric derivatives
psi_s, Q_s, c_t into determining equations, which are then solved by these steps, each printed with the equations that
force it.
- An equation that is a nonzero factor times one derivative d_x F removes x from F's arguments.
- An equation is split into the coefficients of a variable that no unknown in it depends on.
- Two elimination steps are done by hand and checked: Theta from E1's derivative-free equation, then Gamma from E2's.
The result is the general solution, valid for any l. At l = 0 it must contain d_s (planted). For a weight s^(l + 0.1),
which is any l not 0, it must not (the control). gen extends the search to first-order generalized symmetries of
weight 2 within a polynomial ansatz, at integer l. Higher-order generalized symmetries and nonlocal ones are not
searched.
"""
import sys

import sympy as sp

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
s, t, P, Q, C = sp.symbols("s t psi Q c")
l = sp.Symbol("l")
VARS = (s, t, P, Q, C)
NAMES = ("sigma", "tau", "Phi", "Theta", "Gamma")


def deteq(lw):
    """determining equations of psi_t = c Q, Q_t = c psi_s, c_s = 2 s^lw c Q^2 for generic infinitesimals"""
    ps, Qs, ct = sp.symbols("psi_s Q_s c_t")
    F = {n: sp.Function(n)(*VARS) for n in NAMES}
    sig, tau, Phi, The, Gam = (F[n] for n in NAMES)
    cs = 2 * s ** lw * C * Q ** 2
    d = {P: (ps, C * Q), Q: (Qs, C * ps), C: (cs, ct)}
    Ds = lambda f: sp.diff(f, s) + sum(sp.diff(f, u) * d[u][0] for u in (P, Q, C))
    Dt = lambda f: sp.diff(f, t) + sum(sp.diff(f, u) * d[u][1] for u in (P, Q, C))
    eta = {P: Phi, Q: The, C: Gam}
    eta_s = lambda u: Ds(eta[u]) - d[u][0] * Ds(sig) - d[u][1] * Ds(tau)
    eta_t = lambda u: Dt(eta[u]) - d[u][0] * Dt(sig) - d[u][1] * Dt(tau)
    E1 = eta_t(P) - Gam * Q - C * The
    E2 = eta_t(Q) - Gam * ps - C * eta_s(P)
    E3 = eta_s(C) - 2 * lw * s ** (lw - 1) * sig * C * Q ** 2 - 2 * s ** lw * (Gam * Q ** 2 + 2 * C * Q * The)
    out = []
    for k, E in enumerate((E1, E2, E3), 1):
        for mon, co in sp.Poly(sp.expand(E), ps, Qs, ct).terms():
            out.append((f"E{k}{mon}", sp.expand(co)))
    return out, F


def funcs(e):
    return [f for f in e.atoms(sp.Function) if f.func.__name__ in NAMES or f.func.__name__.startswith("f_")]


class Solver:
    def __init__(self, eqs, F):
        self.eqs = eqs                      # list of (label, expr)
        self.sub = dict(F)                  # name -> current expression
        self.log = []

    def apply(self, rep):
        """substitute rep {old function application: new expression} everywhere"""
        self.eqs = [(lab, sp.expand(sp.simplify(e.subs(rep).doit()))) for lab, e in self.eqs]
        self.eqs = [(lab, e) for lab, e in self.eqs if e != 0]
        self.sub = {n: sp.simplify(v.subs(rep).doit()) for n, v in self.sub.items()}

    def prune_once(self):
        """an equation c * d_x f(...) = 0 with c free of unknowns: f does not depend on x"""
        for lab, e in self.eqs:
            ds = list(e.atoms(sp.Derivative))
            fs = funcs(e)
            if len(ds) == 1 and len(fs) == 1 and ds[0].expr == fs[0] and len(ds[0].variables) == 1:
                fac = sp.simplify(e / ds[0])
                if not funcs(fac) and fac != 0:
                    f, x = fs[0], ds[0].variables[0]
                    args = tuple(a for a in f.args if a != x)
                    g = sp.Function(f.func.__name__)(*args)
                    self.log.append(f"{lab}: {sp.factor(e)} = 0  =>  {f.func.__name__} independent of {x}")
                    self.apply({f: g})
                    return True
        return False

    def split_once(self):
        """split an equation by a variable that none of its unknowns depends on (as a polynomial in it)"""
        for i, (lab, e) in enumerate(self.eqs):
            dep = set()
            for f in funcs(e):
                dep |= set(f.args)
            for x in (C, Q, P, t, s):
                if x in dep or not e.has(x):
                    continue
                try:
                    poly = sp.Poly(e, x)
                except sp.PolynomialError:
                    continue
                if poly.degree() < 1:
                    continue
                parts = [(f"{lab}[{x}^{m[0]}]", sp.expand(co)) for m, co in poly.terms()]
                self.log.append(f"{lab}: split by powers of {x} into {len(parts)} equations")
                self.eqs = self.eqs[:i] + [p for p in parts if p[1] != 0] + self.eqs[i + 1:]
                return True
        return False

    def run(self):
        while self.prune_once() or self.split_once():
            pass

    def show(self, head):
        print(f"\n-- {head}: {len(self.eqs)} equations left")
        for lab, e in self.eqs:
            print(f"   {lab}: {sp.factor(e)} = 0")
        print("   infinitesimals: " + ", ".join(f"{n} = {self.sub[n]}" for n in NAMES))


def point(lw, verbose=True):
    """the general point symmetry for the weight s^lw (lw symbolic or a number); returns the solver"""
    eqs, F = deteq(lw)
    S = Solver(eqs, F)
    S.run()
    if verbose:
        for line in S.log:
            print("   " + line)
        S.show("after the automatic steps")
    # by hand (checked): E1(0,0,0) gives Theta; Theta's c-independence then forces Gamma = c g(s, t) and Phi_t = 0
    sub = S.sub
    E1 = dict(S.eqs).get("E1(0, 0, 0)")
    assert E1 is not None, "E1(0,0,0) expected among the remaining equations"
    th = sub["Theta"]
    sol = sp.solve(E1, th)
    assert len(sol) == 1
    dc = sp.expand(sp.simplify(sp.diff(sol[0], C) * C ** 2))
    if verbose:
        print(f"\n   E1(0,0,0) solved for Theta: Theta = {sp.simplify(sol[0])}; Theta does not depend on c, so "
              f"c^2 d_c of that = {sp.factor(dc)} = 0, split by Q:")
    pq = sp.Poly(dc, Q)
    conds = [co for _, co in pq.terms()]
    for co in conds:
        if verbose:
            print(f"      {sp.factor(co)} = 0")
    G = sub["Gamma"]
    g = sp.Function("f_g")(s, t)
    assert sp.simplify(conds[0].subs(G, C * g).doit()) == 0 or sp.simplify(conds[-1].subs(G, C * g).doit()) == 0
    # the Q^1 condition c Gamma_c - Gamma = 0 has the general solution Gamma = c g(s, t); the Q^0 one is Phi_t = 0
    Phi0 = sub["Phi"]
    PhiN = sp.Function("Phi")(*[a for a in Phi0.args if a != t])
    S.apply({G: C * g, Phi0: PhiN})
    S.apply({th: sp.simplify(sol[0].subs({G: C * g, Phi0: PhiN}).doit())})
    S.run()
    if verbose:
        for line in S.log[len(S.log) - 0:]:
            print("   " + line)
        S.show("after Theta and Gamma = c g(s, t)")
    # by hand (checked): E2(1,0,0) is sigma'(s) - 2 g(t) - 2 tau'(t) = 0 (g has lost s above). Its s-derivative is
    # sigma'' = 0, so sigma = sigma0 + sigma1 s, and then g = sigma1/2 - tau'
    eq = dict(S.eqs)
    lab2 = next(k for k in eq if k.startswith("E2(1, 0, 0)"))
    e2 = eq[lab2]
    sg0 = S.sub["sigma"]
    d2 = sp.simplify(sp.diff(e2, s))
    assert sp.simplify(d2 - sp.diff(sg0, s, 2)) == 0 or sp.simplify(d2 + sp.diff(sg0, s, 2)) == 0, d2
    s0, s1, a, b = sp.symbols("sigma0 sigma1 a b")
    gfun = next(f for f in funcs(e2) if f.func.__name__ == "f_g")
    S.apply({sg0: s0 + s1 * s})
    e2 = dict(S.eqs)[lab2]
    sg = sp.solve(e2, gfun)
    assert len(sg) == 1
    if verbose:
        print(f"\n   {lab2}: d_s of it is sigma'' = 0, so sigma = sigma0 + sigma1 s; then {gfun} = {sg[0]}")
    S.apply({gfun: sg[0]})
    # by hand (checked): Phi'' = 0 (E2's Q^2 part), so Phi = a psi + b
    eq = dict(S.eqs)
    ph = S.sub["Phi"]
    lab3 = next(k for k, e in eq.items() if e.has(sp.Derivative(ph, P, 2)))
    assert sp.simplify(eq[lab3] - sp.diff(ph, P, 2)) == 0 or sp.simplify(eq[lab3] + sp.diff(ph, P, 2)) == 0
    if verbose:
        print(f"   {lab3}: Phi'' = 0, so Phi = a psi + b")
    S.apply({ph: a * P + b})
    if verbose:
        S.show("after sigma, g and Phi (only conditions on constants and tau may remain)")
    left = [e for _, e in S.eqs]
    assert all(not funcs(e) or all(f.func.__name__ == "tau" for f in funcs(e)) for e in left)
    S.const = (s0, s1, a, b)
    return S


def final_form(lw):
    """the general solution read off the steps: sigma = s0 + s1 s, tau(t), Phi = a psi + b, Gamma = c (s1/2 - tau'),
    Theta = Q (a - s1/2) ... with the algebraic conditions left; returns the conditions on (s0, s1, a)"""
    s0, s1, a, b = sp.symbols("sigma0 sigma1 a b")
    T = sp.Function("T")(t)
    gen = {"sigma": s0 + s1 * s, "tau": T, "Phi": a * P + b, "Gamma": C * (s1 / 2 - sp.diff(T, t)),
           "Theta": Q * (a - s1 / 2)}
    eqs, F = deteq(lw)
    rest = []
    for lab, e in eqs:
        r = sp.expand(sp.simplify(e.subs({F[n]: gen[n] for n in NAMES}).doit()))
        if r != 0:
            rest.append((lab, sp.factor(r)))
    return gen, rest, (s0, s1, a, b)


def bracket(A, B):
    """[A, B] of vector fields given as dicts var -> coefficient over VARS"""
    out = {}
    for v in VARS:
        out[v] = sp.simplify(sum(A.get(x, 0) * sp.diff(B.get(v, 0), x) - B.get(x, 0) * sp.diff(A.get(v, 0), x)
                                 for x in VARS))
    return out


def lam_of(Zf, X):
    """lambda with [Z, X] = -lambda X, or None"""
    br = bracket(Zf, X)
    lam = None
    for v in VARS:
        xv, bv = sp.simplify(X.get(v, 0)), br[v]
        if xv == 0:
            if sp.simplify(bv) != 0:
                return None
            continue
        q = sp.simplify(-bv / xv)
        if q.free_symbols & set(VARS):
            return None
        if lam is None:
            lam = q
        elif sp.simplify(q - lam) != 0:
            return None
    return lam


def run_point():
    lw_mut = (lambda lv: lv + 1) if "weight" in MUT else (lambda lv: lv)
    print("== the general Lie point symmetry, weight s^l with l symbolic"
          + ("  [MUTATION: the c equation's weight is s^(l + 1)]" if MUT else ""))
    S = point(lw_mut(l))
    gen, rest, (s0, s1, a, b) = final_form(lw_mut(l))
    print("\n== check: the general solution")
    print("   sigma = sigma0 + sigma1 s, tau = T(t), Phi = a psi + b, Gamma = c (sigma1/2 - T'), Theta = Q (a - sigma1/2)")
    print("   left in the determining equations (all must vanish):")
    conds = []
    for lab, r in rest:
        print(f"      {lab}: {r}")
        conds.append(r)
    # the remaining conditions are linear in (sigma0, sigma1, a) with coefficients in s^l, s^(l-1): split by s-powers
    lin = []
    for r in conds:
        r = sp.expand(sp.powsimp(sp.expand(r), force=True))
        for term_group in sp.collect(r, [s ** lw_mut(l), s ** (lw_mut(l) - 1)], evaluate=False).values():
            for co in sp.Poly(sp.expand(term_group), Q, C).coeffs():
                if co != 0:
                    lin.append(sp.factor(co))
    lin = sorted(set(lin), key=str)
    print(f"   conditions on the constants: {[str(x) + ' = 0' for x in lin]}")
    ok = True
    Zf = {s: 2 * s, t: t, P: -l * P, Q: -(l + 1) * Q, C: 0}
    for lv, label, expect_ds in ((0, "l = 0 (planted)", True), (1, "l = 1", False), (2, "l = 2", False),
                                 (sp.Rational(1, 10), "l = 0.1 (control: the weight s^(0 + 0.1))", False),
                                 (sp.Rational(11, 10), "l = 1.1 (control: the weight s^(1 + 0.1))", False)):
        cl = [x.subs(l, lv) for x in lin]
        sol = sp.solve(cl, [s0, s1, a], dict=True)
        sol = sol[0] if sol else {}
        free = [v for v in (s0, s1, a) if v not in sol or sol[v].has(v)]
        has_ds = s0 in free
        Zl = {k: sp.sympify(v).subs(l, lv) for k, v in Zf.items()}
        gens = {"d_s": {s: 1}, "S (second scaling)": {s: s, P: -lv * P / 2, Q: -(lv + 1) * Q / 2, C: C / 2},
                "d_psi": {P: 1}}
        present = []
        if has_ds:
            present.append("d_s")
        if s1 in free or (s1 in sol and sol[s1] != 0):
            present.append("S (second scaling)")
        present.append("d_psi")
        lams = {g_: lam_of(Zl, gens[g_]) for g_ in present}
        gauge = {k: lam_of(Zl, {t: t ** k, C: -k * t ** (k - 1) * C}) for k in (-1, 0, 1, 2)}
        lam2 = [g_ for g_, lv_ in lams.items() if lv_ == 2]
        good = (len(lam2) > 0) == expect_ds
        ok &= good
        print(f"\n   {label}: solution {sol}, free constants {free}")
        print(f"      non-gauge generators and their lambda: {lams}")
        print(f"      gauge t^k d_t - k t^(k-1) c d_c, lambda by k: {gauge} (lambda = 1 - k; k = -1 gives the gauge "
              f"mode at lambda = 2, which moves c(0) and is removed by h0 = 0)")
        print(f"      a non-gauge lambda = 2 generator: {lam2 or 'none'}  "
              f"(expected {'d_s' if expect_ds else 'none'}: {'OK' if good else 'BAD'})")
    print("\nPASS" if ok else "\nFAIL")
    return ok


def mseq():
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(here, "..", "inner_pert", "data")
    l0 = [json.load(open(os.path.join(dd, f"first_order_hier_M{M}_xi0.2.json")))["lam"] for M in (1, 3, 5)]
    l1 = json.load(open(os.path.join(here, "data", "ell_lambda_l1.json")))
    a1, a3 = l1["M1"]["lam"], l1["M3"]["lam"]
    a3lo, a3hi = l1["M3"]["range"]
    d13, d35 = l0[1] - l0[0], l0[2] - l0[1]
    r1, r2 = d35 / d13, (2 - l0[2]) / d35
    print("l = 0 (first_order_hier_M{1,3,5}_xi0.2.json; exact limit 2, GAMMA.md 5):")
    print(f"   lambda* {l0[0]:.9f}, {l0[1]:.9f}, {l0[2]:.9f}; steps M 1->3 {d13:+.6f}, 3->5 {d35:+.6f}, 5->inf {2 - l0[2]:+.6f}")
    print(f"   step ratios (3->5)/(1->3) {r1:+.4f}, (5->inf)/(3->5) {r2:+.4f}: alternating, shrinking 20- to 35-fold")
    D = a3 - a1
    rows = []
    for lab, ra, rb in (("both l = 0 ratios", r1, r2), ("geometric, ratio (3->5)/(1->3)", r1, r1),
                        ("geometric, ratio (5->inf)/(3->5)", r2, r2)):
        if ra == rb:
            tail = D * ra / (1 - ra)
        else:
            tail = D * ra * (1 + rb)
        rows.append((lab, a3 + tail, tail))
    print(f"\nl = 1 (data/ell_lambda_l1.json, from raw I994-mac): lambda* M = 1 {a1:.9f}, M = 3 {a3:.6f} "
          f"(range {a3lo:.6f} to {a3hi:.6f}); step 1->3 {D:+.6f}")
    for lab, lim, tail in rows:
        print(f"   calibrated by {lab}: M = 3 -> inf {tail:+.6f}, lambda_inf {lim:.6f} (gamma {1 / lim:.6f}), lambda_inf - 2 = {lim - 2:+.6f} "
              f"(M = 3 range: {a3lo + tail - 2:+.6f} to {a3hi + tail - 2:+.6f})")
    need = (2 - a3) / D
    print(f"\n   lambda_inf = 2 would need the M = 3 -> inf correction {2 - a3:+.6f}, i.e. {need:+.3f} times the 1 -> 3 step "
          f"(range {(2 - a3hi) / D:+.3f} to {(2 - a3lo) / D:+.3f}), with the step's own sign; l = 0's corrections after "
          f"M = 3 total {(2 - l0[1]) / d13:+.4f} times its 1 -> 3 step, of the opposite sign")
    return True


def gen(lv, DEG=3, TMAX=3, SMAX=2, CMIN=-1, CMAX=2):
    """first-order generalized symmetries of weight 2 at the integer weight lv: characteristics K = (K_psi, K_Q, K_c)
    on the jet (s, t, psi, Q, c, psi_s, Q_s, c_t), each a sum of monomials s^i t^j c^k psi^a Q^b psi_s^p Q_s^q c_t^r
    (0 <= i <= SMAX, |j| <= TMAX, CMIN <= k <= CMAX, a + b + p + q + r <= DEG) of the Z-weight of d_s's own
    characteristic -(psi_s, Q_s, c_s), i.e. (-l-2, -l-3, -2). The linearized equations on solutions,
        D_t K_psi = K_c Q + c K_Q,   D_t K_Q = K_c psi_s + c D_s K_psi,   D_s K_c = 2 s^l (K_c Q^2 + 2 c Q K_Q),
    with the mixed derivatives eliminated (free jet coordinates psi_s, psi_ss, Q_s, Q_ss, c_t, c_tt), give an exact
    linear system for the coefficients. Point symmetries of weight 2 are included (in evolutionary form)."""
    from sympy import QQ
    from sympy.polys.matrices import DomainMatrix
    import itertools
    ps, pss, Qs, Qss, ct, ctt = sp.symbols("psi_s psi_ss Q_s Q_ss c_t c_tt")
    W = s ** (lv + 1 if "weight" in MUT else lv)    # the mutation: the c equation's weight raised by one
    cs = 2 * W * C * Q ** 2
    first = {P: (ps, C * Q), Q: (Qs, C * ps), C: (cs, ct)}
    second = {ps: (pss, sp.expand(cs * Q + C * Qs)), Qs: (Qss, sp.expand(cs * ps + C * pss)),
              ct: (sp.expand(2 * W * (ct * Q ** 2 + 2 * C * Q * C * ps)), ctt)}
    Ds = lambda f: sp.diff(f, s) + sum(sp.diff(f, u) * first[u][0] for u in first) + sum(sp.diff(f, u) * second[u][0] for u in second)
    Dt = lambda f: sp.diff(f, t) + sum(sp.diff(f, u) * first[u][1] for u in first) + sum(sp.diff(f, u) * second[u][1] for u in second)
    wt = {P: -lv, Q: -lv - 1, ps: -lv - 2, Qs: -lv - 3, ct: -1}
    targets = (-lv - 2, -lv - 3, -2)
    jv = (P, Q, ps, Qs, ct)
    comps, unk = [], []
    for m, target in enumerate(targets):
        K = 0
        for e in itertools.product(range(DEG + 1), repeat=5):
            if sum(e) > DEG:
                continue
            w0 = sum(wt[v] * k for v, k in zip(jv, e))
            for i in range(SMAX + 1):
                j = target - 2 * i - w0
                if abs(j) > TMAX:
                    continue
                for k in range(CMIN, CMAX + 1):
                    a_ = sp.Symbol(f"a{m}_{len(unk)}")
                    unk.append(a_)
                    K += a_ * s ** i * t ** j * C ** k * sp.Mul(*[v ** p_ for v, p_ in zip(jv, e)])
        comps.append(K)
    K1, K2, K3 = comps
    conds = [Dt(K1) - K3 * Q - C * K2, Dt(K2) - K3 * ps - C * Ds(K1), Ds(K3) - 2 * W * (K3 * Q ** 2 + 2 * C * Q * K2)]
    rows = []
    gens = (s, t, P, Q, C, ps, pss, Qs, Qss, ct, ctt)
    for cnd in conds:
        num = sp.expand(cnd * t ** (TMAX + 2) * C ** (-CMIN + 2))
        for _, co in sp.Poly(num, *gens).terms():
            rows.append([co.coeff(u_) for u_ in unk])
    M = DomainMatrix.from_list_sympy(len(rows), len(unk), rows).convert_to(QQ)
    ns = M.nullspace().to_Matrix()
    sols = []
    for r in range(ns.rows):
        vec = dict(zip(unk, ns.row(r)))
        sols.append(tuple(sp.factor(sp.expand(K.subs(vec))) for K in comps))
    return sols, len(unk), len(rows)


def run_gen():
    ok = True
    for lv, expect in ((0, True), (1, False), (2, False)):
        sols, nu, nr = gen(lv)
        print(f"\n== l = {lv}: first-order generalized symmetries of weight 2 ({nu} unknown coefficients, {nr} conditions): "
              f"{len(sols)} independent solutions" + ("  [MUTATION: weight s^(l + 1) in the c equation]" if MUT else ""))
        tau = 1 / t                           # the gauge mode of weight 2: tau = 1/t, characteristic -tau (c Q, c psi_s, c_t) + tau' c
        gauge = (-tau * C * Q, -tau * C * sp.Symbol("psi_s"), -sp.diff(tau, t) * C - tau * sp.Symbol("c_t"))
        dsch = (-sp.Symbol("psi_s"), -sp.Symbol("Q_s"), -2 * s ** (lv + 1 if "weight" in MUT else lv) * C * Q ** 2)
        for K in sols:
            print(f"   K = {K}")
        # the span of the solutions against {gauge, d_s}
        def in_span(vecs, target):
            cs_ = sp.symbols(f"x0:{len(vecs)}")
            eqs = []
            for m in range(3):
                e = sp.expand(sum(c_ * v[m] for c_, v in zip(cs_, vecs)) - target[m])
                num = sp.numer(sp.together(e))
                eqs += sp.Poly(num, s, t, P, Q, C, *sp.symbols("psi_s Q_s c_t")).coeffs()
            return bool(sp.solve(eqs, cs_, dict=True)) or all(e == 0 for e in eqs)
        has_gauge = in_span(sols, gauge)
        has_ds = in_span(sols, dsch)
        non_gauge = len(sols) - (1 if has_gauge else 0)
        good = has_gauge and (non_gauge == (1 if expect else 0)) and (has_ds == expect)
        ok &= good
        print(f"   contains the gauge mode (tau = 1/t): {has_gauge}; contains d_s: {has_ds}; solutions beyond the gauge: "
              f"{non_gauge} (expected {1 if expect else 0}, d_s {'present' if expect else 'absent'}): {'OK' if good else 'BAD'}")
    print("\nPASS" if ok else "\nFAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    good = {"point": run_point, "mseq": mseq, "gen": run_gen}[cmd]()
    sys.exit(0 if good else 1)
