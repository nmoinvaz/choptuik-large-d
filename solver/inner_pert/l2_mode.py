#!/usr/bin/env python3
"""Is the l = 2 deformation of the D = infinity inner solution exact? A sympy check (notes/L2_MODE.md, #993;
notes/DIRECTION_REVIEW.md 2.1 (a)).

Spatial R^(n+1) = R^p x R^q (p + q = n + 1), y in R^p, z in R^q, r1 = |y|, r2 = |z|, in inner variables
r1 = sqrt(2 n s) cos(chi), r2 = sqrt(2 n s) sin(chi) (so s = r^2/(2n)). The metric is O(p) x O(q) invariant:

    g = -c^2 dt^2 + n (ds^2 / (2s) + 2 s dchi^2) + (mu / n) dS^2 / |grad S|^2 + r1^2 dOmega_{p-1} + r2^2 dOmega_{q-1},

flat space plus the lapse c and an O(1/n) spatial part along the normal of the level sets of S (|grad S| in the flat
metric). The deformed configuration follows S = s + e dS: c = c(S, t), mu = mu(S, t), phi = Phi(S, t), linearised in e
(F(S, t) -> F(s, t) + e dS F_s). Curvature from the multiply-warped-product formulas, base (t, s, chi) and two sphere
fibres of dimensions m1 = p - 1, m2 = q - 1:

    R_ab = R~_ab - sum_k m_k (D_a D_b U_k + D_a U_k D_b U_k),
    R^(k) = (m_k - 1) e^{-2 U_k} - (box U_k + grad U_k . sum_l m_l grad U_l)     (times the fibre metric),
    box phi = box~ phi + sum_k m_k grad U_k . grad phi,

with e^{U_1} = r1, e^{U_2} = r2. Equations: R_mu nu = kappa d_mu phi d_nu phi (massless scalar), and box phi = 0.

    python3 l2_mode.py --flat         planted: flat space (c = 1, mu = Phi = 0) has zero residuals for any e
    python3 l2_mode.py --check        the whole check (about 3 min): background, l = 0 calibration, l = 2 for p = q and
                                      p = 1, the non-harmonic control; exit 0 iff every case comes out as fixed below
    python3 l2_mode.py --check --mutate=curv    the fibres' curvature term dropped: must fail (exit 1)
"""
import sys

import sympy as sp

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
t, s, chi = sp.symbols("t s chi", positive=True)
n, e, kap = sp.symbols("n e kappa", positive=True)
X = (t, s, chi)
c0, mu0, Ph0 = (sp.Function(f)(s, t) for f in ("c", "mu", "Phi"))


def follow(F, dS):
    return F + e * dS * sp.diff(F, s)


def lin(expr):
    """the part of expr of degree <= 1 in e"""
    return sp.expand(expr.subs(e, 0)) + e * sp.expand(sp.diff(expr, e).subs(e, 0))


A1, B1, C1, V1, V2, L1, P1 = (sp.Function(f)(s, t, chi) for f in ("A", "B", "C", "V1", "V2", "L", "P"))


def setup(dS, p="half", flat=False, corr=None):
    """metric (3x3 base), warps U_k with dimensions m_k, and the scalar, for the deformation dS(s, chi). corr: the
    O(e) corrections {A, B, C, V1, V2, L, P} (default all the unknown functions): g_ss += e A/(2s), g_s chi += e B,
    g_chi chi += 2 s e C (each of relative order 1/n), U_k += e V_k / n, the lapse c += e L and the scalar
    Phi += e P (order one)"""
    S = s + e * dS
    if flat:
        c, mu, Ph = sp.Integer(1), sp.Integer(0), sp.Integer(0)
    else:
        c, mu, Ph = follow(c0, dS), follow(mu0, dS), follow(Ph0, dS)
    K = {"A": A1, "B": B1, "C": C1, "V1": V1, "V2": V2, "L": L1, "P": P1}
    if corr is not None:
        K = {k: corr.get(k, sp.Integer(0)) for k in K}
    if flat:
        K = {k: sp.Integer(0) for k in K}
    c = c + e * K["L"]
    Ph = Ph + e * K["P"]
    Ss, Sc = sp.diff(S, s), sp.diff(S, chi)
    grad2 = 2 * s / n * Ss ** 2 + Sc ** 2 / (2 * n * s)
    g = sp.zeros(3, 3)
    g[0, 0] = -c ** 2
    g[1, 1] = n / (2 * s) + mu / n * Ss ** 2 / grad2
    g[1, 2] = g[2, 1] = mu / n * Ss * Sc / grad2
    g[2, 2] = 2 * n * s + mu / n * Sc ** 2 / grad2
    g[1, 1] += e * K["A"] / (2 * s)
    g[1, 2] += e * K["B"]
    g[2, 1] += e * K["B"]
    g[2, 2] += e * 2 * s * K["C"]
    g = g.applyfunc(lin)
    if p == "half":                                   # p = q = (n + 1)/2
        U = [sp.log(sp.sqrt(2 * n * s) * sp.cos(chi)) + e * K["V1"] / n,
             sp.log(sp.sqrt(2 * n * s) * sp.sin(chi)) + e * K["V2"] / n]
        m = [(n - 1) / 2, (n - 1) / 2]
    else:                                             # p = 1: y in R (no fibre), z in R^n
        U = [sp.log(sp.sqrt(2 * n * s) * sp.sin(chi)) + e * K["V2"] / n]
        m = [n - 1]
    return g, U, m, Ph


def inverse1(g):
    """inverse to first order in e"""
    g0 = g.subs(e, 0)
    g1 = g.applyfunc(lambda x: sp.diff(x, e).subs(e, 0))
    g0i = g0.inv()
    return (g0i - e * g0i * g1 * g0i).applyfunc(sp.expand)


def residuals(g, U, m, Ph):
    gi = inverse1(g)
    d = lambda f, a: sp.diff(f, X[a])
    Gam = [[[lin(sum(gi[a, k] * (d(g[k, b], c) + d(g[k, c], b) - d(g[b, c], k)) for k in range(3)) / 2)
             for c in range(3)] for b in range(3)] for a in range(3)]
    Ric = sp.zeros(3, 3)
    for b in range(3):
        for c in range(b, 3):
            r = sum(d(Gam[a][b][c], a) - d(Gam[a][b][a], c) for a in range(3))
            r += sum(Gam[a][a][k] * Gam[k][b][c] - Gam[a][c][k] * Gam[k][a][b] for a in range(3) for k in range(3))
            Ric[b, c] = Ric[c, b] = lin(r)

    def hess(f, a, b):
        return d(d(f, a), b) - sum(Gam[k][a][b] * d(f, k) for k in range(3))

    def dot(f, h):
        return lin(sum(gi[a, b] * d(f, a) * d(h, b) for a in range(3) for b in range(3)))

    def box(f):
        return lin(sum(gi[a, b] * hess(f, a, b) for a in range(3) for b in range(3)))

    E = sp.zeros(3, 3)
    for a in range(3):
        for b in range(a, 3):
            v = Ric[a, b] - sum(mk * (hess(Uk, a, b) + d(Uk, a) * d(Uk, b)) for mk, Uk in zip(m, U))
            v -= kap * d(Ph, a) * d(Ph, b)
            E[a, b] = E[b, a] = lin(v)
    Emix = (gi * E).applyfunc(lin)                    # E^a_b
    curv = 0 if "curv" in MUT else 1                    # --mutate=curv drops the fibres' own curvature (m - 1) e^{-2U}
    F = [lin(curv * (mk - 1) * sp.exp(-2 * Uk) - box(Uk) - sum(ml * dot(Uk, Ul) for ml, Ul in zip(m, U)))
         for mk, Uk in zip(m, U)]
    W = lin(box(Ph) + sum(mk * dot(Uk, Ph) for mk, Uk in zip(m, U)))
    return Emix, F, W


if __name__ == "__main__":
    if "--flat" in sys.argv:
        ok = True
        for p in ("half", "one"):
            for dS in (s * sp.cos(2 * chi), s * sp.cos(chi) ** 2, sp.Integer(1)):
                Em, F, W = residuals(*setup(dS, p, flat=True))
                vals = [sp.simplify(x) for x in list(Em) + F + [W]]
                good = all(v == 0 for v in vals)
                ok &= good
                print(f"flat, p {p}, dS {dS}: {'all zero' if good else vals}", flush=True)
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- reduction by the background equations
def rhs():
    """the leading-order background (kappa = 1 normalises phi; the script keeps kappa): c_s = kappa Phi_t^2 / c,
    Phi_s = c^-1 (c^-1 Phi_t)_t, and mu = 2 s c_s / c (E^chi_chi at O(1/n)); s acts as the evolution variable"""
    c, P = c0, Ph0
    return {c0: kap * sp.diff(P, t) ** 2 / c, Ph0: sp.diff(sp.diff(P, t) / c, t) / c}


def reduce_bg(expr):
    """eliminate mu and every s-derivative of c and Phi, leaving t-derivatives at fixed s"""
    R = rhs()
    mu_expr = 2 * s * R[c0] / c0
    expr = expr.subs(mu0, mu_expr).doit() if expr.has(mu0) else expr

    def Ds(x):                                           # total s-derivative along the background equations
        out = sp.diff(x, s)                              # explicit s, and the s-derivatives sympy makes
        return out
    for _ in range(8):
        atoms = [a for a in expr.atoms(sp.Derivative) if a.expr in (c0, Ph0) and any(v == s for v, k in a.variable_count)]
        if not atoms:
            break
        rep = {}
        for a in atoms:
            ks = dict(a.variable_count)
            ns_, nt = ks.get(s, 0), ks.get(t, 0)
            base = R[a.expr]
            for _ in range(ns_ - 1):
                base = sp.diff(base, s)
            rep[a] = sp.diff(base, t, nt) if nt else base
        expr = expr.xreplace(rep)
    return expr


def lead_coeff(expr, var=n):
    """(leading power of n, its coefficient) of a rational function of n, with zero coefficients stripped"""
    x = sp.together(expr)
    num, den = sp.fraction(x)
    pn, pd = sp.Poly(sp.expand(num), var), sp.Poly(sp.expand(den), var)
    cs = pn.all_coeffs()
    for k, cf in enumerate(cs):
        if sp.simplify(cf) != 0:
            return pn.degree() - k - pd.degree(), sp.simplify(cf / pd.LC())
    return None, sp.Integer(0)


# ---------------------------------------------------------------- the e-linear residuals at their leading orders
KNAT = {"E^t_t": 0, "E^t_s": 0, "E^t_chi": 0, "W": 0, "E^s_s": -1, "E^chi_chi": -1, "E^s_chi": -1, "E^chi_s": -1}
IDX = {"E^t_t": (0, 0), "E^t_s": (0, 1), "E^t_chi": (0, 2), "E^s_s": (1, 1), "E^chi_chi": (2, 2), "E^s_chi": (1, 2),
       "E^chi_s": (2, 1)}


def laurent_n(x, kmin):
    """[(power, coefficient)] of the expansion of a rational function of n at n = infinity, powers >= kmin, by power
    series division in 1/n"""
    num, den = sp.fraction(sp.together(x))
    N = sp.Poly(sp.expand(num), n).all_coeffs()                     # highest power first
    D = sp.Poly(sp.expand(den), n).all_coeffs()
    top = (len(N) - 1) - (len(D) - 1)
    K = top - kmin + 1
    if K <= 0:
        return []
    N = N + [sp.Integer(0)] * K
    D = D + [sp.Integer(0)] * K
    q = []
    for k in range(K):                                              # sum_{i<=k} q_i D_{k-i} = N_k
        acc = N[k] - sum(q[i] * D[k - i] for i in range(k))
        q.append(sp.together(acc / D[0]))
    return [(top - k, q[k]) for k in range(K)]


def epart(dS, p="half", corr=None, comps=None):
    """{component: [(power of n, on-shell coefficient)]} of the e-linear residual, every power >= the component's
    natural order (the order of its background equation); an empty list means it vanishes there"""
    Em, F, W = residuals(*setup(dS, p, corr=corr))
    C = {k: Em[i] for k, i in IDX.items()}
    C["W"] = W
    for k, Fk in enumerate(F):
        C[f"F{k}"] = Fk
    out = {}
    for name, x in C.items():
        if comps and name not in comps:
            continue
        kn = KNAT.get(name, -1)
        x1 = sp.diff(x, e).subs(e, 0)
        res = []
        for pw, cf in laurent_n(x1, kn):
            v = sp.simplify(reduce_bg(cf))
            if v != 0:
                res.append((pw, v))
        out[name] = res
    return out


# ---------------------------------------------------------------- the check
def background_onshell(p="half"):
    """the e = 0 residuals at their natural orders, reduced by the background equations: all must vanish"""
    Em, F, W = residuals(*setup(sp.Integer(0), p, corr={}))
    C = {k: Em[i] for k, i in IDX.items()}
    C["W"] = W
    for k, Fk in enumerate(F):
        C[f"F{k}"] = Fk
    bad = {}
    for name, x in C.items():
        for pw, cf in laurent_n(x.subs(e, 0), KNAT.get(name, -1)):
            v = sp.simplify(reduce_bg(cf))
            if v != 0:
                bad.setdefault(name, []).append((pw, v))
    return bad


def control_order_one():
    """the non-harmonic control dS = s cos^2 chi (p = q) with every correction free (A, B, C, V1, V2; lapse L = 0):
    the O(1) equations W and E^t_t involve only V = V1 + V2. W fixes V_t; E^t_t is then a condition on the background
    alone. Returned: that condition evaluated on the admissible local data c = 1, Phi = t at one s (c_s = kappa,
    Phi_s = 0 there): nonzero means no correction rescues the control"""
    gen = {"A": A1, "B": B1, "C": C1, "V1": V1, "V2": V2, "L": sp.Integer(0)}
    out = epart(s * sp.cos(chi) ** 2, "half", gen, comps=["E^t_t", "W"])
    Wc = dict(out["W"]).get(0, sp.Integer(0))
    Ec = dict(out["E^t_t"]).get(0, sp.Integer(0))
    Vt = sp.Symbol("Vt")
    Wl = Wc.subs(sp.Derivative(V1, t), Vt - sp.Derivative(V2, t))
    sol = sp.solve(sp.numer(sp.together(Wl)), Vt)
    if len(sol) != 1:
        return None
    Vt_expr = sol[0]                                          # (V1 + V2)_t from W
    El = Ec.subs({sp.Derivative(V1, (t, 2)): sp.diff(Vt_expr, t) - sp.Derivative(V2, (t, 2)),
                  sp.Derivative(V1, t): Vt_expr - sp.Derivative(V2, t)})
    El = sp.simplify(El)
    test = El.subs({c0: sp.Integer(1), Ph0: t}).doit()
    return sp.simplify(test), El


def check():
    zero = {k: sp.Integer(0) for k in ("A", "B", "C", "V1", "V2", "L")}
    ok = True

    def report(label, out, want_zero):
        nonlocal ok
        allz = all(not v for v in out.values())
        good = allz == want_zero
        ok &= good
        nz = [k for k, v in out.items() if v]
        print(f"{label}: {'all zero' if allz else 'nonzero in ' + ', '.join(nz)}  "
              f"({'as required' if good else '** NOT AS REQUIRED **'})", flush=True)

    bad = background_onshell("half")
    good = not bad
    ok &= good
    print(f"background, p = q, on shell at every natural order: {'all zero' if good else 'nonzero in ' + ', '.join(bad)}"
          f"  ({'as required' if good else '** NOT AS REQUIRED **'})", flush=True)
    report("l = 0 (dS = 1) with A = -mu/s", epart(sp.Integer(1), "half", dict(zero, A=-mu0 / s)), True)
    report("l = 0 (dS = 1) without the correction", epart(sp.Integer(1), "half", dict(zero)), False)
    dS = s * sp.cos(2 * chi)
    report("l = 2, p = q (dS = s cos 2chi) with A = mu cos 2chi", epart(dS, "half", dict(zero, A=mu0 * sp.cos(2 * chi))), True)
    report("l = 2, p = q without the correction", epart(dS, "half", dict(zero)), False)
    dS1 = s * (sp.cos(chi) ** 2 - sp.sin(chi) ** 2 / n)
    report("l = 2, p = 1 (dS = s (cos^2 chi - sin^2 chi / n)) with A = mu dS/s",
           epart(dS1, "one", dict(zero, A=mu0 * dS1 / s)), True)
    report("control, p = q (dS = s cos^2 chi = |y|^2 / (2n p)), no correction", epart(s * sp.cos(chi) ** 2, "half", dict(zero)), False)
    r = control_order_one()
    good = r is not None and r[0] != 0
    ok &= good
    print(f"control, p = q, every correction free: E^t_t after W, at c = 1, Phi = t: "
          f"{r[0] if r else 'no unique V_t'}  ({'nonzero, as required' if good else '** NOT AS REQUIRED **'})", flush=True)
    return ok


if __name__ == "__main__" and "--check" in sys.argv:
    good = check()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)


# ---------------------------------------------------------------- the background to O(1/n) (#1001)
hh = sp.Symbol("h", positive=True)                       # h = 1/n in the on-shell rules


def trunc_h(x, order=1):
    x = sp.expand(x)
    return sum(x.coeff(hh, k) * hh ** k for k in range(order + 1))


def bg_rules1():
    """on-shell rules to O(1/n): c_s, Phi_s, mu with their 1/n parts (from E^t_t, W, E^chi_chi one order below
    natural), the 1/n parts reduced by the leading rules. Returns {c0: c_s, Ph0: Phi_s, mu0: mu} as series in h"""
    Em, F, W = residuals(*setup(sp.Integer(0), "half", corr={}))
    Ett = dict(laurent_n(Em[0, 0].subs(e, 0), -1))
    Ww = dict(laurent_n(W.subs(e, 0), -1))
    Ecc = dict(laurent_n(Em[2, 2].subs(e, 0), -2))
    R0 = rhs()
    Rc1 = sp.simplify(reduce_bg(c0 * Ett[-1]))
    RP1 = sp.simplify(reduce_bg(-Ww[-1]))
    X2 = sp.simplify(reduce_bg(Ecc[-2]))
    Rc = R0[c0] + hh * Rc1
    RP = R0[Ph0] + hh * RP1
    Rmu = 2 * s * Rc / c0 - 2 * s * hh * X2
    return {c0: Rc, Ph0: RP, mu0: trunc_h(Rmu)}


def reduce_bg1(expr, rules, order=1):
    """eliminate mu and every s-derivative of c and Phi with the O(1/n) rules, keeping powers of h up to order"""
    expr = expr.subs(mu0, rules[mu0]).doit() if expr.has(mu0) else expr
    for _ in range(10):
        atoms = [a for a in expr.atoms(sp.Derivative) if a.expr in (c0, Ph0) and any(v == s for v, k in a.variable_count)]
        if not atoms:
            break
        rep = {}
        for a in atoms:
            ks = dict(a.variable_count)
            ns_, nt = ks.get(s, 0), ks.get(t, 0)
            base = rules[a.expr]
            for _ in range(ns_ - 1):
                base = sp.diff(base, s)
            rep[a] = sp.diff(base, t, nt) if nt else base
        expr = trunc_h(expr.xreplace(rep), order)
    return trunc_h(expr, order)


def first_order_map_check(rules=None, drop_1n=False):
    """planted: the O(1/n) background in these variables against notes/FIRST_ORDER.md 1's heat system, through
    C = c / sqrt(1 + mu/n) (= alpha/a), Y = 2 Phi_t / C, S = 2 Phi_s, at kappa = 1; and mu against eta. Each returned
    residual must vanish to O(1/n). drop_1n uses the leading rules only (the control)"""
    R = rules or bg_rules1()
    if drop_1n:
        R = {k: v.subs(hh, 0) for k, v in R.items()}
    R = {k: v.subs(kap, 1) for k, v in R.items()}
    Cf = c0 * (1 - hh * mu0 / 2)                          # c / sqrt(1 + h mu) to O(h)
    Yf = 2 * sp.diff(Ph0, t) / Cf
    Sf = 2 * sp.diff(Ph0, s)
    FO1 = sp.diff(Yf, t) - Cf * Sf - hh * Cf * (Sf + 2 * s * sp.diff(Sf, s) + s * Yf ** 2 * Sf / 2)
    FO3 = sp.diff(Cf, s) - Cf * (Yf ** 2 / 4 + hh * (s * Sf ** 2 / 2 - Yf ** 2 / 2 - s * Yf * sp.diff(Yf, s)))
    eta = s * Yf ** 2 / 2 + hh * (-s * Yf ** 2 / 2 + s ** 2 * Sf ** 2 - 2 * s ** 2 * Yf * sp.diff(Yf, s))
    out = {}
    for name, x in (("Y equation", FO1), ("C equation", FO3), ("mu - eta", mu0 - eta)):
        x = x.subs(hh, 0) + hh * sp.diff(x, hh).subs(hh, 0)       # first order in h
        out[name] = sp.simplify(reduce_bg1(x.subs(kap, 1), R))
    return out


def epart1(dS, p="half", corr=None, comps=None, rules=None, drop_1n=False):
    """{component: (O(1) part, O(1/n) part)} of the e-linear residual at its natural order, on shell with the O(1/n)
    background (rules), i.e. the value of x_natural + x_(natural-1)/n reduced to O(1/n). drop_1n: the leading rules only
    (the control)"""
    R = rules or bg_rules1()
    if drop_1n:
        R = {k: v.subs(hh, 0) for k, v in R.items()}
    Em, F, W = residuals(*setup(dS, p, corr=corr))
    C = {k: Em[i] for k, i in IDX.items()}
    C["W"] = W
    for k, Fk in enumerate(F):
        C[f"F{k}"] = Fk
    out = {}
    for name, x in C.items():
        if comps and name not in comps:
            continue
        kn = KNAT.get(name, -1)
        L = dict(laurent_n(sp.diff(x, e).subs(e, 0), kn - 1))
        if any(pw > kn for pw in L):
            raise ValueError(f"{name}: a term above the natural order needs O(1/n^2) rules")
        v = reduce_bg1(L.get(kn, 0) + hh * L.get(kn - 1, 0), R)
        out[name] = (sp.simplify(v.coeff(hh, 0)), sp.simplify(v.coeff(hh, 1)))
    return out


def l0_first_order_corr():
    """the l = 0 shift that translates FIRST_ORDER's (C, Y, S) exactly and keeps mu = eta to O(1/n): A = -d_s^expl eta,
    lapse L = c A / (2n) (so that C = c (1 - mu/(2n)) translates)"""
    Cf = c0 * (1 - mu0 / (2 * n))
    Yf = 2 * sp.diff(Ph0, t) / Cf
    Sf = 2 * sp.diff(Ph0, s)
    deta = Yf ** 2 / 2 + (-Yf ** 2 / 2 + 2 * s * Sf ** 2 - 4 * s * Yf * sp.diff(Yf, s)) / n
    A = -deta
    return {"A": A, "B": 0, "C": 0, "V1": 0, "V2": 0, "L": c0 * A / (2 * n)}


def l0_first_order_check(rules=None, drop_1n=False):
    """planted: the l = 0 shift's O(1/n) e-parts of W and E^t_t are -(1/2c) and -(1/c) times FIRST_ORDER 4's s-translation
    breaking terms of the Y and C equations, C (2 S_s + Y^2 S/2) and C (S^2/2 - Y Y_s); returns the differences"""
    R = rules or bg_rules1()
    out = epart1(sp.Integer(1), "half", l0_first_order_corr(), ["W", "E^t_t"], R, drop_1n)
    Y_, S_ = 2 * sp.diff(Ph0, t) / c0, 2 * sp.diff(Ph0, s)
    predW = -(2 * sp.diff(S_, s) + Y_ ** 2 * S_ / 2) / 2
    predE = -(S_ ** 2 / 2 - Y_ * sp.diff(Y_, s))
    R1 = {k: v.subs(hh, 0) for k, v in (rules or R).items()}
    res = {}
    for name, pred in (("W", predW), ("E^t_t", predE)):
        o0, o1 = out[name]
        d = sp.simplify(reduce_bg(sp.expand(o1 - pred)).subs(kap, 1))
        res[name] = (sp.simplify(o0.subs(kap, 1)), d)
    return res


# ---------------------------------------------------------------- the O(1/n) step, fast zero tests (#1001)
def numval(expr, point=0):
    """the value of an on-shell expression (t-derivatives of c, Phi at fixed s; s, chi explicit) on explicit test
    functions, to 50 digits: a zero test that avoids simplify. Two points with different functions"""
    T = sp.Symbol("T")
    fs = [(sp.sin(T) + sp.Rational(3, 10) * T ** 2 + 1, 1 + sp.Rational(1, 5) * sp.cos(T), sp.Rational(37, 100),
           sp.Rational(61, 100), sp.Rational(43, 100)),
          (sp.exp(T / 3) - sp.Rational(1, 2) * T ** 3, 2 - sp.sin(2 * T) / 7, sp.Rational(-21, 100),
           sp.Rational(113, 100), sp.Rational(9, 10))][point]
    fP, fc, tv, sv, xv = fs
    rep = {}
    for a in expr.atoms(sp.Derivative):
        if a.expr in (c0, Ph0):
            k = dict(a.variable_count).get(t, 0)
            if dict(a.variable_count).get(s, 0):
                raise ValueError("an s-derivative is left: reduce first")
            rep[a] = sp.diff(fc if a.expr == c0 else fP, T, k)
    x = expr.xreplace(rep).xreplace({c0: fc, Ph0: fP})
    x = x.subs({T: tv, t: tv, s: sv, chi: xv, kap: 1, hh: 0})
    return sp.N(x, 50)


def is_zero_num(expr):
    return all(abs(numval(expr, k)) < 1e-30 for k in (0, 1))


def epart1_raw(dS, p="half", corr=None, comps=None, rules=None, drop_1n=False, dS_is_zero=False):
    """epart1 without simplify: {component: (O(1) part, O(1/n) part)}, on shell with the O(1/n) background"""
    R = rules or bg_rules1()
    if drop_1n:
        R = {k: v.subs(hh, 0) for k, v in R.items()}
    Em, F, W = residuals(*setup(dS, p, corr=corr))
    C = {k: Em[i] for k, i in IDX.items()}
    C["W"] = W
    for k, Fk in enumerate(F):
        C[f"F{k}"] = Fk
    out = {}
    for name, x in C.items():
        if comps and name not in comps:
            continue
        kn = KNAT.get(name, -1)
        L = dict(laurent_n(sp.diff(x, e).subs(e, 0), kn - 1))
        if any(pw > kn for pw in L):
            raise ValueError(f"{name}: a term above the natural order needs O(1/n^2) rules")
        v = reduce_bg1(L.get(kn, 0) + hh * L.get(kn - 1, 0), R)
        out[name] = (v.coeff(hh, 0), v.coeff(hh, 1))
    return out


def deformation_as_corrections(dS, Afix):
    """D2 Z0 written as explicit O(e) corrections at dS = 0: L = dS c_s, P = dS Phi_s, and the e-part of the normal
    metric (mu(S)/n) dS^2/|grad S|^2 + e Afix ds^2/(2s) as A/(2s), B, 2 s C"""
    S = s + e * dS
    Ss, Sc = sp.diff(S, s), sp.diff(S, chi)
    grad2 = 2 * s / n * Ss ** 2 + Sc ** 2 / (2 * n * s)
    mu = mu0 + e * dS * sp.diff(mu0, s)
    d1 = lambda x: sp.diff(x, e).subs(e, 0)
    gss = mu / n * Ss ** 2 / grad2 + e * Afix / (2 * s)
    gsc = mu / n * Ss * Sc / grad2
    gcc = mu / n * Sc ** 2 / grad2
    return {"L": dS * sp.diff(c0, s), "P": dS * sp.diff(Ph0, s), "A": sp.simplify(d1(gss) * 2 * s),
            "B": sp.simplify(d1(gsc)), "C": sp.simplify(d1(gcc) / (2 * s)), "V1": 0, "V2": 0}


def natural_raw(dS, p="half", corr=None, comps=None):
    """{component: on-shell (leading rules) natural-order e-part}, without simplify"""
    Em, F, W = residuals(*setup(dS, p, corr=corr))
    C = {k: Em[i] for k, i in IDX.items()}
    C["W"] = W
    for k, Fk in enumerate(F):
        C[f"F{k}"] = Fk
    out = {}
    for name, x in C.items():
        if comps and name not in comps:
            continue
        kn = KNAT.get(name, -1)
        L = dict(laurent_n(sp.diff(x, e).subs(e, 0), kn))
        out[name] = reduce_bg(L.get(kn, sp.Integer(0)))
    return out


def first_order():
    """#1001: the O(1/n) step, every expectation fixed here. Zero tests by numval at two test points (|x| < 1e-30)"""
    ok = True

    def say(label, good, detail=""):
        nonlocal ok
        ok &= good
        print(f"{label}: {detail}  ({'as required' if good else '** NOT AS REQUIRED **'})", flush=True)

    R = bg_rules1()
    # 1. the background on shell to O(1/n)
    Em, F, W = residuals(*setup(sp.Integer(0), "half", corr={}))
    C = {k: Em[i] for k, i in IDX.items()}
    C["W"] = W
    for k, Fk in enumerate(F):
        C[f"F{k}"] = Fk
    bad = []
    for name, x in C.items():
        kn = KNAT.get(name, -1)
        L = dict(laurent_n(x.subs(e, 0), kn - 1))
        if L and not is_zero_num(reduce_bg1(L.get(kn, 0) + hh * L.get(kn - 1, 0), R).subs(hh, 1)):
            bad.append(name)
    say("background, every component on shell to O(1/n)", not bad, "all zero" if not bad else "nonzero in " + ", ".join(bad))
    # 2. planted 1: FIRST_ORDER.md 1 through the map, and its control
    p1 = first_order_map_check(R)
    c1 = first_order_map_check(R, drop_1n=True)
    say("planted 1, the heat system and eta of FIRST_ORDER.md 1", all(v == 0 for v in p1.values()),
        ", ".join(f"{k} {'0' if v == 0 else 'nonzero'}" for k, v in p1.items()))
    say("planted 1 control, leading rules only", all(v != 0 for v in c1.values()),
        ", ".join(f"{k} {'0' if v == 0 else 'nonzero'}" for k, v in c1.items()))
    # 3. planted 2: the l = 0 shift's O(1/n) forcing, and its control
    Y_, S_ = 2 * sp.diff(Ph0, t) / c0, 2 * sp.diff(Ph0, s)
    pred = {"W": -(2 * sp.diff(S_, s) + Y_ ** 2 * S_ / 2) / 2, "E^t_t": -(S_ ** 2 / 2 - Y_ * sp.diff(Y_, s))}
    for lab, drop in (("planted 2, l = 0 forcing = FIRST_ORDER.md 4's s-translation breaking terms", False),
                      ("planted 2 control, leading background only", True)):
        out = epart1_raw(sp.Integer(1), "half", l0_first_order_corr(), ["W", "E^t_t"], R, drop)
        diffs = {k: reduce_bg(sp.expand(out[k][1] - pred[k])) for k in pred}
        zeros = {k: is_zero_num(v) for k, v in diffs.items()}
        o1zero = all(is_zero_num(out[k][0]) for k in pred)
        want = (not drop)
        say(lab, (all(zeros.values()) == want) and o1zero,
            ", ".join(f"{k} {'equal' if z else 'differs'}" for k, z in zeros.items()) + f"; O(1) parts {'zero' if o1zero else 'nonzero'}")
    # 4. the l = 2 forcing F, and its control
    zero = {k: sp.Integer(0) for k in ("A", "B", "C", "V1", "V2", "L", "P")}
    dS = s * sp.cos(2 * chi)
    corr = dict(zero, A=mu0 * sp.cos(2 * chi))
    out = epart1_raw(dS, "half", corr, None, R)
    outc = epart1_raw(dS, "half", corr, None, R, drop_1n=True)
    o1zero = all(is_zero_num(v[0]) for v in out.values())
    nzF = [k for k, v in out.items() if not is_zero_num(v[1])]
    differ = [k for k in out if not is_zero_num(outc[k][1] - out[k][1])]
    say("l = 2, O(1) parts on the O(1/n) shell", o1zero, "all zero" if o1zero else "nonzero")
    say("l = 2, the O(1/n) forcing F", bool(nzF), "nonzero in " + ", ".join(nzF))
    say("l = 2 control, F with the leading background only", bool(differ), "differs in " + ", ".join(differ))
    # 4b. the same for p = 1, where dS = s (cos^2 chi - sin^2 chi / n) carries a 1/n part itself
    Em1, F1_, W1_ = residuals(*setup(sp.Integer(0), "one", corr={}))
    C1 = {k: Em1[i] for k, i in IDX.items()}
    C1["W"] = W1_
    for k, Fk in enumerate(F1_):
        C1[f"F{k}"] = Fk
    bad1 = []
    for name, x in C1.items():
        kn = KNAT.get(name, -1)
        L = dict(laurent_n(x.subs(e, 0), kn - 1))
        if L and not is_zero_num(reduce_bg1(L.get(kn, 0) + hh * L.get(kn - 1, 0), R).subs(hh, 1)):
            bad1.append(name)
    say("p = 1 background on shell to O(1/n), the same rules", not bad1, "all zero" if not bad1 else "nonzero in " + ", ".join(bad1))
    dS1 = s * (sp.cos(chi) ** 2 - sp.sin(chi) ** 2 / n)
    corr1 = dict(zero, A=mu0 * dS1 / s)
    out1 = epart1_raw(dS1, "one", corr1, None, R)
    out1c = epart1_raw(dS1, "one", corr1, None, R, drop_1n=True)
    say("p = 1, l = 2: O(1) parts on the O(1/n) shell", all(is_zero_num(v[0]) for v in out1.values()), "all zero")
    nz1 = [k for k, v in out1.items() if not is_zero_num(v[1])]
    df1 = [k for k in out1 if not is_zero_num(out1c[k][1] - out1[k][1])]
    say("p = 1, l = 2: the O(1/n) forcing F", bool(nz1), "nonzero in " + ", ".join(nz1))
    say("p = 1 control, F with the leading background only", bool(df1), "differs in " + ", ".join(df1))
    # 5. the operator on the deformation, and F_lambda
    dc = deformation_as_corrections(dS, mu0 * sp.cos(2 * chi))
    L0d = natural_raw(sp.Integer(0), "half", dc)
    allz = all(is_zero_num(v) for v in L0d.values())
    say("L0 on D2 Z0 written as corrections", allz, "all zero" if allz else "nonzero")
    lg = sp.log(-t)
    Fl = natural_raw(sp.Integer(0), "half", {k: lg * v for k, v in dc.items()})
    nzl = [k for k, v in Fl.items() if not all(abs(sp.re(numval(v, q))) < 1e-30 for q in (0, 1))]
    say("F_lambda = L0[ln(-t) D2 Z0]", bool(nzl), "nonzero in " + ", ".join(nzl))
    # 6. the conjugation: on the lapse and the scalar, the l = 2 operator is the l = 0 one times sY (V1 + V2 = 0)
    gam, psi = sp.Function("gamma")(s, t), sp.Function("psi")(s, t)
    Y2 = sp.cos(2 * chi)
    rows2 = natural_raw(sp.Integer(0), "half", dict(zero, L=s * Y2 * gam, P=s * Y2 * psi), ["W", "E^t_t"])
    rows0 = natural_raw(sp.Integer(0), "half", dict(zero, L=gam, P=psi), ["W", "E^t_t"])
    conj = {k: sp.simplify(sp.expand(rows2[k] - s * Y2 * rows0[k])) for k in rows2}
    say("conjugation: W and E^t_t at (sY gamma, sY psi) minus sY times the l = 0 rows", all(v == 0 for v in conj.values()),
        ", ".join(f"{k} {'0' if v == 0 else 'nonzero'}" for k, v in conj.items()))
    if "--print" in sys.argv:
        print("\nL0 (the leading operator on general corrections, on shell):")
        for k, v in natural_raw(sp.Integer(0), "half", None).items():
            print(f"  {k}: {sp.expand(v)}")
        print("\nF (l = 2, O(1/n) forcing, per unit e, on shell):")
        for k, v in out.items():
            print(f"  {k}: {sp.expand(v[1])}")
        print("\nF_lambda:")
        for k, v in Fl.items():
            print(f"  {k}: {sp.expand(v)}")
    return ok


if __name__ == "__main__" and "--first-order" in sys.argv:
    good = first_order()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
