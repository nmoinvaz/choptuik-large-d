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


A1, B1, C1, V1, V2, L1 = (sp.Function(f)(s, t, chi) for f in ("A", "B", "C", "V1", "V2", "L"))


def setup(dS, p="half", flat=False, corr=None):
    """metric (3x3 base), warps U_k with dimensions m_k, and the scalar, for the deformation dS(s, chi). corr: the
    O(e) corrections {A, B, C, V1, V2, L} (default all the unknown functions): g_ss += e A/(2s), g_s chi += e B,
    g_chi chi += 2 s e C (each of relative order 1/n), U_k += e V_k / n, and the lapse c += e L (order one)"""
    S = s + e * dS
    if flat:
        c, mu, Ph = sp.Integer(1), sp.Integer(0), sp.Integer(0)
    else:
        c, mu, Ph = follow(c0, dS), follow(mu0, dS), follow(Ph0, dS)
    K = {"A": A1, "B": B1, "C": C1, "V1": V1, "V2": V2, "L": L1}
    if corr is not None:
        K = {k: corr.get(k, sp.Integer(0)) for k in K}
    if flat:
        K = {k: sp.Integer(0) for k in K}
    c = c + e * K["L"]
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
