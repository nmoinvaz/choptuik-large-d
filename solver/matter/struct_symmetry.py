"""(d) Continuous (Lie point) symmetries of the D = infinity inner problems, and the growth rates they generate.

    nice -n 10 python3 struct_symmetry.py           (a few minutes)

Inner systems (struct_inner.py; first-order form, physical inner variables s = r^2/(2n) and t):
  multiplet, l fixed, psi = vphi s^{-l/2}, Q = psi_t / c:   psi_t = c Q,   Q_t = c psi_s,   c_s = 2 s^l c Q^2
      (the same system as d_t(vphi_t/c) = c (vphi_s - l vphi/(2 s)), c_s = 2 vphi_t^2/c, by the point map vphi = s^{l/2} psi;
       l = 0 is GAMMA.md's heat system in potential form; the Yang-Mills U-system is l = 1)
  wave map, P = u_t / c:   u_t = c P,   P_t = c (u_s - sin(2u)/(4 s)),   c_s = 2 eta c P^2
  complex scalar (phi1 + i phi2; the target-space family with no potential):  phi_k,t = c P_k,  P_k,t = c phi_k,s,
      c_s = 2 c (P_1^2 + P_2^2)

Method: the infinitesimal invariance condition of the first prolongation, on solutions (principal derivatives psi_t,
Q_t, c_s eliminated), split by the parametric derivatives psi_s, Q_s, c_t; infinitesimals polynomial in all variables
up to total degree DEG (times {1, e^{2iu}, e^{-2iu}} for the wave map, whose u-dependence is trigonometric); the linear
system over QQ (QQ(i) split into real and imaginary parts) is solved exactly. The solution space is printed as
vector fields and graded by the self-similar scaling Z: a generator with [Z, X] = -lambda X, applied to the self-
similar background, gives a perturbation proportional to e^{lambda tau}.

The search is complete only within the ansatz (polynomial infinitesimals of bounded degree); a non-polynomial point
symmetry, or a generalized/nonlocal one, would not be seen.
"""
import itertools
import sys
import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def lie_point_symmetries(X, U, principal, deg, trig=None, deg_trig=None):
    """X, U: lists of symbols; principal: {(i, j): F} meaning d U[i] / d X[j] = F.
    Returns a list of vector fields {var: coefficient} spanning the solutions within the ansatz."""
    V = list(X) + list(U)
    d = {(m, k): sp.Symbol(f"{U[m]}_{X[k]}") for m in range(len(U)) for k in range(len(X))}
    param = [key for key in d if key not in principal]
    mons = []
    for tot in range(deg + 1):
        for e in itertools.product(range(tot + 1), repeat=len(V)):
            if sum(e) == tot:
                mons.append(sp.Mul(*[v ** p for v, p in zip(V, e)]))
    if trig is not None:   # functions of trig: multiply polynomial monomials (of lower degree) by e^{+-2 i u}
        u = trig
        extra = []
        for mm in mons:
            if sp.Poly(mm, *V).total_degree() <= deg_trig:
                extra += [mm * sp.cos(2 * u), mm * sp.sin(2 * u)]
        mons = mons + extra
    comps = list(V)            # xi^k for X, eta^m for U

    def Dtot(f, k):
        return sp.diff(f, X[k]) + sum(d[(m, k)] * sp.diff(f, U[m]) for m in range(len(U)))

    on_shell = {d[key]: F for key, F in principal.items()}

    def condition(field):
        xi = [field.get(x_, 0) for x_ in X]
        eta = [field.get(u_, 0) for u_ in U]

        def eta_pr(m, k):
            return Dtot(eta[m], k) - sum(d[(m, j)] * Dtot(xi[j], k) for j in range(len(X)))
        out = []
        for (i, j), F in principal.items():
            c = eta_pr(i, j) - (sum(xi[k] * sp.diff(F, X[k]) for k in range(len(X)))
                                + sum(eta[m] * sp.diff(F, U[m]) for m in range(len(U)))
                                + sum(eta_pr(m, k) * sp.diff(F, d[(m, k)]) for (m, k) in param))
            out.append(c.subs(on_shell))
        return out

    w = sp.Symbol("w")
    gens = list(V) + [d[key] for key in param] + ([w] if trig is not None else [])
    cols = []
    rows = {}
    unknowns = [(comp, mm) for comp in comps for mm in mons]
    for col, (comp, mm) in enumerate(unknowns):
        conds = condition({comp: mm})
        for ie, ce in enumerate(conds):
            if trig is not None:
                ce = sp.expand(ce.rewrite(sp.exp)).subs(sp.exp(2 * sp.I * trig), w).subs(sp.exp(-2 * sp.I * trig), 1 / w)
                ce = sp.expand(ce * w ** 4)
            num = sp.numer(sp.together(ce * sp.Symbol("s") ** 0))
            num, den = sp.fraction(sp.together(ce))
            if num == 0:
                continue
            P = sp.Poly(sp.expand(num * sp.Symbol("_den") / sp.Symbol("_den")), *gens)
            # the denominator must not depend on the unknown: it is a fixed power of s (and w); normalize by it
            dd = sp.Poly(den, *gens)
            key_den = dd.as_expr()
            for mon, coef in P.as_dict().items():
                for part, val in (("re", sp.re(coef)), ("im", sp.im(coef))):
                    if val != 0:
                        rows.setdefault((ie, key_den, mon, part), {})[col] = val
    # different denominators for the same equation must be put over a common one; we avoid that by requiring
    # each (equation, denominator) block to vanish separately, which is valid because the denominators are monomials
    # in s and w, and we multiply each block by its own denominator monomial before comparing:
    merged = {}
    for (ie, den, mon, part), row in rows.items():
        dpoly = sp.Poly(den, *gens)
        if len(dpoly.as_dict()) != 1:
            raise RuntimeError("non-monomial denominator: " + str(den))
        (dmon, dcoef), = dpoly.as_dict().items()
        maxden = tuple(8 for _ in gens)
        shifted = tuple(m + (M - dm) for m, M, dm in zip(mon, maxden, dmon))
        tgt = merged.setdefault((ie, shifted, part), {})
        for col, val in row.items():
            tgt[col] = tgt.get(col, 0) + sp.Rational(val) / sp.Rational(dcoef)
    R = [[QQ(sp.Rational(r.get(c, 0)).p, sp.Rational(r.get(c, 0)).q) for c in range(len(unknowns))] for r in merged.values()]
    M = DomainMatrix(R, (len(R), len(unknowns)), QQ)
    N = M.nullspace().to_Matrix()
    fields = []
    for i in range(N.rows):
        fv = {}
        for col, (comp, mm) in enumerate(unknowns):
            if N[i, col] != 0:
                fv[comp] = fv.get(comp, 0) + N[i, col] * mm
        fields.append({k: sp.simplify(v) for k, v in fv.items() if sp.simplify(v) != 0})
    return fields, len(unknowns), M.shape[0]


def bracket(A, B, V):
    out = {}
    for v in V:
        e = sum(A.get(w, 0) * sp.diff(B.get(v, 0), w) - B.get(w, 0) * sp.diff(A.get(v, 0), w) for w in V)
        e = sp.simplify(e)
        if e != 0:
            out[v] = e
    return out


def show(f, V):
    return " + ".join(f"({sp.factor(f[v])}) d_{v}" for v in V if v in f) or "0"


def weight(Z, X, V):
    """lambda with [Z, X] = -lambda X, or None if X is not homogeneous."""
    B = bracket(Z, X, V)
    lam = None
    for v in V:
        a_, b_ = X.get(v, 0), B.get(v, 0)
        if a_ == 0 and b_ == 0:
            continue
        if a_ == 0:
            return None
        q = sp.simplify(-b_ / a_)
        if q.free_symbols:
            return None
        if lam is None:
            lam = q
        elif sp.simplify(q - lam) != 0:
            return None
    return lam


def run(name, X, U, principal, Z, deg, trig=None, deg_trig=None):
    print(f"\n== {name}  (polynomial degree <= {deg}" + (f", trigonometric part degree <= {deg_trig}" if trig is not None else "") + ")")
    V = list(X) + list(U)
    fields, nunk, nrow = lie_point_symmetries(X, U, principal, deg, trig, deg_trig)
    print(f"   {nunk} unknown coefficients, {nrow} linear conditions, solution space of dimension {len(fields)}")
    # Z must be a symmetry; grade the solutions by ad_Z (they are homogeneous combinations after the nullspace basis
    # is diagonalized; the monomial basis makes most of them homogeneous already)
    ZZ = {k: v for k, v in Z.items() if v != 0}
    lams = []
    for f in fields:
        lam = weight(ZZ, f, V)
        lams.append(lam)
        print(f"   lambda = {str(lam):>4}:  " + show(f, V))
    return fields, lams


s, t = sp.symbols("s t", positive=True)
results = {}
for l in (0, 1, 2, 3):
    psi, Q, c = sp.symbols("psi Q c")
    principal = {(0, 1): c * Q, (1, 1): c * sp.Symbol("psi_s"), (2, 0): 2 * s ** l * c * Q ** 2}
    Z = {s: 2 * s, t: t, psi: -l * psi, Q: -(l + 1) * Q, c: 0}
    results[l] = run(f"multiplet l = {l}:  psi_t = c Q, Q_t = c psi_s, c_s = 2 s^{l} c Q^2", [s, t], [psi, Q, c], principal, Z, DEG)

u, P, c = sp.symbols("u P c")
eta = sp.Rational(3, 7)          # a generic value; the result does not depend on it
principal = {(0, 1): c * P, (1, 1): c * (sp.Symbol("u_s") - sp.sin(2 * u) / (4 * s)), (2, 0): 2 * eta * c * P ** 2}
Z = {s: 2 * s, t: t, u: 0, P: -P, c: 0}
results["wm"] = run("wave map (eta = 3/7):  u_t = c P, P_t = c (u_s - sin(2u)/(4s)), c_s = 2 eta c P^2",
                    [s, t], [u, P, c], principal, Z, min(DEG, 3), trig=u, deg_trig=1)

if DEG >= 2:
    f1, f2, P1, P2, c = sp.symbols("f1 f2 P1 P2 c")
    principal = {(0, 1): c * P1, (1, 1): c * P2, (2, 1): c * sp.Symbol("f1_s"), (3, 1): c * sp.Symbol("f2_s"),
                 (4, 0): 2 * c * (P1 ** 2 + P2 ** 2)}
    Z = {s: 2 * s, t: t, f1: 0, f2: 0, P1: -P1, P2: -P2, c: 0}
    results["cs"] = run("complex scalar:  f_t = c P, P_t = c f_s (two components), c_s = 2 c |P|^2",
                        [s, t], [f1, f2, P1, P2, c], principal, Z, 2)

print("\nSummary (positive growth rates generated by point symmetries; lambda = 1 is the time translation):")
ok = True
for key, (fields, lams) in results.items():
    pos = sorted({l_ for l_ in lams if l_ is not None and l_ > 0})
    print(f"   {str(key):>3}: lambda > 0 from symmetries: {pos}")
    has2 = 2 in pos
    expect2 = key in (0, "cs")
    ok = ok and (has2 == expect2) and all(l_ is not None for l_ in lams)
print("\nlambda = 2 (s-translation) present exactly for l = 0 and the complex scalar:", "ok" if ok else "FAIL")
print("\nall ok" if ok else "\nSOME CHECKS FAILED")
