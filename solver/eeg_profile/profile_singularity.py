#!/usr/bin/env python3
"""The type of the D = infinity profile's singularity at theta_0 from the inner equations (#397; pre-registered on
the issue).

Stage 1 (wtc): the local (Weiss-Tabor-Carnevale) expansion of the gauge-fixed inner equations (DINF_ANALYTIC.md 1.1,
form A), Y_s + (V Y)_u = Y_uu, V_u = Y^2/4, about a movable singular manifold psi = u - f(s) = 0, with ln psi and
ln^2 psi terms allowed at every order; and of the planted ODE (A0), Y' = V Y + k1, V' = Y^2/4 (the s-independent
reduction, whose resonance condition -k1/2 = 0 fails for k1 != 0). For each order: the coefficients fixed, the free
ones, and the log terms forced.

    cd solver/eeg_profile
    python3 profile_singularity.py wtc [--order=4]
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "wtc":
        sys.exit(0 if cmd_wtc() else 1)
    raise SystemExit(__doc__)
