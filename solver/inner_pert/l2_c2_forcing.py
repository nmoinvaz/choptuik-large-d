#!/usr/bin/env python3
"""The l = 2 forcing F/(sY) for the c2 computation, from l2_mode.py's O(1/n) reduction (notes/L2_MODE.md 8, 9; #1010).

On the lapse and the scalar the l = 2 problem is the l = 0 problem conjugated by sY (Y = cos 2chi, V1 + V2 = 0,
L2_MODE.md 8.3), so the correction W2 = sY w obeys the l = 0 equations at lambda = 2 forced by Ft = F/(sY). This
script extracts Ft in the W and E^t_t rows as rational functions of the t-derivatives of Phi and c at fixed s
(symbols P1..P8 = Phi_t, Phi_tt, ..., C0..C7 = c, c_t, ...; kappa = 1), checks that they carry no chi (l2 and
l0 carry no explicit s either; the control does, s = T^2 xi^2/2 with the same lambda = 2 weight), and
writes them to data/l2_c2_forcing.json for l2_c2.py. Four forcings:

    l2        the l = 2 deformation's O(1/n) residual (L2_MODE.md 8.3), divided by s cos 2chi
    l2_ctrl   the same with the leading background only (the control of #1010)
    l0        the l = 0 shift's (L2_MODE.md 8.2, FIRST_ORDER.md 4's s-translation breaking terms; the planted case)
    l2_lam    F_lambda = L0[ln(-t) D2 Z0], divided by s cos 2chi (a check of the d/dlambda normalisation)
    G         L0[(0, Phi_s)]: checked here to equal Ft(l2) + Ft(l0) identically (with the control (c_s, 0))

    python3 l2_c2_forcing.py          about 10 min; exit 0 iff every check below holds
"""
import json
import os
import sys
import time

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import l2_mode as M2                                    # noqa: E402
sys.argv = _argv

t, s, chi, kap = M2.t, M2.s, M2.chi, M2.kap
c0, Ph0 = M2.c0, M2.Ph0
NP = 8
P = sp.symbols(f"P1:{NP + 1}")                          # Phi_t, Phi_tt, ... (t-derivatives at fixed s)
C = sp.symbols(f"C0:{NP}")                              # c, c_t, c_tt, ...
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def symbolize(expr):
    """t-derivatives of Phi and c at fixed s -> P_k, C_k; kappa -> 1"""
    rep = {}
    for a in expr.atoms(sp.Derivative):
        if a.expr in (c0, Ph0):
            vc = dict(a.variable_count)
            if vc.get(s, 0):
                raise ValueError("an s-derivative is left")
            k = vc.get(t, 0)
            rep[a] = P[k - 1] if a.expr == Ph0 else C[k]
    x = expr.xreplace(rep).xreplace({c0: C[0]}).subs(kap, 1)
    if x.has(Ph0):
        raise ValueError("undifferentiated Phi")
    return x


def desymbolize(x):
    rep = {P[k]: sp.diff(Ph0, t, k + 1) for k in range(NP)}
    rep.update({C[k]: sp.diff(c0, t, k) if k else c0 for k in range(NP)})
    return x.xreplace(rep)


def reduce_rows(rows):
    """{row: F} -> {row: F / (s cos 2chi)} as a rational function of P, C (t, s allowed), checked free of chi.
    chi is set to the rational point cos chi = 3/5, sin chi = 4/5 (cos 2chi = -7/25), so every coefficient is rational"""
    out = {}
    for k, F in rows.items():
        x = sp.expand_trig(symbolize(F)).subs({sp.cos(chi): sp.Rational(3, 5), sp.sin(chi): sp.Rational(4, 5)})
        if x.has(chi):
            raise ValueError(f"{k}: chi left after the rational point")
        x0 = sp.cancel(sp.together(x / (s * sp.Rational(-7, 25))))
        free = x0.free_symbols - set(P) - set(C) - {t, s}       # s = T^2 xi^2 / 2 is allowed (the control has it)
        if free:
            raise ValueError(f"{k}: explicit {free} left")
        x0 = sp.factor(x0)
        # the reduction holds at other (s, chi): F - s cos 2chi Ft vanishes on both numval test points
        d = F - s * sp.cos(2 * chi) * desymbolize(x0)
        if not M2.is_zero_num(d):
            raise ValueError(f"{k}: F / (s cos 2chi) depends on s or chi")
        out[k] = x0
    return out


def main():
    ok = True
    R = M2.bg_rules1()
    log("O(1/n) background rules")
    zero = {k: sp.Integer(0) for k in ("A", "B", "C", "V1", "V2", "L", "P")}
    dS = s * sp.cos(2 * chi)
    corr = dict(zero, A=M2.mu0 * sp.cos(2 * chi))
    rows = ["W", "E^t_t"]
    res = {}
    for name, drop in (("l2", False), ("l2_ctrl", True)):
        out = M2.epart1_raw(dS, "half", corr, rows, R, drop_1n=drop)
        o1 = all(M2.is_zero_num(out[k][0]) for k in rows)
        ok &= o1
        res[name] = reduce_rows({k: out[k][1] for k in rows})
        log(f"{name}: O(1) parts {'zero' if o1 else 'NONZERO'}; Ft = " + "; ".join(f"{k}: {v}" for k, v in res[name].items()))
    # l = 0 shift: no sY factor (dS = 1); Ft = the O(1/n) part itself
    out = M2.epart1_raw(sp.Integer(1), "half", M2.l0_first_order_corr(), rows, R)
    o1 = all(M2.is_zero_num(out[k][0]) for k in rows)
    ok &= o1
    res["l0"] = {k: sp.factor(symbolize(out[k][1])) for k in rows}
    log(f"l0: O(1) parts {'zero' if o1 else 'NONZERO'}; Ft = " + "; ".join(f"{k}: {v}" for k, v in res["l0"].items()))
    # F_lambda = L0[ln(-t) D2 Z0]: ln(-t) -> its derivative 1/t only (the log parts cancel, L0 annihilates D2 Z0)
    dc = M2.deformation_as_corrections(dS, M2.mu0 * sp.cos(2 * chi))
    lg = sp.Symbol("lg")
    Fl = M2.natural_raw(sp.Integer(0), "half", {k: sp.Function("ln")(t) * v for k, v in dc.items()}, rows)
    Fl2 = {}
    for k, v in Fl.items():
        v = v.doit()
        v = v.replace(lambda e_: isinstance(e_, sp.Derivative) and e_.expr == sp.Function("ln")(t),
                      lambda e_: {1: 1 / t, 2: -1 / t ** 2, 3: 2 / t ** 3}[dict(e_.variable_count)[t]])
        v = v.subs(sp.Function("ln")(t), lg)
        v = sp.expand(v)
        lgpart = v.coeff(lg, 1)
        if not M2.is_zero_num(lgpart):
            ok = False
            log(f"F_lambda {k}: the ln(-t) part does not cancel")
        Fl2[k] = v.coeff(lg, 0)
    res["l2_lam"] = reduce_rows(Fl2)
    log("l2_lam: Ft = " + "; ".join(f"{k}: {v}" for k, v in res["l2_lam"].items()))
    # the decomposition Ft(l2) = -Ft(l0) + L0[(0, Phi_s)] (L0: the l = 0 rows; (0, Phi_s) is the scalar part of the
    # s-translation, a local field of the background); control: (c_s, 0), the lapse part, must not do it
    res["G"] = {}
    for lab, V, want in (("(0, Phi_s)", dict(zero, P=sp.diff(Ph0, s)), True),
                         ("control (c_s, 0)", dict(zero, L=sp.diff(c0, s)), False)):
        Lr = M2.natural_raw(sp.Integer(0), "half", V, rows)
        for k in rows:
            Gk = sp.cancel(symbolize(Lr[k]))
            d = sp.cancel(res["l2"][k] + res["l0"][k] - Gk)
            good = (d == 0) == want
            ok &= good
            if want:
                res["G"][k] = sp.factor(Gk)
            log(f"Ft(l2) + Ft(l0) - L0[{lab}], {k}: {'0' if d == 0 else d}  ({'as required' if good else 'NOT AS REQUIRED'})")
    # the control must differ from l2
    for k in rows:
        d = sp.cancel(res["l2"][k] - res["l2_ctrl"][k])
        good = d != 0
        ok &= good
        log(f"control, {k}: l2 - l2_ctrl {'nonzero' if good else 'ZERO'}")
    fn = os.path.join(HERE, "data", "l2_c2_forcing.json")
    json.dump({name: {k: sp.srepr(v) for k, v in d.items()} for name, d in res.items()}
              | {"symbols": dict(P="Phi_t..Phi_tttt at fixed s", C="c..c_ttt at fixed s", kappa=1,
                                 rows="W and E^t_t, per unit e, divided by s cos 2chi (l0: dS = 1)")},
              open(fn, "w"), indent=1)
    log(f"wrote {fn}")
    return ok


if __name__ == "__main__":
    good = main()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
