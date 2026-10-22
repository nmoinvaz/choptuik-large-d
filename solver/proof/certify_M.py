#!/usr/bin/env python3
"""Point certificate for the M-harmonic inner zero (proof/A_M.md): validated integration plus one Krawczyk test.

The function. For p = (A, wf, B3, C3, ..., BM, CM) (F = A sin t + sum_{nu >= 3} B_nu sin nu t + C_nu cos nu t,
w = wf w_inf, w_inf = 2 pi / 1.64389375) let y+ be the lateral solution continued from the start ray along the path
(ray at angle phi0 from r0 to rho, arc |xi| = rho down to the real axis, real axis from rho to xi1) and
    g(p) = (Re J_1, Im J_1, Re J_3, Im J_3, ..., Re J_M, Im J_M),   J_nu = D+_nu(xi1) - conj(D+_{-nu}(xi1)),
the D-components of the jump at xi1 on the M + 1 real conditions. At M = 1 this is J_D of proof/A_certify.md.

The test (A_M.md section 2). A chart that follows the higher harmonics' zero manifold to first order,
    p = p^ + T x,   T = [[I_2, 0], [C, I_{M-1}]],   C = -(dg_H/dp_H)^-1 dg_H/dp_o at p^   (the tilt),
and the Krawczyk operator of gt(x) = g(p^ + T x) on the box X = [-h, h]^2 x prod [-w_nu, w_nu]^2:
    K(X) = - Y gt(0) + (I - Y Dgt(X)) X,      Y = (mid Dgt(0))^-1 (computed at high precision, then used exactly).
Y is block upper-triangular up to rounding: its outer rows are S^-1 (g_o - L g_H) (the Schur-corrected outer condition,
the leakage L = dg_o/dp_H (dg_H/dp_H)^-1 removed) and its inner rows are the interval Newton for the higher-harmonic
unknowns. Y g and Y Dgt are evaluated by applying the real matrix Y l (l = the jump functionals) to the final Lohner
sets c + Q q (no interval hull is taken before the leakage cancels). If K(X) is in the interior of X, gt has exactly
one zero in X and Dgt is non-singular there (Krawczyk 1969; Moore 1977; Neumaier 1990), i.e. g has exactly one zero in
the parallelepiped P = p^ + T X, and it is simple.

    python3 certify_M.py point M [--p=a,b,...] [--out=f.json] [--log=f.log]     C^1 point run (g, Dg at p^)
    python3 certify_M.py refine M [--p=...] [--iters=k] [--modes=H,full,..] [--tag=t]   Newton refinement of the
                                        centre (non-rigorous numerics) -> data/A_M_refine_M{M}_{tag}.json
    python3 certify_M.py box M (--centre=refine.json [--use_out] | --point=f.json) --h=H --w=W3[,W5] [--out=g.json]
                                        box run over P (tilted columns); its centre-trajectory set also gives g(p^)
    python3 certify_M.py krawczyk --box=g.json [--point=f.json] [--mutate=...] [--theta=t] [--hier]   the test
    python3 certify_M.py controls --box=g.json [--point=f.json]                 all mutation controls (exit 0 iff all
                                                                               verdicts are as expected)
    python3 certify_M.py krawczyk --point=f.json --box=g.json --hier [--theta=t]   the test, then the comparison with
                                                                               hier.py's zeros (--mutate=hier: must fail)
    python3 certify_M.py startcheck M [--mutate]                               rigorous start data vs this file's series
    python3 certify_M.py m1 [--mutate=m1ref]                                    M = 1 reproduction of Theorem A(M=1)
"""
import json
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_mat, arb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_M_sys as CS  # noqa: E402
import certify_M_lohner as CL  # noqa: E402

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
PREC = 256                 # integration
KPREC = 768                # Krawczyk assembly (Y and the products with the Lohner sets)

# numerical zeros (double precision) used as starting points: M = 1 Theorem A(M = 1)'s centre; M = 3 nested/hier
# (solver/inner_pert/logs/first_order_M3.log); M = 5 hier.py --schur (solver/inner_pert/logs/first_order_hier_M5.log)
START = {
    1: ["2.5421480475", "0.7769940136"],
    3: ["2.7264758856199967", "1.0156030948470995", "0.14441947847113112", "0.011087425888287268"],
    5: ["2.718579189229888", "0.9992284031945021", "0.1492035749173264", "0.01471905249087869",
        "0.008405346906375517", "0.0014703451817031647"],
}
# hier.py's zeros for the containment control (two independent double-precision solves each; their spread is the
# numerical accuracy): M = 3 hier (first_order_hier_M3.log) and nested (first_order_M3.log); M = 5 hier at rtol 1e-12
# and 1e-13 (first_order_hier_M5.log, first_order_hier_M5_rtol1e-13.log)
HIER = {
    1: [["2.5421480475188947", "0.7769940136467028"]],
    3: [["2.726475885636556", "1.0156030948641483", "0.1444194784693144", "0.01108742588800987"],
        ["2.7264758856199967", "1.0156030948470995", "0.14441947847113112", "0.011087425888287268"]],
    5: [["2.718579189229888", "0.9992284031945021", "0.1492035749173264", "0.01471905249087869",
         "0.008405346906375517", "0.0014703451817031647"],
        ["2.718579189212617", "0.9992284032109982", "0.14920357491042185", "0.014719052490104231",
         "0.00840534690564208", "0.0014703451815662712"]],
}
# Theorem A(M = 1) (proof/A_certify.md 12.6): the certified zero enclosure on the lower path, h = 1e-9
THM_A1 = {"A": ("2.54214804761338", "2.54214804761975"), "wf": ("0.776994013607438", "0.776994013612010")}


R0 = {1: "0.1", 3: "0.0741", 5: "0.0622"}
ARG = {1: "3pi/8", 3: "5pi/16", 5: "5pi/16"}
NTAY, NJAC = 20, 12          # Taylor orders of the trajectory and of the step Jacobian (M >= 3); M = 1 uses 24, 24
FPM_DIR = os.environ.get("CERTIFY_M_FPM_DIR")   # where fixed_point_M.py lives, if not next to this file


def path_for(M):
    """(r0, phi0, rho, xi1). M = 1: the lower path of Theorem A(M = 1) (ray 3pi/8 from 0.1). M >= 3: the start ray of
    inner_np.py / hier.py, arg xi = 5pi/16, from r0 = inner_np's r0_for (0.0741 at M = 3, 0.0622 at M = 5), where the
    M-harmonic start-value lemma has small Cauchy bounds (fixed_point_M.py's table), then the same lower path: ray to
    0.2, arc at 0.2, real axis to 0.3."""
    phi0 = (3 if M == 1 else arb(5) / 2) * arb.pi() / 8
    return (R0[M], phi0, "0.2", "0.3")


def orders(M):
    return (24, 24) if M == 1 else (NTAY, NJAC)


def fpm_module():
    """fixed_point_M (the M-harmonic start-value lemma, another agent's file) if importable, else None"""
    if FPM_DIR and FPM_DIR not in sys.path:
        sys.path.insert(0, FPM_DIR)
    try:
        import fixed_point_M
        return fixed_point_M
    except ImportError:
        return None


def start_mode(M):
    if M == 1:
        return "fk"
    return "fpm" if fpm_module() is not None and os.environ.get("CERTIFY_M_START") != "placeholder" else "placeholder"


def exact_p(pstr):
    """the centre p^ as exact binary numbers: the midpoints of the decimal strings rounded at PREC bits (within
    2^-PREC relative of the decimals). Point run, box run and the assembly all use this same exact point."""
    old = ctx.prec
    ctx.prec = PREC
    p = [arb(arb(x).mid()) for x in pstr]
    ctx.prec = old
    return p


# ---------------------------------------------------------------- start data

def start_data(M, p, dirs, box=False):
    """values of y+(xi0) at p (point) and the derivative columns along dirs, over p (balls if box).
    M = 1: fixed_point.start_data (rigorous, proof/A_certify.md 4.4-4.5); M >= 3: certify_M_sys.placeholder_start
    (NOT rigorous). Returns (values at the centre, columns, w-components of the columns, info)."""
    path = path_for(M)
    W0 = CS.W0()
    if start_mode(M) == "fk":
        import fixed_point as FP
        ctx.prec = 320
        A, wf = p[0], p[1]
        _, vals_c, _, _, eta_c = FP.start_data(arb(A.mid()), arb(wf.mid()), path[0], +1, arb("1e-3"))
        if box:
            _, _, dAs, dws, eta_x = FP.start_data(A, wf, path[0], +1, arb("1e-3"))
        else:
            _, _, dAs, dws, eta_x = FP.start_data(arb(A.mid()), arb(wf.mid()), path[0], +1, arb("1e-3"))
        ctx.prec = PREC
        cols = [[acb(dv[0]) * dAs[i] + acb(dv[1]) * dws[i] for i in range(len(dAs))] for dv in dirs]
        wcol = [W0 * dv[1] for dv in dirs]
        info = {"start": "fixed_point.start_data (rigorous)", "eta": float(eta_c[0].upper()),
                "eta_der": float(eta_x[1].upper()), "rigorous": True}
        return vals_c, cols, wcol, info
    if start_mode(M) == "fpm":
        import hashlib
        FPM = fpm_module()
        src = FPM.__file__
        sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
        ctx.prec = 320
        pc = [arb(x.mid()) for x in p]
        _, vals_c, ders_c, eta_c = FPM.start_data_M(pc, M, path[0], ARG[M], +1, "1e-3", diag=True)
        if box:
            _, _, ders, eta_x = FPM.start_data_M([arb(x) for x in p], M, path[0], ARG[M], +1, "1e-3", diag=True)
        else:
            ders, eta_x = ders_c, eta_c
        ctx.prec = PREC
        m = M + 1
        cols = [[sum((acb(dv[i]) * ders[i][k] for i in range(m)), acb(0)) for k in range(len(vals_c))] for dv in dirs]
        wcol = [W0 * dv[1] for dv in dirs]
        info = {"start": f"fixed_point_M.start_data_M (rigorous given that lemma; {src}, sha256 {sha}), arg {ARG[M]}, "
                         f"diag splitting", "eta": float(eta_c[0].upper()), "eta_der": float(eta_x[1].upper()),
                "rigorous": True, "fpm_sha256": sha}
        return vals_c, cols, wcol, info
    ctx.prec = PREC
    pc = [arb(x.mid()) for x in p]
    _, vals_c, _, info = CS.placeholder_start(pc, M, path[0], path[1], dirs=[])
    _, _, cols, info2 = CS.placeholder_start(p, M, path[0], path[1], dirs=dirs)
    wcol = [W0 * dv[1] for dv in dirs]
    info = dict(info)
    info["start"] = "placeholder: truncated series + ASSUMED error ball (NOT rigorous)"
    return vals_c, cols, wcol, info


# ---------------------------------------------------------------- jump functionals

def jump_rows(sys_, mutate=None):
    """the real (M + 1) x 2n matrix l with g = l z, z = (Re y, Im y) (n = nd + 1, w included):
    Re J_nu = Re D_nu - Re D_-nu, Im J_nu = Im D_nu + Im D_-nu (J_nu = D_nu - conj D_-nu), nu = 1, 3, .., M.
    Mutations: 'sign' (J = D_nu + conj D_-nu), 'noconj' (J = D_nu - D_-nu), 'negconj' (-conj J, null),
    'scale' (J_nu -> (nu + 1) J_nu, null)."""
    n, M = sys_.n, sys_.M
    rows = []
    for nu in range(1, M + 1, 2):
        ip, im = sys_.idx[("D", nu)], sys_.idx[("D", -nu)]
        re, imr = [0] * (2 * n), [0] * (2 * n)
        a, b = (1, -1)                       # Re: D_nu, D_-nu
        c, d = (1, 1)                        # Im: D_nu, D_-nu
        if mutate == "sign":
            a, b, c, d = 1, 1, 1, -1
        elif mutate == "noconj":
            a, b, c, d = 1, -1, 1, -1
        re[ip], re[im] = a, b
        imr[n + ip], imr[n + im] = c, d
        if mutate == "negconj":              # -conj(J): Re -> -Re J, Im -> +Im J
            re = [-x for x in re]
        if mutate == "scale":
            re = [(nu + 1) * x for x in re]
            imr = [(nu + 1) * x for x in imr]
        rows += [re, imr]
    return arb_mat([[arb(x) for x in r] for r in rows])


def apply_set(R, c, Q, q):
    """R (c + Q q) for a Lohner set in real coordinates: R stack(c) + (R Q) q, all in arb (R a real arb_mat)"""
    n = len(c)
    z = arb_mat([[c[i].real] for i in range(n)] + [[c[i].imag] for i in range(n)])
    qm = arb_mat([[x] for x in q])
    v = R * z + (R * Q) * qm
    return [v[i, 0] for i in range(v.nrows())]


def apply_var(R, var):
    """R (cV_k + QV qV_k) for every derivative column k: returns the (rows x m) list of arb"""
    cV, QV, qV = var
    n, m = cV.nrows(), cV.ncols()
    Z = arb_mat(2 * n, m)
    for i in range(n):
        for j in range(m):
            Z[i, j], Z[n + i, j] = cV[i, j].real, cV[i, j].imag
    V = R * Z + (R * QV) * qV
    return [[V[i, j] for j in range(m)] for i in range(V.nrows())]


# ---------------------------------------------------------------- runs

def point_run(M, pstr, out=None, log=None, verbose=False):
    """C^1 point run at p^: the centre set and the M + 1 columns dy/dp_k along the single trajectory"""
    t0 = time.time()
    ctx.prec = PREC
    s = CS.SysM(M)
    p = exact_p(pstr)
    m = M + 1
    dirs = [[arb(1) if i == k else arb(0) for i in range(m)] for k in range(m)]
    vals, cols, wcol, info = start_data(M, p, dirs, box=False)
    ctx.prec = PREC
    NT, NJ = orders(M)
    lo = CL.start_set(s, vals, CS.W0() * p[1], cols, wcol, dx=None, N=NT, NJ=NJ)
    path = path_for(M)
    if log:
        with open(log, "a") as fh:
            fh.write(f"point run M = {M} at p^ = {pstr}; start: {info['start']}\n")
    steps, _ = CL.integrate(lo, path, verbose=verbose, log=log)
    res = {"kind": "point", "M": M, "p": list(pstr), "path": [path[0], ARG[M], path[2], path[3]], "start": info,
           "steps": steps, "sec": time.time() - t0, "prec": PREC, "sets": CL.dump_lohner(lo)}
    g, Dg = g_Dg(s, lo.c, lo.Q, lo.q, lo.var)
    res["g"] = [CL.s_arb(x) for x in g]
    res["Dg"] = [[CL.s_arb(x) for x in row] for row in Dg]
    res["g_rad"] = max(float(x.rad()) for x in g)
    res["Dg_rad"] = max(float(x.rad()) for row in Dg for x in row)
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh)
    return res


def g_Dg(s, c, Q, q, var, mutate=None):
    ell = jump_rows(s, mutate)
    g = apply_set(ell, c, Q, q)
    Dg = apply_var(ell, var) if var is not None else None
    return g, Dg


def tilt_from(Dg_mid, M):
    """C = -(Dg_HH)^-1 Dg_Ho (the tangent of the zero manifold of g_H at p^), from the point run's Dg midpoints;
    returned as a list of rows of arb points (exact)"""
    m = M + 1
    if M == 1:
        return []
    o, H = [0, 1], list(range(2, m))
    DHH = arb_mat([[Dg_mid[i][j] for j in H] for i in H])
    DHo = arb_mat([[Dg_mid[i][j] for j in o] for i in H])
    Cm = -(DHH.solve(DHo))
    return [[arb(Cm[i, j].mid()) for j in range(2)] for i in range(len(H))]


def T_matrix(Cm, M):
    m = M + 1
    T = [[arb(1) if i == j else arb(0) for j in range(m)] for i in range(m)]
    for i, row in enumerate(Cm):
        for j in range(2):
            T[2 + i][j] = row[j]
    return T


def box_run(M, pstr, Cm, h, ws, out=None, log=None, verbose=False, max_steps=None):
    """C^1 box run over P = p^ + T X, X = [-h, h]^2 x prod [-w, w]^2 (tilted columns dy/dx = dy/dp T); Cm the tilt
    (rows of exact arb), from a point run or the refinement at (or near) p^"""
    t0 = time.time()
    ctx.prec = KPREC
    s = CS.SysM(M)
    m = M + 1
    Cm = [[arb(x) for x in r] for r in Cm]
    T = T_matrix(Cm, M)
    widths = [arb(h), arb(h)] + [arb(ws[(i // 2) % len(ws)]) for i in range(M - 1)]
    # hull of P: p_o in p^_o + [-h, h], p_H in p^_H + C [-h, h] + [-w, w]
    p = exact_p(pstr)
    pbox = [p[i] + arb(0, widths[i]) if i < 2 else
            p[i] + sum((T[i][j] * arb(0, widths[j]) for j in range(2)), arb(0)) + arb(0, widths[i])
            for i in range(m)]
    dirs = [[T[i][k] for i in range(m)] for k in range(m)]
    ctx.prec = PREC
    vals, cols, wcol, info = start_data(M, pbox, dirs, box=True)
    ctx.prec = PREC
    dx = [acb(arb(0, wd)) for wd in widths]
    NT, NJ = orders(M)
    lo = CL.start_set(s, vals, CS.W0() * p[1], cols, wcol, dx=dx, N=NT, NJ=NJ)
    path = path_for(M)
    if log:
        with open(log, "a") as fh:
            fh.write(f"box run M = {M} at p^ = {pstr}, h = {h}, w = {ws}; start: {info['start']}\n")
    lo.max_steps = max_steps
    try:
        steps, _ = CL.integrate(lo, path, verbose=verbose, log=log)
    except CL.PilotStop as e:
        V = lo.var_box()
        X = lo.box()
        res = {"kind": "pilot", "M": M, "p": list(pstr), "h": h, "w": list(ws), "stopped_at": str(e),
               "steps": lo.total_steps, "sec": time.time() - t0, "sec_per_step": (time.time() - t0) / lo.total_steps,
               "state_rad": max(CL.VI.rad_of(x) for x in X[:s.nd]),
               "der_rad": [max(CL.VI.rad_of(V[i, j]) for i in range(s.nd)) for j in range(V.ncols())]}
        if out:
            with open(out, "w") as fh:
                json.dump(res, fh, indent=1)
        print(f"pilot stopped after {res['steps']} steps at {res['stopped_at']}: {res['sec_per_step']:.1f} s/step; "
              f"state radius {res['state_rad']:.2e}; derivative radii per column " +
              ", ".join(f"{x:.1e}" for x in res["der_rad"]))
        return res
    res = {"kind": "box", "M": M, "p": list(pstr), "h": h, "w": list(ws), "C": [[CL.s_arb(x) for x in r] for r in Cm],
           "path": [path[0], ARG[M], path[2], path[3]], "start": info, "steps": steps, "sec": time.time() - t0,
           "prec": PREC, "sets": CL.dump_lohner(lo)}
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh)
    return res


# ---------------------------------------------------------------- the Krawczyk test

def krawczyk_test(bx, pt=None, mutate=None, theta=None, verbose=True):
    """K(X) from a saved box run (Dgt over X, and g(p^) from its centre-trajectory set) and optionally a point run at
    the same centre (then g(p^) and Y come from it). mutate: None, or one of 'sign', 'noconj', 'negconj', 'scale'
    (function mutations), 'plant_out', 'plant_in', 'plant_H' (planted zeros), 'widen' (g(p^) widened). theta: shrink
    the box to theta X (Dgt(X) is valid over every sub-box centred at 0). Returns a dict with ok, K, margins, the zero
    enclosure in p-coordinates."""
    if pt is not None:
        assert pt["p"] == bx["p"] and pt["M"] == bx["M"]
    M = bx["M"]
    m = M + 1
    ctx.prec = KPREC
    s = CS.SysM(M)
    fmut = mutate if mutate in ("sign", "noconj", "negconj", "scale") else None
    ell = jump_rows(s, fmut)
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    T = T_matrix(Cm, M)
    Tm = arb_mat(T)
    cb, Qb, qb, varb = CL.load_lohner(bx["sets"])
    if pt is not None:                             # g(p^) from the point run's centre set; Y from its columns
        c, Q, q, var = CL.load_lohner(pt["sets"])
        Dgt0 = arb_mat(apply_var(ell, var)) * Tm
    else:                                          # both from the box run: its centre set, its column midpoints
        c, Q, q = cb, Qb, qb
        Dgt0 = arb_mat(apply_var(ell, varb))
    g = apply_set(ell, c, Q, q)
    Dmid = arb_mat([[arb(Dgt0[i, j].mid()) for j in range(m)] for i in range(m)])
    Yb = Dmid.inv()
    Y = arb_mat([[arb(Yb[i, j].mid()) for j in range(m)] for i in range(m)])
    Yl = Y * ell
    Yg = apply_set(Yl, c, Q, q)
    YJ = apply_var(Yl, varb)                       # Y Dgt(X): the box run's columns are the tilted ones
    h = arb(bx["h"])
    widths = [h, h] + [arb(bx["w"][(i // 2) % len(bx["w"])]) for i in range(M - 1)]
    if theta is not None:
        widths = [arb(theta) * x for x in widths]
    # planted zeros: g -> g - L d with L the exact midpoint of Dgt(X) (moves the zero by d up to second order)
    if mutate in ("plant_out", "plant_in", "plant_H"):
        J_box = apply_var(ell, varb)
        L = arb_mat([[arb(J_box[i][j].mid()) for j in range(m)] for i in range(m)])
        d = [arb(0)] * m
        if mutate == "plant_out":
            d[0] = 2 * widths[0]
        elif mutate == "plant_in":
            d[0], d[1] = widths[0] / 2, -3 * widths[1] / 10
            for i in range(2, m):
                d[i] = widths[i] / 4
        else:
            d[2] = 2 * widths[2] if m > 2 else 2 * widths[0]
        Ld = L * arb_mat([[x] for x in d])
        YLd = Y * Ld
        Yg = [Yg[i] - YLd[i, 0] for i in range(m)]
    if mutate == "widen":
        # g(p^) widened by 10^3 h |S| in every outer component (S ~ the smallest singular value of the outer block)
        sv = np.linalg.svd(np.array([[float(Dgt0[i, j].mid()) for j in range(2)] for i in range(2)]), compute_uv=False)
        wid = 1e3 * float(h.mid()) * sv[-1]
        gw = [g[i] + (arb(0, wid) if i < 2 else arb(0)) for i in range(m)]
        Ygw = Y * arb_mat([[x] for x in gw])
        Yg = [Ygw[i, 0] for i in range(m)]
    K = []
    for i in range(m):
        t = -Yg[i]
        for j in range(m):
            Mij = (arb(1) if i == j else arb(0)) - YJ[i][j]
            t = t + Mij * arb(0, widths[j])
        K.append(t)
    ok = all(abs(K[i]) < widths[i] for i in range(m))
    margins = [float((widths[i] - abs(K[i]).upper()).lower()) for i in range(m)]
    rel = [mg / float(widths[i].mid()) for i, mg in enumerate(margins)]
    # the zero in p-coordinates: p* in p^ + T K
    p = exact_p(bx["p"])
    pstar = [p[i] + sum((T[i][j] * K[j] for j in range(m)), arb(0)) for i in range(m)]
    res = {"ok": bool(ok), "K": K, "margins": margins, "rel_margins": rel, "pstar": pstar, "widths": widths,
           "g_rad": max(float(x.rad()) for x in g), "YJ_rad": max(float(YJ[i][j].rad()) for i in range(m) for j in range(m)),
           "Yg": Yg, "YJ": YJ, "T": T, "p_hat": p}
    if verbose:
        tag = f" [mutation {mutate}]" if mutate else ""
        print(f"Krawczyk M = {M}{tag}: K(X) in int X: {ok}; relative margins " + ", ".join(f"{r:+.3f}" for r in rel))
    return res


def outward(x, digits=15):
    """decimal strings of the endpoints of the ball x, rounded outward to `digits` significant digits"""
    from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
    out = []
    for e, mode, sgn in ((x.lower(), ROUND_FLOOR, -1), (x.upper(), ROUND_CEILING, +1)):
        d = Decimal(e.str(60, radius=False))         # within 1e-59 relative of the exact endpoint
        nudge = abs(d) * Decimal("1e-55") + Decimal("1e-300")
        d = d + sgn * nudge                           # move outward past the conversion error
        q = Decimal(1).scaleb(d.adjusted() - digits + 1)
        out.append(str(d.quantize(q, rounding=mode)))
    return out


def describe(res, M):
    names = ["A", "w/w_inf"] + [f"{b}{nu}" for nu in range(3, M + 1, 2) for b in ("B", "C")]
    lines = []
    for i, nm in enumerate(names):
        x = res["pstar"][i]
        lo_, hi_ = outward(x)
        lines.append(f"   {nm:8s} in [{lo_}, {hi_}]  (radius {float(x.rad()):.2e})")
    return "\n".join(lines)


def hull_contains(res, pvals):
    """does the certified zero enclosure (p-coordinates) contain the point pvals (strings)?"""
    return all(res["pstar"][i].contains(arb(v)) for i, v in enumerate(pvals))


def export_6a(bx, out):
    """the state box B1 at xi1 over the certified parallelepiped (every state component), in the format lemma 6a(M)
    reads (lemma6_jump_M.py --box=..., keys M, h, xi1, yX_xi1, as lemma6_path_box_1e-9.json): the box of the final
    Lohner set c + Q q + V dx, V = cV + QV qV, which contains y+(xi1; p) for every p in P"""
    ctx.prec = PREC
    s = CS.SysM(bx["M"])
    c, Q, q, var = CL.load_lohner(bx["sets"])
    lo = CL.LohnerM(s, c, Q, q, 1, var)
    lo.dx = [CL.p_acb(z) for z in bx["sets"]["dx"]]
    X = lo.box()
    V = lo.var_box()
    d = {"M": bx["M"], "h": bx["h"], "w": bx["w"], "xi1": bx["path"][3], "path": bx["path"], "p": bx["p"],
         "C": bx["C"], "start": bx["start"], "yX_xi1": [CL.s_acb(x) for x in X[:s.nd]],
         "V": [[CL.s_acb(V[i, j]) for j in range(V.ncols())] for i in range(s.nd)],
         "note": "V = d(state)/d(x) over P in the tilted coordinates x (p = p^ + T x, T = [[I, 0], [C, I]])"}
    with open(out, "w") as fh:
        json.dump(d, fh)
    print(f"exported the state box at xi1 = {d['xi1']} (max radius {max(CL.VI.rad_of(x) for x in X[:s.nd]):.2e}) to {out}")


def hier_check(res, M, mutate=False, verbose=True):
    """the certified zero against hier.py's double-precision zeros (HIER[M], two solves each, their spread = hier's
    accuracy). Two statements, both must hold:
      (a) hier's zeros lie in the axis-aligned hull of the certified parallelepiped P = p^ + T X;
      (b) in every coordinate they agree with the certified zero enclosure p^ + T K(X) to within 3x hier's spread.
    (Literal containment in P itself is not expected: P is thinner in the higher-harmonic directions than hier's own
    accuracy there.) mutate: hier's zeros moved by 100x the spread (at least 1e-9) in A: must fail."""
    ctx.prec = KPREC
    pts = [[arb(v) for v in pt] for pt in HIER[M]]
    m = M + 1
    spread = [max(abs(float(pts[a][i].mid()) - float(pts[b][i].mid())) for a in range(len(pts)) for b in range(len(pts)))
              for i in range(m)]
    spread = [max(s, 1e-15 * abs(float(pts[0][i].mid()))) for i, s in enumerate(spread)]
    if mutate:
        sh = max(100 * spread[0], 1e-9)
        pts = [[p[0] + sh] + p[1:] for p in pts]
    T = res["T"]
    hull = []
    for i in range(m):
        acc = arb(0)
        for j in range(m):
            acc = acc + T[i][j] * arb(0, res["widths"][j])
        hull.append(res["p_hat"][i] + acc)
    ok_a = all(hull[i].contains(p[i]) for p in pts for i in range(m))
    dev = [max(float(abs(p[i] - res["pstar"][i]).upper()) for p in pts) for i in range(m)]
    ok_b = all(dev[i] <= float(res["pstar"][i].rad()) + 3 * spread[i] for i in range(m))
    if verbose:
        names = ["A", "w/w_inf"] + [f"{b}{nu}" for nu in range(3, M + 1, 2) for b in ("B", "C")]
        print(f"hier.py check M = {M}{' [mutated: hier zeros shifted]' if mutate else ''}: (a) inside the hull of P: "
              f"{ok_a}; (b) agreement within 3x hier's spread: {ok_b}")
        for i, nm in enumerate(names):
            print(f"   {nm:8s} |hier - certified| <= {dev[i]:.2e}   hier spread {spread[i]:.1e}   hull half-width "
                  f"{float(hull[i].rad()):.1e}")
    return ok_a and ok_b


def start_check(M, mutate=False):
    """the rigorous start data (fixed_point_M.start_data_M) must contain the midpoints of this file's own series
    (certify_M_sys.series_generic_M, independent code) at the start point; mutate: the series values shifted by 1e-40
    (relative to O(1) values, far above both radii): must fail"""
    if fpm_module() is None:
        print("start check: fixed_point_M.py not importable (set CERTIFY_M_FPM_DIR); skipped")
        return None
    p = exact_p(START[M])
    m = M + 1
    dirs = [[arb(1) if i == k else arb(0) for i in range(m)] for k in range(m)]
    vf, cf, _, info = start_data(M, p, dirs)
    old = os.environ.get("CERTIFY_M_START")
    os.environ["CERTIFY_M_START"] = "placeholder"
    vp, cp, _, _ = start_data(M, p, dirs)
    if old is None:
        del os.environ["CERTIFY_M_START"]
    else:
        os.environ["CERTIFY_M_START"] = old
    sh = arb("1e-40") if mutate else arb(0)
    okv = all(vf[i].contains(acb(vp[i].mid()) + sh) for i in range(len(vf)))
    okd = all(cf[k][i].contains(acb(cp[k][i].mid())) for k in range(m) for i in range(len(vf)))
    dv = max(float(abs(vf[i].mid() - vp[i].mid())) for i in range(len(vf)))
    print(f"start check M = {M}{' [mutated]' if mutate else ''}: {info['start']}\n   contains this file's series values: "
          f"{okv} (max difference {dv:.1e}, radius {max(float(v.rad()) for v in vf):.1e}); derivatives: {okd}")
    return okv and okd


# ---------------------------------------------------------------- refinement (non-rigorous numerics)

def newton_steps(g, Dg, M):
    """the Newton steps from g and Dg (lists of arb points): ('H', higher harmonics only at fixed (A, wf)) and
    ('full'). Computed at KPREC: Dg has condition up to about 1e15 in these raw units."""
    ctx.prec = KPREC
    m = M + 1
    D = arb_mat([[arb(x) for x in row] for row in Dg])
    sF = D.solve(arb_mat([[x] for x in g]))
    full = [-sF[k, 0] for k in range(m)]
    if M == 1:
        return full, full
    H = list(range(2, m))
    DHH = arb_mat([[D[i, j] for j in H] for i in H])
    st = DHH.solve(arb_mat([[g[i]] for i in H]))
    hstep = [arb(0), arb(0)] + [-st[k, 0] for k in range(len(H))]
    return hstep, full


def numerics_point(M, pstr, log=None):
    """g and Dg at p (NOT rigorous: plain high-precision Taylor integration from the series start values), midpoints"""
    ctx.prec = PREC
    s = CS.SysM(M)
    p = exact_p(pstr)
    m = M + 1
    path = path_for(M)
    old = os.environ.get("CERTIFY_M_START")
    os.environ["CERTIFY_M_START"] = "placeholder"
    if M == 1:
        _, vals, ders, _ = CS.placeholder_start(p, M, path[0], path[1])
    else:
        dirs = [[arb(1) if i == k else arb(0) for i in range(m)] for k in range(m)]
        vals, cols_, _, _ = start_data(M, p, dirs)
        ders = cols_
    if old is None:
        del os.environ["CERTIFY_M_START"]
    else:
        os.environ["CERTIFY_M_START"] = old
    ctx.prec = PREC
    W0 = CS.W0()
    cols = [list(ders[k]) + [acb(W0 if k == 1 else 0)] for k in range(m)]
    y, Wm, steps = CL.numerics_run(s, vals, W0 * p[1], cols, path, N=NUM_N, tol=NUM_TOL, log=log)
    ell = jump_rows(s)
    n = s.n
    z = arb_mat([[y[i].real] for i in range(n)] + [[y[i].imag] for i in range(n)])
    Z = arb_mat(2 * n, m)
    for i in range(n):
        for j in range(m):
            Z[i, j], Z[n + i, j] = Wm[i, j].real, Wm[i, j].imag
    gv, Dv = ell * z, ell * Z
    g = [arb(gv[i, 0].mid()) for i in range(m)]
    Dg = [[arb(Dv[i, j].mid()) for j in range(m)] for i in range(m)]
    return g, Dg, steps


NUM_N, NUM_TOL = 30, 1e-32


def refine(M, pstr, iters, tag, modes=None):
    """Newton refinement of the centre (NOT rigorous numerics; any centre is allowed). Iteration k: g, Dg at p_k by
    numerics_point, then an 'H' step (higher harmonics only, at fixed (A, wf)) or a 'full' step; default modes:
    H, full, H, full, ... Saves data/A_M_refine_M{M}_{tag}.json after every iteration (p_in, p_out, the tilt C at
    p_in, |g|, the step sizes), which the box run reads (--centre=...)."""
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"A_M_refine_M{M}_{tag}.log")
    hist = []
    for it in range(iters):
        t0 = time.time()
        g, Dg, steps = numerics_point(M, pstr, log=log)
        sH, sF = newton_steps(g, Dg, M)
        mode = (modes[it] if modes else ("H" if it % 2 == 0 else "full")) if M > 1 else "full"
        step = sH if mode == "H" else sF
        ctx.prec = KPREC
        newp = [exact_p([pstr[i]])[0] + step[i] for i in range(M + 1)]
        newstr = [x.mid().str(50, radius=False) for x in newp]
        Cm = tilt_from(Dg, M)
        rec = {"it": it, "p_in": list(pstr), "mode": mode, "p_out": newstr,
               "g": [x.str(20) for x in g], "Dg": [[x.str(20) for x in r] for r in Dg],
               "full_step": [x.str(10) for x in sF], "H_step": [x.str(10) for x in sH],
               "steps": steps, "sec": time.time() - t0}
        hist.append(rec)
        msg = (f"refine M = {M} it {it}: |g| = {max(abs(float(x)) for x in g):.3e}; H step "
               f"{max([abs(float(x)) for x in sH[2:]] + [0.0]):.2e}; full step (outer) "
               f"{max(abs(float(x)) for x in sF[:2]):.2e}, (inner) {max([abs(float(x)) for x in sF[2:]] + [0.0]):.2e} "
               f"-> {mode} step; {sum(steps)} steps, {time.time() - t0:.0f} s")
        print(msg, flush=True)
        with open(log, "a") as fh:
            fh.write(msg + "\n   p_out = " + ", ".join(newstr) + "\n")
        with open(os.path.join(DATA, f"A_M_refine_M{M}_{tag}.json"), "w") as fh:
            json.dump({"M": M, "p_in": list(pstr), "p_out": newstr, "C": [[x.str(60) for x in r] for r in Cm],
                       "history": hist, "note": "non-rigorous numerics (certify_M_lohner.numerics_run)"}, fh, indent=1)
        pstr = newstr
    return pstr


# ---------------------------------------------------------------- controls

CONTROLS = [
    ("C0", None, True, "clean"),
    ("C1", "sign", False, "jump sign flipped, D_nu + conj D_-nu"),
    ("C2", "noconj", False, "conjugation dropped, D_nu - D_-nu"),
    ("C3", "negconj", True, "-conj J (null: same zeros)"),
    ("C4", "scale", True, "J_nu -> (nu + 1) J_nu (null: same zeros)"),
    ("C5", "plant_out", False, "zero planted outside, moved by 2h in A"),
    ("C6", "plant_in", True, "zero planted inside, moved by (h/2, -3h/10, w/4, ...)"),
    ("C7", "widen", False, "g(p^) widened by 1e3 h |S| in the outer components"),
    ("C8", "plant_H", False, "zero planted outside in the first higher-harmonic coordinate, moved by 2w"),
]


def run_controls(bx, pt=None):
    ok_all = True
    rows = []
    clean = None
    for name, mut, expect, desc in CONTROLS:
        if mut == "plant_H" and bx["M"] == 1:
            continue
        r = krawczyk_test(bx, pt, mut, verbose=False)
        if mut is None:
            clean = r
        verdict = r["ok"] == expect
        extra = ""
        if mut in ("negconj", "scale") and r["ok"] and clean is not None:
            same = all(r["pstar"][i].overlaps(clean["pstar"][i]) for i in range(len(r["pstar"])))
            verdict &= same
            extra = f"; zero enclosure overlaps C0's: {same}"
        ok_all &= verdict
        rows.append((name, desc, "pass" if expect else "fail", "pass" if r["ok"] else "fail", r["rel_margins"], verdict))
        print(f"{name} {desc}: expected {'pass' if expect else 'fail'}, got {'pass' if r['ok'] else 'fail'} "
              f"(relative margins {', '.join(f'{x:+.2e}' for x in r['rel_margins'])}){extra}  "
              f"{'OK' if verdict else 'UNEXPECTED'}", flush=True)
    print("CONTROLS " + ("ALL AS EXPECTED" if ok_all else "SOME UNEXPECTED"))
    return ok_all, rows


def plan(d):
    """the block structure of Dg at the last refinement point (raw jump units at xi1): the outer Schur complement S,
    the inner block Dg_HH, the leakage L = Dg_oH Dg_HH^-1, the tilt C, the Newton step sizes. Numerics, not rigorous."""
    M = d["M"]
    rec = d["history"][-1]
    D = np.array([[float(arb(x).mid()) for x in r] for r in rec["Dg"]])
    print(f"M = {M}, refinement iteration {rec['it']} ({rec['mode']} step), p = {rec['p_in']}")
    print(f"   |g| = {max(abs(float(arb(x).mid())) for x in rec['g']):.3e}; full Newton step: outer "
          f"{max(abs(float(arb(x).mid())) for x in rec['full_step'][:2]):.2e}, inner "
          f"{max([abs(float(arb(x).mid())) for x in rec['full_step'][2:]] + [0.0]):.2e}")
    if M == 1:
        print(f"   Dg singular values {np.linalg.svd(D, compute_uv=False)}")
        return
    o, H = [0, 1], list(range(2, M + 1))
    DHH, DHo, DoH, Doo = D[np.ix_(H, H)], D[np.ix_(H, o)], D[np.ix_(o, H)], D[np.ix_(o, o)]
    L = DoH @ np.linalg.inv(DHH)
    S = Doo - L @ DHo
    C = -np.linalg.solve(DHH, DHo)
    np.set_printoptions(precision=3, linewidth=150)
    print(f"   |Dg_oo| (raw outer block) max {np.abs(Doo).max():.2e}; Schur complement S singular values "
          f"{np.linalg.svd(S, compute_uv=False)}")
    print(f"   Dg_HH singular values {np.linalg.svd(DHH, compute_uv=False)}; |Dg_oH| max {np.abs(DoH).max():.2e}")
    print(f"   leakage L = Dg_oH Dg_HH^-1, max |entry| {np.abs(L).max():.2e}; tilt C =\n{C}")
    print(f"   condition of Dg {np.linalg.cond(D):.2e}; of the tilted, block-triangular Dg T: "
          f"{np.linalg.cond(np.block([[S, DoH], [np.zeros((M - 1, 2)), DHH]])):.2e}")


def centre_and_tilt(o, M):
    """the centre p^ and the tilt C for a box run: --centre=refine.json (the refinement's final centre and the tilt of
    its last Newton run) or --point=point.json (a point run's centre and its Dg)"""
    if "centre" in o:
        with open(o["centre"]) as fh:
            d = json.load(fh)
        ctx.prec = KPREC
        return d["p_out"] if "use_out" in o else d["p_in"], [[arb(x) for x in r] for r in d["C"]]
    with open(o["point"]) as fh:
        pt = json.load(fh)
    ctx.prec = KPREC
    Dg_mid = [[arb(arb(x).mid()) for x in row] for row in pt["Dg"]]
    return pt["p"], tilt_from(Dg_mid, M)


def opts_of(argv):
    return {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}


def main():
    argv = sys.argv[1:]
    mode = argv[0]
    o = opts_of(argv)
    pos = [a for a in argv[1:] if not a.startswith("--")]
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    if mode == "point":
        M = int(pos[0])
        pstr = o["p"].split(",") if "p" in o else START[M]
        r = point_run(M, pstr, out=o.get("out"), log=o.get("log"), verbose="verbose" in o)
        print(f"point run M = {M}: steps {r['steps']}, {r['sec']:.0f} s, g radius {r['g_rad']:.1e}, Dg radius {r['Dg_rad']:.1e}")
        print("   g = " + ", ".join(arb(x).str(8, radius=True) for x in r["g"]))
        return 0
    if mode == "refine":
        M = int(pos[0])
        pstr = o["p"].split(",") if "p" in o else START[M]
        modes = o["modes"].split(",") if "modes" in o else None
        refine(M, pstr, int(o.get("iters", 3)), o.get("tag", "a"), modes)
        return 0
    if mode == "box":
        M = int(pos[0])
        pstr, Cm = centre_and_tilt(o, M)
        ws = o.get("w", "0").split(",")
        r = box_run(M, pstr, Cm, o["h"], ws, out=o.get("out"), log=o.get("log"), verbose="verbose" in o,
                    max_steps=int(o["pilot"]) if "pilot" in o else None)
        print(f"box run M = {M}: steps {r['steps']}, {r['sec']:.0f} s")
        return 0
    if mode in ("krawczyk", "controls"):
        pt = None
        if "point" in o:
            with open(o["point"]) as fh:
                pt = json.load(fh)
        with open(o["box"]) as fh:
            bx = json.load(fh)
        M = bx["M"]
        if mode == "controls":
            ok, _ = run_controls(bx, pt)
            return 0 if ok else 1
        th = o.get("theta")
        r = krawczyk_test(bx, pt, o.get("mutate"), theta=th)
        print(f"   start values: {pt['start']['start']}")
        print(f"   g(p^) radius {r['g_rad']:.1e}; max radius of Y Dgt(X) {r['YJ_rad']:.1e}")
        print(f"   box: h = {bx['h']}, w = {bx['w']}{', shrunk by theta = ' + th if th else ''}")
        print(describe(r, M))
        if "hier" in o and r["ok"]:
            hok = hier_check(r, M, mutate=(o.get("mutate") == "hier"))
            return 0 if hok else 1
        return 0 if r["ok"] else 1
    if mode == "plan":
        with open(o["centre"]) as fh:
            d = json.load(fh)
        plan(d)
        return 0
    if mode == "export6a":
        with open(o["box"]) as fh:
            bx = json.load(fh)
        export_6a(bx, o["out"])
        return 0
    if mode == "startcheck":
        ok = start_check(int(pos[0]), mutate="mutate" in o)
        return 0 if ok else 1
    if mode == "m1":
        return m1_reproduce(o)
    print(__doc__)
    return 2


def m1_reproduce(o):
    """M = 1 through this code path: the lower path of Theorem A(M = 1), fixed_point.start_data, h = 1e-9; the zero
    enclosure must overlap Theorem A(M = 1)'s (and lie inside X)"""
    pf = os.path.join(DATA, "A_M_point_M1.json")
    bf = os.path.join(DATA, "A_M_box_M1_1e-9.json")
    log = os.path.join(LOGS, "A_M_m1.log")
    if not os.path.exists(pf) or "rerun" in o:
        point_run(1, START[1], out=pf, log=log)
    with open(pf) as fh:
        pt = json.load(fh)
    if not os.path.exists(bf) or "rerun" in o:
        box_run(1, START[1], [], "1e-9", ["0"], out=bf, log=log)
    with open(bf) as fh:
        bx = json.load(fh)
    r = krawczyk_test(bx, pt)
    print(describe(r, 1))
    ref = THM_A1
    if o.get("mutate") == "m1ref":       # control: a reference shifted by 1e-11 must not be reproduced
        ref = {"A": ("2.54214804762338", "2.54214804762975"), "wf": ref["wf"]}
    ctx.prec = KPREC
    refA = arb(ref["A"][0]).union(arb(ref["A"][1]))
    refw = arb(ref["wf"][0]).union(arb(ref["wf"][1]))
    ovA, ovw = r["pstar"][0].overlaps(refA), r["pstar"][1].overlaps(refw)
    good = r["ok"] and ovA and ovw
    print(f"M = 1 reproduction: Krawczyk passes {r['ok']}; zero enclosure overlaps Theorem A(M = 1)'s: A {ovA}, "
          f"w/w_inf {ovw}  -> {'REPRODUCED' if good else 'NOT REPRODUCED'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
