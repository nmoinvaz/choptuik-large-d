#!/usr/bin/env python3
"""Second-order (C^2) point certificate for the M-harmonic inner zero (proof/A_M.md section 10).

The function g and the test are those of certify_M.py (g = the D-jumps at xi1 = 0.3 on the M + 1 real conditions);
what changes is the chart and the enclosure of the Jacobian over the box.

Chart (second order). p(x) = p^ + T x + 1/2 (0, Gamma[x_o, x_o]),  T = [[I_2, 0], [C, I_{M-1}]], x = (x_o, x_H):
the inner unknowns are measured from the quadratic approximation of the zero manifold of g_H. C is the tilt of
certify_M.py; Gamma_i,jk (i an inner index, j, k outer) is chosen from numerics (certify_M2_fd.py) so that
D^2_oo gt_H(0) is about 0. Any C and Gamma are allowed: the test is what is rigorous. p(.) is injective with
invertible derivative (unit lower triangular), so zeros of gt(x) = g(p(x)) in X are zeros of g in p(X) and conversely.

Enclosure (mean value form at second order). A C^2 run (certify_M2_lohner.LohnerM2) gives, at xi1,
    gt(0) in l(c + Q q),   Dgt(0) in l V0,   D^2 gt(x) in l W(X) for all x in X
(l the real jump functionals). For x in X, Dgt(x) = Dgt(0) + int_0^1 D^2 gt(s x)[., x] ds, so the interval matrix
    J(X)_ij = (l V0)_ij + sum_k (l W(X))_{i,jk} X_k
contains Dgt(x) for every x in X. The Krawczyk operator with Y = (mid l V0)^-1 (an exact matrix),
    K(X) = -Y gt(0) + (I - Y J(X)) X,   (I - Y J(X))_ij = delta_ij - (Y l V0)_ij - sum_k (Y l W(X))_{i,jk} X_k,
evaluated with Y l applied to the Lohner sets before any interval product (as certify_M.py). If K(X) is in int X,
gt has exactly one zero in X and every matrix in J(X), in particular Dgt(x), is non-singular (Krawczyk 1969; Moore
1977; Neumaier 1990, as in certify_M.py): g has exactly one zero in p(X), and it is simple.

    python3 certify_M2.py chart M --centre=refine.json [--use_out] --fd=data/A_M2_fd_M{M}.json [--out=f]
        the chart (p^, C from the refinement record, Gamma from the finite differences) -> data/A_M2_chart_M{M}.json
    python3 certify_M2.py box M --chart=chart.json --h=H --w=W3[,W5] --out=g.json --log=f.log [--pilot=steps]
                              [--mutate=w0|chart|lin2]
        the C^2 run over X = [-H, H]^2 x prod [-W, W]^2 (checkpointed: resumes from g.json.ckpt)
    python3 certify_M2.py krawczyk --box=g.json [--sub=h,w3[,w5]] [--best] [--shrink] [--hier] [--mutate=...]
    python3 certify_M2.py controls --box=g.json [--sub=...]        C0-C8 of certify_M.py on the C^2 test
    python3 certify_M2.py d2check --box=g.json --fd=data/A_M2_fd_M{M}.json   control D2: l W(X) at xi1 must contain
                                                                  the independent finite-difference D^2 gt(0)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_mat, arb_mat, acb_series, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_M_sys as CS  # noqa: E402
import certify_M_lohner as CL  # noqa: E402
import certify_M as CM  # noqa: E402
import certify_M2_lohner as C2  # noqa: E402

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
PREC, KPREC = CM.PREC, CM.KPREC
CAUCHY = "1e-3"
MUTATE = set()          # "w0": the start values' second derivatives dropped; "chart": the chart term dropped


def diag_for(M):
    """the splitting of fixed_point_M: 'plain' at M = 1 (fixed_point.py exactly), 'diag' at M >= 3 (certify_M.py's)"""
    return M >= 3


# ---------------------------------------------------------------- the chart

def load_chart(f):
    """(p^ strings, C rows of exact arb, Gamma rows [g_00, g_01, g_11] of exact arb, the record)"""
    with open(f) as fh:
        d = json.load(fh)
    ctx.prec = KPREC
    Cm = [[arb(x) for x in r] for r in d["C"]]
    G = [[arb(x) for x in r] for r in d["Gamma"]]
    return d["p"], Cm, G, d


def gam(G, i, j, k):
    """Gamma_{i,jk} (i inner index 0..M-2, j, k in {0, 1})"""
    return G[i][j + k]


def p_of_x(p, T, G, x, M):
    """p(x) = p^ + T x + 1/2 (0, Gamma[x_o, x_o]) for a vector x of arb (balls allowed)"""
    m = M + 1
    out = []
    for i in range(m):
        acc = p[i]
        for j in range(m):
            if T[i][j] != 0:
                acc = acc + T[i][j] * x[j]
        if i >= 2:
            g = G[i - 2]
            acc = acc + (g[0] * (x[0] * x[0]) + 2 * g[1] * (x[0] * x[1]) + g[2] * (x[1] * x[1])) / 2
        out.append(acc)
    return out


def dp_dirs(T, G, x, M):
    """the columns d_k(x) = dp/dx_k (lists of m arb) at x (balls allowed): T_k, plus (0, Gamma[e_k, x_o]) for k outer"""
    m = M + 1
    out = []
    for k in range(m):
        d = [T[i][k] for i in range(m)]
        if k < 2:
            for i in range(2, m):
                d[i] = d[i] + gam(G, i - 2, k, 0) * x[0] + gam(G, i - 2, k, 1) * x[1]
        out.append(d)
    return out


# ---------------------------------------------------------------- start data with second derivatives

def series2(pbox, dirs, M, r0, phi0, N):
    """the truncated series Y_N at xi0 = r0 e^{i phi0} (N terms, the start-value lemma's truncation) as second-order
    jets along each direction d in dirs (lists of m arb balls), over the parameter box pbox: returns, per direction,
    the coefficient lists [c0, c1, c2] per state component, c2 = 1/2 D^2 Y_N[d, d] (exact jets: acb_series of length
    3 through certify_M_sys.series_generic_M, valid for every p in pbox and every direction in the ball d)"""
    m = M + 1
    W = CS.W0()
    xi0 = acb(arb(r0)) * (CS.I * arb(phi0)).exp()
    out = []
    for d in dirs:
        ctx.cap = 3
        pj = [acb_series([acb(pbox[i]), acb(d[i])], prec=3) for i in range(m)]
        Fj = CS.profile(pj, M, acb_series([1], prec=3))
        wj = acb_series([acb(W * pbox[1]), acb(W * d[1])], prec=3)
        dj, slj, gj = CS.series_generic_M(Fj, wj, N, M)
        vj = CS.series_state(dj, slj, gj, acb_series([xi0], prec=3), N, M)
        out.append([[v.coeffs()[i] if len(v.coeffs()) > i else acb(0) for i in range(3)] for v in vj])
    return out


def start_data2(M, pstr, Cm, G, widths, verbose=True):
    """C^2 start data at xi0 for the chart p(x) over X = prod [-widths_k, widths_k] (proof/A_M.md 10.3):
      vals_c   y+(xi0; p^)                              (nd acb, from fixed_point_M.start_data_M at the point p^)
      cols     V0_k = Dy+(xi0; p^) T_k                  (m lists; the lemma's derivatives at p^)
      W0cols   for every pair j <= k, an enclosure of D^2_x y+(xi0; p(x))_jk over x in X:
               D^2 Y_N(p(x))[d_j(x), d_k(x)]   (exact series jets over the hull of p(X), directions over X)
             + D^2 eta(p(x))[d_j(x), d_k(x)]   (Cauchy: |.| <= 4 nu^2 sup|eta| / (c - hw)^2, see below)
             + Dy+(xi0; p(x)) Gamma_jk          (j, k outer; the lemma's derivatives over the hull of p(X))
    eta = y+ - Y_N is holomorphic in the parameters on the complex box p^ + c(1 + i) with |eta_k| <= e_k there
    (start_data_M's Cauchy box at the point p^; it adds a ball of radius e_k / c to every derivative, so
    e_k <= c x the smallest radius among the returned derivative balls of component k). For x real with
    |x_i - p^_i| <= hw and real directions u, v with max norm <= nu, z -> eta(x + z1 u + z2 v) is holomorphic on the
    bidisc |z1|, |z2| < (c - hw) / (2 nu), which lies in that complex box; Cauchy's estimate on it gives
    |D^2 eta(x)[u, v]| <= e_k (2 nu / (c - hw))^2."""
    import fixed_point_M as FPM
    m = M + 1
    path = CM.path_for(M)
    r0, arg = path[0], CM.ARG[M]
    diag = diag_for(M)
    W0 = CS.W0()
    ctx.prec = 320
    p = CM.exact_p(pstr)
    ctx.prec = 320
    T = CM.T_matrix(Cm, M)
    X = [arb(0, wd) for wd in widths]
    pbox = p_of_x(p, T, G, X, M)
    hw = max(float(abs(pbox[i] - p[i]).upper()) for i in range(m))
    c = arb(CAUCHY)
    assert hw < 0.5 * float(c), "the box must lie well inside the Cauchy box"
    t0 = time.time()
    centre = FPM.run_M([acb(v.mid()) for v in p], M, r0, arg, None, verbose=False, kcap=40, diag=diag)
    N = centre["N"]
    _, vals_c, ders_c, eta_c = FPM.start_data_M([arb(v) for v in p], M, r0, arg, +1, CAUCHY, centre=centre, diag=diag)
    nd = len(vals_c)
    ctx.prec = 320
    cols = [[sum((acb(T[i][k]) * ders_c[i][q] for i in range(m)), acb(0)) for q in range(nd)] for k in range(m)]
    wcol = [W0 * T[1][k] for k in range(m)]
    # sup |eta_k| on the Cauchy box at p^, from the radii of the derivative balls
    e_k = [arb(c * min(min(ders_c[i][q].real.rad(), ders_c[i][q].imag.rad()) for i in range(m))) for q in range(nd)]
    dirsX = dp_dirs(T, G, X, M)
    nu = max(float(abs(dirsX[k][i]).upper()) for k in range(m) for i in range(m))
    fac = (2 * arb(nu) / (c - arb(hw))) ** 2
    pairs = C2.pair_list(m)
    # the series jets: diagonal directions, and the sums for the off-diagonal pairs (polarization:
    # D^2[a, b] = c2(a + b) - c2(a) - c2(b), with c2(d) = 1/2 D^2[d, d])
    # consistency with the lemma at the centre: this file's Y_N(p^) and DY_N(p^) T_0 must overlap the lemma's balls
    # (both enclose the same numbers up to eta; at M = 5 eta at the point is below the rounding, both radii are about
    # 1e-95, and containment would compare rounding paths rather than truncations)
    jc = series2(p, [[T[i][0] for i in range(m)]], M, r0, path[1], N)[0]
    okv = all(vals_c[q].overlaps(jc[q][0]) for q in range(nd))
    okd = all(cols[0][q].overlaps(jc[q][1]) for q in range(nd))
    if not (okv and okd):
        raise RuntimeError(f"start data: this file's series does not overlap the lemma's balls (values {okv}, "
                           f"derivatives {okd}): truncation N = {N} mismatch?")
    jd = series2(pbox, dirsX, M, r0, path[1], N)
    sums = [[dirsX[j][i] + dirsX[k][i] for i in range(m)] for (j, k) in pairs if j != k]
    js = series2(pbox, sums, M, r0, path[1], N)
    ders_b = None
    if any(g != 0 for row in G for g in row) and "chart" not in MUTATE:
        _, _, ders_b, _ = FPM.start_data_M([arb(v) for v in pbox], M, r0, arg, +1, CAUCHY, centre=centre, diag=diag)
        ctx.prec = 320
    W0cols = []
    it = iter(js)
    for (j, k) in pairs:
        if j == k:
            d2 = [2 * jd[j][q][2] for q in range(nd)]
        else:
            jsk = next(it)
            d2 = [jsk[q][2] - jd[j][q][2] - jd[k][q][2] for q in range(nd)]
        if "w0" in MUTATE:
            d2 = [acb(0) for _ in range(nd)]
        col = []
        for q in range(nd):
            r = arb(0, (e_k[q] * fac).upper())
            v = d2[q] + acb(r, r)
            if ders_b is not None and j < 2 and k < 2:
                for i in range(2, m):
                    v = v + ders_b[i][q] * gam(G, i - 2, j, k)
            col.append(v)
        W0cols.append(col)
    info = {"start": f"fixed_point_M.start_data_M (rigorous given that lemma; sha256 "
                     f"{hashlib.sha256(open(FPM.__file__, 'rb').read()).hexdigest()[:16]}), arg {arg}, "
                     f"{'diag' if diag else 'plain'} splitting, N = {N}; second derivatives: series jets + Cauchy",
            "rigorous": True, "N": N, "eta": float(eta_c[0].upper()), "eta_der": float(eta_c[1].upper()),
            "eta_d2_bound": max(float((e * fac).upper()) for e in e_k), "hull_halfwidth": hw, "nu": nu,
            "sec": time.time() - t0, "mutate": sorted(MUTATE)}
    if verbose:
        print(f"start data (C^2): N = {N}, sup|eta| on the Cauchy box <= {max(float(e.upper()) for e in e_k):.2e}, "
              f"second-derivative bound of eta {info['eta_d2_bound']:.2e}, consistency with the lemma ok; "
              f"{info['sec']:.0f} s", flush=True)
    return vals_c, cols, wcol, W0cols, info


# ---------------------------------------------------------------- the run

def box2_run(M, chart_file, h, ws, out, log=None, max_steps=None, verbose=False):
    t0 = time.time()
    pstr, Cm, G, rec = load_chart(chart_file)
    m = M + 1
    widths = [arb(h), arb(h)] + [arb(ws[(i // 2) % len(ws)]) for i in range(M - 1)]
    s = CS.SysM(M)
    vals, cols, wcol, W0cols, info = start_data2(M, pstr, Cm, G, widths)
    ctx.prec = PREC
    p = CM.exact_p(pstr)
    ctx.prec = PREC
    pdx = [acb(arb(0, wd)) for wd in widths]
    NT, NJ = CM.orders(M)
    lo = C2.start_set2(s, vals, CS.W0() * p[1], cols, wcol, W0cols, pdx, NT, NJ)
    lo.max_steps = max_steps
    path = CM.path_for(M)
    if log:
        with open(log, "a") as fh:
            fh.write(f"C2 box run M = {M}, chart {chart_file}, h = {h}, w = {ws}; start: {info['start']}; "
                     f"mutations {sorted(MUTATE | C2.MUTATE)}\n")
    key = json.dumps({"M": M, "p": list(pstr), "C": rec["C"], "Gamma": rec["Gamma"], "h": h, "w": list(ws),
                      "path": [path[0], CM.ARG[M], path[2], path[3]], "N": [NT, NJ], "start": info["start"],
                      "mutate": sorted(MUTATE | C2.MUTATE)})
    ckpt = (out + ".ckpt") if max_steps is None else None
    try:
        steps = C2.integrate2(lo, path, verbose=verbose, log=log, ckpt=ckpt, key=key)
    except CL.PilotStop as e:
        rs, rv, rw, rvx = lo.radii()
        res = {"kind": "pilot2", "M": M, "h": h, "w": list(ws), "stopped_at": str(e), "steps": lo.total_steps,
               "sec": time.time() - t0, "sec_per_step": (time.time() - t0) / lo.total_steps, "state_rad": rs,
               "V0_rad": rv, "W_rad": rw, "VX_rad": rvx}
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"pilot stopped after {res['steps']} steps at {res['stopped_at']}: {res['sec_per_step']:.2f} s/step; "
              f"radii: state (box) {rs:.2e}, V0 {rv:.2e}, W {rw:.2e}, V(X) {rvx:.2e}")
        return res
    if ckpt and os.path.exists(ckpt):
        os.replace(ckpt, ckpt + ".done")
    res = {"kind": "box2", "M": M, "p": list(pstr), "h": h, "w": list(ws), "C": rec["C"], "Gamma": rec["Gamma"],
           "chart": chart_file, "path": [path[0], CM.ARG[M], path[2], path[3]], "start": info, "steps": steps,
           "sec": time.time() - t0, "prec": PREC, "orders": [NT, NJ], "mutate": sorted(MUTATE | C2.MUTATE),
           "sets": C2.dump2(lo)}
    with open(out, "w") as fh:
        json.dump(res, fh)
    print(f"C2 box run M = {M}: steps {steps}, {res['sec']:.0f} s -> {out}")
    return res


# ---------------------------------------------------------------- the Krawczyk test (second order)

def assemble(bx, mutate=None):
    """the point pieces of the test from a saved C^2 run: Y (exact), Y gt(0) (list of arb), Y l V0 (m x m arb),
    Y l W(X) (m x P arb), l V0 (m x m arb), T, Gamma, p^"""
    M = bx["M"]
    m = M + 1
    ctx.prec = KPREC
    s = CS.SysM(M)
    fmut = mutate if mutate in ("sign", "noconj", "negconj", "scale") else None
    ell = CM.jump_rows(s, fmut)
    c, Q, q, var, w2, pdx = C2.load2(bx["sets"])
    LV0 = arb_mat(CM.apply_var(ell, var))
    Dmid = arb_mat([[arb(LV0[i, j].mid()) for j in range(m)] for i in range(m)])
    Yb = Dmid.inv()
    Y = arb_mat([[arb(Yb[i, j].mid()) for j in range(m)] for i in range(m)])
    Yl = Y * ell
    Yg = CM.apply_set(Yl, c, Q, q)
    G1 = CM.apply_var(Yl, var)
    G2 = CM.apply_var(Yl, w2)
    Cm = [[arb(x) for x in r] for r in bx["C"]]
    G = [[arb(x) for x in r] for r in bx["Gamma"]]
    return {"M": M, "Y": Y, "Yg": Yg, "G1": G1, "G2": G2, "LV0": LV0, "T": CM.T_matrix(Cm, M), "G": G,
            "p_hat": CM.exact_p(bx["p"]), "ell": ell, "w2": w2}


def krawczyk2_test(bx, mutate=None, theta=None, sub=None, verbose=True, asm=None):
    """K(X) of the module docstring from a saved C^2 run. mutate: as certify_M.krawczyk_test ('sign', 'noconj',
    'negconj', 'scale', 'plant_out', 'plant_in', 'plant_H', ('plant_d', fractions), 'widen'). sub: a sub-box of the
    run's box (the enclosures hold on every sub-box centred at 0); theta: scale the box."""
    M = bx["M"]
    m = M + 1
    A = asm if (asm is not None and mutate not in ("sign", "noconj", "negconj", "scale")) else assemble(bx, mutate)
    ctx.prec = KPREC
    Y, Yg, G1, G2, T, G = A["Y"], list(A["Yg"]), A["G1"], A["G2"], A["T"], A["G"]
    pidx = C2.pair_index(m)
    h = arb(bx["h"])
    widths = [h, h] + [arb(bx["w"][(i // 2) % len(bx["w"])]) for i in range(M - 1)]
    if sub is not None:
        sw = [arb(sub[0]), arb(sub[0])] + [arb(sub[1 + (i // 2) % (len(sub) - 1)]) for i in range(M - 1)]
        assert all(bool(sw[i].upper() <= widths[i].upper()) for i in range(m)), "the sub-box must lie inside the run box"
        widths = sw
    if theta is not None:
        widths = [arb(theta) * x for x in widths]
    X = [arb(0, wd) for wd in widths]
    if mutate in ("plant_out", "plant_in", "plant_H") or (isinstance(mutate, tuple) and mutate[0] == "plant_d"):
        # g -> g - L d with L the exact midpoint of l V0 = Dgt(0): the zero moves by about d, Dgt is unchanged
        L = arb_mat([[arb(A["LV0"][i, j].mid()) for j in range(m)] for i in range(m)])
        d = [arb(0)] * m
        if mutate == "plant_out":
            d[0] = 2 * widths[0]
        elif mutate == "plant_in":
            d[0], d[1] = widths[0] / 2, -3 * widths[1] / 10
            for i in range(2, m):
                d[i] = widths[i] / 4
        elif isinstance(mutate, tuple):
            d = [arb(mutate[1][i]) * widths[i] for i in range(m)]
        else:
            d[2] = 2 * widths[2] if m > 2 else 2 * widths[0]
        YLd = Y * (L * arb_mat([[x] for x in d]))
        Yg = [Yg[i] - YLd[i, 0] for i in range(m)]
    if mutate == "widen":
        sv = np.linalg.svd(np.array([[float(A["LV0"][i, j].mid()) for j in range(2)] for i in range(2)]),
                           compute_uv=False)
        wid = 1e3 * float(h.mid()) * sv[-1]
        ell, gC = A["ell"], None
        c, Q, q, _, _, _ = C2.load2(bx["sets"])
        g = CM.apply_set(ell, c, Q, q)
        gw = [g[i] + (arb(0, wid) if i < 2 else arb(0)) for i in range(m)]
        Ygw = Y * arb_mat([[x] for x in gw])
        Yg = [Ygw[i, 0] for i in range(m)]
    # I - Y J(X), entrywise as an interval matrix, then times X
    Mx = [[(arb(1) if i == j else arb(0)) - G1[i][j] - sum((G2[i][pidx[(j, k)]] * X[k] for k in range(m)), arb(0))
           for j in range(m)] for i in range(m)]
    K = [-Yg[i] + sum((Mx[i][j] * X[j] for j in range(m)), arb(0)) for i in range(m)]
    ok = all(abs(K[i]) < widths[i] for i in range(m))
    margins = [float((widths[i] - abs(K[i]).upper()).lower()) for i in range(m)]
    rel = [mg / float(widths[i].mid()) for i, mg in enumerate(margins)]
    p = A["p_hat"]
    pstar = p_of_x(p, T, G, K, M)
    hull = p_of_x(p, T, G, X, M)
    res = {"ok": bool(ok), "K": K, "margins": margins, "rel_margins": rel, "pstar": pstar, "hull": hull,
           "widths": widths, "T": T, "p_hat": p, "Yg": Yg,
           "YJ_rad": max(float(Mx[i][j].rad()) for i in range(m) for j in range(m))}
    if verbose:
        tag = f" [mutation {mutate if isinstance(mutate, str) else mutate[0]}]" if mutate else ""
        print(f"Krawczyk (C^2) M = {M}{tag}: K(X) in int X: {ok}; relative margins " + ", ".join(f"{r:+.3f}" for r in rel))
    return res


def contributions(bx, sub=None, asm=None):
    """diagnostic: the relative contribution of every term to |K_i| / w_i (|Y g|, |I - Y l V0| X, and the
    second-order terms |Y l W|_{i,jk} X_j X_k summed per pair)"""
    M = bx["M"]
    m = M + 1
    A = asm or assemble(bx)
    pidx = C2.pair_index(m)
    h = float(bx["h"])
    widths = [h, h] + [float(bx["w"][(i // 2) % len(bx["w"])]) for i in range(M - 1)]
    if sub is not None:
        widths = [float(sub[0])] * 2 + [float(sub[1 + (i // 2) % (len(sub) - 1)]) for i in range(M - 1)]
    out = np.zeros((m, 2 + len(C2.pair_list(m))))
    for i in range(m):
        out[i, 0] = float(abs(A["Yg"][i]).upper()) / widths[i]
        out[i, 1] = sum(float(abs((1 if i == j else 0) - A["G1"][i][j]).upper()) * widths[j] for j in range(m)) / widths[i]
        for p, (j, k) in enumerate(C2.pair_list(m)):
            mult = 1 if j == k else 2
            out[i, 2 + p] = mult * float(abs(A["G2"][i][p]).upper()) * widths[j] * widths[k] / widths[i]
    return out


def best_outer2(bx, lo=1e-6, asm=None):
    h = float(bx["h"])
    k = 0
    while 0.9 ** k >= lo:
        hp = h * 0.9 ** k
        r = krawczyk2_test(bx, sub=[repr(hp)] + list(bx["w"]), verbose=False, asm=asm)
        if r["ok"]:
            return hp, r
        k += 1
    return None, None


def search2(bx, asm=None, facs=None):
    """sub-boxes of the run's box (the enclosures hold on each): every combination of factors (one for the outer
    half-width, one per inner width) from facs; returns the passing sub-box with the largest outer half-width (ties:
    the largest inner widths) and its result, or (None, None)"""
    facs = facs or [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]
    h = float(bx["h"])
    ws = [float(x) for x in bx["w"]]
    import itertools
    best, rbest = None, None
    for combo in itertools.product(facs, repeat=1 + len(ws)):
        sub = [repr(h * combo[0])] + [repr(w * f) for w, f in zip(ws, combo[1:])]
        key = (h * combo[0],) + tuple(w * f for w, f in zip(ws, combo[1:]))
        if best is not None and key <= best:
            continue
        r = krawczyk2_test(bx, sub=sub, verbose=False, asm=asm)
        if r["ok"]:
            best, rbest = key, (sub, r)
    return (rbest[0], rbest[1]) if rbest else (None, None)


def shrink2(bx, sub=None, lo=1e-4, asm=None):
    r1 = krawczyk2_test(bx, sub=sub, verbose=False, asm=asm)
    if not r1["ok"]:
        return None, None
    best, rbest, th = 1.0, r1, 1.0
    while th * 0.9 >= lo:
        th *= 0.9
        r = krawczyk2_test(bx, theta=repr(th), sub=sub, verbose=False, asm=asm)
        if not r["ok"]:
            break
        best, rbest = th, r
    return best, rbest


def hier_check2(res, M, mutate=False):
    """certify_M.hier_check with the hull of the curved parallelepiped p(X)"""
    ctx.prec = KPREC
    pts = [[arb(v) for v in pt] for pt in CM.HIER[M]]
    m = M + 1
    spread = [max(abs(float(pts[a][i].mid()) - float(pts[b][i].mid())) for a in range(len(pts)) for b in range(len(pts)))
              for i in range(m)]
    spread = [max(sv, 1e-15 * abs(float(pts[0][i].mid()))) for i, sv in enumerate(spread)]
    if mutate:
        sh = max(100 * spread[0], 1e-9)
        pts = [[pp[0] + sh] + pp[1:] for pp in pts]
    hull = res["hull"]
    ins = [all(hull[i].contains(pp[i]) for i in range(m)) for pp in pts]
    dev = [max(float(abs(pp[i] - res["pstar"][i]).upper()) for pp in pts) for i in range(m)]
    ok_b = all(dev[i] <= float(res["pstar"][i].rad()) + 3 * spread[i] for i in range(m))
    names = CM_names(M)
    print(f"hier.py check M = {M}{' [mutated]' if mutate else ''}: (a) inside the hull of p(X): {all(ins)} "
          f"(per solve {ins}); (b) agreement within 3x hier's spread: {ok_b}")
    for i, nm in enumerate(names):
        print(f"   {nm:8s} |hier - certified| <= {dev[i]:.2e}   hier spread {spread[i]:.1e}   hull half-width "
              f"{float(hull[i].rad()):.1e}")
    return all(ins) and ok_b


# the certified first-order enclosures (endpoints as published): Theorem A(M = 1), proof/A_certify.md 12.6; the
# M = 3 certificate, proof/A_M.md 6.2
REFERENCE = {
    "thmA1": {"M": 1, "box": [("2.54214804761338", "2.54214804761975"), ("0.776994013607438", "0.776994013612010")]},
    "AM3": {"M": 3, "box": [("2.72647588563032", "2.72647588563144"), ("1.01560309485743", "1.01560309485804"),
                            ("0.144419478469924", "0.144419478470230"), ("0.0110874258881079", "0.0110874258881255")]},
}


def reference_check(res, M, name, mutate=False):
    """planted truth: the C^2 zero enclosure must overlap the certified first-order one in every coordinate (both
    contain the unique zero if both certificates hold). mutate: the reference shifted by 10 of its widths in A
    (must not overlap)."""
    ref = REFERENCE[name]
    assert ref["M"] == M
    ctx.prec = KPREC
    ok = True
    for i, (lo_, hi_) in enumerate(ref["box"]):
        a, b = arb(lo_), arb(hi_)
        if mutate and i == 0:
            a, b = a + 10 * (b - a), b + 10 * (b - a)
        rb = a.union(b)
        ov = res["pstar"][i].overlaps(rb)
        inside = bool(rb.contains(res["pstar"][i]))
        ok &= ov
        print(f"   {CM_names(M)[i]:8s} overlaps the reference ({name}{', shifted' if mutate and i == 0 else ''}): {ov}; "
              f"C^2 enclosure inside it: {inside}")
    print(f"reference check ({name}): {'REPRODUCED' if ok else 'NOT REPRODUCED'}")
    return ok


def CM_names(M):
    return ["A", "w/w_inf"] + [f"{b}{nu}" for nu in range(3, M + 1, 2) for b in ("B", "C")]


def run_controls2(bx, sub=None):
    """certify_M.CONTROLS on the C^2 test (same mutations and expected verdicts)"""
    ok_all, clean = True, None
    asm = assemble(bx)
    for name, mut, expect, desc in CM.CONTROLS:
        if mut == "plant_H" and bx["M"] == 1:
            continue
        if mut == "plant_half":
            fr = [0.5 * max(x, 0.0) * (-1) ** i for i, x in enumerate(clean["rel_margins"])]
            r = krawczyk2_test(bx, ("plant_d", fr), sub=sub, verbose=False, asm=asm)
        else:
            r = krawczyk2_test(bx, mut, sub=sub, verbose=False, asm=asm)
        if mut is None:
            clean = r
        verdict = r["ok"] == expect
        extra = ""
        if mut in ("negconj", "scale") and r["ok"] and clean is not None:
            same = all(r["pstar"][i].overlaps(clean["pstar"][i]) for i in range(len(r["pstar"])))
            verdict &= same
            extra = f"; zero enclosure overlaps C0's: {same}"
        ok_all &= verdict
        print(f"{name} {desc}: expected {'pass' if expect else 'fail'}, got {'pass' if r['ok'] else 'fail'} "
              f"(relative margins {', '.join(f'{x:+.2e}' for x in r['rel_margins'])}){extra}  "
              f"{'OK' if verdict else 'UNEXPECTED'}", flush=True)
    print("CONTROLS " + ("ALL AS EXPECTED" if ok_all else "SOME UNEXPECTED"))
    return ok_all


# ---------------------------------------------------------------- control D2: the second-derivative enclosure

D2_REL_FLOOR = 1e-10      # relative floor of the D2 tolerance (written down before the runs, proof/A_M.md 10.5)


def d2check(bx, fdfile, verbose=True):
    """the rigorous enclosure Y l W(X) at xi1 (D^2 gt over X, in the run's chart) must contain Y times the independent
    finite-difference D^2 gt(0) (certify_M2_fd.py: central differences of the C^1 numerics in the first-order chart,
    converted to the run's chart by D^2 gt2[., j, k] = D^2 gt1[., j, k] + Dgt1[., H] Gamma_jk for j, k outer).
    Entry (i, jk) passes if |Y FD - mid| <= rad + 3 |Y| err + D2_REL_FLOOR |Y FD| (err = the Richardson difference).
    Returns (all pass, the worst ratio |Y FD - mid| / tolerance, the worst relative deviation)."""
    M = bx["M"]
    m = M + 1
    A = assemble(bx)
    fd = json.load(open(fdfile))
    D2 = np.array(fd["D2"])
    err = np.nan_to_num(np.array(fd["err"]), nan=0.0)
    G = [[float(arb(x).mid()) for x in r] for r in bx["Gamma"]]
    D1 = np.array([[float(A["LV0"][i, j].mid()) for j in range(m)] for i in range(m)])
    D2c = D2.copy()
    for j in range(2):
        for k in range(2):
            for i in range(2, m):
                D2c[:, j, k] += D1[:, i] * G[i - 2][j + k]
    Yn = np.array([[float(A["Y"][i, j].mid()) for j in range(m)] for i in range(m)])
    worst, worst_rel, ok = 0.0, 0.0, True
    rows = []
    for p, (j, k) in enumerate(C2.pair_list(m)):
        yfd = Yn @ D2c[:, j, k]
        yerr = np.abs(Yn) @ err[:, j, k]
        for i in range(m):
            z = A["G2"][i][p]
            dev = abs(yfd[i] - float(z.mid()))
            tol = float(z.rad()) + 3 * yerr[i] + D2_REL_FLOOR * abs(yfd[i]) + 1e-300
            worst = max(worst, dev / tol)
            worst_rel = max(worst_rel, dev / (abs(yfd[i]) + 1e-300))
            ok &= dev <= tol
            rows.append((i, j, k, yfd[i], float(z.mid()), float(z.rad()), 3 * yerr[i]))
    if verbose:
        print(f"D2 check M = {M} ({bx.get('mutate') or 'no mutation'}): Y l W(X) contains Y D^2 gt(0) from the "
              f"finite differences: {ok}; worst |dev| / tolerance {worst:.2e}, worst relative deviation "
              f"{worst_rel:.2e}")
        big = sorted(rows, key=lambda r: -abs(r[3]))[:6]
        for r in big:
            print(f"   (Y D^2 gt)_{r[0]},{r[1]}{r[2]}: FD {r[3]:+.6e}  enclosure {r[4]:+.6e} +- {r[5]:.1e}  "
                  f"(FD error {r[6]:.1e})")
    return ok, worst, worst_rel


# ---------------------------------------------------------------- the chart from the finite differences

def make_chart(M, centre, use_out, fdfile, out):
    """p^ and C from the refinement record (or point run); Gamma_{i,jk} = -[(Dgt1_HH)^-1 D^2_jk gt1_H]_i from the
    finite differences (numerics; any Gamma is allowed). M = 1: no inner block, Gamma empty."""
    ctx.prec = KPREC
    o = {"point": centre} if "_point_" in os.path.basename(centre) else {"centre": centre}
    if use_out:
        o["use_out"] = True
    pstr, Cm = CM.centre_and_tilt(o, M)
    m = M + 1
    G = []
    if M > 1:
        fd = json.load(open(fdfile))
        D2 = np.array(fd["D2"])
        # Dgt1 at 0 from the FD record's centre estimate if present, else from the refinement's Dg times T
        T = np.array(fd["T"])
        with open(centre) as fh:
            rec = json.load(fh)
        Dg = np.array([[float(arb(x).mid()) for x in r] for r in rec["history"][-1]["Dg"]])
        D1 = Dg @ T
        H = list(range(2, m))
        DHH = D1[np.ix_(H, H)]
        for i in range(M - 1):
            G.append([None, None, None])
        for (j, k), slot in (((0, 0), 0), ((0, 1), 1), ((1, 1), 2)):
            g = -np.linalg.solve(DHH, D2[H, j, k])
            for i in range(M - 1):
                G[i][slot] = g[i]
        G = [[repr(float(x)) for x in r] for r in G]
    d = {"M": M, "p": list(pstr), "C": [[x.str(60) for x in r] for r in Cm], "Gamma": G, "centre": centre,
         "fd": fdfile, "note": "chart p(x) = p^ + T x + (0, Gamma[x_o, x_o]/2); Gamma rows [g_00, g_01, g_11] "
                               "(numerics; any chart is allowed)"}
    with open(out, "w") as fh:
        json.dump(d, fh, indent=1)
    print(f"chart M = {M}: Gamma = {G} -> {out}")
    return d


def export6a2(bx, out, sub=None):
    """the state box at xi1 over the certified curved parallelepiped p(X') (X' the run's box or a sub-box), for
    lemma 6a(M), in certify_M.export_6a's format (keys M, h, xi1, yX_xi1, V): Y(X') = c + Q q + V0 x + W(X)[x, x]/2
    and V(X') = V0 + W(X)[., x], which contain y+(xi1; p(x)) and its x-derivative for every x in X'"""
    ctx.prec = PREC
    M = bx["M"]
    m = M + 1
    s = CS.SysM(M)
    c, Q, q, var, w2, pdx = C2.load2(bx["sets"])
    if sub is not None:
        ws = [arb(sub[0]), arb(sub[0])] + [arb(sub[1 + (i // 2) % (len(sub) - 1)]) for i in range(M - 1)]
        assert all(bool(ws[i].upper() <= pdx[i].real.rad()) for i in range(m)), "the sub-box must lie inside the run box"
        pdx = [acb(arb(0, x)) for x in ws]
    lo = C2.LohnerM2(s, c, Q, q, 1, var, w2, pdx)
    X = lo.YX_box()
    V = lo.VX_box()
    d = {"M": M, "h": sub[0] if sub else bx["h"], "w": sub[1:] if sub else bx["w"], "xi1": bx["path"][3],
         "path": bx["path"], "p": bx["p"], "C": bx["C"], "Gamma": bx["Gamma"], "start": bx["start"],
         "yX_xi1": [CL.s_acb(x) for x in X[:s.nd]],
         "V": [[CL.s_acb(V[i, j]) for j in range(V.ncols())] for i in range(s.nd)],
         "note": "V = d(state)/d(x) over p(X) in the chart p = p^ + T x + (0, Gamma[x_o, x_o]/2) (certify_M2.py)"}
    with open(out, "w") as fh:
        json.dump(d, fh)
    print(f"exported the state box at xi1 = {d['xi1']} (max radius {max(CL.VI.rad_of(x) for x in X[:s.nd]):.2e}) to {out}")


def diag1(bxfile, fdfile=None):
    """diagnostic of a first-order (certify_M.py) box run's Krawczyk test: the radius of Y Dgt(X) per entry, its
    relative contribution rad_ij w_j / w_i to |K_i| / w_i, and (with fdfile, the finite differences of
    certify_M2_fd.py in the same first-order chart) the true variation of Y Dgt over the box,
    sum_k |Y D^2 gt|_{i,jk} w_k, for comparison: their ratio is the excess of the C^1 enclosure"""
    with open(bxfile) as fh:
        bx = json.load(fh)
    M = bx["M"]
    m = M + 1
    r = CM.krawczyk_test(bx, verbose=True)
    YJ = r["YJ"]
    rad = np.array([[float(YJ[i][j].rad()) for j in range(m)] for i in range(m)])
    w = np.array([float(x.mid()) for x in r["widths"]])
    np.set_printoptions(precision=2, linewidth=200)
    print(f"radius of Y Dgt(X) (box {bxfile}):\n{rad}")
    print(f"relative contribution to |K_i| / w_i, rad_ij w_j / w_i:\n{rad * w[None, :] / w[:, None]}")
    print("|Y gt(0)|_i / w_i: " + ", ".join(f"{float(abs(x).upper()) / w[i]:.1e}" for i, x in enumerate(r["Yg"])))
    if fdfile:
        fd = json.load(open(fdfile))
        D2 = np.array(fd["D2"])
        ctx.prec = KPREC
        s = CS.SysM(M)
        ell = CM.jump_rows(s)
        _, _, _, varb = CL.load_lohner(bx["sets"])
        D1 = np.array([[float(x.mid()) for x in row] for row in CM.apply_var(ell, varb)])
        Yn = np.linalg.inv(D1)
        YD2 = np.einsum("ia,ajk->ijk", Yn, D2)
        var = np.einsum("ijk,k->ij", np.abs(YD2), w)
        print(f"true variation of Y Dgt over the box, sum_k |Y D^2 gt|_(i,jk) w_k (finite differences):\n{var}")
        print(f"excess of the C^1 enclosure, radius / true variation:\n{rad / var}")


def opts_of(argv):
    return {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}


def main():
    argv = sys.argv[1:]
    mode = argv[0]
    o = opts_of(argv)
    pos = [a for a in argv[1:] if not a.startswith("--")]
    for mu in (o.get("mutate") or "").split(","):
        if mu in ("w0", "chart"):
            MUTATE.add(mu)
        if mu == "lin2":
            C2.MUTATE.add(mu)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    if mode == "chart":
        M = int(pos[0])
        make_chart(M, o["centre"], "use_out" in o, o.get("fd"), o.get("out", os.path.join(DATA, f"A_M2_chart_M{M}.json")))
        return 0
    if mode == "box":
        M = int(pos[0])
        ws = o.get("w", "0").split(",")
        box2_run(M, o["chart"], o["h"], ws, o["out"], log=o.get("log"),
                 max_steps=int(o["pilot"]) if "pilot" in o else None, verbose="verbose" in o)
        return 0
    if mode == "diag1":
        diag1(o["box"], o.get("fd"))
        return 0
    with open(o["box"]) as fh:
        bx = json.load(fh)
    M = bx["M"]
    sub = o["sub"].split(",") if "sub" in o else None
    if mode == "controls":
        return 0 if run_controls2(bx, sub) else 1
    if mode == "export6a":
        export6a2(bx, o["out"], sub)
        return 0
    if mode == "d2check":
        ok, _, _ = d2check(bx, o["fd"])
        return 0 if ok else 1
    if mode == "krawczyk":
        asm = assemble(bx)
        if "best" in o:
            hp, _ = best_outer2(bx, asm=asm)
            print(f"   largest passing outer half-width with the run's inner widths: {hp}")
            if hp is not None:
                sub = [repr(hp)] + list(bx["w"])
        if "search" in o:
            sb, _ = search2(bx, asm=asm)
            print(f"   passing sub-box with the largest outer half-width (factor grid): {sb}")
            if sb is not None:
                sub = sb
        mut = o.get("mutate")
        mut = None if mut in (None, "hier", "ref") else mut
        r = krawczyk2_test(bx, mut, theta=o.get("theta"), sub=sub, asm=asm)
        print(f"   start values: {bx['start']['start']}; run mutations: {bx.get('mutate')}")
        print(f"   gt(0): |Y g| per row / width " + ", ".join(f"{float(abs(r['Yg'][i]).upper()) / float(r['widths'][i].mid()):.1e}"
                                                         for i in range(M + 1)))
        print(f"   box of the run: h = {bx['h']}, w = {bx['w']}; tested on: half-widths " +
              ", ".join(f"{float(x.mid()):.3g}" for x in r["widths"]))
        np.set_printoptions(precision=2, linewidth=200)
        print("   contributions to |K_i| / w_i (columns: |Y g|, first order, then each pair jk):")
        print(contributions(bx, sub, asm))
        print(CM.describe(r, M))
        if "shrink" in o and r["ok"]:
            th, rs = shrink2(bx, sub=sub, asm=asm)
            print(f"   smallest passing box: theta = {th:.4g}; relative margins " +
                  ", ".join(f"{x:+.3f}" for x in rs["rel_margins"]) + "\n   the zero (endpoints rounded outward):")
            print(CM.describe(rs, M))
            # the zero lies in both enclosures (the test passes on both boxes): keep their intersection, as
            # certify_M.m1_reproduce does
            r = dict(r, pstar=[a.intersection(b) for a, b in zip(r["pstar"], rs["pstar"])])
        if "hier" in o and r["ok"]:
            return 0 if hier_check2(r, M, mutate=(o.get("mutate") == "hier")) else 1
        if "reference" in o and r["ok"]:
            return 0 if reference_check(r, M, o["reference"], mutate=(o.get("mutate") == "ref")) else 1
        return 0 if r["ok"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
