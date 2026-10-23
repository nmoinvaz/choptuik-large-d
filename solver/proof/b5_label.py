#!/usr/bin/env python3
"""Lemma L (proof/B_chain.md, section 2): the labels of fast-type solutions of the inner secant equation on the
Stokes line, M = 1, certified in ball arithmetic (python-flint) over the complex parameter box K of B5(1).

Setting. z = xi^2 real in (0, Z_L]; p in K (complex). The inner system at eps = 0 in the state (P, Phi, C) of
proof/B5_inner.md section 1 (b5_system.Model.f_0):
    P' = Phi/(2z),   Phi' = Phi/(2z^2) + G0(z, P, Phi, C),   C' = [C E0]/2,   E0 = [P P]/2,
with z G0 analytic (b5_formulas.zG0). psi = Y+ - Y- (Y+ the lateral sum of lemma S, Y-(z; p) = sigma(Y+(z; p^bar)))
solves the secant equation
    psi_P' = psi_Phi/(2z),  psi_Phi' = psi_Phi/(2z^2) + (J_P psi_P + J_F psi_Phi + J_C psi_C)/z,
    psi_C' = K_P psi_P + K_C psi_C,
with J = int_0^1 D(z G0)(Y- + s psi) ds and K the same for the C-row: fixed functions of z, enclosed below on cells
by the Jacobian blocks of b5_formulas evaluated on lemma S's enclosure of the inner solution (which contains Y+, Y- and
the segment between them, because lemma S's eta-ball and the formal series are sigma-symmetric).

The transport generator. G(p) = J_F(0) = [Ct(0) .] - (3/2 + s), Ct(0) = [F^2] (b5_formulas.jac_blocks at z = 0), a
2 x 2 matrix with eigenvalues alpha +- i sigma, alpha = <F^2> - 3/2; G = V Lambda V^-1 (V enclosed over K).

Claims certified here (for every p in K):
  (L1) Fast-type solutions. For every k in C^2 there is a solution Psi(z) k of the secant equation on (0, Z_L] with
       psi_Phi = e^{-1/(2z)} V z^Lambda a~(z), a~(0+) = V^-1 k, and
           |a~(z) - a~(0)|_inf <= eps_L(z) |a~(0)|_inf,   eps_L(z) = exp(int_0^z g) - 1,
           g(t) = t^-delta (gamma_F + gamma_P) + 2 gamma_C kappa_P e^{kappa_C t} t^2,
       gamma_F = sup ||V^-1 ((J_F - G)/t) V||, gamma_P = sup ||V^-1 J_P V||, gamma_C = sup ||V^-1 J_C||,
       kappa_P = sup ||K_P V||, kappa_C = sup ||K_C|| (inf-norms, sup over the cells), delta = max |Re(lambda_1 -
       lambda_2)| over K. Proof (Volterra, Groenwall): proof/B_chain.md section 2.
       Equivalently, the label map L_z(Y) := e^{1/(2z)} z^{-G} Y_Phi satisfies L_z(Psi k) = (I + R(z)) k,
       R(z) = V R~(z) V^-1, ||R~(z)||_inf <= eps_L(z).
  (L2) Identification. The flat estimate of lemma 6a (lemma6_sector.flat_estimate), rerun on lemma S over K at
       Z_f, gives ||T0|| < 1, so the solutions of the secant equation that are O(z^k) at 0 form a space of dimension
       <= 2; the Psi k are such solutions and k -> Psi k is injective; so psi_in = Y+ - Y- = Psi k(p) for a unique
       k(p) = lim_{z -> 0} L_z(psi_in): the Stokes multiplier vector, holomorphic in p.

    python3 b5_label.py [--ZL=0.0081] [--Zf=0.004489] [--ncell=60] [--json=b5_label.json]
    python3 b5_label.py check            formula of (J_F - G)/z vs b5_formulas.jac_blocks (mpmath); point
                                          containment of the cell enclosures; certified eps_L vs the label variation
                                          of a double-precision integration of the linearized inner system
    python3 b5_label.py check --mutate=Ct2     drops the [Ct2 .] term of (J_F - G)/z: the formula check must fail
    python3 b5_label.py check --mutate=scale   scales every coupling enclosure by 0.01: the certified eps_L falls below
                                          the numerical label variation, and that check must fail
"""
import json
import math
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_mat, acb_poly, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5_bounds as BB  # noqa: E402
import b5_formulas as BF  # noqa: E402
import fixed_point as FP  # noqa: E402
import lemma6_sector as LS  # noqa: E402

ODD, EVEN = BF.ODD, BF.EVEN
proj = BF.proj
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
PREC = 192
up, ball, amax = BB.up, BB.ball, BB.amax
BOX = ("2e-6", "1e-6")          # the complex parameter box K of B5(1) (b5_bounds_k0.3_Z0.0085.json)
RP, KW = "0.14", 12             # lemma S as rerun by b5_bounds.py


# ----------------------------------------------------------------------------------------------------------------------
# the formula: (J_F - G) dPhi / z, analytic (no division by z)
# ----------------------------------------------------------------------------------------------------------------------
def jf_div(o, z, P, C, Ct, Ct2, dPhi):
    """(J_F(z) - G) dPhi / z, with J_F = d(z G0)/dPhi (b5_formulas.jac_blocks 'F'), G = [Ct(0) .] - (D2 + D)/2,
    Ct = (C - 1)/z, Ct2 = (Ct - Ct(0))/z:
        1/2 [Ct [Ct dPhi]] + [Ct2 dPhi] - 1/2 ([Ct D2 dQ] - D2 [Ct dQ]) + 1/2 [C E0 dQ],   dQ = Cop^-1 dPhi."""
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    CE = o.ee(C, E0)
    dQ = o.Cinv(C, dPhi)
    t1 = o.sc(o.one / 2, o.op(Ct, o.op(Ct, dPhi)))
    t2 = o.op(Ct2, dPhi)
    t3 = o.sc(-o.one / 2, o.add(o.op(Ct, o.D(dQ, 2)), o.sc(-o.one, o.D(o.op(Ct, dQ), 2))))
    t4 = o.sc(o.one / 2, o.op(CE, dQ))
    terms = [t1, t3, t4] if MUT == "Ct2" else [t1, t2, t3, t4]
    return o.add(*terms)


def G_of(o, Ct0):
    """G = [Ct(0) .] - (D2 + D)/2 as a 2 x 2 list (rows, cols in ODD order)"""
    return [[Ct0.get(a - b, o.zero) - ((o.one * 3 / 2 + o.s[a]) if a == b else o.zero) for b in ODD] for a in ODD]


# ----------------------------------------------------------------------------------------------------------------------
# eigen-decomposition of G over the box (closed form for 2 x 2)
# ----------------------------------------------------------------------------------------------------------------------
def eigen(Gm):
    """G = [[g11, g12], [g21, g22]] (acb balls): lambda_{1,2} = m +- r, r = sqrt(((g11 - g22)/2)^2 + g12 g21);
    V columns (g12, lambda_j - g11); returns lambda, V, V^-1 as acb_mat, and delta, mu_lo, mu_hi (arb bounds)"""
    g11, g12, g21, g22 = Gm[0][0], Gm[0][1], Gm[1][0], Gm[1][1]
    m = (g11 + g22) / 2
    d = (g11 - g22) / 2
    # d^2 + g12 g21 = -(w^2 - <F^2>_2 <F^2>_-2) is near the negative real axis (the principal branch cut of sqrt):
    # take r = i sqrt(-(d^2 + g12 g21)), whose argument is near 0
    r = acb(0, 1) * (-(d * d + g12 * g21)).sqrt()
    lam = [m + r, m - r]
    cols = [(g12, lam[j] - g11) for j in range(2)]
    # columns scaled to unit length (by an exact float at the midpoint): any scaling is a valid V; this one keeps the
    # inf-norms of V^-1 M V small
    sc = [arb(1 / float((abs(a) ** 2 + abs(b) ** 2).sqrt().mid())) for a, b in cols]
    V = acb_mat([[cols[0][0] * sc[0], cols[1][0] * sc[1]], [cols[0][1] * sc[0], cols[1][1] * sc[1]]])
    Vi = V.inv()
    re = [l.real for l in lam]
    delta = arb(abs(re[0] - re[1]).upper())
    mu_lo = arb(min(x.lower() for x in re))
    mu_hi = arb(max(x.upper() for x in re))
    return lam, V, Vi, delta, mu_lo, mu_hi


def matnorm(M, rows, cols):
    """inf-norm (max row sum of moduli), upper bound"""
    return arb(max(sum((up(M[i][j]) for j in range(cols)), arb(0)).upper() for i in range(rows)))


def tomat(cols_dicts, rows_keys):
    """list of column dicts (one per input direction) -> list of rows"""
    return [[col[r] for col in cols_dicts] for r in rows_keys]


def mm(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][l] * B[l][j] for l in range(k)), acb(0)) for j in range(m)] for i in range(n)]


def aslist(M):
    return [[M[i, j] for j in range(M.ncols())] for i in range(M.nrows())]


# ----------------------------------------------------------------------------------------------------------------------
# inner representation, flat estimate (L2)
# ----------------------------------------------------------------------------------------------------------------------
def setup(box=BOX):
    ctx.prec = PREC
    rep = BB.inner_rep(RP, KW, box[0], box[1])
    dp = BB.derived_polys(rep)
    ctx.prec = PREC
    return rep, dp


def flat_identification(rep, Zf):
    """lemma 6a's flat estimate (lemma6_sector.flat_estimate, unchanged) on lemma S over the complex box"""
    ctx.prec = 320
    sa, N, P, Q, C, R = FP.setup(rep["A"], rep["w"], RP, 320, rep["N"])
    S = dict(k=rep["k"], R=rep["R"], box="1", sa=sa, P=rep["P"], Q=rep["Q"], C=rep["C"], rho=rep["rho"],
             wts=rep["wts"])
    out = LS.flat_estimate(S, Zf, verbose=False)
    ctx.prec = PREC
    return out


# ----------------------------------------------------------------------------------------------------------------------
# cells on the real segment and the coefficient bounds
# ----------------------------------------------------------------------------------------------------------------------
def real_cells(ZL, n):
    """cells [t1, t2] covering (0, ZL]: the first (0, t0] with t0 = ZL/n, then n - 1 uniform cells"""
    ZL = arb(ZL)
    edges = [arb(0)] + [ZL * j / n for j in range(1, n + 1)]
    out = []
    for i in range(n):
        t1, t2 = edges[i], edges[i + 1]
        mid, rad = (t1 + t2) / 2, (t2 - t1) / 2
        zb = acb(arb(mid.mid(), (rad + mid.rad()).upper()))
        out.append((zb, t1, t2))
    return out


def cell_data(rep, dp, z, rmax):
    q = BB.cell_inner(rep, dp, z, rmax, arb(0))
    eta = rep["eta"]
    e2c = {k: eta[("c", k)][0] * rmax ** (eta[("c", k)][1] - 2) for k in EVEN}
    Ct2 = {}
    for k in EVEN:
        p2, ok = BB.shift_down(dp["Ct"][k], 1)
        Ct2[k] = p2(z) + ball(e2c[k])
    q["Ct2"] = Ct2
    return q


def cell_coefs(o, q, V, Vi, scale=None):
    """gamma_F, gamma_P, gamma_C, kappa_P, kappa_C on the cell (upper bounds) and the raw matrices"""
    z, P, Pt, C, Ct, Ct2 = q["z"], q["P"], q["Pt"], q["C"], q["Ct"], q["Ct2"]
    Phi = o.sc(z, Pt)
    unit = lambda comps, j: {k: (acb(1) if k == j else acb(0)) for k in comps}
    NF = tomat([jf_div(o, z, P, C, Ct, Ct2, unit(ODD, j)) for j in ODD], ODD)
    JP = tomat([BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dP=unit(ODD, j))["P"] for j in ODD], ODD)
    JC = tomat([BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dC=unit(EVEN, j))["C"] for j in EVEN], ODD)
    E0 = {k: v / 2 for k, v in proj(P, P, EVEN, acb(0)).items()}
    KP = tomat([{k: v / 2 for k, v in proj(C, proj(P, unit(ODD, j), EVEN, acb(0)), EVEN, acb(0)).items()} for j in ODD],
               EVEN)
    KC = tomat([{k: v / 2 for k, v in proj(unit(EVEN, j), E0, EVEN, acb(0)).items()} for j in EVEN], EVEN)
    Vl, Vil = aslist(V), aslist(Vi)
    tF = mm(mm(Vil, NF), Vl)
    tP = mm(mm(Vil, JP), Vl)
    tC = mm(Vil, JC)
    kP = mm(KP, Vl)
    out = dict(gF=matnorm(tF, 2, 2), gP=matnorm(tP, 2, 2), gC=matnorm(tC, 2, 3), kP=matnorm(kP, 3, 2),
               kC=matnorm(KC, 3, 3))
    if scale is not None:
        out = {k_: v * arb(scale) for k_, v in out.items()}
    return out, dict(NF=NF, JP=JP, JC=JC, KP=KP, KC=KC)


# ----------------------------------------------------------------------------------------------------------------------
# the certificate
# ----------------------------------------------------------------------------------------------------------------------
def run(ZL="0.0081", Zf="0.004489", ncell=60, box=BOX, verbose=True, scale=None):
    t_start = time.time()
    rep, dp = setup(box)
    ctx.prec = PREC
    o = BB.acb_ops(rep["w"])
    # G over the box from the constant term of Ct (= [F^2] exactly, as a function of p)
    Ct0 = {k: dp["Ct"][k].coeffs()[0] if dp["Ct"][k].degree() >= 0 else acb(0) for k in EVEN}
    Gm = G_of(o, Ct0)
    lam, V, Vi, delta, mu_lo, mu_hi = eigen(Gm)
    if not delta < arb("0.01"):
        raise ValueError("delta too large")
    flat = flat_identification(rep, Zf)
    cells = real_cells(ZL, ncell)
    I = arb(0)
    table = []
    worst = {"gF": arb(0), "gP": arb(0), "gC": arb(0), "kP": arb(0), "kC": arb(0)}
    for i, (zb, t1, t2) in enumerate(cells):
        q = cell_data(rep, dp, zb, up(t2))
        cf, _ = cell_coefs(o, q, V, Vi, scale)
        for k_, v in cf.items():
            worst[k_] = max(worst[k_], v, key=float)
        if i == 0:
            lin = t2 ** (1 - delta) / (1 - delta)
        else:
            lin = (t2 - t1) * t1 ** (-delta)
        cubic = (t2 ** 3 - t1 ** 3) / 3
        I = I + (cf["gF"] + cf["gP"]) * lin + 2 * cf["gC"] * cf["kP"] * (cf["kC"] * t2).exp() * cubic
        table.append((float(t2.upper()), float(I.upper()), float((I.exp() - 1).upper())))
    epsL = I.exp() - 1
    out = dict(ZL=ZL, Zf=Zf, ncell=ncell, box=list(box), rp=RP, k=KW, lemmaS_ok=rep["ok"], lemmaS_lip=float(rep["lip"].upper()),
               flat_Lambda=float(flat["lam"].upper()), flat_ok=flat["ok"], delta=float(delta.upper()),
               mu_lo=float(mu_lo.lower()), mu_hi=float(mu_hi.upper()),
               lam=[[lam[j].real.str(15), lam[j].imag.str(15)] for j in range(2)],
               V=[[[V[i, j].real.str(20), V[i, j].imag.str(20)] for j in range(2)] for i in range(2)],
               Vinv_norm=float(matnorm(aslist(Vi), 2, 2).upper()), V_norm=float(matnorm(aslist(V), 2, 2).upper()),
               coef_sup={k_: float(v.upper()) for k_, v in worst.items()},
               int_g=float(I.upper()), epsL=float(epsL.upper()), table=table[:: max(1, ncell // 12)] + [table[-1]],
               ok=bool(rep["ok"] and flat["ok"] and epsL < 1), time=time.time() - t_start)
    if verbose:
        print(f"lemma S over K = x^ + [+-{box[0]}]^2 + i[+-{box[1]}]^2: contraction {out['lemmaS_lip']:.4f}, ok {rep['ok']}")
        print(f"(L2) flat estimate on (0, {Zf}]: ||T0|| <= {out['flat_Lambda']:.4f} < 1: {flat['ok']}")
        print(f"G eigenvalues {lam[0].mid().str(8)}, {lam[1].mid().str(8)}; delta = |Re(l1 - l2)| <= {out['delta']:.2e}; "
              f"Re lambda in [{out['mu_lo']:.6f}, {out['mu_hi']:.6f}]; ||V|| <= {out['V_norm']:.4f}, ||V^-1|| <= "
              f"{out['Vinv_norm']:.4f}")
        print("(L1) coefficient sups over the cells: " + ", ".join(f"{k_} {v:.4g}" for k_, v in out["coef_sup"].items()))
        for (z, Iv, e) in out["table"]:
            print(f"   z = {z:.6f}: int_0^z g <= {Iv:.5f}, eps_L(z) <= {e:.5f}")
        print(f"=> ||R~(Z_L)||_inf <= eps_L({ZL}) = {out['epsL']:.5f}  (< 1: {bool(epsL < 1)});  all ok: {out['ok']}  "
              f"[{out['time']:.0f} s]")
    return out, dict(rep=rep, dp=dp, V=V, Vi=Vi, lam=lam, o=o)


# ----------------------------------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------------------------------
def check_formula(npts=4):
    """(J_F - G) dPhi / z vs (jac_blocks F(z) - jac_blocks F(z ~ 0)) / z at random complex states, mpmath; the state
    at z must have C = 1 + z Ct with Ct(0) shared: take C = 1 + z (c0 + z c1), Ct2 = c1 + ..."""
    import mpmath as mp
    import random
    mp.mp.dps = 50
    rnd = random.Random(31)
    worst = 0.0
    for _ in range(npts):
        cr = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        w = mp.mpf(rnd.uniform(1, 4))
        o = BF.mp_ops(w)
        z = mp.mpc(rnd.uniform(0.001, 0.01), rnd.uniform(-0.003, 0.003))
        P = {m: 2 * cr() for m in ODD}
        Pt = {m: 5 * cr() for m in ODD}
        Phi = o.sc(z, Pt)
        # C with C(0) = 1 and Ct(0) = [P P]/4 (as on the inner solution: Ct(0) = [F^2], P(0) = 2F), plus z-terms
        Ct0 = {k: v / 4 for k, v in proj(P, P, EVEN, mp.mpc(0)).items()}
        c1 = {k: cr() for k in EVEN}
        Ct = {k: Ct0[k] + z * c1[k] for k in EVEN}
        C = {k: (1 if k == 0 else 0) + z * Ct[k] for k in EVEN}
        G = G_of(o, Ct0)
        for j in ODD:
            u = {k: (mp.mpf(1) if k == j else mp.mpf(0)) for k in ODD}
            lhs = jf_div(o, z, P, C, Ct, c1, u)
            JF = BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dPhi=u)["F"]
            for i, m in enumerate(ODD):
                ref = (JF[m] - G[i][ODD.index(j)]) / z
                worst = max(worst, float(abs(lhs[m] - ref) / (1 + abs(ref))))
    return worst


def numeric_label_variation(zs, z0=1e-4):
    """double precision, NOT rigorous: the label propagation R~(z) of the linearized inner system along the real
    axis at x^ (inner solution from the series, b5_numeric.Inner), integrated in the rescaled variables
    u = e^{1/(2z)} psi from z0 with a~(z0) = e_i (eigen-coordinates); returns ||R~(z)||_inf at the given z"""
    import b5_numeric as BN
    import b5_system as BS
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm, eig
    w = BN.omega()
    inner = BN.Inner()
    A = float(BN.A_STAR)
    F = {1: -0.5j * A, -1: 0.5j * A}
    F2 = BS.proj(F, F, EVEN)
    Gn = np.array([[F2.get(a - b, 0) - ((1.5 + 1j * a * w) if a == b else 0) for b in ODD] for a in ODD])
    lam, Vn = eig(Gn)
    Vin = np.linalg.inv(Vn)
    npm = BN.NP(w)
    keys = [("P", m) for m in ODD] + [("F", m) for m in ODD] + [("C", k) for k in EVEN]

    def jac_at(z):
        Y, _ = BN.state_in(inner, z, w)
        J = np.zeros((7, 7), complex)
        h = 1e-7
        base = None
        for j, cj in enumerate(keys):
            Yp, Ym = dict(Y), dict(Y)
            Yp[cj] = Y[cj] + h
            Ym[cj] = Y[cj] - h
            fp, fm = npm.rhs0(z, Yp), npm.rhs0(z, Ym)
            for i, ci in enumerate(keys):
                J[i, j] = (fp[ci] - fm[ci]) / (2 * h)
        return J      # rows: P' (= Phi/(2z)), G0 (Phi' minus Phi/(2z^2)), C'

    def rhs(z, u):
        J = jac_at(z)
        du = J @ u
        # u = e^{1/(2z)} psi: u' = psi' e^{..} + u/(2z^2) (e^{1/2z})' ... careful: (e^{1/(2z)})' = -e^{1/(2z)}/(2z^2)
        # psi_Phi' = psi_Phi/(2z^2) + G0-part  ->  u_Phi' = G0-part(u);  slow rows: u' = J u - u/(2z^2)
        out = du.copy()
        out[0:2] = du[0:2] - u[0:2] / (2 * z * z)
        out[4:7] = du[4:7] - u[4:7] / (2 * z * z)
        return out

    res = {}
    cols = []
    for i in range(2):
        a0 = np.zeros(2, complex)
        a0[i] = 1.0
        uF = expm(Gn * np.log(z0)) @ (Vn @ a0)
        u0 = np.concatenate([z0 * uF, uF, np.zeros(3, complex)])
        sol = solve_ivp(rhs, (z0, max(zs)), u0, method="BDF", rtol=1e-10, atol=1e-14, dense_output=True)
        cols.append(sol)
    for z in zs:
        Rt = np.zeros((2, 2), complex)
        for i in range(2):
            u = cols[i].sol(z)
            a = Vin @ (expm(-Gn * np.log(z)) @ u[2:4])
            Rt[:, i] = a
        Rt -= np.eye(2)
        res[z] = float(np.max(np.sum(np.abs(Rt), axis=1)))
    return res


def check_main():
    ok = True
    fw = check_formula()
    f_ok = fw < 1e-30
    print(f"formula (J_F - G)/z vs jac_blocks: worst relative difference {fw:.2e} ({'PASS' if f_ok else 'FAIL'}, < 1e-30)")
    ok &= f_ok
    # certified eps_L vs the numerical label variation (the certified bound must dominate)
    scale = "0.01" if MUT == "scale" else None
    out, _ = run(ZL="0.0081", ncell=30, verbose=False, scale=scale)
    zs = [z for (z, _, _) in out["table"]]
    num = numeric_label_variation(zs)
    worst = 0.0
    for (z, Iv, e) in out["table"]:
        n_ = num[z]
        print(f"   z = {z:.5f}: numerical ||R~|| = {n_:.4f} (from z0 = 1e-4, not rigorous) <= certified eps_L = {e:.4f}"
              f": {n_ <= e}")
        worst = max(worst, n_ - e)
    d_ok = worst <= 0
    print(f"certified eps_L dominates the numerical label variation: {'PASS' if d_ok else 'FAIL'}"
          + (f"   [MUTATION {MUT}]" if MUT else ""))
    ok &= d_ok
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        ok = check_main()
        if MUT:
            print(f"MUTATION {MUT}: a check must FAIL: {'caught' if not ok else 'NOT caught'}")
            sys.exit(0 if not ok else 3)
        sys.exit(0 if ok else 1)
    out, _ = run(ZL=OPTS.get("ZL", "0.0081"), Zf=OPTS.get("Zf", "0.004489"), ncell=int(OPTS.get("ncell", "60")))
    if "json" in OPTS:
        with open(OPTS["json"], "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    sys.exit(0 if out["ok"] else 1)
