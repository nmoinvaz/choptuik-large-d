#!/usr/bin/env python3
"""Lemma B5(1) for M = 1 (proof/B5_inner.md, sections 3-4): the constants of the fixed-point map in ball arithmetic
(python-flint arb/acb), and the resulting n0(1).

What is computed (every number an upper bound, rigorous):
  1. The inner solution on the working domain W_n (|z| <= Z, 0 <= arg z <= pi/2 + kappa) as Y_N + eta: Y_N the
     truncated formal series (series_arb.py), eta bounded by lemma S of proof/A_certify.md section 11.3, rerun here
     over a COMPLEX parameter box around the certified zero (lemma6_sector.py's construction, unchanged). Lemma S's
     domain D contains W_n when Z <= R+ sin(pi/4 - kappa) (and kappa <= 0.385, so that no sector lemma is needed).
  2. A covering of {|z| <= Z, 0 <= arg z <= pi/2 + kappa} by a disc cell around 0 and polar cells; on each cell,
     enclosures of the inner quantities (P, Phi/z, C, (C-1)/z, Q, Q' = S/(2z^2)) including eta.
  3. The Lipschitz matrix of the eps -> 0 part of the W-system (Jacobian blocks of b5_formulas.jac_blocks and of the
     C-row), the kernel factors of section 3, and the eps-part (Cauchy-Schwarz estimate on the scaled exact field
     b5_formulas.g_scaled over a complex t-disc and a state polydisc).
  4. The forcing: R_Phi = sup |z r_Phi/eps| (b5_formulas.res_phi), R_C = sup |[C E0]|/2, TF = sup |z F_vt|.
  5. Perron weights, the contraction constant Lambda < 1, the bound rho (as an affine function of the data delta of
     hypothesis (H8)), and the conditions on n.

    python3 b5_bounds.py [--kappa=0.3] [--Z=0.008] [--rp=0.14] [--k=12] [--box=1e-9,1e-6] [--n0exp=100] [--gamma=0.3]
          [--kappa1=0.6] [--json=out.json] [--mutate=NAME]
      --box=re,im: parameter box x^ + [-re, re]^2 + i[-im, im]^2 (complex, so that Cauchy estimates give C^1)
      --n0exp: the run assumes n >= exp(n0exp) (eps0 = e^-n0exp, t0 = e^-2 gamma n0exp) and checks that it is enough
    mutations (each must make the check fail, i.e. Lambda >= 1 or a failed consistency test):
      --mutate=a        fast-kernel rate taken as sin(kappa) instead of sin(kappa)/2 is NOT a mutation of the input
                        (it makes the bound smaller); instead: --mutate=jac doubles the Phi <- P Jacobian block
                        (the dominant coupling) and the run must lose the contraction at the default Z
      --mutate=series   one series coefficient off by 1e-6 relative: lemma S must fail
"""
import json
import math
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_mat, acb_poly, ctx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed_point as FP  # noqa: E402
import lemma6_sector as LS  # noqa: E402
import b5_formulas as BF  # noqa: E402

ODD, EVEN = BF.ODD, BF.EVEN
proj = BF.proj
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
PREC = 192
# the data bound assumed in (H8): on the inner arc |W_P| <= dP eps, |W_Phi| <= dF eps r_n, |W_C| <= dC eps,
# |W_vt| <= dv eps (every harmonic). Leading-order values (b5_data.py): dP ~ 0 (K_1 = 0), dC = A^2/2 = 3.23,
# dv = A^2 = 6.46, dF = 2|P^(1)_1|; the reference values below include a margin.
DELTA_REF = {k_: float(v) for k_, v in zip("PFCv", OPTS.get("delta", "1,200,10,20").split(","))}


def up(x):
    return arb(abs(x).upper())


def ball(r):
    r = arb(r)
    return acb(arb(0, r.upper()), arb(0, r.upper()))


def amax(d):
    return max((up(v) for v in d.values()), key=lambda x: float(x))


def solve_acb(A, b):
    n = len(b)
    M = acb_mat([[acb(A[i][j]) for j in range(n)] for i in range(n)])
    x = M.solve(acb_mat([[acb(v)] for v in b]))
    return [x[i, 0] for i in range(n)]


def acb_ops(w):
    return BF.Ops(w, solve_acb, acb(0), acb(1), acb(0, 1))


# ----------------------------------------------------------------------------------------------------------------------
# 1. the inner solution: lemma S over a complex parameter box
# ----------------------------------------------------------------------------------------------------------------------
def inner_rep(rp, k, box_re, box_im):
    """lemma S over the complex box (the representation used), preceded by a strict run at the box centre: there the
    structural vanishing of the residual below order N is checked to rounding level (over a box the coefficient balls
    are as wide as the box, so only containment of 0 can be checked)"""
    ctx.prec = 320
    FP.Maj.BOX = False
    centre = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), rp, verbose=False)
    N, kmins = centre["N"], centre["kmins"]
    strict = _lemma_S(rp, k, acb(FP.A_STAR), acb(FP.WF_STAR), N, kmins, box=False)
    rep = _lemma_S(rp, k, acb(arb(FP.A_STAR, arb(box_re)), arb(0, arb(box_im))),
                   acb(arb(FP.WF_STAR, arb(box_re)), arb(0, arb(box_im))), N, kmins, box=True)
    rep["ok_centre"] = strict["ok"]
    rep["ok"] = rep["ok"] and strict["ok"]
    return rep


def _lemma_S(rp, k, A, wf, N, kmins, box):
    ctx.prec = 320
    FP.Maj.BOX = box
    W0 = 2 * arb.pi() / arb("1.64389375")
    w = acb(W0) * wf
    sq2 = arb(2).sqrt()
    Rp = arb(rp) ** 2
    a_eff = (sq2 / 4 - k * Rp / 2).lower()
    sa, N, P, Q, C, R = FP.setup(A, w, rp, 320, N)
    if MUT == "series":
        cs = P[1].coeffs()
        cs[5] = cs[5] * (1 + acb("1e-6"))
        P[1] = acb_poly(cs)
    FP.Maj.R = R
    Np, Nv, Nc, V = FP.residual_numerators(sa, P, Q, C)
    res, lows = {}, []
    kin = iter(kmins)
    for mu in sa.odd:
        m, km, ok = FP.tail_maj(Np[mu], 1, arb(0.5), next(kin)); res[("p", mu)] = m; lows.append(ok)
        m, km, ok = FP.tail_maj(Nv[mu], 2, arb(0.5), next(kin)); res[("v", mu)] = m; lows.append(ok)
    for nu in sa.even:
        m, km, ok = FP.tail_maj(Nc[nu], 0, arb(1), next(kin)); res[("c", nu)] = m; lows.append(ok)
    with LS.patched(a_eff):
        L, comps, kw, rows = FP.lipschitz_and_forcing(sa, P, Q, C, k, arb("1e-20"), [1.0] * 7)
        wts = LS.perron_weights(L, LS.sym_index(comps))
        Fv = [FP.intbound(res[cp], "fast" if cp[0] == "v" else "slow", kw[cp]) for cp in comps]
        F = arb(max((arb(wts[j]) * Fv[j]).upper() for j in range(len(comps))))
        lip0 = LS.weighted_norm(L, wts)
        rho = 4 * F / (1 - lip0) if lip0 < 1 else arb(1)
        L, comps, kw, rows = FP.lipschitz_and_forcing(sa, P, Q, C, k, rho, wts)
        lip = LS.weighted_norm(L, wts)
    FP.Maj.BOX = False
    selfmap = F + lip * rho
    ok = bool(lip < 1) and bool(selfmap < rho) and all(lows)
    ctx.prec = PREC
    # eta bounds: |eta_j(z)| <= (rho / w_j) |z|^{kw_j} on D
    eta = {cp: (rho / arb(wts[i]), kw[cp]) for i, cp in enumerate(comps)}
    return dict(N=N, k=k, R=R, rp=rp, lip=lip, ok=ok, P=P, Q=Q, C=C, w=w, eta=eta, rho=rho, wts=wts, comps=comps,
                a_eff=arb(a_eff), A=A, wf=wf)


# ----------------------------------------------------------------------------------------------------------------------
# 2. polynomials derived from Y_N (exact divisions by z, after checking that the dropped coefficients contain 0)
# ----------------------------------------------------------------------------------------------------------------------
def pconv(A, B, keep):
    out = {k: acb_poly([0]) for k in keep}
    for i, a in A.items():
        for j, b in B.items():
            if i + j in out:
                out[i + j] = out[i + j] + a * b
    return out


def shift_down(p, m):
    cs = p.coeffs()
    lows_ok = all(c.contains(0) for c in cs[:m])
    return acb_poly(cs[m:] if len(cs) > m else [0]), lows_ok


def derived_polys(rep):
    w = rep["w"]
    P, Q, C = rep["P"], rep["Q"], rep["C"]
    X = acb_poly([0, 1])
    s = {m: acb(0, 1) * m * w for m in ODD}
    CQ = pconv(C, Q, ODD)
    Phi = {m: CQ[m] - (1 + s[m]) * P[m] for m in ODD}
    Pt, oks = {}, []
    for m in ODD:
        Pt[m], ok = shift_down(Phi[m], 1); oks.append(ok)
    Ct = {}
    for k in EVEN:
        Ct[k], ok = shift_down(C[k] - (1 if k == 0 else 0), 1); oks.append(ok)
    E = {k: v * acb(0.5) for k, v in pconv(P, P, EVEN).items()}
    CE = pconv(C, E, EVEN)
    CPhi = pconv(C, Phi, ODD)
    CEP = pconv(CE, P, ODD)
    S = {m: CPhi[m] - X * (2 + s[m]) * Q[m] + X * CEP[m] for m in ODD}
    Sz2 = {}
    for m in ODD:
        Sz2[m], ok = shift_down(S[m], 2); oks.append(ok)
    return dict(Pt=Pt, Ct=Ct, Sz2=Sz2, ok=all(oks))


# ----------------------------------------------------------------------------------------------------------------------
# 3. cells
# ----------------------------------------------------------------------------------------------------------------------
def make_cells(Z, alpha, n_r=6, d_theta=0.08, c0=0.12):
    """disc |z| <= c0 Z (a square ball) and polar cells covering c0 Z <= |z| <= Z, 0 <= arg z <= alpha"""
    cells = []
    R0 = arb(c0) * arb(Z)
    cells.append((ball(R0 * arb(2).sqrt() / 2 * arb(2).sqrt()), arb(0), R0 * arb(2).sqrt()))  # square of half-side R0
    radii = [R0 * (arb(Z) / R0) ** (arb(i) / n_r) for i in range(n_r + 1)]
    nth = max(1, int(math.ceil(float(alpha) / d_theta)))
    for i in range(n_r):
        r1, r2 = radii[i], radii[i + 1]
        rm, rr = (r1 + r2) / 2, (r2 - r1) / 2
        rb = arb(rm.mid(), (rr + rm.rad() + rr.rad()).upper())
        for j in range(nth):
            t1, t2 = arb(alpha) * j / nth, arb(alpha) * (j + 1) / nth
            tm, tr = (t1 + t2) / 2, (t2 - t1) / 2
            tb = arb(tm.mid(), (tr + tm.rad() + tr.rad()).upper())
            z = acb(rb * tb.cos(), rb * tb.sin())
            cells.append((z, r1, r2))
    return cells


# ----------------------------------------------------------------------------------------------------------------------
# 4. per-cell enclosures of the inner quantities
# ----------------------------------------------------------------------------------------------------------------------
def cell_inner(rep, dp, z, rmax, wpad):
    """balls for P, Pt = Phi/z, C, Ct = (C-1)/z, Q, Qp = S/(2z^2) on the cell (inner solution + eta), plus the
    W-perturbation pad wpad on P, Pt, C, Ct (the state is Y_in + W with |W| inside the bootstrap ball)"""
    w = rep["w"]
    s = {m: acb(0, 1) * m * w for m in ODD}
    D = {m: up(1 + s[m]) for m in ODD}
    eta = rep["eta"]
    ep = {m: eta[("p", m)][0] * rmax ** eta[("p", m)][1] for m in ODD}          # |eta_p|
    ev = {m: eta[("v", m)][0] * rmax ** eta[("v", m)][1] for m in ODD}
    ec = {k: eta[("c", k)][0] * rmax ** eta[("c", k)][1] for k in EVEN}
    eq = {m: ev[m] + D[m] * ep[m] for m in ODD}
    P = {m: rep["P"][m](z) + ball(ep[m]) + ball(wpad) for m in ODD}
    Q = {m: rep["Q"][m](z) + ball(eq[m]) for m in ODD}
    C = {k: rep["C"][k](z) + ball(ec[k]) + ball(wpad) for k in EVEN}
    # |eta|/|z| and |eta|/|z|^2 bounds
    e1p = {m: eta[("p", m)][0] * rmax ** (eta[("p", m)][1] - 1) for m in ODD}
    e1v = {m: eta[("v", m)][0] * rmax ** (eta[("v", m)][1] - 1) for m in ODD}
    e1c = {k: eta[("c", k)][0] * rmax ** (eta[("c", k)][1] - 1) for k in EVEN}
    e2p = {m: eta[("p", m)][0] * rmax ** (eta[("p", m)][1] - 2) for m in ODD}
    e2v = {m: eta[("v", m)][0] * rmax ** (eta[("v", m)][1] - 2) for m in ODD}
    e2c = {k: eta[("c", k)][0] * rmax ** (eta[("c", k)][1] - 2) for k in EVEN}
    Ct = {k: dp["Ct"][k](z) + ball(e1c[k]) + ball(wpad) for k in EVEN}
    # Phi/z: Phi(Y_in) - Phi(Y_N) = [eta_c Q] + [C_N eta_Q] - D eta_p, divided by z
    sQ = sum((up(v) for v in Q.values()), arb(0))
    sC = sum((up(v) for v in C.values()), arb(0))
    e1max = max([*e1p.values(), *e1v.values(), *e1c.values()], key=lambda x: float(x))
    Dmax = max(D.values(), key=lambda x: float(x))
    dPhi_z = (sQ + sC * (1 + Dmax) + Dmax) * e1max * 2
    Pt = {m: dp["Pt"][m](z) + ball(dPhi_z) + ball(wpad) for m in ODD}
    # Q' = S(Y_in)/(2 z^2): S_N/z^2 /2 + (S(Y_in) - S(Y_N))/(2 z^2); |DS| bounded generously on the cell
    sP = sum((up(v) for v in P.values()), arb(0))
    D2max = max((up(2 + s[m]) for m in ODD), key=lambda x: float(x))
    zmax = up(z)
    LS_ = (sC * sC + sC * Dmax) + zmax * D2max + (sC * Dmax + 2 * zmax * sC * sP * sP) + \
          (2 * sC * sQ + Dmax * sP + zmax * sP * sP * sP)            # |dS/dQ| + |dS/dP| + |dS/dC| (convolution sums)
    e2max = max([*e2p.values(), *e2v.values(), *e2c.values()], key=lambda x: float(x))
    dS = LS_ * e2max * 2 * (1 + Dmax)
    Qp = {m: dp["Sz2"][m](z) / 2 + ball(dS / 2) for m in ODD}
    return dict(P=P, Q=Q, C=C, Ct=Ct, Pt=Pt, Qp=Qp, z=z, vpad=3 * wpad)


def dQhat(o, z, e, P, Pt, C):
    """dQh with Q(exact) = Q0 + eps dQh at vt = 0 (the formula of b5_formulas.res_phi)"""
    one = o.one
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    CE = o.ee(C, E0)
    B0 = lambda X: o.op(CE, X)
    Cop = lambda X: o.op(C, X)
    Q0 = o.Cinv(C, o.add(o.sc(z, Pt), o.D(P)))
    A = [[sum((C.get(a_ - c_, o.zero) * ((one - e * (1 + o.s[c_])) if c_ == b_ else o.zero) for c_ in ODD), o.zero)
          + e * (1 - e) * z * CE.get(a_ - b_, o.zero) for b_ in ODD] for a_ in ODD]
    rhs = o.add(o.sc(-one, Cop(Cop(Pt))), o.sc(-(1 - e), Cop(B0(P))), Cop(o.D(Q0)), o.sc(-(1 - e) * z, B0(Q0)))
    return dict(zip(ODD, o.solve_(A, [rhs[m] for m in ODD]))), Q0


# ----------------------------------------------------------------------------------------------------------------------
# 5. per-cell bounds
# ----------------------------------------------------------------------------------------------------------------------
COMPS = [("P", m) for m in ODD] + [("F", m) for m in ODD] + [("C", k) for k in EVEN] + [("v", k) for k in EVEN]


def unit(comps, j):
    return {k: (acb(1) if k == j else acb(0)) for k in comps}


def cell_jac(o, q):
    """main (eps -> 0) Jacobian entries on the cell: Phi row (scaled: coefficient of |W_j|/|t|), C row (O(1))"""
    z, P, Pt, C, Ct = q["z"], q["P"], q["Pt"], q["C"], q["Ct"]
    Phi = o.sc(z, Pt)
    J = {}
    for j in ODD:
        col = BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dP=unit(ODD, j))["P"]
        for m in ODD:
            J[(("F", m), ("P", j))] = up(col[m])
        col = BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dPhi=unit(ODD, j))["F"]
        for m in ODD:
            J[(("F", m), ("F", j))] = up(col[m])
    for j in EVEN:
        col = BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dC=unit(EVEN, j))["C"]
        for m in ODD:
            J[(("F", m), ("C", j))] = up(col[m])
    if MUT == "jac":
        for key in list(J):
            if key[0][0] == "F" and key[1][0] == "P":
                J[key] = J[key] * 2
    # C row: C' = [C (E0 + vt)]/2 -> d/dP = [C [P dP]]/2, d/dC = [dC (E0 + vt)]/2, d/dvt = [C dvt]/2  (vt = pad only)
    E0 = {k: v / 2 for k, v in proj(P, P, EVEN, acb(0)).items()}
    for j in ODD:
        col = {k: v / 2 for k, v in proj(C, proj(P, unit(ODD, j), EVEN, acb(0)), EVEN, acb(0)).items()}
        for k in EVEN:
            J[(("C", k), ("P", j))] = up(col[k])
    for j in EVEN:
        col = {k: v / 2 for k, v in proj(unit(EVEN, j), E0, EVEN, acb(0)).items()}
        for k in EVEN:
            J[(("C", k), ("C", j))] = up(col[k]) + arb(q.get("vpad", 0))
        col = {k: v / 2 for k, v in proj(C, unit(EVEN, j), EVEN, acb(0)).items()}
        for k in EVEN:
            J[(("C", k), ("v", j))] = up(col[k])
    return J


def cell_residual(o, q, eps0, t0):
    z, P, Pt, C, Qp, Q = q["z"], q["P"], q["Pt"], q["C"], q["Qp"], q["Q"]
    e = acb(arb(eps0 / 2, eps0 / 2))
    o.t = acb(arb(0, t0), arb(0, t0))
    rp = BF.res_phi(o, z, e, P, Pt, C, Qp)
    R_phi = amax(rp)
    E0 = {k: v / 2 for k, v in proj(P, P, EVEN, acb(0)).items()}
    R_C = amax({k: v / 2 for k, v in proj(C, E0, EVEN, acb(0)).items()})
    # exact Q at the inner state = Q0 + eps dQh
    dQh, Q0 = dQhat(o, z, e, P, Pt, C)
    Qex = {m: Q0[m] + e * dQh[m] for m in ODD}
    TF = amax(BF.tF_vt(o, z, e, P, Pt, Qex))
    return R_phi, R_C, TF


def cell_vt_lip(o, q, eps0, Z, rc="0.2"):
    """Lipschitz bound of the vt-row integrand t*Xi in the state (P, Pt, C, vt) (all components summed), by Cauchy's
    estimate on a polydisc of radius rc in (P, Pt) and 0.01 in C, with Q recomputed from the ball; plus the explicit
    vt-terms |t| (|E0| + 2|vt|)/2. Only enters multiplied by 2 eps0."""
    z = q["z"]
    r = arb(rc)
    e = acb(arb(eps0 / 2, eps0 / 2))
    Pb = {m: q["P"][m] + ball(r) for m in ODD}
    Ptb = {m: q["Pt"][m] + ball(r) for m in ODD}
    Cb = {k: q["C"][k] + ball("0.01") for k in EVEN}
    dQh, Q0 = dQhat(o, z, e, Pb, Ptb, Cb)
    Qb = {m: Q0[m] + e * dQh[m] for m in ODD}
    sup = amax(BF.tF_vt(o, z, e, Pb, Ptb, Qb))
    E0 = {k: v / 2 for k, v in proj(q["P"], q["P"], EVEN, acb(0)).items()}
    ncomp = len(ODD) * 2 + len(EVEN)
    return sup / (r / 2) * ncomp + sup / (arb("0.01") / 2) * len(EVEN) + arb(Z) * (amax(E0) + 1)


def cell_eps_part(o, q, t0, tau1="0.01", rad=None):
    """Cauchy-Schwarz bound of the eps-part Lipschitz coefficients ell_j (coefficient of |W_j|/|t|, and for Phi of
    |W_Phi|/|t|^2): g(z, t', Y') on |t'| <= tau1 (complex) and a state polydisc of radii rad around Y_in; for
    |t| <= t0: |d_{Y_j}(g(t) - g(0))| <= 4 G t0 / (rad_j tau1)"""
    rad = rad or {"P": arb("0.2"), "F": arb("2"), "C": arb("0.01"), "v": arb("0.2")}
    z = q["z"]
    tau = arb(tau1)
    tb = acb(arb(0, tau), arb(0, tau))
    P = {m: q["P"][m] + ball(rad["P"]) for m in ODD}
    Pt = {m: q["Pt"][m] + ball(rad["F"]) for m in ODD}
    C = {k: q["C"][k] + ball(rad["C"]) for k in EVEN}
    vt = {k: ball(rad["v"]) for k in EVEN}
    g = BF.g_scaled(o, z, tb, P, Pt, C, vt)
    G = amax(g)
    ell = {key: 4 * G * arb(t0) / (rad[key] * tau) for key in rad}
    return ell, G


# ----------------------------------------------------------------------------------------------------------------------
# 6. assembly
# ----------------------------------------------------------------------------------------------------------------------
def assemble(J, ell, Z, kappa, eps0, vt_lip):
    """the Lipschitz matrix L of the fixed-point map (unweighted entries, norm sup w_j |W_j|/(eps m_j)),
    m_P = m_C = m_vt = 1, m_Phi = |z|; kernel factors of proof/B5_inner.md section 3"""
    n = len(COMPS)
    idx = {c: i for i, c in enumerate(COMPS)}
    L = [[arb(0)] * n for _ in range(n)]
    sk = arb(kappa).sin()
    a = sk / 2
    Za = arb(Z)
    f1 = 1 / (a - sk * Za)              # fast kernel, integrand |t|^-1: |z| / (a - sin k Z)
    f2 = 1 / (a - 2 * sk * Za)          # fast kernel, integrand |t|^0: |z|^2 / (a - 2 sin k Z)
    if not (a - 2 * sk * Za > 0):
        raise ValueError("fast kernel not integrable: need Z < 1/4")
    for m in ODD:
        L[idx[("P", m)]][idx[("F", m)]] = Za / 2
    for (ci, cj), g in J.items():
        i, j = idx[ci], idx[cj]
        if ci[0] == "F":
            L[i][j] = g * (Za * f2 if cj[0] == "F" else f1)
        else:
            L[i][j] = g * Za
    # eps part of the Phi row: |d_j Delta| <= ell_j/|t| (j = P, C, vt) and ell_F/|t|^2 (Phi): integrand q = -1 always
    for ci in COMPS:
        if ci[0] != "F":
            continue
        for cj in COMPS:
            L[idx[ci]][idx[cj]] = L[idx[ci]][idx[cj]] + ell[cj[0]] * f1
    # eps part of the C row: -eps [C (E0 + vt)]/2: Lipschitz eps0 * (|J_C| + 1), integrated radially (q = 0)
    for ci in COMPS:
        if ci[0] != "C":
            continue
        for cj in COMPS:
            if cj[0] in ("P", "C", "v"):
                L[idx[ci]][idx[cj]] = L[idx[ci]][idx[cj]] + arb(eps0) * (J.get((ci, cj), arb(0)) + 1) * Za
    # vt row: the kernel gives 2/(n+1) <= 2 eps0 times the Lipschitz constant of z (vt-row forcing)
    for ci in COMPS:
        if ci[0] != "v":
            continue
        for cj in COMPS:
            L[idx[ci]][idx[cj]] = 2 * arb(eps0) * vt_lip
    return L, idx


def perron(L):
    Lf = np.array([[float(x.upper()) for x in row] for row in L]) + 1e-12
    ev, evec = np.linalg.eig(Lf)
    i = int(np.argmax(ev.real))
    u = np.abs(evec[:, i].real) + 1e-6 * np.abs(evec[:, i].real).max()
    w = 1.0 / u
    # symmetric in mu -> -mu
    idx = {c: i for i, c in enumerate(COMPS)}
    w = [min(w[idx[c]], w[idx[(c[0], -c[1])]]) for c in COMPS]
    return w


def weighted(L, w):
    return arb(max((sum(arb(w[i]) * L[i][j] / arb(w[j]) for j in range(len(L)))).upper() for i in range(len(L))))


def run(kappa=0.3, Z=0.008, rp="0.14", k=12, box=("1e-9", "1e-6"), n0exp=100, gamma=0.3, kappa1=0.6,
        rhoW=arb("1e12"), verbose=True):
    t_start = time.time()
    ctx.prec = PREC
    alpha = arb.pi() / 2 + arb(kappa)
    rep = inner_rep(rp, k, box[0], box[1])
    Rp = arb(rp) ** 2
    cover = Rp * (arb.pi() / 4 - arb(kappa)).sin()
    dom_ok = bool(arb(Z) <= cover) and kappa <= 0.385
    dp = derived_polys(rep)
    ctx.prec = PREC
    eps0 = arb(-n0exp).exp()
    t0 = arb(-2 * gamma * n0exp).exp()
    o = acb_ops(rep["w"])
    cells = make_cells(Z, alpha)
    wpad = rhoW * (eps0 if eps0 > t0 else t0)          # |W_j|/m_j <= rho eps/w_j; w_j >= 1 after normalization
    J, R_phi, R_C, TF, ell, Gmax = {}, arb(0), arb(0), arb(0), {}, arb(0)
    vt_lip = arb(0)
    for (z, r1, r2) in cells:
        q = cell_inner(rep, dp, z, up(r2), wpad)
        Jc = cell_jac(o, q)
        for key, v in Jc.items():
            J[key] = v if key not in J else (v if v > J[key] else J[key])
        rph, rc, tf = cell_residual(o, q, eps0, t0)
        R_phi, R_C, TF = max(R_phi, rph, key=float), max(R_C, rc, key=float), max(TF, tf, key=float)
        el, G = cell_eps_part(o, q, t0)
        for key, v in el.items():
            ell[key] = v if key not in ell else max(ell[key], v, key=float)
        Gmax = max(Gmax, G, key=float)
        vt_lip = max(vt_lip, cell_vt_lip(o, q, eps0, Z), key=float)
    L, idx = assemble(J, ell, Z, kappa, eps0, vt_lip)
    w = perron(L)
    lam = weighted(L, w)
    sk = arb(kappa).sin()
    a = sk / 2
    # forcing (weighted): Phi row R_phi/(a - sin k Z), C row R_C Z, vt row 2 TF
    Fj = {}
    for c in COMPS:
        if c[0] == "F":
            Fj[c] = R_phi / (a - sk * arb(Z))
        elif c[0] == "C":
            Fj[c] = R_C * arb(Z)
        elif c[0] == "v":
            Fj[c] = 2 * TF
        else:
            Fj[c] = arb(0)
    F = arb(max((arb(w[idx[c]]) * Fj[c]).upper() for c in COMPS))
    out = dict(kappa=kappa, Z=Z, rp=rp, k=k, box=box, n0exp=n0exp, gamma=gamma, lemmaS_lip=float(rep["lip"].upper()),
               lemmaS_ok=rep["ok"], polys_ok=dp["ok"], dom_ok=dom_ok, cover=float(cover.lower()),
               lam=float(lam.upper()), F=float(F.upper()), weights=w, R_phi=float(R_phi.upper()),
               R_C=float(R_C.upper()), TF=float(TF.upper()), Gmax=float(Gmax.upper()),
               ell={k_: float(v.upper()) for k_, v in ell.items()}, ncells=len(cells))
    # componentwise super-solution: b_j = F_j + delta_j (data of (H8)), find x with b + L x < x (rigorous); then the
    # bootstrap gives |W_j| <= eps m_j x_j. Done for the data delta = DELTA_REF and for delta = 0 (forcing only).
    Lf = np.array([[float(x_.upper()) for x_ in row] for row in L])
    supsol = {}
    zero = arb(0)
    cases = [("data", {c: Fj[c] + arb(DELTA_REF[c[0]]) for c in COMPS}),
             ("forcing", dict(Fj))]
    cases += [(f"unit_{blk}", {c: (arb(1) if c[0] == blk else zero) for c in COMPS}) for blk in "PFCv"]
    for tag, bdict in cases:
        b0 = [bdict[c] for c in COMPS]
        bmax = max(float(v.upper()) for v in b0)
        b = [v + arb(1e-6 * bmax) for v in b0]           # a positive floor, so that every row has slack
        bf = np.array([float(v.upper()) for v in b])
        try:
            # solve with a margin (1e-3 relative + 1e-3 bmax per row), so that b + L x < x holds with room to spare
            xf = np.linalg.solve(np.eye(len(COMPS)) - Lf, bf * (1 + 1e-3) + 1e-3 * bmax)
        except np.linalg.LinAlgError:
            xf = np.full(len(COMPS), np.inf)
        good = bool(np.all(np.isfinite(xf)) and np.all(xf > 0))
        if good:
            xa = [arb(float(v)) for v in xf]
            for i in range(len(COMPS)):
                lhs = b[i] + sum((L[i][j] * xa[j] for j in range(len(COMPS))), arb(0))
                if not lhs < xa[i]:
                    good = False
            if good and not max(xa, key=float) <= rhoW:
                good = False
        supsol[tag] = dict(ok=good, x={f"{c[0]}{c[1]:+d}": (float(xa[i].upper()) if good else None)
                                       for i, c in enumerate(COMPS)})
    out["supersolution"] = supsol
    out["delta_ref"] = DELTA_REF
    # the criterion of the bootstrap (proof/B5_inner.md section 3) is the super-solution b + L x < x; for a nonnegative L
    # it implies spectral radius < 1. The Perron-weighted norm lam is reported for information.
    ok = out["lemmaS_ok"] and out["polys_ok"] and dom_ok and supsol["data"]["ok"]
    out["ok"] = bool(ok)
    # weighted max rows for the report
    out["Lmax_entries"] = sorted(((float(L[i][j].upper()), f"{COMPS[i][0]}{COMPS[i][1]:+d}<-{COMPS[j][0]}{COMPS[j][1]:+d}")
                                  for i in range(len(COMPS)) for j in range(len(COMPS))), reverse=True)[:10]
    # n0 from the domain condition: Z_n / cos(kappa) <= Z, Z_n = kappa1/ln n
    out["ln_n0_domain"] = kappa1 / (Z * math.cos(kappa))
    out["time"] = time.time() - t_start
    if verbose:
        print(f"kappa = {kappa}, Z = {Z} (lemma S: rp = {rp}, k = {k}, contraction {out['lemmaS_lip']:.4f}, ok "
              f"{out['lemmaS_ok']}; D covers |z| <= {out['cover']:.5f} in 0 <= arg z <= pi/2 + kappa: {dom_ok}); "
              f"derived polynomials exact: {dp['ok']}; {len(cells)} cells")
        print(f"  Lambda <= {out['lam']:.4f}; forcing (weighted) F <= {out['F']:.4g}; R_Phi <= {out['R_phi']:.4g}, "
              f"R_C <= {out['R_C']:.4g}, TF <= {out['TF']:.4g}; eps-part ell = "
              + ", ".join(f"{k_} {v:.2e}" for k_, v in out["ell"].items()) + f" (G <= {out['Gmax']:.3g})")
        print("  largest L entries: " + "; ".join(f"{s} {v:.3g}" for v, s in out["Lmax_entries"][:6]))
        print(f"  Perron weights " + ", ".join(f"{c[0]}{c[1]:+d} {wv:.4g}" for c, wv in zip(COMPS, w)))
        for tag, ss in out["supersolution"].items():
            lab = {"data": f"forcing + data delta = {DELTA_REF}", "forcing": "forcing only"}.get(tag, f"unit data in {tag[5:]}")
            print(f"  super-solution ({lab}): certified {ss['ok']}: x = "
                  + (", ".join(f"{k_} {v:.4g}" for k_, v in ss["x"].items()) if ss["ok"] else "-"))
        print(f"  => contraction: {ok};  domain condition gives ln n0 >= kappa1/(Z cos kappa) = {out['ln_n0_domain']:.2f} "
              f"(kappa1 = {kappa1}); the run assumed ln n >= {n0exp}  [{out['time']:.0f} s]")
    return out


def point_checks(kappa=0.3, rp="0.14", k=12, box=("2e-6", "1e-6"), npts=4):
    """the rigorous enclosures on a tiny cell (a point) must contain values computed independently: the inner
    solution from the double-precision series (b5_numeric.Inner), z r_Phi/eps from b5_system.Model in mpmath, and
    the Jacobian blocks from finite differences of b5_system.Model.f_0. Returns the worst excess (0 if contained)."""
    import mpmath as mp
    import b5_numeric as BN
    import b5_system as BS
    ctx.prec = PREC
    rep = inner_rep(rp, k, box[0], box[1])
    dp = derived_polys(rep)
    ctx.prec = PREC
    o = acb_ops(rep["w"])
    inner = BN.Inner()
    wf_ = float(BN.omega())
    rng = np.random.default_rng(7)
    worst = {"inner": 0.0, "res": 0.0, "jac": 0.0}
    for _ in range(npts):
        r = rng.uniform(5e-4, 6e-3)
        th = rng.uniform(0, np.pi / 2 + kappa)
        zc = r * np.exp(1j * th)
        z = acb(zc.real, zc.imag)
        q = cell_inner(rep, dp, z, up(z), arb(0))
        Yd, qn = BN.state_in(inner, zc, wf_)
        # inner values: P, Pt = Phi/z, C, Qp = Q'
        num = dict(P=qn["P"], Pt={m: qn["Phi"][m] / zc for m in ODD}, C=qn["C"], Qp=qn["dQ"])
        for key in ("P", "Pt", "C", "Qp"):
            for h, val in num[key].items():
                bx = q[key][h]
                err = max(abs(float(bx.real.mid()) - val.real) - float(bx.real.rad()),
                          abs(float(bx.imag.mid()) - val.imag) - float(bx.imag.rad()), 0.0)
                worst["inner"] = max(worst["inner"], err / (1 + abs(val)))
        # residual z r_Phi/eps: enclosure at eps -> 0 (t ball [0, 1e-12]) vs the Model at eps = 1e-14
        mp.mp.dps = 40
        eps = mp.mpf("1e-14")
        w = mp.mpf(wf_)
        zm = mp.mpc(zc)
        P = {m: mp.mpc(qn["P"][m]) for m in ODD}
        Phi = {m: mp.mpc(qn["Phi"][m]) for m in ODD}
        C = {k_: mp.mpc(qn["C"][k_]) for k_ in EVEN}
        vt = {k_: mp.mpc(0) for k_ in EVEN}
        _, dPe, _, _ = BS.Model(w, eps, mp).f_eps(zm, P, Phi, C, vt)
        _, dP0, _, _ = BS.Model(w, 0, mp).f_0(zm, P, Phi, C, vt)
        o.t = acb(arb(0, 1e-12), arb(0, 1e-12))
        rpb = BF.res_phi(o, z, acb(arb(5e-15, 5e-15)), q["P"], q["Pt"], q["C"], q["Qp"])
        for m in ODD:
            val = complex((dPe[m] - dP0[m]) * zm / eps)
            bx = rpb[m]
            err = max(abs(float(bx.real.mid()) - val.real) - float(bx.real.rad()),
                      abs(float(bx.imag.mid()) - val.imag) - float(bx.imag.rad()), 0.0)
            worst["res"] = max(worst["res"], err / (1 + abs(val)))
        # Jacobian blocks (complex enclosures, b5_formulas.jac_blocks) vs central differences of b5_system's G0
        npm = BN.NP(wf_)
        h = 1e-5

        def cd(cj):
            Yp, Ym = dict(Yd), dict(Yd)
            Yp[cj] = Yd[cj] + h
            Ym[cj] = Yd[cj] - h
            fp, fm = npm.rhs0(zc, Yp), npm.rhs0(zc, Ym)
            return {m: (fp[("F", m)] - fm[("F", m)]) / (2 * h) * zc for m in ODD}
        Phi_b = o.sc(z, q["Pt"])
        for blk, comps, kw in (("P", ODD, "dP"), ("F", ODD, "dPhi"), ("C", EVEN, "dC")):
            for j in comps:
                Jb = BF.jac_blocks(o, z, q["P"], Phi_b, q["Pt"], q["Ct"], q["C"], **{kw: unit(comps, j)})[blk]
                fd = cd((blk, j))
                for m in ODD:
                    bx, val = Jb[m], fd[m]
                    err = max(abs(float(bx.real.mid()) - val.real) - float(bx.real.rad()),
                              abs(float(bx.imag.mid()) - val.imag) - float(bx.imag.rad()), 0.0)
                    worst["jac"] = max(worst["jac"], err / (1 + abs(val)))
    # tolerances: double-precision inputs (inner 1e-12, res 1e-8), central differences (jac 1e-6)
    print("  raw excess beyond the enclosures (relative): " + ", ".join(f"{k_} {v:.2e}" for k_, v in worst.items()))
    worst["inner"] = 0.0 if worst["inner"] < 1e-12 else worst["inner"]
    worst["res"] = 0.0 if worst["res"] < 1e-8 else worst["res"]
    worst["jac"] = 0.0 if worst["jac"] < 1e-6 else worst["jac"]
    return worst


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        wv = point_checks()
        ok = all(v == 0.0 for v in wv.values())
        print("point checks (rigorous enclosure must contain the independent value; excess 0 = contained): "
              + ", ".join(f"{k_} {v:.3g}" for k_, v in wv.items()) + f"  -> {'PASS' if ok else 'FAIL'}")
        if MUT:
            print(f"MUTATION {MUT}: the check must FAIL: {'caught' if not ok else 'NOT caught'}")
            sys.exit(0 if not ok else 3)
        sys.exit(0 if ok else 1)
    kap = float(OPTS.get("kappa", "0.3"))
    Zs = [float(x) for x in OPTS.get("Z", "0.008").split(",")]
    box = tuple(OPTS.get("box", "2e-6,1e-6").split(","))
    outs = []
    k1 = float(OPTS.get("kappa1", "0.6"))
    for Z in Zs:
        # by default the run assumes exactly the n0 that the domain condition gives, ln n0 = kappa1/(Z cos kappa)
        n0e = float(OPTS["n0exp"]) if "n0exp" in OPTS else k1 / (Z * math.cos(kap))
        outs.append(run(kappa=kap, Z=Z, rp=OPTS.get("rp", "0.14"), k=int(OPTS.get("k", "12")), box=box,
                        n0exp=n0e, gamma=float(OPTS.get("gamma", "0.3")), kappa1=k1))
    if "json" in OPTS:
        with open(OPTS["json"], "w") as fh:
            json.dump(outs, fh, indent=1, default=str)
    if MUT:
        bad = not all(o_["ok"] for o_ in outs)
        print(f"MUTATION {MUT}: the run must FAIL: {'caught' if bad else 'NOT caught'}")
        sys.exit(0 if bad else 3)
    sys.exit(0 if all(o_["ok"] for o_ in outs) else 1)
