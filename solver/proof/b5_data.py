#!/usr/bin/env python3
"""Lemma B5(1) (proof/B5_inner.md, section 5): the leading values of the data delta of hypothesis (H8), from the
first-order inner correction Z^(1) (numerical, mpmath; NOT rigorous).

On the inner arc |z| = r_n the difference W = Y_out - Z_in is, to leading order, eps Z^(1)(r_n), where Z^(1) is the
O(1/n) correction of the formal solution with the data of notes/FIRST_ORDER.md section 2: profile correction K_1 = 0
(Z_in is built with the actual profile), gauge c^(1)(0) = -g_1 = -[F^2] (c(x = 1) = 1), vt^(1) slaved. Its first
coefficients follow from the order-1/z balance of the Phi-row of the W-system (b5_formulas):
    phi_1/2 + J_PhiP(0) p_0 + J_PhiC(0) c_0 + R(0) = 0,   p_0 = K_1 = 0,  c_0 = -[F^2],
where R(0) = lim z r_Phi/eps and J(0) = lim z dG0/dY, so Phi^(1) = phi_1 z + ..., P^(1) = (phi_1/2) z + ...; and
vt^(1)(0) = -E0(0) = -2[F^2]. Hence, as n -> infinity,
    delta_P -> 0 (|W_P| ~ eps |phi_1|/2 r_n),  delta_Phi -> max|phi_1|,  delta_C -> max|[F^2]|,  delta_vt -> max|2[F^2]|.
Check: phi_1 is compared with a direct evaluation of the slaved first-order Phi^(1) from the residual of the exact model
along a ray (Phi^(1)(z)/z at small z by the variation-of-constants integral would need the lateral sum; here the
order-by-order identity is checked by recomputing R(0) and J(0) at two radii, 1e-7 and 1e-9, which must agree).

    python3 b5_data.py [--mutate=gauge]      gauge c_0 = 0 instead of -[F^2]: delta_Phi changes (reported)
"""
import sys

import mpmath as mp
import numpy as np

import b5_formulas as BF
import b5_numeric as BN

ODD, EVEN = BF.ODD, BF.EVEN
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)


def leading(zr):
    mp.mp.dps = 40
    w = mp.mpf(BN.omega())
    o = BF.mp_ops(w)
    inner = BN.Inner()
    zc = complex(zr)
    Y, q = BN.state_in(inner, zc, float(w))
    z = mp.mpc(zc)
    P = {m: mp.mpc(q["P"][m]) for m in ODD}
    Pt = {m: mp.mpc(q["Phi"][m]) / z for m in ODD}
    C = {k: mp.mpc(q["C"][k]) for k in EVEN}
    Qp = {m: mp.mpc(q["dQ"][m]) for m in ODD}
    Ct = {k: (C[k] - (1 if k == 0 else 0)) / z for k in EVEN}
    Phi = o.sc(z, Pt)
    o.t = mp.mpc(0)
    R0 = BF.res_phi(o, z, mp.mpf(0), P, Pt, C, Qp)
    F2 = {k: v / 4 for k, v in BF.proj(P, P, EVEN, mp.mpc(0)).items()}       # [F^2] = [P P]/4 at z -> 0 (P = 2F)
    c0 = {k: -F2[k] for k in EVEN}
    JC = BF.jac_blocks(o, z, P, Phi, Pt, Ct, C, dC=c0)["C"]
    phi1 = {m: -2 * (JC[m] + R0[m]) for m in ODD}
    return phi1, F2, R0


def jet_cross_check():
    """the O(1/n) inner series of notes/FIRST_ORDER.md (solver/inner_pert/first_order_jet.py: a separate numpy jet
    recursion of the O(1/n) inner equations, which shares no code with b5_formulas) for the M = 1 model at the same
    point, K_1 = 0, gauge -F^2: its y-coefficient of xi^3 is P^(1)_1 = phi_1/2"""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for sub in ("inner_pert", "inner_stokes", "inner_borel", ""):
        sys.path.insert(0, os.path.join(here, "..", sub))
    from first_order_jet import JetModel, Dir, F2h, W0
    A = float(BN.A_STAR)
    w = W0 * float(BN.WF_STAR)
    Fh = {1: A / 2j, -1: -A / 2j}
    g1 = F2h(Fh, 1)
    gauge = {k: (0 if MUT == "gauge" else -v) for k, v in g1.items()}
    jm = JetModel(1, w, Fh, [Dir("n", q1=True)], [0, 1], {1: ({}, gauge)})
    jm.series_at(1.0, 4, stop=False)
    L = jm.L
    return {m: complex(jm.ser["D"][1][1][m + L]) for m in ODD}


if __name__ == "__main__":
    p_a, F2, R0 = leading(1e-7)
    p_b, _, _ = leading(1e-9)
    agree = float(max(abs(p_a[m] - p_b[m]) / abs(p_a[m]) for m in ODD))
    a1 = jet_cross_check()
    xdev = float(max(abs(complex(p_b[m]) / 2 - a1[m]) / abs(a1[m]) for m in ODD))
    print("cross-check with first_order_jet.py (independent O(1/n) recursion): P^(1)_1 = "
          + ", ".join(f"mu {m:+d}: {a1[m]:.6g}" for m in ODD) + f"; relative deviation from phi_1/2: {xdev:.2e}")
    dPhi = max(abs(p_a[m]) for m in ODD)
    dC = max(abs(v) for v in F2.values())
    dv = 2 * dC
    print(f"R(0) = lim z r_Phi/eps: " + ", ".join(f"mu {m:+d}: {mp.nstr(abs(R0[m]), 6)}" for m in ODD))
    print(f"phi_1 (Phi^(1) = phi_1 z + ...): " + ", ".join(f"mu {m:+d}: {mp.nstr(p_a[m], 6)}" for m in ODD)
          + f"   (z = 1e-7 vs 1e-9: relative difference {mp.nstr(agree, 3)})")
    print(f"leading data: delta_P -> 0 (|W_P| ~ eps |phi_1|/2 r_n), delta_Phi -> {mp.nstr(dPhi, 6)}, "
          f"delta_C -> max|[F^2]| = {mp.nstr(dC, 6)}, delta_vt -> 2 max|[F^2]| = {mp.nstr(dv, 6)}"
          + ("   [MUTATION gauge: c_0 = 0]" if MUT else ""))
    ok = agree < 1e-5 and xdev < 1e-6
    print(f"checks: z-independence {agree:.1e} (< 1e-5), agreement with first_order_jet {xdev:.1e} (< 1e-6): "
          f"{'PASS' if ok else 'FAIL'}")
    if MUT:
        # the mutation is applied to the jet side only (the b5 side keeps the true gauge) -> must disagree
        sys.exit(0 if not ok else 3)
    sys.exit(0 if ok else 1)
