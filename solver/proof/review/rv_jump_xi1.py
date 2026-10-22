#!/usr/bin/env python3
"""Review check R6 (agent RV, double precision, NOT rigorous): is the certified zero of J_D(xi1 = 0.3) a zero of the
whole Stokes jump?

The certificate proves that g = kappa J_D, J_D = D+_{+1}(xi1) - conj(D+_{-1}(xi1)), has one simple zero in the box,
for xi1 = 0.3. The selection rule is that the jump y+ - y- vanishes, i.e. every component at every xi1. Here, with
inner_stokes/lateral_np.py's solver (rtol 1e-13), each complex component c of the jump (D_{+1}, S_{+1}, C_{+2}, C_0)
at xi1 = 0.2, 0.25, 0.3, 0.35 gives its own Newton estimate of where it vanishes,
    x_c = x^ - (dc/dx)^-1 c(x^)     (real 2x2, finite differences),
and these are compared with the certified enclosure A in [2.5421480475256, 2.5421480477075],
w/w_inf in [0.776994013544111, 0.776994013675337]. If they all fall at the certified zero (to the solver's accuracy),
the jump vanishes there as a whole, numerically; the proof of that is the missing lemma of the review (item 6a).

Second part: path dependence (review item 6b). y+ on the real axis is defined by continuation along the ray to xi1,
then the arc |xi| = xi1. That equals the continuation of the Borel sum along the real axis only if y+ is analytic in
the region between the paths. Compared here: y+(0.3) along ray-arc(0.3) and along ray-arc(0.2)-real axis, and
y+(0.35) along ray-arc(0.3)-real axis and along ray-arc(0.35), with max |y| on the arcs.

    python3 rv_jump_xi1.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "inner_stokes"))
from lateral_np import NP, series_data, Fharm, W0  # noqa: E402

XH = np.array([2.5421480475, 0.7769940136])
KA = (2.5421480475256, 2.5421480477075)
KW = (0.776994013544111, 0.776994013675337)


def jumps(A, wf, r1, r0=0.1):
    w = W0 * wf
    phi0 = 3 * np.pi / 8
    Y0, jt, err = series_data(Fharm(A), w, 1, r0 * np.exp(1j * phi0))
    nm = NP(1, w)
    Y = nm.path(Y0, [("ray", phi0, r0, r1), ("arc", r1, phi0, 0.0)], rtol=1e-13, atol=1e-16)
    D, S, C = Y[0:2], Y[2:4], Y[4:7]           # D_-1, D_+1; S_-1, S_+1; C_-2, C_0, C_+2
    return {"D+1": D[1] - np.conj(D[0]), "S+1": S[1] - np.conj(S[0]), "C+2": C[2] - np.conj(C[0]),
            "C0": C[1] - np.conj(C[1])}


if __name__ == "__main__":
    hd = 1e-6
    print(f"certified: A in [{KA[0]}, {KA[1]}], w/w_inf in [{KW[0]}, {KW[1]}]")
    for r1 in (0.2, 0.25, 0.3, 0.35):
        j0 = jumps(*XH, r1)
        ja = jumps(XH[0] + hd, XH[1], r1)
        jw = jumps(XH[0], XH[1] + hd, r1)
        parts = []
        xm = np.array([(KA[0] + KA[1]) / 2, (KW[0] + KW[1]) / 2])
        for k in j0:
            J = np.array([[(ja[k] - j0[k]).real, (jw[k] - j0[k]).real], [(ja[k] - j0[k]).imag, (jw[k] - j0[k]).imag]]) / hd
            pred = np.array([j0[k].real, j0[k].imag]) + J @ (xm - XH)        # linear prediction at the certified zero
            scale = np.abs(J).max() * 1e-10                                    # what a 1e-10 move of x changes
            if k == "C0":                                                      # C0 - conj(C0) = 2i Im C0: one real equation
                parts.append(f"{k}: |jump| {abs(j0[k]):.1e}, at the certified zero {abs(pred[1]):.1e} "
                             f"(a 1e-10 move: {scale:.1e})")
                continue
            x = XH - np.linalg.solve(J, np.array([j0[k].real, j0[k].imag]))
            parts.append(f"{k}: |jump| {abs(j0[k]):.1e}, zero at x^ + ({x[0] - XH[0]:+.2e}, {x[1] - XH[1]:+.2e}), "
                         f"at the certified zero {np.abs(pred).max():.1e} (a 1e-10 move: {scale:.1e})")
        print(f"xi1 = {r1}:\n    " + "\n    ".join(parts), flush=True)
    print(f"(the certified zero is at x^ + ({KA[0] - XH[0]:+.2e}..{KA[1] - XH[0]:+.2e}, "
          f"{KW[0] - XH[1]:+.2e}..{KW[1] - XH[1]:+.2e}))")
    A, wf = XH
    w = W0 * wf
    phi0 = 3 * np.pi / 8
    Y0, _, _ = series_data(Fharm(A), w, 1, 0.1 * np.exp(1j * phi0))
    nm = NP(1, w)

    def run(segs):
        return nm.path(Y0, segs, rtol=1e-13, atol=1e-16)
    pa = run([("ray", phi0, 0.1, 0.3), ("arc", 0.3, phi0, 0.0)])
    pb = run([("ray", phi0, 0.1, 0.2), ("arc", 0.2, phi0, 0.0), ("real", 0.2, 0.3)])
    pc = run([("ray", phi0, 0.1, 0.3), ("arc", 0.3, phi0, 0.0), ("real", 0.3, 0.35)])
    pd = run([("ray", phi0, 0.1, 0.35), ("arc", 0.35, phi0, 0.0)])
    print(f"path dependence: y+(0.3) along ray-arc(0.3) vs ray-arc(0.2)-real: max difference {np.abs(pa - pb).max():.1e}")
    print(f"                 y+(0.35) along ray-arc(0.3)-real vs ray-arc(0.35): max difference {np.abs(pc - pd).max():.1e} "
          f"(|y| {np.abs(pc).max():.2f} vs {np.abs(pd).max():.2f})")
    for r1 in (0.3, 0.32, 0.34):
        Yc, ph, m = run([("ray", phi0, 0.1, r1)]), phi0, 0.0
        for k in range(1, 25):
            ph2 = phi0 * (1 - k / 24)
            Yc = nm.path(Yc, [("arc", r1, ph, ph2)], rtol=1e-12, atol=1e-15)
            ph, m = ph2, max(m, np.abs(Yc).max())
        print(f"                 max |y+| on the arc |xi| = {r1}: {m:.1f}")
