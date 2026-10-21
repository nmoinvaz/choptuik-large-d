#!/usr/bin/env python3
"""Krawczyk / interval-Newton test for the M = 1 inner zero (step iv of proof/A_certify.md), its accuracy budget, and a
validation on mock enclosures.

The map. With x = (A, wf) real (wf = w/w_inf) and x^ the box centre,
    g(x) = kappa J_D(x),   J_D = D+_{+1}(xi1) - conj(D+_{-1}(xi1)),   kappa = xi1^(-A^2)/N at x = x^ (a constant),
    N = 2 pi i K'(s) f(xi1) 2F_1,
split into (Re g, Im g). kappa is a fixed non-zero constant (rigorous in arb, as in lin_check.py), so g has the same
zeros as the selection condition J_1 = 0 (R_1 = 0) and, at a zero at the centre, the same Jacobian as
pm_dryrun.G = R_1 xi1^(-A^2) (G = g N(x^) xi1^(-A^2) / (N(x) xi1^(-A^2) at x^), and J_D = 0 there). Its Jacobian
needs only dD+_{+-1}(xi1)/dx, i.e. the variational equations carried through the validated integration from the C^1
start data of fixed_point.start_data; no derivative of N or of xi1^(-A^2) is needed.

The test. Y = mid(J(X))^-1 (any real matrix), X = x^ + [-h, h]^2,
    K(X) = x^ - Y g(x^) + (I - Y J(X)) (X - x^).
If K(X) lies in the interior of X, g has exactly one zero in X and it is simple (Krawczyk 1969; Moore). Needed:
an enclosure of g(x^) (a point evaluation) and an enclosure J(X) of the Jacobian over the whole box.

    python3 krawczyk.py [--xi1=0.3]   accuracy budget from the double-precision dry run, and the mock validation
                                      (default xi1 = 0.2, the radius of lateral_np.py and pm_dryrun.py)
"""
import os
import sys
import time
import numpy as np
from flint import arb, acb, arb_mat, ctx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_stokes"))

XI1 = "0.2"
XHAT = ("2.5421480475", "0.7769940136")


def krawczyk(xhat, g_hat, JX, h):
    """xhat: 2 arb (exact centre), g_hat: 2 arb (enclosure of g(xhat)), JX: 2x2 list of arb (enclosure over X),
    h: 2 arb half-widths. Returns (ok, K as 2 arb, margins h - |K - xhat|)."""
    Jm = np.array([[float(JX[i][j].mid()) for j in range(2)] for i in range(2)])
    Yf = np.linalg.inv(Jm)
    Y = [[arb(float(Yf[i][j])) for j in range(2)] for i in range(2)]        # exact binary floats: any matrix is allowed
    dX = [arb(0, h[0].upper()), arb(0, h[1].upper())]
    K = []
    for i in range(2):
        t = xhat[i] - (Y[i][0] * g_hat[0] + Y[i][1] * g_hat[1])
        for j in range(2):
            Mij = (arb(1) if i == j else arb(0)) - (Y[i][0] * JX[0][j] + Y[i][1] * JX[1][j])
            t = t + Mij * dX[j]
        K.append(t)
    ok = all((abs(K[i] - xhat[i]) < h[i]) for i in range(2))
    margins = [float((h[i] - abs(K[i] - xhat[i]).upper()).lower()) for i in range(2)]
    return bool(ok), K, margins


def normalization(xhat, xi1=XI1, prec=256):
    """N(x^) = 2 pi i K'(s) f(xi1) 2F_1 and xi1^(-A^2), rigorous (arb), s = i w"""
    ctx.prec = prec
    I = acb(0, 1)
    A, wf = arb(xhat[0]), arb(xhat[1])
    W0 = 2 * arb.pi() / arb("1.64389375")
    w = W0 * wf
    s = I * w
    a, b = (1 + s) / 2, (2 + s) / 2
    Kp = acb(2) ** (-s - acb(0.5)) / (a.gamma() * b.gamma())
    x1 = arb(xi1)
    X = 1 / (2 * x1 * x1)
    a2, b2 = 1 - a, 1 - b
    f = acb(x1) ** (-2 * s) * (-acb(X)).exp() * acb(X) ** a2 * acb(X).hypgeom_u(a2, 1 + a2 - b2)
    F1 = -I * A / 2
    N = 2 * arb.pi() * I * Kp * f * 2 * F1
    return N, acb(x1) ** (-(A * A))


def g_from_states(Dm1, Dp1, dDm1, dDp1, N, xs):
    """g(x^) and J(X) from enclosures of D+_{-1}(xi1), D+_{+1}(xi1) at the centre (acb) and of their derivatives
    dD/dA, dD/dwf over the box X (lists of 2 acb), for real parameters: J_D = D+1 - conj(D-1), d conj(f)/dx = conj(df/dx).
    xs = xi1^(-A^2) and N at the centre: kappa = xs/N is a constant, see the docstring."""
    c = xs / N
    JD = Dp1 - Dm1.conjugate()
    gv = JD * c
    dJ = [(dDp1[k] - dDm1[k].conjugate()) * c for k in range(2)]
    g_hat = [gv.real, gv.imag]
    JX = [[dJ[0].real, dJ[1].real], [dJ[0].imag, dJ[1].imag]]
    return g_hat, JX


def budget(verbose=True):
    """double-precision dry run (NOT rigorous): Jacobian at the centre and at the corners of boxes, and the enclosure
    widths of g(x^) and J(X) that the Krawczyk test tolerates"""
    import pm_dryrun as PM
    t0 = time.time()
    x0 = np.array([float(XHAT[0]), float(XHAT[1])])

    def jac(p, hh=1e-6):
        g0 = PM.G(*p)
        return np.array([(PM.G(p[0] + hh, p[1]) - g0) / hh, (PM.G(p[0], p[1] + hh) - g0) / hh]).T, g0
    J0, g0 = jac(x0)
    sv = np.linalg.svd(J0, compute_uv=False)
    corners = [np.array([sa, sb]) for sa in (-1, 1) for sb in (-1, 1)]
    hb = 1e-3
    Jc = [jac(x0 + hb * c)[0] for c in corners]
    var = np.max([np.abs(J - J0) for J in Jc], axis=0)          # entrywise variation over the h = 1e-3 box
    hess = var / hb                                              # rough second-derivative scale
    Y = np.linalg.inv(J0)
    aY = np.abs(Y)
    rows = []
    for h in (1e-3, 1e-5):
        varh = hess * h
        # Krawczyk: |Y| (|g^| + delta_g) + |Y| (varh + delta_J) h 1 < h, componentwise (2-vectors)
        base = aY @ np.abs(g0) + (aY @ varh) @ np.array([h, h])
        slack = h - base
        rowsum = aY.sum(axis=1)
        dg_max = np.min(slack / rowsum)                          # delta_J = 0
        dJ_max = np.min(slack / (h * aY.sum(axis=1) * 2))        # delta_g = 0, delta_J the same in all 4 entries
        rows.append((h, varh.max(), dg_max, dJ_max))
    if verbose:
        print(f"[dry run, double precision, NOT rigorous; {time.time() - t0:.0f} s]")
        print(f"  centre x^ = ({XHAT[0]}, {XHAT[1]}); |g(x^)| = {np.abs(g0).max():.1e} (double-precision floor)")
        print(f"  J0 = dg/d(A, wf) = {np.round(J0, 3).tolist()}, singular values {sv[0]:.1f}, {sv[1]:.1f}")
        print(f"  variation of J over the h = 1e-3 box (entrywise max |J(corner) - J0|): {np.round(var, 4).tolist()}")
        print("  tolerated enclosure radii (one at a time; split between them in practice):")
        for h, vh, dg, dJ in rows:
            print(f"    h = {h:.0e}: J varies by <= {vh:.2e} over X; g(x^) radius <= {dg:.2e} (with exact J(X)); "
                  f"J(X) extra radius <= {dJ:.2e} per entry (with exact g(x^))")
    return J0, g0, Jc, hess, rows


def mock(J0, g0, hess, verbose=True):
    """validate the test logic on mock enclosures: the dry-run values with inflated radii"""
    ctx.prec = 128
    xhat = [arb(XHAT[0]), arb(XHAT[1])]
    res = []
    for h, fg, fJ, shift in ((1e-3, 0.25, 0.25, 0.0), (1e-3, 2.5, 0.25, 0.0), (1e-5, 0.25, 0.25, 0.0),
                             (1e-5, 0.25, 0.25, 3.0)):
        J = J0
        Y = np.linalg.inv(J)
        aY = np.abs(Y)
        varh = hess * h
        base = aY @ np.abs(g0) + (aY @ varh) @ np.array([h, h])
        slack = h - base
        dg = fg * np.min(slack / aY.sum(axis=1))
        dJ = fJ * np.min(slack / (h * aY.sum(axis=1) * 2))
        # optionally move the centre off the zero by 'shift' box half-widths along A: the box must then fail
        gc = g0 + J @ np.array([shift * h, 0.0])
        g_hat = [arb(float(gc[i]), dg + abs(gc[i]) * 1e-15) for i in range(2)]
        JX = [[arb(float(J[i][j]), float(varh[i][j]) + dJ) for j in range(2)] for i in range(2)]
        ok, K, marg = krawczyk(xhat, g_hat, JX, [arb(h), arb(h)])
        res.append((h, fg, fJ, shift, ok))
        if verbose:
            print(f"  mock h = {h:.0e}, g radius {fg:.2f} x budget, J radius {fJ:.2f} x budget, centre off by {shift} h: "
                  f"K(X) in int X: {ok} (margins {marg[0]:.2e}, {marg[1]:.2e})")
    return res


if __name__ == "__main__":
    ctx.prec = 256
    XI1 = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--xi1=")), XI1)
    import pm_dryrun
    pm_dryrun.XI1 = float(XI1)
    print(f"matching radius xi1 = {XI1}")
    N, xs = normalization(XHAT, XI1)
    print(f"normalization (rigorous): N(x^) = {N.str(12)}, xi1^(-A^2) = {xs.str(12)}, |xi1^(-A^2)/N| = {abs(xs / N).str(8)}")
    J0, g0, Jc, hess, rows = budget()
    c = float(abs(xs / N).mid())
    print("  in terms of the raw jump J_D = D+_{+1}(xi1) - conj(D+_{-1}(xi1)) (what the validated integration must enclose):")
    for h, vh, dg, dJ in rows:
        print(f"    h = {h:.0e}: |J_D(x^)| radius <= {dg / c:.2e}; dJ_D/d(A, wf) over X: extra radius <= {dJ / c:.2e} "
              f"(on top of the true variation {vh / c:.2e})")
    print("mock validation of the test (enclosures = dry-run values with inflated radii):")
    mock(J0, g0, hess)
