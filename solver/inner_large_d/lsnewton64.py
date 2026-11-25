"""Newton with the weak directions split off (a Lyapunov-Schmidt reduction), in float64 (#1090).

The float64 counterpart of solver/factored/lsnewton.py (Track B, double-double). The scaled Jacobian
diag(1/r) J diag(1/c) = U S V^T splits the unknowns into strong coordinates a (singular values >= tau S_max) and weak
ones b: y = y0 + (V_s a + V_w b)/c.
  - inner: for fixed b, the strong rows U_s^T F/r = 0 are solved for a by the chord iteration with S_s;
  - outer: the reduced residual g(b) = U_w^T F/r (r components) is solved by Newton with a difference Jacobian
    (each column an inner solve), with backtracking on |g|.
Plain Newton's basin in the weak directions is about sigma_min^2 / curvature (Track B, #1088); the reduction leaves
the curvature that the inner solve does not absorb. A numerical method (Kielhoefer, Bifurcation Theory, Springer 2012,
ch. I.2); no claim beyond the logged iterations.
"""
import numpy as np


def split(J, tau):
    r = np.linalg.norm(J, axis=1)
    Jr = J / r[:, None]
    c = np.linalg.norm(Jr, axis=0)
    U, S, Vt = np.linalg.svd(Jr / c[None, :])
    weak = S < tau * S[0]
    return r, c, U[:, ~weak], S[~weak], Vt[~weak].T, U[:, weak], Vt[weak].T, S[weak]


def ls_solve(residual, y0, J, tau=1e-6, outer=10, inner=8, tol=1e-15, log=print, hb=1e-7):
    r, c, Us, Ss, Vs, Uw, Vw, Sw = split(J, tau)
    log("    LS: %d strong, %d weak; weak sv %s" % (len(Ss), len(Sw), " ".join("%.1e" % v for v in Sw)))

    def y_of(a, b):
        return y0 + (Vs @ a + Vw @ b) / c

    def inner_solve(b, a):
        for _ in range(inner):
            F = residual(y_of(a, b)) / r
            gs = Us.T @ F
            a = a - gs / Ss
            if np.max(np.abs(gs)) < 0.1 * tol:
                break
        F = residual(y_of(a, b)) / r
        return a, F

    a = np.zeros(len(Ss))
    b = np.zeros(len(Sw))
    a, F = inner_solve(b, a)
    g = Uw.T @ F
    best = (np.max(np.abs(F * r)), y_of(a, b))
    for it in range(outer):
        if len(b) == 0:
            break
        Jg = np.zeros((len(b), len(b)))
        for j in range(len(b)):
            bj = b.copy()
            bj[j] += hb
            aj, Fj = inner_solve(bj, a.copy())
            Jg[:, j] = (Uw.T @ Fj - g) / hb
        if not (np.all(np.isfinite(Jg)) and np.all(np.isfinite(g))):
            log("    LS outer %d: non-finite reduced Jacobian; kept the best point" % it)
            break
        db = np.linalg.lstsq(Jg, -g, rcond=None)[0]
        lam = 1.0
        for _ in range(10):
            bn = b + lam * db
            an, Fn = inner_solve(bn, a.copy())
            gn = Uw.T @ Fn
            if np.max(np.abs(gn)) < np.max(np.abs(g)):
                break
            lam *= 0.5
        if not np.all(np.isfinite(Fn)):
            log("    LS outer %d: no finite step along the weak directions; kept the best point" % it)
            break
        a, b, F, g = an, bn, Fn, gn
        mf = float(np.max(np.abs(F * r)))
        log("    LS outer %d: max|F| %.2e |g| %.2e |db| %.1e lambda %g" % (it, mf, np.max(np.abs(g)), np.max(np.abs(db)), lam))
        if mf < best[0]:
            best = (mf, y_of(a, b))
        if mf < tol:
            break
    return best[1], best[0]
