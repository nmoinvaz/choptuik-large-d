# Copied unchanged from solver/factored/lsnewton.py on wip/large-d-factored (Track B of #1088, commit 77e86e79).
"""Newton with the weak directions split off (a Lyapunov-Schmidt reduction), in double-double (#1088).

The DSS Jacobian has a few near-null directions (the high harmonics' selection conditions, with Delta in their span;
LARGE_D_WALL), along which the residual is strongly curved: plain Newton's basin there is about sigma_min^2 / curvature,
and a damped step crawls (fsolve.py, fseries_sdd.py). Here the scaled Jacobian's SVD splits the unknowns into strong
coordinates a and weak ones b (singular values below tau):
  - inner: for fixed b, the strong rows U_s^T F(y(a, b)) = 0 are solved for a by the chord iteration with Sigma_s;
  - outer: the reduced residual g(b) = U_w^T F(y(a(b), b)) (r components) is solved by Newton with a difference
    Jacobian (each column an inner solve).
The reduced problem's curvature is what is left after the inner solve absorbs the strong rows' quadratic terms.
Status: a numerical method, standard in bifurcation analysis (Lyapunov-Schmidt; e.g. Kielhoefer, Bifurcation Theory,
Springer 2012, ch. I.2); no claim beyond the logged iterations.
"""
import numpy as np
import mpmath as mp

import ddnp as d
from ddnp import DD


def scaled_svd(J, dps=40):
    """SVD of diag(1/r) J diag(1/c) in mpmath (r: row norms, c: column norms of the row-scaled matrix)."""
    Jf = J.h + J.l
    r = np.linalg.norm(Jf, axis=1)
    Jr = Jf / r[:, None]
    c = np.linalg.norm(Jr, axis=0)
    with mp.workdps(dps):
        A = mp.matrix(Jf.shape[0], Jf.shape[1])
        for i in range(Jf.shape[0]):
            for j in range(Jf.shape[1]):
                A[i, j] = (mp.mpf(float(J.h[i, j])) + mp.mpf(float(J.l[i, j]))) / (mp.mpf(r[i]) * mp.mpf(c[j]))
        U, S, V = mp.svd_r(A)
        Uf = np.array([[float(U[i, j]) for j in range(U.cols)] for i in range(U.rows)])
        Sf = np.array([float(S[i]) for i in range(len(S))])
        Vf = np.array([[float(V[i, j]) for j in range(V.cols)] for i in range(V.rows)])
    order = np.argsort(-Sf)
    return r, c, Uf[:, order], Sf[order], Vf[order, :]


def ls_solve(residual, y0, J, tau=1e-8, outer=8, inner=6, log=print, tol=1e-30, hb=None):
    r, c, U, S, Vt = scaled_svd(J)
    weak = np.where(S < tau * S[0])[0]
    strong = np.where(S >= tau * S[0])[0]
    Us, Uw = U[:, strong], U[:, weak]
    Vs, Vw = Vt[strong, :].T, Vt[weak, :].T
    Ss = S[strong]
    log("LS: %d strong, %d weak directions; weak singular values %s" % (len(strong), len(weak),
                                                                     " ".join("%.2e" % v for v in S[weak])))

    # J = diag(r) Js diag(c), so the scaled coordinates are s = c * (y - y0): y = y0 + (Vs a + Vw b) / c, a, b as DD
    def y_dd(a, b):
        s = d.matvec(DD(Vs), a) + d.matvec(DD(Vw), b)
        return y0 + s / DD(c)

    def strong_norm(F):
        Fs = d.matvec(DD(Us.T), F / DD(r))
        return Fs, float(np.linalg.norm(Fs.h)) if np.all(np.isfinite(Fs.h)) else np.inf

    def inner_solve(a, b):
        try:
            return _inner_solve(a, b)
        except (ValueError, FloatingPointError, OverflowError):
            Fbad = d.zeros((len(r),))
            Fbad.h[:] = np.inf
            return a, Fbad

    def _inner_solve(a, b):
        with np.errstate(all="ignore"):
            F = residual(y_dd(a, b))
        Fs, ns = strong_norm(F)
        for it in range(inner):
            da = Fs / DD(Ss)
            lam = 1.0
            for _ in range(20):
                at = a - da * lam
                try:
                    with np.errstate(all="ignore"):
                        Ft = residual(y_dd(at, b))
                    Fst, nst = strong_norm(Ft)
                except (ValueError, FloatingPointError, OverflowError):
                    nst = np.inf
                if nst < ns or nst == 0.0:
                    break
                lam *= 0.5
            if not nst < ns:
                break
            a, F, Fs, ns = at, Ft, Fst, nst
            if float(np.max(np.abs((da * lam).h))) < 1e-31 * max(1.0, float(np.max(np.abs(a.h)))):
                break
        return a, F

    nw = len(weak)
    a = d.zeros((len(strong),))
    b = d.zeros((nw,))
    a, F = inner_solve(a, b)
    g = d.matvec(DD(Uw.T), F / DD(r))
    hist = []
    for it in range(outer):
        nF = float(np.max(np.abs(F.h)))
        ng = float(np.max(np.abs(g.h))) if nw else 0.0
        log("LS outer %d: max|F| %.3e |g| %.3e |b| %.3e" % (it, nF, ng, float(np.max(np.abs(b.h))) if nw else 0.0))
        hist.append((nF, ng))
        if nF < tol or nw == 0:
            break
        # difference Jacobian of g(b)
        h = hb if hb is not None else max(1e-14, 1e-3 * ng / max(S[weak].min(), 1e-300))
        G = np.zeros((nw, nw))
        db = np.full(nw, np.nan)
        for j in range(nw):
            bj = b.copy()
            bj[j] = bj[j] + h
            aj, Fj = inner_solve(a.copy(), bj)
            gj = d.matvec(DD(Uw.T), Fj / DD(r))
            G[:, j] = ((gj - g) * (1.0 / h)).f()
        if np.all(np.isfinite(G)):
            db = -np.linalg.solve(G, g.f())
        # backtracking on |g|
        lam = 1.0
        accepted = False
        if np.all(np.isfinite(db)):
            for _ in range(14):
                bt = b + DD(lam * db)
                at, Ft = inner_solve(a.copy(), bt)
                gt = d.matvec(DD(Uw.T), Ft / DD(r))
                ngt = float(np.max(np.abs(gt.h)))
                if np.isfinite(ngt) and np.all(np.isfinite(Ft.h)) and ngt < (1 - 1e-4 * lam) * ng:
                    accepted = True
                    break
                lam *= 0.5
        if accepted:
            a, b, F, g = at, bt, Ft, gt
        else:
            log("    no step along the weak directions reduces |g|; kept the point")
        log("    db %.3e lambda %.3g, reduced Jacobian sv %s" % (float(np.max(np.abs(db))), lam,
                                                                " ".join("%.2e" % v for v in np.linalg.svd(G, compute_uv=False))))
    return y_dd(a, b), F, hist, S
