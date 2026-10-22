#!/usr/bin/env python3
"""Hirschmann-Eardley kappa-model (kappa = 1: axion-dilaton) in D = n + 2 dimensions, spherical symmetry.

Model (HE97, gr-qc/9511052, eq. (4), written in D dimensions):
    S = int d^D x sqrt(-g) [ R - 2 |grad F|^2 / (1 - kappa |F|^2)^2 ],
    R_ab = h (F_a Fbar_b + Fbar_a F_b),   box F = -2 kappa Fbar (grad F . grad F) / (1 - kappa |F|^2),   h = (1 - kappa|F|^2)^-2.
Metric ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2.  Self-similar coordinates tau = -ln(-t), x = r/(-t), and
    F = (-t)^{i omega} psi(tau, x) = e^{-i omega tau} psi,   P = (a/alpha)(psi_tau - i omega psi + x psi_x),
    b = alpha/a,   u = a^2 - 1,   w = (x/X)^2  (X = x_h, the self-similarity horizon b(X) = X),   v = u/w.
Equations (derived and checked in css_equations.py):
    psi_tau = b P + i omega psi - 2 w psi_w
    P_tau   = (i omega - 1) P - 2 w P_w + X^-2 [4 w b psi_ww + 2 (n+1) b psi_w + 4 w b_w psi_w]
              + b (2 kappa psibar / rho) (4 w psi_w^2 / X^2 - P^2),        rho = 1 - kappa |psi|^2
    Hamiltonian:  v + w v_w = (1 + w v) [ (X^2/n) h (|P|^2 + 4 w |psi_w|^2 / X^2) - (n-1) v / 2 ]
    slicing:      b_w = (n-1) b v / 2
    momentum:     v_tau = (8 (1 + w v)/n) b h Re(P psibar_w) - 2 (v + w v_w)
The CSS background has psi = f(x), P = -p with p = (i omega f - x f')/b (HE95a's p), tau-independent.

Discretization: Chebyshev-Gauss-Lobatto collocation in w on [0, 1] (w = 0 centre, w = 1 horizon), every equation
collocated at every node (regularity at both regular-singular ends comes from the polynomial basis), except that
the slicing row at w = 0 is replaced by the gauge b(0) = 1 (alpha = 1 on the axis).  Jacobians by complex step
(all residuals are written in real arithmetic, so they are analytic in the real unknowns).
"""
import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# the PDE terms, as plain arithmetic (works for numpy arrays and for sympy symbols)
# ---------------------------------------------------------------------------------------------------------------
def pde_terms(n, kap, om, X, w, sR, sI, pR, pI, v, b, sRw, sIw, pRw, pIw, vw, bw, sRww, sIww):
    """Return (psi_tau R, psi_tau I, P_tau R, P_tau I, Ham, slicing, v_tau from momentum)."""
    rho = 1 - kap * (sR * sR + sI * sI)
    h = 1 / (rho * rho)
    # psi_tau = b P + i om psi - 2 w psi_w
    dsR = b * pR - om * sI - 2 * w * sRw
    dsI = b * pI + om * sR - 2 * w * sIw
    # Laplacian part
    LR = (4 * w * b * sRww + 2 * (n + 1) * b * sRw + 4 * w * bw * sRw) / X ** 2
    LI = (4 * w * b * sIww + 2 * (n + 1) * b * sIw + 4 * w * bw * sIw) / X ** 2
    # Q = 4 w psi_w^2 / X^2 - P^2  (complex square)
    QR = 4 * w * (sRw * sRw - sIw * sIw) / X ** 2 - (pR * pR - pI * pI)
    QI = 4 * w * (2 * sRw * sIw) / X ** 2 - 2 * pR * pI
    # 2 kappa psibar / rho * Q
    cR = 2 * kap / rho
    NR = cR * (sR * QR + sI * QI)
    NI = cR * (sR * QI - sI * QR)
    dpR = -pR - om * pI - 2 * w * pRw + LR + b * NR
    dpI = om * pR - pI - 2 * w * pIw + LI + b * NI
    E = (pR * pR + pI * pI) + 4 * w * (sRw * sRw + sIw * sIw) / X ** 2
    ham = v + w * vw - (1 + w * v) * ((X ** 2 / n) * h * E - (n - 1) * v / 2)
    sli = bw - (n - 1) * b * v / 2
    vtau = (8 * (1 + w * v) / n) * b * h * (pR * sRw + pI * sIw) - 2 * (v + w * vw)
    return dsR, dsI, dpR, dpI, ham, sli, vtau


# ---------------------------------------------------------------------------------------------------------------
# Chebyshev collocation
# ---------------------------------------------------------------------------------------------------------------
def cheb(N):
    """Gauss-Lobatto nodes on [0,1] (w_0 = 0 ... w_N = 1) and first-derivative matrix."""
    j = np.arange(N + 1)
    xc = np.cos(np.pi * j / N)                     # 1 ... -1
    c = np.ones(N + 1); c[0] = c[-1] = 2.0
    c = c * (-1.0) ** j
    dX = xc[:, None] - xc[None, :]
    Dm = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    Dm = Dm - np.diag(Dm.sum(axis=1))
    # map w = (1 - xc)/2  -> d/dw = -2 d/dxc ; reorder so that w increases
    w = (1 - xc) / 2
    Dw = -2 * Dm
    return w, Dw


class Grid:
    """Chebyshev-Gauss-Lobatto nodes in zeta on [0,1], mapped to w = (e^{A zeta} - 1)/(e^A - 1) (A = 0: w = zeta).
    The map clusters nodes at the centre, where the solution has a complex singularity close to w = 0 (where
    kappa f fbar = 1 in the complexified x-plane); it is analytic, so it keeps spectral convergence."""
    def __init__(self, N, A=0.0):
        self.N = N
        self.A = A
        self.z, Dz = cheb(N)
        if A == 0:
            self.w = self.z.copy(); wp = np.ones_like(self.z)
        else:
            self.w = np.expm1(A * self.z) / np.expm1(A)
            wp = A * np.exp(A * self.z) / np.expm1(A)
            self.w[0] = 0.0; self.w[-1] = 1.0
        self.D1 = Dz / wp[:, None]
        self.D2 = self.D1 @ self.D1
        self.M = N + 1

    def zeta_of_w(self, w):
        w = np.asarray(w, dtype=float)
        return w if self.A == 0 else np.log1p(w * np.expm1(self.A)) / self.A

    def interp_matrix(self, wnew):
        """Barycentric interpolation matrix from the nodes to points wnew."""
        N = self.N
        wts = (-1.0) ** np.arange(N + 1); wts[0] *= 0.5; wts[-1] *= 0.5
        wnew = np.atleast_1d(self.zeta_of_w(wnew))
        d = wnew[:, None] - self.z[None, :]
        exact = np.isclose(d, 0, atol=1e-15)
        d[exact] = 1.0
        T = wts[None, :] / d
        T = T / T.sum(axis=1, keepdims=True)
        rows = np.where(exact.any(axis=1))[0]
        for r_ in rows:
            T[r_] = 0.0
            T[r_, np.where(exact[r_])[0][0]] = 1.0
        return T


NF = 6   # fields: sR, sI, pR, pI, v, b


def split(Z, M):
    return [Z[k * M:(k + 1) * M] for k in range(NF)]


def residual_fields(g, Z, n, kap, om, X):
    M = g.M
    sR, sI, pR, pI, v, b = split(Z, M)
    D1, D2, w = g.D1, g.D2, g.w
    out = pde_terms(n, kap, om, X, w, sR, sI, pR, pI, v, b,
                    D1 @ sR, D1 @ sI, D1 @ pR, D1 @ pI, D1 @ v, D1 @ b, D2 @ sR, D2 @ sI)
    return out


def bg_residual(g, Y, n, kap):
    """Background (steady) residual.  Y = [fields (6M), omega, X]."""
    M = g.M
    Z = Y[:NF * M]; om = Y[NF * M]; X = Y[NF * M + 1]
    dsR, dsI, dpR, dpI, ham, sli, _ = residual_fields(g, Z, n, kap, om, X)
    sR, sI, pR, pI, v, b = split(Z, M)
    sli = sli.copy() if hasattr(sli, 'copy') else sli
    sli[0] = b[0] - 1.0
    extra = np.array([sI[0], b[-1] - X], dtype=Y.dtype)
    return np.concatenate([dsR, dsI, dpR, dpI, ham, sli, extra])


def cs_jacobian(fun, Y, h=1e-30):
    Y = np.asarray(Y, dtype=float)
    f0 = fun(Y.astype(complex)).real
    J = np.empty((f0.size, Y.size))
    Yc = Y.astype(complex)
    for k in range(Y.size):
        Yc[k] += 1j * h
        J[:, k] = fun(Yc).imag / h
        Yc[k] -= 1j * h
    return f0, J


def newton_bg(g, Y0, n, kap, tol=1e-12, maxit=40, verbose=False, soft=1e-8):
    """Damped Newton.  Stops at |F| < tol, or when |F| < soft and an iteration no longer halves |F| (roundoff)."""
    Y = np.array(Y0, dtype=float)
    prev = np.inf
    for it in range(maxit):
        F, J = cs_jacobian(lambda y: bg_residual(g, y, n, kap), Y)
        nr = np.max(np.abs(F))
        if verbose:
            print("  newton it %d  |F| = %.3e  omega = %.12f  X = %.12f" % (it, nr, Y[NF * g.M], Y[NF * g.M + 1]))
        if nr < tol or (nr < soft and nr > 0.5 * prev):
            return Y, nr, it
        prev = nr
        dY = np.linalg.solve(J, -F)
        # damping
        lam = 1.0
        for _ in range(20):
            Yt = Y + lam * dY
            nt = np.max(np.abs(bg_residual(g, Yt, n, kap)))
            if np.isfinite(nt) and nt < nr * (1 - 1e-4 * lam) or lam < 1e-3:
                break
            lam *= 0.5
        Y = Yt
    F = bg_residual(g, Y, n, kap)
    return Y, np.max(np.abs(F)), maxit


# ---------------------------------------------------------------------------------------------------------------
# perturbations
# ---------------------------------------------------------------------------------------------------------------
def pert_residual(g, Z, n, kap, om, X):
    """Linear-perturbation operator rows: [psi_tau (2), P_tau (2), Ham, slicing with delta b(0) = 0]."""
    M = g.M
    dsR, dsI, dpR, dpI, ham, sli, _ = residual_fields(g, Z, n, kap, om, X)
    b = split(Z, M)[5]
    sli = sli.copy()
    sli[0] = b[0]
    return np.concatenate([dsR, dsI, dpR, dpI, ham, sli])


def pert_matrices(g, Ybg, n, kap):
    """Return (J, B) with  lambda B dZ = J dZ  (B = identity on the 4 evolved fields, 0 on constraints)."""
    M = g.M
    Z0 = Ybg[:NF * M]; om = Ybg[NF * M]; X = Ybg[NF * M + 1]
    _, J = cs_jacobian(lambda z: pert_residual(g, z, n, kap, om, X), Z0)
    B = np.zeros_like(J)
    idx = np.arange(4 * M)
    B[idx, idx] = 1.0
    return J, B


def momentum_check(g, Ybg, n, kap, lam, dZ):
    """Residual of the linearized momentum constraint  lambda dv = delta(v_tau)  for an eigenvector dZ (complex)."""
    M = g.M
    Z0 = Ybg[:NF * M].astype(complex); om = Ybg[NF * M]; X = Ybg[NF * M + 1]
    def vt(z):
        return residual_fields(g, z, n, kap, om, X)[6]
    eps = 1e-30
    # directional derivative by complex step, separately for Re and Im parts of dZ
    dR = vt(Z0 + 1j * eps * dZ.real).imag / eps
    dI = vt(Z0 + 1j * eps * dZ.imag).imag / eps
    dvt = dR + 1j * dI
    dv = split(dZ, M)[4]
    scale = max(np.max(np.abs(dvt)), np.max(np.abs(lam * dv)), 1e-300)
    return np.max(np.abs(lam * dv - dvt)) / scale


def spectrum(g, Ybg, n, kap):
    import scipy.linalg as sl
    J, B = pert_matrices(g, Ybg, n, kap)
    lam, V = sl.eig(J, B)
    ok = np.isfinite(lam) & (np.abs(lam) < 1e6)
    return lam[ok], V[:, ok]


# ---------------------------------------------------------------------------------------------------------------
# helpers to build / move solutions
# ---------------------------------------------------------------------------------------------------------------
def unpack(g, Y):
    M = g.M
    Z = Y[:NF * M]
    d = dict(zip(["sR", "sI", "pR", "pI", "v", "b"], split(Z, M)))
    d["omega"] = Y[NF * M]; d["X"] = Y[NF * M + 1]
    return d


def regrid(g_old, Y, g_new):
    T = g_old.interp_matrix(g_new.w)
    M = g_old.M
    parts = [T @ Y[k * M:(k + 1) * M] for k in range(NF)]
    return np.concatenate(parts + [Y[NF * M:]])
