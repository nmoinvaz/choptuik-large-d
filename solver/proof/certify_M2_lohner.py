#!/usr/bin/env python3
"""Second-order (C^2) validated integration for the M-harmonic certificate (proof/A_M.md section 10).

What a run of LohnerM2 proves (given its start sets). Let x range over a parameter box X = prod [-w_k, w_k] (real,
centred at 0) and y(t; x) be the solution of the system certify_M_sys.SysM along the path, with y(t0; x) in the start
sets. At every step, for every x in X:
  (a) y(t; 0) lies in the centre set  c + Q q;
  (b) V(t; 0) = dy/dx (t; 0) lies in the derivative sets  cV + QV qV  (m columns);
  (c) W(t; x) = d^2 y / dx dx (t; x) lies in the second-derivative sets  cW + QW qW  (one column per pair j <= k).
All three are Lohner sets in real coordinates (validated_int.Lohner's representation). From (a)-(c) and Taylor's
theorem with integral remainder, for every x in X
    y(t; x) in Y(X) := c + Q q + V0 x + 1/2 W(X)[x, x],        V(t; x) in V(X) := V0 + W(X)[x, .],
(V0, W(X) the boxes of the sets in (b), (c)), which are the sets the step's enclosures are evaluated on.

One step from t0 to t0 + h (the flow map phi of the step): with
  J0  an enclosure of D phi over the centre set (validated_int's Taylor-plus-remainder J, on the centre box),
  JX  the same over Y(X),
  H   an enclosure of D^2 phi(y)[a, b] for every y in Y(X) and every column pair a in V(X)_j, b in V(X)_k,
the chain rule gives  y(0) -> phi(y(0)),  V0 -> D phi(y(0)) V0 in J0 V0,  and
    W(x)_jk -> D phi(y(x)) W(x)_jk + D^2 phi(y(x))[V(x)_j, V(x)_k]   in   JX W(X)_jk + H_jk,
and each is put back into Lohner form (centre, QR frame, box) as validated_int.Lohner does for the C^1 columns.

H is the solution at tau = h of the second variational equation
    Psi' = A(tau) Psi + B(tau)[Phi(tau) a, Phi(tau) b],   Psi(0) = 0,
with A = xi' F_y(y(tau)) and B = xi' F_yy(y(tau)) (F the right-hand side, F_yy its second derivative, secondvar
below): its Taylor coefficients Psi_0..Psi_NJ from the trajectory series over Y(X) (the coefficient recursion
Psi_{l+1} = (sum_i A_{l-i} Psi_i + beta_l) / (l + 1)), plus the remainder h^(NJ+1) Psi^(NJ+1)(s)/(NJ+1)! with s in
[0, h], enclosed by the same recursion started at the interval time T = t0 + [0, h] from the a-priori enclosures
(Yt of the trajectory, Vt of Phi, Pt of Psi; apriori_Psi below): by the chain rule
Psi(s + sigma) = Phi(s + sigma, s) Psi(s) + D^2 phi_{s+sigma <- s}(y(s))[Phi(s) a, Phi(s) b], so its coefficient NJ+1
in sigma is the recursion's output from (Psi(s), Phi(s) a, Phi(s) b) in (Pt, Vt a, Vt b). The integral form of the
remainder (the mean of a function with values in a box lies in the box) makes this valid for complex values, as for
J in certify_M_lohner.LohnerM.

    python3 certify_M2_lohner.py selftest [--mutate=lin2]
        secondvar against central differences of SysM.lin (M = 1, 3, 5) and its symmetry; one step's H series against
        central differences of the step Jacobian series; one LohnerM2 step's W against central differences of two
        LohnerM C^1 point steps. --mutate=lin2 (one sign in secondvar flipped) must FAIL.
"""
import json
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_mat, arb_mat, acb_series, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validated_int as VI  # noqa: E402
import lemma6_common  # noqa: E402
import certify_M_sys as CS  # noqa: E402
import certify_M_lohner as CL  # noqa: E402

I = CS.I
pr = CS.pr
MUTATE = set()          # "lin2": one sign of secondvar flipped (controls)


# ---------------------------------------------------------------- the second derivative of the right-hand side

def firstvar(s, cache, U, xi, ixi, zero):
    """the first variation of SysM.rhs at the cached point along U (nd + 1 entries, U[-1] = dw), as the tuple of
    direction parts and first-variation intermediates that secondvar needs (the same formulas as SysM.lin)"""
    D, S, C, w, E, cE, dD = cache
    uD, uS, uC, uw = s.unpack(U)
    ODD, EVEN = s.ODD, s.EVEN
    ixi2 = ixi * ixi
    p1, p2 = pr(uC, S, ODD, zero), pr(C, uS, ODD, zero)
    ddD = {m: (p1[m] + p2[m] - I * m * (uw * xi * D[m] + w * xi * uD[m])) * ixi2 for m in ODD}
    dE = pr(D, uD, EVEN, zero)
    q1, q2 = pr(uC, E, EVEN, zero), pr(C, dE, EVEN, zero)
    dcE = {v: q1[v] + q2[v] for v in EVEN}
    return (uD, uS, uC, uw, ddD, dE, dcE)


def secondvar(s, cache, fa, fb, xi, ixi, zero):
    """F_yy[a, b]: the second derivative of SysM.rhs at the cached point, bilinear in the directions a, b given by
    their firstvar tuples. rhs: dD = ([C S] - i mu w xi D)/xi^2, E = [D D]/2, cE = [C E],
    r2 = [C xi dD] + [cE D] - [C D], dS = (r2 - i mu w xi S)/xi^2, dC = cE/xi; differentiate each product twice
    (product rule; d_a d_b of a product of two factors is the sum of the two cross terms plus the factor times the
    second variation). The w-component of the result is 0 (w is constant)."""
    D, S, C, w, E, cE, dD = cache
    aD, aS, aC, aw, addD, adE, adcE = fa
    bD, bS, bC, bw, bddD, bdE, bdcE = fb
    ODD, EVEN = s.ODD, s.EVEN
    ixi2 = ixi * ixi
    c1, c2 = pr(aC, bS, ODD, zero), pr(bC, aS, ODD, zero)
    d2dD = {m: (c1[m] + c2[m] - I * m * xi * (aw * bD[m] + bw * aD[m])) * ixi2 for m in ODD}
    d2E = pr(aD, bD, EVEN, zero)
    e1, e2, e3 = pr(aC, bdE, EVEN, zero), pr(bC, adE, EVEN, zero), pr(C, d2E, EVEN, zero)
    d2cE = {v: e1[v] + e2[v] + e3[v] for v in EVEN}
    xa = {m: xi * addD[m] for m in ODD}
    xb = {m: xi * bddD[m] for m in ODD}
    x2 = {m: xi * d2dD[m] for m in ODD}
    t = [pr(aC, xb, ODD, zero), pr(bC, xa, ODD, zero), pr(C, x2, ODD, zero), pr(d2cE, D, ODD, zero),
         pr(adcE, bD, ODD, zero), pr(bdcE, aD, ODD, zero), pr(aC, bD, ODD, zero), pr(bC, aD, ODD, zero)]
    sg = -1 if "lin2" in MUTATE else 1
    d2r2 = {m: t[0][m] + t[1][m] + t[2][m] + t[3][m] + sg * (t[4][m] + t[5][m]) - t[6][m] - t[7][m] for m in ODD}
    d2dS = {m: (d2r2[m] - I * m * xi * (aw * bS[m] + bw * aS[m])) * ixi2 for m in ODD}
    d2dC = {v: d2cE[v] * ixi for v in EVEN}
    return s.pack(d2dD, d2dS, d2dC, zero)


# ---------------------------------------------------------------- series recursions

def taylor_VA(s, ctxv, N):
    """certify_M_lohner.taylor_V, also returning the coefficient matrices A_0..A_N of A(tau) = xi' F_y(Y(tau)):
    V_{k+1} = (1/(k+1)) sum_{j <= k} A_{k-j} V_j, V_0 = I. Returns (V list of N + 1, A list of N + 1)."""
    cache, xs, ixs, dxs, zero = ctxv
    L = N + 1
    ctx.cap = L
    n, nd = s.n, s.nd
    Acols = []
    for j in range(n):
        F = s.lin_unit(cache, j, xs, ixs, zero)
        Acols.append([(F[i] * dxs).coeffs() for i in range(nd)])
    Ak = []
    for k in range(N + 1):
        Mk = acb_mat(n, n)
        for j in range(n):
            col = Acols[j]
            for i in range(nd):
                cs = col[i]
                if k < len(cs):
                    Mk[i, j] = cs[k]
        Ak.append(Mk)
    V = [acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])]
    for k in range(N):
        acc = Ak[k] * V[0]
        for j in range(1, k + 1):
            acc = acc + Ak[k - j] * V[j]
        V.append(acc * acb(arb(1) / (k + 1)))
    return V, Ak


def pair_sources(s, ctxv, Vk, acols, pairs, K):
    """the coefficients beta_0..beta_K (n x P acb_mat each) of beta_jk(tau) = xi'(tau) F_yy(Y(tau))[v_j(tau), v_k(tau)]
    with v_j(tau) = Phi(tau) a_j = sum_l Vk[l] a_j tau^l (acols: n x m acb_mat of the initial vectors a_j), on the
    series context ctxv (length >= K + 1) and the Phi coefficients Vk (>= K + 1 of them)"""
    cache, xs, ixs, dxs, zero = ctxv
    L = K + 1
    ctx.cap = L
    n, nd = s.n, s.nd
    m = acols.ncols()
    VA = [Vk[l] * acols for l in range(L)]
    fv = []
    for j in range(m):
        U = [acb_series([VA[l][i, j] for l in range(L)], prec=L) for i in range(n)]
        fv.append(firstvar(s, cache, U, xs, ixs, zero))
    P = len(pairs)
    beta = [acb_mat(n, P) for _ in range(L)]
    for p, (j, k) in enumerate(pairs):
        F = secondvar(s, cache, fv[j], fv[k], xs, ixs, zero)
        for i in range(nd):
            cs = (F[i] * dxs).coeffs()
            for l in range(min(L, len(cs))):
                beta[l][i, p] = cs[l]
    return beta


def psi_coeffs(Ak, beta, Psi0, K):
    """Psi_0..Psi_K of Psi' = A(tau) Psi + beta(tau), Psi(0) = Psi0 (n x P): Psi_{l+1} = (sum_{i <= l} A_{l-i} Psi_i
    + beta_l) / (l + 1); needs A_0..A_{K-1} and beta_0..beta_{K-1}"""
    Psi = [Psi0]
    for l in range(K):
        acc = beta[l] + Ak[l] * Psi[0]
        for i in range(1, l + 1):
            acc = acc + Ak[l - i] * Psi[i]
        Psi.append(acc * acb(arb(1) / (l + 1)))
    return Psi


def horner(Ms, hh, K):
    """sum_{l <= K} Ms[l] h^l"""
    acc = Ms[K]
    for l in range(K - 1, -1, -1):
        acc = acc * hh + Ms[l]
    return acc


def _infl_col(Mt, fac):
    """inflate every entry of the n x P acb_mat by fac in its radius plus, per column, 1e-2 x the column's largest
    radius (a per-pair floor: the pairs differ by many orders of magnitude). Only an initial guess: containment is
    verified by apriori_Psi."""
    n, P = Mt.nrows(), Mt.ncols()
    out = acb_mat(n, P)
    for p in range(P):
        fl = 1e-60
        for i in range(n):
            fl = max(fl, float(Mt[i, p].real.rad()), float(Mt[i, p].imag.rad()))
        fl = arb(1e-2 * fl + 1e-60)
        for i in range(n):
            z = Mt[i, p]
            out[i, p] = acb(arb(z.real.mid(), z.real.rad() * fac + fl), arb(z.imag.mid(), z.imag.rad() * fac + fl))
    return out


def apriori_Psi(s, leg, T, Yt, Vt, acols, pairs, h):
    """Pt (n x P) with [0, h] (Fy Pt + B) contained in Pt, where Fy = xi' F_y(Yt) and B_jk = xi' F_yy(Yt)[Vt a_j, Vt a_k]
    at the interval time T: then Psi_jk(tau) is in Pt for all tau in [0, h] (the Picard operator of the linear
    equation Psi' = A Psi + beta, Psi(0) = 0, maps functions with values in Pt into itself). None if not found."""
    n, nd = s.n, s.nd
    xi = leg.xi(T)
    dxi = leg.dxi(xi)
    ixi = 1 / xi
    zero = acb(0)
    _, cache = s.rhs(Yt, xi, ixi, zero)
    Fy = acb_mat(n, n)
    for j in range(n):
        e = [acb(1) if i == j else acb(0) for i in range(n)]
        col = s.lin(cache, e, xi, ixi, zero)
        for i in range(nd):
            Fy[i, j] = col[i] * dxi
    Va = Vt * acols
    m = acols.ncols()
    fv = [firstvar(s, cache, [Va[i, j] for i in range(n)], xi, ixi, zero) for j in range(m)]
    P = len(pairs)
    B = acb_mat(n, P)
    for p, (j, k) in enumerate(pairs):
        F = secondvar(s, cache, fv[j], fv[k], xi, ixi, zero)
        for i in range(nd):
            B[i, p] = F[i] * dxi
    Hb = acb(arb(h / 2, h / 2))
    Pt = _infl_col(Hb * B, 2.0)
    for it in range(30):
        Z = Hb * (Fy * Pt + B)
        if all(Pt[i, p].contains(Z[i, p]) for i in range(n) for p in range(P)):
            return Z
        U = acb_mat([[Pt[i, p].union(Z[i, p]) for p in range(P)] for i in range(n)])
        Pt = _infl_col(U, 1.5)
    return None


# ---------------------------------------------------------------- the C^2 Lohner set

def pair_list(m):
    return [(j, k) for j in range(m) for k in range(j, m)]


def pair_index(m):
    pi = {}
    for p, (j, k) in enumerate(pair_list(m)):
        pi[(j, k)] = pi[(k, j)] = p
    return pi


class LohnerM2(CL.LohnerM):
    """the C^2 set of the module docstring: centre set (c, Q, q), V0 = (cV, QV, qV) along the centre trajectory, and
    W = (cW, QW, qW) over the parameter box pdx (acb balls 0 +- w_k). self.dx stays None, so box() and the inherited
    progress code see the centre set only."""

    def __init__(self, sys_, c, Q, q, N, var, w2, pdx, NJ=None):
        super().__init__(sys_, c, Q, q, N, var)
        if isinstance(w2[1], acb_mat):
            cW, QW, qW = w2
            w2 = (cW, VI.realify(QW), VI.stack(qW))
        self.w2 = w2
        self.pdx = list(pdx)
        self.m = len(self.pdx)
        self.pairs = pair_list(self.m)
        self.pidx = pair_index(self.m)
        self.NJ = NJ

    def W_box(self):
        cW, QW, qW = self.w2
        return cW + VI.unstack(QW * qW)

    def VX_box(self, V0=None, W=None):
        """V(X) = V0 + W(X)[., x] over the box: n x m acb_mat"""
        V0 = self.var_box() if V0 is None else V0
        W = self.W_box() if W is None else W
        n, m = self.s.n, self.m
        out = acb_mat(n, m)
        for j in range(m):
            for i in range(n):
                acc = V0[i, j]
                for k in range(m):
                    acc = acc + W[i, self.pidx[(j, k)]] * self.pdx[k]
                out[i, j] = acc
        return out

    def YX_box(self, V0=None, W=None):
        """Y(X) = c + Q q + V0 x + 1/2 W(X)[x, x] over the box: list of n acb (products of the balls pdx_j pdx_k are
        formed as products, never as powers: python-flint 0.9's arb(0, r)**2 is NaN)"""
        V0 = self.var_box() if V0 is None else V0
        W = self.W_box() if W is None else W
        X0 = self.box()
        n, m = self.s.n, self.m
        out = []
        for i in range(n):
            acc = X0[i]
            for j in range(m):
                acc = acc + V0[i, j] * self.pdx[j]
            for (j, k) in self.pairs:
                t = W[i, self.pidx[(j, k)]] * (self.pdx[j] * self.pdx[k])
                acc = acc + (t if j != k else t / 2)
            out.append(acc)
        return out

    def _step(self, leg, t0, h):
        s, N, n, nd = self.s, self.N, self.s.n, self.s.nd
        NJ = self.NJ or N
        V0 = self.var_box()
        W = self.W_box()
        VX = self.VX_box(V0, W)
        X = self.YX_box(V0, W)
        X0 = self.box()
        Yt, T = lemma6_common.apriori(s, leg, t0, h, X)
        if Yt is None:
            return False
        Vt = lemma6_common.apriori_V(s, leg, T, Yt, h)
        if Vt is None:
            return False
        Pt = apriori_Psi(s, leg, T, Yt, Vt, VX, self.pairs, h)
        if Pt is None:
            return False
        hh = acb(h)
        hN1 = hh ** (N + 1)
        # the centre trajectory (as LohnerM)
        Yc, _ = CL.taylor_Y(s, leg, acb(t0), self.c, N)
        Yr, ctr = CL.taylor_Y(s, leg, T, Yt, N + 1)
        phic = [VI.evalpoly(Yc[i], hh, N) + hN1 * VI.coef(Yr[i], N + 1) for i in range(nd)] + self.c[nd:]
        # the remainders of J and H at the interval time T from the a-priori sets
        ctrJ = CL.truncate_ctx(ctr, NJ + 2) if NJ < N else ctr
        VR, AR = taylor_VA(s, ctrJ, NJ + 1)
        remJ = hh ** (NJ + 1) * (VR[NJ + 1] * Vt)
        # J0 over the centre set, JX over Y(X)
        _, ctx0 = CL.taylor_Y(s, leg, acb(t0), X0, NJ)
        Vk0, _ = taylor_VA(s, ctx0, NJ)
        J0 = horner(Vk0, hh, NJ) + remJ
        _, ctxX = CL.taylor_Y(s, leg, acb(t0), X, NJ)
        VkX, AkX = taylor_VA(s, ctxX, NJ)
        JX = horner(VkX, hh, NJ) + remJ
        # H over Y(X) and the column pairs of V(X)
        P = len(self.pairs)
        beta = pair_sources(s, ctxX, VkX, VX, self.pairs, NJ - 1)
        Psi = psi_coeffs(AkX, beta, acb_mat(n, P), NJ)
        betaR = pair_sources(s, ctrJ, VR, Vt * VX, self.pairs, NJ)
        PsiR = psi_coeffs(AR, betaR, Pt, NJ + 1)
        H = horner(Psi, hh, NJ) + hh ** (NJ + 1) * PsiR[NJ + 1]
        # Lohner updates (real coordinates), as validated_int.Lohner.step
        J0R, JXR = VI.realify(J0), VI.realify(JX)
        cnew = [acb(v.mid()) for v in phic]
        e = VI.stack(acb_mat([[phic[i] - cnew[i]] for i in range(n)]))
        A = J0R * self.Q
        Qn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(A), [float(abs(x).upper()) for x in self.q]))
        Qi = Qn.inv()
        qn = (Qi * A) * arb_mat([[x] for x in self.q]) + Qi * e
        cV, QV, qV = self.var
        m = cV.ncols()
        JC = J0 * cV
        cVn = acb_mat([[acb(JC[i, j].mid()) for j in range(m)] for i in range(n)])
        AV = J0R * QV
        QVn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(AV), [max(float(abs(qV[i, j]).upper()) for j in range(m))
                                                         for i in range(2 * n)]))
        QVi = QVn.inv()
        varn = (cVn, QVn, (QVi * AV) * qV + QVi * VI.stack(JC - cVn))
        cW, QW, qW = self.w2
        JW = JX * cW + H
        cWn = acb_mat([[acb(JW[i, p].mid()) for p in range(P)] for i in range(n)])
        AW = JXR * QW
        QWn = VI.from_np_r(VI.qr_pivot(VI.to_np_r(AW), [max(float(abs(qW[i, p]).upper()) for p in range(P))
                                                         for i in range(2 * n)]))
        QWi = QWn.inv()
        w2n = (cWn, QWn, (QWi * AW) * qW + QWi * VI.stack(JW - cWn))
        self.c, self.Q, self.q, self.var, self.w2 = cnew, Qn, [qn[i, 0] for i in range(2 * n)], varn, w2n
        return True

    def radii(self):
        """(state radius over the box, V0 radius, W radius, V(X) radius): diagnostics for the progress lines"""
        V0, W = self.var_box(), self.W_box()
        nd = self.s.nd
        X = self.YX_box(V0, W)
        rs = max(VI.rad_of(x) for x in X[:nd])
        rv = max(VI.rad_of(V0[i, j]) for i in range(nd) for j in range(V0.ncols()))
        rw = max(VI.rad_of(W[i, p]) for i in range(nd) for p in range(W.ncols()))
        VX = self.VX_box(V0, W)
        rvx = max(VI.rad_of(VX[i, j]) for i in range(nd) for j in range(VX.ncols()))
        return rs, rv, rw, rvx

    def run(self, leg, h0, verbose=False, log=None, t_start=0.0, steps_start=0, ckpt=None):
        """certify_M_lohner.LohnerM.run (t: 0 -> 1 on the grid 2^-30 Z) with the C^2 radii in the progress lines"""
        q = 2.0 ** -30
        hmax = getattr(self, "hmax", 1.0)
        t, h, steps = t_start, min(h0, hmax), steps_start
        last = time.time()
        while t < 1.0:
            if ckpt is not None and time.time() - last > CL.CKPT_SEC:
                ckpt(t, h, steps)
                last = time.time()
            if getattr(self, "max_steps", None) is not None and getattr(self, "total_steps", 0) >= self.max_steps:
                raise CL.PilotStop(f"{leg.kind} t = {t:.6f}")
            hs = min(max(round(h / q), 1) * q, 1.0 - t)
            if not self.step(leg, t, hs):
                h = hs / 2
                if h < 1e-9:
                    raise RuntimeError(f"step size underflow at t = {t} on the {leg.kind} leg")
                continue
            t = t + hs
            steps += 1
            self.total_steps = getattr(self, "total_steps", 0) + 1
            h = min(hs * 1.25, hmax)
            if (verbose or log) and steps % 25 == 0:
                rs, rv, rw, rvx = self.radii()
                msg = (f"   {leg.kind} t {t:.6f} h {hs:.2e} steps {steps}  state radius (box) {rs:.2e}  V0 radius "
                       f"{rv:.2e}  W radius {rw:.2e}  V(X) radius {rvx:.2e}  [{time.time() - CL.T0:.0f} s]")
                if verbose:
                    print(msg, flush=True)
                if log:
                    with open(log, "a") as fh:
                        fh.write(msg + "\n")
        return steps


# ---------------------------------------------------------------- start set, runs, serialization

def start_set2(sys_, vals_c, w_c, cols, wcol, W0cols, pdx, N, NJ):
    """the C^2 Lohner set at xi0: vals_c (nd acb) the state at x = 0, w_c = w at x = 0; cols (m lists of nd acb) and
    wcol (m arb) the derivative columns at x = 0; W0cols (P lists of nd acb) the second-derivative columns over the
    box (their w-components are exactly 0: w = W0 (wf^ + x_1) is affine in x); pdx the box (acb balls)"""
    n, nd = sys_.n, sys_.nd
    c = [acb(v.mid()) for v in vals_c] + [acb(w_c.mid())]
    q = [vals_c[i] - c[i] for i in range(nd)] + [acb(w_c) - c[nd]]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    m = len(cols)
    V0 = acb_mat(n, m)
    for j in range(m):
        for i in range(nd):
            V0[i, j] = cols[j][i]
        V0[nd, j] = acb(wcol[j])
    cV = acb_mat([[acb(V0[i, j].mid()) for j in range(m)] for i in range(n)])
    P = len(W0cols)
    W0 = acb_mat(n, P)
    for p in range(P):
        for i in range(nd):
            W0[i, p] = W0cols[p][i]
    cW = acb_mat([[acb(W0[i, p].mid()) for p in range(P)] for i in range(n)])
    lo = LohnerM2(sys_, c, Q, q, N, (cV, Q, V0 - cV), (cW, Q, W0 - cW), pdx, NJ)
    return lo


def dump2(lo):
    """the final C^2 sets exactly (centre, V0, W; Q matrices as float hex) and the box"""
    out = CL.dump_lohner(lo)
    cW, QW, qW = lo.w2
    assert CL.check_exact_float(QW)
    out["cW"] = [[CL.s_acb(cW[i, p]) for p in range(cW.ncols())] for i in range(cW.nrows())]
    out["QW"] = CL.s_mat_float(QW)
    out["qW"] = [[CL.s_arb(qW[i, p]) for p in range(qW.ncols())] for i in range(qW.nrows())]
    out["pdx"] = [CL.s_acb(z) for z in lo.pdx]
    return out


def load2(d):
    """inverse of dump2: (c, Q, q, var, w2, pdx)"""
    c, Q, q, var = CL.load_lohner(d)
    cW = acb_mat([[CL.p_acb(v) for v in row] for row in d["cW"]])
    QW = CL.p_mat_float(d["QW"])
    qW = arb_mat([[arb(x) for x in row] for row in d["qW"]])
    return c, Q, q, var, (cW, QW, qW), [CL.p_acb(z) for z in d["pdx"]]


def save_ckpt2(path_file, key, lo, k, t, h, steps_done, steps_leg):
    d = {"key": key, "leg": k, "t": float(t).hex(), "h": float(h).hex(), "steps_done": steps_done,
         "steps_leg": steps_leg, "total_steps": getattr(lo, "total_steps", 0), "sets": dump2(lo),
         "prec": ctx.prec, "time": time.time()}
    tmp = path_file + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh)
    os.replace(tmp, path_file)


def integrate2(lo, path, h0=1e-3, verbose=False, log=None, ckpt=None, key=None):
    """certify_M_lohner.integrate for LohnerM2: the legs of path, with checkpoints (the exact C^2 sets) about every
    CKPT_SEC seconds and at the end of each leg, resumed from when the key matches"""
    steps = []
    k0, t0_, h_, s0 = 0, 0.0, h0, 0
    if ckpt and os.path.exists(ckpt):
        with open(ckpt) as fh:
            d = json.load(fh)
        if d["key"] != key:
            raise RuntimeError(f"checkpoint {ckpt} belongs to another run (key mismatch); remove it to start afresh")
        lo.c, lo.Q, lo.q, lo.var, lo.w2, lo.pdx = load2(d["sets"])
        lo.total_steps = d["total_steps"]
        k0, t0_, h_, s0, steps = d["leg"], float.fromhex(d["t"]), float.fromhex(d["h"]), d["steps_leg"], d["steps_done"]
        if log:
            with open(log, "a") as fh:
                fh.write(f"resumed from {ckpt}: leg {k0}, t {t0_:.6f}, {lo.total_steps} steps done\n")
    for k, leg in enumerate(CS.legs_for(path)):
        if k < k0:
            continue
        t1 = time.time()
        cb = None
        if ckpt:
            def cb(t, h, nst, k=k):
                save_ckpt2(ckpt, key, lo, k, t, h, steps, nst)
        ts, ss, hh = (t0_, s0, h_) if k == k0 else (0.0, 0, h0)
        steps.append(lo.run(leg, hh, verbose, log, t_start=ts, steps_start=ss, ckpt=cb))
        if log:
            rs, rv, rw, rvx = lo.radii()
            with open(log, "a") as fh:
                fh.write(f"leg {leg.kind} done: {steps[-1]} steps, {time.time() - t1:.0f} s; state radius (box) {rs:.2e}, "
                         f"V0 {rv:.2e}, W {rw:.2e}, V(X) {rvx:.2e}\n")
        if ckpt:
            save_ckpt2(ckpt, key, lo, k + 1, 0.0, h0, steps, 0)
    return steps


# ---------------------------------------------------------------- self-tests

def _rand_state(s, rng, scale=0.5):
    Y = [acb(complex(*(scale * rng.normal(size=2)))) for _ in range(s.nd)]
    return Y + [acb(3.8)]


def check_secondvar(verbose=True):
    """secondvar against the central difference of SysM.lin, (F_y(y + e b) a - F_y(y - e b) a) / (2 e), at 256 bits
    with e = 1e-30 (truncation error ~ e^2 |F_yyyy|), at a random complex state and directions with nonzero w
    components, M = 1, 3, 5; and its symmetry in (a, b). Returns the worst relative error."""
    rng = np.random.default_rng(11)
    ctx.prec = 256
    worst = 0.0
    for M in (1, 3, 5):
        s = CS.SysM(M)
        Y = _rand_state(s, rng)
        xi = acb(0.15, 0.07)
        ixi = 1 / xi
        zero = acb(0)
        a = [acb(complex(*rng.normal(size=2))) for _ in range(s.nd)] + [acb(0.3)]
        b = [acb(complex(*rng.normal(size=2))) for _ in range(s.nd)] + [acb(-0.7)]
        _, cache = s.rhs(Y, xi, ixi, zero)
        F2 = secondvar(s, cache, firstvar(s, cache, a, xi, ixi, zero), firstvar(s, cache, b, xi, ixi, zero), xi, ixi,
                       zero)
        F2s = secondvar(s, cache, firstvar(s, cache, b, xi, ixi, zero), firstvar(s, cache, a, xi, ixi, zero), xi, ixi,
                        zero)
        e = arb("1e-30")
        Yp = [Y[i] + e * b[i] for i in range(s.n)]
        Ym = [Y[i] - e * b[i] for i in range(s.n)]
        _, cp = s.rhs(Yp, xi, ixi, zero)
        _, cm = s.rhs(Ym, xi, ixi, zero)
        Lp, Lm = s.lin(cp, a, xi, ixi, zero), s.lin(cm, a, xi, ixi, zero)
        err = max(float(abs((Lp[i] - Lm[i]) / (2 * e) - F2[i]).upper()) / (1 + float(abs(F2[i]).upper()))
                  for i in range(s.nd))
        sym = max(float(abs(F2[i] - F2s[i]).upper()) for i in range(s.nd))
        scale = max(float(abs(F2[i]).upper()) for i in range(s.nd))
        worst = max(worst, err, sym)
        if verbose:
            print(f"M = {M}: secondvar vs central difference of lin: {err:.1e} (|F_yy| up to {scale:.1e}); "
                  f"symmetry {sym:.1e}")
    return worst


def check_H_series(verbose=True):
    """one step's second variation series against central differences of the first variation series: for a point
    state y, sum_l Psi_l h^l (pair_sources + psi_coeffs) must equal (J(y + e b) a - J(y - e b) a) / (2 e), J = sum_l V_l
    h^l (taylor_VA on the point series), to the truncation error; M = 3, ray leg, h = 2^-12. Returns the relative
    error."""
    rng = np.random.default_rng(5)
    ctx.prec = 256
    M = 3
    s = CS.SysM(M)
    phi0 = arb(5) / 16 * arb.pi()
    leg = CS.LegM("ray", "0.0741", "0.2", phi0)
    p = [arb("2.72647588563088"), arb("1.0156030948577"), arb("0.144419478470077"), arb("0.0110874258881167")]
    xi0, vals, _ = CS.start_series(p, M, "0.0741", phi0, 60, dirs=[])
    Y = [acb(v.mid()) for v in vals] + [acb((CS.W0() * p[1]).mid())]
    a = acb_mat([[complex(*(1e-2 * rng.normal(size=2)))] for _ in range(s.nd)] + [[0.4]])
    b = acb_mat([[complex(*(1e-2 * rng.normal(size=2)))] for _ in range(s.nd)] + [[0.0]])
    NJ, h = 12, 2.0 ** -12
    hh = acb(h)
    _, ctxv = CL.taylor_Y(s, leg, acb(0), Y, NJ)
    Vk, Ak = taylor_VA(s, ctxv, NJ)
    ab = acb_mat(s.n, 2)
    for i in range(s.n):
        ab[i, 0], ab[i, 1] = a[i, 0], b[i, 0]
    beta = pair_sources(s, ctxv, Vk, ab, [(0, 1)], NJ - 1)
    Psi = psi_coeffs(Ak, beta, acb_mat(s.n, 1), NJ)
    Hs = horner(Psi, hh, NJ)
    e = arb("1e-25")
    Js = []
    for sg in (1, -1):
        Yb = [Y[i] + sg * e * b[i, 0] for i in range(s.n)]
        _, cb = CL.taylor_Y(s, leg, acb(0), Yb, NJ)
        Vb, _ = taylor_VA(s, cb, NJ)
        Js.append(horner(Vb, hh, NJ) * a)
    fd = (Js[0] - Js[1]) * acb(1 / (2 * e))
    err = max(float(abs(fd[i, 0] - Hs[i, 0]).upper()) for i in range(s.nd)) / \
        max(float(abs(Hs[i, 0]).upper()) for i in range(s.nd))
    if verbose:
        print(f"M = 3: H series (second variation, one step h = 2^-12) vs central difference of the J series: "
              f"relative {err:.1e} (|H| up to {max(float(abs(Hs[i, 0]).upper()) for i in range(s.nd)):.1e})")
    return err


def check_W_step(verbose=True):
    """one full LohnerM2 step from a point centre (box of half-width 1e-12 in two parameter directions) against two
    LohnerM C^1 point steps at y +- e V0 b: W_new(.)_{jk} must contain (V_new(+e e_k) - V_new(-e e_k))_j / (2 e) up to
    the central difference's error; M = 1, ray leg. Returns (contained, relative mismatch of midpoints)."""
    rng = np.random.default_rng(3)
    ctx.prec = 256
    s = CS.SysM(1)
    phi0 = 3 * arb.pi() / 8
    leg = CS.LegM("ray", "0.1", "0.2", phi0)
    p = [arb("2.5421480475"), arb("0.7769940136")]
    xi0, vals, ders = CS.start_series(p, 1, "0.1", phi0, 40)
    W0 = CS.W0()
    c = [acb(v.mid()) for v in vals]
    cols = [[acb(v.mid()) for v in ders[k]] for k in range(2)]
    wcol = [arb(0), W0]
    Wc = [[acb(complex(*(1e-3 * rng.normal(size=2)))) for _ in range(s.nd)] for _ in range(3)]
    pdx = [acb(arb(0, 1e-12)), acb(arb(0, 1e-12))]
    lo = start_set2(s, c, W0 * p[1], cols, wcol, Wc, pdx, 24, 12)
    h = 2.0 ** -12
    ok = lo.step(leg, 0.0, h)
    Wn = lo.W_box()
    # finite differences along x_k: y(e e_k) = y + e V0_k + e^2/2 W_kk, V(e e_k) = V0 + e W_{.k}
    e = 1e-20
    worst, contained = 0.0, ok
    pidx = pair_index(2)
    for k in range(2):
        Vn = []
        for sg in (1, -1):
            ee = sg * e
            yk = [c[i] + ee * cols[k][i] + ee * ee / 2 * Wc[pidx[(k, k)]][i] for i in range(s.nd)] + \
                 [acb(W0 * p[1] + ee * wcol[k])]
            Vk = [[cols[j][i] + ee * Wc[pidx[(j, k)]][i] for i in range(s.nd)] for j in range(2)]
            l1 = CL.start_set(s, yk[:s.nd], W0 * p[1] + ee * wcol[k], Vk, wcol, dx=None, N=24, NJ=12)
            l1.step(leg, 0.0, h)
            Vn.append(l1.var_box())
        for j in range(2):
            for i in range(s.nd):
                fd = (Vn[0][i, j] - Vn[1][i, j]) / (2 * e)
                wv = Wn[i, pidx[(j, k)]]
                d = float(abs(fd.mid() - wv.mid()))
                tol = float(wv.rad()) + 1e-12 * float(abs(wv).upper()) + float(fd.rad())
                worst = max(worst, d / (float(abs(wv).upper()) + 1e-300))
                contained &= d <= tol
    if verbose:
        print(f"M = 1: one LohnerM2 step's W vs central differences of two C^1 steps: step ok {ok}, contained "
              f"{contained}, relative midpoint mismatch {worst:.1e}")
    return contained, worst


def selftest():
    e1 = check_secondvar()
    e2 = check_H_series()
    okw, e3 = check_W_step()
    ok = e1 < 1e-40 and e2 < 1e-20 and okw and e3 < 1e-6
    print("SELFTEST2 " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--mutate="):
            MUTATE.add(a.split("=", 1)[1])
    if args and args[0] == "selftest":
        sys.exit(0 if selftest() else 1)
    print(__doc__)
