#!/usr/bin/env python3
"""The real DSS's imaginary sector (complex_imag.py) with its sensitive parts in double-double (#375).

complex_imag.py's det M(lambda) is rounding noise at D = 60 (matter/COMPLEX_CRITICAL.md section 3). Its sensitivity
lies in the propagators nearest the light cone and in the light-cone block. A relative change of 1e-14 in the
propagator from x = 0.6 to 0.7 moves the phase of det M by 2 rad, while the centre block and the segments up to 0.5
move it by at most 2e-10 rad. This module recomputes, in double-double (ddnp.py, about 32 digits):

  - the propagators of the last NDD segments (--dd=NDD, default all node-to-node segments). Background and test field
    are integrated together by a Taylor method of order K (--K) from the stored double-double node values (hi + lo of
    the hp state), with the step chosen from the last two terms so that their relative size is at most --tol;
  - the light-cone block H, from the test field's light-cone series (complex_imag.TestField.horizon's recursion), on the
    background's 34-digit light-cone series, with double-double solves of each order;
  - the LU of M (banded, partial pivoting) and ln det M = sum ln u_ii + i pi (row swaps).

The centre series, the propagation from x_s to the first node, and any segment not in double-double are complex_imag.py's
(DOP853 in double, --rtol). Counting (B0, U0, FULL, --plant) is complex_imag.count_box, unchanged.

Shares with hp: the stored state and the equations of hp/src/model.rs (derived from Box psi = 0 in complex_imag_eqs.py).
Shares with complex_imag.py: the background reconstruction in double (DOP853, rtol 1e-13), the centre series, the
34-digit light-cone series of the background, the candidate poles and count_box. The double-double arithmetic is
ddnp.py's (Dekker, Knuth, Hida-Li-Bailey), not multidouble's.

    python3 complex_imag_dd.py check STATE.npz [--dd=N] [--tol=1e-20] [--K=32] [--out=F]
          bg node mismatch per double-double segment, the lambda = 0 null vector, double-double against double det
    python3 complex_imag_dd.py det STATE.npz LAM [LAM ...] [--tol=...] [--dd=N] [--mutate=M] [--out=F]
    python3 complex_imag_dd.py tolscan STATE.npz [--tols=1e-12,1e-14,1e-16,1e-18,1e-20] [--lams=...] [--out=F]
    python3 complex_imag_dd.py count STATE.npz B0 U0 FULL [--plant=Z] [--mutate=coupling|noG] [--evlog=F] [--out=F]
    python3 complex_imag_dd.py selftest            the banded LU against numpy and mpmath, with a mutated control
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complex_imag as ci  # noqa: E402
import ddnp as dd  # noqa: E402
from ddnp import R, C  # noqa: E402

LU_MUTATE = None


def mp2dd(arr):
    a = np.asarray(arr, dtype=object)
    flat = a.ravel()
    h = np.array([float(x) for x in flat])
    l = np.array([float(x - mp.mpf(float(x))) for x in flat])
    return R(h.reshape(a.shape), l.reshape(a.shape))


def rvec(x, M):
    """a double (or R scalar) broadcast to a length-M R vector"""
    if isinstance(x, R):
        return R(np.full(M, float(x.h)), np.full(M, float(x.l)))
    return R(np.full(M, float(x)))


# ------------------------------------------------------------------ LU in double-double
def lu_factor(A, lb=None, ub=None):
    """LU with partial pivoting of a complex double-double matrix A (C, N x N), in place. lb, ub: the lower and upper
    bandwidths of A (None: dense); with pivoting U's upper bandwidth is at most lb + ub. Returns (A, perm, lndet)."""
    N = A.shape[0]
    perm = np.arange(N)
    lnabs, phase, swaps = 0.0, 0.0, 0
    ubu = None if (lb is None or ub is None) else lb + ub
    for k in range(N):
        r1 = N if lb is None else min(N, k + lb + 1)
        c1 = N if ubu is None else min(N, k + ubu + 1)
        mag = np.abs(A.re.h[k:r1, k]) + np.abs(A.im.h[k:r1, k])
        piv = k + int(np.argmax(mag))
        if LU_MUTATE == "nopivot":
            piv = k
        if piv != k:
            for arr in (A.re.h, A.re.l, A.im.h, A.im.l):
                arr[[k, piv], :] = arr[[piv, k], :]
            perm[[k, piv]] = perm[[piv, k]]
            swaps += 1
        p = A[k, k]
        pz = complex(p.re.h + p.re.l, p.im.h + p.im.l)
        lnabs += math.log(abs(pz))
        phase += math.atan2(pz.imag, pz.real)
        if r1 > k + 1:
            l = dd.cdiv(A[k + 1:r1, k], C(R(p.re.h, p.re.l), R(p.im.h, p.im.l)))
            A[k + 1:r1, k] = l
            if c1 > k + 1:
                u = A[k, k + 1:c1]
                A[k + 1:r1, k + 1:c1] = dd.csub(A[k + 1:r1, k + 1:c1], dd.cmul(l[:, None], u[None, :]))
    return A, perm, complex(lnabs, math.remainder(phase + math.pi * swaps, 2 * math.pi))


def lu_solve(A, perm, B):
    """solve with the factors of lu_factor (dense); B: C (N x k)"""
    N = A.shape[0]
    X = B[perm].copy() if B.re.h.ndim == 2 else B[perm].reshape(-1, 1)
    for k in range(N):                                   # unit lower
        if k + 1 < N:
            X[k + 1:] = dd.csub(X[k + 1:], dd.cmul(A[k + 1:, k][:, None], X[k][None, :]))
    for k in range(N - 1, -1, -1):                       # upper
        X[k] = dd.cdiv(X[k], C(R(A.re.h[k, k], A.re.l[k, k]), R(A.im.h[k, k], A.im.l[k, k])))
        if k > 0:
            X[:k] = dd.csub(X[:k], dd.cmul(A[:k, k][:, None], X[k][None, :]))
    return X


# ------------------------------------------------------------------ the double-double background and Taylor step
class DDState:
    """double-double node values (hi + lo of the hp state), oD, n, and the background's light-cone series"""

    def __init__(self, bg):
        zf = np.load(bg.path, allow_pickle=True)
        zh, zl = zf["zh"], zf["zl"]
        M, K = bg.M, bg.K
        self.M = M
        self.node = []
        for i in range(K):
            o = 2 * M + 4 * M * i
            self.node.append([R(zh[o + M * f:o + M * (f + 1)].copy(), zl[o + M * f:o + M * (f + 1)].copy())
                              for f in range(4)])
        mp.mp.dps = bg.dps
        self.oD = mp2dd(bg.oD_mp)
        n_mp = mp.mpf(bg.D_mp) - 2
        self.n = mp2dd([n_mp])[0]
        self.nm1 = mp2dd([n_mp - 1])[0]
        self.hn = mp2dd([n_mp / 2])[0]
        self.hs = {k: mp2dd(v) for k, v in bg.hs_mp.items()}


def cpv(F, H, k, lo=0):
    """sum_{j=lo}^{k} F_j H_{k-j} for stacked R vectors (rows = orders)"""
    if k < lo:
        return R(np.zeros(F.h.shape[1:]))
    return dd.dsum(dd.mul(F[lo:k + 1], H[k - lo::-1]), 0)


def cdot(Fs, Ps, w=None):
    """sum_t w_t Fs_t[:, None] Ps_t for an R vector stack Fs (n, M) and a C matrix stack Ps (n, M, nc)"""
    if w is not None:
        Fs = dd.mul_d(Fs, w[:, None])
    return dd.cdsum(dd.rcmul(R(Fs.h[:, :, None], Fs.l[:, :, None]), Ps), 0)


def cpm(F, P, k, lo=0):
    """sum_{j=lo}^{k} F_j[:, None] P_{k-j}"""
    return cdot(F[lo:k + 1], P[k - lo::-1])


def matvec(A, v):
    return dd.matmul(A, v.reshape(-1, 1)).reshape(-1)


def taylor_series(st, x0, ybg, Y, lam, K, kcoup=1.0, gG=1.0):
    """Taylor coefficients in xi = x - x0 (x0 a double) of the background (p, m, a, c: R stacks (K+1, M)) and of the
    test field Y = (P; Q) (C stacks (K+1, M, nc)) at x0"""
    M = st.M
    nm1, hn = st.nm1, st.hn
    X0 = rvec(x0, M)
    x0sq = R(*dd.two_prod(np.full(M, x0), np.full(M, x0)))
    p, m, a, c = (dd.zeros((K + 1, M)) for _ in range(4))
    G, A2, S, sm, sp, pm = (dd.zeros((K + 1, M)) for _ in range(6))
    p[0], m[0], a[0], c[0] = ybg
    for k in range(K + 1):
        pm[k] = dd.add(p[k], m[k])
        A2[k] = cpv(a, a, k)
        S[k] = dd.add(cpv(p, p, k), cpv(m, m, k))
        G[k] = dd.mul(rvec(nm1, M), dd.sub(cpv(c, A2, k), c[k]))
        xc = dd.mul_d(c[k], x0)
        if k >= 1:
            xc = dd.add(xc, c[k - 1])
        x2 = x0sq if k == 0 else rvec(2 * x0 if k == 1 else (1.0 if k == 2 else 0.0), M)
        sm[k] = dd.sub(x2, xc)
        sp[k] = dd.add(x2, xc)
        if k == K:
            break
        Hk = dd.add(cpv(a, S, k), dd.mul(rvec(nm1, M), dd.sub(a[k], cpv(a, A2, k))))
        c[k + 1] = dd.div(dd.sub(G[k], dd.mul_d(c[k], float(k))), dd.mul_d(X0, float(k + 1)))
        a[k + 1] = dd.div(dd.sub(Hk, dd.mul_d(a[k], float(2 * k))), dd.mul_d(X0, float(2 * (k + 1))))
        oDp = matvec(st.oD, p[k])
        oDm = matvec(st.oD, m[k])
        xoDp = dd.mul_d(oDp, x0)
        xoDm = dd.mul_d(oDm, x0)
        if k >= 1:
            xoDp = dd.add(xoDp, matvec(st.oD, p[k - 1]))
            xoDm = dd.add(xoDm, matvec(st.oD, m[k - 1]))
        cpmk = dd.mul(rvec(hn, M), cpv(c, pm, k))
        Rp = dd.add(dd.sub(dd.sub(cpv(G, p, k), cpv(c, p, k)), xoDp), cpmk)
        Rm = dd.sub(dd.sub(dd.add(cpv(c, m, k), dd.neg(cpv(G, m, k))), xoDm), cpmk)
        if k >= 1:
            w = np.arange(k, 0, -1, dtype=float)
            Rp = dd.sub(Rp, dd.dsum(dd.mul(dd.mul_d(sm[1:k + 1], w[:, None]), p[k:0:-1]), 0))
            Rm = dd.sub(Rm, dd.dsum(dd.mul(dd.mul_d(sp[1:k + 1], w[:, None]), m[k:0:-1]), 0))
        p[k + 1] = dd.div(Rp, dd.mul_d(sm[0], float(k + 1)))
        m[k + 1] = dd.div(Rm, dd.mul_d(sp[0], float(k + 1)))
    bgs = dict(p=p, m=m, a=a, c=c)
    if Y is None:
        return bgs, None
    nc = Y.shape[1]
    hnv = rvec(hn, M)
    wP = dd.add(dd.sub(dd.mul_d(G, gG), c), dd.mul_d(dd.mul(R(hnv.h[None], hnv.l[None]), c), kcoup))
    cpP = dd.mul_d(dd.mul(R(hnv.h[None], hnv.l[None]), c), kcoup)
    wQ = dd.add(dd.sub(dd.mul_d(G, gG), c), dd.mul(R(hnv.h[None], hnv.l[None]), c))
    cpQ = dd.mul(R(hnv.h[None], hnv.l[None]), c)
    P, Q, LP, LQ = (dd.czeros((K + 1, M, nc)) for _ in range(4))
    P[0] = Y[:M]
    Q[0] = Y[M:]
    sm0 = sm[0]
    sp0 = sp[0]
    for k in range(K):
        LP[k] = dd.cadd(dd.cmul_c(P[k], lam), dd.cmatmul(st.oD, P[k]))
        LQ[k] = dd.cadd(dd.cmul_c(Q[k], lam), dd.cmatmul(st.oD, Q[k]))
        xLP = dd.cmul_d(LP[k], x0)
        xLQ = dd.cmul_d(LQ[k], x0)
        if k >= 1:
            xLP = dd.cadd(xLP, LP[k - 1])
            xLQ = dd.cadd(xLQ, LQ[k - 1])
        RP = dd.cadd(dd.csub(cpm(wP, P, k), xLP), cpm(cpP, Q, k))
        RQ = dd.csub(dd.csub(dd.cneg(xLQ), cpm(wQ, Q, k)), cpm(cpQ, P, k))
        if k >= 1:
            w = np.arange(k, 0, -1, dtype=float)                 # j = 1..k: (k + 1 - j) P_{k+1-j}
            RP = dd.csub(RP, cdot(sm[1:k + 1], P[k:0:-1], w))
            RQ = dd.csub(RQ, cdot(sp[1:k + 1], Q[k:0:-1], w))
        dP = dd.mul_d(sm0, float(k + 1))
        dQ = dd.mul_d(sp0, float(k + 1))
        P[k + 1] = dd.rdiv_c(RP, R(dP.h[:, None], dP.l[:, None]))
        Q[k + 1] = dd.rdiv_c(RQ, R(dQ.h[:, None], dQ.l[:, None]))
    return bgs, (P, Q)


def horner_r(F, h):
    acc = F[F.h.shape[0] - 1]
    for k in range(F.h.shape[0] - 2, -1, -1):
        acc = dd.add(dd.mul_d(acc, h), F[k])
    return acc


def horner_c(F, h):
    acc = F[F.re.h.shape[0] - 1]
    for k in range(F.re.h.shape[0] - 2, -1, -1):
        acc = dd.cadd(dd.cmul_d(acc, h), F[k])
    return acc


def step_size(bgs, ts, K, tol):
    """largest h with |f_k| h^k <= tol |f_0| for k = K - 1, K: per background field, and per column of the test
    field (P and Q rows together)"""
    h = np.inf
    for k in (K - 1, K):
        for f in ("p", "m", "a", "c"):
            a0, ak = np.max(np.abs(bgs[f].h[0])), np.max(np.abs(bgs[f].h[k]))
            if ak > 0 and a0 > 0:
                h = min(h, (tol * a0 / ak) ** (1.0 / k))
        if ts is not None:
            col = lambda X, i: np.maximum(np.max(np.abs(X[0].re.h[i]) + np.abs(X[0].im.h[i]), axis=0),
                                          np.max(np.abs(X[1].re.h[i]) + np.abs(X[1].im.h[i]), axis=0))
            a0, ak = col(ts, 0), col(ts, k)
            ok = (a0 > 0) & (ak > 0)
            if np.any(ok):
                h = min(h, float(np.min((tol * a0[ok] / ak[ok]) ** (1.0 / k))))
    return 0.9 * h


def dd_segment(st, x0, x1, ybg, Y, lam, tol, K=32, kcoup=1.0, gG=1.0, stats=None):
    """integrate background (and test field Y, if given) from x0 to x1 in double-double"""
    x = x0
    nstep = 0
    while x < x1:
        bgs, ts = taylor_series(st, x, ybg, Y, lam, K, kcoup, gG)
        h = step_size(bgs, ts, K, tol)
        if not h > 1e-12 * max(1.0, abs(x)):
            raise RuntimeError("step size %g at x = %r" % (h, x))
        xn = x1 if x + h >= x1 else x + h
        h = xn - x                       # exact (Sterbenz): x and xn within a factor 2 for these steps
        ybg = [horner_r(bgs[f], h) for f in ("p", "m", "a", "c")]
        if ts is not None:
            Pn, Qn = horner_c(ts[0], h), horner_c(ts[1], h)
            Y = C(R(np.concatenate([Pn.re.h, Qn.re.h]), np.concatenate([Pn.re.l, Qn.re.l])),
                  R(np.concatenate([Pn.im.h, Qn.im.h]), np.concatenate([Pn.im.l, Qn.im.l])))
        x = xn
        nstep += 1
    if stats is not None:
        stats.append(nstep)
    return ybg, Y


# ------------------------------------------------------------------ the light-cone block in double-double
def solve_refine(oD, lam, kap, B, log, iters=4):
    """(lam + oD + diag(kap)) X = B in double-double: LU in double (scipy), residuals in double-double, refined until
    the correction is below 1e-33 of X (at most iters times); log gets the last relative correction"""
    import scipy.linalg as sl
    M = oD.h.shape[0]
    d = dd.add_d(kap, 0.0)
    dz = C(dd.add_d(d, lam.real), R(np.full(M, lam.imag)))          # lam + kap, the diagonal
    Ad = oD.h + oD.l + np.diag(dz.re.h + 1j * dz.im.h)
    fac = sl.lu_factor(Ad)
    X = dd.from_complex(sl.lu_solve(fac, B.to_complex()))
    rel = 1.0
    col = C(R(dz.re.h[:, None], dz.re.l[:, None]), R(dz.im.h[:, None], dz.im.l[:, None]))
    for _ in range(iters):
        Rr = dd.csub(B, dd.cadd(dd.cmatmul(oD, X), dd.cmul(col, X)))
        dX = sl.lu_solve(fac, Rr.to_complex())
        X = dd.cadd(X, dd.from_complex(dX))
        rel = float(np.max(np.abs(dX)) / max(dd.maxabs(X), 1e-300))
        if rel < 1e-33:
            break
    log.append(rel)
    return X


def horizon_dd(st, bg, lam, x, kcoup=1.0, gG=1.0, ks=None, stats=None):
    """complex_imag.TestField.horizon's series for Q_h = unit vectors, in double-double, evaluated at x: C (2M, M)"""
    M = st.M
    K = bg.ks if ks is None else ks
    hs = st.hs
    cc, e, v = hs["c"], hs["e"], hs["v"]
    G = dd.mul_d(hs["G"], gG)
    hn = rvec(st.hn, M)
    hp_ = dd.mul_d(hn, kcoup)
    P = dd.czeros((K + 1, M, M))
    Q = dd.czeros((K + 2, M, M))
    Q.re.h[0] = np.eye(M)
    G0 = G[0]
    refine_log = []
    one = R(np.ones(M))
    col = lambda r: R(r.h[:, None], r.l[:, None])

    def Lmul(X):
        return dd.cadd(dd.cmul_c(X, lam), dd.cmatmul(st.oD, X))
    for k in range(K + 1):
        kap = dd.sub(dd.add(dd.neg(dd.mul_d(e[1], float(k))), dd.sub(one, G0)), hp_)
        rhs = dd.rcmul(col(hp_), Q[k])
        if k >= 1:
            rhs = dd.cadd(rhs, Lmul(P[k - 1]))
        if k >= 2:
            w = np.arange(k - 1, 0, -1, dtype=float)            # j = 2..k: (k + 1 - j) P_{k+1-j}
            rhs = dd.cadd(rhs, cdot(e[2:k + 1], P[k - 1:0:-1], w))
        if k >= 1:
            rhs = dd.cadd(rhs, cpm(dd.sub(G, cc[:K + 1]), P, k, lo=1))
            rhs = dd.cadd(rhs, dd.rcmul(col(hp_), dd.cadd(cpm(cc, P, k, lo=1), cpm(cc, Q, k, lo=1))))
        P[k] = solve_refine(st.oD, lam, kap, rhs, refine_log)
        rq = Lmul(dd.csub(Q[k], Q[k - 1]) if k >= 1 else Q[k])
        if k >= 1:
            w = np.arange(k, 0, -1, dtype=float)                 # j = 1..k: (k + 1 - j) Q_{k+1-j}
            rq = dd.csub(rq, cdot(v[1:k + 1], Q[k:0:-1], w))
        rq = dd.cadd(rq, cpm(dd.sub(G, cc[:K + 1]), Q, k))
        rq = dd.cadd(rq, dd.rcmul(col(hn), dd.cadd(cpm(cc, P, k), cpm(cc, Q, k))))
        Q[k + 1] = dd.rdiv_c(rq, col(dd.mul_d(v[0], float(k + 1))))
    u = 1.0 - x                                                   # exact for x in [0.5, 1]
    Ph, Qh = horner_c(P, u), horner_c(Q[:K + 1], u)
    tail = max(dd.maxabs(P[K]), dd.maxabs(Q[K])) * u ** K
    out = dd.czeros((2 * M, M))
    out[:M] = Ph
    out[M:] = Qh
    if stats is not None:
        stats["refine_last_rel"] = max(refine_log)
    return out, tail


# ------------------------------------------------------------------ the determinant
class DDTestField:
    """det M with matching points: the stored nodes, and nsub - 1 equally spaced points inside each double-double
    segment (and inside the double ones with nsub_double). det M does not change in exact arithmetic when a matching
    point is added (eliminate the new unknowns: det(-I) = 1), but each piece's propagator is then well conditioned."""

    def __init__(self, bg, mutate=None, ndd=None, tol=1e-20, K=32, rtol=1e-13, kf=None, nsub=5, nsub_double=1):
        if kf is not None:
            raise SystemExit("kf is not implemented in double-double")
        self.bg = bg
        self.st = DDState(bg)
        self.tf = ci.TestField(bg, mutate, rtol=rtol)       # centre series, double segments, poles
        self.mutate = mutate
        self.kcoup, self.gG = self.tf.kcoup, self.tf.gG
        self.tol, self.K = tol, K
        L = len(bg.X)
        assert bg.X == bg.nodes, "x_e beyond the last node is not supported here"
        self.ndd = L - 1 if ndd is None else ndd
        self.nsub, self.nsub_double = nsub, nsub_double
        self.last_steps = []
        self.last_tail = None
        self.last_times = {}
        self.hstats = {}

    def poles(self):
        return self.tf.poles()

    def pieces(self):
        bg = self.bg
        L = len(bg.X)
        si = 1 if bg.pre else 0
        out = []
        for b in range(1, L):
            x0, x1, seg = bg.segs[si + b - 1]
            isdd = b >= L - self.ndd
            k = self.nsub if isdd else self.nsub_double
            xs = [x0 + (x1 - x0) * j / k for j in range(k)] + [x1]
            out += [(xs[j], xs[j + 1], isdd, seg, b, j == 0) for j in range(k)]
        return out

    def matrix_dd(self, lam):
        bg, M, st = self.bg, self.bg.M, self.st
        pcs = self.pieces()
        Lp = len(pcs) + 1
        N = 2 * M * (Lp + 1)
        A = dd.czeros((N, N))
        I2 = np.eye(2 * M)
        t0 = time.time()
        Cm = self.tf.centre(lam, bg.xs)
        if bg.pre:
            x0, x1, seg = bg.segs[0]
            Cm = self.tf.propagate(seg, x0, x1, Cm, lam)
        A[0:2 * M, 0:M] = dd.from_complex(Cm)
        A.re.h[0:2 * M, M:3 * M] = -I2
        self.last_steps = []
        ybg = None
        for i, (x0, x1, isdd, seg, b, first) in enumerate(pcs, start=1):
            r0 = 2 * M * i
            if isdd:
                if first:
                    ybg = st.node[b - 1]                    # the stored double-double node starts each segment
                Y0 = C(R(I2.copy()), R(np.zeros((2 * M, 2 * M))))
                ybg, Phi = dd_segment(st, x0, x1, ybg, Y0, lam, self.tol, self.K, self.kcoup, self.gG,
                                      self.last_steps)
            else:
                Phi = dd.from_complex(self.tf.propagate(seg, x0, x1, I2.astype(complex), lam))
            A[r0:r0 + 2 * M, M + 2 * M * (i - 1):M + 2 * M * i] = Phi
            A.re.h[r0:r0 + 2 * M, M + 2 * M * i:M + 2 * M * (i + 1)] = -I2
        t1 = time.time()
        H, self.last_tail = horizon_dd(st, bg, lam, bg.X[-1], self.kcoup, self.gG, stats=self.hstats)
        t2 = time.time()
        r0 = 2 * M * Lp
        A.re.h[r0:r0 + 2 * M, M + 2 * M * (Lp - 1):M + 2 * M * Lp] = -I2
        A[r0:r0 + 2 * M, M + 2 * M * Lp:] = H
        self.last_times = dict(propagators=round(t1 - t0, 2), horizon=round(t2 - t1, 2), N=N)
        return A

    def lndet(self, lam):
        lam = complex(lam)
        A = self.matrix_dd(lam)
        M = self.bg.M
        t = time.time()
        _, _, v = lu_factor(A, lb=3 * M, ub=3 * M)
        self.last_times["lu"] = round(time.time() - t, 2)
        return v


# ------------------------------------------------------------------ checks
def check(bg, ddtf, out_emit):
    st, M = ddtf.st, bg.M
    L = len(bg.X)
    rec = dict(kind="check_dd", state=os.path.basename(bg.path), D=bg.D, tol=ddtf.tol, K=ddtf.K, ndd=ddtf.ndd,
               nsub=ddtf.nsub, nsub_double=ddtf.nsub_double)
    # 1. the background integrated in double-double from node b-1 to node b against the stored node b
    mism = []
    for b in range(max(1, L - ddtf.ndd), L):
        steps = []
        y1, _ = dd_segment(st, bg.X[b - 1], bg.X[b], st.node[b - 1], None, 0.0, ddtf.tol, ddtf.K, stats=steps)
        d = max(float(np.max(np.abs(dd.sub(y1[f], st.node[b][f]).h))) for f in range(4))
        mism.append(dict(x0=bg.X[b - 1], x1=bg.X[b], maxdiff=d, steps=steps[0]))
    rec["bg_node_mismatch_dd"] = mism
    # 2. lambda = 0: the background's (p, m) is a test-field solution; propagated by the double-double propagator
    nul = []
    for b in range(max(1, L - ddtf.ndd), L):
        z0 = C(R(np.concatenate([st.node[b - 1][0].h, st.node[b - 1][1].h])[:, None],
                 np.concatenate([st.node[b - 1][0].l, st.node[b - 1][1].l])[:, None]), dd.zeros((2 * M, 1)))
        _, z1 = dd_segment(st, bg.X[b - 1], bg.X[b], st.node[b - 1], z0, 0.0, ddtf.tol, ddtf.K, ddtf.kcoup, ddtf.gG)
        ref = np.concatenate([st.node[b][0].h, st.node[b][1].h]) + np.concatenate([st.node[b][0].l, st.node[b][1].l])
        got = z1.to_complex()[:, 0]
        nul.append(dict(x0=bg.X[b - 1], x1=bg.X[b], maxdiff=float(np.max(np.abs(got - ref))),
                        maxval=float(np.max(np.abs(ref)))))
    rec["null_lambda0_segment"] = nul
    # 3. the light-cone block at lambda = 0 applied to the background's m_h reproduces (p, m) at the last node
    Hh, tail = horizon_dd(st, bg, 0j, bg.X[-1], ddtf.kcoup, ddtf.gG)
    zf = np.load(bg.path, allow_pickle=True)
    o = 2 * M + 4 * M * bg.K
    mh = zf["zh"][o:o + M] + zf["zl"][o:o + M]
    got = Hh.to_complex() @ mh
    ref = np.concatenate([st.node[-1][0].h + st.node[-1][0].l, st.node[-1][1].h + st.node[-1][1].l])
    rec["null_lambda0_horizon"] = dict(maxdiff=float(np.max(np.abs(got - ref))), tail=tail)
    # 4. double-double against double ln det
    lam = 1.3 + 0.4j
    t = time.time()
    v_dd = ddtf.lndet(lam)
    rec["lndet_dd"] = [v_dd.real, v_dd.imag]
    rec["seconds_dd"] = round(time.time() - t, 1)
    rec["steps"] = ddtf.last_steps
    rec["times"] = ddtf.last_times
    rec["horizon_tail"] = ddtf.last_tail
    rec["horizon_refine_last_rel"] = ddtf.hstats.get("refine_last_rel")
    # 5. matching points: det M is unchanged in exact arithmetic when nsub changes
    keep = ddtf.nsub
    ddtf.nsub = 1
    v_1 = ddtf.lndet(lam)
    ddtf.nsub = keep
    rec["lndet_dd_nsub1"] = [v_1.real, v_1.imag]
    rec["nsub_minus_nsub1"] = [v_dd.real - v_1.real, math.remainder(v_dd.imag - v_1.imag, 2 * math.pi)]
    v_d = ddtf.tf.lndet(lam)
    rec["lndet_double"] = [v_d.real, v_d.imag]
    rec["dd_minus_double"] = [v_dd.real - v_d.real, math.remainder(v_dd.imag - v_d.imag, 2 * math.pi)]
    out_emit(rec)


def selftest():
    """the banded double-double LU against mpmath on a random banded complex matrix; mutated control: no pivoting on
    a matrix whose first pivot is tiny"""
    global LU_MUTATE
    mp.mp.dps = 50
    rng = np.random.default_rng(3)
    N, lb, ub = 40, 5, 7
    Z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    mask = np.fromfunction(lambda i, j: (j - i <= ub) & (i - j <= lb), (N, N))
    Z = Z * mask
    Z[0, 0] = 1e-30                                   # forces a swap at k = 0; without it, growth 1e30
    ref = mp.log(mp.det(mp.matrix(Z.tolist())))
    ok = True
    res = {}
    for mut in (None, "nopivot"):
        LU_MUTATE = mut
        _, _, v = lu_factor(dd.from_complex(Z), lb=lb, ub=ub)
        err = abs(mp.mpc(v.real, v.imag) - ref)
        err = min(err, abs(mp.mpc(v.real, v.imag) - ref + 2j * mp.pi), abs(mp.mpc(v.real, v.imag) - ref - 2j * mp.pi))
        res[str(mut)] = float(err)
    LU_MUTATE = None
    ok = res["None"] < 1e-12
    # with no pivoting the tiny pivot loses digits: the control must be visibly worse (or the selftest is blind)
    ctl = res["nopivot"] > 1e3 * max(res["None"], 1e-16)
    # solve against numpy on a dense well-conditioned matrix
    A = rng.standard_normal((12, 12)) + 1j * rng.standard_normal((12, 12)) + 5 * np.eye(12)
    B = rng.standard_normal((12, 3)) + 0j
    fac, perm, _ = lu_factor(dd.from_complex(A))
    X = lu_solve(fac, perm, dd.from_complex(B)).to_complex()
    serr = float(np.max(np.abs(X - np.linalg.solve(A, B))))
    ok = ok and ctl and serr < 1e-13
    print("selftest: ln det banded LU error %.1e (limit 1e-12), no-pivot control %.1e (%s), solve vs numpy %.1e: %s"
          % (res["None"], res["nopivot"], "fires" if ctl else "DOES NOT FIRE", serr, "PASS" if ok else "FAIL"))
    return ok


def main():
    pos, o = ci.opts_of(sys.argv[1:])
    if not pos:
        raise SystemExit(__doc__)
    if pos[0] == "selftest":
        sys.exit(0 if selftest() else 1)
    out = o.get("out")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
    mutate = o.get("mutate") or None
    bg = ci.Background(pos[1], ks_series=int(o.get("ks", "100")))
    ndd = int(o["dd"]) if "dd" in o else None
    ddtf = DDTestField(bg, mutate, ndd=ndd, tol=float(o.get("tol", "1e-20")), K=int(o.get("K", "32")),
                       rtol=float(o.get("rtol", "1e-13")), nsub=int(o.get("nsub", "5")),
                       nsub_double=int(o.get("nsub_double", "1")))
    info = dict(state=os.path.basename(bg.path), D=bg.D, M=bg.M, nodes=bg.nodes, omega=bg.omega, mutate=mutate,
                tol=ddtf.tol, K=ddtf.K, ndd=ddtf.ndd, rtol=ddtf.tf.rtol, ks=bg.ks, nsub=ddtf.nsub,
                nsub_double=ddtf.nsub_double)
    if pos[0] == "check":
        check(bg, ddtf, emit)
        return
    if pos[0] == "det":
        for s in pos[2:]:
            lam = complex(s.replace(" ", ""))
            t = time.time()
            v = ddtf.lndet(lam)
            emit(dict(kind="det_dd", **info, lam=[lam.real, lam.imag], lndet=[v.real, v.imag],
                      seconds=round(time.time() - t, 1), steps=ddtf.last_steps, times=ddtf.last_times,
                      horizon_tail=ddtf.last_tail))
        return
    if pos[0] == "tolscan":
        tols = [float(t) for t in o.get("tols", "1e-12,1e-14,1e-16,1e-18,1e-20").split(",")]
        om = bg.omega
        lams = [complex(s) for s in o["lams"].split(",")] if "lams" in o else \
            [0.1 + 0.3j, 1.3 + 0.4j, 3.0 + 1.0j, 5.0 + 0.5 * om * 1j, 0.15 + 1.0 * om * 1j, 5.9 + 1.4 * om * 1j]
        for lam in lams:
            for tl in tols:
                ddtf.tol = tl
                t = time.time()
                v = ddtf.lndet(lam)
                emit(dict(kind="tolscan_dd", **{**info, "tol": tl}, lam=[lam.real, lam.imag], lndet=[v.real, v.imag],
                          seconds=round(time.time() - t, 1), steps=ddtf.last_steps))
        return
    if pos[0] == "count":
        plant = complex(o["plant"].replace(" ", "")) if "plant" in o else None
        boxes = ci.std_boxes(bg.omega)
        evlog = o.get("evlog")
        ci.load_cache(evlog, mutate)

        def log(r):
            r["mutate"] = mutate
            if evlog:
                with open(evlog, "a") as fo:
                    fo.write(json.dumps(r) + "\n")
        for spec in pos[2:]:
            name, bx = (spec, boxes[spec]) if spec in boxes else \
                (spec.split(":")[0], tuple(float(v) for v in spec.split(":")[1:5]))
            t = time.time()
            res = ci.count_box(ddtf, bx, plant=plant, tol=float(o.get("wtol", math.pi / 4)), log=log,
                               h0=float(o.get("h0", "0.25")))
            emit(dict(kind="count_dd", **info, box_name=name, box=list(bx), plant=None if plant is None else
                      [plant.real, plant.imag], seconds=round(time.time() - t, 1), **res))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
