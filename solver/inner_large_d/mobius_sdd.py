"""mobius_s.py in double-double: form S with the horizon series in the Moebius variable, every operation rounded to
double-double, with exact forward-mode Jacobians (ddnp.Dual) (#1090).

The centre series is Track B's (solver/factored/fseries_sdd.py, FactoredSDD.centre, copied); the horizon recurrence is
mobius_s.FormS.horizon written in ddnp operations. The Moebius coefficients h, e, z (and t* = 1/(2 + a)) are formed in
mpmath and rounded to double-double, since t* is not a binary fraction.
"""
import numpy as np
import mpmath as mp

import ddnp as d
from ddnp import DD


def _col(x):
    if isinstance(x, d.Dual):
        return d.Dual(DD(x.v.h[:, None], x.v.l[:, None]), DD(x.t.h[:, None], x.t.l[:, None]))
    return DD(x.h[:, None], x.l[:, None])


def _dd(v):
    with mp.workdps(50):
        x = mp.mpf(v)
        h = float(x)
        return DD(np.array(h), np.array(float(x - h)))


class FormSDD:
    def __init__(self, D, M, a=1, kc=160, ks=None, kb=150, zK=0.5):
        self.D = float(D)
        self.n = n = self.D - 2.0
        self.M = M
        self.kc = kc
        self.ks = ks if ks is not None else int(n + 14 * np.sqrt(n) + 80)
        self.kb = kb
        self.zK = zK
        self.Do = d.dmat_odd_dd(M)
        self.De = d.dmat_even_dd(M)
        self.phase = d.phase_dd(M)
        self.pi = d.pi_dd()
        self.dual = False
        K = self.ks + 4
        with mp.workdps(50):
            a = mp.mpf(a)
            ts = (1 - mp.mpf(zK)) / (1 + a * mp.mpf(zK))
            h = [1 / (1 + a), (a - 1) / (1 + a) * ts, -a / (1 + a) * ts ** 2]
            e = [1 / (1 + a), -2 / (1 + a) * ts, 1 / (1 + a) * ts ** 2]
            zs = [mp.mpf(1)] + [-(1 + a) * (-a) ** (j - 1) * ts ** j for j in range(1, K)]
            hh = h + [mp.mpf(0)] * (K - 3)
            ee = e + [mp.mpf(0)] * (K - 3)
            self.io = float(1 / ts) if (1 / ts) == int(1 / ts) else None
            self.io_dd = _dd(1 / ts)
            rep = lambda L: d.from_mp([v for v in L for _ in range(M)], (K, M))  # noqa: E731
            self.ZS = rep(zs)
            self.H = rep(hh)
            self.E = rep(ee)
            self.h = [_dd(v) for v in h]
            self.e = [_dd(v) for v in e]
            self.ts = float(ts)

    def J(self, k, lo, hi):
        if hi < lo:
            return np.arange(0)
        if self.kb is None or hi - lo + 1 <= 2 * self.kb + 2:
            return np.arange(lo, hi + 1)
        return np.concatenate([np.arange(lo, lo + self.kb + 1), np.arange(max(lo + self.kb + 1, k - self.kb), hi + 1)])

    def cv(self, A, B, k, lo=0, hi=None):
        hi = k if hi is None else hi
        J = self.J(k, lo, hi)
        if len(J) == 0:
            return d.zeros((self.M,))
        return d.dot0(A[J], B[k - J])

    def xsolve(self, om, qs0, rhs):
        A = self.De * om - d.diag(qs0 + (self.n - 1))
        return d.solve(A, rhs)

    # ------------------------------------------------------------------ centre (Track B, FactoredSDD.centre)
    def centre(self, p1, c0, om):
        n, K, M = self.n, self.kc, self.M
        Z = lambda: d.zeros_like_kind((K + 1, M), self.dual)  # noqa: E731
        U, V, X, beta, c, G, W, qs = Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z()
        c[0] = c0
        V[0] = p1 * 2.0
        for k in range(1, K + 1):
            j = np.arange(1, k)[:, None].astype(float)
            num = V[k - 1] * (2.0 * (k - 1) + 1.0) + om * d.matvec(self.Do, V[k - 1])
            if k > 1:
                wts = c[1:k] * (2 * (k - j) + (n - 1)) + G[1:k]
                num = num - d.dot0(wts, U[k - 1:0:-1])
            U[k] = num / (c0 * (2.0 * k + n - 1))
            W[k] = d.cauchy(U, V, k, 1, k)
            cW = d.cauchy(c, W, k, 0, k - 1)
            qs[k] = (d.cauchy(U, U, k, 1, k - 1) + d.cauchy(V, V, k - 1, 0, k - 1)) * 0.5 - cW
            rhs = qs[k] + d.cauchy(qs, X, k, 1, k)
            X[k] = self.xsolve(om, qs[0], rhs)
            beta[k] = X[k] * (-(n - 1))
            G[k] = d.cauchy(c, beta, k, 0, k - 1) - d.cauchy(X, G, k, 1, k - 1)
            c[k] = G[k] * (1.0 / (2 * k)) if (2 * k) & (2 * k - 1) == 0 else G[k] / float(2 * k)
            num = U[k] * (2.0 * k) + om * d.matvec(self.Do, U[k])
            if k > 1:
                num = num - d.dot0(c[1:k] * (2.0 * (k - j)), V[k - 1:0:-1])
            num = num - d.cauchy(G, V, k, 1, k)
            V[k] = num / (c0 * (2.0 * k))
        zp = DD(np.repeat((self.zK ** np.arange(K + 1))[:, None], M, axis=1))
        out = {nm: d.dot0(Xs, zp) for nm, Xs in (("U", U), ("V", V), ("c", c))}
        out["tail"] = max(float(np.max(np.abs(Y.h[-1]))) * self.zK ** K for Y in (U, V, c, X))
        return out

    # ------------------------------------------------------------------ horizon (Moebius variable, tau-units)
    def horizon(self, Q0, om, keep=False):
        n, ks, M = self.n, self.ks, self.M
        h, e, io = self.h, self.e, self.io_dd
        K = ks + 3
        Z = lambda: d.zeros_like_kind((K, M), self.dual)  # noqa: E731
        U, V, X, G, c, q, dU, dV, Qn, VVc, UVc, hc = (Z() for _ in range(12))
        c[0] = 1.0
        Qn[0] = Q0
        Ax = self.De * om - d.diag(Q0 * Q0 * 0.5 + (n - 1))
        two_io_h0 = io * h[0] * 2.0
        for k in range(0, ks + 1):
            U[k] = Qn[k] * 0.5
            V[k] = Qn[k] * (-0.5)
            VVc[k] = self.cv(V, V, k)
            UVc[k] = self.cv(U, V, k)
            q[k] = (self.cv(U, U, k) + self.cv(self.ZS, VVc, k)) * 0.5 - self.cv(c, UVc, k)
            rhs = q[k] + self.cv(q, X, k, 1, k) if k >= 1 else q[k]
            X[k] = d.solve(Ax, rhs)
            num = self.cv(c, X, k) * (-(n - 1))
            if k >= 1:
                num = num - self.cv(G, X, k, 0, k - 1)
            G[k] = num / (X[0] + 1.0)
            s = G[k] * (-0.5)
            for i in range(1, min(k, 2) + 1):
                s = s - c[k - i + 1] * (io * h[i] * float(k - i + 1))
            c[k + 1] = s / (io * h[0] * float(k + 1))
            hk = c[k] * h[0]
            if k >= 1:
                hk = hk + c[k - 1] * h[1]
            if k >= 2:
                hk = hk + c[k - 2] * h[2]
            hc[k] = hk
            if k >= 1:
                dU[k - 1] = U[k] * float(k)
                dV[k - 1] = V[k] * float(k)
                m = self.J(k, 0, k - 1)
                hcm = hc[k - m]
                deriv = (d.dot0(hcm - self.H[k - m], dU[m]) + d.dot0(hcm - self.E[k - m], dV[m])) * (io * 2.0)
            else:
                deriv = None
            zVk = self.cv(self.ZS, V, k)
            r = (om * d.matvec(self.Do, U[k] + zVk) - self.cv(G, U, k) - self.cv(G, V, k) + zVk
                 - self.cv(c, U, k) * (n - 1))
            if deriv is not None:
                r = r + deriv
            Ldiag = c[1] * (io * h[0] * (2.0 * k)) + (io * (h[1] - e[1]) * float(k)) - G[0] - (n / 2 - 1)
            Lk = self.Do * om + d.diag(Ldiag)
            Pk = d.solve(Lk, -r)
            U[k] = U[k] + Pk * 0.5
            V[k] = V[k] + Pk * 0.5
            if k >= 1:
                dU[k - 1] = U[k] * float(k)
                dV[k - 1] = V[k] * float(k)
            VVc[k] = self.cv(V, V, k)
            UVc[k] = self.cv(U, V, k)
            dU[k] = 0.0
            dV[k] = 0.0
            Ak = None
            for i in range(0, min(k, 2) + 1):
                mm = k - i
                t = (dU[mm] - self.cv(c, dV, mm)) * h[i]
                Ak = t if Ak is None else Ak + t
            Ak = Ak * (io * (-2.0)) + om * d.matvec(self.Do, U[k]) - self.cv(G, V, k)
            Qn[k + 1] = Ak / (two_io_h0 * float(k + 1))
        out = {"U": d.sum0(U[:ks + 1]), "V": d.sum0(V[:ks + 1]), "c": d.sum0(c[:ks + 1])}
        out["tail"] = max(float(np.max(np.abs(Y.h[ks]))) for Y in (U, V, c))
        if keep:
            out["series"] = {"U": U[:ks + 1], "V": V[:ks + 1], "c": c[:ks + 1]}
        return out

    # ------------------------------------------------------------------ residual, Jacobian
    def residual(self, y):
        M = self.M
        self.dual = isinstance(y, d.Dual)
        p1, c0, Q0, Delta = y[0:M], y[M:2 * M], y[2 * M:3 * M], y[3 * M]
        om = (self.pi * 2.0) / Delta
        ce = self.centre(p1, c0, om)
        ho = self.horizon(Q0, om)
        ph = d.dot0(DD(self.phase.h[:, None], self.phase.l[:, None]), _col(p1))
        R = d.zeros_like_kind((3 * M + 1,), self.dual)
        R[0:M] = ce["U"] - ho["U"]
        R[M:2 * M] = ce["V"] - ho["V"]
        R[2 * M:3 * M] = ce["c"] - ho["c"]
        R[3 * M] = ph[0]
        self.last = (ce, ho)
        return R

    def jacobian(self, y):
        """The exact Jacobian in double-double by forward-mode differentiation, one column per evaluation."""
        L = len(y)
        J = d.zeros((L, L))
        for i in range(L):
            ev = np.zeros(L)
            ev[i] = 1.0
            col = self.residual(d.Dual(y, DD(ev))).t
            J.h[:, i] = col.h
            J.l[:, i] = col.l
        self.dual = False
        return J
