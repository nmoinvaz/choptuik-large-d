"""Form S (fseries_s.py) in double-double: the same recurrences, every operation rounded to double-double (#1088).

The arithmetic is ddnp.py's (multidouble's Python package). The tau matrices, pi and the phase weights are formed in
mpmath and rounded to double-double. The Jacobian is by central differences in double-double (truncation h^2, rounding
eps_dd/h with eps_dd ~ 1e-32), and the Newton step is solved in double-double.

Usage (a solve): python3 fseries_sdd.py --D=100 --M=8 --start=STATE [--iters=8] [--out=FILE.json] [--h=2e-11]
  STATE: an hp npz or a json of fsolve.py / this script; rescaled to D as in fsolve.py when its D differs.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402


def _col(x):
    return x[:, None] if not isinstance(x, d.Dual) else d.Dual(DD(x.v.h[:, None], x.v.l[:, None]), DD(x.t.h[:, None], x.t.l[:, None]))


class FactoredSDD:
    def __init__(self, D, M, kc=160, ks=None, zK=0.5, mutate="", kb=None):
        self.Dd = float(D)
        self.n = self.Dd - 2.0
        n = self.n
        self.M = M
        self.kc = kc
        self.ks = ks if ks is not None else int(n + 14 * np.sqrt(n) + 80)
        self.zK = zK
        self.wK = 1.0 - zK
        self.Do = d.dmat_odd_dd(M)
        self.De = d.dmat_even_dd(M)
        self.phase = d.phase_dd(M)
        self.pi = d.pi_dd()
        self.mutate = mutate
        self.stats = {}
        self.dual = False
        self.kb = kb

    def mv(self, A, x):
        return d.matvec(A, x)

    def jset(self, k, lo, hi):
        """Indices j in [lo, hi] of the products A_j B_{k-j} kept: all of them, or with self.kb set (the banded
        recurrence) those with j <= lo + kb or k - j <= kb, where one factor is in the slow part. The pairs dropped have
        both factors past order kb: the shadow's square, zero at a solution together with its first derivative."""
        if self.kb is None or hi - lo + 1 <= 2 * self.kb + 2:
            return np.arange(lo, hi + 1)
        a = np.arange(lo, min(hi, lo + self.kb) + 1)
        b = np.arange(max(lo, k - self.kb), hi + 1)
        return np.unique(np.concatenate([a, b]))

    def pair(self, A, B, k, lo, hi):
        """sum over the kept j in [lo, hi] of A_j B_{k-j}"""
        if hi < lo:
            return d.zeros((self.M,))
        J = self.jset(k, lo, hi)
        return d.dot0(A[J], B[k - J])

    def xsolve(self, om, qs0, rhs):
        A = self.De * om - d.diag(qs0 + (self.n - 1))
        return d.solve(A, rhs)

    # ------------------------------------------------------------------ centre
    def centre(self, p1, c0, om):
        n, K, M = self.n, self.kc, self.M
        Z = lambda: d.zeros_like_kind((K + 1, M), self.dual)  # noqa: E731
        U, V, X, beta, c, G, W, qs = Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z()
        c[0] = c0
        V[0] = p1 * 2.0
        for k in range(1, K + 1):
            j = np.arange(1, k)[:, None].astype(float)
            num = V[k - 1] * (2.0 * (k - 1) + 1.0) + om * self.mv(self.Do, V[k - 1])
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
            num = U[k] * (2.0 * k) + om * self.mv(self.Do, U[k])
            if k > 1:
                num = num - d.dot0(c[1:k] * (2.0 * (k - j)), V[k - 1:0:-1])
            num = num - d.cauchy(G, V, k, 1, k)
            V[k] = num / (c0 * (2.0 * k))
        zp = DD(np.repeat((self.zK ** np.arange(K + 1))[:, None], M, axis=1))
        out = {nm: d.dot0(Xs, zp) for nm, Xs in (("U", U), ("V", V), ("c", c), ("X", X))}
        out["tail"] = max(float(np.max(np.abs(Y.h[-1]))) * self.zK ** K for Y in (U, V, c, X))
        return out

    # ------------------------------------------------------------------ horizon
    def horizon(self, Q0, om):
        n, ks, M = self.n, self.ks, self.M
        lam = self.wK
        Z = lambda r=ks + 1: d.zeros_like_kind((r, M), self.dual)  # noqa: E731
        U, V, DU, DV, X, beta, G, cG, H, f, W, V2, qs = (Z() for _ in range(13))
        c, c2 = Z(ks + 2), Z(ks + 2)
        c[0] = 1.0
        c2[0] = 1.0
        qs[0] = Q0 * Q0 * 0.5
        X[0] = self.xsolve(om, qs[0], qs[0])
        beta[0] = X[0] * (-(n - 1))
        B0 = X[0] + 1.0
        G[0] = beta[0] / B0
        g = G[0]
        c[1] = G[0] * (-lam / 2)
        c2[1] = c[1] * 2.0
        e = -c2[1] - lam
        f[0] = e * (1.0 / lam)
        eprev = e
        cG[0] = G[0]
        H[0] = cG[0] + (n - 1)
        A0 = self.Do * om - d.diag(g + (n / 2 - 1))
        P = d.solve(A0, Q0 * (n / 2))
        U[0] = (P + Q0) * 0.5
        V[0] = (P - Q0) * 0.5
        DU[0] = self.mv(self.Do, U[0])
        DV[0] = self.mv(self.Do, V[0])
        V2[0] = V[0] * V[0]
        W[0] = U[0] * V[0]
        one_g = 1.0 - g
        for k in range(1, ks + 1):
            J = self.jset(k, 1, k)  # the pairs (j, k - j) kept: all, or the two bands of the banded recurrence
            jj = J[:, None].astype(float)
            Ur = U[k - J]
            Vr = V[k - J]
            DUr = DU[k - J]
            DVr = DV[k - J]
            fk = f[J] * (k - jj)
            cJ = c[J]
            cGk0 = self.pair(c, G, k, 1, k)
            cG[k] = cGk0
            H[k] = cGk0 + c2[k] * (n - 1)
            G[k] = 0.0
            r1 = (d.dot0(fk, Vr) * (-2.0) + om * d.dot0(cJ, DUr) - om * DV[k - 1] * lam
                  - d.dot0(cG[J], Vr) - V[k - 1] * lam - d.dot0(G[J] + cJ * (n - 1), Ur))
            cz = cJ - c[J - 1] * lam
            cgz = (cJ - G[J]) - (c[J - 1] - G[J - 1]) * lam
            r2 = (d.dot0(fk, Ur) * (-2.0) - om * DU[k - 1] * lam + om * d.dot0(cz, DVr) + d.dot0(cgz, Vr)
                  - d.dot0(H[J], Ur))
            Qk = (r1 - r2) / (one_g * (2.0 * k))
            U2p = self.pair(U, U, k, 1, k - 1)
            V2p = self.pair(V, V, k, 1, k - 1)
            Wp = self.pair(U, V, k, 1, k - 1)
            cWlow = self.pair(c, W, k, 1, k)
            qs[k] = Q0 * Qk + (U2p + V2p - V2[k - 1] * lam) * 0.5 - Wp - cWlow
            rhs = qs[k] + self.pair(qs, X, k, 1, k)
            X[k] = self.xsolve(om, qs[0], rhs)
            beta[k] = X[k] * (-(n - 1))
            G[k] = (self.pair(c, beta, k, 0, k) - self.pair(X, G, k, 1, k)) / B0
            cG[k] = cGk0 + G[k]
            H[k] = cG[k] + c2[k] * (n - 1)
            c[k + 1] = (c[k] * (2.0 * k) - G[k]) * (lam / (2 * (k + 1)))
            c2[k + 1] = self.pair(c, c, k + 1, 0, k + 1)
            e = -c2[k + 1]
            f[k] = (e - eprev * lam) * (1.0 / lam)
            eprev = e
            r2 = r2 - G[k] * (V[0] + U[0])
            Ak = self.Do * om + d.diag(one_g * float(k) - (g + (n / 2 - 1)))
            if self.mutate == "sigma":
                Ak = self.Do * om + d.diag(one_g * float(k) - (n / 2 - 1))
            Pk = d.solve(Ak, -r2 - Qk * (one_g * float(k) - n / 2))
            U[k] = (Pk + Qk) * 0.5
            V[k] = (Pk - Qk) * 0.5
            DU[k] = self.mv(self.Do, U[k])
            DV[k] = self.mv(self.Do, V[k])
            V2[k] = self.pair(V, V, k, 0, k)
            W[k] = self.pair(U, V, k, 0, k)
        out = {"U": d.sum0(U), "V": d.sum0(V), "c": d.sum0(c[:ks + 1]), "X": d.sum0(X), "X0": X[0]}
        out["tail"] = max(float(np.max(np.abs(Y.h[-1]))) for Y in (U, V, c[:ks + 1], X))
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

    def jacobian(self, y, hrel=None):
        """The exact Jacobian in double-double by forward-mode differentiation, one column per evaluation."""
        L = len(y)
        J = d.zeros((L, L))
        for i in range(L):
            e = np.zeros(L)
            e[i] = 1.0
            col = self.residual(d.Dual(y, DD(e))).t
            J.h[:, i] = col.h
            J.l[:, i] = col.l
        self.dual = False
        return J

    def jacobian_fd(self, y, hrel=2.0**-36):
        L = len(y)
        J = d.zeros((L, L))
        for i in range(L):
            h = hrel * max(1e-3, abs(float(y.h[i])))
            h = 2.0 ** np.round(np.log2(h))
            yp = y.copy()
            ym = y.copy()
            yp[i] = y[i] + h
            ym[i] = y[i] - h
            col = (self.residual(yp) - self.residual(ym)) * (0.5 / h)
            J.h[:, i] = col.h
            J.l[:, i] = col.l
        return J


def load_y(start, M, D):
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from fsolve import load_start
    y4, D0 = load_start(start, M)
    y = np.concatenate([y4[:3 * M], y4[4 * M:]])
    yl = np.zeros_like(y)
    if not start.endswith(".npz"):
        js = json.load(open(start))
        if js.get("M") == M and "p1_lo" in js:
            yl = np.concatenate([js["p1_lo"], js["c0_lo"], js["Q0_lo"], [js["Delta_lo"]]])
    n, n0 = D - 2, D0 - 2
    if abs(D - D0) > 1e-12:
        s = np.sqrt(n0 / n)
        y[0:M] *= s
        y[2 * M:3 * M] *= s
        y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
        yl[:] = 0.0
    return DD(y, yl), D0


if __name__ == "__main__":
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    D = float(opts["D"])
    M = int(opts.get("M", 8))
    iters = int(opts.get("iters", 8))
    y, D0 = load_y(opts["start"], M, D)
    if "Delta" in opts:
        y[3 * M] = float(opts["Delta"])
    f = FactoredSDD(D, M, kc=int(opts.get("kc", 160)), ks=int(opts["ks"]) if "ks" in opts else None,
                    mutate=opts.get("mutate", ""))
    log = lambda s: print(s, flush=True)  # noqa: E731
    log("dd form S: D %.6f n %.4f M %d kc %d ks %d start %s (D0 %.6f)" % (D, f.n, M, f.kc, f.ks, opts["start"].rsplit("/", 1)[-1], D0))
    t0 = time.time()
    hist = []
    J = None
    for it in range(iters + 1):
        F = f.residual(y)
        nf = float(np.max(np.abs(F.h)))
        log("it %d max|F| %.3e Delta %.20f  tails c %.1e h %.1e (%.0f s)" % (it, nf, float(y.h[3 * M]) + 0.0, f.last[0]["tail"],
                                                                         f.last[1]["tail"], time.time() - t0))
        log("    Delta (dd) %s" % repr((float(y.h[3 * M]), float(y.l[3 * M]))))
        hist.append({"it": it, "maxF": nf, "Delta": [float(y.h[3 * M]), float(y.l[3 * M])], "t": time.time() - t0})
        if it == iters or nf < 1e-30:
            break
        if J is None or opts.get("chord", "0") == "0" or it % int(opts.get("chord", "1")) == 0:
            J = f.jacobian(y, hrel=float(opts.get("h", 2.0**-36)))
        dy = d.solve(J, -F)
        rs = np.linalg.norm(J.h, axis=1)
        r0 = np.linalg.norm(F.h / rs)
        alpha = 1.0
        for _ in range(12):
            with np.errstate(all="ignore"):
                Ft = f.residual(y + dy * alpha)
            rt = np.linalg.norm(Ft.h / rs)
            if np.isfinite(rt) and rt < (1 - 1e-4 * alpha) * r0:
                break
            alpha *= 0.5
        y = y + dy * alpha
        log("   step max|dy| %.3e dDelta %.3e alpha %.3g" % (float(np.max(np.abs(dy.h))), float(dy.h[3 * M]), alpha))
    # conditioning of the (rounded) Jacobian, in mpmath at 40 digits
    import mpmath as mp
    with mp.workdps(40):
        Jm = mp.matrix(J.h.shape[0], J.h.shape[1])
        for a in range(J.h.shape[0]):
            for b in range(J.h.shape[1]):
                Jm[a, b] = mp.mpf(float(J.h[a, b])) + mp.mpf(float(J.l[a, b]))
        r = [mp.sqrt(sum(Jm[a, b]**2 for b in range(Jm.cols))) for a in range(Jm.rows)]
        for a in range(Jm.rows):
            for b in range(Jm.cols):
                Jm[a, b] /= r[a]
        cn = [mp.sqrt(sum(Jm[a, b]**2 for a in range(Jm.rows))) for b in range(Jm.cols)]
        for a in range(Jm.rows):
            for b in range(Jm.cols):
                Jm[a, b] /= cn[b]
        sv = mp.svd_r(Jm, compute_uv=False)
        sv = sorted([float(v) for v in sv], reverse=True)
    log("scaled singular values: max %.3e min %.3e cond %.3e; smallest five %s" % (
        sv[0], sv[-1], sv[0] / sv[-1], " ".join("%.2e" % v for v in sv[-5:])))
    log("Delta %.20f (+ %.3e)" % (float(y.h[3 * M]), float(y.l[3 * M])))
    if "out" in opts:
        js = {"D": D, "M": M, "kc": f.kc, "ks": f.ks, "form": "S-dd", "p1": list(y.h[0:M]), "c0": list(y.h[M:2 * M]),
              "Q0": list(y.h[2 * M:3 * M]), "Delta": float(y.h[3 * M]), "p1_lo": list(y.l[0:M]), "c0_lo": list(y.l[M:2 * M]),
              "Q0_lo": list(y.l[2 * M:3 * M]), "Delta_lo": float(y.l[3 * M]), "hist": hist, "sv_scaled": sv,
              "start": opts["start"], "seconds": time.time() - t0, "mutate": opts.get("mutate", "")}
        json.dump(js, open(opts["out"], "w"), indent=1)
