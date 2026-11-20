#!/usr/bin/env python3
"""The l = 1 mode matrix M(lambda) of dss_ell.DSSEll in double-double (#1025, route (c)).

Same discrete operator as DSSEll.mode_matrix (frozen RK4 step counts, the stored double state z, the same x grid), built
in double-double so that M's small singular values (the decaying-mode ladder of matter/ELL.md section 6) are resolved
far below double rounding. Differences of method, all exact in exact arithmetic:
  - the state is carried in parity-restricted Fourier coefficients (p, m, psi odd; a, c even), so d/dtau is elementwise
    (k omega) and the truncation to k <= kc is the projection back to coefficients; products are formed on the same
    N-point grid;
  - derivatives are carried as dual numbers (value plus one tangent per input direction) instead of a complex step;
  - the tau operators (coefficients <-> grid values, d/dtau on the grid) are built exactly in mpmath and rounded to
    double-double; products with them use Ozaki's error-free splitting into float64 BLAS products;
  - the horizon series' periodic solves are a float64 LU refined with double-double residuals.

    python ell_ddm.py selftest [--mutate=slices|ops]
        operators against DSSEll's float64 ones (1e-14), the Ozaki product against ddnp.matmul (1e-30 relative), and
        dual-number derivatives against the analytic ones in double-double (1e-28); --mutate=slices (2 slices) and
        --mutate=ops (one harmonic off) must fail.
References: Ozaki, Ogita, Oishi and Rump, Numer. Algorithms 59, 95 (2012) (error-free splitting of matrix products);
double-double arithmetic as in ddnp.py (Dekker 1971; Hida, Li and Bailey 2001)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ddnp as dd                                    # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
NSLICE = 2 if MUTATE == "slices" else 5


# ---------------------------------------------------------------- exact tau operators
def tau_operators(N, kodd, keven):
    """Exact (mpmath, 40 digits) operators, rounded to double-double, in DSSEll's conventions:
    Bodd (n_odd x N), Beven (n_even x N): coefficients -> grid values (c @ B, DSSSpec.phys);
    Sodd (N x n_odd), Seven (N x n_even): grid values -> coefficients (f @ S, DSSSpec.spec);
    Dm (N x N): d/dtheta on the grid, f @ Dm (DSSEll._Dm: rfft, times i k, Nyquist zeroed, irfft)."""
    import mpmath as mp
    mp.mp.dps = 40
    kodd = [int(k) + (1 if MUTATE == "ops" else 0) for k in kodd]
    keven = [int(k) for k in keven]
    th = [2 * mp.pi * j / N for j in range(N)]

    def todd(rows):
        a = np.array([[float(v) for v in r] for r in rows])
        b = np.array([[float(v - float(v)) for v in r] for r in rows])
        return dd.R(a, b)
    Bo = [[2 * mp.cos(k * t) for t in th] for k in kodd] + [[-2 * mp.sin(k * t) for t in th] for k in kodd]
    Be = [[mp.mpf(1)] * N] + [[2 * mp.cos(k * t) for t in th] for k in keven[1:]] + \
         [[-2 * mp.sin(k * t) for t in th] for k in keven[1:]]
    So = [[mp.cos(k * t) / N for k in kodd] + [-mp.sin(k * t) / N for k in kodd] for t in th]
    Se = [[mp.mpf(1) / N] + [mp.cos(k * t) / N for k in keven[1:]] + [-mp.sin(k * t) / N for k in keven[1:]] for t in th]
    d = [-(mp.mpf(2) / N) * mp.fsum(k * mp.sin(2 * mp.pi * k * r / N) for k in range(1, N // 2)) for r in range(N)]
    Dm = [[d[(j - i) % N] for j in range(N)] for i in range(N)]
    return dict(Bodd=todd(Bo), Beven=todd(Be), Sodd=todd(So), Seven=todd(Se), Dm=todd(Dm))


# ---------------------------------------------------------------- Ozaki products
def _slices(h, l, axis, n):
    """exact split of the double-double array h + l into NSLICE float64 slices along rows (axis=-1 scales per row) or
    columns (axis=-2), each with few enough bits that products of slices summed over n terms are exact in float64"""
    rho = math.ceil((53 + math.log2(max(n, 2))) / 2)
    rh, rl = h.copy(), l.copy()
    out = []
    for _ in range(NSLICE):
        mx = np.max(np.abs(rh), axis=axis, keepdims=True)
        tau = np.where(mx > 0, np.ceil(np.log2(np.where(mx > 0, mx, 1.0))), 0.0)
        sig = 0.75 * np.exp2(rho + tau)
        s = (rh + sig) - sig
        out.append(s)
        rh = rh - s                                          # exact
        rh, rl = dd.two_sum(rh, rl)
    return out


class Op:
    """a fixed double-double operator W (n x m), pre-split for X @ W"""

    def __init__(self, W):
        self.W = W
        self.n = W.shape[0]
        self.sl = _slices(W.h, W.l, axis=-2, n=self.n)

    def rmul(self, X):
        """X @ W for a double-double X (..., n); exact slice products accumulated in double-double"""
        shp = X.shape
        Xh = X.h.reshape(-1, shp[-1]); Xl = X.l.reshape(-1, shp[-1])
        xs = _slices(Xh, Xl, axis=-1, n=self.n)
        acc = None
        for p in range(NSLICE):
            for q in range(NSLICE - p):
                P = xs[p] @ self.sl[q]
                acc = dd.R(P) if acc is None else dd.add(acc, dd.R(P))
        return dd.R(acc.h.reshape(shp[:-1] + (-1,)), acc.l.reshape(shp[:-1] + (-1,)))


# ---------------------------------------------------------------- dual double-double numbers
def _R(x):
    return x if isinstance(x, dd.R) else dd.R(np.asarray(x, dtype=float))


class DT:
    """value v (dd.R, broadcast over the member axis) and tangents t (dd.R with one row per input direction, or None)"""
    __slots__ = ("v", "t")

    def __init__(self, v, t=None):
        self.v = v
        self.t = t

    def __add__(self, o):
        if isinstance(o, DT):
            t = self.t if o.t is None else (o.t if self.t is None else dd.add(self.t, o.t))
            return DT(dd.add(self.v, o.v), t)
        return DT(dd.add_d(self.v, np.asarray(o, dtype=float)), self.t)
    __radd__ = __add__

    def __neg__(self):
        return DT(dd.neg(self.v), None if self.t is None else dd.neg(self.t))

    def __sub__(self, o):
        return self + (-o if isinstance(o, DT) else -np.asarray(o, dtype=float))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, DT):
            v = dd.mul(self.v, o.v)
            ts = []
            if o.t is not None:
                ts.append(dd.mul(self.v, o.t))
            if self.t is not None:
                ts.append(dd.mul(self.t, o.v))
            t = None if not ts else (ts[0] if len(ts) == 1 else dd.add(ts[0], ts[1]))
            return DT(v, t)
        if isinstance(o, dd.R):
            return DT(dd.mul(self.v, o), None if self.t is None else dd.mul(self.t, o))
        o = np.asarray(o, dtype=float)
        return DT(dd.mul_d(self.v, o), None if self.t is None else dd.mul_d(self.t, o))
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, DT):
            q = dd.div(self.v, o.v)
            t = self.t
            if o.t is not None:
                qt = dd.mul(q, o.t)
                t = dd.neg(qt) if t is None else dd.sub(t, qt)
            return DT(q, None if t is None else dd.div(t, _bcast(o.v, t)))
        o = _R(o)
        return DT(dd.div(self.v, _bcast(o, self.v)), None if self.t is None else dd.div(self.t, _bcast(o, self.t)))

    def __rtruediv__(self, o):
        q = dd.div(_bcast(_R(o), self.v), self.v)
        return DT(q, None if self.t is None else dd.neg(dd.div(dd.mul(q, self.t), _bcast(self.v, self.t))))

    def lin(self, f):
        """a linear map f applied to value and tangents"""
        return DT(f(self.v), None if self.t is None else f(self.t))


def _bcast(b, a):
    """b broadcast to a's shape (ddnp.div reads b.h elementwise)"""
    return dd.R(np.broadcast_to(b.h, np.broadcast_shapes(b.h.shape, a.h.shape)).copy(),
                np.broadcast_to(b.l, np.broadcast_shapes(b.l.shape, a.l.shape)).copy())


def selftest():
    sys.path.insert(0, HERE)
    from dss_ell import DSSEll, NODES_STD
    ok = True
    S = DSSEll(kc=15, nodes=NODES_STD, n=11.0, ell=1, Delta_ref=1.0, Ks=30, u0=0.04)
    ops = tau_operators(S.N, S.kodd, S.keven)
    chk = dict(Bodd=S._B["odd"], Beven=S._B["even"], Sodd=S._S["odd"], Seven=S._S["even"], Dm=S._Dm)
    errs = {k: float(np.max(np.abs(ops[k].h + ops[k].l - v)) / max(1.0, np.max(np.abs(v)))) for k, v in chk.items()}
    good = max(errs.values()) < 1e-14
    print(f"selftest operators vs DSSEll's float64 ones (k_c 15, N {S.N}): " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f" (bar 1e-14): {'OK' if good else 'BAD'}", flush=True)
    ok &= good
    g = np.random.default_rng(1)
    X = dd.R(g.standard_normal((7, S.N)) * np.exp2(g.integers(-30, 30, (7, 1))))
    X = dd.add(X, dd.R(X.h * 1e-17 * g.standard_normal(X.shape)))
    ref = dd.matmul(X, ops["Dm"])
    got = Op(ops["Dm"]).rmul(X)
    rel = float(np.max(np.abs((got.h - ref.h) + (got.l - ref.l)) / np.max(np.abs(ref.h), axis=1, keepdims=True)))
    good = rel < 1e-30
    print(f"selftest Ozaki product ({NSLICE} slices) vs ddnp.matmul: largest error {rel:.1e} of the row's largest entry (bar "
          f"1e-30): {'OK' if good else 'BAD'}", flush=True)
    ok &= good
    # dual numbers: f(u) = u^3 / (1 + u) - 2 u; its tangent against f'(u) t = ((2u^3 + 3u^2) / (1 + u)^2 - 2) t in dd
    u0 = dd.R(g.uniform(0.2, 2.0, (1, 5)))
    tt = dd.R(g.standard_normal((3, 5)))
    f = lambda u: (u * u * u) / (1.0 + u) - u * 2.0
    D = f(DT(u0, tt))
    u2 = dd.mul(u0, u0); u3 = dd.mul(u2, u0); w = dd.add_d(u0, 1.0)
    fp = dd.add_d(dd.div(dd.add(dd.mul_d(u3, 2.0), dd.mul_d(u2, 3.0)), dd.mul(w, w)), -2.0)
    ex = dd.mul(_bcast(fp, tt), tt)
    rel = float(np.max(np.abs((D.t.h - ex.h) + (D.t.l - ex.l))) / np.max(np.abs(ex.h)))
    good = rel < 1e-28
    print(f"selftest dual derivative vs the analytic one in double-double: {rel:.1e} (bar 1e-28): {'OK' if good else 'BAD'}",
          flush=True)
    ok &= good
    print(f"selftest: {'PASS' if ok else 'FAIL'}" + (f" [mutate {MUTATE}]" if MUTATE else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        sys.exit(selftest())
