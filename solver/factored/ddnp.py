"""Double-double arrays on numpy, with multidouble's Python package for the arithmetic (#1088).

DD holds hi and lo float64 arrays of one shape; + - * / broadcast like numpy and round once per operation to
double-double (multidouble's elementwise add, sub, mul, div). dot0 sums products over axis 0 with multidouble's
dd_dot, matvec and solve use its dd_matvec and dd_solve (LU with partial pivoting).

Dual holds a DD value v and a DD tangent t (forward-mode differentiation in double-double): every operation applies
the product and quotient rules, so a residual evaluated at Dual(y, e_i) returns column i of the exact Jacobian,
rounded to double-double. dot0, matvec, solve, diag, sum0 and cauchy accept either type.
"""
import mpmath as mp
import multidouble as md
import numpy as np


class DD:
    __slots__ = ("h", "l")

    def __init__(self, h, l=None):
        self.h = np.array(h, dtype=float)
        self.l = np.zeros_like(self.h) if l is None else np.array(l, dtype=float)

    @property
    def shape(self):
        return self.h.shape

    def __len__(self):
        return len(self.h)

    def __getitem__(self, i):
        return DD(self.h[i], self.l[i])

    def __setitem__(self, i, v):
        v = asdd(v)
        self.h[i] = v.h
        self.l[i] = v.l

    def copy(self):
        return DD(self.h.copy(), self.l.copy())

    def __neg__(self):
        return DD(-self.h, -self.l)

    def __add__(self, o):
        if type(o).__name__ == "Dual":
            return o + self
        return _op(md.add, self, o)

    __radd__ = __add__

    def __sub__(self, o):
        if type(o).__name__ == "Dual":
            return -(o - self)
        return _op(md.sub, self, o)

    def __rsub__(self, o):
        return _op(md.sub, o, self)

    def __mul__(self, o):
        if type(o).__name__ == "Dual":
            return o * self
        return _op(md.mul, self, o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if type(o).__name__ == "Dual":
            return o.__rtruediv__(self)
        return _op(md.div, self, o)

    def __rtruediv__(self, o):
        return _op(md.div, o, self)

    def f(self):
        return self.h + self.l

    def mp(self):
        return [mp.mpf(float(a)) + mp.mpf(float(b)) for a, b in zip(self.h.ravel(), self.l.ravel())]


def asdd(x):
    if isinstance(x, DD):
        return x
    return DD(x)


def _op(fn, a, b):
    a = asdd(a)
    b = asdd(b)
    ah, bh = np.broadcast_arrays(a.h, b.h)
    al, bl = np.broadcast_arrays(a.l, b.l)
    shape = ah.shape
    rh, rl = fn(np.ascontiguousarray(ah).ravel(), np.ascontiguousarray(al).ravel(),
                np.ascontiguousarray(bh).ravel(), np.ascontiguousarray(bl).ravel())
    return DD(np.asarray(rh).reshape(shape), np.asarray(rl).reshape(shape))


def zeros(shape):
    return DD(np.zeros(shape))


def from_mp(vals, shape=None):
    hs, ls = [], []
    for v in vals:
        h = float(v)
        hs.append(h)
        ls.append(float(v - h))
    d = DD(np.array(hs), np.array(ls))
    if shape is not None:
        d = DD(d.h.reshape(shape), d.l.reshape(shape))
    return d


def dot0(A, B):
    """sum_j A[j, m] B[j, m] for each m (A, B of shape (L, M)), in double-double."""
    L, M = A.h.shape
    if L == 0:
        return zeros((M,))
    oh = np.empty(M)
    ol = np.empty(M)
    Ah = np.ascontiguousarray(A.h.T)
    Al = np.ascontiguousarray(A.l.T)
    Bh = np.ascontiguousarray(B.h.T)
    Bl = np.ascontiguousarray(B.l.T)
    for m in range(M):
        oh[m], ol[m] = md.dd_dot(Ah[m], Al[m], Bh[m], Bl[m])
    return DD(oh, ol)


def cauchy(A, B, k, lo, hi):
    """sum_{j=lo}^{hi} A[j] B[k-j], pointwise in the last axis."""
    M = A.h.shape[1]
    if hi < lo:
        return zeros((M,))
    Bs = B[k - hi:k - lo + 1]
    return dot0(A[lo:hi + 1], DD(Bs.h[::-1], Bs.l[::-1]))


def matvec(Am, x):
    h, l = md.dd_matvec(np.ascontiguousarray(Am.h), np.ascontiguousarray(Am.l), np.ascontiguousarray(x.h),
                        np.ascontiguousarray(x.l))
    return DD(np.asarray(h), np.asarray(l))


def solve(Am, b):
    h, l = md.dd_solve(np.ascontiguousarray(Am.h), np.ascontiguousarray(Am.l), np.ascontiguousarray(b.h),
                       np.ascontiguousarray(b.l))
    return DD(np.asarray(h), np.asarray(l))


def diag(v):
    M = len(v.h)
    out = zeros((M, M))
    out.h[np.arange(M), np.arange(M)] = v.h
    out.l[np.arange(M), np.arange(M)] = v.l
    return out


def sum0(A):
    L, M = A.h.shape
    return dot0(A, DD(np.ones((L, M))))


def sqrt(x):
    """sqrt in double-double by one Newton step on the float64 root."""
    r = np.sqrt(x.h)
    rr = DD(r)
    return rr + (x - rr * rr) / (2.0 * rr)


def pi_dd():
    with mp.workdps(50):
        return from_mp([mp.pi])[0]


def dmat_odd_dd(M):
    with mp.workdps(50):
        N = 2 * M
        vals = []
        for i in range(M):
            for j in range(M):
                v = mp.mpf(0)
                for jj, sgn in ((j, 1), (j + M, -1)):
                    if jj != i:
                        v += sgn * mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
                vals.append(v)
        return from_mp(vals, (M, M))


def dmat_even_dd(M):
    with mp.workdps(50):
        vals = []
        for i in range(M):
            for j in range(M):
                vals.append(mp.mpf(0) if i == j else (-1) ** (i - j) * mp.cot((i - j) * mp.pi / M))
        return from_mp(vals, (M, M))


def phase_dd(M):
    with mp.workdps(50):
        return from_mp([mp.sin(mp.pi * j / M) for j in range(M)])


class Dual:
    __slots__ = ("v", "t")

    def __init__(self, v, t=None):
        self.v = asdd(v)
        self.t = zeros(self.v.shape) if t is None else asdd(t)

    @property
    def shape(self):
        return self.v.shape

    @property
    def h(self):
        return self.v.h

    @property
    def l(self):
        return self.v.l

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return Dual(self.v[i], self.t[i])

    def __setitem__(self, i, x):
        if isinstance(x, Dual):
            self.v[i] = x.v
            self.t[i] = x.t
        else:
            self.v[i] = x
            self.t[i] = 0.0

    def copy(self):
        return Dual(self.v.copy(), self.t.copy())

    def f(self):
        return self.v.f()

    def __neg__(self):
        return Dual(-self.v, -self.t)

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v, self.t + o.t)
        return Dual(self.v + o, self.t)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v - o.v, self.t - o.t)
        return Dual(self.v - o, self.t)

    def __rsub__(self, o):
        return Dual(o - self.v, -self.t)

    def __mul__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v * o.v, self.t * o.v + self.v * o.t)
        return Dual(self.v * o, self.t * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            q = self.v / o.v
            return Dual(q, (self.t - q * o.t) / o.v)
        return Dual(self.v / o, self.t / o)

    def __rtruediv__(self, o):
        q = asdd(o) / self.v
        return Dual(q, -(q * self.t) / self.v)


def zeros_like_kind(shape, dual):
    return Dual(zeros(shape)) if dual else zeros(shape)


_dot0_dd = dot0
_matvec_dd = matvec
_solve_dd = solve
_diag_dd = diag
_sum0_dd = sum0


def dot0(A, B):  # noqa: F811
    da, db = isinstance(A, Dual), isinstance(B, Dual)
    if not (da or db):
        return _dot0_dd(A, B)
    if da and db:
        return Dual(_dot0_dd(A.v, B.v), _dot0_dd(A.t, B.v) + _dot0_dd(A.v, B.t))
    if da:
        return Dual(_dot0_dd(A.v, B), _dot0_dd(A.t, B))
    return Dual(_dot0_dd(A, B.v), _dot0_dd(A, B.t))


def cauchy(A, B, k, lo, hi):  # noqa: F811
    M = A.shape[1]
    if hi < lo:
        return zeros((M,))
    return dot0(A[lo:hi + 1], B[k - hi:k - lo + 1][::-1])


def matvec(Am, x):  # noqa: F811
    if isinstance(Am, Dual):
        xv = x.v if isinstance(x, Dual) else x
        out_t = _matvec_dd(Am.t, xv)
        if isinstance(x, Dual):
            out_t = out_t + _matvec_dd(Am.v, x.t)
        return Dual(_matvec_dd(Am.v, xv), out_t)
    if isinstance(x, Dual):
        return Dual(_matvec_dd(Am, x.v), _matvec_dd(Am, x.t))
    return _matvec_dd(Am, x)


def solve(Am, b):  # noqa: F811
    if not isinstance(Am, Dual) and not isinstance(b, Dual):
        return _solve_dd(Am, b)
    Av = Am.v if isinstance(Am, Dual) else Am
    bv = b.v if isinstance(b, Dual) else b
    x = _solve_dd(Av, bv)
    rhs = b.t if isinstance(b, Dual) else zeros(bv.shape)
    if isinstance(Am, Dual):
        rhs = rhs - _matvec_dd(Am.t, x)
    return Dual(x, _solve_dd(Av, rhs))


def diag(v):  # noqa: F811
    if isinstance(v, Dual):
        return Dual(_diag_dd(v.v), _diag_dd(v.t))
    return _diag_dd(v)


def sum0(A):  # noqa: F811
    if isinstance(A, Dual):
        return Dual(_sum0_dd(A.v), _sum0_dd(A.t))
    return _sum0_dd(A)
