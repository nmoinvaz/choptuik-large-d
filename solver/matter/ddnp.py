"""Double-double arithmetic on numpy arrays (about 32 digits): real values as (hi, lo) pairs of float64 arrays.

Written for complex_imag_dd.py (#375) as an implementation separate from multidouble and hp. Error-free transforms
after Dekker (Numer. Math. 18, 224 (1971)) and Knuth (TAOCP vol. 2); the double-double operations follow Hida, Li and
Bailey's QD library (2001), in its accurate ("IEEE") addition.

    R(hi, lo)            a real double-double array; C(re, im) a complex one (two R)
    add, sub, mul, mul_d, div, neg, dsum (pairwise sum along an axis)
    cadd, csub, cmul, rcmul (real times complex), cmul_c (complex times complex double), cdiv, cmatmul (real matrix
    times complex matrix)

    python3 ddnp.py selftest [--mutate=twoprod]       each operation against mpmath at 60 digits
"""
import sys

import numpy as np

SPLIT = 134217729.0          # 2^27 + 1
MUTATE = None                # selftest control: 'twoprod' drops the product's error term


class R:
    __slots__ = ("h", "l")

    def __init__(self, h, l=None):
        self.h = np.asarray(h, dtype=float)
        self.l = np.zeros_like(self.h) if l is None else np.asarray(l, dtype=float)

    def __getitem__(self, k):
        return R(self.h[k], self.l[k])

    def __setitem__(self, k, v):
        self.h[k] = v.h
        self.l[k] = v.l

    @property
    def shape(self):
        return self.h.shape

    def copy(self):
        return R(self.h.copy(), self.l.copy())

    def reshape(self, *s):
        return R(self.h.reshape(*s), self.l.reshape(*s))

    @property
    def T(self):
        return R(self.h.T, self.l.T)


class C:
    __slots__ = ("re", "im")

    def __init__(self, re, im=None):
        self.re = re
        self.im = R(np.zeros_like(re.h)) if im is None else im

    def __getitem__(self, k):
        return C(self.re[k], self.im[k])

    def __setitem__(self, k, v):
        self.re[k] = v.re
        self.im[k] = v.im

    @property
    def shape(self):
        return self.re.shape

    def copy(self):
        return C(self.re.copy(), self.im.copy())

    def reshape(self, *s):
        return C(self.re.reshape(*s), self.im.reshape(*s))

    @property
    def T(self):
        return C(self.re.T, self.im.T)

    def to_complex(self):
        return (self.re.h + self.re.l) + 1j * (self.im.h + self.im.l)


def zeros(shape):
    return R(np.zeros(shape), np.zeros(shape))


def czeros(shape):
    return C(zeros(shape), zeros(shape))


def from_complex(z):
    z = np.asarray(z, dtype=complex)
    return C(R(z.real.copy()), R(z.imag.copy()))


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def qts(a, b):
    s = a + b
    return s, b - (s - a)


def split(a):
    t = SPLIT * a
    hi = t - (t - a)
    return hi, a - hi


def two_prod(a, b):
    p = a * b
    if MUTATE == "twoprod":
        return p, 0.0 * p
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def add(a, b):
    s, e = two_sum(a.h, b.h)
    t, f = two_sum(a.l, b.l)
    e = e + t
    s, e = qts(s, e)
    e = e + f
    s, e = qts(s, e)
    return R(s, e)


def neg(a):
    return R(-a.h, -a.l)


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    p, e = two_prod(a.h, b.h)
    e = e + (a.h * b.l + a.l * b.h)
    p, e = qts(p, e)
    return R(p, e)


def mul_d(a, d):
    """a (double-double) times d (float64 array or scalar)"""
    p, e = two_prod(a.h, d)
    e = e + a.l * d
    p, e = qts(p, e)
    return R(p, e)


def add_d(a, d):
    s, e = two_sum(a.h, d)
    e = e + a.l
    s, e = qts(s, e)
    return R(s, e)


def div(a, b):
    q1 = a.h / b.h
    r = sub(a, mul_d(b, q1))
    q2 = r.h / b.h
    r = sub(r, mul_d(b, q2))
    q3 = r.h / b.h
    q1, q2 = qts(q1, q2)
    return add(R(q1, q2), R(q3))


def dsum(a, axis=0):
    """pairwise double-double sum along an axis"""
    h = np.moveaxis(a.h, axis, 0)
    l = np.moveaxis(a.l, axis, 0)
    x = R(h, l)
    while x.h.shape[0] > 1:
        n = x.h.shape[0]
        if n % 2:
            x = R(np.concatenate([x.h, np.zeros_like(x.h[:1])]), np.concatenate([x.l, np.zeros_like(x.l[:1])]))
            n += 1
        x = add(x[0:n:2], x[1:n:2])
    return x[0]


# ---------------------------------------------------------------- complex
def cadd(a, b):
    return C(add(a.re, b.re), add(a.im, b.im))


def csub(a, b):
    return C(sub(a.re, b.re), sub(a.im, b.im))


def cneg(a):
    return C(neg(a.re), neg(a.im))


def cmul(a, b):
    return C(sub(mul(a.re, b.re), mul(a.im, b.im)), add(mul(a.re, b.im), mul(a.im, b.re)))


def rcmul(r, c):
    """real double-double r times complex double-double c (broadcasting)"""
    return C(mul(r, c.re), mul(r, c.im))


def cmul_d(c, d):
    """complex double-double c times real float64 d"""
    return C(mul_d(c.re, d), mul_d(c.im, d))


def cmul_c(c, z):
    """complex double-double c times a complex double scalar z (exact double parts)"""
    zr, zi = float(np.real(z)), float(np.imag(z))
    return C(sub(mul_d(c.re, zr), mul_d(c.im, zi)), add(mul_d(c.re, zi), mul_d(c.im, zr)))


def cabs2(a):
    return add(mul(a.re, a.re), mul(a.im, a.im))


def cdiv(a, b):
    n = cabs2(b)
    t = cmul(a, C(b.re, neg(b.im)))
    return C(div(t.re, n), div(t.im, n))


def rdiv_c(c, r):
    """complex c divided by real r"""
    return C(div(c.re, r), div(c.im, r))


def cdsum(a, axis=0):
    return C(dsum(a.re, axis), dsum(a.im, axis))


def matmul(A, Y):
    """real double-double A (n x m) times real double-double Y (m x k)"""
    return dsum(mul(R(A.h[:, :, None], A.l[:, :, None]), R(Y.h[None, :, :], Y.l[None, :, :])), axis=1)


def cmatmul(A, Y):
    """real double-double matrix A (n x m) times complex double-double Y (m x k); Y may be (m,)"""
    vec = Y.re.h.ndim == 1
    if vec:
        Y = Y.reshape(-1, 1)
    out = C(matmul(A, Y.re), matmul(A, Y.im))
    return out.reshape(-1) if vec else out


def maxabs(c):
    return float(np.max(np.abs(c.re.h) + np.abs(c.im.h)))


# ---------------------------------------------------------------- selftest
def selftest():
    import mpmath as mp
    mp.mp.dps = 60
    rng = np.random.default_rng(7)
    n = 200
    # inputs with full double-double content: hi + lo, lo ~ 1e-17 hi, across magnitudes
    def rnd():
        h = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)
        l = h * rng.standard_normal(n) * 1e-17
        h, l = qts(h, l)
        return R(h, l)
    to_mp = lambda x, i: mp.mpf(float(x.h[i])) + mp.mpf(float(x.l[i]))
    a, b = rnd(), rnd()
    worst = {}

    def rel(name, got, ref):
        e = max(abs((to_mp(got, i) - ref[i]) / ref[i]) for i in range(n))
        worst[name] = float(e)
    rel("add", add(a, b), [to_mp(a, i) + to_mp(b, i) for i in range(n)])
    rel("mul", mul(a, b), [to_mp(a, i) * to_mp(b, i) for i in range(n)])
    rel("div", div(a, b), [to_mp(a, i) / to_mp(b, i) for i in range(n)])
    rel("mul_d", mul_d(a, b.h), [to_mp(a, i) * mp.mpf(float(b.h[i])) for i in range(n)])
    # sums of same-sign terms (cancellation is a property of the data, not the sum)
    m = 37
    X = R(np.abs(rng.standard_normal((m, n))), np.abs(rng.standard_normal((m, n))) * 1e-17)
    s = dsum(X, 0)
    ref = [sum(mp.mpf(float(X.h[j, i])) + mp.mpf(float(X.l[j, i])) for j in range(m)) for i in range(n)]
    rel("dsum", s, ref)
    # complex matmul against mpmath
    k = 6
    A = R(rng.standard_normal((k, k)), rng.standard_normal((k, k)) * 1e-17)
    Y = C(R(rng.standard_normal((k, 3)), rng.standard_normal((k, 3)) * 1e-17),
          R(rng.standard_normal((k, 3)), rng.standard_normal((k, 3)) * 1e-17))
    P = cmatmul(A, Y)
    mA = mp.matrix([[mp.mpf(float(A.h[i, j])) + mp.mpf(float(A.l[i, j])) for j in range(k)] for i in range(k)])
    mY = mp.matrix([[mp.mpc(mp.mpf(float(Y.re.h[i, j])) + mp.mpf(float(Y.re.l[i, j])),
                            mp.mpf(float(Y.im.h[i, j])) + mp.mpf(float(Y.im.l[i, j]))) for j in range(3)]
                    for i in range(k)])
    mP = mA * mY
    e = max(abs(mp.mpc(mp.mpf(float(P.re.h[i, j])) + mp.mpf(float(P.re.l[i, j])),
                       mp.mpf(float(P.im.h[i, j])) + mp.mpf(float(P.im.l[i, j]))) - mP[i, j]) /
            mp.norm(mP) for i in range(k) for j in range(3))
    worst["cmatmul"] = float(e)
    za = C(rnd(), rnd()); zb = C(rnd(), rnd())
    q = cdiv(za, zb)
    e = max(abs((mp.mpc(to_mp(q.re, i), to_mp(q.im, i)) - mp.mpc(to_mp(za.re, i), to_mp(za.im, i)) /
                 mp.mpc(to_mp(zb.re, i), to_mp(zb.im, i))) / (mp.mpc(to_mp(za.re, i), to_mp(za.im, i)) /
                                                              mp.mpc(to_mp(zb.re, i), to_mp(zb.im, i))))
            for i in range(n))
    worst["cdiv"] = float(e)
    ok = all(v < 1e-30 for v in worst.values())
    print("ddnp selftest (largest relative error against mpmath, limit 1e-30):",
          {k: "%.1e" % v for k, v in worst.items()}, "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        if any(a == "--mutate=twoprod" for a in sys.argv):
            MUTATE = "twoprod"
        sys.exit(0 if selftest() else 1)
    raise SystemExit(__doc__)
