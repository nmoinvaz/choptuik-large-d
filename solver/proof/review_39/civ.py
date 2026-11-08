"""Rectangular complex intervals on mpmath's interval primitives (mpmath.libmp.libmpi), for the #39 review.

A real interval is an mpi pair (a, b) of raw mpf values; every operation rounds a down and b up at PREC bits
(mpi_add, mpi_mul, ... with an explicit precision; prec = 0 would be exact). A complex interval is (re, im), the
rectangle re x i im, which contains every exact value. No python-flint anywhere in this review's arithmetic.
"""
from mpmath.libmp import from_man_exp, from_int, from_str, mpf_pi, mpf_sqrt, mpf_add, mpf_mul, mpf_lt, mpf_le
from mpmath.libmp import round_floor, round_ceiling, fzero, finf, fninf, mpf_neg, mpf_abs, mpf_div, mpf_cmp
from mpmath.libmp import to_float
from mpmath.libmp.libmpi import mpi_add, mpi_sub, mpi_mul, mpi_div, mpi_neg, mpi_abs, mpi_cos_sin, mpi_sqrt
from mpmath.libmp.libmpi import mpi_pow_int  # noqa: F401

PREC = 128
Z0 = (fzero, fzero)                     # the real interval [0, 0]
CZ = (Z0, Z0)                           # the complex interval 0


def setprec(p):
    global PREC
    PREC = p


# ------------------------------------------------------------------ real intervals

def ri(x):
    """an exact real (int or (man, exp)) as a point interval"""
    if isinstance(x, int):
        v = from_int(x)
    else:
        v = from_man_exp(int(x[0]), int(x[1]))
    return (v, v)


def ri_dec(s):
    """a decimal string, rounded outward"""
    return (from_str(s, PREC, round_floor), from_str(s, PREC, round_ceiling))


def ri_ball(mid, rad):
    """[mid - rad, mid + rad] for exact (man, exp) pairs, rounded outward"""
    m = from_man_exp(int(mid[0]), int(mid[1]))
    r = from_man_exp(int(rad[0]), int(rad[1]))
    return (mpf_add(m, mpf_neg(r), PREC, round_floor), mpf_add(m, r, PREC, round_ceiling))


def radd(x, y):
    return mpi_add(x, y, PREC)


def rsub(x, y):
    return mpi_sub(x, y, PREC)


def rmul(x, y):
    return mpi_mul(x, y, PREC)


def rdiv(x, y):
    return mpi_div(x, y, PREC)


def rneg(x):
    return mpi_neg(x)


def rpi():
    return (mpf_pi(PREC, round_floor), mpf_pi(PREC, round_ceiling))


def rcos_sin(x):
    return mpi_cos_sin(x, PREC)


def rsqrt(x):
    return mpi_sqrt(x, PREC)


def rhull(x, y):
    a = x[0] if mpf_le(x[0], y[0]) else y[0]
    b = x[1] if mpf_le(y[1], x[1]) else y[1]
    return (a, b)


def rlt(x, y):
    """proven x < y"""
    return mpf_lt(x[1], y[0])


def rle(x, y):
    return mpf_le(x[1], y[0])


def rgt(x, y):
    return mpf_lt(y[1], x[0])


def rup(x):
    """the upper endpoint as a point interval"""
    return (x[1], x[1])


def rmaxup(xs):
    """a point interval >= every element of every interval in xs"""
    m = fninf
    for x in xs:
        if mpf_lt(m, x[1]):
            m = x[1]
    return (m, m)


def rinflate(x, e):
    """x + [-e, e] for a real e >= 0 given as an mpf (rounded outward)"""
    return (mpf_add(x[0], mpf_neg(e), PREC, round_floor), mpf_add(x[1], e, PREC, round_ceiling))


def rcontains0(x):
    return mpf_le(x[0], fzero) and mpf_le(fzero, x[1])


def rf(x):
    """float of the upper endpoint (for printing only)"""
    return to_float(x[1])


def rmag(x):
    """an interval [0, u] with u >= |t| for every t in x"""
    a = mpf_abs(x[0])
    b = mpf_abs(x[1])
    u = b if mpf_le(a, b) else a
    return (fzero, u)


# ------------------------------------------------------------------ complex intervals

def ci(re, im=Z0):
    return (re, im)


def cadd(x, y):
    return (mpi_add(x[0], y[0], PREC), mpi_add(x[1], y[1], PREC))


def csub(x, y):
    return (mpi_sub(x[0], y[0], PREC), mpi_sub(x[1], y[1], PREC))


def cneg(x):
    return (mpi_neg(x[0]), mpi_neg(x[1]))


def cmul(x, y):
    p = PREC
    xr, xi = x
    yr, yi = y
    return (mpi_sub(mpi_mul(xr, yr, p), mpi_mul(xi, yi, p), p), mpi_add(mpi_mul(xr, yi, p), mpi_mul(xi, yr, p), p))


def cmulr(x, r):
    """complex interval times real interval"""
    return (mpi_mul(x[0], r, PREC), mpi_mul(x[1], r, PREC))


def cconj(x):
    return (x[0], mpi_neg(x[1]))


def cinv(x):
    """1/x = conj(x)/|x|^2, |x|^2 > 0 required"""
    xr, xi = x
    d = mpi_add(mpi_mul(xr, xr, PREC), mpi_mul(xi, xi, PREC), PREC)
    if not mpf_lt(fzero, d[0]):
        raise ZeroDivisionError("complex interval contains 0")
    return (mpi_div(xr, d, PREC), mpi_div(mpi_neg(xi), d, PREC))


def cdiv(x, y):
    return cmul(x, cinv(y))


def cabs(x):
    """an interval containing |t| for every t in x"""
    xr, xi = x
    # |t|^2 over the rectangle: squares of the absolute intervals
    ar, ai = mpi_abs(xr), mpi_abs(xi)
    s = mpi_add(mpi_mul(ar, ar, PREC), mpi_mul(ai, ai, PREC), PREC)
    return mpi_sqrt(s, PREC)


def cabsup(x):
    """an upper bound of |t| over x, as a point interval"""
    a = cabs(x)
    return (a[1], a[1])


def cabslow(x):
    a = cabs(x)
    return (a[0], a[0])


def cinflate(x, e):
    """the rectangle x + [-e, e] + i[-e, e] (contains the closed disc of radius e around every point of x)"""
    return (rinflate(x[0], e), rinflate(x[1], e))


def chull(x, y):
    return (rhull(x[0], y[0]), rhull(x[1], y[1]))


def ccontains0(x):
    return rcontains0(x[0]) and rcontains0(x[1])


def cfrom_me(p):
    """a stored [[[mid man, exp], [rad man, exp]], [[...], [...]]] complex ball as a rectangle"""
    return (ri_ball(p[0][0], p[0][1]), ri_ball(p[1][0], p[1][1]))


def cexpi(phi):
    """e^{i phi} for a real interval phi"""
    c, s = mpi_cos_sin(phi, PREC)
    return (c, s)


def crad(x):
    """max half-width of the rectangle (float, for printing)"""
    return max(to_float(mpf_add(x[0][1], mpf_neg(x[0][0]))) / 2, to_float(mpf_add(x[1][1], mpf_neg(x[1][0]))) / 2)


def cmid_float(x):
    return complex((to_float(x[0][0]) + to_float(x[0][1])) / 2, (to_float(x[1][0]) + to_float(x[1][1])) / 2)
