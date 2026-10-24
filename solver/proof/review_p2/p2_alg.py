"""P2 review: a bound algebra for the independent Lemma LC check (p2_lc.py), written from B8_constants.md sections 3-5
and B_chain.md section 13, not from b8_alg.py.

Numbers. Upper bounds are raw mpmath floats rounded upward (mpmath.libmp, round_ceiling), lower bounds rounded
downward; transcendental constants come from mpmath.iv intervals (outward). Every number R carries a sparse gradient
g (index -> nonnegative upper bound): the derivative of the majorant along a perturbation of the class constants.
The Lipschitz columns of Lemma LC are these gradients (forward differentiation of the majorant), instead of the
secants of b8m_lc.py: by the product rule |fg - f~g~| <= |f||g - g~| + |g~||f - f~| and
||g^-1 - g~^-1|| <= ||g^-1|| ||g~^-1|| ||g - g~||, the difference of a node is bounded by the derivative of its
majorant in the direction of the input differences (B_chain.md 13.4); no convexity is needed.

Bounds. On each region an upper bound is a sum of terms c X^a theta^b n^e (X = 1 + |x|, theta = |x|/X), kept in a
dict (b, e) -> (a, c); two terms with the same (b, e) are merged into the larger a, the other coefficient times
Xlo^(a' - a) <= 1. On the far region theta is replaced by its bound at once. The supremum of a term over a region for
every n >= n0: X^a <= Xhi^a or Xlo^a, theta^b <= thhi^b or thlo^b, n^e <= n0^e (e <= 0); on the far region the better
of Xlo^a n0^e (a, e <= 0) and Xlo^(a + 2e) rho_max^(-e) (a + 2e <= 0, e <= 0; X^2 = n rho).
"""
from mpmath import iv, mp
from mpmath.libmp import (from_int, mpf_add, mpf_div, mpf_mul, mpf_sub, mpf_lt, round_ceiling as UP_,
                          round_floor as DN_, fzero, fone)

PREC = 120
iv.prec = 160


def up_iv(x):
    """raw upper end of an interval/number"""
    return iv.mpf(x)._mpi_[1]


def lo_iv(x):
    return iv.mpf(x)._mpi_[0]


def r2iv(r):
    return iv.mpf(mp.make_mpf(r))


def fl(r):
    return float(mp.make_mpf(r))


def add_u(a, b):
    return mpf_add(a, b, PREC, UP_)


def mul_u(a, b):
    return mpf_mul(a, b, PREC, UP_)


def div_u(a, b):
    return mpf_div(a, b, PREC, UP_)


def sub_d(a, b):
    return mpf_sub(a, b, PREC, DN_)


def mul_d(a, b):
    return mpf_mul(a, b, PREC, DN_)


def div_d(a, b):
    return mpf_div(a, b, PREC, DN_)


def pow_u(x, k):
    """x^k for x > 0 raw (upper end), k integer; for k < 0 uses a lower bound of x^|k|"""
    if k == 0:
        return fone
    if k > 0:
        r = fone
        for _ in range(k):
            r = mul_u(r, x)
        return r
    r = fone
    for _ in range(-k):
        r = mul_d(r, x)
    return div_u(fone, r)


def pow_d(x, k):
    if k == 0:
        return fone
    if k > 0:
        r = fone
        for _ in range(k):
            r = mul_d(r, x)
        return r
    r = fone
    for _ in range(-k):
        r = mul_u(r, x)
    return div_d(fone, r)


def gt(a, b):
    return mpf_lt(b, a)


def rmax(a, b):
    return a if gt(a, b) else b


class R:
    """a nonnegative upper bound with a sparse nonnegative gradient"""
    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v = v
        self.g = g if g is not None else {}

    @staticmethod
    def c(x):
        return R(up_iv(x))

    def __add__(self, o):
        g = dict(self.g)
        for i, u in o.g.items():
            g[i] = add_u(g[i], u) if i in g else u
        return R(add_u(self.v, o.v), g)

    def __mul__(self, o):
        if not isinstance(o, R):
            o = R(o)
        g = {i: mul_u(u, o.v) for i, u in self.g.items()}
        for i, u in o.g.items():
            t = mul_u(self.v, u)
            g[i] = add_u(g[i], t) if i in g else t
        return R(mul_u(self.v, o.v), g)

    def scale(self, s):
        """times a raw nonnegative constant"""
        return R(mul_u(self.v, s), {i: mul_u(u, s) for i, u in self.g.items()})

    def f(self):
        return fl(self.v)


class Lo:
    """a positive lower bound l, with the rate g at which it may decrease along the perturbation"""
    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v = v
        self.g = g if g is not None else {}

    def minus(self, s):
        """l - s for an upper bound s (R)"""
        g = dict(self.g)
        for i, u in s.g.items():
            g[i] = add_u(g[i], u) if i in g else u
        return Lo(sub_d(self.v, s.v), g)

    def times_one_minus(self, s):
        """l (1 - s)"""
        v = mul_d(self.v, sub_d(fone, s.v))
        g = {i: mul_u(u, sub_d(fone, fzero)) for i, u in self.g.items()}
        for i, u in s.g.items():
            t = mul_u(self.v, u)
            g[i] = add_u(g[i], t) if i in g else t
        return Lo(v, g)

    def times(self, o):
        v = mul_d(self.v, o.v)
        g = {i: mul_u(u, o.v) for i, u in self.g.items()}
        for i, u in o.g.items():
            t = mul_u(self.v, u)
            g[i] = add_u(g[i], t) if i in g else t
        return Lo(v, g)

    def positive(self):
        return gt(self.v, fzero)

    def inv(self):
        """upper bound of 1/l with gradient g/l^2"""
        if not self.positive():
            raise ValueError("lower bound not positive")
        l2 = mul_d(self.v, self.v)
        return R(div_u(fone, self.v), {i: div_u(u, l2) for i, u in self.g.items()})


ZR = R(fzero)


# ------------------------------------------------------------------------------------------------------ regions
class Reg:
    """region boxes: Xlo, Xhi (None: far), thlo, thhi, rho (sup of rho = X^2/n), far; and n0"""
    regs = {}
    n0 = None          # raw lower bound of n0 (n^e <= n0^e for e <= 0 uses it)
    n0_up = None

    @classmethod
    def set(cls, boxes, n0_iv):
        cls.regs = boxes
        cls.n0 = lo_iv(n0_iv)
        cls.n0_up = up_iv(n0_iv)


def theta_far_up(r, b):
    bx = Reg.regs[r]
    if b == 0:
        return fone
    return pow_u(bx["thhi"], b) if b > 0 else pow_u(bx["thlo"], b)


class Bd:
    """upper bound per region: {r: {(b, e): (a, R)}}"""
    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = t if t is not None else {r: {} for r in Reg.regs}

    @staticmethod
    def mono(c, a=0, b=0, e=0):
        """c X^a theta^b n^e on every region (c an R, a raw constant, or a dict region -> R)"""
        out = Bd()
        for r, bx in Reg.regs.items():
            cr = c[r] if isinstance(c, dict) else c
            if not isinstance(cr, R):
                cr = R(cr)
            if bx["far"]:
                _merge(r, out.t[r], (0, e), a, cr.scale(theta_far_up(r, b)))
            else:
                _merge(r, out.t[r], (b, e), a, cr)
        return out

    def __add__(self, o):
        out = Bd({r: dict(self.t[r]) for r in Reg.regs})
        for r in Reg.regs:
            for k, (a, c) in o.t[r].items():
                _merge(r, out.t[r], k, a, c)
        return out

    def __mul__(self, o):
        if isinstance(o, Bd):
            out = Bd()
            for r in Reg.regs:
                for (b1, e1), (a1, c1) in self.t[r].items():
                    for (b2, e2), (a2, c2) in o.t[r].items():
                        _merge(r, out.t[r], (b1 + b2, e1 + e2), a1 + a2, c1 * c2)
            return out
        if isinstance(o, R):
            return Bd({r: {k: (a, c * o) for k, (a, c) in self.t[r].items()} for r in Reg.regs})
        return Bd({r: {k: (a, c.scale(o)) for k, (a, c) in self.t[r].items()} for r in Reg.regs})

    __rmul__ = __mul__

    def only(self, regs):
        return Bd({r: (dict(self.t[r]) if r in regs else {}) for r in Reg.regs})


def _merge(r, d, key, a, c):
    if key not in d:
        d[key] = (a, c)
        return
    a0, c0 = d[key]
    Xlo = Reg.regs[r]["Xlo"]
    if a >= a0:
        d[key] = (a, c + c0.scale(pow_u(Xlo, a0 - a)))
    else:
        d[key] = (a0, c0 + c.scale(pow_u(Xlo, a - a0)))


def term_factor(r, key, a):
    """sup over region r (all n >= n0) of X^a theta^b n^e, or None if unbounded"""
    b, e = key
    bx = Reg.regs[r]
    if b >= 0:
        th = pow_u(bx["thhi"], b)
    elif gt(bx["thlo"], fzero):
        th = pow_u(bx["thlo"], b)
    else:
        return None
    if e > 0:
        return None
    ne = pow_u(Reg.n0, e)                      # n0^e with e <= 0 (n0 lower bound -> upper bound of n0^e)
    if not bx["far"]:
        xf = pow_u(bx["Xhi"], a) if a > 0 else pow_u(bx["Xlo"], a)
        return mul_u(mul_u(th, xf), ne)
    best = None
    if a <= 0:
        best = mul_u(mul_u(th, pow_u(bx["Xlo"], a)), ne)
    if a + 2 * e <= 0:
        v = mul_u(mul_u(th, pow_u(bx["Xlo"], a + 2 * e)), pow_u(bx["rho"], -e))
        best = v if best is None or gt(best, v) else best
    return best


def bsup(B, r):
    s = ZR
    for key, (a, c) in B.t[r].items():
        f = term_factor(r, key, a)
        if f is None:
            raise ValueError(f"unbounded term {key} X^{a} on {r}")
        s = s + c.scale(f)
    return s


def rmaxR(lst):
    """the candidate with the largest value (its gradient is that candidate's)"""
    best = None
    for x in lst:
        if best is None or gt(x.v, best.v):
            best = x
    return best


def lo_mono(c, a=0, b=0, e=0):
    """a one-term lower bound per region: {r: (Lo, a, b, e)} (theta bounded below on far regions)"""
    out = {}
    for r, bx in Reg.regs.items():
        cr = c[r] if isinstance(c, dict) else c
        if not isinstance(cr, Lo):
            cr = Lo(cr)
        if bx["far"] and b != 0:
            f = pow_d(bx["thlo"], b) if b > 0 else pow_d(bx["thhi"], b)
            cr = Lo(mul_d(cr.v, f), {i: mul_u(u, f) for i, u in cr.g.items()})
            out[r] = (cr, a, 0, e)
        else:
            out[r] = (cr, a, b, e)
    return out


def inv_lo(lob):
    """upper bound of 1/f from a one-term lower bound of |f| (or of 1/||f^-1||)"""
    out = Bd()
    for r, (cl, a, b, e) in lob.items():
        out.t[r][(-b, -e)] = (-a, cl.inv())
    return out


def lo_times(l1, l2):
    return {r: (l1[r][0].times(l2[r][0]), l1[r][1] + l2[r][1], l1[r][2] + l2[r][2], l1[r][3] + l2[r][3]) for r in l1}


class Q:
    """|f| <= v, |x f'| <= d, |x (x f')'| <= d2 (None if not tracked); lo a one-term lower bound (or None)"""
    __slots__ = ("v", "d", "d2", "lo")

    def __init__(self, v, d=None, d2=None, lo=None):
        self.v, self.d, self.d2, self.lo = v, d, d2, lo

    def __add__(self, o):
        return Q(self.v + o.v, _a(self.d, o.d), _a(self.d2, o.d2))

    def __mul__(self, o):
        if not isinstance(o, Q):
            return Q(self.v * o, None if self.d is None else self.d * o, None if self.d2 is None else self.d2 * o)
        d = None if (self.d is None or o.d is None) else self.d * o.v + self.v * o.d
        d2 = None
        if None not in (self.d, o.d, self.d2, o.d2):
            d2 = self.d2 * o.v + self.d * o.d * R.c(2) + self.v * o.d2
        return Q(self.v * o.v, d, d2)

    __rmul__ = __mul__


def _a(a, b):
    return None if (a is None or b is None) else a + b


def qinv(g):
    """g^-1 from g.lo: value 1/lo, |x (g^-1)'| <= |x g'|/lo^2, second derivative by the product rule"""
    il = inv_lo(g.lo)
    d = None if g.d is None else g.d * il * il
    d2 = None
    if g.d is not None and g.d2 is not None:
        d2 = g.d2 * il * il + g.d * g.d * il * il * il * R.c(2)
    return Q(il, d, d2)


def const(c):
    return Q(Bd.mono(R.c(c)), Bd(), Bd())


def one_plus(small):
    """the operator I + small (norm <= 1 + |small|), lower bound 1 - sup|small| per region"""
    q = const(1) + small
    lob = {}
    for r in Reg.regs:
        s = bsup(small.v, r)
        l = Lo(fone).minus(s)
        if not l.positive():
            raise ValueError(f"1 - small not positive on {r}")
        lob[r] = (l, 0, 0, 0)
    q.lo = lob
    return q


def from_desc(dsc, name=""):
    """Lemma W with the region discounts: theta^m0 X^w rho^jmin sum_jm c_jm thhi^(m - m0) Xlo^(m - 2j - w) rho_r^(j - jmin)
    as one monomial per region (X^(w + 2 jmin) theta^m0 n^-jmin)"""
    if dsc is None:
        return Bd()
    m0, w, jmin = dsc["m0"], dsc["w"], dsc["jmin"]
    cr = {}
    for r, bx in Reg.regs.items():
        s = fzero
        for (j, m), c in dsc["coef"].items():
            cc = up_iv(c)
            t = mul_u(mul_u(cc, pow_u(bx["thhi"], m - m0)), pow_u(bx["Xlo"], m - 2 * j - w))
            t = mul_u(t, pow_u(bx["rho"], j - jmin))
            s = add_u(s, t)
        cr[r] = R(s)
    return Bd.mono(cr, a=w + 2 * jmin, b=m0, e=-jmin)
