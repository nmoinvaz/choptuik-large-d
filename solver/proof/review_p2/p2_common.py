"""Shared helpers of the P2 review scripts (independent of solver/proof's machinery: mpmath interval arithmetic,
mpmath.iv, instead of python-flint's arb).

Only parsing of the certificates' saved outputs is done here; nothing from solver/proof is imported.

mpmath.iv pitfalls handled here: x.a and x.b are degenerate intervals, and comparisons of intervals return None when
undecided. Every decision below is therefore made on exact endpoints (mp.mpf) obtained with lo()/hi().
"""
import json
import os
import re

from mpmath import iv, mp

iv.prec = 256
mp.prec = 256

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)          # solver/proof (data files only)

_BALL = re.compile(r"^\[\s*([-+0-9.eE]*)\s*\+/-\s*([0-9.eE+-]+)\s*\]$")


def lo(x):
    """exact lower endpoint (mp.mpf) of an interval or number"""
    return mp.make_mpf(iv.mpf(x)._mpi_[0])


def hi(x):
    """exact upper endpoint (mp.mpf)"""
    return mp.make_mpf(iv.mpf(x)._mpi_[1])


def I(a, b=None):
    """the interval [a, b] (or the point a) from mp.mpf / strings, exactly"""
    if b is None:
        return iv.mpf(a)
    return iv.mpf([a, b])


def ivball(s):
    """an arb string '[m +/- r]', '[+/- r]' or a plain decimal -> an mpmath interval containing it (outward)"""
    s = str(s).strip()
    m = _BALL.match(s)
    if m:
        mid = m.group(1)
        rad = hi(iv.mpf(m.group(2)))
        c = iv.mpf(mid) if mid not in ("", "+", "-") else iv.mpf(0)
        return iv.mpf([lo(c - rad), hi(c + rad)])
    if s.startswith("["):
        raise ValueError(f"cannot parse ball {s!r}")
    return iv.mpf(s)


def ivup(x):
    """[hi(x), hi(x)]"""
    return iv.mpf(hi(x))


def absup(x):
    """upper bound (mp.mpf) of |x| for a real interval"""
    return max(abs(lo(x)), abs(hi(x)))


def abslo(x):
    """lower bound (mp.mpf) of |x| for a real interval (0 if it contains 0)"""
    a, b = lo(x), hi(x)
    if a <= 0 <= b:
        return mp.mpf(0)
    return min(abs(a), abs(b))


def sym(r):
    """the interval [-r, r] for r >= 0 (interval or number)"""
    r = hi(r)
    return iv.mpf([-r, r])


class C:
    """a rectangular complex interval (re, im both mpmath intervals)"""

    def __init__(self, re, im=0):
        self.re, self.im = iv.mpf(re), iv.mpf(im)

    @staticmethod
    def parse(pair):
        return C(ivball(pair[0]), ivball(pair[1]))

    def __add__(self, o):
        o = cz(o)
        return C(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __sub__(self, o):
        o = cz(o)
        return C(self.re - o.re, self.im - o.im)

    def __rsub__(self, o):
        return cz(o) - self

    def __neg__(self):
        return C(-self.re, -self.im)

    def __mul__(self, o):
        o = cz(o)
        return C(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = cz(o)
        d = o.re * o.re + o.im * o.im
        if lo(d) <= 0:
            raise ZeroDivisionError("complex interval division by a set containing 0")
        return C((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)

    def conj(self):
        return C(self.re, -self.im)

    def absup(self):
        """upper bound (mp.mpf) of the modulus"""
        return hi(iv.sqrt(iv.mpf(absup(self.re)) ** 2 + iv.mpf(absup(self.im)) ** 2))

    def abslo(self):
        """lower bound (mp.mpf) of the modulus"""
        return lo(iv.sqrt(iv.mpf(abslo(self.re)) ** 2 + iv.mpf(abslo(self.im)) ** 2))

    def __repr__(self):
        return f"C({self.re}, {self.im})"


def cz(o):
    if isinstance(o, C):
        return o
    return C(o, 0)


def cexp(z):
    """exp of a complex interval"""
    z = cz(z)
    e = iv.exp(z.re)
    return C(e * iv.cos(z.im), e * iv.sin(z.im))


def load(name):
    with open(os.path.join(PROOF, name)) as fh:
        return json.load(fh)
