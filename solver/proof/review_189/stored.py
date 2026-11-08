"""Exact readers for the stored certificate files (#189 review), in rational arithmetic (fractions.Fraction).

No python-flint here. A stored real is one of
  * an arb string "[m +/- r]", "[+/- r]", or a bare decimal "m" (python-flint's arb.str with radius=True, which prints an
    interval containing the ball, so [m - r, m + r] read exactly encloses it);
  * a pair [mid, rad] of decimal strings (lemma6_jump_M.enc_arb);
  * a pair [[man, exp], [man, exp]] of binary mantissa and exponent (review_39/dump_inputs.py).
Each is returned as an exact interval (lo, hi) of Fractions.
"""
from fractions import Fraction
import math


def frac_dec(s):
    return Fraction(s.strip())


def arb_str(s):
    """an arb string -> (lo, hi)"""
    s = s.strip()
    if s.startswith("["):
        body = s[1:-1].strip()
        if body.startswith("+/-"):
            r = frac_dec(body[3:])
            return (-r, r)
        m, r = body.split("+/-")
        m, r = frac_dec(m), frac_dec(r)
        return (m - r, m + r)
    m = frac_dec(s)
    return (m, m)


def manexp(p):
    """[[man, exp], [man, exp]] (mid, rad) -> (lo, hi)"""
    (mm, me), (rm, re_) = p
    m = Fraction(int(mm)) * (Fraction(2) ** int(me))
    r = Fraction(int(rm)) * (Fraction(2) ** int(re_))
    return (m - r, m + r)


def midrad(p):
    """[mid, rad] decimal strings -> (lo, hi)"""
    m, r = frac_dec(p[0]), frac_dec(p[1])
    return (m - r, m + r)


def any_real(p):
    if isinstance(p, str):
        return arb_str(p)
    if isinstance(p[0], list):
        return manexp(p)
    return midrad(p)


# ---------------------------------------------------------------- exact interval arithmetic (Fractions)

def iadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def imul(a, b):
    ps = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(ps), max(ps))


def ihull(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]))


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def rad(a):
    return (a[1] - a[0]) / 2


def mid(a):
    return (a[1] + a[0]) / 2


# ---------------------------------------------------------------- decimal output with directed rounding

def dec_down(x, sig=70):
    """a decimal string <= x with about sig significant digits"""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(float(x)))) if float(x) != 0 else -400
    k = sig - e
    v = x * Fraction(10) ** k
    n = math.floor(v)
    return f"{n}e{-k}"


def dec_up(x, sig=70):
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(float(x)))) if float(x) != 0 else -400
    k = sig - e
    v = x * Fraction(10) ** k
    n = math.ceil(v)
    return f"{n}e{-k}"


def iv_line(a):
    return f"{dec_down(a[0])} {dec_up(a[1])}"
