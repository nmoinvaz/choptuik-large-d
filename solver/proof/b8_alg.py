#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, Lemma W'): the compact bound algebra on a family of regions.

Each region r has a box: X = 1 + |x| in [Xlo, Xhi] (Xhi = None: unbounded, the far region), theta = |x|/X in
[thlo, thhi], and rho = X^2/n <= rho_hi (for every n >= n_0).  An upper bound is kept per region as a dict
key -> (a, c), meaning sum over keys of c X^a theta^b n^e with key = (b, e):
  - two terms with the same key are merged into the larger exponent a, the other coefficient multiplied by
    Xlo^(a' - a) <= 1 (valid since X >= Xlo >= 1, exact at X = Xlo);
  - on the far region theta is replaced at once by its bound (thhi^b, b >= 0; thlo^b, b < 0) and b = 0 in the key.
Lower bounds (divisors) are one term per region; on the far region theta is replaced by thlo^b (b >= 0) or thhi^b.
The sup of a term over a region for every n >= n_0: X^a <= Xhi^a (a > 0, Xhi finite) or Xlo^a (a <= 0);
theta^b <= thhi^b (b >= 0) or thlo^b (b < 0, thlo > 0); n^e <= n_0^e (e <= 0).  On the far region the better of
Xlo^a n_0^e (a, e <= 0) and Xlo^(a+2e) rho_hi^(-e) (a + 2e <= 0, e <= 0; X^2 = n rho <= n rho_hi).  Anything else
is reported as unbounded.  Every constant is a python-flint ball; only upper (lower) ends are used.
"""
from flint import arb

REG = []          # region names, set by configure()
BOX = {}          # region -> dict(Xlo, Xhi, thlo, thhi, rho, far); "n0" -> n_0


def configure(boxes, n0):
    REG.clear()
    REG.extend(boxes.keys())
    BOX.clear()
    BOX.update(boxes)
    BOX["n0"] = n0


def up(x):
    return arb(arb(x).upper())


def lo(x):
    return arb(arb(x).lower())


def _merge(r, d, key, a, c):
    if key not in d:
        d[key] = (a, c)
        return
    a0, c0 = d[key]
    Xlo = BOX[r]["Xlo"]
    if a >= a0:
        d[key] = (a, up(c + c0 * Xlo ** (a0 - a)))
    else:
        d[key] = (a0, up(c0 + c * Xlo ** (a - a0)))


def _theta_far(r, eT, lower=False):
    bx = BOX[r]
    if eT == 0:
        return arb(1)
    if lower:
        return bx["thlo"] ** eT if eT > 0 else bx["thhi"] ** eT
    return bx["thhi"] ** eT if eT > 0 else bx["thlo"] ** eT


class B:
    def __init__(self, t=None):
        self.t = t if t is not None else {r: {} for r in REG}

    @staticmethod
    def mono(c, eX=0, eT=0, eR=0, eN=0):
        b = B()
        a, e = eX + 2 * eR, eN - eR
        for r in REG:
            cr = up(abs(arb(c[r] if isinstance(c, dict) else c)))
            if BOX[r]["far"]:
                _merge(r, b.t[r], (0, e), a, up(cr * _theta_far(r, eT)))
            else:
                _merge(r, b.t[r], (eT, e), a, cr)
        return b

    def __add__(self, o):
        out = B({r: dict(self.t[r]) for r in REG})
        for r in REG:
            for k, (a, c) in o.t[r].items():
                _merge(r, out.t[r], k, a, c)
        return out

    def __mul__(self, o):
        if not isinstance(o, B):
            s = abs(arb(o))
            return B({r: {k: (a, up(c * s)) for k, (a, c) in self.t[r].items()} for r in REG})
        out = B()
        for r in REG:
            for k1, (a1, c1) in self.t[r].items():
                for k2, (a2, c2) in o.t[r].items():
                    _merge(r, out.t[r], (k1[0] + k2[0], k1[1] + k2[1]), a1 + a2, up(c1 * c2))
        return out

    __rmul__ = __mul__


def lo_mono(c, eX=0, eT=0, eN=0):
    b = B()
    for r in REG:
        cr = lo(c[r] if isinstance(c, dict) else c)
        if BOX[r]["far"]:
            b.t[r][(0, eN)] = (eX, lo(cr * _theta_far(r, eT, lower=True)))
        else:
            b.t[r][(eT, eN)] = (eX, cr)
    return b


def term_sup(r, key, a, c):
    b, e = key
    bx = BOX[r]
    n0 = BOX["n0"]
    if b >= 0:
        th = bx["thhi"] ** b
    elif bx["thlo"] > 0:
        th = bx["thlo"] ** b
    else:
        return None
    if not bx["far"]:
        if e > 0:
            return None
        xf = bx["Xhi"] ** a if a > 0 else bx["Xlo"] ** a
        return up(c * th * xf * n0 ** e)
    best = None
    if a <= 0 and e <= 0:
        best = up(c * th * bx["Xlo"] ** a * n0 ** e)
    if a + 2 * e <= 0 and e <= 0:
        v = up(c * th * bx["Xlo"] ** (a + 2 * e) * bx["rho"] ** (-e))
        best = v if best is None or v < best else best
    return best


def bsup_r(b, r):
    s = arb(0)
    for k, (a, c) in b.t[r].items():
        v = term_sup(r, k, a, c)
        if v is None:
            raise ValueError(f"unbounded term {k}, X^{a}, on region {r}")
        s = up(s + v)
    return s


def bsup(b, regions):
    return max((bsup_r(b, r) for r in regions), key=lambda v: float(v))


def inv_lo(lob):
    out = B()
    for r in REG:
        (k, (a, c)), = lob.t[r].items()
        out.t[r][(-k[0], -k[1])] = (-a, up(1 / c))
    return out


def scale_lo(l1, l2):
    out = B()
    for r in REG:
        (k1, (a1, c1)), = l1.t[r].items()
        (k2, (a2, c2)), = l2.t[r].items()
        out.t[r][(k1[0] + k2[0], k1[1] + k2[1])] = (a1 + a2, lo(c1 * c2))
    return out


class Q:
    """|f| <= v, |x f'| <= d, |x (x f')'| <= d2 (None if not tracked); lo: one-term lower bound of |f|, or for an
    operator I + small the bound ||f^-1|| <= 1/lo"""

    def __init__(self, v, d=None, d2=None, lo=None):
        self.v, self.d, self.d2, self.lo = v, d, d2, lo

    def __add__(self, o):
        return Q(self.v + o.v, _add(self.d, o.d), _add(self.d2, o.d2))

    def __mul__(self, o):
        if not isinstance(o, Q):
            return Q(self.v * o, None if self.d is None else self.d * o, None if self.d2 is None else self.d2 * o)
        d = None if (self.d is None or o.d is None) else self.d * o.v + self.v * o.d
        d2 = None
        if None not in (self.d, o.d, self.d2, o.d2):
            d2 = self.d2 * o.v + self.d * o.d * arb(2) + self.v * o.d2
        return Q(self.v * o.v, d, d2)

    __rmul__ = __mul__


def _add(a, b):
    return None if (a is None or b is None) else a + b


def const(c):
    return Q(B.mono(c), B(), B())


def qinv(g):
    il = inv_lo(g.lo)
    d = None if g.d is None else g.d * il * il
    d2 = None
    if g.d is not None and g.d2 is not None:
        d2 = g.d2 * il * il + g.d * g.d * il * il * il * arb(2)
    return Q(il, d, d2)


def lower_one_minus(b):
    out = B()
    for r in REG:
        s = bsup_r(b, r)
        v = lo(1 - s)
        if not v > 0:
            raise ValueError(f"1 - {float(s):.3g} not positive on {r}")
        out.t[r][(0, 0)] = (0, v)
    return out


def from_desc(d, dd=None, dd2=None):
    """theta^m0 X^w sum_j C_j rho^j per region (Lemma W' with the region discounts), one term rho^jmin"""
    def mk(dsc):
        if dsc is None:
            return None
        m0, w = dsc["m0"], dsc["w"]
        jmin = min(j for j, _ in dsc["coef"])
        cr = {}
        for r in REG:
            bx = BOX[r]
            s = arb(0)
            for (j, m), c in dsc["coef"].items():
                s += c * bx["thhi"] ** (m - m0) * bx["Xlo"] ** (m - 2 * j - w) * bx["rho"] ** (j - jmin)
            cr[r] = up(s)
        return B.mono(cr, eX=w, eT=m0, eR=jmin)
    return Q(mk(d), mk(dd), mk(dd2))
