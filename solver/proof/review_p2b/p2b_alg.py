#!/usr/bin/env python3
"""P2b review: the bound algebra used by p2b_lc.py and p2b_lcj.py, written from the statements (B8_constants.md 1.3,
3, 5; B_chain.md 13), not from solver/proof/b8_alg.py or b8m_bounds.py (not opened before the numbers were written
down). Arithmetic: mpmath interval arithmetic (iv), every quantity an upper bound carried as a point interval at the
upper endpoint of the interval operation that produced it (so each step rounds outward); lower bounds are taken from
lower endpoints. No python-flint.

Representation (own design). A bound on a cell is a Bd: a weight monomial W = X^a theta^b rho^e (X = 1 + |x|,
theta = |x|/X, rho = X^2/n) and a number v with |node| <= v W at every point of the cell and for every n >= n0. Sums
convert to a common weight with the cell's supremum of the weight ratio. Each Bd also carries forward-mode
derivatives d[i] = dv/dt_i with respect to the inputs (the band constants, scaled), which give the Lipschitz matrix
(B_chain.md 13.4) directly as derivatives of the majorant; no secant and no convexity argument is needed, because
every operation's derivative bound is itself a bound on the node's difference (product rule, inverse rule
|g^-1 - g~^-1| <= |g^-1||g~^-1||g - g~|, linearity of the integrals, maxima taken componentwise).

A Jet is (B0, B1, B2): bounds of a node and of its first two log-derivatives x d/dx (None when not available).
"""
from mpmath import iv, mp

iv.prec = 106
mp.prec = 106


def up(x):
    """point interval at the upper endpoint"""
    x = iv.mpf(x)
    return x.b


def lo(x):
    return iv.mpf(x).a


def mpup(x):
    return mp.make_mpf(iv.mpf(x)._mpi_[1])


def mplo(x):
    return mp.make_mpf(iv.mpf(x)._mpi_[0])


INF = None
ZEROI = iv.mpf(0)
ONEI = iv.mpf(1)

# ---------------------------------------------------------------- constants (B8_constants.md 1.3, B_chain.md 13)
K = 4
KAPPA = iv.mpf("0.3")
KAPPA1 = iv.mpf("0.6")
DELTA = iv.mpf("0.05")
HD = DELTA / 2                       # delta/2
ALPHA = iv.pi / 4 + KAPPA / 2


def _asin_enclosure(y):
    """[lo, hi] with sin(lo) <= y <= sin(hi) checked in interval arithmetic (sin increasing on [0, pi/2]);
    mpmath's iv has no asin"""
    mp.prec = 200
    c = mp.asin(mp.mpf(mpup(y)))
    a, b = c - mp.mpf(2) ** -90, c + mp.mpf(2) ** -90
    mp.prec = 106
    if not (mpup(iv.sin(iv.mpf(a))) <= mplo(y) and mpup(y) <= mplo(iv.sin(iv.mpf(b)))):
        raise ValueError("asin enclosure")
    return iv.mpf([a, b])


PHID = _asin_enclosure(HD)           # phi_delta = arcsin(delta/2)
SK = iv.sin(KAPPA)
SALPHA = iv.sin(ALPHA) / ALPHA       # s_alpha = sin(alpha)/alpha
R1, XF = iv.mpf("0.05"), iv.mpf(20)


class Setting:
    def __init__(self, ln_n0="73.89"):
        self.ln_n0 = ln_n0
        self.n0 = iv.exp(iv.mpf(ln_n0))
        self.n0lo = lo(self.n0)
        self.nu0 = up(1 / self.n0lo)                    # 1/n <= nu0
        # R'_n = (kappa1 n/(cos kappa ln n))^(1/2); rho_max = (1 + R'_n)^2/n, decreasing in n (Lemma D)
        n = self.n0lo
        Rp = iv.sqrt(KAPPA1 * n / (iv.cos(KAPPA) * iv.log(n)))
        self.Rp = Rp
        self.rho_max = up((1 + Rp) ** 2 / n)
        self.eps1 = up(1 / (self.n0lo - 1))           # 1/(n - 1)
        self.c1mnu = up(1 / (1 - self.nu0))            # 1/(1 - nu0) (eps1 <= nu c1mnu)


# ---------------------------------------------------------------- cells
class Cell:
    """a set of points x (all n >= n0): kind 'sector' (0 <= arg x <= alpha, r in [rlo, rhi], outside the open disc
    |x - 1| < delta/2), 'ray' (arg x = alpha), 'L' (|x - 1| <= delta/2, Im x >= 0), 'far' (sector, |x| >= 20;
    rhi may be None = infinity), 'real' (x real in [rlo, rhi], x > 1; for Lemma J)"""

    def __init__(self, st, band, kind, rlo, rhi):
        self.st, self.band, self.kind = st, band, kind
        rlo = iv.mpf(rlo)
        self.rlo = rlo
        self.rhi = None if rhi is None else iv.mpf(rhi)
        self.far = band == "F"
        if kind == "L":
            rlo, rhi = 1 - HD, 1 + HD
            self.rlo, self.rhi = rlo, rhi
        self.Xlo = lo(1 + self.rlo)
        self.Xhi = None if self.rhi is None else up(1 + self.rhi)
        self.thlo = lo(self.rlo / (1 + self.rlo))
        self.thhi = ONEI if self.rhi is None else up(self.rhi / (1 + self.rhi))
        if self.far:
            cap = st.rho_max
            if self.Xhi is not None:
                cap = min(mpup(cap), mpup(self.Xhi ** 2 * st.nu0))
                cap = iv.mpf(cap)
            self.rho_cap = cap
        else:
            self.rho_cap = up(self.Xhi ** 2 * st.nu0)
        self._ms = {}
        self.geom()

    # sup over the cell (and n >= n0) of X^da theta^db rho^de
    def msup(self, da, db, de):
        key = (da, db, de)
        if key in self._ms:
            return self._ms[key]
        if de < 0:
            raise ValueError(f"weight ratio rho^{de} unbounded as n -> infinity (cell {self.band}/{self.kind})")
        if db >= 0:
            th = self.thhi ** db
        else:
            if mplo(self.thlo) <= 0:
                raise ValueError(f"theta^{db} unbounded at x = 0")
            th = self.thlo ** db
        vals = []
        # nu-form: X^(da + 2 de) theta^db nu0^de, valid on every cell with X bounded
        if self.Xhi is not None:
            p = da + 2 * de
            Xp = self.Xhi ** p if p >= 0 else self.Xlo ** p
            vals.append(up(Xp * th * self.st.nu0 ** de))
        if self.far:
            if da > 0 and self.Xhi is None:
                if not vals:
                    raise ValueError(f"X^{da} unbounded on the far cell")
            else:
                Xp = ONEI if da == 0 else (self.Xhi ** da if da > 0 else self.Xlo ** da)
                vals.append(up(Xp * th * self.rho_cap ** de))
        if not vals:
            raise ValueError("no bound")
        v = min(vals, key=mpup)
        self._ms[key] = v
        return v

    def corner(self, w):
        """X^a theta^b rho^e at the cell's upper corner (X = Xhi, or 2 Xlo when unbounded; rho = rho_cap), a float
        used only to choose between valid representations"""
        X = mpup(self.Xhi) if self.Xhi is not None else 2 * mpup(self.Xlo)
        th = mpup(self.thhi)
        rho = mpup(self.rho_cap)
        return mp.mpf(X) ** w[0] * mp.mpf(th) ** w[1] * mp.mpf(rho) ** w[2]

    def geom(self):
        """inf and sup of |x^2 - 1|/X^2 and sup |x|^2/|x^2 - 1| over the cell. On sector and ray cells the radius
        range is split into sub-intervals (each with decoupled interval bounds) and the extreme is taken over them"""
        k, rlo, rhi = self.kind, self.rlo, self.rhi
        if k == "L":
            self.d2lo = None                         # not used: the bounded forms of B_chain.md 13.3
            self.d2hi = up(HD * (2 + HD) / self.Xlo ** 2)
            self.x2d = None
            self.xp1lo = lo(2 - HD)                  # |x + 1| >= 2 - delta/2
            self.xp1hi = up(2 + HD)
            return
        if self.far:
            # |x^2 - 1|/X^2 >= (r^2 - 1)/(1 + r)^2 = (r - 1)/(r + 1), increasing in r; |x|^2/|x^2 - 1| <= r^2/(r^2 - 1)
            self.d2lo = lo((rlo - 1) / (rlo + 1))
            self.d2hi = ONEI
            self.x2d = up(rlo ** 2 / (rlo ** 2 - 1))
            return
        nsub = 200
        a, b = mplo(rlo), mpup(rhi)
        if a > 0:
            pts = [mp.mpf(a) * (mp.mpf(b) / a) ** (mp.mpf(i) / nsub) for i in range(nsub + 1)]
        else:
            pts = [mp.mpf(b) * mp.mpf(i) / nsub for i in range(nsub + 1)]
        pts[0], pts[-1] = a, b
        d2lo = d2hi = x2d = None
        dabs = None
        Drmax = None
        for p, q in zip(pts[:-1], pts[1:]):
            r = iv.mpf([p, q])
            X = 1 + r
            if k == "ray":
                D = iv.sqrt(1 + 2 * r ** 2 * SK + r ** 4)        # |x^2 - 1| on the ray, exactly
                m, M = lo(D), up(D)
            elif k == "real":
                m, M = lo(r ** 2 - 1), up(r ** 2 - 1)
            else:
                # sector: |x^2 - 1| >= |r^2 - 1| (|x^2-1|^2 = r^4 - 2 r^2 cos 2phi + 1); outside the open disc,
                # with d = |x - 1| >= delta/2 and |x + 1|^2 = 2 r^2 + 2 - d^2 (parallelogram law),
                # |x^2 - 1| = d (2 r^2 + 2 - d^2)^(1/2), increasing in d while d^2 <= r^2 + 1 (true for
                # 0 <= arg x <= alpha < pi/2), so |x^2 - 1| >= (delta/2)(2 r^2 + 2 - delta^2/4)^(1/2)
                if q < 1:
                    m = lo(1 - r ** 2)
                elif p > 1:
                    m = lo(r ** 2 - 1)
                else:
                    m = ZEROI
                if p <= mpup(1 + HD) and q >= mplo(1 - HD):
                    m2 = lo(HD * iv.sqrt(2 * r ** 2 + 2 - HD ** 2))
                    if mplo(m2) > mplo(m):
                        m = m2
                M = up(r ** 2 + 1)
            if mplo(m) <= 0:
                raise ValueError(f"no lower bound of |x^2 - 1| on {self.kind} cell at r = {p}")
            v1, v2, v3 = lo(iv.mpf(m) / X ** 2), up(iv.mpf(M) / X ** 2), up(r ** 2 / iv.mpf(m))
            d2lo = v1 if d2lo is None or mplo(v1) < mplo(d2lo) else d2lo
            d2hi = v2 if d2hi is None or mpup(v2) > mpup(d2hi) else d2hi
            x2d = v3 if x2d is None or mpup(v3) > mpup(x2d) else x2d
            dabs = iv.mpf(m) if dabs is None or mplo(m) < mplo(dabs) else dabs
            Drmax = iv.mpf(M) if Drmax is None or mpup(M) > mpup(Drmax) else Drmax
        self.d2lo, self.d2hi, self.x2d = d2lo, d2hi, x2d
        self.dabs_lo = lo(dabs)
        if k == "ray":
            self.Dr = up(Drmax)

    def __repr__(self):
        return f"Cell({self.band},{self.kind},[{mp.nstr(mpup(self.rlo), 6)},{'inf' if self.rhi is None else mp.nstr(mpup(self.rhi), 6)}])"


# ---------------------------------------------------------------- bounds with weights and forward derivatives
class Bd:
    __slots__ = ("w", "v", "d")

    def __init__(self, w, v, d=None):
        self.w = w
        self.v = up(v)
        self.d = d or {}

    def __repr__(self):
        return f"Bd(w={self.w}, v={mp.nstr(mpup(self.v), 6)}, nd={len(self.d)})"


def bconst(c, w=(0, 0, 0)):
    return Bd(w, c)


# Derivative entries: normally numbers in the value's weight (Lemma LC's Lipschitz matrix). With OWNW = True (Lemma
# J's pointwise secant coefficients) each entry is a list of (weight, number) terms in its own weight, because the
# derivative of a node with respect to a pointwise perturbation scales differently in n from the node itself.
OWNW = False


def wadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def dzero(d):
    if OWNW:
        return all(mpup(v) == 0 for ts in d.values() for (_, v) in ts)
    return all(mpup(a) == 0 for a in d.values())


def pick_weight(cell, ws_vs):
    """choose a common weight for terms [(w, v)]: candidates each term's weight and (max a, min b, min e); the
    smallest bound at the cell's upper corner (any valid choice gives a valid bound). Returns (w, total, factors)."""
    cands = {(max(w[0] for w, _ in ws_vs), min(w[1] for w, _ in ws_vs), min(w[2] for w, _ in ws_vs))}
    cands.update(w for w, _ in ws_vs)
    best = None
    for w in cands:
        try:
            fs = [cell.msup(x[0] - w[0], x[1] - w[1], x[2] - w[2]) if x != w else ONEI for x, _ in ws_vs]
        except ValueError:
            continue
        tot = ZEROI
        for (_, v), f in zip(ws_vs, fs):
            tot = tot + v * f
        size = mpup(tot) * cell.corner(w)
        if best is None or size < best[3]:
            best = (w, up(tot), fs, size)
    if best is None:
        raise ValueError("no common weight")
    return best[0], best[1], best[2]


def bmul(x, y):
    if x is None or y is None:
        return None
    w = wadd(x.w, y.w)
    d = {}
    if OWNW:
        for i, ts in x.d.items():
            d[i] = [(wadd(y.w, tw), up(y.v * tv)) for tw, tv in ts]
        for i, ts in y.d.items():
            d[i] = d.get(i, []) + [(wadd(x.w, tw), up(x.v * tv)) for tw, tv in ts]
        return Bd(w, x.v * y.v, d)
    for i, a in x.d.items():
        d[i] = up(a * y.v)
    for i, a in y.d.items():
        d[i] = up(d[i] + a * x.v) if i in d else up(a * x.v)
    return Bd(w, x.v * y.v, d)


def bscale(x, c):
    if x is None:
        return None
    if OWNW:
        return Bd(x.w, x.v * c, {i: [(tw, up(tv * c)) for tw, tv in ts] for i, ts in x.d.items()})
    return Bd(x.w, x.v * c, {i: up(a * c) for i, a in x.d.items()})


def bto(x, cell, w):
    """the same bound in weight w (factor = sup of the weight ratio over the cell); own-weight derivative entries
    are left in their weights"""
    if x is None:
        return None
    if x.w == tuple(w):
        return x
    if mpup(x.v) == 0 and dzero(x.d):
        return Bd(tuple(w), ZEROI, dict(x.d))
    f = cell.msup(x.w[0] - w[0], x.w[1] - w[1], x.w[2] - w[2])
    if OWNW:
        return Bd(tuple(w), x.v * f, dict(x.d))
    return Bd(tuple(w), x.v * f, {i: up(a * f) for i, a in x.d.items()})


def badd(cell, *xs):
    """sum of bounds in one weight, chosen by pick_weight (the choice is made at the point of evaluation and the
    derivatives follow the same chain)"""
    xs = [x for x in xs if x is not None and not (mpup(x.v) == 0 and dzero(x.d))]
    if not xs:
        return Bd((0, 0, 0), ZEROI)
    if len(xs) == 1:
        return xs[0]
    w, tot, fs = pick_weight(cell, [(x.w, x.v) for x in xs])
    d = {}
    if OWNW:
        for x in xs:
            for i, ts in x.d.items():
                d[i] = d.get(i, []) + ts
        for i in d:
            if len(d[i]) > 1:
                tw, tv, _ = pick_weight(cell, d[i])
                d[i] = [(tw, tv)]
        return Bd(w, tot, d)
    for x, f in zip(xs, fs):
        for i, a2 in x.d.items():
            t = up(a2 * f)
            d[i] = up(d[i] + t) if i in d else t
    return Bd(w, tot, d)


def bnum(x, cell):
    """dimensionless number: sup over the cell of the bound (weight 1)"""
    return bto(x, cell, (0, 0, 0))


def binvlb(L, s, w=(0, 0, 0)):
    """1/(L - s) for a lower bound L (number) and a dimensionless Bd s; weight w attached"""
    den = lo(iv.mpf(lo(L)) - s.v)
    if mpup(den) <= 0:
        raise ValueError("inverse: lower bound lost")
    v = up(1 / iv.mpf(den))
    if OWNW:
        d = {i: [(wadd(tuple(w), tw), up(tv * v * v)) for tw, tv in ts] for i, ts in s.d.items()}
    else:
        d = {i: up(a * v * v) for i, a in s.d.items()}
    return Bd(tuple(w), v, d)


def dsup(x, cell, i, w=(0, 0, 0)):
    """OWNW: sup over the cell of the derivative entry i of x, in weight w"""
    tot = ZEROI
    for tw, tv in x.d.get(i, []):
        tot = tot + tv * cell.msup(tw[0] - w[0], tw[1] - w[1], tw[2] - w[2])
    return up(tot)


def bmax(*xs):
    """componentwise max (value and each derivative); same weight required"""
    if OWNW:
        raise ValueError("bmax is not used with own-weight derivatives")
    xs = [x for x in xs if x is not None]
    w = xs[0].w
    v = xs[0].v
    d = dict(xs[0].d)
    for x in xs[1:]:
        assert x.w == w
        if mpup(x.v) > mpup(v):
            v = x.v
        for i, a in x.d.items():
            if i not in d or mpup(a) > mpup(d[i]):
                d[i] = a
    return Bd(w, v, d)


# ---------------------------------------------------------------- jets
class Jet:
    __slots__ = ("c",)

    def __init__(self, b0, b1=None, b2=None):
        self.c = [b0, b1, b2]

    def __getitem__(self, k):
        return self.c[k]


def jadd(cell, *js):
    out = []
    for k in range(3):
        comps = [j[k] for j in js if j is not None]
        if any(c is None for c in comps) or not comps:
            out.append(None)
        else:
            out.append(badd(cell, *comps))
    return Jet(*out)


def jmul(cell, x, y):
    c0 = bmul(x[0], y[0])
    c1 = None
    if x[1] is not None and y[1] is not None:
        c1 = badd(cell, bmul(x[1], y[0]), bmul(x[0], y[1]))
    c2 = None
    if c1 is not None and x[2] is not None and y[2] is not None:
        c2 = badd(cell, bmul(x[2], y[0]), bscale(bmul(x[1], y[1]), 2), bmul(x[0], y[2]))
    return Jet(c0, c1, c2)


def jprod(cell, *js):
    out = js[0]
    for j in js[1:]:
        out = jmul(cell, out, j)
    return out


def jscale(x, c):
    return Jet(*[None if b is None else bscale(b, c) for b in x.c])


def jinv(cell, E, L=ONEI):
    """g = (L I + E)^-1 (dimensionless): |g| <= 1/(L - |E|), g' = -g E' g, g'' = -g'E'g - gE''g - gE'g' (x d/dx);
    E's components are converted to numbers (suprema over the cell)"""
    s = bnum(E[0], cell)
    g0 = binvlb(L, s)
    g1 = g2 = None
    if E[1] is not None:
        e1 = bnum(E[1], cell)
        g1 = bmul(bmul(g0, g0), e1)
        if E[2] is not None:
            e2 = bnum(E[2], cell)
            g2 = badd(cell, bscale(bmul(bmul(g0, g1), e1), 2), bmul(bmul(g0, g0), e2))
    return Jet(g0, g1, g2)


def jconst(c, w=(0, 0, 0)):
    """an x-independent quantity: derivatives 0"""
    return Jet(Bd(w, c), Bd(w, ZEROI), Bd(w, ZEROI))


def jmono(a, b, e, k1, k2, c=ONEI):
    """c X^a theta^b rho^e-type monomial in x with x d/dx factors k1, k2 (|x| = theta X: (1, 1, 0), 1, 1)"""
    return Jet(Bd((a, b, e), c), Bd((a, b, e), c * k1), Bd((a, b, e), c * k2))


# ---------------------------------------------------------------- known polynomials
class Poly:
    """a known polynomial quantity: per column (operators) or a single column (vectors/scalars), a list of
    (j, m, C) with |coef of eps^j x^m| <= C (summed over the rows/harmonics). Weight: X^a theta^b rho^e with
    b = min m, e = min j, a = max(m - 2j), so every term's ratio is bounded on every cell (including X -> infinity)."""

    def __init__(self, name, cols, mult=None):
        self.name = name
        self.cols = [[(j, m, iv.mpf(C) * (1 if mult is None else mult(m))) for (j, m, C) in col] for col in cols]
        terms = [t for col in self.cols for t in col]
        if not terms:
            self.w = (0, 0, 0)
            self.empty = True
            return
        self.empty = False
        self.w = (max(m - 2 * j for j, m, _ in terms), min(m for _, m, _ in terms), min(j for j, _, _ in terms))
        self._cache = {}

    def bound(self, cell):
        if self.empty:
            return Bd((0, 0, 0), ZEROI)
        key = id(cell)
        if key in self._cache:
            return self._cache[key]
        a, b, e = self.w
        best = None
        for col in self.cols:
            v = ZEROI
            for j, m, C in col:
                v = v + C * cell.msup(m - 2 * j - a, m - b, j - e)
            v = up(v)
            if best is None or mpup(v) > mpup(best):
                best = v
        out = Bd(self.w, best)
        self._cache[key] = out
        return out


def poly_jet(name, cols):
    """Jet of a known polynomial: (P, x P', (x d/dx)^2 P) with coefficients |m|^k C"""
    p0 = Poly(name, cols)
    p1 = Poly(name + "'", cols, mult=lambda m: abs(m))
    p2 = Poly(name + "''", cols, mult=lambda m: m * m)
    for p in (p1, p2):
        p.w = p0.w           # the same weight (the m = 0 terms carry the factor 0)
    return p0, p1, p2


class KJet:
    """a known jet evaluated per cell with caching"""

    def __init__(self, name, cols):
        self.p = poly_jet(name, cols)

    def __call__(self, cell):
        return Jet(*(q.bound(cell) for q in self.p))
