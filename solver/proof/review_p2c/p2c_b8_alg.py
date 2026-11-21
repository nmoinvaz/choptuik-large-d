#!/usr/bin/env python3
"""P2 item 4 (#1061): two arithmetics for one set of formulas, written for this review (b8_alg.py not opened).

Every formula of the bootstrap (p2c_b8_boot.py) is written once against the interface below and evaluated in two ways.
- Point: exact values at one point (x, n) with given remainders, in acb balls, as jets (f, x f', (x d/dx)^2 f). It
  checks Lemma R's identities and serves as the planted truth for the bounds.
- Cell: upper bounds of norms on a cell, uniform in n >= n0, as jets of majorants. A majorant is a sum of terms
  c * eps^e * h_a * w^[a > 0] * (r or X)^s, with h_a one of the four bootstrap constants (linear; a product of two is
  folded with h_b <= Hmax_b), w = theta rho^(K+1) carried exactly, and the cell's geometry in s:
    B cells (r_lo > 0, r <= x_F): r folded at the worst end at once, s = 0, e is the eps-power.
    Z cell  (0 <= r <= r_a):      s is the power of r (merged to the smallest one; r <= r_a).
    F       (r >= x_F, rho <= rho_max): e is the power of rho = X^2/n, s the power of X (merged to the largest one;
                                  X >= X_lo); X and rho are bounded independently.
  Norms: l1 over harmonics, induced l1 for operators; |[fg]| <= |f||g|, |[g .]| <= |g|.
  Inverses: (I + E)^-1 <= 1/(1 - sup E), d/dx by -T E' T, the sup taken over the cell at h = Hmax.
"""
from flint import acb, arb

import p2c_b8_series as SER
from p2c_b8_series import K, add, sub, scale, hp, matvec, matmat, M_e, M_o, eye, norm

NH = 5            # term index a: 0 none, 1 h_V, 2 h_S, 3 h_beta, 4 h_c
ONE = arb(1)


def apow(b, k):
    """b^k for an exact positive ball b and integer k (no ** on balls that may contain 0)."""
    if k == 0:
        return ONE
    if k < 0:
        return ONE / apow(b, -k)
    out = b
    for _ in range(k - 1):
        out = out * b
    return out


class Cell:
    """mode 'B', 'Z' or 'F'; kind 'sector' (0 <= arg x <= alpha, outside |x - 1| < delta/2) or 'ray' (arg = alpha)."""

    def __init__(self, mode, eps0, Hmax, kappa, delta, r_lo=None, r_hi=None, xF=None, rho_max=None, kind="sector"):
        self.mode, self.kind, self.eps0 = mode, kind, eps0
        self.Hmax = [arb(repr(float(v))) for v in Hmax]          # (h_V, h_S, h_beta, h_c) <= Hmax
        self.kappa, self.delta = kappa, delta
        sk = kappa.sin()
        self._p = {}
        if mode in ("B", "Z"):
            self.r_lo, self.r_hi = r_lo, r_hi
            X_lo, X_hi = 1 + r_lo, 1 + r_hi
            self.X_lo, self.X_hi = X_lo, X_hi
            self.th_lo, self.th_hi = r_lo / X_lo, r_hi / X_hi
            self.Ecap = 2 * K + 4
            if kind == "ray":
                self.L_lo = (1 + 2 * r_lo * r_lo * sk + apow(r_lo, 4)).sqrt()
                self.L_hi = (1 + 2 * r_hi * r_hi * sk + apow(r_hi, 4)).sqrt()
            else:
                lo = arb(0) if (r_lo <= 1 and r_hi >= 1) else (r_lo * r_lo - 1 if r_lo > 1 else 1 - r_hi * r_hi)
                self.L_lo = max((lo, delta / 2 * (1 + r_lo * r_lo).sqrt()), key=lambda z: z.lower())
                self.L_hi = 1 + r_hi * r_hi
            if mode == "B":
                self.wf_hi = self.th_hi * apow(X_hi, 2 * K + 2)        # w = wf eps^(K+1)
                self.wf_lo = self.th_lo * apow(X_lo, 2 * K + 2)
                self.inv_th = ONE / self.th_lo
        else:
            self.xF = xF
            self.X_lo = 1 + xF
            self.th_F = xF / (1 + xF)
            self.rho_max = rho_max
            self.Ecap = 2 * K + 8
            self.lam = self.X_lo / (self.X_lo - 2)                        # X^2/|x^2 - 1| <= lam on F
        if mode != "F" and self.L_lo.lower() <= 0:
            raise ValueError("cell meets x^2 = 1")

    def label(self):
        if self.mode == "F":
            return f"F[{self.xF}..]"
        return f"{self.mode}{'r' if self.kind == 'ray' else ''}[{self.r_lo.str(4, radius=False)},{self.r_hi.str(4, radius=False)}]"

    def pw(self, base, k):
        key = (base, k)
        if key not in self._p:
            b = {"ra": getattr(self, "r_hi", None), "X": getattr(self, "X_lo", None), "eps": self.eps0,
                 "rho": getattr(self, "rho_max", None)}[base]
            self._p[key] = apow(b, k)
        return self._p[key]


class Maj:
    __slots__ = ("cell", "t")

    def __init__(self, cell, t=None):
        self.cell, self.t = cell, (t if t is not None else {})

    def copy(self):
        return Maj(self.cell, {k: list(v) for k, v in self.t.items()})

    def _ins(self, e, a, c, s):
        cl = self.cell
        if e > cl.Ecap:
            c = c * cl.pw("rho" if cl.mode == "F" else "eps", e - cl.Ecap)
            e = cl.Ecap
        if cl.mode == "B":
            s = 0
        key = (e, a)
        old = self.t.get(key)
        if old is None:
            self.t[key] = [c, s]
            return
        c0, s0 = old
        if s0 == s:
            old[0] = c0 + c
        elif cl.mode == "Z":
            sm = min(s0, s)
            old[0] = c0 * cl.pw("ra", s0 - sm) + c * cl.pw("ra", s - sm)
            old[1] = sm
        else:
            sm = max(s0, s)
            old[0] = c0 * cl.pw("X", s0 - sm) + c * cl.pw("X", s - sm)
            old[1] = sm

    def __add__(self, o):
        r = self.copy()
        for (e, a), (c, s) in o.t.items():
            r._ins(e, a, c, s)
        return r

    def smul(self, k):
        return Maj(self.cell, {key: [c * k, s] for key, (c, s) in self.t.items()})

    def __mul__(self, o):
        cl = self.cell
        r = Maj(cl)
        for (e1, a1), (c1, s1) in self.t.items():
            for (e2, a2), (c2, s2) in o.t.items():
                c, e, s = c1 * c2, e1 + e2, s1 + s2
                if a1 and a2:
                    # h_a1 h_a2 w^2 <= (h_a1 Hmax_a2 + Hmax_a1 h_a2) w w_hi / 2
                    fc, fe, fs = wfold(cl)
                    half = c * fc / 2
                    r._ins(e + fe, a1, half * cl.Hmax[a2 - 1], s + fs)
                    r._ins(e + fe, a2, half * cl.Hmax[a1 - 1], s + fs)
                else:
                    r._ins(e, a1 or a2, c, s)
        return r

    def sup(self):
        """A number bounding the majorant on the cell for every n >= n0 and every h <= Hmax."""
        cl = self.cell
        tot = arb(0)
        for (e, a), (c, s) in self.t.items():
            if a:
                fc, fe, fs = wfold(cl)
                c, e, s = c * fc * cl.Hmax[a - 1], e + fe, s + fs
            tot += c * evalterm(cl, e, s, f"sup {cl.label()}")
        return tot

    def ratio(self, weight="w"):
        """Coefficients (const, h_V, h_S, h_beta, h_c) of an affine bound of |Q|/weight, weight w or (n - 1) w."""
        cl = self.cell
        src = self
        if weight == "wn":
            src = self * inv_nm1(cl)
        out = [arb(0) for _ in range(NH)]
        for (e, a), (c, s) in src.t.items():
            if a:
                out[a] += c * evalterm(cl, e, s, f"ratio {cl.label()}")
            else:
                # divide by w >= w_lo
                if cl.mode == "B":
                    out[0] += c / cl.wf_lo * evalterm(cl, e - (K + 1), s, f"ratio0 {cl.label()}")
                elif cl.mode == "Z":
                    out[0] += c * (1 + cl.r_hi) * evalterm(cl, e - (K + 1), s - 1, f"ratio0 {cl.label()}")
                else:
                    out[0] += c / cl.th_F * evalterm(cl, e - (K + 1), s, f"ratio0 {cl.label()}")
        return out


class BoundError(Exception):
    pass


def evalterm(cl, e, s, where):
    if cl.mode == "F":
        if e < 0 or s > 0:
            raise BoundError(f"{where}: unbounded term rho^{e} X^{s}")
        return cl.pw("rho", e) * cl.pw("X", s)
    if e < 0:
        raise BoundError(f"{where}: unbounded term eps^{e}")
    if cl.mode == "Z":
        if s < 0:
            raise BoundError(f"{where}: unbounded term r^{s}")
        return cl.pw("eps", e) * cl.pw("ra", s)
    return cl.pw("eps", e)


def wfold(cl):
    """w <= fc eps^fe r^fs (B, Z) or fc rho^fe (F), for folding one remainder factor into a number."""
    if cl.mode == "B":
        return cl.wf_hi, K + 1, 0
    if cl.mode == "Z":
        return apow(1 + cl.r_hi, 2 * K + 1), K + 1, 1
    return ONE, K + 1, 0


def const(cl, c, e=0, s=0, a=0):
    m = Maj(cl)
    m._ins(e, a, arb(c), s)
    return m


def inv_nm1(cl):
    """1/(n - 1) = eps/(1 - eps) <= eps/(1 - eps0); on F eps = rho X^-2."""
    c = ONE / (1 - cl.eps0)
    return const(cl, c, 1, -2 if cl.mode == "F" else 0)


# ---------------------------------------------------------------- jets

def jlen(*js):
    return min(len(j) for j in js)


class CellArith:
    """Majorant arithmetic on one cell. Objects are jets (lists) of Maj."""

    point = False

    def __init__(self, cell, atoms, omega_abs, order=2):
        self.cl, self.at, self.om = cell, atoms, omega_abs
        self.order = order
        self._K = {}

    # atoms
    def K(self, name):
        if name not in self._K:
            self._K[name] = poly_maj(self.cl, self.at[name], self.order)
        return self._K[name]

    def zero(self, n=1):
        return [Maj(self.cl) for _ in range(n)]

    def add(self, a, b):
        n = jlen(a, b)
        return [a[i] + b[i] for i in range(n)]

    sub = add

    def neg(self, a):
        return a

    def mul(self, a, b):
        n = jlen(a, b)
        out = [a[0] * b[0]]
        if n > 1:
            out.append(a[1] * b[0] + a[0] * b[1])
        if n > 2:
            out.append(a[2] * b[0] + (a[1] * b[1]).smul(arb(2)) + a[0] * b[2])
        return out

    mv = mm = hp = sc = mul

    def Me(self, c):
        return c

    Mo = Me

    def I(self):
        return [const(self.cl, 1), Maj(self.cl), Maj(self.cl)]

    def Om(self):
        return [const(self.cl, self.om), Maj(self.cl), Maj(self.cl)]

    def ninv(self, E):
        s = E[0].sup()
        if s.upper() >= 1:
            raise BoundError(f"Neumann bound lost on {self.cl.label()} (sup {s.upper().str(4, radius=False)})")
        t0 = ONE / (1 - s)
        out = [const(self.cl, t0)]
        if len(E) > 1:
            out.append(E[1].smul(t0 * t0))
        if len(E) > 2:
            out.append(E[2].smul(t0 * t0) + (E[1] * E[1]).smul(2 * t0 * t0 * t0))
        return out

    sinv = ninv

    def dshift(self, j):
        return j[1:]

    def trunc(self, j, n):
        return j[:n + 1]

    # scalar functions of x and n
    def x(self, k):
        cl = self.cl
        if cl.mode == "B":
            v = apow(cl.r_hi, k) if k >= 0 else apow(cl.r_lo, k)
            return [const(cl, v), const(cl, abs(k) * v), const(cl, k * k * v)]
        if cl.mode == "Z":
            return [const(cl, 1, 0, k), const(cl, abs(k), 0, k), const(cl, k * k, 0, k)]
        c = ONE if k >= 0 else apow(cl.th_F, k)
        return [const(cl, c, 0, k), const(cl, abs(k) * c, 0, k), const(cl, k * k * c, 0, k)]

    def inv_x2m1(self):
        cl = self.cl
        if cl.mode == "B":
            L, r2 = cl.L_lo, cl.r_hi * cl.r_hi
            return [const(cl, ONE / L), const(cl, 2 * r2 / (L * L)), const(cl, 4 * r2 / (L * L) + 8 * r2 * r2 / (L * L * L))]
        if cl.mode == "Z":
            L = cl.L_lo
            return [const(cl, ONE / L), const(cl, 2 / (L * L), 0, 2),
                    const(cl, 4 / (L * L) + 8 * cl.r_hi * cl.r_hi / (L * L * L), 0, 2)]
        lam = cl.lam
        return [const(cl, lam, 0, -2), const(cl, 2 * lam * lam, 0, -2), const(cl, 4 * lam * lam + 8 * lam * lam * lam, 0, -2)]

    def x2m1(self):
        cl = self.cl
        if cl.mode == "B":
            r2 = cl.r_hi * cl.r_hi
            return [const(cl, cl.L_hi), const(cl, 2 * r2), const(cl, 4 * r2)]
        if cl.mode == "Z":
            return [const(cl, 1 + cl.r_hi * cl.r_hi), const(cl, 2, 0, 2), const(cl, 4, 0, 2)]
        return [const(cl, 1, 0, 2), const(cl, 2, 0, 2), const(cl, 4, 0, 2)]

    def nm1(self):
        cl = self.cl
        return [const(cl, 1, -1, 2 if cl.mode == "F" else 0), Maj(cl), Maj(cl)]

    def inv_nm1(self):
        return [inv_nm1(self.cl), Maj(self.cl), Maj(self.cl)]

    def eps(self):
        cl = self.cl
        return [const(cl, 1, 1, -2 if cl.mode == "F" else 0), Maj(cl), Maj(cl)]

    def one(self):
        return self.I()

    def rem(self, name, h_index, w1=False):
        """Value majorant of a remainder under (B_h): h w, or h w1 = h w/theta."""
        cl = self.cl
        if not w1:
            return [const(cl, 1, 0, 0, h_index)]
        if cl.mode == "B":
            return [const(cl, cl.inv_th, 0, 0, h_index)]
        if cl.mode == "Z":
            return [const(cl, 1 + cl.r_hi, 0, -1, h_index)]
        return [const(cl, ONE / cl.th_F, 0, 0, h_index)]


def poly_maj(cl, P, order):
    """Jet of majorants of a polynomial sum_{j,m} p_jm eps^j x^m: sum |p_jm| eps^j r^m (and m, m^2 times it)."""
    out = [Maj(cl) for _ in range(order + 1)]
    for (j, m), v in P.items():
        nv = norm(v)
        if nv.upper() == 0:
            continue
        for p in range(order + 1):
            if p and m == 0:
                continue
            c = nv * (m ** p)
            if cl.mode == "B":
                out[p]._ins(j, 0, c * apow(cl.r_hi, m), 0)
            elif cl.mode == "Z":
                out[p]._ins(j, 0, c, m)
            else:
                out[p]._ins(j, 0, c, m - 2 * j)            # eps^j r^m <= rho^j X^(m - 2j)
    return out


# ---------------------------------------------------------------- the point arithmetic

class PointArith:
    """Exact values at a point, as jets (f, D f, D^2 f) with D = x d/dx."""

    point = True

    def __init__(self, atoms, x, n, omega_mat, order=2):
        self.at, self.x_, self.n = atoms, x, n
        self.eps_ = 1 / n
        self.Omm = omega_mat
        self.order = order
        self._K = {}

    def K(self, name):
        if name not in self._K:
            P = self.at[name]
            self._K[name] = [SER.peval(P, self.eps_, self.x_, p) for p in range(self.order + 1)]
        return self._K[name]

    def add(self, a, b):
        n = jlen(a, b)
        return [add(a[i], b[i]) for i in range(n)]

    def sub(self, a, b):
        n = jlen(a, b)
        return [sub(a[i], b[i]) for i in range(n)]

    def neg(self, a):
        return [scale(v, acb(-1)) for v in a]

    def _bil(self, a, b, f):
        n = jlen(a, b)
        out = [f(a[0], b[0])]
        if n > 1:
            out.append(add(f(a[1], b[0]), f(a[0], b[1])))
        if n > 2:
            out.append(add(add(f(a[2], b[0]), scale(f(a[1], b[1]), acb(2))), f(a[0], b[2])))
        return out

    def mv(self, A, v):
        return self._bil(A, v, matvec)

    def mm(self, A, B):
        return self._bil(A, B, matmat)

    def hp(self, f, g):
        return self._bil(f, g, hp)

    def sc(self, s, a):
        return self._bil(s, a, lambda u, v: scale(v, u))

    def mul(self, a, b):
        return self._bil(a, b, lambda u, v: u * v)

    def Me(self, c):
        return [M_e(v) for v in c]

    def Mo(self, s):
        return [M_o(v) for v in s]

    def I(self):
        z = [[acb(0)] * 2 for _ in range(2)]
        return [eye(2), z, z]

    def Om(self):
        z = [[acb(0)] * 2 for _ in range(2)]
        return [self.Omm, z, z]

    def ninv(self, E):
        n = len(E)
        M = add(eye(2), E[0])
        d = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        T = [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]
        out = [T]
        if n > 1:
            T1 = scale(matmat(matmat(T, E[1]), T), acb(-1))
            out.append(T1)
        if n > 2:
            a = scale(matmat(matmat(T, E[2]), T), acb(-1))
            b = scale(matmat(matmat(matmat(matmat(T, E[1]), T), E[1]), T), acb(2))
            out.append(add(a, b))
        return out

    def sinv(self, d):
        t0 = 1 / (1 + d[0])
        out = [t0]
        if len(d) > 1:
            out.append(-t0 * t0 * d[1])
        if len(d) > 2:
            out.append(-t0 * t0 * d[2] + 2 * t0 * t0 * t0 * d[1] * d[1])
        return out

    def dshift(self, j):
        return j[1:]

    def trunc(self, j, n):
        return j[:n + 1]

    def x(self, k):
        v = self.x_ ** k
        return [v, v * k, v * k * k]

    def inv_x2m1(self):
        x2 = self.x_ * self.x_
        u = 1 / (x2 - 1)
        return [u, -2 * x2 * u * u, -4 * x2 * u * u + 8 * x2 * x2 * u * u * u]

    def x2m1(self):
        x2 = self.x_ * self.x_
        return [x2 - 1, 2 * x2, 4 * x2]

    def nm1(self):
        return [self.n - 1, acb(0), acb(0)]

    def inv_nm1(self):
        return [1 / (self.n - 1), acb(0), acb(0)]

    def eps(self):
        return [self.eps_, acb(0), acb(0)]

    def one(self):
        return [acb(1), acb(0), acb(0)]
