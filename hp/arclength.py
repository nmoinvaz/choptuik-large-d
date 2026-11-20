#!/usr/bin/env python3
"""Pseudo-arclength continuation through a fold (#842): the asymptotic numerical method's series in an arclength s of
(z, D), with the normalization <z_1, z_1> + h_1^2 = 1 and <z_1, z_k> + h_1 h_k = 0 for k >= 2 (h_k the coefficients
of D(s) - D_0), and a bordered Newton corrector, regular at a simple fold.

The series at a base is the base's D-series w(h) (the chain's own, anm_run.py's coefficients_o) composed with
h(s) = sum_k h_k s^k: z(s) = sum_m w_m h(s)^m, so z_k = w_1 h_k + q_k with q_k = sum_{m>=2} w_m [s^k] h(s)^m, and the
normalization gives h_1 = +-(1 + |w_1|^2)^(-1/2), h_k = -<w_1, q_k> / (1 + |w_1|^2). This is the solution of the
bordered order-k system [J F_D; z_1^T h_1] [z_k; h_k] = [-r_k; 0]; solving that system directly needs the residual's
jet along a D(s) that is not linear in s, and the kernel's jet takes D only as D_0 + h (multishoot's Param::var). A
base exactly at the fold (h_1 = 0) has no D-series, and near it the composition's sum cancels: compose() returns each
order's condition number sum_m |w_m| |[s^k] h^m| / |z_k|, the factor by which it loses precision. No base is placed
where the unit tangent's |t_D| is below --guard (0.05); the step is lengthened past the turn or shortened.

The corrector: Newton on (z, D) with the pseudo-arclength row g = <t_z, z - z_p> + t_D (D - D_p) = 0, t the series'
unit tangent at the step's end, by block elimination (Keller's bordering): a = -J^-1 F, b = J^-1 F_D,
dD = -(g + <t_z, a>) / (t_D - <t_z, b>), dz = a - b dD. Each iterate's denominator t_D - <t_z, b> and |b| are recorded:
the elimination loses about |b| |t_z| / |t_D - <t_z, b>| digits when J is nearly singular (#842, coordinator's note).

The turn: where D'(s) = 0 inside a step's [0, a], on that step's series: s*, D* = D(s*), z(s*).

References: H. B. Keller, Numerical solution of bifurcation and nonlinear eigenvalue problems, in Applications of
Bifurcation Theory (Academic Press, 1977) 359-384 (pseudo-arclength, bordering); B. Cochelin, A path-following
technique via an asymptotic-numerical method, Computers & Structures 53 (1994) 1181-1192 (the series in s).

    python3 arclength.py plant [--border 0] [--mutate compose] [--dps 50] [--N 24] [--steps 60] [--out FILE]
    python3 arclength.py selfcheck          (the plant passes, and --border 0 and --mutate compose each fail)

`plant` is the planted fold of #842 (Toy, its docstring); it prints one line per step and the checks, and exits 1 when
a check fails.
"""
import json
import math
import os
import random
import sys
import time

import mpmath as mp
import numpy as np


# ------------------------------------------------------------------ truncated power series and vectors (mpmath)
class TS:
    """a truncated power series c_0 + c_1 s + ... + c_n s^n with mpf coefficients"""
    __slots__ = ("c",)

    def __init__(self, c):
        self.c = list(c)

    def __add__(self, o):
        if isinstance(o, TS):
            return TS([a + b for a, b in zip(self.c, o.c)])
        c = list(self.c)
        c[0] += o
        return TS(c)
    __radd__ = __add__

    def __neg__(self):
        return TS([-a for a in self.c])

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, TS):
            return TS([mp.fsum(self.c[j] * o.c[k - j] for j in range(k + 1)) for k in range(len(self.c))])
        return TS([a * o for a in self.c])
    __rmul__ = __mul__

    def __truediv__(self, o):
        return TS([a / o for a in self.c])


def ts_sin(u):
    """sin of a series, by sin' = cos u', cos' = -sin u'"""
    n = len(u.c)
    S, C = [mp.sin(u.c[0])] + [mp.mpf(0)] * (n - 1), [mp.cos(u.c[0])] + [mp.mpf(0)] * (n - 1)
    for k in range(1, n):
        S[k] = mp.fsum(j * u.c[j] * C[k - j] for j in range(1, k + 1)) / k
        C[k] = -mp.fsum(j * u.c[j] * S[k - j] for j in range(1, k + 1)) / k
    return TS(S)


def sin_(x):
    return ts_sin(x) if isinstance(x, TS) else mp.sin(x)


def o8(z):
    """a vector's parts as contiguous float64 arrays (octo_corrector.o8, here so that the plant needs no kernel)"""
    return tuple(np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z)


def omax(F):
    """max |F| of a residual in parts, to f64 accuracy (octo_corrector.omax)"""
    return float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())


def vadd(u, v, a=1):
    """u + a v"""
    return [x + a * y for x, y in zip(u, v)]


def vscale(u, a):
    return [a * x for x in u]


def vdot(u, v):
    return mp.fdot(u, v)


def vnorm(u):
    return mp.sqrt(mp.fdot(u, u))


def vmax(u):
    return max(abs(x) for x in u)


def horner(Z, s, deriv=False):
    """sum_k Z_k s^k (Z a list of vectors or of scalars), or its derivative in s"""
    vec = isinstance(Z[0], list)
    terms = [(k * Z[k] if not vec else vscale(Z[k], k)) for k in range(1, len(Z))] if deriv else list(Z)
    p = terms[-1]
    for t in reversed(terms[:-1]):
        p = vadd(t, p, s) if vec else t + s * p
    return p


# ------------------------------------------------------------------ the series: composition and its tangent
def compose(W, sign, mutate=""):
    """the arclength series from the D-series W = [w_0, ..., w_N] (vectors), oriented by sign (the sign of h_1):
    (Z, H, cond, mag), Z[k] the vectors z_k (z_0 = w_0), H[k] the scalars h_k (H[0] = 0), mag[k] the norm of the
    composition's sum of magnitudes sum_m |w_m| |[s^k] h^m| and cond[k] = mag[k] / |z_k|, its condition number at
    order k. mutate 'compose' (the control) drops q_k's m >= 2 terms."""
    N = len(W) - 1
    a = 1 + vdot(W[1], W[1])
    H = [mp.mpf(0), sign / mp.sqrt(a)]
    Z = [W[0], vscale(W[1], H[1])]
    cond = [mp.mpf(1), mp.mpf(1)]
    magn = [vnorm(W[0]), vnorm(W[1]) * abs(H[1])]
    c = {1: {1: H[1]}}                       # c[m][k] = [s^k] h(s)^m
    for k in range(2, N + 1):
        for m in range(2, k + 1):
            c.setdefault(m, {})
            c[m][k] = mp.fsum(H[j] * c[m - 1].get(k - j, 0) for j in range(1, k - m + 2))
        q = [mp.mpf(0)] * len(W[0])
        mag = [mp.mpf(0)] * len(W[0])
        if mutate != "compose":
            for m in range(2, k + 1):
                if c[m][k]:
                    q = vadd(q, W[m], c[m][k])
                    mag = vadd(mag, [abs(x) for x in W[m]], abs(c[m][k]))
        hk = -vdot(W[1], q) / a
        H.append(hk)
        c[1][k] = hk
        zk = vadd(q, W[1], hk)
        Z.append(zk)
        mag = vadd(mag, [abs(x) for x in W[1]], abs(hk))
        nz = vnorm(zk)
        magn.append(vnorm(mag))
        cond.append(magn[-1] / nz if nz else mp.inf)
    return Z, H, cond, magn


def unit_tangent(Z, H, s):
    """(t_z, t_D): the series' derivative at s, normalized"""
    tz, tD = horner(Z, s, deriv=True), horner(H, s, deriv=True)
    n = mp.sqrt(vdot(tz, tz) + tD * tD)
    return vscale(tz, 1 / n), tD / n


def tD_poly(Z, H):
    """|t_D(s)| as a fast float64 function of s, from the Gram matrix of the z_k's leading digits"""
    N = len(Z) - 1
    Zf = np.array([[float(x) for x in Z[k]] for k in range(1, N + 1)])
    G = Zf @ Zf.T
    Hf = np.array([float(H[k]) for k in range(1, N + 1)])
    ks = np.arange(1, N + 1)

    def f(s):
        p = ks * float(s) ** (ks - 1)
        dz2 = float(p @ G @ p)
        dD = float(p @ Hf)
        return abs(dD) / math.sqrt(dz2 + dD * dD)
    return f


def turn_in(H, a):
    """the zeros s* of D'(s) in (0, a]: [(s*, D(s*) - D_0)], by a sign scan and Newton in mpmath"""
    dH = [k * H[k] for k in range(1, len(H))]
    d2H = [k * dH[k] for k in range(1, len(dH))]
    ev = lambda P, s: horner(P, s)  # noqa: E731
    out = []
    n = 400
    grid = [a * j / n for j in range(n + 1)]
    vals = [ev(dH, s) for s in grid]
    for j in range(n):
        if vals[j] == 0 or vals[j] * vals[j + 1] < 0:
            s = (grid[j] + grid[j + 1]) / 2
            for _ in range(60):
                ds = ev(dH, s) / ev(d2H, s)
                s -= ds
                if abs(ds) < mp.mpf(10) ** (-mp.mp.dps + 5):
                    break
            out.append((s, horner(H, s)))
    return out


def edge_points(fD, a_hi, guard, n=2000):
    """the s in (0, a_hi] where |t_D(s)| crosses guard, ascending (float64 scan, then bisection)"""
    xs = [a_hi * j / n for j in range(n + 1)]
    v = [fD(x) - guard for x in xs]
    out = []
    for j in range(n):
        if v[j] * v[j + 1] < 0:
            lo, hi = xs[j], xs[j + 1]
            for _ in range(60):
                mid = (lo + hi) / 2
                if (fD(mid) - guard) * v[j] < 0:
                    hi = mid
                else:
                    lo = mid
            out.append((lo + hi) / 2)
    return out


def bordered_step(a, b, g, tz, tD):
    """Keller's block elimination of [J F_D; t_z^T t_D] [dz; dD] = [-F; -g], given a = -J^-1 F and b = J^-1 F_D:
    (dz, dD, denominator)"""
    den = tD - vdot(tz, b)
    dD = -(g + vdot(tz, a)) / den
    return vadd(a, b, -dD), dD, den


# ------------------------------------------------------------------ the planted fold
class Toy:
    """#842's planted fold: six unknowns x = Q y (Q orthogonal), sigma = y_0, g(D) = D + D^2/8,
    R_0 = g(D) - g(D*) + sigma^2, R_i = y_i - p_i - c_i sigma^2 - d_i sigma^3 + e_i sin(D) R_0 (i = 1..5),
    F = T R + 0.1 (R_0 R_1, R_1 R_2, ..., R_5 R_0). The branches: y_i(sigma) = p_i + c_i sigma^2 + d_i sigma^3 and
    D(sigma) = g^-1(g(D*) - sigma^2), sigma < 0 before the turn at D* (sigma = 0) and sigma > 0 after it. J is singular
    at sigma = 0 (R_0's row vanishes) and F_D is not in its range there (g' > 0), so the bordered matrix is regular.
    T, Q, p, c, d, e are drawn from random.Random(seed)."""
    n = 6

    def __init__(self, seed=842, Dstar=mp.mpf("1.3")):
        rng = random.Random(seed)
        u = lambda: mp.mpf(rng.uniform(-1, 1))  # noqa: E731
        n = self.n
        Q, _ = mp.qr(mp.matrix([[u() for _ in range(n)] for _ in range(n)]))
        self.Q = [[Q[i, j] for j in range(n)] for i in range(n)]
        self.T = [[(1 if i == j else 0) + mp.mpf("0.3") * u() for j in range(n)] for i in range(n)]
        self.p, self.c, self.d, self.e = ([u() for _ in range(n)] for _ in range(4))
        self.Dstar = Dstar
        self.gs = self.g(Dstar)

    def g(self, D):
        return D + D * D / 8

    def ginv(self, v):
        return -4 + mp.sqrt(16 + 8 * v)

    def truth(self, sig):
        """(x, D) on the branch at sigma"""
        y = [sig] + [self.p[i] + self.c[i] * sig ** 2 + self.d[i] * sig ** 3 for i in range(1, self.n)]
        return [mp.fsum(self.Q[i][j] * y[j] for j in range(self.n)) for i in range(self.n)], self.ginv(self.gs - sig ** 2)

    def coord(self, x):
        """y = Q^T x"""
        return [mp.fsum(self.Q[j][i] * x[j] for j in range(self.n)) for i in range(self.n)]

    def F(self, x, D):
        n = self.n
        y = [sum(self.Q[j][i] * x[j] for j in range(n)) for i in range(n)]
        sig = y[0]
        R0 = self.g(D) - self.gs + sig * sig
        sD = sin_(D)
        R = [R0] + [y[i] - self.p[i] - self.c[i] * sig * sig - self.d[i] * sig * sig * sig + self.e[i] * sD * R0 for i in range(1, n)]
        return [sum(self.T[i][j] * R[j] for j in range(n)) + mp.mpf("0.1") * R[i] * R[(i + 1) % n] for i in range(n)]

    def jet(self, Zs, Ds, nout):
        """the coefficients 0..nout-1 of F(z(s), D(s)), z(s) = sum Zs[k] s^k, D(s) = sum Ds[k] s^k"""
        pad = lambda cs: list(cs[:nout]) + [mp.mpf(0)] * (nout - len(cs[:nout]))  # noqa: E731
        x = [TS(pad([Z[i] for Z in Zs])) for i in range(self.n)]
        Fs = self.F(x, TS(pad(list(Ds))))
        return [[f.c[k] for f in Fs] for k in range(nout)]

    def jac(self, x, D):
        """J (an mp.matrix) and F_D at (x, D)"""
        n = self.n
        J = mp.matrix(n, n)
        for j in range(n):
            e = [mp.mpf(1 if i == j else 0) for i in range(n)]
            col = self.jet([x, e], [D], 2)[1]
            for i in range(n):
                J[i, j] = col[i]
        return J, self.jet([x], [D, 1], 2)[1]

    def solve(self, J, b):
        return list(mp.lu_solve(J, mp.matrix(b)))

    # the system's interface to continue()
    def dseries(self, x, D, N):
        """the D-series w_0..w_N at (x, D)"""
        J, _ = self.jac(x, D)
        W = [list(x)]
        for k in range(1, N + 1):
            r = self.jet(W, [D, 1], k + 1)[k]
            W.append(self.solve(J, vscale(r, -1)))
        return W

    def direct(self, x, D, N, sign):
        """the bordered recursion with the toy's jet along D(s) (the second route to compose()'s series)"""
        J, FD = self.jac(x, D)
        n = self.n
        w1 = self.solve(J, vscale(FD, -1))
        H = [D, sign / mp.sqrt(1 + vdot(w1, w1))]
        Z = [list(x), vscale(w1, H[1])]
        B = mp.matrix(n + 1, n + 1)
        for i in range(n):
            for j in range(n):
                B[i, j] = J[i, j]
            B[i, n] = FD[i]
            B[n, i] = Z[1][i]
        B[n, n] = H[1]
        for k in range(2, N + 1):
            r = self.jet(Z, H, k + 1)[k]
            sol = list(mp.lu_solve(B, mp.matrix(vscale(r, -1) + [0])))
            Z.append(sol[:n])
            H.append(sol[n])
        H[0] = mp.mpf(0)
        return Z, H

    def correct(self, xp, Dp, tz, tD, border=True, tol=None, it_max=30):
        """bordered Newton (border=False: Newton at fixed D = Dp) with halving on a rising max |F|: (x, D, converged,
        record); the record lists each iterate's max |F|, lam, and with the border its denominator and |b|"""
        tol = tol if tol is not None else mp.mpf(10) ** (-(mp.mp.dps - 10))
        x, D = list(xp), Dp
        its = []
        F = self.F(x, D)
        nf = vmax(F)
        for it in range(it_max):
            if nf <= tol:
                return x, D, True, dict(its=its, F=float(nf))
            J, FD = self.jac(x, D)
            a = self.solve(J, vscale(F, -1))
            if border:
                b = self.solve(J, FD)
                g = vdot(tz, vadd(x, xp, -1)) + tD * (D - Dp)
                dz, dD, den = bordered_step(a, b, g, tz, tD)
            else:
                dz, dD, den, b = a, mp.mpf(0), None, None
            lam = mp.mpf(1)
            for _ in range(12):
                xt, Dt = vadd(x, dz, lam), D + lam * dD
                Ft = self.F(xt, Dt)
                if vmax(Ft) < nf:
                    break
                lam /= 2
            else:
                return x, D, False, dict(its=its, F=float(nf), why="no decrease")
            its.append(dict(F=float(nf), lam=float(lam), den=None if den is None else float(den),
                            b=None if b is None else float(vnorm(b))))
            x, D, F, nf = xt, Dt, Ft, vmax(Ft)
        return x, D, nf <= tol, dict(its=its, F=float(nf))


# ------------------------------------------------------------------ the driver
def continue_arc(sysm, x0, D0, sign0, opt, stop=None, log=print, on_step=None):
    """the continuation from the base (x0, D0) in the direction of sign0 (the sign of D's change): one record per
    step. opt: N, eps, h_max, grow, a0 (the first step's length, at most a_max; 0: a_max), hold (a step that needed
    a halving keeps its length), steps, border, guard, edge (place one base at |t_D| = guard on the approach to the
    first turn), mutate, direct (compare compose() with sysm.direct at each base), keep_x (the points' vectors in the
    records; the chain saves its states instead), turns_max (stop after the step whose series holds that many turns
    in all; 0: never). on_step(rec) is called after each step."""
    N = opt["N"]
    x, D, sign = list(x0), D0, sign0
    t_prev = None
    a_prev = None
    recs = []
    edge_done = not opt.get("edge")
    turns = 0
    for step in range(opt["steps"]):
        W = sysm.dseries(x, D, N)
        if t_prev is not None:
            # orientation: the new tangent (w_1, 1) sign overlaps the previous series' tangent at its end
            ov = vdot(t_prev[0], W[1]) + t_prev[1]
            sign = 1 if ov > 0 else -1
        rec = dict(step=step, D0=float(D), D0_str=mp.nstr(D, 40), sign=sign)
        if opt["border"]:
            Z, H, cond, magn = compose(W, sign, opt.get("mutate", ""))
            rec["cond_max"] = float(max(cond))
            rec["cond"] = [float(c) for c in cond]
            if opt.get("direct"):
                # the second route: each order's difference and the composition's condition number, both scaled by
                # the bordered recursion's coefficient (a composed z_k = 0 would make its own cond infinite)
                Zd, Hd = sysm.direct(x, D, N, sign)
                diffs, dcond = [], []
                for k in range(1, N + 1):
                    vk = mp.sqrt(vdot(Zd[k], Zd[k]) + Hd[k] ** 2)
                    dk = mp.sqrt(vdot(vadd(Z[k], Zd[k], -1), vadd(Z[k], Zd[k], -1)) + (H[k] - Hd[k]) ** 2)
                    diffs.append(float(dk / vk))
                    dcond.append(float(magn[k] / vk))
                rec["direct_rel"] = diffs
                rec["direct_cond"] = dcond
        else:
            Z = [W[k] if k == 0 else vscale(W[k], mp.mpf(sign) ** k) for k in range(N + 1)]
            H = [mp.mpf(0), mp.mpf(sign)] + [mp.mpf(0)] * (N - 1)
        H1 = mp.sqrt(vdot(Z[1], Z[1]) + H[1] ** 2)
        HN = mp.sqrt(vdot(Z[N], Z[N]) + H[N] ** 2)
        a_max = float((opt["eps"] * H1 / HN) ** (mp.mpf(1) / (N - 1))) if HN else opt["h_max"]
        a = min(a_max, opt["h_max"], (a_prev * opt["grow"]) if a_prev else (opt.get("a0") or opt["h_max"]))
        rec.update(a_max=a_max, tD0=float(abs(H[1]) / H1))
        how = "a_max"
        if opt["border"] and opt["guard"]:
            fD = tD_poly(Z, H)
            pts = edge_points(fD, a_max, opt["guard"])
            if not edge_done and pts and fD(0) > opt["guard"] and pts[0] <= a:
                a, how, edge_done = pts[0], "edge", True
            elif fD(a) < opt["guard"]:
                hi = [p for p in pts if p > a]
                lo = [p for p in pts if p < a]
                if hi:
                    a, how = hi[0] * (1 + 1e-6), "guard, lengthened"
                elif lo:
                    a, how = lo[-1] * (1 - 1e-6), "guard, shortened"
        rec["how"] = how
        conv = False
        for attempt in range(4):
            am = mp.mpf(a)
            xp, Dp = horner(Z, am), D + horner(H, am)
            tz, tD = unit_tangent(Z, H, am)
            Fp = vmax(sysm.F(xp, Dp)) if hasattr(sysm, "F") else None
            xc, Dc, conv, crec = sysm.correct(xp, Dp, tz, tD, border=opt["border"])
            rec.setdefault("attempts", []).append(dict(a=a, D_pred=float(Dp), F_pred=None if Fp is None else float(Fp),
                                                      converged=conv, D=float(Dc), corrector=crec))
            if conv:
                break
            a /= 2
        if conv and opt["border"]:
            for s_, dD_ in turn_in(H, am):
                xs_ = horner(Z, s_)
                rec.setdefault("turn", []).append(dict(s=float(s_), D=mp.nstr(D + dD_, 40), Delta=mp.nstr(xs_[-1], 40),
                                                       x=[mp.nstr(v, 40) for v in xs_] if opt.get("keep_x", True) else None))
                turns += 1
        rec["converged"] = conv
        if conv:
            rec.update(a=a, D=float(Dc), D_str=mp.nstr(Dc, 40), Delta=mp.nstr(xc[-1], 40), tD_end=float(abs(tD)),
                       x=[mp.nstr(v, 40) for v in xc] if opt.get("keep_x", True) else None)
        recs.append(rec)
        if on_step is not None:
            on_step(rec)
        log(f"step {step}: D0 {mp.nstr(D, 15)}, sign {sign:+d}, a_max {a_max:.3e}, a {a:.3e} ({how}), |t_D| base {rec['tD0']:.3f}"
            + (f", cond max {rec['cond_max']:.1e}" if "cond_max" in rec else "")
            + (f", direct max rel {max(rec['direct_rel']):.1e}" if "direct_rel" in rec else "")
            + (f" -> D {mp.nstr(Dc, 15)} in {len(crec['its'])} its, max|F| {crec['F']:.1e}" if conv else " -> NOT converged")
            + ("".join(f"; turn at s {t['s']:.4f}, D* {t['D'][:22]}" for t in rec.get("turn", []))))
        if not conv:
            break
        if opt.get("turns_max") and turns >= opt["turns_max"]:
            log(f"stopped: turn {turns} found (turns_max {opt['turns_max']})")
            break
        t_prev = (tz, tD)
        # hold (anm_run.py --a_hold 1): a step that needed a halving keeps its length for the next one
        a_prev = a / opt["grow"] if opt.get("hold") and attempt > 0 else a
        x, D = xc, Dc
        if stop is not None and stop(x, D):
            break
    return recs

# ------------------------------------------------------------------ the chain (anm_run.py --arclength 1)
DPS_SERIES = 140        # the series, its sums and the border's dot products: above octo-double's 128 digits
DPS_KERNEL = 50         # the held values and every number the kernel's side forms, at anm_run.py's precision


def to_mp_vec(z):
    """a vector in parts (two to eight float64 arrays) as mpf, each entry its parts' exact sum"""
    P = [np.asarray(p, dtype=float) for p in z]
    return [mp.fsum(mp.mpf(float(P[q][i])) for q in range(len(P))) for i in range(len(P[0]))]


def to_parts(v, n=8):
    """an mpf vector as n float64 arrays, each entry split exactly (at DPS_SERIES)"""
    out = [np.zeros(len(v)) for _ in range(n)]
    for i, x in enumerate(v):
        r = mp.mpf(x)
        for q in range(n):
            f = float(r)
            out[q][i] = f
            r -= f
    return tuple(np.ascontiguousarray(p) for p in out)


def dd_round(D):
    """D rounded to double-double, as an exact mpf: the kernel forms n = D - 2 there"""
    hi = float(D)
    return mp.mpf(hi) + mp.mpf(float(D - hi))


class BorderedCore:
    """the pinned octo-double system in the unknowns y = (D, z) (D first, so Delta stays the last unknown), with the
    pseudo-arclength row g = <t_z, z - z_p> + t_D (D - D_p) first, for newton_od. D is double-double: each step's dD
    is rounded so that D + dD is, and dz is taken for the rounded dD. The row is reported as 0 once |g| is within
    max(ulp(D), H) / |t_D| (H = --arc_dhold; the rounding of D moves the point along the branch by at most ulp(D)).
    D is held (dD = 0, the step a: the fixed-D Newton step) when the bordered dD is at most H and the row is within
    that bound: D resolves about 1e-30 in double-double, so a smaller move would be rounded to 0 or to a jump of up to
    an ulp, and the arclength row needs no such precision. newton_o: Keller's block
    elimination through the LU at the point (sysm.at(D)), a = -J^-1 (F - c), b = J^-1 F_D, each call's denominator
    t_D - <t_z, b> and |b| logged; olu_solve: the same elimination (transposed: through J^-T)."""

    def __init__(self, sysm, zp, Dp, tz, tD):
        self.sysm, self.L = sysm, sysm.L
        self.size = self.L + 1
        self.zp, self.Dp, self.tz, self.tD = zp, mp.mpf(Dp), tz, mp.mpf(tD)
        self.hold_max = mp.mpf(sysm.opt.get("arc_dhold", 0) or 0)
        self.last = None
        self.log = []

    def split(self, y):
        y = [np.asarray(p, dtype=float) for p in y]
        with mp.workdps(DPS_SERIES):
            D = dd_round(mp.fsum(mp.mpf(float(p[0])) for p in y))
        return D, tuple(np.ascontiguousarray(p[1:]) for p in y)

    def join(self, d0, z):
        """(scalar, vector in parts) as one eight-part vector; the scalar an mpf or float"""
        with mp.workdps(DPS_SERIES):
            r, head = mp.mpf(d0), []
            for _ in range(8):
                f = float(r)
                head.append(f)
                r -= f
        z = list(z) + [np.zeros(self.L)] * (8 - len(z))
        return tuple(np.ascontiguousarray(np.concatenate(([head[q]], np.asarray(z[q], dtype=float)))) for q in range(8))

    def g(self, D, z):
        with mp.workdps(DPS_SERIES):
            return vdot(self.tz, vadd(to_mp_vec(z), self.zp, -1)) + self.tD * (D - self.Dp)

    def g_tol(self, D):
        return max(abs(D) * mp.mpf(2) ** -105, self.hold_max) / abs(self.tD)

    def g_eff(self, D, z):
        g = self.g(D, z)
        return mp.mpf(0) if abs(g) <= self.g_tol(D) else g

    def residual_o(self, y, mutate=0):
        D, z = self.split(y)
        e = self.sysm.at(D)
        return self.join(self.g_eff(D, z), o8(e["po"].residual_o(z, mutate=mutate)))

    def newton_o(self, y, mutate=0):
        D, z = self.split(y)
        e = self.sysm.at(D)
        d, Fh, Fl, pmin, pmax = e["po"].newton_o(z, mutate=mutate)
        FD = self.sysm.F_D(e, z)
        b = o8(e["sn"].olu_solve([np.ascontiguousarray(p) for p in FD]))
        with mp.workdps(DPS_SERIES):
            am, bm = to_mp_vec(d), to_mp_vec(b)
            g = self.g(D, z)
            den = self.tD - vdot(self.tz, bm)
            dD = -(g + vdot(self.tz, am)) / den
            hold = bool(abs(dD) <= self.hold_max and abs(g) <= self.g_tol(D))
            dDr = mp.mpf(0) if hold else dd_round(D + dD) - D
            step = self.join(dDr, to_parts(vadd(am, bm, -dDr)))
            ge = self.g_eff(D, z)
            self.last = dict(D=D, e=e, b=bm, FD=FD, den=den, hold=hold)
            self.log.append(dict(D=mp.nstr(D, 34), den=float(den), b=float(vnorm(bm)), dD=float(dD), dD_rounded=float(dDr), g=float(g),
                                 hold=hold))
            print(f"      border at D {mp.nstr(D, 34)}: g {float(g):+.2e}, dD {float(dD):+.2e} -> "
                  f"{'held' if hold else format(float(dDr), '+.2e')}, den {float(den):.4f}, |b| {float(vnorm(bm)):.4f}, "
                  f"|a| {float(max(abs(x) for x in am)):.2e}", flush=True)
            gh = float(ge)
            gl = float(ge - gh)
        return step, np.concatenate(([gh], np.asarray(Fh))), np.concatenate(([gl], np.asarray(Fl))), pmin, pmax

    def olu_solve(self, rhs, trans=False):
        rhs = [np.asarray(p, dtype=float) for p in rhs] + [np.zeros(self.size)] * (8 - len(rhs))
        e, last = self.last["e"], self.last
        rz = [np.ascontiguousarray(p[1:]) for p in rhs]
        with mp.workdps(DPS_SERIES):
            rg = mp.fsum(mp.mpf(float(p[0])) for p in rhs)
            if last.get("hold"):        # D held: the map is block-diag(1, J), as newton_o's step
                return self.join(rg, e["sn"].olu_solve(rz, trans))
            if not trans:
                a = to_mp_vec(e["sn"].olu_solve(rz))
                yD = (rg - vdot(self.tz, a)) / last["den"]
                return self.join(yD, to_parts(vadd(a, last["b"], -yD)))
            c = to_mp_vec(e["sn"].olu_solve(rz, True))
            if "eT" not in last:
                last["eT"] = to_mp_vec(e["sn"].olu_solve(list(to_parts(self.tz)), True))
                last["FDm"] = to_mp_vec(last["FD"])
            yD = (rg - vdot(last["FDm"], c)) / (self.tD - vdot(last["FDm"], last["eT"]))
            return self.join(yD, to_parts(vadd(c, last["eT"], -yD)))


def _bordered_elim_class():
    """BorderedElim, an ElimO (so that newton_od takes its projection paths) around a BorderedCore: the point's z is
    projected by the ElimO at its own D (sysm.at(D)), and newton_o's step includes the projection's displacement, as
    ElimO's. The start's failed projection at max|F| above 1e-6 raises reseed.StartNotProjected, as ElimO's."""
    from octo_corrector import ElimO
    from reseed import StartNotProjected

    class BorderedElim(ElimO):
        def __init__(self, core, log=False):        # noqa: super().__init__ builds a PinnedQ this class does not use
            self.so, self.log, self.why, self.first = core, log, [], True
            self.counts = dict(u0=0, u0q=0, ladder=0, failed=0, skipped=0)

        def project(self, y):
            from multidouble import od_axpy  # noqa: F401
            core = self.so
            D, z = core.split(y)
            e = core.sysm.at(D)
            first, self.first = self.first, False
            if e["eo"] is None:
                return o8(y)
            c0, n0 = dict(e["eo"].counts), len(e["eo"].why)
            zp = e["eo"].project(z)
            for k in self.counts:
                self.counts[k] += e["eo"].counts.get(k, 0) - c0.get(k, 0)
            self.why.extend(e["eo"].why[n0:])
            if first and e["eo"].counts["failed"] > c0["failed"] and omax(e["po"].residual_o(z)) > 1e-6:
                raise StartNotProjected(dict(self.counts))
            return core.join(D, zp)

        def residual_o(self, y, mutate=0):
            return self.so.residual_o(self.project(y), mutate=mutate)

        def newton_o(self, y, mutate=0):
            from multidouble import od_axpy
            y = o8(y)
            yp = self.project(y)
            st = self.so.newton_o(yp, mutate=mutate)
            disp = o8(od_axpy(-1.0, *y, *yp))
            return (o8(od_axpy(1.0, *disp, *st[0])), *st[1:])
    return BorderedElim


class ChainSys:
    """the chain's system for continue_arc, from anm_run.py's context c (its base setup, solvers and held fit): the
    D-series at a base by anm_run's coefficients_o (the octo-double jet with the held fit's jets), and the bordered
    corrector (newton_od on BorderedElim, with the chain's flags). Every D is double-double; at(D) builds and keeps
    (the last three) the solver, its pinned wrappers, the re-seeder and the held slope there."""

    def __init__(self, c, z8, D):
        self.c, self.opt = c, c["opt"]
        self.base8, self.D = o8(z8), D
        self.L = len(self.base8[0])
        self.cache = {}
        self.n_pts = 0
        if c["hfit"] is None and c["trend"]:
            raise SystemExit("--arclength 1 takes the held values from --held_fit (or constant), not --trend")

    def at(self, D):
        if not (mp.isfinite(D) and D > 2):
            # n = D - 2 > 0 is the kernel's and the held fit's domain; a corrector iterate outside it (a blown-up
            # chord, |dw| 1.9e23 in #1008) is refused as a failed evaluation, not a TypeError from an mpc sqrt
            raise ValueError(f"D {mp.nstr(D, 6)} outside the system's domain D > 2")
        from octo_corrector import ElimO, PinnedO
        from pinning import band_field
        from reseed import Reseeder
        c, opt = self.c, self.opt
        key = (float(D), float(D - float(D)))
        if key in self.cache:
            return self.cache[key]
        with mp.workdps(DPS_KERNEL):
            Dd = mp.mpf(key[0]) + mp.mpf(key[1])
            sn = c["mk"](Dd)
            sn.set_segs(c["segs"])
            spn = c["wrap_at"](sn, Dd)
            po = PinnedO(spn.s, spn.ch, spn.cl, c8=spn.c8)
            rs = (Reseeder(dict(M=c["M"], nodes=c["nodes"], kt=c["kt"], kc=c["kc"], ks=c["ks"]), Dd, threads=opt["threads"],
                           horizon_alpha=opt["horizon_alpha"], qd=bool(opt["reseed_q"]), od=opt["reseed_o"], qd_start=opt["reseed_start"],
                           exact_recip=opt.get("exact_recip", 0))
                  if opt["reseed"] else None)
            eo = ElimO(po, rs, switch=opt["elim_switch"], start_F=None, log=bool(opt["log_proj"])) if opt["reseed"] == 2 else None
            held1 = (band_field(c["hfit"].jets(Dd, 1, c["ks_held"], exact=c["held_x"])[1], c["ks_held"], c["M"], parts=c["hparts"])
                     if c["hfit"] is not None else None)
        e = dict(D=Dd, sn=sn, spn=spn, po=po, rs=rs, eo=eo, held1=held1)
        while len(self.cache) >= 3:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = e
        return e

    def F_D(self, e, z):
        """the pinned residual's D-derivative at fixed z: the kernel's order-1 jet coefficient minus the held slope"""
        from multidouble import od_add_dd, od_axpy
        F = e["sn"].residual_jet_o([o8(z)], 2)
        f1 = [np.array(np.asarray(p), dtype=float) for p in F[1]]
        t = e["held1"]
        if t is not None:
            rows = self.c["rows"]
            if len(t) == 2:
                r = od_add_dd(*(np.ascontiguousarray(p[rows]) for p in f1), -np.asarray(t[0]), -np.asarray(t[1]))
            else:
                r = od_axpy(-1.0, *(np.ascontiguousarray(np.asarray(v)) for v in t), *(np.ascontiguousarray(p[rows]) for p in f1))
            for p, v in zip(f1, r):
                p[rows] = v
        return o8(f1)

    def newton_args(self):
        from octo_corrector import od_qn_kw
        opt = self.opt
        return dict(tol=opt["tol_od"], it_max=opt["qd_it"], verbose=True, junk_w=opt["od_junk"], lam_stall=opt["lam_stall"],
                    accept=opt["od_accept"], strong_chord=int(opt["od_strong"]), full_drop=opt["od_full_drop"],
                    abort_theta=opt["od_abort_theta"], abort_it=opt["od_abort_it"], **od_qn_kw(opt))

    def polish(self):
        """the base solved at D rounded to double-double (its held values there): the chain's corrector at fixed D"""
        from octo_corrector import newton_od
        with mp.workdps(DPS_SERIES):
            self.D = dd_round(self.D)
        e = self.at(self.D)
        so = e["eo"] if e["eo"] is not None else e["po"]
        F0 = omax(e["po"].residual_o(self.base8))
        z8, hist, conv = newton_od(so, self.base8, **self.newton_args())
        if e["eo"] is not None:
            z8 = e["eo"].project(z8)
        F1 = omax(e["po"].residual_o(z8))
        with mp.workdps(DPS_SERIES):
            dDelta = float(mp.fsum(mp.mpf(float(p[-1])) for p in z8) - mp.fsum(mp.mpf(float(p[-1])) for p in self.base8))
        self.base8 = o8(z8)
        return dict(D=mp.nstr(self.D, 34), F_before=F0, F_after=F1, steps=len(hist), converged=bool(conv), dDelta=dDelta)

    def series_from_file(self, path, D, N):
        """the series of an earlier run's arcjet file (--arc_jet), refused unless its D, N and exact_recip are this
        base's (a file without exact_recip predates the flag and was made with it off)"""
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        er_file, er_run = int(meta.get("exact_recip", 0)), int(self.opt.get("exact_recip", 0))
        if er_file != er_run:
            raise SystemExit(f"--arc_jet {path}: its series was made with exact_recip {er_file}, this run takes {er_run}")
        with mp.workdps(DPS_SERIES):
            dD = abs(mp.mpf(meta["D_str"]) - D)
            if meta.get("N") != N or dD > abs(D) * mp.mpf(10) ** -32:
                raise SystemExit(f"--arc_jet {path}: its D {meta['D_str']} / N {meta.get('N')} are not the base's "
                                 f"{mp.nstr(D, 34)} / {N}")
            Zo = [tuple(np.asarray(d[f"z{q}"][k], dtype=float) for q in range(8)) for k in range(N + 1)]
            self.series_info = dict(pivots=tuple(meta["pivots"]), norms=meta["norms"], t_coef=0.0, from_file=path)
            norms = meta["norms"]
            print(f"  series at D {mp.nstr(D, 15)} read from {path}: |w_1| {norms[1]:.3e}, |w_N| {norms[N]:.3e}", flush=True)
            return [to_mp_vec(z) for z in Zo]

    def dseries(self, x, D, N):
        c = self.c
        from pinning import band_field
        if self.opt.get("arc_jet") and not getattr(self, "jet_read", False):
            self.jet_read = True
            self.at(D)
            return self.series_from_file(self.opt["arc_jet"], D, N)
        e = self.at(D)
        t0 = time.time()
        with mp.workdps(DPS_KERNEL):
            pmin, pmax = e["sn"].newton_o(list(self.base8))[3:5]
            held = None
            if c["hfit"] is not None:
                jets = c["hfit"].jets(e["D"], N, c["ks_held"], exact=c["held_x"])
                held = ((lambda j: band_field(jets[j], c["ks_held"], c["M"], parts=c["hparts"])), c["rows"])
            Zo, norms = c["coefficients_o"](e["sn"], self.base8, N, held=held)
        self.series_info = dict(pivots=(float(pmin), float(pmax)), norms=norms, t_coef=round(time.time() - t0, 1))
        if self.opt["save_jet"]:
            jp = os.path.join(c["out"], f"arcjet_{self.n_pts:03d}_D{float(D):.6f}.npz")
            np.savez(jp, **{f"z{q}": np.array([np.asarray(z[q]) for z in Zo]) for q in range(8)},
                     meta=json.dumps(dict(D_str=mp.nstr(D, 34), N=N, parts=8, jet="o", norms=norms, pivots=(float(pmin), float(pmax)),
                                          exact_recip=int(self.opt.get("exact_recip", 0)))))
        print(f"  series at D {mp.nstr(D, 15)}: pivots {pmin:.1e}..{pmax:.1e}, |w_1| {norms[1]:.3e}, |w_N| {norms[N]:.3e}, "
              f"{self.series_info['t_coef']} s", flush=True)
        return [to_mp_vec(z) for z in Zo]

    def correct(self, xp, Dp, tz, tD, border=True):
        from octo_corrector import newton_od, q4_of_o, with_horizon
        from reseed import StartNotProjected
        opt, t0 = self.opt, time.time()
        with mp.workdps(DPS_SERIES):
            Dd = dd_round(Dp)
            # the prediction moved along the tangent to the double-double D the corrector runs at: left at Dp, its
            # Delta is off by about w_1 (Dd - Dp) ~ 1e-31, which the re-seed's projection does not see (#842 pilot 2)
            xs = vadd(xp, tz, (Dd - Dp) / tD)
            pred8 = to_parts(xs)
        e = self.at(Dd)
        rec = dict(D_pred=mp.nstr(Dp, 34), D_start=mp.nstr(Dd, 34), D_round=float(Dd - Dp), Delta_shift=float(xs[-1] - xp[-1]))
        if opt["reseed"]:
            zr, rsd = e["rs"].reseed(q4_of_o(pred8))
            rec["reseed"] = dict(ok=rsd["ok"], seconds=rsd["seconds"], dropped=rsd["dropped"])
            if rsd["ok"]:
                pred8 = with_horizon(pred8, zr, e["rs"])
        rec["F_pred"] = omax(e["po"].residual_o(pred8))
        print(f"  attempt at D {mp.nstr(Dd, 15)} (the prediction moved {rec['D_round']:+.2e} in D, {rec['Delta_shift']:+.2e} in Delta): "
              f"predicted max|F| {rec['F_pred']:.2e}"
              + (f", re-seed {'ok' if rec['reseed']['ok'] else 'FAILED'}" if opt["reseed"] else ""), flush=True)
        if border:
            core = BorderedCore(self, xp, Dp, tz, tD)
            so = _bordered_elim_class()(core, log=bool(opt["log_proj"]))
            y0 = core.join(Dd, pred8)
        else:
            core = None
            so = e["eo"] if e["eo"] is not None else e["po"]
            y0 = pred8
        try:
            y8, hist, conv = newton_od(so, y0, **self.newton_args())
        except StartNotProjected as ex:
            rec.update(its=[], F=float("nan"), why=f"start not projected {ex.args[0]}", seconds=round(time.time() - t0, 1))
            return xp, Dp, False, rec
        if border:
            y8 = so.project(y8)
            D, z8 = core.split(y8)
            rec["border"] = core.log
            rec["counts"] = so.counts
        else:
            if e["eo"] is not None:
                y8 = e["eo"].project(y8)
            D, z8 = Dd, o8(y8)
        ee = self.at(D)
        Fz = omax(ee["po"].residual_o(z8))
        rec.update(its=[dict(F=h.get("F"), lam=h.get("lam")) for h in hist], F=Fz, converged_od=bool(conv),
                   seconds=round(time.time() - t0, 1), D=mp.nstr(D, 34))
        print(f"  corrector: {len(hist)} steps, pinned max|F| {Fz:.2e} at D {mp.nstr(D, 20)}, {'converged' if conv else 'NOT converged'}, "
              f"{rec['seconds']} s" + (f"; denominators {', '.join(format(r['den'], '.2e') for r in core.log)}, |b| "
                                        f"{', '.join(format(r['b'], '.2e') for r in core.log)}" if core else ""), flush=True)
        if not conv:
            return xp, Dp, False, rec
        self.base8, self.D = o8(z8), D
        self.n_pts += 1
        c = self.c
        from octo_corrector import save_o
        p = os.path.join(c["out"], f"arc_{self.n_pts:03d}_D{float(D):.6f}.npz")
        save_o(p, z8, dict(D=float(D), D_str=mp.nstr(D, 34), M=c["M"], nodes=c["nodes"], kt=c["kt"], kc=c["kc"], ks=c["ks"],
                           kfreeze=c["kf"], pin_held=c["ks_held"], N=c["N"], source=opt["state"], parts=8, arclength=True,
                           tD=float(tD), segs=[[float(x) for x in sg] for sg in c["segs"]],
                           exact_recip=int(opt.get("exact_recip", 0))))
        rec["state"] = os.path.basename(p)
        with mp.workdps(DPS_SERIES):
            return to_mp_vec(z8), D, True, rec


def run_chain(c):
    """anm_run.py --arclength 1: the continuation from the run's base (after anm_run's own base setup), records in
    OUT/arc.jsonl, states OUT/arc_<n>_D<D>.npz; stops after --steps steps, below --D_min, at a failed step, or after
    the step whose series holds the --turns_max-th turn (2: the mirror's own turn, if it has one)"""
    from octo_corrector import load_o
    opt = c["opt"]
    if opt.get("arc_diag"):
        return run_floor(c) if int(opt["arc_diag"]) == 2 else run_diag(c)
    z8 = c["last_q"]["z8"] if "z8" in c["last_q"] else load_o(c["state_path"])[0]
    sysm = ChainSys(c, z8, c["D"])
    pol = sysm.polish()
    print(f"base polished at D rounded to double-double ({pol['D']}): max|F| {pol['F_before']:.2e} -> {pol['F_after']:.2e} in "
          f"{pol['steps']} steps, {'converged' if pol['converged'] else 'NOT converged'}, Delta moved {pol['dDelta']:+.2e}", flush=True)
    rp = os.path.join(c["out"], "arc.jsonl")
    with open(rp, "a") as f:
        f.write(json.dumps(dict(base=opt["state"], polish=pol, opt={k: v for k, v in opt.items()})) + "\n")
    if not pol["converged"]:
        print("stopped: the base polish did not converge", flush=True)
        return 1

    def on_step(rec):
        with open(rp, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    aopt = dict(N=opt["N"], eps=mp.mpf(opt["eps"]), h_max=opt["h_max"], grow=opt["a_grow"], a0=opt["a0"], hold=bool(opt["a_hold"]), steps=opt["steps"], border=int(opt["border"]),
                guard=opt["guard"], edge=0, mutate="", direct=0, keep_x=False, turns_max=opt["turns_max"])
    stop = (lambda x, D: D < opt["D_min"]) if opt["D_min"] else None
    with mp.workdps(DPS_SERIES):
        x0 = to_mp_vec(sysm.base8)
        recs = continue_arc(sysm, x0, sysm.D, int(opt["arc_sign"]), aopt, stop=stop,
                            log=lambda s: print(s, flush=True), on_step=on_step)
    print("done", flush=True)
    return 0 if recs and recs[-1]["converged"] else 1


def run_diag(c):
    """anm_run.py --arclength 1 --arc_diag 1: the adapter's pieces at the polished base, where each step must be at
    the base's own level and each pair must agree. (1) The fixed-D Newton step. (2) b = J^-1 F_D, with F_D formed
    before and after the factorization, against -w_1 of the order-1 series. (3) The four weakest singular values of J
    and of the bordered matrix (junk_subspace_o). (4) The bordered step, BorderedCore then BorderedElim, with the
    series' tangent and s = 0. (5) F_D against central differences in D at h 1e-22 and 1e-26. OUT/arc_diag.json."""
    from octo_corrector import junk_subspace_o, load_o
    opt = c["opt"]
    z8 = c["last_q"]["z8"] if "z8" in c["last_q"] else load_o(c["state_path"])[0]
    sysm = ChainSys(c, z8, c["D"])
    out = dict(polish=sysm.polish())
    t0 = time.time()

    def say(k, v):
        out[k] = v
        print(f"diag {k}: {json.dumps(v, default=str)} ({time.time() - t0:.0f} s)", flush=True)

    def rel(u, v):
        return float(vmax(vadd(u, v, -1)) / vmax(v))
    with mp.workdps(DPS_SERIES):
        D0, z0 = sysm.D, sysm.base8
        e = sysm.at(D0)
        FD_pre = to_mp_vec(sysm.F_D(e, z0))
        st = e["po"].newton_o(z0)
        a0 = to_mp_vec(st[0])
        say("newton_base", dict(F=omax(e["po"].residual_o(z0)), step_max=float(vmax(a0)), step_Delta=float(a0[-1]),
                                pivots=[float(st[3]), float(st[4])]))
        FD = sysm.F_D(e, z0)
        FDm = to_mp_vec(FD)
        b = to_mp_vec(e["sn"].olu_solve([np.ascontiguousarray(p) for p in FD]))
        b_pre = to_mp_vec(e["sn"].olu_solve(list(to_parts(FD_pre))))
        held = None
        if c["hfit"] is not None:
            from pinning import band_field
            with mp.workdps(DPS_KERNEL):
                jets = c["hfit"].jets(e["D"], 1, c["ks_held"], exact=c["held_x"])
            held = ((lambda j: band_field(jets[j], c["ks_held"], c["M"], parts=c["hparts"])), c["rows"])
        with mp.workdps(DPS_KERNEL):
            Zo, norms = c["coefficients_o"](e["sn"], z0, 1, held=held)
        w1 = to_mp_vec(Zo[1])
        say("b", dict(FD_max=float(vmax(FDm)), FD_pre_vs_post=rel(FD_pre, FDm), b_norm=float(vnorm(b)), w1_norm=float(vnorm(w1)),
                      b_plus_w1=rel(vscale(b, -1), w1), b_pre_vs_post=rel(b_pre, b), b_Delta=float(b[-1]), w1_Delta=float(w1[-1])))
        e["po"].newton_o(z0)
        _, sJ, sJd = junk_subspace_o(e["po"], 4, delta_max=2.0)
        tD = 1 / mp.sqrt(1 + vdot(w1, w1))
        tz = vscale(w1, tD)
        core = BorderedCore(sysm, to_mp_vec(z0), D0, tz, tD)
        y0 = core.join(D0, z0)
        stc = core.newton_o(y0)
        sc = to_mp_vec(stc[0])
        _, sB, sBd = junk_subspace_o(core, 4, delta_max=2.0)
        say("sigma", dict(J=sorted(float(x) for x in list(sJ) + list(sJd)), bordered=sorted(float(x) for x in list(sB) + list(sBd))))
        BE = _bordered_elim_class()(core)
        ste = BE.newton_o(y0)
        se = to_mp_vec(ste[0])
        say("bordered_step", dict(core_D=float(sc[0]), core_max=float(vmax(sc[1:])), core_Delta=float(sc[-1]), elim_D=float(se[0]),
                                  elim_max=float(vmax(se[1:])), elim_Delta=float(se[-1]), log=core.log, counts=BE.counts))
        for h in (mp.mpf("1e-22"), mp.mpf("1e-26")):
            Dp, Dm = dd_round(D0 + h), dd_round(D0 - h)
            Fp = to_mp_vec(sysm.at(Dp)["po"].residual_o(z0))
            Fm = to_mp_vec(sysm.at(Dm)["po"].residual_o(z0))
            fd = vscale(vadd(Fp, Fm, -1), 1 / (Dp - Dm))
            r = c["rows"]
            held_rows = set(range(r.start, r.stop))
            dif = vadd(fd, FDm, -1)
            say(f"FD_h{float(h):.0e}", dict(h_used=float(Dp - D0), rel=rel(fd, FDm),
                                            held_rows=float(max(abs(dif[i]) for i in held_rows) / vmax(FDm)),
                                            other_rows=float(max(abs(dif[i]) for i in range(len(dif)) if i not in held_rows) / vmax(FDm)),
                                            worst_row=int(max(range(len(dif)), key=lambda i: abs(dif[i])))))
    with open(os.path.join(c["out"], "arc_diag.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("done", flush=True)
    return 0


def run_floor(c):
    """anm_run.py --arclength 1 --arc_diag 2 (#1008): where the floor of the predicted max|F| comes from (about 1e-36
    at D 259.4-259.5 for every arclength step a, so the corrector's first |dz| is that floor times the gain). At the
    stored base, not polished, with the series of --arc_jet (or formed), for each a of --floor_a (a = 0 is the base):
    the prediction as correct() forms it, moved along the tangent to its double-double D; its octo-double max|F|
    before and after the re-seed, with the re-seed's octo-double rung residuals (Reseeder.last_o); and where each max
    sits (the held rows, the rows at the horizon data's offsets, the rest, the five largest rows). --floor_polish 1 polishes the base first: the held
    values are taken relative to the run's own first state, so a later state of a run misses this run's held rows by
    the fit's offset between the two states (7.7e-38 at run2's arc step 7, as run2's own polish removed 4.4e-38).
    Control: w_1 scaled by 1 + --floor_mutate at the largest a must raise max|F| before the re-seed by at least 100
    times, or the probe cannot see an error of the series. Exit 1 when the control does not fire or the base's own
    max|F| (a = 0, before the re-seed) exceeds 1e-60. OUT/arc_floor.json."""
    from octo_corrector import load_o, q4_of_o, with_horizon
    opt = c["opt"]
    z8 = c["last_q"]["z8"] if "z8" in c["last_q"] else load_o(c["state_path"])[0]
    sysm = ChainSys(c, z8, c["D"])
    with mp.workdps(DPS_SERIES):
        sysm.D = dd_round(sysm.D)
    t0 = time.time()
    rows = c["rows"]
    pol = sysm.polish() if int(opt["floor_polish"]) else None
    if pol is not None:
        print(f"floor: base polished: max|F| {pol['F_before']:.2e} -> {pol['F_after']:.2e} in {pol['steps']} steps "
              f"({time.time() - t0:.0f} s)", flush=True)

    hz = None

    def where(F):
        v = np.abs(np.asarray(F[0], dtype=float) + np.asarray(F[1], dtype=float))
        held = np.zeros(len(v), dtype=bool)
        if rows is not None:
            held[rows] = True
        hor = np.zeros(len(v), dtype=bool)
        if hz is not None:
            hor[hz] = True
        top = np.argsort(v)[::-1][:5]
        return dict(max=float(v.max()), held_rows=float(v[held].max()) if held.any() else None,
                    other_rows=float(v[~held].max()), horizon_rows=float(v[hor].max()) if hor.any() else None,
                    rest_rows=float(v[~held & ~hor].max()), top=[[int(i), float(v[i])] for i in top])

    def probe(W, a, tag):
        with mp.workdps(DPS_SERIES):
            Z, H, _, _ = compose(W, int(opt["arc_sign"]), "")
            am = mp.mpf(a)
            xp, Dp = horner(Z, am), sysm.D + horner(H, am)
            tz, tD = unit_tangent(Z, H, am)
            Dd = dd_round(Dp)
            pred8 = to_parts(vadd(xp, tz, (Dd - Dp) / tD))
        e = sysm.at(Dd)
        rec = dict(tag=tag, a=a, D=mp.nstr(Dd, 34), D_round=float(Dd - Dp), before=where(e["po"].residual_o(pred8)))
        if e["rs"] is not None:
            ts = time.time()
            zr, rsd = e["rs"].reseed(q4_of_o(pred8))
            rec["reseed"] = dict(ok=bool(rsd["ok"]), seconds=round(time.time() - ts, 1))
            if rsd["ok"]:
                rec["after"] = where(e["po"].residual_o(with_horizon(pred8, zr, e["rs"])))
                rec["refine_o"] = dict(rungs=[float(x) for x in (getattr(e["rs"], "last_o", None) or [])],
                                       end=getattr(e["rs"], "last_o_end", ""))
        aft = rec.get("after")

        def say(w):
            return (f"{w['max']:.2e} (held " + ("none" if w["held_rows"] is None else f"{w['held_rows']:.2e}")
                    + ", horizon " + ("none" if w["horizon_rows"] is None else f"{w['horizon_rows']:.2e}")
                    + f", rest {w['rest_rows']:.2e})")
        print(f"floor {tag}: a {a:g}, D {mp.nstr(Dd, 15)}: max|F| before the re-seed {say(rec['before'])}, after "
              + (say(aft) if aft else "none")
              + (f", refine_o {['%.2e' % x for x in rec['refine_o']['rungs']]} {rec['refine_o']['end']}" if "refine_o" in rec else "")
              + f" ({time.time() - t0:.0f} s)", flush=True)
        return rec

    with mp.workdps(DPS_SERIES):
        W = sysm.dseries(to_mp_vec(sysm.base8), sysm.D, opt["N"])
    e0 = sysm.at(sysm.D)
    if e0["rs"] is not None:
        # the rows at the horizon data's offsets (m_h, a_h; 2M of them): the rows the re-seed's data feed
        hz = slice(int(e0["rs"].off["mh"]), int(e0["rs"].off["mh"]) + 2 * int(e0["rs"].M))
    out = dict(D=mp.nstr(sysm.D, 34), state=opt["state"], arc_jet=opt["arc_jet"], polish=pol, base=where(e0["po"].residual_o(sysm.base8)),
               offsets={k: int(v) for k, v in dict(e0["rs"].off).items() if isinstance(v, (int, np.integer))} if e0["rs"] is not None else None,
               points=[])
    if str(opt["floor_dh"]):
        # (#1008) the residual's smoothness in D at the base: R(h) = F(z_0, D + h) - (D_h - D) F_D(z_0), D_h the
        # double-double D + h. A smooth F leaves R ~ h^2 F_DD; a D-dependent coefficient rounded below octo-double
        # leaves a remainder that does not fall with h. Control: F_D scaled by 1 + --floor_mutate * 1e14 must raise R
        # in proportion to h.
        FD = to_mp_vec(sysm.F_D(e0, sysm.base8))
        out["dh"] = []
        for h in [float(s) for s in str(opt["floor_dh"]).split(",")]:
            with mp.workdps(DPS_SERIES):
                Dh = dd_round(sysm.D + mp.mpf(h))
            e = sysm.at(Dh)
            with mp.workdps(DPS_SERIES):
                F1 = to_mp_vec(e["po"].residual_o(sysm.base8))
                hu = Dh - sysm.D
                R = vadd(F1, FD, -hu)
                Rm = vadd(F1, vscale(FD, 1 + mp.mpf(opt["floor_mutate"]) * mp.mpf(10) ** 14), -hu)
            rec = dict(h=h, h_used=float(hu), R=where(to_parts(R)), R_control=where(to_parts(Rm)))
            out["dh"].append(rec)
            w, wm = rec["R"], rec["R_control"]
            print(f"floor dh {h:g} (used {float(hu):.3e}): R max {w['max']:.2e} (held {w['held_rows']:.2e}, horizon "
                  f"{w['horizon_rows']:.2e}, rest {w['rest_rows']:.2e}, top rows {[i for i, _ in w['top']]}); "
                  f"control {wm['max']:.2e} ({time.time() - t0:.0f} s)", flush=True)
        with open(os.path.join(c["out"], "arc_floor.json"), "w") as f:
            json.dump(out, f, indent=1, default=str)
        print("done", flush=True)
        return 0
    alist = [float(s) for s in str(opt["floor_a"]).split(",")]
    for a in alist:
        out["points"].append(probe(W, a, "series"))
    with mp.workdps(DPS_SERIES):
        Wm = list(W)
        Wm[1] = vscale(W[1], 1 + mp.mpf(opt["floor_mutate"]))
    ctl = probe(Wm, alist[-1], f"control (w_1 x (1 + {opt['floor_mutate']:g}))")
    ref = out["points"][-1]["before"]["max"]
    ctl["fires"] = bool(ctl["before"]["max"] >= 100 * ref)
    out["control"] = ctl
    base_ok = out["points"][0]["a"] == 0 and out["points"][0]["before"]["max"] <= 1e-60
    out["base_ok"] = bool(base_ok)
    with open(os.path.join(c["out"], "arc_floor.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"floor control: {'FIRES' if ctl['fires'] else 'DOES NOT FIRE'} ({ctl['before']['max']:.2e} against "
          f"{ref:.2e}); base at a = 0: {'ok' if base_ok else 'NOT at its own level, or a = 0 not first'}", flush=True)
    print("done", flush=True)
    return 0 if (ctl["fires"] and base_ok) else 1



def cmd_plant(argv):
    opt = dict(N=24, eps=1e-30, h_max=1.0, grow=2.0, steps=60, border=1, guard=0.05, edge=1, mutate="", direct=1, dps=50,
               sig0=-0.8, sig1=0.8, out="")
    for a in argv:
        k, _, v = a.lstrip("-").partition("=")
        if k not in opt:
            raise SystemExit(f"unknown option {a}")
        opt[k] = type(opt[k])(v) if not isinstance(opt[k], str) else v
    mp.mp.dps = opt["dps"]
    toy = Toy()
    x0, D0 = toy.truth(mp.mpf(opt["sig0"]))
    opt["eps"] = mp.mpf(opt["eps"])
    recs = continue_arc(toy, x0, D0, +1, opt, stop=lambda x, D: toy.coord(x)[0] >= opt["sig1"])
    # (a) every converged point on the curve
    dist, sigs = [], []
    for r in recs:
        if not r["converged"]:
            continue
        x = [mp.mpf(v) for v in r["x"]]
        y = toy.coord(x)
        xt, Dt = toy.truth(y[0])
        dist.append(max(vmax(vadd(x, xt, -1)), abs(mp.mpf(r["D_str"]) - Dt)))
        sigs.append(float(y[0]))
    worst = float(max(dist)) if dist else float("inf")
    ok_a = bool(dist) and worst <= 1e-30
    # (b) both branches
    nneg, npos = sum(1 for s in sigs if s <= -0.3), sum(1 for s in sigs if s >= 0.3)
    ok_b = nneg >= 2 and npos >= 2
    # (c) the turn
    xs, Ds = toy.truth(mp.mpf(0))
    tr = [(r["step"], t) for r in recs for t in r.get("turn", [])]
    errs = [(st, float(abs(mp.mpf(t["D"]) - Ds)), float(vmax(vadd([mp.mpf(v) for v in t["x"]], xs, -1)))) for st, t in tr]
    ok_c = bool(errs) and all(e[1] <= 1e-10 and e[2] <= 1e-10 for e in errs)
    # (d) the composed series against the bordered recursion: at most 100 x 10^-dps x the composition's condition
    # number at every order (amended before the run, after the coordinator's note on the guard's edge), both scaled
    # by the bordered coefficient (amended after the first run, whose control left the bar infinite where z_k = 0)
    unit = 10.0 ** (-opt["dps"])
    dmarg = []
    for r in recs:
        if "direct_rel" in r:
            dmarg.append(max(d / (100 * unit * max(1.0, c)) for d, c in zip(r["direct_rel"], r["direct_cond"])))
    ok_d = bool(dmarg) and max(dmarg) <= 1
    edge = [r for r in recs if r.get("how") == "edge"]
    after_edge = [r for r in recs if edge and r["step"] == edge[0]["step"] + 1]
    summ = dict(opt={k: (float(v) if isinstance(v, mp.mpf) else v) for k, v in opt.items()},
                a=dict(ok=ok_a, worst=worst, n=len(dist)), b=dict(ok=ok_b, n_neg=nneg, n_pos=npos, sigmas=sigs),
                c=dict(ok=ok_c, turns=[dict(step=e[0], dD=e[1], dx=e[2]) for e in errs]),
                d=dict(ok=ok_d, worst_margin=max(dmarg) if dmarg else None),
                edge=dict(placed=bool(edge), step=edge[0]["step"] if edge else None,
                          tD=edge[0].get("tD_end") if edge else None,
                          next_base_tD=after_edge[0]["tD0"] if after_edge else None,
                          next_cond_max=after_edge[0].get("cond_max") if after_edge else None,
                          next_direct_max=max(after_edge[0]["direct_rel"]) if after_edge and "direct_rel" in after_edge[0] else None),
                records=recs)
    print(f"(a) on the curve: {len(dist)} points, worst {worst:.1e} (bar 1e-30) -> {'pass' if ok_a else 'FAIL'}")
    print(f"(b) both branches: {nneg} with sigma <= -0.3, {npos} with sigma >= +0.3 (2 each) -> {'pass' if ok_b else 'FAIL'}")
    print(f"(c) the turn: " + (", ".join(f"step {e[0]}: |D - D*| {e[1]:.1e}, |x - x*| {e[2]:.1e}" for e in errs) or "none found")
          + f" (bar 1e-10) -> {'pass' if ok_c else 'FAIL'}")
    print(f"(d) composed against bordered: worst ratio to 100 x 1e-{opt['dps']} x cond {summ['d']['worst_margin']} (bar 1) -> {'pass' if ok_d else 'FAIL'}")
    print(f"edge base: {summ['edge']}")
    if opt["out"]:
        with open(opt["out"], "w") as f:
            json.dump(summ, f, indent=1, default=str)
    ok = ok_a and ok_b and ok_c and ok_d
    print("plant: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_selfcheck(argv):
    """the plant passes and each control fails"""
    import subprocess
    runs = [("plant", [], 0), ("--border 0", ["--border=0"], 1), ("--mutate compose", ["--mutate=compose"], 1)]
    good = True
    for name, extra, want in runs:
        r = subprocess.run([sys.executable, __file__, "plant"] + extra + argv, capture_output=True, text=True)
        last = [ln for ln in r.stdout.splitlines() if ln.startswith(("(a)", "(b)", "(c)", "(d)", "plant:"))]
        print(f"{name}: exit {r.returncode} (wanted {want})")
        for ln in last:
            print("   " + ln)
        good &= (r.returncode == want)
    print("selfcheck: " + ("PASS" if good else "FAIL"))
    return 0 if good else 1


def _delta_of_state(path):
    """(D, Delta) of an eight-part state file (save_o's keys zh, zl, z2..z7, meta D_str), each its parts' exact sum"""
    f = np.load(path, allow_pickle=True)
    meta = json.loads(str(f["meta"]))
    keys = ["zh", "zl"] + [f"z{q}" for q in range(2, 8)]
    return mp.mpf(meta["D_str"]), mp.fsum(mp.mpf(float(f[k][-1])) for k in keys if k in f)


def _jet_delta(path, D):
    """Delta on a stored D-series (anm_run.py --save_jet 1, eight parts) at D, by Horner in u = D - D_b with D_b the
    series' D rounded to double-double, and the last two terms' size there (the truncation estimate)"""
    f = np.load(path, allow_pickle=True)
    meta = json.loads(str(f["meta"]))
    Db = dd_round(mp.mpf(meta["D_str"]))
    zk = [mp.fsum(mp.mpf(float(f[f"z{q}"][k][-1])) for q in range(8)) for k in range(f["z0"].shape[0])]
    u = D - Db
    val = mp.mpf(0)
    for c in reversed(zk):
        val = val * u + c
    N = len(zk) - 1
    return val, abs(zk[N] * u ** N) + abs(zk[N - 1] * u ** (N - 1)), float(u)


def cmd_pilotcheck(argv):
    """#842's pilot: the corrected point's Delta against chain_nsr's stored series at another base, at the point's own
    D. pass: |Delta - Delta_jet(D)| <= --bar (1e-30) + the jet's truncation estimate there. --shift dD evaluates the
    jet at D + dD (the control: 1e-28 moves Delta by about 1.3e-29 x 10, so it must fail). --state2: a state of
    chain_nsr's own near the jet's base, the jet's format check (the same rule)."""
    opt = dict(pilot="", jet="", state2="", bar=1e-30, shift=0.0)
    for a in argv:
        k, _, v = a.lstrip("-").partition("=")
        if k not in opt:
            raise SystemExit(f"unknown option {a}")
        opt[k] = type(opt[k])(v)
    mp.mp.dps = DPS_SERIES
    good = True
    for name, path in (("state2", opt["state2"]), ("pilot", opt["pilot"])):
        if not path:
            continue
        D, dl = _delta_of_state(path)
        val, trunc, u = _jet_delta(opt["jet"], D + mp.mpf(opt["shift"]))
        diff = abs(dl - val)
        ok = diff <= opt["bar"] + trunc
        good &= ok
        print(f"{name} {os.path.basename(path)}: D {mp.nstr(D, 25)}, u {u:+.3e}, Delta {mp.nstr(dl, 40)}, jet "
              f"{mp.nstr(val, 40)}, |diff| {float(diff):.2e} (bar {opt['bar']:.0e} + truncation {float(trunc):.1e})"
              f"{', shift ' + format(opt['shift'], '.0e') if opt['shift'] else ''} -> {'pass' if ok else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pilotcheck":
        sys.exit(cmd_pilotcheck(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "plant":
        sys.exit(cmd_plant(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        sys.exit(cmd_selfcheck(sys.argv[2:]))
    raise SystemExit(__doc__)
