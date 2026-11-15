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
import random
import sys

import mpmath as mp


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
    import numpy as np
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
def continue_arc(sysm, x0, D0, sign0, opt, stop=None, log=print):
    """the continuation from the base (x0, D0) in the direction of sign0 (the sign of D's change): one record per
    step. opt: N, eps, h_max, grow, steps, border, guard, edge (place one base at |t_D| = guard on the approach to
    the first turn), mutate, direct (compare compose() with sysm.direct at each base)."""
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
        a = min(a_max, opt["h_max"], (a_prev * opt["grow"]) if a_prev else opt["h_max"])
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
                rec.setdefault("turn", []).append(dict(s=float(s_), D=mp.nstr(D + dD_, 40), x=[mp.nstr(v, 40) for v in horner(Z, s_)]))
                turns += 1
        rec["converged"] = conv
        if conv:
            rec.update(a=a, D=float(Dc), D_str=mp.nstr(Dc, 40), x=[mp.nstr(v, 40) for v in xc], tD_end=float(abs(tD)))
        recs.append(rec)
        log(f"step {step}: D0 {mp.nstr(D, 15)}, sign {sign:+d}, a_max {a_max:.3e}, a {a:.3e} ({how}), |t_D| base {rec['tD0']:.3f}"
            + (f", cond max {rec['cond_max']:.1e}" if "cond_max" in rec else "")
            + (f", direct max rel {max(rec['direct_rel']):.1e}" if "direct_rel" in rec else "")
            + (f" -> D {mp.nstr(Dc, 15)} in {len(crec['its'])} its, max|F| {crec['F']:.1e}" if conv else " -> NOT converged")
            + ("".join(f"; turn at s {t['s']:.4f}, D* {t['D'][:22]}" for t in rec.get("turn", []))))
        if not conv:
            break
        t_prev = (tz, tD)
        a_prev = a
        x, D = xc, Dc
        if stop is not None and stop(x, D):
            break
    return recs


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plant":
        sys.exit(cmd_plant(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        sys.exit(cmd_selfcheck(sys.argv[2:]))
    raise SystemExit(__doc__)
