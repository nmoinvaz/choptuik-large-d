#!/usr/bin/env python3
"""Nonlinear toy of the horizon block: does the large-D Newton basin belong to the formulation?

Fields b (the metric field, b = a^2 in the real a-equation) and y (the field with the regular singular point, the real p)
on u in [0, u0], x = 1 - u, ' = d/du:

    x b' = (N + k(u)) b^2 - (N + y^2) b          (Bernoulli/Riccati: w = 1/b obeys x w' = (N + y^2) w - (N + k))
    u y' = s y + u (al + ga N (b - 1)) y + g(u)  (exponents 0 and s at u = 0; the coupling N (b - 1) is the real G)

N = 2 s + 1, as n - 1 against s = n/2 - 1 in the real horizon block (c1 = 0, one harmonic, omega = 0). The real
a-equation x b_x = (N + q^2) b - N b^2 (hp/src/model.rs, H) is the k = 0 case with q^2 = p^2 + m^2. The mass mode
dw ~ x^-N grows inward from the horizon and the b^2 term turns it into a movable simple pole of b where it reaches O(1).
Planted truth: y* = 1 - u/2 + 0.3 u^2 and w* = 1/b* = 0.998 (1 - 0.05 u - 0.02 u^2) solve the system exactly, which
fixes the polynomial k and the series g (computed at dps + 60). The far side is an algebraic condition at the node
u0 = 0.3, phi(b0, y0) = (y0 - y0*) + (b0 - b0*) + (b0 - b0*)^2/2 + (y0 - y0*)^2/2 = 0, standing in for the interior.
"""
import math
import os
import random
import sys
import time

import mpmath as mp
from mpmath import fdot, mpf

U0 = "0.3"
STRICT = True        # a residual whose horizon series does not converge is undefined (rejected)


def pmul(a, b):
    out = [mpf(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def padd(*ps):
    L = max(len(p) for p in ps)
    out = [mpf(0)] * L
    for p in ps:
        for i, x in enumerate(p):
            out[i] += x
    return out


def shift(c, uc, nout):
    """Taylor coefficients of f(uc + t) from those of f(u) at 0 (first nout)"""
    L = len(c)
    pw = [mpf(1)]
    for _ in range(L):
        pw.append(pw[-1] * uc)
    out = []
    for m in range(min(nout, L)):
        # sum_{j >= m} C(j, m) c_j uc^(j - m)
        terms = []
        bin_ = mpf(1)
        for j in range(m, L):
            if j > m:
                bin_ = bin_ * j / (j - m)
            terms.append(bin_ * c[j] * pw[j - m])
        out.append(mp.fsum(terms))
    return out + [mpf(0)] * max(0, nout - len(out))


class Toy:
    def __init__(self, s, dps, al="0.5", ga="0.5", mutate=None, Lg=220):
        self.dps = dps
        self.mutate = mutate
        with mp.workdps(dps + 60):
            s_ = mpf(s)
            N = 2 * s_ + 1
            Y = [mpf(1), mpf("-0.5"), mpf("0.3")]
            W0 = mpf("0.998")
            w = [W0, -W0 * mpf("0.05"), -W0 * mpf("0.02")]
            y2 = pmul(Y, Y)
            wp = [w[1], 2 * w[2]]
            xwp = pmul([mpf(1), mpf(-1)], wp)
            k = padd([N * c for c in w], pmul(y2, w), [-N], [-c for c in xwp])
            # b* = 1/w* as a series
            bs = [1 / w[0]]
            for j in range(1, Lg + 2):
                acc = mpf(0)
                for i in range(1, min(j, 2) + 1):
                    acc += w[i] * bs[j - i]
                bs.append(-acc / w[0])
            alr, gar = mpf(al), mpf(ga)
            yp = [Y[1], 2 * Y[2]]
            uyp = [mpf(0)] + yp                                # u y*'
            bm1 = [bs[0] - 1] + bs[1:]
            cpl = [alr + gar * N * bm1[0]] + [gar * N * c for c in bm1[1:]]
            cy = pmul(cpl[:Lg + 1], Y)[:Lg + 1]                 # (al + ga N (b* - 1)) y*
            g = padd(uyp, [-s_ * c for c in Y], [mpf(0)] + [-c for c in cy[:Lg]])[:Lg + 1]
            u0 = mpf(U0)
            self.ref = dict(bh=1 / w[0], b0=1 / (w[0] + w[1] * u0 + w[2] * u0 ** 2), y0=Y[0] + Y[1] * u0 + Y[2] * u0 ** 2)
        with mp.workdps(dps):
            self.s = +s_
            self.N = +N
            self.al = +alr
            self.ga = +gar * (1 + mpf("1e-6")) if mutate == "gamma" else +gar
            self.k = [+c for c in k]
            self.g = [+c for c in g]
            self.u0 = +u0
            self.tol = mpf(10) ** (-(dps // 2) if mutate == "half" else -dps - 3)
        self.kmax = 8 * dps + 900
        self.tol_dyn = None
        self._gcache = {}
        self.kloc = int(1.8 * dps) + 30

    # ------------------------------------------------------------------ series at the horizon
    def hseries(self, bh, u, tang=True, kfix=None, coeffs=False):
        """the regular family's (b, y) at u for horizon datum b(0) = bh, summed until the terms fall below tol;
        with tang, their derivatives in bh. Returns (B, Y, dB, dY, terms, converged[, coefficient lists])"""
        s, N, al, ga, g, kp = self.s, self.N, self.al, self.ga, self.g, self.k
        Lg, dk = len(g), len(kp)
        be, et = [mpf(bh)], [-g[0] / s]
        dbe, det = [mpf(1)], [mpf(0)]
        b2, y2, db2, dy2 = [], [], [], []
        upw = mpf(1)
        B, Yv, dB, dY = be[0], et[0], dbe[0], det[0]
        scale = max(abs(B), abs(Yv), 1)
        small, conv = 0, False
        t_prev, tt_prev = None, None
        kmax = kfix if kfix else self.kmax
        tolh = max(self.tol, self.tol_dyn) if self.tol_dyn is not None else self.tol
        j = 0
        while j < kmax:
            b2.append(fdot(be, be[::-1]))
            y2.append(fdot(et, et[::-1]))
            y2b = fdot(y2, be[::-1])
            by = fdot(be, et[::-1])
            kb2 = fdot(kp[:min(dk, j + 1)], b2[j::-1][:min(dk, j + 1)])
            rb = N * b2[j] + kb2 - N * be[j] - y2b
            nb = (rb + j * be[j]) / (j + 1)
            Fj = al * et[j] + ga * N * (by - et[j])
            gj = g[j + 1] if j + 1 < Lg else 0
            ny = (Fj + gj) / (j + 1 - s)
            if tang:
                db2.append(2 * fdot(be, dbe[::-1]))
                dy2.append(2 * fdot(et, det[::-1]))
                dy2b = fdot(dy2, be[::-1]) + fdot(y2, dbe[::-1])
                dby = fdot(dbe, et[::-1]) + fdot(be, det[::-1])
                dkb2 = fdot(kp[:min(dk, j + 1)], db2[j::-1][:min(dk, j + 1)])
                drb = N * db2[j] + dkb2 - N * dbe[j] - dy2b
                dnb = (drb + j * dbe[j]) / (j + 1)
                dF = al * det[j] + ga * N * (dby - det[j])
                dny = dF / (j + 1 - s)
                dbe.append(dnb)
                det.append(dny)
            be.append(nb)
            et.append(ny)
            j += 1
            upw *= u
            tb, ty = nb * upw, ny * upw
            B += tb
            Yv += ty
            if tang:
                dB += dbe[-1] * upw
                dY += det[-1] * upw
            t = max(abs(tb), abs(ty))
            tt = max(abs(dbe[-1]), abs(det[-1])) * upw if tang else mpf(0)
            if kfix is None and self.mutate == "half" and j > 2 and t < tolh * scale:
                conv = True                          # the control: stop at the first term below 10^-(P/2)
                break
            if kfix is None:
                # geometric tail estimate t rho/(1 - rho) from the last ratio, for the values and the tangents
                rho = t / t_prev if t_prev else mpf(0)
                rt = tt / tt_prev if (tang and tt_prev) else mpf(0)
                if rho < 1 and rt < 1:
                    tail = t * rho / (1 - rho) + t
                    ttail = (tt * rt / (1 - rt) + tt) / max(abs(dB), abs(dY), 1) if (tang and self.mutate != "half") else 0
                    if tail < tolh * scale and ttail < tolh:
                        small += 1
                        if small >= 3:
                            conv = True
                            break
                    else:
                        small = 0
                else:
                    small = 0
                if t > mpf(10) ** 15 * scale:        # divergent: the series is off its domain at u
                    break
            t_prev, tt_prev = t, tt
        out = (B, Yv, dB, dY, j, conv)
        if coeffs:
            out = out + (be, et)
        return out

    # ------------------------------------------------------------------ Taylor steps between nodes
    def local(self, uc, bc, yc, tang=True):
        """Taylor coefficients about uc of the solution through (bc, yc); with tang, the tangents in bc (q = 0) and
        yc (q = 1) as dB[q], dY[q]"""
        s, N, al, ga = self.s, self.N, self.al, self.ga
        K = self.kloc
        key = mp.nstr(uc, self.dps + 5)
        if key not in self._gcache:
            self._gcache[key] = (shift(self.g, uc, K + 2), shift(self.k, uc, len(self.k)))
        gl, kl = self._gcache[key]
        dk = len(kl)
        be, et = [mpf(bc)], [mpf(yc)]
        dB = [[mpf(1)], [mpf(0)]]
        dY = [[mpf(0)], [mpf(1)]]
        b2, y2, Fs = [], [], []
        db2 = [[], []]
        dy2 = [[], []]
        dFs = [[], []]
        x0 = 1 - uc
        for j in range(K):
            ber, etr = be[::-1], et[::-1]
            b2.append(fdot(be, ber))
            y2.append(fdot(et, etr))
            y2b = fdot(y2, ber)
            by = fdot(be, etr)
            m = min(dk, j + 1)
            kb2 = fdot(kl[:m], b2[j::-1][:m])
            rb = N * b2[j] + kb2 - N * be[j] - y2b
            nb = (rb + j * be[j]) / (x0 * (j + 1))
            Fj = al * et[j] + ga * N * (by - et[j])
            Fs.append(Fj)
            Fm = Fs[j - 1] if j >= 1 else 0
            ny = ((s - j) * et[j] + uc * Fj + Fm + gl[j]) / (uc * (j + 1))
            if tang:
                for q in range(2):
                    dbq, deq = dB[q], dY[q]
                    db2[q].append(2 * fdot(be, dbq[::-1]))
                    dy2[q].append(2 * fdot(et, deq[::-1]))
                    dy2b = fdot(dy2[q], ber) + fdot(y2, dbq[::-1])
                    dby = fdot(dbq, etr) + fdot(be, deq[::-1])
                    dkb2 = fdot(kl[:m], db2[q][j::-1][:m])
                    drb = N * db2[q][j] + dkb2 - N * dbq[j] - dy2b
                    dF = al * deq[j] + ga * N * (dby - deq[j])
                    dFs[q].append(dF)
                    dFm = dFs[q][j - 1] if j >= 1 else 0
                    dbq.append((drb + j * dbq[j]) / (x0 * (j + 1)))
                    deq.append(((s - j) * deq[j] + uc * dF + dFm) / (uc * (j + 1)))
            be.append(nb)
            et.append(ny)
        return be, et, dB, dY

    def integrate(self, u_from, u_to, b, y, tang=True, theta="0.25", nmax=400):
        """Taylor steps from u_from to u_to (either direction); returns (b, y, J = d(b, y)/d(b, y)_start, steps, ok)"""
        u = mpf(u_from)
        u_to = mpf(u_to)
        J = [[mpf(1), mpf(0)], [mpf(0), mpf(1)]]
        th = mpf(theta)
        steps = 0
        while abs(u - u_to) > mpf(10) ** (-self.dps) and steps < nmax:
            be, et, dbe, det = self.local(u, b, y, tang)
            K = len(be) - 1
            sgn = 1 if u_to > u else -1
            h = min(th * u, mpf(8) / self.N, abs(u_to - u))
            scale = 1 + abs(b) + abs(y)
            ok = False
            tolh = max(self.tol, self.tol_dyn) if self.tol_dyn is not None else self.tol
            for _ in range(60):
                tail = max(max(abs(be[j]), abs(et[j])) * h ** j for j in (K - 2, K - 1, K))
                if tail < tolh * scale:
                    ok = True
                    break
                h *= mpf("0.6")
            if not ok:
                return b, y, J, steps, False
            hs = sgn * h
            pw = [mpf(1)]
            for _ in range(K):
                pw.append(pw[-1] * hs)
            nb, ny = fdot(be, pw), fdot(et, pw)
            if tang:
                Js = [[fdot(dbe[0], pw), fdot(dbe[1], pw)], [fdot(det[0], pw), fdot(det[1], pw)]]
                J = [[Js[0][0] * J[0][0] + Js[0][1] * J[1][0], Js[0][0] * J[0][1] + Js[0][1] * J[1][1]],
                     [Js[1][0] * J[0][0] + Js[1][1] * J[1][0], Js[1][0] * J[0][1] + Js[1][1] * J[1][1]]]
            b, y = nb, ny
            u = u + hs if abs(u_to - u) > h else u_to
            steps += 1
            if not (mp.isfinite(b) and mp.isfinite(y)) or abs(b) > 1e30 or abs(y) > 1e30:
                return b, y, J, steps, False
        return b, y, J, steps, True

    # ------------------------------------------------------------------ the far condition at the node
    def phi(self, b0, y0):
        db, dy = b0 - self.ref["b0"], y0 - self.ref["y0"]
        return dy + db + db * db / 2 + dy * dy / 2, 1 + db, 1 + dy

    # ------------------------------------------------------------------ horizon-data inversion
    def invert(self, u, btarget, bh0, maxit=60):
        """b_h with B(b_h; u) = btarget: Newton on r = 1/B - 1/btarget in w_h = 1/b_h (the Bernoulli map is nearly
        affine there), halving a step whose series does not converge or whose |r| does not fall"""
        wh = 1 / mpf(bh0)
        tolh = max(self.tol, self.tol_dyn) if self.tol_dyn is not None else self.tol
        B, Yv, dB, dY, nt, conv = self.hseries(1 / wh, u)
        if not conv:
            return None
        r = 1 / B - 1 / btarget
        for it in range(maxit):
            dr = dB / (B * B * wh * wh)
            step = -r / dr
            lam = mpf(1)
            for _ in range(40):
                wt = wh + lam * step
                Bt, Yt, dBt, dYt, ntt, convt = self.hseries(1 / wt, u)
                if convt:
                    rt = 1 / Bt - 1 / btarget
                    if abs(rt) < abs(r) or abs(rt) <= 10 * tolh * abs(1 / btarget):
                        break
                lam /= 2
            else:
                return 1 / wh
            wh, B, dB, r = wt, Bt, dBt, rt
            if abs(lam * step) < mpf(10) ** (-self.dps + 1) * abs(wh) or abs(r) <= tolh * abs(1 / btarget):
                break
        return 1 / wh


# ---------------------------------------------------------------------- formulations
def res_series(T):
    """F1, ours: unknowns (b_h, b0, y0); the horizon series summed at u0 matched to the node, and the far condition"""
    def F(z):
        bh, b0, y0 = z
        B, Yv, dB, dY, nt, conv = T.hseries(bh, T.u0)
        ph, pb, py = T.phi(b0, y0)
        if STRICT and not conv:
            # the residual is defined only where the series converges at u0 (the hypothesis' domain)
            big = mpf(10) ** 50
            return [big, big, ph], [[1, 0, 0], [0, 1, 0], [0, pb, py]], dict(nt=nt, conv=False)
        return [B - b0, Yv - y0, ph], [[dB, mpf(-1), mpf(0)], [dY, mpf(0), mpf(-1)], [mpf(0), pb, py]], dict(nt=nt, conv=conv)
    return F


def res_inward(T, ue):
    """F2, inward: unknowns (b0, y0, b_h); the node integrated toward the horizon to ue, matched to the series there"""
    ue = mpf(ue)

    def F(z):
        b0, y0, bh = z
        be_, ye_, J, steps, ok = T.integrate(T.u0, ue, b0, y0)
        B, Yv, dB, dY, nt, conv = T.hseries(bh, ue)
        ph, pb, py = T.phi(b0, y0)
        if not ok or (STRICT and not conv):
            big = mpf(10) ** 50
            return [big, big, ph], [[1, 0, 0], [0, 1, 0], [pb, py, 0]], dict(ok=False)
        return ([B - be_, Yv - ye_, ph], [[-J[0][0], -J[0][1], dB], [-J[1][0], -J[1][1], dY], [pb, py, mpf(0)]],
                dict(nt=nt, conv=conv, steps=steps))
    return F


def reseed(T, b0, y0, levels=5):
    """the horizon datum of the regular member through the node's b0: integrate the node toward the horizon to
    u0/2^levels (the mass mode and the singular branch both decay that way), invert the series there starting from
    the integrated b, then invert again at u0/2^(levels-1), ..., u0, each from the last"""
    us = [T.u0 / 2 ** i for i in range(levels + 1)]
    keep = T.tol_dyn
    T.tol_dyn = mpf("1e-8")
    bs = [mpf(b0)]
    b, y = mpf(b0), mpf(y0)
    for i in range(1, levels + 1):
        b, y, J, steps, ok = T.integrate(us[i - 1], us[i], b, y, tang=False)
        if not ok:
            T.tol_dyn = keep
            return None
        bs.append(b)
    bh = bs[-1]
    for i in range(levels, -1, -1):
        bh = T.invert(us[i], bs[i], bh)
        if bh is None:
            break
    T.tol_dyn = keep
    return bh


def res_elim(T):
    """F3e, horizon data eliminated: unknowns (b0, y0); b_h = b_h(b0) by inversion at u0 (warm-started), residual
    [Y(b_h; u0) - y0, phi]"""
    st = {}

    def F(z):
        b0, y0 = z
        bh0 = st.get("bh")
        if bh0 is None:
            bh0 = reseed(T, b0, y0)
        bh = T.invert(T.u0, b0, bh0) if bh0 is not None else None
        ph, pb, py = T.phi(b0, y0)
        if bh is None:
            big = mpf(10) ** 50
            return [big, ph], [[1, 0], [pb, py]], dict(ok=False)
        B, Yv, dB, dY, nt, conv = T.hseries(bh, T.u0)
        st["bh"] = bh
        # d Y_reg / d b0 along the family = dY/dB
        return [Yv - y0, ph], [[dY / dB, mpf(-1)], [pb, py]], dict(nt=nt, conv=conv, bh=bh)
    return F, st


def pade_at(c, u, L):
    """[L/L] Pade approximant of sum c_k u^k at u, from the scaled coefficients c_k u^k (evaluated at 1)"""
    a = [c[k] * u ** k for k in range(2 * L + 1)]
    p, q = mp.pade(a, L, L)
    return mp.polyval(p[::-1], 1) / mp.polyval(q[::-1], 1)


def res_pade(T, L=24):
    """F4, ours continued past the pole: unknowns (b_h, b0, y0); b and y at u0 from [L/L] Pade approximants of the
    horizon series (2L + 1 coefficients) instead of its partial sum; the b_h column by central differences"""
    def ev(bh):
        B, Yv, dB, dY, nt, conv, be, et = T.hseries(bh, T.u0, tang=False, kfix=2 * L + 1, coeffs=True)
        return pade_at(be, T.u0, L), pade_at(et, T.u0, L)

    def F(z):
        bh, b0, y0 = z
        try:
            B, Yv = ev(bh)
            h = abs(bh) * mpf(10) ** (-T.dps // 3)
            Bp, Yp = ev(bh + h)
            Bm, Ym = ev(bh - h)
        except (ZeroDivisionError, ValueError):
            big = mpf(10) ** 50
            return [big, big, big], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dict(ok=False)
        dB, dY = (Bp - Bm) / (2 * h), (Yp - Ym) / (2 * h)
        ph, pb, py = T.phi(b0, y0)
        return [B - b0, Yv - y0, ph], [[dB, mpf(-1), mpf(0)], [dY, mpf(0), mpf(-1)], [mpf(0), pb, py]], dict()
    return F


def norm(v):
    return max(abs(x) for x in v)


def newton(F, z0, dps, maxit=20, T=None):
    """Newton with backtracking on max|F|; with T, inexact: the horizon series is summed to 1e-4 |F| of the current
    iterate (never below T.tol), so that iterates far off pay only for the digits they use"""
    z = [mpf(x) for x in z0]
    if T is not None:
        T.tol_dyn = None                 # the start at full accuracy, so that every comparison is consistent
    if T is not None and os.environ.get("INEXACT_START"):
        # the start summed to 1e-8, tightened until |F| is resolved (|F| >= 1e3 x the tolerance) or at full accuracy:
        # the series' domain is then where it converges at all, not where it converges to P digits within kmax
        tau = mpf("1e-8")
        while True:
            T.tol_dyn = tau
            Fv, J, info = F(z)
            nF = norm(Fv)
            if nF >= 1000 * tau or tau <= T.tol:
                break
            tau = max(nF * mpf("1e-6"), T.tol)
    else:
        Fv, J, info = F(z)
        nF = norm(Fv)
    hist = [float(mp.log10(nF)) if nF > 0 else -999]
    n = len(z)
    for it in range(maxit):
        if nF == 0:
            break
        try:
            dz = mp.lu_solve(mp.matrix(J), mp.matrix([-f for f in Fv]))
        except (ZeroDivisionError, ValueError):
            break
        lam = mpf(1)
        acc = False
        if T is not None:
            T.tol_dyn = nF * mpf("1e-6")
        for _ in range(14):
            zt = [z[i] + lam * dz[i] for i in range(n)]
            Ft, Jt, it_ = F(zt)
            nt = norm(Ft)
            if mp.isfinite(nt) and nt < nF:
                acc = True
                break
            lam /= 2
        if not acc:
            break
        stepn = max(abs(lam * dz[i]) for i in range(n))
        z, Fv, J, nF = zt, Ft, Jt, nt
        hist.append(float(mp.log10(nF)) if nF > 0 else -999)
        if stepn < mpf(10) ** (-dps - 2):
            break
    if T is not None:
        T.tol_dyn = None
    return z, nF, hist

