"""Larson-Penston / Hunter self-similar isothermal collapse, high precision.

System (Guo-Hadzic-Jang, CMP 386 (2021), eqs. (1.9)-(1.10)), ' = d/dy:
  rho' = -2 y w rho (rho - w) / B,   w' = (1 - 3w)/y + 2 y w^2 (rho - w) / B,   B = 1 - y^2 w^2.
Sonic point y*: B = 0, rho = w = 1/y*. Origin: w(0) = 1/3, rho(0) = rho0.
Series via python-flint arb_series; Taylor stepping at regular points by Picard iteration.
"""
from flint import arb, arb_series, ctx
ctx.cap = 256

def mid(x):
    return arb(x.mid())

def ser(coeffs, n):
    return arb_series([mid(c) for c in coeffs], prec=n)

def rhs(R, W, Y, n, mut=1):
    B = 1 - Y * Y * W * W
    f = -2 * Y * W * R * (R - W) / B
    g = (1 - 3 * W) / Y + 2 * mut * Y * W * W * (R - W) / B
    return f, g

def step_series(y0, r0, w0, n, mut=1):
    """Taylor series of the solution about a regular point y0, length n (Picard)."""
    Y = arb_series([y0, 1], prec=n)
    R = arb_series([r0], prec=n)
    W = arb_series([w0], prec=n)
    for _ in range(n + 1):
        f, g = rhs(R, W, Y, n, mut)
        R = arb_series([r0], prec=n) + f.integral()
        W = arb_series([w0], prec=n) + g.integral()
        R = ser(R.coeffs()[:n], n); W = ser(W.coeffs()[:n], n)
    return R.coeffs(), W.coeffs()

def evalpoly(c, h):
    s = arb(0)
    for a in reversed(c):
        s = s * h + a
    return s

def radius_est(cs, n):
    # crude radius of convergence from the last coefficients
    import math
    vals = []
    for c in cs:
        for k in range(n // 2, n):
            a = abs(float(c[k].mid())) if k < len(c) else 0.0
            if a > 0:
                vals.append(a ** (-1.0 / k))
    return min(vals) if vals else 1.0

def integrate(y0, r0, w0, y1, n=48, tol_digits=None, mut=1):
    """Taylor-step from y0 to y1 (either direction)."""
    if tol_digits is None:
        tol_digits = 0.30103 * ctx.prec * 0.9
    y, r, w = arb(y0), arb(r0), arb(w0)
    y1 = arb(y1)
    sgn = 1 if float(y1.mid()) > float(y.mid()) else -1
    steps = 0
    while True:
        rc, wc = step_series(y, r, w, n, mut)
        rad = radius_est([rc, wc], n)
        h = rad * 10 ** (-tol_digits / n) * 0.8
        rem = abs(float((y1 - y).mid()))
        last = h >= rem
        hh = (y1 - y) if last else arb(h) * sgn
        r, w = mid(evalpoly(rc, hh)), mid(evalpoly(wc, hh))
        y = y1 if last else mid(y + hh)
        steps += 1
        if last:
            return r, w, steps

def origin_series(rho0, n, mut=1):
    """Series about y = 0 with w(0) = 1/3, rho(0) = rho0 (one-parameter regular family)."""
    Y = arb_series([0, 1], prec=n)
    R = arb_series([rho0], prec=n)
    W = arb_series([arb(1) / 3], prec=n)
    for _ in range(n + 2):
        B = 1 - Y * Y * W * W
        f = -2 * Y * W * R * (R - W) / B
        h = 2 * mut * Y * Y * W * W * (R - W) / B      # y w' + 3 w - 1 = h
        R = arb_series([rho0], prec=n) + f.integral()
        hc = (1 + h).coeffs() + [arb(0)] * n
        W = arb_series([hc[k] / (k + 3) for k in range(n)], prec=n)
        R = ser(R.coeffs()[:n], n); W = ser(W.coeffs()[:n], n)
    return R.coeffs(), W.coeffs()

def sonic_series(ys, n, branch='LP', mut=1, d1=0):
    """Series in x = y - y* at the sonic point. LP: (r1, w1) = (-1/y*^2, (1/y*)(1 - 2/y*)).
    Hunter: (r1, w1) = ((1/y*)(1 - 3/y*), 0). d1 perturbs w1 (mutation control)."""
    ys = arb(ys)
    r = [1 / ys] + [arb(0)] * (n - 1)
    w = [1 / ys] + [arb(0)] * (n - 1)
    if branch == 'LP':
        r[1], w[1] = -1 / ys ** 2, (1 / ys) * (1 - 2 / ys) + d1
    else:
        r[1], w[1] = (1 / ys) * (1 - 3 / ys), arb(0) + d1
    Y = arb_series([ys, 1], prec=n)

    def resid(rr, ww, m):
        R = arb_series(rr[:m + 1], prec=m + 1)
        W = arb_series(ww[:m + 1], prec=m + 1)
        Yt = arb_series([ys, 1], prec=m + 1)
        B = 1 - Yt * Yt * W * W
        F = B * R.derivative() + 2 * Yt * W * R * (R - W)
        G = Yt * B * W.derivative() - (1 - 3 * W) * B - 2 * mut * Yt * Yt * W * W * (R - W)
        Fc = F.coeffs() + [arb(0)] * (m + 2); Gc = G.coeffs() + [arb(0)] * (m + 2)
        return Fc[m], Gc[m]
    for m in range(1, n):
        # derivative() of a length-(m+1) series drops the top term; extend by one so that x^m of R' is kept
        def res(a, b):
            rr = r[:m] + [a] + [arb(0)]; ww = w[:m] + [b] + [arb(0)]
            R = arb_series(rr, prec=m + 2); W = arb_series(ww, prec=m + 2)
            Yt = arb_series([ys, 1], prec=m + 2)
            B = 1 - Yt * Yt * W * W
            F = B * R.derivative() + 2 * Yt * W * R * (R - W)
            G = Yt * B * W.derivative() - (1 - 3 * W) * B - 2 * mut * Yt * Yt * W * W * (R - W)
            Fc = F.coeffs() + [arb(0)] * (m + 3); Gc = G.coeffs() + [arb(0)] * (m + 3)
            return Fc[m], Gc[m]
        if m == 1:
            f1, g1 = res(r[1], w[1])
            continue
        F0, G0 = res(arb(0), arb(0))
        Fa, Ga = res(arb(1), arb(0))
        Fb, Gb = res(arb(0), arb(1))
        a11, a12, a21, a22 = Fa - F0, Fb - F0, Ga - G0, Gb - G0
        det = a11 * a22 - a12 * a21
        r[m] = mid((-F0 * a22 + G0 * a12) / det)
        w[m] = mid((-a11 * G0 + a21 * F0) / det)
    return r, w, (f1, g1)
