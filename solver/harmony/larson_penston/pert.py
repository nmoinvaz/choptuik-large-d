"""Radial perturbations e^{sigma s} (s = -ln(-t)) of a self-similar isothermal solution (LP or Hunter A).
Linearized self-similar Euler-Poisson (U = y(w - 1), dM = -y^2[(y+U)dR + R dU]/(sigma - 1)):
  A v' + C v = 0, v = (dR, dU), A = [[yw, R], [1/R, yw]],
  C = [[sigma + 2 + U' + 2U/y, R' + 2R/y], [-R'/R^2 - 2yw/(sigma-1), sigma + U' - 2R/(sigma-1)]].
Regular at the centre (dU ~ y) and analytic at the sonic point; f(sigma) = det[v_centre, v_sonic] at y_m.
usage: pert.py <scriptdir> <bits> <n> <branch> <y*> <rho0> <ym> <ya> <ds> <sigma0> <sigma1> [mutC]"""
import sys, time, math
sys.path.insert(0, sys.argv[1])
from flint import arb, arb_series, ctx
from lp_core import *
bits, n = int(sys.argv[2]), int(sys.argv[3]); ctx.prec = bits
br = sys.argv[4]; ys, rho0 = arb(sys.argv[5]), arb(sys.argv[6])
ym, ya, ds = arb(sys.argv[7]), arb(sys.argv[8]), arb(sys.argv[9])
s0, s1 = arb(sys.argv[10]), arb(sys.argv[11])
mutC = arb(sys.argv[12]) if len(sys.argv) > 12 else arb(1)   # mutation: scales the self-gravity term

def mats(Rb, Wb, Y, sig, L):
    U = Y * (Wb - 1); Up = U.derivative(); Rp = Rb.derivative()
    A = [[Y * Wb, Rb], [1 / Rb, Y * Wb]]
    C = [[sig + 2 + Up + 2 * U / Y, Rp + 2 * Rb / Y],
         [-Rp / (Rb * Rb) - mutC * 2 * Y * Wb / (sig - 1), sig + Up - mutC * 2 * Rb / (sig - 1)]]
    return A, C

def coeff(s, k):
    c = s.coeffs(); return c[k] if k < len(c) else arb(0)

def resid_series(Rb, Wb, Y, p, q, sig):
    A, C = mats(Rb, Wb, Y, sig, None)
    pp, qp = p.derivative(), q.derivative()
    e1 = A[0][0] * pp + A[0][1] * qp + C[0][0] * p + C[0][1] * q
    e2 = A[1][0] * pp + A[1][1] * qp + C[1][0] * p + C[1][1] * q
    return e1, e2

def sonic_pert(rc, wc, sig, N):
    """Analytic perturbation at the sonic point, v_n = a_n r + b_n s, r = (1, -y*), s = (1, 0), b_0 = 1."""
    a = [arb(0)] * (N + 2); b = [arb(0)] * (N + 2); b[0] = arb(1)
    for m in range(N):
        L = m + 3
        Rb = arb_series(rc[:L], prec=L); Wb = arb_series(wc[:L], prec=L); Y = arb_series([ys, 1], prec=L)
        def ev(am, bm1):
            aa = a[:m] + [am] + [arb(0)] * 2; bb = b[:m + 1] + [bm1] + [arb(0)]
            p = arb_series([aa[k] + bb[k] for k in range(L)], prec=L)
            q = arb_series([-ys * aa[k] for k in range(L)], prec=L)
            e1, e2 = resid_series(Rb, Wb, Y, p, q, sig)
            return coeff(e1, m), coeff(e2, m)
        sc = arb(2) ** int(max(0, math.log2(1e-300 + abs(float(a[m - 1].mid())) + abs(float(b[m].mid()))) + 4 * m)) if m > 0 else arb(1)
        F0 = ev(arb(0), arb(0)); Fa = ev(sc, arb(0)); Fb = ev(arb(0), sc)
        a11, a21 = (Fa[0] - F0[0]) / sc, (Fa[1] - F0[1]) / sc; a12, a22 = (Fb[0] - F0[0]) / sc, (Fb[1] - F0[1]) / sc
        det = a11 * a22 - a12 * a21
        a[m] = mid((-F0[0] * a22 + a12 * F0[1]) / det); b[m + 1] = mid((-a11 * F0[1] + a21 * F0[0]) / det)
    p = [a[k] + b[k] for k in range(N)]; q = [-ys * a[k] for k in range(N)]
    return p, q

def centre_pert(rc, wc, sig, N):
    """Regular perturbation at y = 0: dU odd with q_1 = 1, dR even."""
    p = [arb(0)] * (N + 2); q = [arb(0)] * (N + 2)
    q[1] = arb(1); p[0] = mid(-3 * rc[0] * q[1] / sig)
    for j in range(1, N // 2):
        L = 2 * j + 3
        Rb = arb_series(rc[:L], prec=L); Wb = arb_series(wc[:L], prec=L); Y = arb_series([0, 1], prec=L)
        def ev(pv, qv):
            pp = list(p); qq = list(q); pp[2 * j] = pv; qq[2 * j + 1] = qv
            P = arb_series(pp[:L], prec=L); Q = arb_series(qq[:L], prec=L)
            # avoid 1/Y: multiply both equations by Y
            A, C = mats(Rb, Wb, Y, sig, None) if False else (None, None)
            U = Y * (Wb - 1); Up = U.derivative(); Rp = Rb.derivative()
            e1 = Y * (Y * Wb * P.derivative() + Rb * Q.derivative() + (sig + 2 + Up) * P + Rp * Q) + 2 * U * P + 2 * Rb * Q
            e2 = Y * (P.derivative() / Rb + Y * Wb * Q.derivative() + (-Rp / (Rb * Rb) - mutC * 2 * Y * Wb / (sig - 1)) * P
                      + (sig + Up - mutC * 2 * Rb / (sig - 1)) * Q)
            return coeff(e1, 2 * j + 1), coeff(e2, 2 * j)     # Y*continuity at y^{2j+1}, Y*momentum at y^{2j}
        sc = arb(2) ** int(max(0, math.log2(1e-300 + abs(float(p[2 * j - 2].mid())) + abs(float(q[2 * j - 1].mid()))) + 12 * j))
        F0 = ev(arb(0), arb(0)); Fa = ev(sc, arb(0)); Fb = ev(arb(0), sc)
        a11, a21 = (Fa[0] - F0[0]) / sc, (Fa[1] - F0[1]) / sc; a12, a22 = (Fb[0] - F0[0]) / sc, (Fb[1] - F0[1]) / sc
        det = a11 * a22 - a12 * a21
        p[2 * j] = mid((-F0[0] * a22 + a12 * F0[1]) / det); q[2 * j + 1] = mid((-a11 * F0[1] + a21 * F0[0]) / det)
    return p[:N], q[:N]

def step_full(y0, r0, w0, p0, q0, sig, N):
    Y = arb_series([y0, 1], prec=N)
    R = arb_series([r0], prec=N); W = arb_series([w0], prec=N)
    P = arb_series([p0], prec=N); Q = arb_series([q0], prec=N)
    for _ in range(N + 1):
        f, g = rhs(R, W, Y, N)
        A, C = mats(R, W, Y, sig, None)
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        c1 = C[0][0] * P + C[0][1] * Q; c2 = C[1][0] * P + C[1][1] * Q
        dp = -(A[1][1] * c1 - A[0][1] * c2) / det; dq = -(-A[1][0] * c1 + A[0][0] * c2) / det
        R = ser((arb_series([r0], prec=N) + f.integral()).coeffs()[:N], N)
        W = ser((arb_series([w0], prec=N) + g.integral()).coeffs()[:N], N)
        P = ser((arb_series([p0], prec=N) + dp.integral()).coeffs()[:N], N)
        Q = ser((arb_series([q0], prec=N) + dq.integral()).coeffs()[:N], N)
    return [S.coeffs() for S in (R, W, P, Q)]

def integ_full(y0, st, y1, sig, N):
    import lp_core
    tol = 0.30103 * bits * 0.9
    y = y0; sgn = 1 if float((y1 - y0).mid()) > 0 else -1
    while True:
        cs = step_full(y, *st, sig, N)
        rad = lp_core.radius_est(cs, N)
        h = rad * 10 ** (-tol / N) * 0.8
        rem = abs(float((y1 - y).mid())); last = h >= rem
        hh = (y1 - y) if last else arb(h) * sgn
        st = [mid(evalpoly(c, hh)) for c in cs]
        y = y1 if last else mid(y + hh)
        if last:
            return st

rcs, wcs, _ = sonic_series(ys, n + 4, br)
rco, wco = origin_series(rho0, n + 4)

def f(sig):
    ps, qs = sonic_pert(rcs, wcs, sig, n)
    st = [mid(evalpoly(c[:n], -ds)) for c in (rcs, wcs, ps, qs)]
    sr = integ_full(mid(ys - ds), st, ym, sig, n)
    pc, qc = centre_pert(rco, wco, sig, n)
    st = [mid(evalpoly(c[:n], ya)) for c in (rco, wco, pc, qc)]
    sl = integ_full(ya, st, ym, sig, n)
    return mid(sl[2] * sr[3] - sl[3] * sr[2]), (sl, sr)

t0 = time.time()
x0, x1 = s0, s1; f0 = f(x0)[0]; f1 = f(x1)[0]; print("f0", f0.str(10), "f1", f1.str(10), flush=True)
for it in range(30):
    x2 = mid(x1 - f1 * (x1 - x0) / (f1 - f0))
    x0, f0 = x1, f1; x1 = x2; f1, (sl, sr) = f(x1)
    print('it', it, 'sigma', x1.mid().str(25, radius=False), '|f| %.3e' % abs(float(f1.mid())), 'bg mismatch %.1e' % abs(float((sl[0] - sr[0]).mid())), '%.0fs' % (time.time() - t0), flush=True)
    if abs(float((x1 - x0).mid())) < 2.0 ** (-bits * 0.8):
        break
d = int(bits * 0.30103) - 6
print('RESULT', br, 'bits', bits, 'n', n, 'ym', sys.argv[7], 'ya', sys.argv[8], 'ds', sys.argv[9], 'mutC', sys.argv[12] if len(sys.argv) > 12 else 1)
print('sigma =', x1.mid().str(d, radius=False))
print('gamma = 1/sigma =', (1 / x1).mid().str(d, radius=False))
