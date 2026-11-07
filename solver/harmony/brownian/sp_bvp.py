"""High-precision shooting for the Sire-Chavanis scaling equation, SC02 Eq. (97),

  S'' = alpha S - D S^2 + x S' (1 - S) - (D+1) S'/x ,   S'(0) = 0, S(0) = 2 z / D,

with the decaying condition at infinity (no e^{x^2/2} mode), imposed as x S' + alpha S = 0 at
x = X (the growing mode is e^{X^2/2} larger, so the truncation error is suppressed by it).
Taylor-series integration (order N, adaptive step) in mpmath.

alpha(z): the root in alpha of the shooting residual at fixed z.
alpha_max(D) = max_z alpha(z) (the microcanonical exponent, SC02 Sec. 3.4).

Planted truth: alpha = 2 must give S(0) = 4/(D-2) (SC02 (100)).
--mutate=geom puts D in place of D+1 (the planted check must fail).
"""
import sys
import mpmath as mp

MUT = False

def taylor_coeffs(xc, a0, a1, alpha, D, N):
    a = [a0, a1]
    for k in range(0, N - 1):
        # need a_{k+2}
        b = lambda j: (j + 1)*a[j + 1] if 0 <= j and j + 1 < len(a) else mp.mpf(0)
        def p(j):
            if j < 0: return mp.mpf(0)
            return mp.fsum(a[i]*a[j - i] for i in range(j + 1))
        def q(j):
            if j < 0: return mp.mpf(0)
            return mp.fsum(a[i]*(j - i + 1)*a[j - i + 1] for i in range(j + 1))
        am1 = a[k - 1] if k >= 1 else mp.mpf(0)
        geo = D if MUT else D + 1
        rhs = (alpha*(xc*a[k] + am1) - D*(xc*p(k) + p(k - 1))
               + (xc**2*b(k) + 2*xc*b(k - 1) + b(k - 2))
               - (xc**2*q(k) + 2*xc*q(k - 1) + q(k - 2)) - geo*b(k))
        rhs -= (k + 1)*k*a[k + 1]
        a.append(rhs/(xc*(k + 2)*(k + 1)))
    return a

def centre_series(S0, alpha, D, J):
    c = [S0]
    geo = D if MUT else D + 1
    for j in range(1, J + 1):
        sq = mp.fsum(c[i]*c[j - 1 - i] for i in range(j))
        xs = 2*(j - 1)*c[j - 1]
        sxs = mp.fsum(c[i]*2*(j - 1 - i)*c[j - 1 - i] for i in range(j))
        # S'' + geo S'/x at x^{2j-2}: c_j [2j(2j-1) + geo 2j]
        c.append((alpha*c[j - 1] - D*sq + xs - sxs)/(2*j*(2*j - 1) + geo*2*j))
    return c

def shoot(z, alpha, D, X, N=40, x1=mp.mpf('0.5'), tol=None):
    if tol is None:
        tol = mp.mpf(10)**(-mp.mp.dps + 5)
    S0 = 2*z/D
    c = centre_series(S0, alpha, D, 3*N)
    S = mp.fsum(c[j]*x1**(2*j) for j in range(len(c)))
    dS = mp.fsum(2*j*c[j]*x1**(2*j - 1) for j in range(1, len(c)))
    x = x1
    while x < X:
        a = taylor_coeffs(x, S, dS, alpha, D, N)
        # step from the tail of the series
        rad = min(abs(a[k])**(-mp.mpf(1)/k) for k in range(N - 5, N + 1) if a[k] != 0)
        h = min(rad*mp.mpf('0.5'), X - x)
        # check tail size
        while abs(a[N])*h**N > tol*(1 + abs(S)):
            h /= 2
        S, dS = mp.fsum(a[k]*h**k for k in range(N + 1)), mp.fsum(k*a[k]*h**(k - 1) for k in range(1, N + 1))
        x += h
        if abs(S) > mp.mpf(10)**50:
            break
    return X*dS + alpha*S, S

def alpha_of_z(z, D, X, a_guess, **kw):
    f = lambda al: shoot(z, al, D, X, **kw)[0]
    return mp.findroot(f, (a_guess, a_guess + mp.mpf('1e-6')), solver='secant', tol=mp.mpf(10)**(-mp.mp.dps + 8))

def z_of_alpha(alpha, D, X, z_guess, **kw):
    f = lambda z: shoot(z, alpha, D, X, **kw)[0]
    return mp.findroot(f, (z_guess, z_guess*(1 + mp.mpf('1e-6'))), solver='secant', tol=mp.mpf(10)**(-mp.mp.dps + 8))

def alpha_max(D, X, z0, a0, dz=None, iters=4, verbose=True, **kw):
    if dz is None:
        dz = mp.mpf(10)**(-mp.mp.dps//4)
    z = mp.mpf(z0)
    ag = mp.mpf(a0)
    for it in range(iters):
        am = alpha_of_z(z - dz, D, X, ag, **kw)
        a0_ = alpha_of_z(z, D, X, ag, **kw)
        ap = alpha_of_z(z + dz, D, X, ag, **kw)
        d1 = (ap - am)/(2*dz)
        d2 = (ap - 2*a0_ + am)/dz**2
        step = -d1/d2
        amax = a0_ - d1**2/(2*d2)
        if verbose:
            print('  it %d z=%s alpha(z)=%s step=%s alpha_max~%s' % (it, mp.nstr(z, 25), mp.nstr(a0_, 30), mp.nstr(step, 5), mp.nstr(amax, 35)), flush=True)
        z += step
        ag = a0_
        if abs(step) < dz:
            break
    a_fin = alpha_of_z(z, D, X, ag, **kw)
    return a_fin, z, d2

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    MUT = '--mutate=geom' in sys.argv
    D = mp.mpf(args[0]) if args else mp.mpf(3)
    dps = int(args[1]) if len(args) > 1 else 50
    X = mp.mpf(args[2]) if len(args) > 2 else mp.mpf(11)
    N = int(args[3]) if len(args) > 3 else 40
    mp.mp.dps = dps
    # planted truth: alpha = 2 gives S(0) = 4/(D-2)
    zc = 2*D/(D - 2)
    zp = z_of_alpha(mp.mpf(2), D, X, zc*(1 + mp.mpf('0.01')), N=N)
    err = abs(zp/zc - 1)
    print('planted alpha=2: z = %s, exact %s, rel err %s' % (mp.nstr(zp, 30), mp.nstr(zc, 30), mp.nstr(err, 3)), flush=True)
    planted_ok = err < mp.mpf(10)**(-dps//2)
    print('planted check', 'passes' if planted_ok else 'FAILS', flush=True)
    if not planted_ok:
        sys.exit(1)
