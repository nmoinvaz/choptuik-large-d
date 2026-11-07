"""Two-sided shooting for the Sire-Chavanis scaling equation, SC02 Eq. (97):

  S'' = alpha S - D S^2 + x S' (1 - S) - (D+1) S'/x ,   S'(0) = 0, S(0) = 2 z / D ,

decaying at infinity, S ~ A x^-alpha (no e^{x^2/2} mode). Outward from 0 (centre series, then
Taylor steps) and inward from X (data A X^-alpha (1, -alpha/X); the error transverse to the
decaying manifold shrinks like e^{-(X^2 - xm^2)/2} on the way in), matched in S, S' at xm.
python-flint arb (midpoints only) at `prec` bits; Taylor order N with adaptive steps.

alpha(z): for fixed z, (A, alpha) from the two matching conditions (Newton).
alpha_max(D) = max_z alpha(z) (the microcanonical exponent of SC02 Sec. 3.4).

Planted truth (run first, same code path): alpha = 2 must give S(0) = 4/(D-2), SC02 (100);
--mutate=geom puts D in place of D+1 in the equation and must make it fail.
"""
import sys, json, time
import flint
from flint import arb

MUT = False

def mid(x):
    return x.mid()

def taylor(xc, a0, a1, alpha, D, N):
    geo = D if MUT else D + 1
    a = [a0, a1]
    p = []   # S^2 coefficients
    q = []   # S S' coefficients
    b = [a1] # S' coefficients b_k = (k+1) a_{k+1}
    xc2 = xc*xc
    zero = arb(0)
    for k in range(0, N - 1):
        p.append(sum((a[i]*a[k - i] for i in range(k + 1)), zero))
        q.append(sum((a[i]*b[k - i] for i in range(k + 1)), zero))
        am1 = a[k - 1] if k >= 1 else zero
        bk1 = b[k - 1] if k >= 1 else zero
        bk2 = b[k - 2] if k >= 2 else zero
        pk1 = p[k - 1] if k >= 1 else zero
        qk1 = q[k - 1] if k >= 1 else zero
        qk2 = q[k - 2] if k >= 2 else zero
        rhs = (alpha*(xc*a[k] + am1) - D*(xc*p[k] + pk1)
               + (xc2*b[k] + 2*xc*bk1 + bk2)
               - (xc2*q[k] + 2*xc*qk1 + qk2) - geo*b[k]
               - (k + 1)*k*a[k + 1])
        ak2 = mid(rhs/(xc*((k + 2)*(k + 1))))
        a.append(ak2)
        b.append((k + 2)*ak2)
    return a

def centre(S0, alpha, D, J):
    geo = D if MUT else D + 1
    c = [S0]
    for j in range(1, J + 1):
        sq = sum((c[i]*c[j - 1 - i] for i in range(j)), arb(0))
        sxs = sum((c[i]*(2*(j - 1 - i))*c[j - 1 - i] for i in range(j)), arb(0))
        c.append(mid((alpha*c[j - 1] - D*sq + 2*(j - 1)*c[j - 1] - sxs)/(2*j*(2*j - 1) + geo*2*j)))
    return c

def integrate(x, S, dS, xend, alpha, D, N, tol):
    nsteps = 0
    sgn = 1 if xend > x else -1
    while (xend - x)*sgn > 0:
        a = taylor(x, S, dS, alpha, D, N)
        r = min(abs(a[k]).mid()**(arb(-1)/k) for k in range(N - 4, N + 1) if a[k] != 0)
        h = min(r*arb('0.4'), abs(xend - x))
        scale = abs(S) + abs(dS)*h
        while abs(a[N])*h**N > tol*scale:
            h = h*arb('0.7')
        h = mid(h)*sgn
        hp = [arb(1)]
        for k in range(N):
            hp.append(hp[-1]*h)
        S, dS = (mid(sum((a[k]*hp[k] for k in range(N + 1)), arb(0))),
                 mid(sum((k*a[k]*hp[k - 1] for k in range(1, N + 1)), arb(0))))
        x = mid(x + h)
        nsteps += 1
    return S, dS, nsteps

class Prob:
    def __init__(self, D, X, xm, N, x1='0.5'):
        self.D, self.X, self.xm, self.N = arb(D), arb(X), arb(xm), N
        self.x1 = arb(x1)
        self.tol = arb(2)**(-flint.ctx.prec + 8)
    def out(self, z, alpha):
        S0 = 2*z/self.D
        c = centre(S0, alpha, self.D, 4*self.N)
        x1 = self.x1
        if S0 > 1:
            x1 = mid(self.x1/S0.sqrt())
        S = sum((c[j]*x1**(2*j) for j in range(len(c))), arb(0))
        dS = sum((2*j*c[j]*x1**(2*j - 1) for j in range(1, len(c))), arb(0))
        S, dS, n = integrate(x1, mid(S), mid(dS), self.xm, alpha, self.D, self.N, self.tol)
        return S, dS
    def inn(self, A, alpha):
        X = self.X
        # S ~ A x^-a [1 + a(D-a)/(2x^2) + A (a-D)/a x^-a]  (linear and first nonlinear correction)
        D = self.D
        c1 = alpha*(D - alpha)/2
        d1 = A*(alpha - D)/alpha
        xa = X**(-alpha)
        S = A*xa*(1 + c1/X**2 + d1*xa)
        dS = A*xa*(-alpha/X + c1*(-alpha - 2)/X**3 + d1*(-2*alpha)*xa/X)
        S, dS, n = integrate(X, mid(S), mid(dS), self.xm, alpha, self.D, self.N, self.tol)
        return S, dS
    def A_guess(self, z, alpha):
        # S(X) from the leading 1/D outer profile sigma_0 = 2z/(1+(z-1)x^2/D), then invert the
        # inward data formula for A (two fixed-point passes for the nonlinear correction)
        D, X = self.D, self.X
        SX = 2*z/(1 + (z - 1)*X*X/D)/D
        xa = X**(-alpha)
        A = SX/xa
        for _ in range(3):
            A = SX/(xa*(1 + alpha*(D - alpha)/2/X**2 + A*(alpha - D)/alpha*xa))
        return mid(A)
    def F(self, z, A, alpha):
        So, dSo = self.out(z, alpha)
        Si, dSi = self.inn(A, alpha)
        return So - Si, dSo - dSi

def solve2(f, u, v, h, itmax=30, tol=None):
    """Newton for f(u, v) = (0, 0) with a forward-difference Jacobian."""
    prev = arb(1)
    for it in range(itmax):
        f1, f2 = f(u, v)
        a1, a2 = f(u + h, v)
        b1, b2 = f(u, v + h)
        J11, J21 = (a1 - f1)/h, (a2 - f2)/h
        J12, J22 = (b1 - f1)/h, (b2 - f2)/h
        det = J11*J22 - J12*J21
        du = -(J22*f1 - J12*f2)/det
        dv = -(-J21*f1 + J11*f2)/det
        u, v = mid(u + du), mid(v + dv)
        nrm = abs(du) + abs(dv)
        if nrm < tol:
            break
        # stagnation at the roundoff floor
        if it > 2 and nrm < arb(2)**(-flint.ctx.prec//2) and nrm > prev*arb('0.25'):
            break
        prev = nrm
    return u, v, it

def alpha_of_z(P, z, A0, al0, tol):
    h = arb(2)**(-flint.ctx.prec//2)
    A, al, it = solve2(lambda A, al: P.F(z, A, al), A0, al0, h, tol=tol)
    return al, A

def z_of_alpha(P, alpha, A0, z0, tol):
    h = arb(2)**(-flint.ctx.prec//2)
    A, z, it = solve2(lambda A, z: P.F(z, A, alpha), A0, z0, h, tol=tol)
    return z, A

def alpha_max(P, z0, A0, al0, tol, verbose=True):
    """Maximum of alpha over u = ln z by Newton on d alpha/du (central differences)."""
    du = arb(2)**(-flint.ctx.prec//4)
    u, A, al = arb(z0).log(), arb(A0), arb(al0)
    for it in range(12):
        am, _ = alpha_of_z(P, (u - du).exp(), A, al, tol)
        a0, A = alpha_of_z(P, u.exp(), A, al, tol)
        ap, _ = alpha_of_z(P, (u + du).exp(), A, al, tol)
        d1 = (ap - am)/(2*du)
        d2 = (ap - 2*a0 + am)/(du*du)
        step = mid(-d1/d2)
        if step > arb('0.5'): step = arb('0.5')
        if step < arb('-0.5'): step = arb('-0.5')
        if verbose:
            print('   it %d  z = %s  alpha(z) = %s  du = %s' % (it, u.exp().str(20, radius=False), a0.str(35, radius=False), step.str(3, radius=False)), flush=True)
        u = mid(u + step)
        al = a0
        if abs(step) < arb(2)**(-flint.ctx.prec//2 + 16):
            break
        if it > 3 and abs(step) < arb(2)**(-flint.ctx.prec//3):
            break
    a_fin, A = alpha_of_z(P, u.exp(), A, al, tol)
    return a_fin, u.exp(), A, d2

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    MUT = '--mutate=geom' in sys.argv
    D = args[0] if args else '3'
    flint.ctx.prec = int(args[1]) if len(args) > 1 else 160
    X = args[2] if len(args) > 2 else '12'
    N = int(args[3]) if len(args) > 3 else 30
    xm = args[4] if len(args) > 4 else '2'
    P = Prob(D, X, xm, N)
    tol = arb(2)**(-flint.ctx.prec + 20)
    Dv = arb(D)
    t0 = time.time()
    zc = 2*Dv/(Dv - 2)
    # planted truth
    zp, Ap = z_of_alpha(P, arb(2), arb(4), zc*arb('1.02'), tol)
    rel = abs(zp/zc - 1)
    print('D=%s prec=%d X=%s N=%d xm=%s' % (D, flint.ctx.prec, X, N, xm))
    print('planted alpha=2: z = %s (exact %s), rel err %s, A = %s  [%.1fs]' % (
        zp.str(30, radius=False), zc.str(30, radius=False), rel.str(3, radius=False), Ap.str(20, radius=False), time.time() - t0), flush=True)
    ok = rel < arb(2)**(-flint.ctx.prec//2)
    print('planted check', 'passes' if ok else 'FAILS', flush=True)
    if not ok or '--planted-only' in sys.argv:
        sys.exit(0 if ok else 1)
    # alpha_max: start from the 1/D expansion, z ~ 4 (+ shift), alpha ~ 2 + 1/(2D) + 11/(16 D^2)
    z0 = arb(args[5]) if len(args) > 5 else 4 + (arb(41)/2 - 6*arb(2).log())/Dv
    al0 = 2 + 1/(2*Dv) + arb(11)/(16*Dv*Dv)
    am, zs, As, d2 = alpha_max(P, z0, Ap, al0, tol)
    print('alpha_max(D=%s) = %s' % (D, am.str(40, radius=False)))
    print('  at z = %s, S(0) = %s, A = %s, d2alpha/dz2 = %s  [%.1fs]' % (zs.str(25, radius=False), (2*zs/Dv).str(25, radius=False), As.str(20, radius=False), d2.str(8, radius=False), time.time() - t0), flush=True)
    rec = dict(D=D, prec=flint.ctx.prec, X=X, N=N, xm=xm, alpha_max=am.str(60, radius=False), z=zs.str(40, radius=False), A=As.str(30, radius=False), planted_rel=rel.str(5, radius=False), secs=round(time.time() - t0, 1))
    if '--json' in sys.argv:
        with open(sys.argv[sys.argv.index('--json') + 1] if False else 'alpha_max_runs.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
