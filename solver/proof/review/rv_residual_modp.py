#!/usr/bin/env python3
"""Review check R1 (agent RV): the residual numerators of the truncated M = 1 series vanish IDENTICALLY below the orders
fixed_point.py drops them at.

fixed_point.tail_maj discards every residual coefficient below kmin (kmin read off at the box centre as the first
coefficient ball not containing 0). That is sound only if those coefficients are exactly zero as polynomials in (A, w).
Here the whole computation is redone in exact arithmetic over GF(p), p a prime = 1 mod 4 (so that i = sqrt(-1) exists
in GF(p)), at random (A, w):

  * the series by the recursion of series_arb / fixed_point.series_jet (ported, generic number type);
  * the residual of the ORIGINAL xi-form system (I1)-(I3) of lateral_np.py, written out independently here, and
  * the residual numerators Np, Nv, Nc of fixed_point.py's (P, V, C) system, also written out independently,
and the first nonzero order of each is compared with fixed_point's kmins [53, 53, 53, 53, 52, 52, 52].
Every coefficient is a polynomial in (A, w) with rational coefficients whose denominators are products of integers
<= 2N + 2 (invertible mod p); a nonzero polynomial of degree d vanishes at a random point of GF(p)^2 with probability
<= d/p (Schwartz-Zippel), d < 300, p ~ 2^61: three random points make an error below 1e-50.

    python3 rv_residual_modp.py [N=53]
"""
import random
import sys

def is_prime(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


P = (1 << 61) + 1
while not (P % 4 == 1 and is_prime(P)):
    P += 4 if P % 4 == 1 else 1
# i = sqrt(-1): a^((p-1)/4) for a non-residue a
for a in range(2, 1000):
    if pow(a, (P - 1) // 2, P) == P - 1:
        IM = pow(a, (P - 1) // 4, P)
        break
assert IM * IM % P == P - 1


def inv(x):
    return pow(x % P, P - 2, P)


INV2 = inv(2)
ODD, EVEN = (-1, 1), (-2, 0, 2)


# ------------------------------------------------------------------ the series (port of series_arb.SeriesArb.series)
def series(A, w, J):
    def mul(X, Y, keep):
        out = {k: 0 for k in keep}
        for i, xi in X.items():
            for j, yj in Y.items():
                if i + j in out:
                    out[i + j] = (out[i + j] + xi * yj) % P
        return out
    Fh = {1: (-IM * A * INV2) % P, -1: (IM * A * INV2) % P}
    d = [{mu: 2 * Fh[mu] % P for mu in ODD}]
    s, g = [], [{nu: (1 if nu == 0 else 0) for nu in EVEN}]
    e = [{nu: 0 for nu in EVEN}]
    ce = [{nu: 0 for nu in EVEN}]
    for j in range(J):
        sj = {mu: (2 * j + 1 + IM * mu * w) * d[j][mu] % P for mu in ODD}
        for i in range(1, j + 1):
            t = mul(g[i], s[j - i], ODD)
            for mu in ODD:
                sj[mu] = (sj[mu] - t[mu]) % P
        s.append(sj)
        m = j + 1
        em = {nu: 0 for nu in EVEN}
        for i in range(m):
            t = mul(d[i], d[m - 1 - i], EVEN)
            for nu in EVEN:
                em[nu] = (em[nu] + t[nu] * INV2) % P
        e.append(em)
        cm = {nu: 0 for nu in EVEN}
        for i in range(m):
            t = mul(g[i], e[m - i], EVEN)
            for nu in EVEN:
                cm[nu] = (cm[nu] + t[nu]) % P
        ce.append(cm)
        g.append({nu: cm[nu] * inv(2 * m) % P for nu in EVEN})
        num = {mu: (2 * j + 2 + IM * mu * w) * s[j][mu] % P for mu in ODD}
        for i in range(1, j + 2):
            t = mul(g[i], d[j + 1 - i], ODD)
            for mu in ODD:
                num[mu] = (num[mu] - 2 * (j + 1 - i) * t[mu]) % P
        for mm in range(1, j + 2):
            t = mul(ce[mm], d[j + 1 - mm], ODD)
            for mu in ODD:
                num[mu] = (num[mu] - t[mu]) % P
        d.append({mu: num[mu] * inv(2 * j + 2) % P for mu in ODD})
    return d, s, g


# ------------------------------------------------------------------ polynomials mod P (lists, index = power)
def padd(a, b):
    n = max(len(a), len(b))
    return [((a[k] if k < len(a) else 0) + (b[k] if k < len(b) else 0)) % P for k in range(n)]


def psc(a, c):
    return [x * c % P for x in a]


def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1) if a and b else [0]
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % P
    return out


def pder(a):
    return [k * a[k] % P for k in range(1, len(a))] or [0]


def shift(a, m):              # multiply by X^m
    return [0] * m + list(a)


def hconv(Ad, Bd, keep):      # harmonic convolution of dicts of polynomials, projected
    out = {k: [0] for k in keep}
    for i, a in Ad.items():
        for j, b in Bd.items():
            if i + j in keep:
                out[i + j] = padd(out[i + j], pmul(a, b))
    return out


def first_nonzero(a):
    return next((k for k, x in enumerate(a) if x % P), None)


def xi_form(d, s, g, N):
    """ODE residual of the xi-form system (lateral_np.NP.rhs, multiplied out), polynomials in xi"""
    D = {mu: [0] * (2 * N + 1) for mu in ODD}
    S = {mu: [0] * (2 * N + 2) for mu in ODD}
    C = {nu: [0] * (2 * N) for nu in EVEN}
    for j in range(N):
        for mu in ODD:
            D[mu][2 * j + 1] = d[j][mu]
            S[mu][2 * j + 2] = s[j][mu]
        for nu in EVEN:
            C[nu][2 * j] = g[j][nu]
    X = [0, 1]
    dD = {mu: pder(D[mu]) for mu in ODD}
    dS = {mu: pder(S[mu]) for mu in ODD}
    dC = {nu: pder(C[nu]) for nu in EVEN}
    # (I1) xi^2 D' + i mu w xi D - [C S] ;  (I2) xi^2 S' + i mu w xi S - ([C xi D'] + [C E D] - [C D]), E = [D^2/2]
    # (I3) xi C' - [C E]
    CS = hconv(C, S, ODD)
    E = {k: psc(v, INV2) for k, v in hconv(D, D, EVEN).items()}
    CE = hconv(C, E, EVEN)
    CED = hconv(CE, D, ODD)
    CD = hconv(C, D, ODD)
    CxdD = hconv(C, {mu: pmul(X, dD[mu]) for mu in ODD}, ODD)
    W = W_
    r1 = {mu: padd(padd(shift(dD[mu], 2), psc(shift(D[mu], 1), IM * mu * W % P)), psc(CS[mu], P - 1)) for mu in ODD}
    r2 = {mu: padd(padd(shift(dS[mu], 2), psc(shift(S[mu], 1), IM * mu * W % P)),
                   psc(padd(padd(CxdD[mu], CED[mu]), psc(CD[mu], P - 1)), P - 1)) for mu in ODD}
    r3 = {nu: padd(shift(dC[nu], 1), psc(CE[nu], P - 1)) for nu in EVEN}
    return r1, r2, r3


def pvc_form(d, s, g, N):
    """the residual numerators of fixed_point.py, written out from its (P, V, C) system (z = xi^2):
    Np = V + [(C-1)Q] - 2 z P',  Nv = V + [(C-1)Q] + [(C-1)W] + z(-(2+s)Q - (1+s)W + [C E^ P]) - 2 z^2 V',
    Nc = [C E^]/2 - C',  W = V + [(C-1)Q],  E^ = [P^2/2]"""
    Pp = {mu: [d[j][mu] for j in range(N)] for mu in ODD}
    Qp = {mu: [s[j][mu] for j in range(N)] for mu in ODD}
    Cp = {nu: [g[j][nu] for j in range(N)] for nu in EVEN}
    sm = {mu: IM * mu * W_ % P for mu in ODD}
    V = {mu: padd(Qp[mu], psc(Pp[mu], P - (1 + sm[mu]) % P)) for mu in ODD}
    Cm1 = {nu: (padd(Cp[nu], [P - 1]) if nu == 0 else Cp[nu]) for nu in EVEN}
    CQ = hconv(Cm1, Qp, ODD)
    Wd = {mu: padd(V[mu], CQ[mu]) for mu in ODD}
    CW = hconv(Cm1, Wd, ODD)
    Eh = {k: psc(v, INV2) for k, v in hconv(Pp, Pp, EVEN).items()}
    CE = hconv(Cp, Eh, EVEN)
    CEP = hconv(CE, Pp, ODD)
    Np = {mu: padd(padd(V[mu], CQ[mu]), psc(shift(pder(Pp[mu]), 1), P - 2)) for mu in ODD}
    Nv = {}
    for mu in ODD:
        inner = padd(padd(psc(Qp[mu], P - (2 + sm[mu]) % P), psc(Wd[mu], P - (1 + sm[mu]) % P)), CEP[mu])
        Nv[mu] = padd(padd(padd(V[mu], CQ[mu]), padd(CW[mu], shift(inner, 1))), psc(shift(pder(V[mu]), 2), P - 2))
    Nc = {nu: padd(psc(CE[nu], INV2), psc(pder(Cp[nu]), P - 1)) for nu in EVEN}
    return Np, Nv, Nc


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 53
    print(f"GF(p), p = {P} (= 1 mod 4), i = {IM}; truncation N = {N}")
    rng = random.Random(20261021)
    claimed = [53, 53, 53, 53, 52, 52, 52]          # fixed_point.py at the centre (order p-1, v-1, p+1, v+1, c-2, c0, c2)
    allok = True
    for trial in range(3):
        A_, W_ = rng.randrange(1, P), rng.randrange(1, P)
        d, s, g = series(A_, W_, N + 1)
        r1, r2, r3 = xi_form(d, s, g, N)
        xi_orders = {f"I1{mu:+d}": first_nonzero(r1[mu]) for mu in ODD}
        xi_orders.update({f"I2{mu:+d}": first_nonzero(r2[mu]) for mu in ODD})
        xi_orders.update({f"I3{nu:+d}": first_nonzero(r3[nu]) for nu in EVEN})
        Np, Nv, Nc = pvc_form(d, s, g, N)
        ords = []
        for mu in ODD:
            ords += [first_nonzero(Np[mu]), first_nonzero(Nv[mu])]
        ords += [first_nonzero(Nc[nu]) for nu in EVEN]
        ok = ords == claimed
        # the two systems are the same: with z = xi^2, D = xi P, S = z Q, the xi-form residuals are
        # r1 = -z Np,  r2 = -xi (Nv + (1 + s) z Np - [C Np]),  r3 = -2 z Nc   (exact polynomial identities, for any
        # polynomials P, Q, C, not only the series)
        def toxi(pz, odd_shift=0):
            out = [0] * (2 * len(pz) + odd_shift)
            for k, c in enumerate(pz):
                out[2 * k + odd_shift] = c
            return out

        def eq(a, b):
            n = max(len(a), len(b))
            return all(((a[k] if k < len(a) else 0) - (b[k] if k < len(b) else 0)) % P == 0 for k in range(n))
        ident = all(eq(r1[mu], toxi(psc(shift(Np[mu], 1), P - 1))) for mu in ODD)
        ident &= all(eq(r3[nu], toxi(psc(shift(Nc[nu], 1), P - 2))) for nu in EVEN)
        Cp = {nu: [g[j][nu] for j in range(N)] for nu in EVEN}
        CNp = hconv(Cp, Np, ODD)
        ident &= all(eq(r2[mu], toxi(psc(padd(padd(Nv[mu], psc(shift(Np[mu], 1), (1 + IM * mu * W_) % P)),
                                              psc(CNp[mu], P - 1)), P - 1), 1)) for mu in ODD)
        print(f"          identities r1 = -z Np, r2 = -xi (Nv + (1+s) z Np - [C Np]), r3 = -2 z Nc hold exactly: {ident}")
        ok &= ident
        allok &= ok
        print(f" point {trial}: xi-form residual first nonzero powers of xi {xi_orders}")
        print(f"          (P,V,C) numerators first nonzero powers of z {ords}; equal to fixed_point's kmins: {ok}")
    print("VERDICT:", "vanishing orders confirmed (exactly zero below kmin, nonzero at kmin)" if allok else "MISMATCH")
