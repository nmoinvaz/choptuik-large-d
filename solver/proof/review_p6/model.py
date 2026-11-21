"""Order-3 one-step model of P6 route 3, written from amp_flint.py for the #1062 review.

Target harmonic nu (odd >= 3). Inputs: the background (+1, d0 = -i), (-1, d0 = +i) and one perturbation (mu, p),
kept to first order in p. Each input is its Kummer sequence d0 T^lam_l. Three forms of the increment dR_j/K_nu:

  literal(nu, mu, J): amp_flint's pass_even(2), pass_odd(3) and the order-3 part of S_at over labels, with p carried
      as a dual number (p^2 dropped), then (R_{j+1} - R_j)/K_nu. Uses psi and Gamma; mp only.
  darboux(nu, mu, J, C): sum over ordered (lapse pair | outer) triples of -b_j/T^nu_{j+1} + de-mixing + psi
      difference. No psi, no recursion; C = mp or iv.
  split(nu, mu, j, C): the same increment, term by term: explicit terms by family, and the interior.

References: the amplitude recursion is amp_flint.py's (its docstring); the Darboux form is D_entire.md 6.1.
"""
import mpmath
from mpmath import mp, iv

OMEGA_NUM, OMEGA_DEN = 160000, 263023            # 1.64389375 = 263023/160000 exactly; w = 2 pi / 1.64389375

MUTATE = set()                                   # model-level planted errors, set by the drivers


def ctx(kind='mp', prec=320):
    C = mp if kind == 'mp' else iv
    C.prec = prec
    w = 2 * C.pi * C.mpf(OMEGA_NUM) / C.mpf(OMEGA_DEN)
    return C, w


def cplx(C, re, im):
    return C.mpc(re, im)


def ab(C, w, lam):
    """alpha_lam, beta_lam"""
    return cplx(C, C.mpf(1) / 2, lam * w / 2), cplx(C, C.mpf(1), lam * w / 2)


def kummer(C, w, lam, L):
    """T^lam_l, l = 0..L: T_0 = 1, T_{m+1} = T_m (m + alpha)(m + beta)/(m + 1)"""
    a, b = ab(C, w, lam)
    T = [cplx(C, 1, 0)]
    for m in range(L):
        T.append(T[-1] * (m + a) * (m + b) / (m + 1))
    return T


def inputs(C, mu):
    return [('b+', 1, cplx(C, 0, -1)), ('b-', -1, cplx(C, 0, 1)), ('p', mu, cplx(C, 1, 0))]


def triples(C, nu, mu):
    """ordered (x1, x2 | x3) with exactly one perturbation and labels adding to nu"""
    inp = inputs(C, mu)
    out = []
    for x1 in inp:
        for x2 in inp:
            for x3 in inp:
                npert = sum(x[0] == 'p' for x in (x1, x2, x3))
                if npert == 1 and x1[1] + x2[1] + x3[1] == nu:
                    out.append((x1, x2, x3))
    if 'droppair' in MUTATE:                     # M6: drop one ordered lapse pair
        out = out[1:]
    return out


# ---------------------------------------------------------------- literal form (amp_flint over labels, dual in p)

def _dmul(x, y):
    return (x[0] * y[0], x[0] * y[1] + x[1] * y[0])


def _dadd(x, y):
    return (x[0] + y[0], x[1] + y[1])


def literal(nu, mu, J, prec=320):
    """(R_j / K_nu) for j = 0..J, p-coefficient, from amp_flint's passes written over labels; mp only"""
    C, w = ctx('mp', prec)
    Z = (C.mpc(0), C.mpc(0))
    labs = sorted({1, -1, mu})
    d1 = {}                                       # d1[lam][l] = dual
    for name, lam, d0 in inputs(C, mu):
        T = kummer(C, w, lam, J + 2)
        if lam not in d1:
            d1[lam] = [Z] * (J + 3)
        for l in range(J + 3):
            v = d0 * T[l]
            add = (C.mpc(0), v) if name == 'p' else (v, C.mpc(0))
            d1[lam][l] = _dadd(d1[lam][l], add)
    # pass_even(2): e_m[h] = 1/4 sum_{lam1+lam2=h} sum_{l<m} d1[lam1]_l d1[lam2]_{m-1-l};  g_m = e_m/(2m)
    pairs = [(l1, l2) for l1 in labs for l2 in labs]
    if 'droppair' in MUTATE:
        pairs = pairs[1:]
    g = {}
    for l1, l2 in pairs:
        h = l1 + l2
        if h not in g:
            g[h] = [Z] * (J + 3)
        for m in range(1, J + 3):
            acc = Z
            for l in range(m):
                acc = _dadd(acc, _dmul(d1[l1][l], d1[l2][m - 1 - l]))
            g[h][m] = _dadd(g[h][m], (acc[0] / (8 * m), acc[1] / (8 * m)))
    s1 = {lam: [((2 * l + 1 + cplx(C, 0, lam * w)) * x[0], (2 * l + 1 + cplx(C, 0, lam * w)) * x[1])
                for l, x in enumerate(d1[lam])] for lam in labs}
    # pass_odd(3) at label nu: s_j = (2j+1+i nu w) d_j - Ss_j, d_{j+1} = (2j+2+i nu w) s_j/(4(j+1)) - Sd_{j+1}
    def conv_at(j, src, imax):
        acc = Z
        for h, gh in g.items():
            lam = nu - h
            if lam not in src:
                continue
            for i in range(1, imax + 1):
                acc = _dadd(acc, _dmul(gh[i], src[lam][j - i]))
        return acc
    snu = cplx(C, 0, nu * w)
    d3 = [Z]
    for j in range(J + 1):
        Ss = conv_at(j, s1, j)
        sj = _dadd(((2 * j + 1 + snu) * d3[j][0], (2 * j + 1 + snu) * d3[j][1]), (-Ss[0], -Ss[1]))
        Sd = conv_at(j + 1, d1, j + 1)
        cj = (2 * j + 2 + snu) / (4 * (j + 1))
        d3.append((cj * sj[0] - Sd[0], cj * sj[1] - Sd[1]))
    # S_at at order 3: R_j = base_j [d3_j + sum_lam E[nu-lam] d1_j[lam] + d1_j[nu] (psi(a+j)+psi(b+j)) h2]
    F = {lam: (d1[lam][0][0] / 2, d1[lam][0][1] / 2) for lam in labs}
    F2 = {}
    for l1 in labs:
        for l2 in labs:
            F2[l1 + l2] = _dadd(F2.get(l1 + l2, Z), _dmul(F[l1], F[l2]))
    E = {h: ((-v[0] / (cplx(C, 0, h * w)), -v[1] / (cplx(C, 0, h * w))) if h != 0 else Z) for h, v in F2.items()}
    h2 = (F2.get(0, Z)[0] / 2, F2.get(0, Z)[1] / 2)
    a, b = ab(C, w, nu)
    K = C.rgamma(a) * C.rgamma(b)
    R = []
    for j in range(J + 1):
        z = d3[j]
        for lam in labs:
            if nu - lam in E:
                z = _dadd(z, _dmul(E[nu - lam], d1[lam][j]))
        if nu in d1:
            ps = C.psi(0, a + j) + C.psi(0, b + j)
            z = _dadd(z, _dmul((d1[nu][j][0] * ps, d1[nu][j][1] * ps), h2))
        base = C.factorial(j) * C.rgamma(a + j) * C.rgamma(b + j)
        R.append(base * z[1] / K)
    return R


# ---------------------------------------------------------------- Darboux form over triples (mp or iv)

class Seqs:
    """Kummer sequences and lapse convolutions for one (nu, mu), up to index L"""

    def __init__(self, nu, mu, L, C, w):
        self.nu, self.mu, self.L, self.C, self.w = nu, mu, L, C, w
        self.T = {lam: kummer(C, w, lam, L + 1) for lam in {1, -1, mu, nu}}
        self.tr = triples(C, nu, mu)
        self.G = {}                               # G[(lam1, lam2)][m] = sum_{l<m} T^l1_l T^l2_{m-1-l}, m = 0..L
        for x1, x2, _ in self.tr:
            key = (x1[1], x2[1])
            if key in self.G:
                continue
            T1, T2 = self.T[key[0]], self.T[key[1]]
            row = [cplx(C, 0, 0)]
            for m in range(1, L + 1):
                acc = cplx(C, 0, 0)
                for l in range(m):
                    acc += T1[l] * T2[m - 1 - l]
                row.append(acc)
            self.G[key] = row

    def g(self, x1, x2, i):
        return x1[2] * x2[2] * self.G[(x1[1], x2[1])][i] / (8 * i)


def darboux(nu, mu, J, C=None, w=None, S=None):
    """dR_j/K_nu for j = 0..J-1, p-coefficient, summed over triples"""
    if C is None:
        C, w = ctx('mp')
    S = S or Seqs(nu, mu, J + 1, C, w)
    T, Tn = S.T, S.T[nu]
    a, b = ab(C, w, nu)
    snu = cplx(C, 0, nu * w)
    out = []
    for j in range(J):
        cj = (2 * j + 2 + snu) / (4 * (j + 1))
        inc = cplx(C, 0, 0)
        for x1, x2, x3 in S.tr:
            l3, d3 = x3[1], x3[2]
            T3 = T[l3]
            Ss = cplx(C, 0, 0)
            for i in range(1, j + 1):
                Ss += S.g(x1, x2, i) * (2 * (j - i) + 1 + cplx(C, 0, l3 * w)) * T3[j - i]
            Sd = cplx(C, 0, 0)
            for i in range(1, j + 2):
                Sd += S.g(x1, x2, i) * T3[j + 1 - i]
            inc -= d3 * (cj * Ss + Sd) / Tn[j + 1]
            h = x1[1] + x2[1]
            if h == 0:
                inc += x1[2] * x2[2] / 8 * d3 * (1 / (a + j) + 1 / (b + j))
            else:
                E = -(x1[2] * x2[2] / 4) / cplx(C, 0, h * w)
                inc += E * d3 * (T3[j + 1] / Tn[j + 1] - T3[j] / Tn[j])
        out.append(inc)
    return out


# ---------------------------------------------------------------- term-by-term split (j >= 3)

def split(nu, mu, j, C=None, w=None, S=None, scale_g2=1, drop_far=False):
    """(explicit terms by family label, interior) of dR_j/K_nu; their sum is darboux()[j]"""
    assert j >= 3
    if C is None:
        C, w = ctx('mp')
    S = S or Seqs(nu, mu, j + 1, C, w)
    T, Tn = S.T, S.T[nu]
    a, b = ab(C, w, nu)
    snu = cplx(C, 0, nu * w)
    cj = (2 * j + 2 + snu) / (4 * (j + 1))
    fam = {}
    inter = cplx(C, 0, 0)

    def put(lam, v):
        fam[lam] = fam.get(lam, cplx(C, 0, 0)) + v

    N = Tn[j + 1]
    for x1, x2, x3 in S.tr:
        l1, l2, l3, d3 = x1[1], x2[1], x3[1], x3[2]
        T3 = T[l3]
        pre = lambda l: 2 * l + 1 + cplx(C, 0, l3 * w)
        g1, g2 = S.g(x1, x2, 1), S.g(x1, x2, 2) * scale_g2
        put(l3, -d3 * cj * g1 * pre(j - 1) * T3[j - 1] / N)            # Ss, i = 1
        put(l3, -d3 * cj * g2 * pre(j - 2) * T3[j - 2] / N)            # Ss, i = 2
        put(l3, -d3 * g1 * T3[j] / N)                                  # Sd, i = 1
        put(l3, -d3 * g2 * T3[j - 1] / N)                              # Sd, i = 2
        far = x1[2] * x2[2] / (8 * (j + 1)) * d3 / N
        put(l2, -far * T[l1][0] * T[l2][j])                           # Sd, i = j + 1, l = 0
        if not drop_far:
            put(l1, -far * T[l1][j] * T[l2][0])                       # Sd, i = j + 1, l = j
        h = l1 + l2
        if h == 0:
            put(nu, x1[2] * x2[2] / 8 * d3 * (1 / (a + j) + 1 / (b + j)))
        else:
            E = -(x1[2] * x2[2] / 4) / cplx(C, 0, h * w)
            if 'demix' in MUTATE:
                E = E * 3 / 2
            put(l3, E * d3 * (T3[j + 1] / N - T3[j] / Tn[j]))
        # interior
        acc = cplx(C, 0, 0)
        for i in range(3, j + 1):
            acc += cj * S.g(x1, x2, i) * pre(j - i) * T3[j - i]
        for i in range(3, j + 1):
            acc += S.g(x1, x2, i) * T3[j + 1 - i]
        inner = cplx(C, 0, 0)
        for l in range(1, j):
            inner += T[l1][l] * T[l2][j - l]
        acc += x1[2] * x2[2] / (8 * (j + 1)) * inner
        inter -= d3 * acc / N
    return fam, inter


def interior_abs_terms(nu, mu, j, C=None, w=None, S=None):
    """sum over every single interior product term of its modulus (the nu = 1 equality test of interior.py)"""
    if C is None:
        C, w = ctx('mp')
    S = S or Seqs(nu, mu, j + 1, C, w)
    T, Tn = S.T, S.T[nu]
    snu = cplx(C, 0, nu * w)
    cj = (2 * j + 2 + snu) / (4 * (j + 1))
    N = abs(Tn[j + 1])
    tot = C.mpf(0)
    for x1, x2, x3 in S.tr:
        l1, l2, l3, d3 = x1[1], x2[1], x3[1], x3[2]
        dd = abs(x1[2] * x2[2] * d3)
        for i in range(3, j + 1):
            for l in range(i):
                tot += dd * abs(cj) / (8 * i) * abs(T[l1][l] * T[l2][i - 1 - l]) * \
                    abs(2 * (j - i) + 1 + cplx(C, 0, l3 * w)) * abs(T[l3][j - i]) / N
                tot += dd / (8 * i) * abs(T[l1][l] * T[l2][i - 1 - l]) * abs(T[l3][j + 1 - i]) / N
        for l in range(1, j):
            tot += dd / (8 * (j + 1)) * abs(T[l1][l] * T[l2][j - l]) / N
    return tot


# ---------------------------------------------------------------- outward rounding for quoted numbers

def _endpoint(x, hi):
    """the exact endpoint of an iv interval (or the exact value of an mpf or int) as a Fraction"""
    from fractions import Fraction
    with mpmath.workprec(4000):                    # wide enough that the conversion is exact
        if hasattr(x, 'a'):
            x = x.b if hi else x.a
        v = mpmath.mpf(x)
    sign, man, exp, _ = v._mpf_                    # man_exp drops the sign; _mpf_ keeps it
    if man == 0:
        return Fraction(0)
    return (-1 if sign else 1) * Fraction(int(man)) * (Fraction(2) ** int(exp))


def _decimal(F, d, upward):
    """F rounded to d significant digits, up (upward) or down, in exact rational arithmetic, as a string"""
    import math
    from fractions import Fraction
    if F == 0:
        return '0'
    sgn = -1 if F < 0 else 1
    A = abs(F)
    e = int(math.floor(math.log10(float(A)))) if 1e-300 < float(A) < 1e300 else len(str(A.numerator)) - len(str(A.denominator))
    while Fraction(10) ** e > A:
        e -= 1
    while Fraction(10) ** (e + 1) <= A:
        e += 1
    k = d - 1 - e
    s = A * Fraction(10) ** k
    up_mag = (sgn > 0) == upward                   # round the magnitude up for an upper end of a positive number
    r = math.ceil(s) if up_mag else math.floor(s)
    if r == 10 ** d:                               # carried into the next decade
        r //= 10
        k -= 1
        e += 1
    digits = str(r)
    if -6 <= e <= 8:
        if k <= 0:
            out = digits + '0' * (-k)
        else:
            digits = digits.rjust(k + 1, '0')
            out = digits[:-k] + '.' + digits[-k:]
    else:
        out = digits[0] + ('.' + digits[1:] if len(digits) > 1 else '') + f'e{e}'
    return ('-' if sgn < 0 else '') + out


def up(x, d=5):
    """the upper end of x rounded up to d significant digits, as a string (exact rational arithmetic)"""
    return _decimal(_endpoint(x, True), d, True)


def dn(x, d=5):
    """the lower end of x rounded down to d significant digits, as a string (exact rational arithmetic)"""
    return _decimal(_endpoint(x, False), d, False)


def ivjson(x, d=45):
    """an interval as [lower end rounded down, upper end rounded up], d digits, for JSON"""
    return [dn(x, d), up(x, d)]
