"""The explicit part of route 3's tail, re-derived for the #1062 review (notes/P6_route3_review.md 2, 5).

For one perturbation mu = nu + k (k = -2, 0, 2) the explicit terms of dR_j/K_nu are, per ordered triple
(x1, x2 | x3), the near end i = 1, 2 of S^s_j and S^d_{j+1}, the far end i = j + 1 of S^d at l = 0 and l = j, and the
de-mixing (lapse harmonic h != 0) or psi (h = 0) term. Each is rho^lam_j times a rational function of j and nu.
C_lam(j) collects family lam. Everything is built here with sympy Polys over QQ_I from model.py's definitions, not
from route3_cancel.py.

Checks:
  - exact: in the variables s = 1/nu, u = nu/j, j^2 C_lam / nu^d = Ntil(s, u) / prod ftil(s, u) with Ntil a polynomial.
    This holds iff C_lam = O(j^-2) at fixed nu (the O(1/j) cancellation, PC3) and the nu^d growth is right.
  - numeric (PC2, part): sum_lam C_lam(j) rho^lam_j equals model.split()'s explicit terms.
  - the certified sup of |Ntil| / prod |ftil| over s in [0, 1/3], u in [0, 1/2], by adaptive bisection in mpmath.iv.

    python3 explicit.py                 the seven sups (about 10 min)
    python3 explicit.py --mutate=demix  M1, E x 3/2: the exact check fails, exit 1
"""
import sys
import time
from collections import Counter
import sympy as sp
from mpmath import mp, iv
import model

j, nu, w = sp.symbols('j nu w')
s, u = sp.symbols('s u')
I = sp.I
GENS = (j, nu, w)


def P(e):
    return sp.Poly(e, *GENS, domain='QQ_I')


# factor keys: ('A', q, lam) = 2j - 2q + 1 + i lam w = 2(j - q + alpha_lam); ('B', q, lam) = 2(j - q + beta_lam);
# ('J',) = j + 1; ('H', lam3) = nu - lam3; ('W',) = w
def fexpr(key):
    if key[0] == 'A':
        return 2 * j - 2 * key[1] + 1 + I * key[2] * w
    if key[0] == 'B':
        return 2 * j - 2 * key[1] + 2 + I * key[2] * w
    if key[0] == 'J':
        return j + 1
    if key[0] == 'H':
        return nu - key[1]
    return w


class Term:
    def __init__(self, num, den=None):
        self.num = num if isinstance(num, sp.Poly) else P(num)
        self.den = Counter(den or {})

    def mul(self, num, den=()):
        t = Term(self.num * (num if isinstance(num, sp.Poly) else P(num)), self.den)
        for k in den:
            t.den[k] += 1
        return t


def R_ratio(lam, k):
    """T^lam_{j-k}/T^nu_{j+1} / rho^lam_j = 1/(a^nu_j prod_{q=1..k} a^lam_{j-q}), a_m = 4(m+1)/(FA FB)"""
    num = 4 * (j + 1)
    den = [('A', 0, nu), ('B', 0, nu)]
    for q in range(1, k + 1):
        num = num * 4 * (j - q + 1)
        den += [('A', q, lam), ('B', q, lam)]
    return Term(num, Counter(den))


def families(k, mutate=()):
    """{family label (sympy expr): [Term]} for the perturbation mu = nu + k"""
    mu = nu + k
    inp = [('b+', sp.Integer(1), -I), ('b-', sp.Integer(-1), I), ('p', mu, sp.Integer(1))]
    fam = {}

    def put(lam, t):
        fam.setdefault(sp.expand(lam), []).append(t)

    for x1 in inp:
        for x2 in inp:
            for x3 in inp:
                if sum(x[0] == 'p' for x in (x1, x2, x3)) != 1:
                    continue
                if sp.expand(x1[1] + x2[1] + x3[1] - nu) != 0:
                    continue
                l1, l2, l3 = x1[1], x2[1], x3[1]
                dd, d3 = x1[2] * x2[2], x3[2]
                T1 = lambda lam: (1 + I * lam * w) / 2 * (2 + I * lam * w) / 2      # T^lam_1 = alpha beta
                g1 = dd / 8
                g2 = dd * (T1(l1) + T1(l2)) / 16
                if 'g2' in mutate:
                    g2 = g2 * sp.Rational(101, 100)
                cj = Term(2 * j + 2 + I * nu * w, {('J',): 1}).mul(sp.Rational(1, 4))
                pre = lambda l: 2 * l + 1 + I * l3 * w
                put(l3, cj.mul(-d3 * g1 * pre(j - 1)).mul(R_ratio(l3, 1).num, R_ratio(l3, 1).den.elements()))
                put(l3, cj.mul(-d3 * g2 * pre(j - 2)).mul(R_ratio(l3, 2).num, R_ratio(l3, 2).den.elements()))
                put(l3, R_ratio(l3, 0).mul(-d3 * g1))
                put(l3, R_ratio(l3, 1).mul(-d3 * g2))
                far = Term(-dd * d3 / 8, {('J',): 1})
                put(l2, far.mul(R_ratio(l2, 0).num, R_ratio(l2, 0).den.elements()))
                if 'dropfar' not in mutate:
                    put(l1, far.mul(R_ratio(l1, 0).num, R_ratio(l1, 0).den.elements()))
                h = sp.expand(l1 + l2)
                if h == 0:
                    put(nu, Term(dd / 8 * d3 * 2, {('A', 0, nu): 1}))
                    put(nu, Term(dd / 8 * d3 * 2, {('B', 0, nu): 1}))
                else:
                    Ec = -dd / 4 / I
                    if 'demix' in mutate:
                        Ec = Ec * sp.Rational(3, 2)
                    hden = [('W',)]
                    if h.free_symbols:                        # h = nu - lam3, lam3 = +-1
                        hden.append(('H', l3))
                        hc = 1
                    else:
                        hc = 1 / h
                    diff = fexpr(('A', 0, l3)) * fexpr(('B', 0, l3)) - fexpr(('A', 0, nu)) * fexpr(('B', 0, nu))
                    put(l3, Term(Ec * hc * d3 * diff, Counter(hden + [('A', 0, nu), ('B', 0, nu)])))
    return fam


def combine(terms):
    """sum of Terms over the common denominator: (N poly, Counter of factor multiplicities)"""
    D = Counter()
    for t in terms:
        for kk, m in t.den.items():
            D[kk] = max(D[kk], m)
    N = P(0)
    for t in terms:
        x = t.num
        for kk, m in D.items():
            e = m - t.den.get(kk, 0)
            if e:
                x = x * P(fexpr(kk)) ** e
        N = N + x
    return N, D


def scaled(N, D, d):
    """Ntil(s, u) = s^(d-2+nJ+nH) u^(nJ-2) N(1/(su), 1/s, w); returns (Ntil dict, ok, worst exponents)"""
    nJ = sum(m for kk, m in D.items() if kk[0] in 'ABJ')
    nH = sum(m for kk, m in D.items() if kk[0] == 'H')
    ea0, eb0 = d - 2 + nJ + nH, nJ - 2
    out = {}
    ok = True
    worst = (10 ** 9, 10 ** 9)
    for (p, q, r), c in N.terms():
        if c == 0:
            continue
        ea, eb = ea0 - p - q, eb0 - p
        worst = (min(worst[0], ea), min(worst[1], eb))
        if ea < 0 or eb < 0:
            ok = False
        out.setdefault((ea, eb), []).append((r, c))
    return out, ok, worst, nJ, nH


def ftil(kk, k):
    """the scaled factor as a function (s, u, w) -> (re, im), for mp or iv numbers"""
    if kk[0] in 'AB':
        q, lam = kk[1], sp.expand(kk[2])
        c0 = (1 if kk[0] == 'A' else 2) - 2 * q
        a_nu = lam.coeff(nu)                  # lam = a_nu nu + b
        b = lam - a_nu * nu
        # 2 + c0 s u + i w (a_nu u + b s u)
        return lambda S, U, W: (2 + c0 * S * U, W * (int(a_nu) * U + int(b) * S * U))
    if kk[0] == 'J':
        return lambda S, U, W: (1 + S * U, 0 * S)
    if kk[0] == 'H':
        lam3 = int(kk[1])
        return lambda S, U, W: (1 - lam3 * S, 0 * S)
    return lambda S, U, W: (W, 0 * S)


def coeff_iv(rc, W):
    """sum_r c_r w^r with c_r in QQ_I, as (re, im) iv intervals"""
    re, im = iv.mpf(0), iv.mpf(0)
    for r, c in rc:
        cr, ci = sp.Rational(sp.re(sp.sympify(c))), sp.Rational(sp.im(sp.sympify(c)))
        wr = W ** r
        re += iv.mpf(cr.p) / cr.q * wr
        im += iv.mpf(ci.p) / ci.q * wr
    return re, im


class Bound:
    """|Ntil| / prod |ftil| on boxes, in mpmath.iv"""

    def __init__(self, Nt, D, k):
        self.W = 2 * iv.pi * 160000 / iv.mpf(263023)
        self.c = {e: coeff_iv(rc, self.W) for e, rc in Nt.items()}
        self.fs = [(ftil(kk, k), m) for kk, m in D.items()]
        self.ea = max(e[0] for e in self.c)
        self.eb = max(e[1] for e in self.c)
        # Horner layout: rows over ea, each a poly in u
        self.rows = {}
        for (ea, eb), v in self.c.items():
            self.rows.setdefault(ea, {})[eb] = v

    def num(self, S, U):
        re, im = iv.mpf(0), iv.mpf(0)
        for ea in range(self.ea, -1, -1):
            row = self.rows.get(ea, {})
            rr, ri = iv.mpf(0), iv.mpf(0)
            for eb in range(self.eb, -1, -1):
                cr, ci = row.get(eb, (0, 0))
                rr = rr * U + cr
                ri = ri * U + ci
            re = re * S + rr
            im = im * S + ri
        return re, im

    def value(self, S, U):
        """interval enclosing |Ntil|/|Dtil| over the box S x U"""
        re, im = self.num(S, U)
        top = iv.sqrt(re ** 2 + im ** 2)
        den = iv.mpf(1)
        for f, m in self.fs:
            fr, fi = f(S, U, self.W)
            fr, fi = iv.mpf(fr), iv.mpf(fi)
            mag = iv.sqrt(fr ** 2 + fi ** 2)
            for _ in range(m):
                den = den * mag
        return top / den


def sup_box(B, smax, umax, tol=0.005, maxboxes=400000):
    """certified upper bound U and point lower bound L of sup |Ntil|/|Dtil| over [0, smax] x [0, umax]"""
    import heapq
    best = iv.mpf(0)
    def ival(a, b):
        return iv.mpf([a, b])
    def evalbox(sa, sb, ua, ub):
        v = B.value(ival(sa, sb), ival(ua, ub))
        c = B.value(iv.mpf((sa + sb) / 2), iv.mpf((ua + ub) / 2))
        return v, c
    heap = []
    sa, sb, ua, ub = mp.mpf(0), mp.mpf(smax), mp.mpf(0), mp.mpf(umax)
    ns, nu_ = 8, 8
    for a in range(ns):
        for b in range(nu_):
            box = (sa + (sb - sa) * a / ns, sa + (sb - sa) * (a + 1) / ns, ua + (ub - ua) * b / nu_, ua + (ub - ua) * (b + 1) / nu_)
            v, c = evalbox(*box)
            best = max(best, c.a)
            heapq.heappush(heap, (-float(v.b), box, v))
    n = len(heap)
    while True:
        negv, box, v = heap[0]
        if v.b <= best.a * (1 + tol) or n >= maxboxes:
            break
        heapq.heappop(heap)
        s0, s1, u0, u1 = box
        sm, um = (s0 + s1) / 2, (u0 + u1) / 2
        for bx in ((s0, sm, u0, um), (sm, s1, u0, um), (s0, sm, um, u1), (sm, s1, um, u1)):
            vv, cc = evalbox(*bx)
            best = max(best, cc.a)
            heapq.heappush(heap, (-float(vv.b), bx, vv))
            n += 1
    Ub = max(item[2].b for item in heap)
    return Ub, best.a, n


def check_numeric(k, Cl, nus=(3, 5, 9, 21), js=(10, 25, 60, 200)):
    """sum_lam C_lam(j) rho^lam_j against model.split()'s explicit terms (mp)"""
    C, W = model.ctx('mp', 320)
    worst = mp.mpf(0)
    for nv in nus:
        for jj in js:
            mu = nv + k
            S_ = model.Seqs(nv, mu, jj + 1, C, W)
            fam, inter = model.split(nv, mu, jj, C, W, S_)
            target = sum(fam.values())
            tot = mp.mpc(0)
            for lam, (N, D) in Cl.items():
                lv = int(lam.subs(nu, nv))
                rho = model.kummer(C, W, lv, jj)[jj] / S_.T[nv][jj]
                num = complex_eval(N, jj, nv, W)
                den = mp.mpc(1)
                for kk, m in D.items():
                    den *= complex_eval(P(fexpr(kk)), jj, nv, W) ** m
                tot += num / den * rho
            worst = max(worst, abs(tot - target) / abs(target))
    return worst


def complex_eval(Pl, jj, nv, W):
    acc = mp.mpc(0)
    for (p, q, r), c in Pl.terms():
        c = sp.sympify(c)
        acc += mp.mpc(mp.mpf(sp.Rational(sp.re(c)).p) / sp.Rational(sp.re(c)).q,
                      mp.mpf(sp.Rational(sp.im(c)).p) / sp.Rational(sp.im(c)).q) * mp.mpf(jj) ** p * mp.mpf(nv) ** q * W ** r
    return acc


def main():
    mutate = set()
    for a in sys.argv[1:]:
        if a.startswith('--mutate='):
            mutate.add(a.split('=', 1)[1])
    quick = '--quick' in sys.argv
    iv.prec = 80
    mp.prec = 320
    ok_all = True
    res = {}
    t0 = time.time()
    for k in (-2, 0, 2):
        fam = families(k, mutate)
        Cl = {lam: combine(ts) for lam, ts in fam.items()}
        num_err = check_numeric(k, Cl, nus=(3, 9), js=(10, 60)) if quick else check_numeric(k, Cl)
        print(f"mu = nu{k:+d}: sum C_lam rho_lam vs model.split explicit terms: {mp.nstr(num_err, 3)}", flush=True)
        if num_err > mp.mpf('1e-30'):
            ok_all = False
        for lam, (N, D) in Cl.items():
            d = 2 if not lam.free_symbols else 1
            Nt, ok, worst, nJ, nH = scaled(N, D, d)
            tag = f"mu = nu{k:+d}, family {lam}"
            print(f"  {tag}: deg_j D = {nJ}, max deg_j N = {N.degree(j)}, d = {d}, scaled exponents >= {worst}: "
                  f"{'bounded' if ok else 'UNBOUNDED'}", flush=True)
            if not ok:
                ok_all = False
                continue
            if quick:
                continue
            B = Bound(Nt, D, k)
            Ub, Lb, n = sup_box(B, mp.mpf(1) / 3, mp.mpf(1) / 2)
            res[f"{k}:{lam}"] = {'U': str(Ub), 'L': str(Lb), 'd': d}
            print(f"    sup j^2|C|/nu^{d}: U <= {model.up(Ub, 5)} (certified), L >= {model.dn(Lb, 5)} (point), {n} boxes,"
                  f" [{time.time() - t0:.0f}s]", flush=True)
    out = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--json=')]
    if out and ok_all:
        import json
        json.dump(res, open(out[0], 'w'), indent=1)
    print(f"explicit.py {'PASS' if ok_all else 'FAIL'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
