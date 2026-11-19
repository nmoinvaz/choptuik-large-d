#!/usr/bin/env python3
"""P6 route 3: a remainder bound |DR_j / K_nu| <= C_mu |p| nu / j^2 for j >= j0 = max(kappa nu, J0), with explicit
constants (notes/P6_route3.md section 10, #990).

The model is route3_cancel.py's nu case with one perturbation of label mu in {nu - 2, nu, nu + 2} and amplitude p on the
A sin theta background (b_p = -i, b_m = i). DR_j / K_nu splits exactly as

    DR_j / K_nu = sum_lam C_lam(j) rho^lam_j + r_j,

C_lam the exact terms of route3_cancel.tail() (near end i = 1, 2, far end i = j + 1 at l = 0 and j, de-mixing, psi) and
r_j the interior convolution terms (Ss and Sd at 3 <= i <= j, and g_{j+1}'s inner terms 1 <= l <= j - 1).

Ingredients, all at the physical w = 2 pi / 1.64389375 (an arb ball):
  S   certified sup over s = 1/nu in [0, 1/3] and u = nu/j in [0, 1/kappa] of j^2 |C_lam| / nu^d (d = 1 for the
      family mu, d = 2 for the families +-1), by interval evaluation of the exact rational function on a grid of boxes;
  Cr  certified sup over x = 1/j in [0, 1/J0] of j^2 |r_j| / (nu P_mu |p|), from the reduction lemma (every interior
      triple product is at most P_mu t_a t_b t_c / t_{j+1}, t = |T^1|) and D_entire.md 6.3's log-convexity grouping;
  P   sup over nu >= 3 of P_{nu+2} = |T^{nu+2}_l / T^nu_l| (closed form, <= e^{pi w} / sqrt(1 - e^{-6 pi w}));
  W   sup over odd nu >= 3 of nu |rho^{+-1}_{j0(nu)}|: an arb product for nu <= 99 and a hand bound beyond.
Then C_mu = S_mu W_mu + S_+ W + S_- W + Cr P_mu, with W_mu = P_mu = 1 for mu = nu - 2, nu and P for mu = nu + 2.

    python3 route3_bound.py --constants [--kappa=K] [--J0=J]    the constants (default kappa 2, J0 10)
    python3 route3_bound.py --check [--kappa=K]                  the bound against the exact recursion, nu = 3..21,
                                                                 j0 <= j <= 1600; exit 0 iff it holds everywhere
    python3 route3_bound.py --check --mutate=plant               a planted violation (3 x the bound added at 5 j's
                                                                 per nu and perturbation): must be caught, exit 1
    python3 route3_bound.py --check --mutate=psi                 h2 x 3/2 in the recursion: an O(1/j) term; reports
                                                                 where it crosses the bound
    python3 route3_bound.py --lemmas                             numerical checks of the hand lemmas
"""
import sys

import mpmath as mp
import sympy as sp
from flint import acb, arb, ctx

import route3_cancel as R
from exact_sigma import ev

ctx.prec = 128
MUT = R.MUT
WV = acb(2 * arb.pi() / arb("1.64389375"))
W_R = WV.real
PERTS = {"m": -2, "0": 0, "p": 2}
NTR = {"m": 3, "0": 6, "p": 3}                    # ordered (lapse pair, y label) triples with one perturbation factor
s_, u_ = sp.symbols("s u", positive=True)


def opt(name, default):
    return type(default)(next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default))


# ---------------------------------------------------------------- S: the C_lam part, certified on (s, u) boxes
def families(pk):
    """{family label (sympy): (Phi(s, u) = j^2 C_lam / nu^d as numerator, denominator), d} for perturbation pk"""
    inp = [(1, -sp.I, 0), (-1, sp.I, 0), (R.nu + PERTS[pk], sp.Integer(1), 1)]
    fam = R.tail(inp, R.nu, 1, mutate="")
    out = {}
    for lam, C in fam.items():
        if all(t == 0 for t in sp.Add.make_args(C)):
            continue
        d = 2 if lam in (1, -1) else 1
        e = sp.cancel(sp.together((C * R.j**2 / R.nu**d).subs({R.j: 1 / (s_ * u_), R.nu: 1 / s_})))
        n, dd = sp.fraction(e)
        # interval evaluation: the denominator as a product of its factors and the numerator in Horner form, both to
        # limit the overestimation of naive interval arithmetic
        out[lam] = (sp.horner(sp.expand(n), wrt=u_), sp.factor(dd), d)
    return out


def box(a, b):
    return acb(arb((a + b) / 2, (b - a) / 2))


def sup_box(num, den, smax, umax, N):
    """(certified upper bound, a lower bound from the box centres) of |num/den| over [0, smax] x [0, umax]"""
    up, lo = arb(0), arb(0)
    for i in range(N):
        for k in range(N):
            sb = box(arb(smax) * i / N, arb(smax) * (i + 1) / N)
            ub = box(arb(umax) * k / N, arb(umax) * (k + 1) / N)
            env = {s_: sb, u_: ub, R.w: WV}
            dv = ev(den, env)
            if dv.contains(0):
                return None, lo                                     # a denominator box contains 0: refine
            v = abs(ev(num, env) / dv)
            up = max(up, v.upper())
            sc, uc = acb(arb(smax) * (i + arb(0.5)) / N), acb(arb(umax) * (k + arb(0.5)) / N)
            envc = {s_: sc, u_: uc, R.w: WV}
            lo = max(lo, abs(ev(num, envc) / ev(den, envc)).lower())
    return up, lo


def S_all(kappa, N=48):
    out = {}
    for pk in PERTS:
        for lam, (n, d, deg) in families(pk).items():
            up, lo = sup_box(n, d, arb(1) / 3, arb(1) / kappa, N)
            out[(pk, str(lam))] = (up, lo, deg)
    return out


# ---------------------------------------------------------------- Cr: the interior, by the reduction lemma
def r_m(m):
    w2 = W_R * W_R / 4
    return ((arb(m) + 0.5) ** 2 + w2).sqrt() * ((arb(m) + 1) ** 2 + w2).sqrt() / (m + 1)


def Cr_sup(kappa, J0, ntr):
    """certified sup over x in [0, 1/J0] of j^2 |r_j| / (nu P |p|), for every nu >= 3 and j >= max(kappa nu, J0)"""
    assert J0 >= 10
    W = (W_R * W_R / 4).upper()
    m0 = int(2 * float(W)) + 2
    for m in range(1, m0 + 1):                                      # r_m increasing for m >= 1 (D_entire.md 6.3)
        assert (r_m(m + 1) - r_m(m)) > 0, "r_m not increasing"
    t1, t2, t3 = r_m(0), r_m(0) * r_m(1), r_m(0) * r_m(1) * r_m(2)
    t1, t2, t3 = t1.upper(), t2.upper(), t3.upper()
    x = arb(1) / (2 * J0) + arb(0, 0.5 / J0 * (1 + 1e-9))          # contains [0, 1/J0]
    om = W_R.upper()
    LWx = (arb(1) / kappa + 2 * x) * om                              # Lambda w x, Lambda = |label of the y slot| <= nu + 2

    def inv(k):
        p = arb(1)
        for q in range(k + 1):
            p *= 1 - (q - 0.5) * x
        return 1 / p

    g = (2 + t1) / 8                                                 # per triple: lapse sum L_n <= (2 + t1) t_n
    c = (2 + 2 * x + om / kappa) / (4 * (1 + x))                     # |c_j| for j >= kappa nu
    A = (g / 3 * t2 * (2 - 5 * x + LWx) * inv(3)                     # Ss i = 3
         + g / 4 * t3 * (2 + LWx) * (1 - 6 * x) * inv(4)             # Ss 4 <= i <= j - 3
         + g * t2 * (5 * x + LWx) * x / (1 - 2 * x) * inv(3)         # Ss i = j - 2
         + g * t1 * (3 * x + LWx) / (1 - x) * inv(2))                # Ss i = j - 1
    B = (g / 3 * t2 * inv(2) + g / 4 * t3 * (1 - 5 * x) * inv(3)    # Sd i = 3, 4 <= i <= j - 2
         + g * t2 * x / (1 - x) * inv(2) + g * t1 * inv(1))          # Sd i = j - 1, j
    Fe = (2 * t1 * inv(1) + (1 - 3 * x) * t2 * inv(2)) / (8 * (1 + x))   # Sd i = j + 1, 1 <= l <= j - 1
    # j^2 |r_j| / nu = j^3 |r_j| x s with s = 1/nu <= 1/3; the Ss i = j term carries 1 + Lambda w, Lambda = nu + 2:
    # x s (1 + (nu + 2) w) = x s + x (1 + 2 s) w <= x/3 + 5 x w / 3
    A5 = g * inv(1) * (x / 3 + 5 * x * om / 3)
    tot = ntr * (x / 3 * (c * A + B + Fe) + c * A5)
    return abs(tot).upper()


# ---------------------------------------------------------------- P and W: the weights
def P_closed(nu):
    """P_{nu+2} = |Gamma(a_nu)Gamma(b_nu) / (Gamma(a_{nu+2})Gamma(b_{nu+2}))| = sqrt(nu sinh(pi (nu+2) w) /
    ((nu+2) sinh(pi nu w))), from |Gamma(1/2 + iy)|^2 = pi/cosh(pi y), |Gamma(1 + iy)|^2 = pi y/sinh(pi y)"""
    pi = arb.pi()
    return (arb(nu) * (pi * (nu + 2) * W_R).sinh() / ((nu + 2) * (pi * nu * W_R).sinh())).sqrt()


def P_sup():
    """sup over nu >= 3 of P_{nu+2} <= e^{pi w} / sqrt(1 - e^{-6 pi w}) (sinh(A+B)/sinh(A) <= e^B / (1 - e^{-2A}))"""
    pi = arb.pi()
    return ((pi * W_R).exp() / (1 - (-6 * pi * W_R).exp()).sqrt()).upper()


def rho_pm(nu, j0):
    """|rho^{+-1}_{j0}| = prod_{m < j0} |a^1_m / a^nu_m| (|T^{-1}| = |T^1| for real w), an arb upper bound"""
    p = arb(1)
    w2 = W_R * W_R / 4
    for m in range(j0):
        num = ((arb(m) + 0.5) ** 2 + w2) * ((arb(m) + 1) ** 2 + w2)
        den = ((arb(m) + 0.5) ** 2 + nu * nu * w2) * ((arb(m) + 1) ** 2 + nu * nu * w2)
        p *= (num / den).sqrt()
    return p


def j0_of(nu, kappa, J0):
    return max(int(mp.ceil(kappa * nu)), J0)


def W_sup(kappa, J0, nu_max=99):
    """sup over odd nu >= 3 of nu |rho^{+-1}_{j0(nu)}|: arb products for nu <= nu_max, and for nu > nu_max the hand bound
    nu q^M with M = floor(nu w / 4) <= j0, q = ((nu w/4 + 1)^2 + w^2/4) / (nu^2 w^2 / 4), evaluated at nu_max + 2;
    it decreases in nu, since q decreases and nu q^(nu w/4 - 1) has log-derivative 1/nu + (w/4) log q < 0 once q < e^{-1}"""
    best = arb(0)
    for nu in range(3, nu_max + 1, 2):
        best = max(best, (nu * rho_pm(nu, j0_of(nu, kappa, J0))).upper())
    nu = nu_max + 2
    q = ((nu * W_R / 4 + 1) ** 2 + W_R * W_R / 4) / (nu * nu * W_R * W_R / 4)
    assert q.upper() < arb(-1).exp().lower()
    M = int(float((nu * W_R / 4).lower()))                          # M <= floor(nu w / 4) <= j0, factors m < M
    tail = nu * q ** M
    return max(best, tail.upper()), tail.upper()


# ---------------------------------------------------------------- assembly
def constants(kappa, J0, N=48, quiet=False):
    S = S_all(kappa, N)
    P = P_sup()
    W, Wtail = W_sup(kappa, J0)
    Cr = {pk: Cr_sup(kappa, J0, NTR[pk]) for pk in PERTS}
    C = {}
    for pk, sh in PERTS.items():
        w_mu = P if pk == "p" else arb(1)
        tot = arb(0)
        for (q, lam), (up, lo, deg) in S.items():
            if q != pk:
                continue
            tot += up * (W if deg == 2 else w_mu)
        tot += Cr[pk] * w_mu
        C[pk] = tot.upper()
    if not quiet:
        print(f"kappa = {kappa}, J0 = {J0}, w = {W_R.str(12)}, grid {N} x {N}")
        print("| perturbation | family | sup j^2 |C_lam| / nu^d, certified | lower bound (box centres) | d |")
        print("|---|---|---|---|---|")
        for (pk, lam), (up, lo, deg) in S.items():
            print(f"| {pk} | {lam} | {up.str(5, radius=False)} | {lo.str(5, radius=False)} | {deg} |")
        print(f"P = sup P_(nu+2) <= {P.str(6, radius=False)}  (e^(pi w) = {(arb.pi() * W_R).exp().str(6, radius=False)})")
        print(f"W = sup nu |rho^(+-1)_j0| <= {W.str(4, radius=False)}  (hand tail at nu = 101: {Wtail.str(3, radius=False)})")
        for pk in PERTS:
            print(f"Cr({pk}) = sup j^2 |r_j| / (nu P |p|) <= {Cr[pk].str(5, radius=False)}  ({NTR[pk]} triples)")
        for pk in PERTS:
            print(f"C_{pk} = {C[pk].str(6, radius=False)}:  |DR_j/K_nu| <= C_{pk} |p| nu / j^2  for j >= max({kappa} nu, {J0})")
    return C, S, P, W, Cr


# ---------------------------------------------------------------- the check against the exact recursion
def check(kappa, J0, Jmax=1600):
    C, S, P, W, Cr = constants(kappa, J0, quiet=True)
    from flint import ctx as fctx
    fctx.prec = 512                                              # the recursion loses ~120 digits at nu 21, J 1600 (#986)
    wv = acb(2 * arb.pi() / arb("1.64389375"))                   # w rebuilt at this precision (its radius propagates)
    ok = True
    worst = {}
    rows = []
    for nu in range(3, 23, 2):
        j0 = j0_of(nu, kappa, J0)
        for pk, sh in PERTS.items():
            amp = acb("0.7", "0.3")
            labels = [(1, acb(0, -1), 0), (-1, acb(0, 1), 0), (nu + sh, amp, 1)]
            if sh != 0:
                labels.append((nu, acb(0), 1))
            mut = "psi" if "psi" in MUT else ""
            DR, T, TN = R.dr_builder(labels, nu, Jmax, wv, 1, mut)
            Cb = float(C[pk].upper())
            plant = set(range(j0, Jmax, max(1, (Jmax - j0) // 5))[:5]) if "plant" in MUT else set()
            mx, at, nviol, first, rmax = 0.0, None, 0, None, 0.0
            for jv in range(j0, Jmax):
                val = DR(jv) / amp
                a = float(abs(val).upper())
                rel = float(val.rad()) / max(float(abs(val).mid()), 1e-300)
                rmax = max(rmax, rel)
                if rel > 1e-6:
                    raise SystemExit(f"precision lost at nu {nu}, {pk}, j {jv}: relative radius {rel:.1e}")
                bnd = Cb * nu / jv ** 2
                if jv in plant:
                    a += 3 * bnd                                        # planted violation
                rat = a / bnd
                if rat > mx:
                    mx, at = rat, jv
                if a > bnd:
                    nviol += 1
                    first = first or jv
            rows.append((nu, pk, j0, mx, at, nviol, first, rmax))
            ok &= nviol == 0
            worst[pk] = max(worst.get(pk, 0.0), mx)
    print(f"kappa = {kappa}, J0 = {J0}; C_m = {float(C['m']):.4g}, C_0 = {float(C['0']):.4g}, C_p = {float(C['p']):.4g}")
    print("| nu | perturbation | j0 | max |DR/K| / bound | at j | violations | first | max relative radius |")
    print("|---|---|---|---|---|---|---|---|")
    for nu, pk, j0, mx, at, nv, first, rm in rows:
        print(f"| {nu} | {pk} | {j0} | {mx:.3e} | {at} | {nv} | {first if first else '-'} | {rm:.1e} |")
    print("worst ratio by perturbation: " + ", ".join(f"{k} {v:.3e}" for k, v in worst.items()))
    return ok


def lemmas():
    ok = True
    mp.mp.dps = 40
    w = mp.mpf(2) * mp.pi / mp.mpf("1.64389375")
    for y in (mp.mpf("0.7"), w * 3 / 2, w * 21 / 2):                       # the two Gamma identities (DLMF 5.4)
        e1 = abs(mp.gamma(mp.mpf(0.5) + 1j * y)) ** 2 - mp.pi / mp.cosh(mp.pi * y)
        e2 = abs(mp.gamma(1 + 1j * y)) ** 2 - mp.pi * y / mp.sinh(mp.pi * y)
        ok &= abs(e1) < mp.mpf(10) ** -30 * (mp.pi / mp.cosh(mp.pi * y)) and abs(e2) < mp.mpf(10) ** -30 * mp.pi * y / mp.sinh(mp.pi * y)
    print(f"Gamma identities at three y: {'ok' if ok else 'FAIL'}")
    for nu in (3, 9, 21):                  # P closed form against mpmath's Gamma ratio; finite products increase to it
        al = lambda l: ((1 + 1j * l * w) / 2, (2 + 1j * l * w) / 2)
        (A1, B1), (A0, B0) = al(nu + 2), al(nu)
        gr = abs(mp.gamma(A0) * mp.gamma(B0) / (mp.gamma(A1) * mp.gamma(B1)))
        cl = mp.mpf(P_closed(nu).mid().str(30, radius=False))
        prods, prod, m = [], mp.mpf(1), 0
        for M in (100, 1000, 10000):
            while m < M:
                prod *= abs((m + A1) * (m + B1)) / abs((m + A0) * (m + B0))
                m += 1
            prods.append(prod)
        good = abs(gr / cl - 1) < mp.mpf(10) ** -25 and prods[0] < prods[1] < prods[2] < cl
        ok &= good
        print(f"P_(nu+2) at nu = {nu}: closed form {mp.nstr(cl, 12)}, Gamma ratio {mp.nstr(gr, 12)}, products to "
              f"m < 1e2, 1e3, 1e4 {[mp.nstr(v, 8) for v in prods]}  {'ok' if good else 'FAIL'}")
    for nu in (3, 9, 21):                                                   # |rho^{+-1}_j| decreasing, |a^nu_m| >= |a^1_m|
        vals = [rho_pm(nu, jv).mid() for jv in (5, 50, 500)]
        good = vals[0] > vals[1] > vals[2]
        ok &= good
        print(f"|rho^(+-1)_j| at nu = {nu}, j = 5, 50, 500: {[v.str(4, radius=False) for v in vals]}  {'decreasing' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    kappa = opt("kappa", 2.0)
    J0 = opt("J0", 10)
    if "--constants" in sys.argv:
        constants(kappa, J0, N=opt("grid", 48))
        good = True
    elif "--check" in sys.argv:
        good = check(kappa, J0)
    elif "--lemmas" in sys.argv:
        good = lemmas()
    else:
        print(__doc__)
        sys.exit(2)
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
