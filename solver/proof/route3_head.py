#!/usr/bin/env python3
"""P6 route 3: the head j < j0 = max(2 nu, 10) of the Darboux sum (notes/P6_route3.md section 11, #1015).

route3_bound.py certifies |DR_j / K_nu| <= C_mu |p| nu / j^2 for j >= j0 in the one-A^2-step model (section 10). The
normalised multiplier is R_0 + sum_{j >= 0} DR_j / K_nu, so the head is the finite sum over 0 <= j < j0, about 2 nu
terms. This script measures it on the exact recursion (route3_cancel.dr_builder), one perturbation of label
mu in {nu - 2, nu, nu + 2} at a time, at the physical w.

    python3 route3_head.py --lemmas               the closed forms and inequalities of the closed form G (section 11.2)
    python3 route3_head.py --check                B7's per-term bound against the exact recursion at every head j,
                                                 nu = 3..21, and sum_j B_j <= G at nu = 3..201; exit 0 iff both hold
    python3 route3_head.py --check --mutate=plant 3 x the bound added at five head j per run: must be caught, exit 1
    python3 route3_head.py --check --mutate=gsmall  G / 100: the second check must fail, exit 1
    python3 route3_head.py --certify [--nu1=99]   Lambda_head = sup e^(-0.736 nu) H_mu(nu): H from the exact recursion in
                                                 arb (512 bits) for nu <= nu1, the closed form G at nu1 + 2 beyond; and
                                                 the whole Darboux sum with section 10's tail
    python3 route3_head.py --scope [--numax=21]   for odd nu = 3..numax and each perturbation: the largest
                                                 j^2 |DR_j/K_nu| / (nu |p|) over the head (against C_mu), the head's
                                                 sum of |DR_j/K_nu| / |p| (the finite-sum bound) and its signed value,
                                                 and the growth in nu of that sum against e^(-0.736 nu)
"""
import math
import sys

from flint import acb, arb, ctx

import route3_cancel as R

MUT = R.MUT
PERTS = {"m": -2, "0": 0, "p": 2}
C_TAIL = {"m": 28.449, "0": 56.874, "p": 4.7661e6}           # section 10.2, kappa = 2, J0 = 10
SIGMA = 0.736                                                # (star)'s rate, B_tail_M7.md 8.4


def opt(name, default):
    return type(default)(next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default))


def j0_of(nu):
    return max(2 * nu, 10)


def head_terms(nu, pk, prec=256, extra=4):
    """|DR_j / K_nu| / |p| for 0 <= j < j0 + extra, and the signed head sum / p, from the exact recursion"""
    ctx.prec = prec
    wv = acb(2 * arb.pi() / arb("1.64389375"))
    amp = acb("0.7", "0.3")
    sh = PERTS[pk]
    labels = [(1, acb(0, -1), 0), (-1, acb(0, 1), 0), (nu + sh, amp, 1)]
    if sh != 0:
        labels.append((nu, acb(0), 1))
    j0 = j0_of(nu)
    J = j0 + extra + 2
    DR, T, TN = R.dr_builder(labels, nu, J, wv, 1, "psi" if "psi" in MUT else "")
    vals, signed = [], acb(0)
    for jv in range(j0 + extra):
        v = DR(jv) / amp
        rel = float(v.rad()) / max(float(abs(v).mid()), 1e-300)
        if rel > 1e-6:
            raise SystemExit(f"precision lost at nu {nu}, {pk}, j {jv}: relative radius {rel:.1e}")
        vals.append(v)
        if jv < j0:
            signed += v
    return vals, signed, j0


def majorants(nu, pk, prec=256):
    """termwise (finite-sum) bounds of |DR_j / K_nu| / |p| for 0 <= j < j0, from the Darboux form
    DR_j/K = -b_j / T^nu_{j+1} + sum_r E_r d0_r (rho^r_{j+1} - rho^r_j) + psi term, b_j = c_j Ss_j + Sd_{j+1},
    with the triangle inequality on every convolution term (order3's sums in absolute value):
      exact    every |T| exact;
      reduced  route3_bound's B3: |T^(+-1)_l| = t_l, |T^mu_l| -> P_mu t_l, |T^nu_(j+1)| -> t_(j+1) (t = |T^1|), and
               |rho^r| <= 1, or P for r = nu + 2 (B2); prefactors exact in nu.
    Returns two lists over j."""
    ctx.prec = prec
    wv = acb(2 * arb.pi() / arb("1.64389375"))
    W = wv.real
    Iv = acb(0, 1)
    sh = PERTS[pk]
    mu = nu + sh
    j0 = j0_of(nu)
    L = j0 + 3
    labs = [(1, acb(0, -1), 0), (-1, acb(0, 1), 0), (mu, acb(1), 1)]
    T = {}
    for lam in {1, -1, mu, nu}:
        a, b = (1 + Iv * lam * wv) / 2, (2 + Iv * lam * wv) / 2
        tt = [acb(1)]
        for l in range(L):
            tt.append(tt[-1] * (l + a) * (l + b) / (l + 1))
        T[lam] = [abs(x) for x in tt]
    t = T[1]
    pi = arb.pi()
    P = (arb(nu) * (pi * (nu + 2) * W).sinh() / ((nu + 2) * (pi * nu * W).sinh())).sqrt() if sh == 2 else arb(1)

    def mag(lam, l, red):
        if not red:
            return T[lam][l]
        return t[l] * (P if lam == mu else 1)

    out = {False: [], True: []}
    for red in (False, True):
        d = {lam: [abs(d0) * mag(lam, l, red) for l in range(L)] for lam, d0, _ in labs}
        s = {lam: [abs(2 * l + 1 + Iv * lam * wv) * d[lam][l] for l in range(L)] for lam, _, _ in labs}
        lapse = {}
        for lp, _, gp in labs:
            for lq, _, gq in labs:
                if gp + gq > 1:
                    continue
                key = (lp + lq, gp + gq)
                acc = lapse.setdefault(key, [arb(0)] * L)
                for m in range(1, L):
                    acc[m] += sum((d[lp][l] * d[lq][m - 1 - l] for l in range(m)), arb(0)) / (8 * m)
        chans = [(lapse[(nu - lr, 1 - gr)], lr) for lr, _, gr in labs if (nu - lr, 1 - gr) in lapse]
        # de-mixing and psi constants (exact values; only their size enters)
        F2 = lambda h, gm: sum((d0p * d0q / 4 for lp, d0p, gp in labs for lq, d0q, gq in labs
                                if lp + lq == h and gp + gq == gm), acb(0))
        aN, bN = (1 + Iv * nu * wv) / 2, (2 + Iv * nu * wv) / 2
        for jv in range(j0):
            cj = abs(2 * jv + 2 + Iv * nu * wv) / (4 * (jv + 1))
            ss = sum((g[i] * s[lr][jv - i] for g, lr in chans for i in range(1, jv + 1)), arb(0))
            sd = sum((g[i] * d[lr][jv + 1 - i] for g, lr in chans for i in range(1, jv + 2)), arb(0))
            den = t[jv + 1] if red else T[nu][jv + 1]
            v = (cj * ss + sd) / den
            for lr, d0r, gr in labs:
                h = nu - lr
                if h != 0:
                    E = abs(F2(h, 1 - gr)) / (abs(h) * W)
                    if red:
                        rho = P if lr == nu + 2 else arb(1)
                        v += E * abs(d0r) * 2 * rho
                    else:
                        v += E * abs(d0r) * (T[lr][jv + 1] / T[nu][jv + 1] + T[lr][jv] / T[nu][jv])
                else:
                    v += abs(F2(0, 1 - gr)) / 2 * abs(d0r) * (1 / abs(aN + jv) + 1 / abs(bN + jv))
            out[red].append(float(v.upper()))
    return out[False], out[True]


def scope(numax):
    rows = []
    for nu in range(3, numax + 1, 2):
        for pk in PERTS:
            vals, signed, j0 = head_terms(nu, pk)
            a = [float(abs(v).mid()) for v in vals]
            q = [(jv * jv * a[jv] / nu, jv) for jv in range(1, j0)]
            qmax, qat = max(q)
            qtail = max(jv * jv * a[jv] / nu for jv in range(j0, len(a)))
            Habs = sum(a[:j0])
            mx, mr = majorants(nu, pk)
            if any(mx[k] < a[k] * (1 - 1e-12) for k in range(j0)):
                raise SystemExit(f"a termwise majorant is below the exact term at nu {nu}, {pk}")
            rows.append(dict(nu=nu, pk=pk, j0=j0, a0=a[0], qmax=qmax, qat=qat, qtail=qtail, Habs=Habs,
                             Hsig=float(abs(signed).mid()), amax=max(a[:j0]), amaxat=max(range(j0), key=lambda k: a[k]),
                             Mx=sum(mx), Mr=sum(mr)))
            print(f"nu {nu:2d} {pk}: j0 {j0:2d}  |DR_0| {a[0]:.3e}  max_j j^2|DR|/nu {qmax:.3e} at j {qat:2d} "
                  f"(C_mu {C_TAIL[pk]:.4g}; first tail j's {qtail:.3e})  head sum |.| {Habs:.4e}, signed {rows[-1]['Hsig']:.4e}"
                  f"  max term {rows[-1]['amax']:.3e} at j {rows[-1]['amaxat']}", flush=True)
    print()
    print("| nu | perturbation | j0 | max j^2|DR_j/K|/(nu|p|), head | at j | / C_mu | head sum of |DR_j/K|/|p| | signed head | "
          "largest term, at j | termwise, exact T | termwise, reduced (B2, B3) | e^(-0.736 nu) x reduced |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['nu']} | {r['pk']} | {r['j0']} | {r['qmax']:.4g} | {r['qat']} | {r['qmax'] / C_TAIL[r['pk']]:.2e} | "
              f"{r['Habs']:.4g} | {r['Hsig']:.4g} | {r['amax']:.3g}, {r['amaxat']} | {r['Mx']:.4g} | {r['Mr']:.4g} | "
              f"{math.exp(-SIGMA * r['nu']) * r['Mr']:.3g} |")
    # growth of the head sum: slopes of log H against log nu and against nu, over nu >= 9
    for pk, key in [(p_, k_) for k_ in ("Habs", "Mx", "Mr") for p_ in PERTS]:
        pts = [(r["nu"], r[key]) for r in rows if r["pk"] == pk and r["nu"] >= 9]
        if len(pts) >= 3:
            xs = [math.log(n) for n, _ in pts]
            ys = [math.log(h) for _, h in pts]
            zs = [n for n, _ in pts]
            sl = lambda X, Y: (len(X) * sum(x * y for x, y in zip(X, Y)) - sum(X) * sum(Y)) / (len(X) * sum(x * x for x in X) - sum(X) ** 2)
            print(f"{key}, perturbation {pk}: ~ nu^{sl(xs, ys):.2f} (log-log slope), rate {sl(zs, ys):+.3f} per unit nu "
                  f"(against sigma = {SIGMA}), over nu = {pts[0][0]}..{pts[-1][0]}")
    return rows


# ---------------------------------------------------------------- the certificate
NTR = {"m": 3, "0": 6, "p": 3}       # (lapse pair, y label) triples with one perturbation factor (route3_bound.NTR)
NY = {"m": 1, "0": 2, "p": 1}        # of which the y label is the perturbation's (the rest carry +-1)


class Consts:
    """omega, r_m = |a^1_m|, t_1 = r_0, harmonic numbers, at the current precision (built after ctx.prec is set)"""

    def __init__(self):
        self.Iv = acb(0, 1)
        self.wv = acb(2 * arb.pi() / arb("1.64389375"))
        self.W = self.wv.real
        self.w2 = self.W * self.W / 4
        self._r, self._H = {}, {0: arb(0)}

    def r(self, m):
        if m not in self._r:
            self._r[m] = ((arb(m) + 0.5) ** 2 + self.w2).sqrt() * ((arb(m) + 1) ** 2 + self.w2).sqrt() / (m + 1)
        return self._r[m]

    def H(self, n):
        while n not in self._H:
            k = max(self._H) + 1
            self._H[k] = self._H[k - 1] + arb(1) / k
        return self._H[n]

    def P(self, nu):
        """P_(nu+2), route3_bound's closed form (B2)"""
        pi = arb.pi()
        return (arb(nu) * (pi * (nu + 2) * self.W).sinh() / ((nu + 2) * (pi * nu * self.W).sinh())).sqrt()

    def P_sup(self):
        pi = arb.pi()
        return (pi * self.W).exp() / (1 - (-6 * pi * self.W).exp()).sqrt()


def labels_of(nu, pk):
    return [(1, acb(0, -1), 0), (-1, acb(0, 1), 0), (nu + PERTS[pk], acb(1), 1)]


def F2(labs, h, gm):
    return sum((dp * dq / 4 for lp, dp, gp in labs for lq, dq, gq in labs if lp + lq == h and gp + gq == gm), acb(0))


def demix_psi(K, nu, pk, jv=None, Pw=None):
    """sum_r |E_r| 2 w_r (|rho^r| <= w_r: 1 for |lambda_r| <= nu, P_(nu+2) for nu + 2, B2) and the psi term; with jv
    None the psi term at its bound 2/(nu w) per 1/|alpha + j|, otherwise exact"""
    labs = labels_of(nu, pk)
    out = arb(0)
    aN, bN = (1 + K.Iv * nu * K.wv) / 2, (2 + K.Iv * nu * K.wv) / 2
    for lr, d0r, gr in labs:
        h = nu - lr
        if h != 0:
            wr = (K.P(nu) if Pw is None else Pw) if lr == nu + 2 else arb(1)
            out += abs(F2(labs, h, 1 - gr)) / (abs(h) * K.W) * abs(d0r) * 2 * wr
        else:
            h2 = abs(F2(labs, 0, 1 - gr)) / 2
            out += h2 * abs(d0r) * (4 / (nu * K.W) if jv is None else 1 / abs(aN + jv) + 1 / abs(bN + jv))
    return out


def B_term(K, nu, pk, jv):
    """the per-term bound (B7): |DR_j / K_nu| / |p| <= P_mu A0 [c_j (sum over the Ss triples of sum_i |2(j - i) + 1 +
    i lam w| / i) / (r_(j-1) r_j) + NTR H_(j+1) / r_j] + de-mixing + psi, A0 = (2 + t_1)^2 / 8. From B3 (each triple
    product at most P_mu t_a t_b t_c / t_(j+1)), D_entire.md 6.3 twice (sum_l t_l t_(n-l) <= (2 + t_1) t_n) and B2"""
    sh = PERTS[pk]
    mu = nu + sh
    Pm = K.P(nu) if sh == 2 else arb(1)
    A0 = (2 + K.r(0)) ** 2 / 8
    cj = abs(2 * jv + 2 + K.Iv * nu * K.wv) / (4 * (jv + 1))
    ss = arb(0)
    if jv >= 1:
        sy = sum((abs(2 * (jv - i) + 1 + K.Iv * mu * K.wv) / i for i in range(1, jv + 1)), arb(0))
        sb = sum((abs(2 * (jv - i) + 1 + K.Iv * K.wv) / i for i in range(1, jv + 1)), arb(0))
        ss = (NY[pk] * sy + (NTR[pk] - NY[pk]) * sb) / (K.r(jv - 1) * K.r(jv))
    sd = NTR[pk] * K.H(jv + 1) / K.r(jv)
    return Pm * A0 * (cj * ss + sd) + demix_psi(K, nu, pk, jv)


def S_ab():
    """S_a = sum_(j>=1) 1/((j+1)(j-1/2)) = (2/3)(1 + 2 ln 2), S_b = sum_(j>=1) 1/((j+1)(j^2-1/4)) = (8/3)(1 - ln 2),
    by partial fractions and psi(2) - psi(1/2) = 1 + 2 ln 2, psi(3/2) - psi(1/2) = 2"""
    l2 = arb(2).log()
    return 2 * (1 + 2 * l2) / 3, 8 * (1 - l2) / 3


def G_closed(K, nu, pk, Pm=None):
    """a closed form >= sum_(j < j0) B_term (hand inequalities): c_j <= 1/2 + nu w/(4(j+1)), |2(j-i)+1+i lam w| <=
    2(j-i)+1+(nu+2)w, H_j <= 1 + ln j0, r_(j-1) r_j >= j^2 - 1/4 and r_j >= j + 1/2 (B1), sum_(j=1)^(N) 1/(j-1/2) <=
    2 + ln(2N - 1), sum_(j>=1) 1/(j^2-1/4) = 2, sum_(j=1)^(N) 1/(j+1/2) <= ln(2N+1). Each piece is >= 0 and of the
    form nu^a L1 L2 with a <= 2 and logarithms L of nu, so e^(-0.736 nu) G decreases for nu >= 7"""
    j0 = j0_of(nu)
    sh = PERTS[pk]
    if Pm is None:
        Pm = K.P(nu) if sh == 2 else arb(1)
    A0 = (2 + K.r(0)) ** 2 / 8
    Sa, Sb = S_ab()
    Lam = arb(nu + 2)
    l0 = 1 + arb(j0).log()
    T1 = 2 + arb(2 * j0 - 1).log()
    T2 = Lam * K.W                                   # (1/2) Lam w sum 1/(j^2 - 1/4) = Lam w
    T3 = nu * K.W / 2 * Sa
    T4 = nu * K.W * Lam * K.W / 4 * Sb
    ssum = NTR[pk] * l0 * (T1 + T2 + T3 + T4)
    sdsum = NTR[pk] * l0 * (1 / K.r(0) + arb(2 * j0 - 1).log())
    return Pm * A0 * (ssum + sdsum) + j0 * demix_psi(K, nu, pk, None, Pm if sh == 2 else None)


def certify(nu1=99, prec=512):
    ctx.prec = prec
    K = Consts()
    out = {}
    worst = {pk: (arb(0), None) for pk in PERTS}
    allw = {pk: (arb(0), None) for pk in PERTS}
    rows = []
    for nu in range(3, nu1 + 1, 2):
        for pk in PERTS:
            p = prec
            while True:
                try:
                    vals, signed, j0 = head_terms(nu, pk, prec=p, extra=0)
                    break
                except SystemExit:
                    p *= 2
                    if p > 4096:
                        raise
            ctx.prec = prec
            H = sum((abs(v).upper() for v in vals), arb(0))
            e = (-arb("0.736") * nu).exp()
            tail = arb(C_TAIL[pk]) * nu / (j0 - 1)               # section 10: sum_(j >= j0) <= C_mu nu / (j0 - 1)
            for key, val, store in (("head", H, worst), ("all", H + tail, allw)):
                v = (e * val).upper()
                if v > store[pk][0]:
                    store[pk] = (v, nu)
            rows.append((nu, pk, j0, H.upper(), p))
            out[(nu, pk)] = H.upper()
    # beyond nu1: the closed form at nu1 + 2 with P_(nu+2) <= P_sup, e^(-0.736 nu) G decreasing for nu >= 7
    nb = nu1 + 2
    beyond = {pk: ((-arb("0.736") * nb).exp() * G_closed(K, nb, pk, K.P_sup() if pk == "p" else None)).upper()
              for pk in PERTS}
    print(f"head sums H_mu(nu) = sum_(j < j0) |DR_j / K_nu| / |p| from the exact recursion ({prec} bits, raised where the "
          f"guard asks), odd nu = 3..{nu1}:")
    print("| nu | H, mu = nu - 2 | H, mu = nu | H, mu = nu + 2 |")
    print("|---|---|---|---|")
    for nu in [3, 5, 7, 9, 11, 21, 41, 61, 81, nu1]:
        if nu <= nu1:
            print(f"| {nu} | " + " | ".join(out[(nu, pk)].str(5, radius=False) for pk in PERTS) + " |")
    print(f"precision used: {sorted({r[4] for r in rows})} bits")
    print("| perturbation | sup e^(-0.736 nu) H, nu <= %d (at nu) | beyond: e^(-0.736 nu) G at nu = %d | Lambda_head | "
          "sup e^(-0.736 nu) (H + C_mu nu/(j0 - 1)), nu <= %d (at nu) |" % (nu1, nb, nu1))
    print("|---|---|---|---|---|")
    res = {}
    for pk in PERTS:
        lam_head = max(worst[pk][0], beyond[pk])
        tb = (-arb("0.736") * nb).exp() * (G_closed(K, nb, pk, K.P_sup() if pk == "p" else None)
                                          + arb(C_TAIL[pk]) * nb / (j0_of(nb) - 1))
        lam_all = max(allw[pk][0], tb.upper())
        res[pk] = (lam_head, lam_all)
        print(f"| {pk} | {worst[pk][0].str(5, radius=False)} ({worst[pk][1]}) | {beyond[pk].str(3, radius=False)} | "
              f"{lam_head.str(5, radius=False)} | {allw[pk][0].str(5, radius=False)} ({allw[pk][1]}); beyond "
              f"{tb.upper().str(3, radius=False)} |")
    return res


def check(numax=21, prec=256):
    """B_term against the exact recursion for every head j at nu = 3..numax, and sum_j B_term <= G_closed at
    nu = 3..201; --mutate=plant adds 3 x B_term at five head j per nu and perturbation, which must be caught"""
    ok = True
    ctx.prec = prec
    K = Consts()
    worst = {pk: 0.0 for pk in PERTS}
    nviol_all = 0
    for nu in range(3, numax + 1, 2):
        for pk in PERTS:
            vals, signed, j0 = head_terms(nu, pk, prec=prec, extra=0)
            ctx.prec = prec
            plant = set(range(0, j0, max(1, j0 // 5))[:5]) if "plant" in MUT else set()
            nv = 0
            for jv in range(j0):
                a = abs(vals[jv]).upper()
                b = B_term(K, nu, pk, jv)
                if jv in plant:
                    a = a + 3 * b
                rat = float((a / b).upper())
                worst[pk] = max(worst[pk], rat)
                if not (a < b):
                    nv += 1
            nviol_all += nv
            if nv:
                print(f"nu {nu} {pk}: {nv} head terms above B_term", flush=True)
    ok &= nviol_all == 0
    print(f"B_term against the exact recursion, every head j, odd nu = 3..{numax}: worst exact/bound "
          + ", ".join(f"{k} {v:.3e}" for k, v in worst.items()) + f"; violations {nviol_all}")
    gw = {pk: 0.0 for pk in PERTS}
    gbad = 0
    for nu in range(3, 202, 2):
        for pk in PERTS:
            s = sum((B_term(K, nu, pk, jv) for jv in range(j0_of(nu))), arb(0))
            g = G_closed(K, nu, pk) * (arb(1) / 100 if "gsmall" in MUT else 1)   # --mutate=gsmall: G / 100
            gw[pk] = max(gw[pk], float((s / g).upper()))
            gbad += 0 if s < g else 1
    ok &= gbad == 0
    print(f"sum_(j < j0) B_term <= G_closed at odd nu = 3..201: worst ratio " + ", ".join(f"{k} {v:.3f}" for k, v in gw.items())
          + f"; failures {gbad}")
    return ok


def lemmas():
    """numerical checks of the closed forms and inequalities G_closed uses (each is proved by hand in section 11.2)"""
    ctx.prec = 128
    ok = True
    Sa, Sb = S_ab()
    N = 200000
    pa = sum(1.0 / ((j + 1) * (j - 0.5)) for j in range(1, N))
    pb = sum(1.0 / ((j + 1) * (j * j - 0.25)) for j in range(1, N))
    ta, tb = 1.0 / N, 1.0 / (2 * N * N)                   # tails beyond N, to leading order
    da, db = abs(pa + ta - float(Sa.mid())), abs(pb + tb - float(Sb.mid()))
    good = da < 1e-9 and db < 1e-9
    ok &= good
    print(f"S_a = (2/3)(1 + 2 ln 2) = {Sa.str(10, radius=False)}, partial sum + tail {pa + ta:.10f}; S_b = (8/3)(1 - ln 2) = "
          f"{Sb.str(10, radius=False)}, partial sum + tail {pb + tb:.10f}  ({'ok' if good else 'FAIL'})")
    tel = sum(1.0 / (j * j - 0.25) for j in range(1, N))
    good = abs(tel + 1.0 / N - 2) < 1e-9
    ok &= good
    print(f"sum 1/(j^2 - 1/4) = 2: partial sum + tail {tel + 1.0 / N:.12f}  ({'ok' if good else 'FAIL'})")
    for M in (1, 2, 5, 10, 100, 1000):
        a = sum(1.0 / (j - 0.5) for j in range(1, M + 1))
        b = sum(1.0 / (j + 0.5) for j in range(1, M + 1))
        good = a <= 2 + math.log(2 * M - 1) + 1e-12 and b <= math.log(2 * M + 1) + 1e-12
        ok &= good
        print(f"N = {M}: sum 1/(j - 1/2) = {a:.6f} <= 2 + ln(2N - 1) = {2 + math.log(2 * M - 1):.6f}; sum 1/(j + 1/2) = "
              f"{b:.6f} <= ln(2N + 1) = {math.log(2 * M + 1):.6f}  ({'ok' if good else 'FAIL'})")
    K = Consts()
    for nu in (7, 9, 21, 101, 1001):                     # e^(-0.736 nu) G decreasing from nu = 7 (sampled)
        for pk in PERTS:
            Ps = K.P_sup() if pk == "p" else None
            g0 = (-arb("0.736") * nu).exp() * G_closed(K, nu, pk, Ps)
            g1 = (-arb("0.736") * (nu + 2)).exp() * G_closed(K, nu + 2, pk, Ps)
            good = g1 < g0
            ok &= good
            if not good:
                print(f"e^(-0.736 nu) G not decreasing at nu {nu}, {pk}")
    print(f"e^(-0.736 nu) G decreasing at the sampled nu: {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--lemmas" in sys.argv:
        good = lemmas()
        print("PASS" if good else "FAIL")
        sys.exit(0 if good else 1)
    if "--scope" in sys.argv:
        scope(opt("numax", 21))
        sys.exit(0)
    if "--check" in sys.argv:
        good = check(opt("numax", 21))
        print("PASS" if good else "FAIL")
        sys.exit(0 if good else 1)
    if "--certify" in sys.argv:
        certify(opt("nu1", 99))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
