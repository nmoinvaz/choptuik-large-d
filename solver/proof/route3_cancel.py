#!/usr/bin/env python3
"""P6 route 3 (notes/P6_route3.md, #982): are the two O(1/j) cancellations of the Darboux increment identities in nu?

The model is order 3 of amp_flint's recursion (solver/inner_amp/amp_flint.py) for a target harmonic nu. Its order-1
inputs are a list of labels lam with initial values d0_lam, each the exact linear (Kummer) sequence
d~_l[lam] = d0_lam T^lam_l, T^lam_l = (alpha_lam)_l (beta_lam)_l / l!, alpha = (1 + i lam w)/2, beta = (2 + i lam w)/2.
The increment of the normalised multiplier sequence (proof/D_entire.md section 6.1) is

    DR_j / K_nu = - b_j / T^nu_{j+1} + sum_lam E[nu - lam] d0_lam (rho^lam_{j+1} - rho^lam_j)
                  + h2 d0_nu (1/(alpha_nu + j) + 1/(beta_nu + j)),                rho^lam_j = T^lam_j / T^nu_j,

b_j = c_j Ss_j + Sd_{j+1} the order-3 forcing, E[h] = -F2_h / (i h w) for h != 0 (E[0] = 0), F2_h = sum over ordered pairs
lam1 + lam2 = h of F_lam1 F_lam2 with F_lam = d0_lam / 2, h2 = F2_0 / 2. tail() splits it as sum over families lam of
C_lam(j) rho^lam_j + O(j^-3), C_lam rational in j, from the terms of exact_sigma._sym_tail: the near end i = 1, 2 of
both convolutions, the far end i = j + 1 at l = 0 and l = j, the de-mixing and the psi normalisation.

Cases (cases()):
  h1     inputs +1, -1 with d0 = -i, +i, target 1. Must equal exact_sigma._sym_tail() term for term (F, G).
  h3     the same inputs, target 3: the first tuned order of harmonic 3, exactly.
  nu     inputs +1, -1 (background, d0 symbols b_p, b_m) and Kummer perturbations of labels nu - 2, nu, nu + 2
         (amplitudes p_m, p_0, p_p), target nu symbolic; only the terms linear in the perturbations. This is the
         one-A^2-step coupling of a linear harmonic of label mu = nu - 2, nu, nu + 2 into harmonic nu.
The test: lim j C_lam(j), the O(1/j) coefficient of every family, vanishes identically (in nu, w and the d0's).

    python3 route3_cancel.py                    the symbolic checks; exit 0 iff all hold
    python3 route3_cancel.py --mutate=demix     E scaled by 3/2: must fail (exit 1)
    python3 route3_cancel.py --mutate=psi       h2 scaled by 3/2: must fail (exit 1)
    python3 route3_cancel.py --c2               also print lim j^2 C_lam(j) of the nu case, by family, as polynomials
    python3 route3_cancel.py --selfcheck        the numeric recursion against exact_sigma.sequences at harmonic 1 (planted
                                                truth); --mutate=selfcheck (the sign of one input) must fail
    python3 route3_cancel.py --matrix           the numeric checks at nu = 3, 5, 7 with every mode and mutation, J = 800;
                                                exit 0 iff each run passes or fails as fixed in matrix()
    python3 route3_cancel.py --numeric NU [J] [--wf=X] [--pert=m|0|p|all]
                                                the exact recursion of the nu case at nu = NU, numerically (python-flint),
                                                with no use of tail(); then j |DR_j/K| -> 0 and the remainder after
                                                tail()'s C_lam is O(j^-3). Exit 0 iff both pass. --mutate=demix or psi
                                                must fail the first, --mutate=near2 (tail() without its i = 2 terms)
                                                the second. --pert keeps one perturbation (labels nu - 2, nu,
                                                nu + 2); --wf sets w (default 1, the physical w). The results are
                                                divided by sum |rho^lam_j|, so a family with small |rho| is visible only
                                                where it is not dominated: one perturbation at a time at wf = 1, and
                                                all of them at small wf, where every |rho| is of order 1
"""
import sys

import sympy as sp

j, w = sp.symbols("j w", positive=True)
nu = sp.Symbol("nu", real=True)
I = sp.I
MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def ab(lam):
    return (1 + I * lam * w) / 2, (2 + I * lam * w) / 2


def A(lam, l):
    """a^lam_l = (l + alpha)(l + beta)/(l + 1), the step T^lam_{l+1} / T^lam_l"""
    a, b = ab(lam)
    return (l + a) * (l + b) / (l + 1)


def same(x, y):
    return sp.expand(x - y) == 0


def tail(inputs, target, deg=None, mutate=None):
    """{family label: C(j)} with DR_j/K_target = sum C rho^fam_j + O(j^-3). inputs: list of (lam, d0, degree); with deg
    set, only products whose degrees add up to deg are kept (deg = 1: linear in the perturbations)."""
    mutate = MUT if mutate is None else mutate
    fam = {}
    ok = (lambda d: True) if deg is None else (lambda d: d == deg)

    def add(lam, expr):
        k = sp.expand(lam)
        fam[k] = fam.get(k, 0) + expr

    def ratio(lam, k):
        """T^lam_{j-k} / T^target_{j+1}, without the factor rho^lam_j"""
        if same(lam, target):
            return 1 / sp.Mul(*[A(target, j - q) for q in range(k + 1)])
        return 1 / (A(target, j) * sp.Mul(*[A(lam, j - q) for q in range(1, k + 1)]))

    def T1(lam):
        a, b = ab(lam)
        return a * b

    c = (2 * j + 2 + I * target * w) / (4 * (j + 1))
    n = len(inputs)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                (lp, dp, gp), (lq, dq, gq), (lr, dr, gr) = inputs[p], inputs[q], inputs[r]
                if not ok(gp + gq + gr) or not same(lp + lq + lr, target):
                    continue
                g1 = sp.Rational(1, 8) * dp * dq                                    # lapse (p, q) at m = 1
                g2 = sp.Rational(1, 16) * dp * dq * (T1(lq) + T1(lp))              # and at m = 2
                sr = I * lr * w
                add(lr, -c * g1 * (2 * j - 1 + sr) * dr * ratio(lr, 1))           # i = 1 in Ss(j)
                add(lr, -g1 * dr * ratio(lr, 0))                                  # i = 1 in Sd(j+1)
                if "near2" not in mutate:
                    add(lr, -c * g2 * (2 * j - 3 + sr) * dr * ratio(lr, 2))       # i = 2 in Ss(j)
                    add(lr, -g2 * dr * ratio(lr, 1))                              # i = 2 in Sd(j+1)
                far = -dr * dp * dq / (8 * (j + 1))                               # i = j+1 in Sd(j+1):
                add(lq, far * ratio(lq, 0))                                       #   l = 0
                add(lp, far * ratio(lp, 0))                                       #   l = j
    at, bt = ab(target)
    for r in range(n):
        lr, dr, gr = inputs[r]
        h = sp.expand(target - lr)
        F2 = sum((dp * dq / 4 for lp, dp, gp in inputs for lq, dq, gq in inputs
                  if ok(gp + gq + gr) and same(lp + lq, h)), sp.Integer(0))
        if h == 0:                                                                  # psi normalisation
            h2 = F2 / 2 * (sp.Rational(3, 2) if "psi" in mutate else 1)
            add(lr, h2 * dr * (1 / (j + at) + 1 / (j + bt)))
        else:                                                                       # de-mixing
            E = -F2 / (I * h * w) * (sp.Rational(3, 2) if "demix" in mutate else 1)
            add(lr, E * dr * (A(lr, j) / A(target, j) - 1))
    return fam


# ---------------------------------------------------------------- truncated Laurent series in x = 1/j
CAP = 6                                   # powers of x kept in intermediate products


def _zero(c):
    return sp.expand(c) == 0


def _add(a, b):
    (ca, Pa), (cb, Pb) = a, b
    P = min(Pa, Pb)
    c = {}
    for d in (ca, cb):
        for k, v in d.items():
            if k <= P:
                c[k] = c.get(k, 0) + v
    return {k: sp.expand(v) for k, v in c.items()}, P


def _mul(a, b):
    (ca, Pa), (cb, Pb) = a, b
    ma, mb = min(ca), min(cb)
    P = min(ma + Pb, mb + Pa, CAP)
    c = {}
    for ka, va in ca.items():
        for kb, vb in cb.items():
            if ka + kb <= P:
                c[ka + kb] = c.get(ka + kb, 0) + va * vb
    c = {k: sp.expand(v) for k, v in c.items()}
    return (c if c else {P: sp.Integer(0)}), P


def _inv(a):
    ca, Pa = a
    ks = sorted(k for k in ca if not _zero(ca[k]))
    m = ks[0]
    lead = ca[m]
    R = min(Pa - m, CAP + m)                                  # relative order kept
    t = ({k - m: sp.expand(-ca[k] / lead) for k in ca if m < k <= m + R}, R)
    if not t[0]:
        t = ({R: sp.Integer(0)}, R)
    acc, term = ({0: sp.Integer(1)}, R), ({0: sp.Integer(1)}, R)
    for _ in range(R):
        term = _mul(term, t)
        acc = _add(acc, term)
    return {k - m: sp.expand(v / lead) for k, v in acc[0].items()}, R - m


def ser(e):
    """(coefficients by power of x = 1/j, valid up to power P) for a rational function of j"""
    if not e.has(j):
        return {0: e}, 10 ** 6
    if e == j:
        return {-1: sp.Integer(1)}, 10 ** 6
    if e.is_Add:
        out = None
        for t in e.args:
            out = ser(t) if out is None else _add(out, ser(t))
        return out
    if e.is_Mul:
        out = None
        for t in e.args:
            out = ser(t) if out is None else _mul(out, ser(t))
        return out
    if e.is_Pow and e.exp.is_Integer:
        n = int(e.exp)
        b = ser(e.base)
        if n < 0:
            b, n = _inv(b), -n
        out = ({0: sp.Integer(1)}, 10 ** 6)
        for _ in range(n):
            out = _mul(out, b)
        return out
    raise ValueError(f"cannot expand {e!r}")


def laurent(terms):
    """c_{-1}, c_0, c_1, c_2 of sum(terms) at j = oo (DR = ... + c_1/j + c_2/j^2 + ...)"""
    tot = ({k: sp.Integer(0) for k in range(-1, 3)}, 10 ** 6)
    for t in terms:
        tot = _add(tot, ser(t))
    c, P = tot
    assert P >= 2, f"series precision too low ({P})"
    return [sp.cancel(sp.together(c.get(k, 0))) for k in range(-1, 3)]


def coeff_inf(expr, power):
    """lim_{j -> oo} j^power expr for a rational function of j (sp.oo if it diverges)"""
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    if num == 0:
        return sp.Integer(0)
    pn, pd = sp.Poly(num, j), sp.Poly(den, j)
    gap = pd.degree() - pn.degree()
    if gap > power:
        return sp.Integer(0)
    if gap < power:
        return sp.oo
    return sp.cancel(pn.LC() / pd.LC())


def expand1(expr):
    """(c1, c2): DR = c1/j + c2/j^2 + O(j^-3)"""
    c0 = coeff_inf(expr, 0)
    if c0 != 0:
        return sp.oo, sp.oo
    c1 = coeff_inf(expr, 1)
    return c1, coeff_inf(expr - c1 / j, 2)


bp, bm, pm, p0, pp = sp.symbols("b_p b_m p_m p_0 p_p")


def cases():
    bg = [(1, -I, 0), (-1, I, 0)]
    nuin = [(1, bp, 0), (-1, bm, 0), (nu - 2, pm, 1), (nu, p0, 1), (nu + 2, pp, 1)]
    return {"h1": (bg, 1, None), "h3": (bg, 3, None), "nu": (nuin, nu, 1)}


def symbolic():
    ok = True
    out = {}
    for name, (inp, tgt, deg) in cases().items():
        fam = tail(inp, tgt, deg)
        out[name] = fam
        for lam, C in sorted(fam.items(), key=lambda kv: str(kv[0])):
            cm1, c0, c1, c2 = laurent(sp.Add.make_args(C))
            zero = cm1 == 0 and c0 == 0 and c1 == 0
            ok &= bool(zero)
            print(f"{name:3s} family {str(lam):7s}: lim j C = {c1 if c1 == 0 else sp.factor(c1)}   "
                  f"{'(zero)' if zero else '** NOT ZERO **'}", flush=True)
    # reproduction: h1 against exact_sigma._sym_tail, exactly
    from exact_sigma import _sym_tail, jj, ww
    F0, G0, _ = _sym_tail()
    F0, G0 = F0.subs({jj: j, ww: w}), G0.subs({jj: j, ww: w})
    h1 = {k: sp.cancel(sp.together(v)) for k, v in out["h1"].items()}
    dF = sp.cancel(h1[sp.Integer(1)] - F0)
    dG = sp.cancel(h1[sp.Integer(-1)] - G0)
    rep = dF == 0 and dG == 0
    print(f"h1 against exact_sigma._sym_tail: F - F0 = {dF}, G - G0 = {dG}   {'(identical)' if rep else '** DIFFER **'}")
    j2F = coeff_inf(h1[sp.Integer(1)], 2)
    j2G = coeff_inf(h1[sp.Integer(-1)], 2)
    print(f"h1: j^2 F -> {sp.factor(j2F)},  j^2 G -> {sp.factor(j2G)}   (D_entire.md 6.2: i(7 - w^2 - 2iw)/8, "
          f"i(7 - w^2 + 5iw)/16)")
    want = (sp.expand(j2F - I * (7 - w**2 - 2 * I * w) / 8) == 0 and sp.expand(j2G - I * (7 - w**2 + 5 * I * w) / 16) == 0)
    # the series engine against the exact rational functions, on h1 and h3
    agree = True
    for name in ("h1", "h3"):
        for lam, C in out[name].items():
            ex = sp.cancel(sp.together(C))
            ls = laurent(sp.Add.make_args(C))
            agree &= all(sp.cancel(ls[k + 1] - coeff_inf(ex, k)) == 0 for k in (-1, 0, 1)) and \
                sp.cancel(ls[3] - coeff_inf(ex - ls[2] / j, 2)) == 0
    print(f"series engine against the exact rational functions (h1, h3): {'agree' if agree else '** DISAGREE **'}")
    if not MUT:
        ok &= bool(rep) and bool(want) and bool(agree)
    if "--c2" in sys.argv:
        print("\nnu case, lim j^2 C_lam(j) with b_p = -i, b_m = i (the A sin theta background); growth in nu:")
        for lam, C in sorted(out["nu"].items(), key=lambda kv: str(kv[0])):
            c2 = sp.cancel(sp.together(laurent(sp.Add.make_args(C))[3].subs({bp: -I, bm: I})))
            for a in (pm, p0, pp):
                ca = sp.cancel(sp.together(sp.expand(c2).coeff(a)))
                if ca == 0:
                    continue
                n_, d_ = sp.fraction(ca)
                dn, dd = sp.Poly(n_, nu).degree(), sp.Poly(d_, nu).degree()
                lead = sp.factor(sp.Poly(n_, nu).LC() / sp.Poly(d_, nu).LC())
                print(f"  family {str(lam):7s} [{a}]: ~ nu^{dn - dd} x ({lead}) as nu -> oo", flush=True)
    return ok


# ---------------------------------------------------------------- the exact recursion, numerically (no use of tail())
def order3(labels, NU, J, wv, deg):
    """the exact order-3 recursion for target harmonic NU from order-1 Kummer inputs (lam, d0, degree), keeping the
    products of total degree deg: T^lam_l, the input sequences, and d~^(3)_j for j <= J + 1 (python-flint balls)"""
    from flint import acb
    Iv = acb(0, 1)
    L = J + 3
    T = {}
    for lam, _, _ in labels:
        if lam in T:
            continue
        a, b = (1 + Iv * lam * wv) / 2, (2 + Iv * lam * wv) / 2
        t = [acb(1)]
        for l in range(L):
            t.append(t[-1] * (l + a) * (l + b) / (l + 1))
        T[lam] = t
    seq = []                                                        # (lam, deg, d~_l, s~_l)
    for lam, d0, g in labels:
        d = [d0 * x for x in T[lam]]
        seq.append((lam, g, d, [(2 * l + 1 + Iv * lam * wv) * d[l] for l in range(L)]))
    lapse = {}                                                      # (harmonic, deg) -> g_m, m = 0..L-1
    for lp, gp, dp, _ in seq:
        for lq, gq, dq, _ in seq:
            if gp + gq > deg:
                continue
            key = (lp + lq, gp + gq)
            acc = lapse.setdefault(key, [acb(0)] * L)
            for m in range(1, L):
                s = acb(0)
                for l in range(m):
                    s += dp[l] * dq[m - 1 - l]
                acc[m] += s / (8 * m)
    chans = [(lapse[(NU - lr, deg - gr)], sr, dr) for lr, gr, dr, sr in seq if (NU - lr, deg - gr) in lapse]
    sN = Iv * NU * wv
    d3 = [acb(0)]
    for jv in range(J + 1):
        ss, sd = acb(0), acb(0)
        for g, sr, dr in chans:
            for i in range(1, jv + 1):
                ss += g[i] * sr[jv - i]
            for i in range(1, jv + 2):
                sd += g[i] * dr[jv + 1 - i]
        sj = (2 * jv + 1 + sN) * d3[jv] - ss
        d3.append((2 * jv + 2 + sN) * sj / (4 * (jv + 1)) - sd)
    return T, seq, d3


def selfcheck(J=200):
    """planted truth for order3(): the background alone (labels +-1, degree 0) at target 1 is the harmonic-1 order-3
    sequence of exact_sigma.sequences, an earlier implementation of the same recursion"""
    from flint import acb, arb, ctx
    from exact_sigma import sequences
    ctx.prec = 256
    wv = acb(arb(2) * arb.pi() / arb("1.64389375"))
    Iv = acb(0, 1)
    flip = "selfcheck" in MUT                                    # mutation: the wrong sign of d0 on label -1
    _, _, d3 = order3([(1, -Iv, 0), (-1, -Iv if flip else Iv, 0)], 1, J, wv, 0)
    d3ref = sequences(wv, J)[4]
    rel = max(float(abs(d3[i] - d3ref[i]).mid()) / float(abs(d3ref[i]).mid()) for i in range(1, J + 1))
    print(f"order3 against exact_sigma.sequences, harmonic 1 at order 3, j <= {J}: max relative difference {rel:.2e}"
          f"  (pass: < 1e-60)")
    return rel < 1e-60


def numeric(NU, J, wf=1.0, pert="all", mut=None, quiet=False):
    mut = MUT if mut is None else mut
    from flint import acb, arb, ctx
    ctx.prec = 256
    wv = acb(arb(str(wf)) * 2 * arb.pi() / arb("1.64389375"))     # w = wf * 2 pi / 1.64389375, as exact_sigma.py
    Iv = acb(0, 1)
    eps = acb(10) ** -40
    amp = {"m": (1, 0), "0": ("0.7", "0.3"), "p": ("-0.4", "1.1")}
    on = lambda k: pert in ("all", k)
    labels = [(1, -Iv, 0), (-1, Iv, 0)] + [(NU + sh, eps * (acb(*amp[k]) if on(k) else acb(0)), 1)
                                          for sh, k in ((-2, "m"), (0, "0"), (2, "p"))]
    T, seq, d3 = order3(labels, NU, J, wv, 1)
    TN = T[NU]
    aN, bN = (1 + sN) / 2, (2 + sN) / 2
    sc = {"demix": 1, "psi": 1}
    if "demix" in mut:
        sc["demix"] = acb(3) / 2
    if "psi" in mut:
        sc["psi"] = acb(3) / 2

    def F2(h, gmax):
        return sum((dp[0] * dq[0] / 4 for lp, gp, dp, _ in seq for lq, gq, dq, _ in seq
                    if lp + lq == h and gp + gq == gmax), acb(0))

    def R(jv):
        """R_j / K_nu without the psi part's j-independent constant (only DR is used)"""
        v = d3[jv] / TN[jv]
        for lr, gr, dr, _ in seq:
            h = NU - lr
            if h != 0:
                v += -F2(h, 1 - gr) / (Iv * h * wv) * sc["demix"] * dr[jv] / TN[jv]
        return v

    def dpsi(jv):
        out = acb(0)
        for lr, gr, dr, _ in seq:
            if lr == NU:
                out += F2(0, 1 - gr) / 2 * sc["psi"] * dr[0] * (1 / (aN + jv) + 1 / (bN + jv))
        return out

    DR = lambda jv: (R(jv + 1) - R(jv) + dpsi(jv)) / eps
    # tail()'s C_lam at nu = NU, evaluated with the same labels (degree 1)
    samp = {"m": sp.Integer(1), "0": sp.Rational(7, 10) + sp.Rational(3, 10) * I, "p": -sp.Rational(4, 10) + sp.Rational(11, 10) * I}
    sym_in = [(1, -I, 0), (-1, I, 0)] + [(NU + sh, samp[k] if on(k) else sp.Integer(0), 1)
                                        for sh, k in ((-2, "m"), (0, "0"), (2, "p"))]
    fam = tail(sym_in, sp.Integer(NU), 1, mutate=mut)
    import mpmath
    mpmath.mp.dps = 50
    wmp = mpmath.mpf(wv.real.str(60, radius=False))
    fn = {lam: sp.lambdify((j, w), C_, "mpmath") for lam, C_ in fam.items()}

    def C(lam, jv):
        z = fn[lam](mpmath.mpf(jv), wmp)
        return acb(arb(mpmath.nstr(z.real, 45)), arb(mpmath.nstr(z.imag, 45)))

    fams = [lam for lam in fam if not all(t == 0 for t in sp.Add.make_args(fam[lam]))]

    def resid(jv):
        v = DR(jv)
        for lam in fams:
            rho = T[int(lam)][jv] / TN[jv]
            v -= C(lam, jv) * rho
        return v

    def norm(jv):
        return sum(float(abs(T[int(lam)][jv] / TN[jv]).mid()) for lam in fams)

    def wmax(lo, hi, f):
        pts = sorted({int(round(lo * (hi / lo) ** (k / 8))) for k in range(9)})
        return max(f(jv) for jv in pts)

    lo, hi = (J // 8, J // 4), (J // 2, J)
    f1 = lambda jv: float(abs(DR(jv)).mid()) * jv / norm(jv)
    f2 = lambda jv: float(abs(DR(jv)).mid()) * jv ** 2 / norm(jv)
    f3 = lambda jv: float(abs(resid(jv)).mid()) * jv ** 3 / norm(jv)
    m1 = [wmax(*lo, f1), wmax(*hi, f1)]
    m2 = [wmax(*lo, f2), wmax(*hi, f2)]
    m3 = [wmax(*lo, f3), wmax(*hi, f3)]
    r1, r3 = m1[1] / m1[0], m3[1] / m3[0]
    if not quiet:
        print(f"nu = {NU}, J = {J}, wf = {wf}, perturbation {pert}, families {[int(x) for x in fams]}; divided by sum |rho^lam_j|, "
              f"max over j in {lo} and {hi}")
        print(f"  j   |DR/K|:             {m1[0]:.4e}, {m1[1]:.4e}  ratio {r1:.3f}  (pass: < 0.5; an O(1/j) part left: ~1)")
        print(f"  j^2 |DR/K|:             {m2[0]:.4e}, {m2[1]:.4e}")
        print(f"  j^3 |DR/K - sum C rho|: {m3[0]:.4e}, {m3[1]:.4e}  ratio {r3:.3f}  (pass: < 2; a missed j^-2 term: ~4)")
    numeric.last = (r1, r3)
    return r1 < 0.5 and r3 < 2


def matrix(J=800):
    """every nu in (3, 5, 7), mode and mutation; the pattern each mutation must show is fixed here, before the runs:
    demix fails the O(1/j) test where a label-shift family is visible (one perturbation nu -+ 2 at wf = 1, all at small
    wf), psi where the same-label family is (perturbation nu at wf = 1, all at small wf), near2 the remainder test
    everywhere; the unmutated model passes everywhere"""
    modes = [("1", "m"), ("1", "0"), ("1", "p"), ("0.05", "all")]
    expect = {"": lambda pert: (True, True), "demix": lambda pert: (pert == "0", True),
              "psi": lambda pert: (pert in ("m", "p"), True), "near2": lambda pert: (True, False)}
    good = True
    print("| nu | wf | perturbation | mutation | j\\|DR\\| ratio | j^3 remainder ratio | expected | got |")
    print("|---|---|---|---|---|---|---|---|")
    for NU in (3, 5, 7):
        for wf, pert in modes:
            for mut in ("", "demix", "psi", "near2"):
                numeric(NU, J, float(wf), pert, mut, quiet=True)
                r1, r3 = numeric.last
                got = (r1 < 0.5, r3 < 2)
                want = expect[mut](pert)
                good &= got == want
                fmt = lambda t: "pass" if all(t) else ("fail 1/j" if not t[0] else "fail rem")
                print(f"| {NU} | {wf} | {pert} | {mut or 'none'} | {r1:.3f} | {r3:.3f} | {fmt(want)} | {fmt(got)}"
                      f"{'' if got == want else ' **'} |", flush=True)
    return good


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        good = selfcheck()
    elif "--matrix" in sys.argv:
        good = matrix()
    elif "--numeric" in sys.argv:
        k = sys.argv.index("--numeric")
        NU = int(sys.argv[k + 1])
        J = int(sys.argv[k + 2]) if len(sys.argv) > k + 2 and sys.argv[k + 2].isdigit() else 800
        wf = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--wf=")), "1"))
        pert = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pert=")), "all")
        good = numeric(NU, J, wf, pert)
    else:
        good = symbolic()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
