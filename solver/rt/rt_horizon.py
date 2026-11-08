"""The horizon's singular branches of the RT equations (issue #163), symbolically, and the planted checks of the
horizon rows of rt_model.py (`horizon=K`).

    python3 solver/rt/rt_horizon.py symbolic [STATE.npz] [--mutate=a4]
    python3 solver/rt/rt_horizon.py exponents STATE.npz [STATE.npz ...]
    python3 solver/rt/rt_horizon.py planted --raw RAW [--mutate=taylor] [--out LOG.jsonl]

symbolic. The equations are EEG's Eq. (5) as rt_eqs_check.py transcribes and checks them (its function Omegas, copied
here; Ric = (D - 2) dphi dphi). In the parity-extended form the unknowns are omega_i = omega_i^- on [-1, 1] and
omega_i^+(t, xi) = -omega_i(t, -xi). The collocated equations are Omega_{2,3,4}^+ / xi, in which D^+ acts on omega^-. At
xi = -1, D^+ = d_t + mu s d_s with s = 1 + xi, so each is m_t + mu s m_s + G(m, p, omega_1; xi) with m = omega^-. The
script derives A_ij = dG_i / dm_j at s = 0, shows that it is lower triangular in the order (omega_2, omega_4, omega_3),
and gives its diagonal a_2, a_4 and a_3 = mu. A Frobenius solution m_j ~ s^lambda v(t) of the linearization therefore
has lambda = -(<a_j> + i k) / mu for harmonic k, with <.> the mean over t and v = e^{ikt} exp(-int (a_j - <a_j>) dt).
At order n of the Taylor expansion the operator on the coefficient m_{j,n} is d_t + mu n + A (checked on a series),
so the analytic solution is fixed order by order except at a resonance n = lambda (k = 0 only). With a state it also
compares A at xi = -1 with the Jacobian entries of rt_model's residual there, which are exactly A since (1 + xi) and
the Fourier matrices' diagonals vanish. --mutate=a4 scales the symbolic a_4 by 1 + 1e-3; the comparison must fail.

exponents. lambda_2 = -<a_2>/mu and lambda_4 = -<a_4>/mu (k = 0 and k = 1/2 real parts) of each state at xi = -1.

planted (criteria written before the first run, in the docstring and I163-mac/README.md):
  (a) At a manufactured analytic state (low-degree Chebyshev polynomials in xi times low harmonics in t, with the
      unknowns' symmetries, so that the grid holds it exactly), the unscaled horizon rows R_n (n = 0..7) of rt_model
      (f64) and rt_mp (arb, 192 bits) equal the Taylor coefficients at xi = -1 of Omega_{2,3,4}^+ / xi that sympy
      computes from the closed-form fields: relative error below 1e-30 in arb, and in f64 the rows as scaled in the
      residual within 1e-12 of N^2, their largest entry. (Revised after the first run, whose f64 criterion was a
      relative error of 1e-9 on the unscaled rows. Those carry the rounding of entries up to 1e13 at order 7 and gave
      4.6e-5, while the f64 rows serve only the f64 Jacobian of the preconditioner. The f64 rows are now the rounded
      200-bit rows instead of repeated f64 products, which had lost 1e-2 at order 7.)
  (b) Through newton_mp with the LUdd preconditioner (rt_run.py's path), the forced system F_K(z) - F_K(z*) = 0 with
      z* the D = 50.007491 root of I145-mac resampled to 16 x 41 and K = 7, started from z* plus 1e-6 of a smooth
      perturbation, returns z* to max|z - z*| < 1e-30 within 10 Newton steps.
  (c) The order-0 row equals the collocated row at xi = -1 to 1e-40 (arb).
--mutate=taylor scales one coefficient of the series formula by 1 + 1e-3 (rt_model's taylor_mut); (a) and (c) must
then fail and the script exit 1.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import json
import argparse
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

t, X, mu, s_ = sp.symbols("t xi mu s")


def Omegas(s, w1, w2, w3, w4, Dop, Dn):
    """EEG Eq. (5a)-(5d), copied from rt_eqs_check.py (which checks it against the covariant equations)."""
    p, q = s, -s
    O1 = (X * (Dop(s, w1) + s * mu * w1) + mu * (Dn - 2) * w1
          + X / 4 * ((Dn - 3) * (w1 ** 2 + w2[1] ** 2 + w2[-1] ** 2 - 2 * w2[q] * (w2[p] + w1))
                     - 2 * (Dn - 1) * w1 * w2[p] + 4 * w3[p] * (w1 + w2[p] - w2[q]) - 4 * w4[p] ** 2))
    O2 = (X * (Dop(s, w2[q]) + s * mu * w2[q]) + mu * ((Dn - 3) * w1 + w2[1] + w2[-1])
          + X / 4 * ((Dn - 3) * (w1 ** 2 + w2[1] ** 2 + w2[-1] ** 2 - 2 * w1 * (w2[1] + w2[-1]))
                     - 2 * (Dn - 1) * w2[1] * w2[-1]))
    O3 = (X * (Dop(s, w3[q]) + s * mu * w3[q]) - mu / 2 * (Dn - 3) * (Dn - 2) * w1
          + X / 4 * (Dn - 2) * ((Dn - 3) * ((w2[1] + w2[-1]) * w1 - sp.Rational(1, 2) * w1 ** 2
                                            - sp.Rational(1, 2) * (w2[1] - w2[-1]) ** 2) - 2 * w4[1] * w4[-1]))
    O4 = (X * (Dop(s, w4[q]) + s * mu * w4[q]) + mu / 2 * (Dn - 2) * (w4[1] + w4[-1])
          - X / 2 * (Dn - 2) * (w4[1] * w2[-1] + w4[-1] * w2[1]))
    return [O1, O2, O3, O4]


# ------------------------------------------------------------------------------------------------ symbolic
def symbolic(state=None, mutate=None):
    Dn = sp.Symbol("D")
    fails = []
    # pointwise symbols at xi = -1: m_j = omega_j^-, p_j = omega_j^+, a1 = omega_1, and d_j = D^+ m_j
    m = {j: sp.Symbol(f"m{j}") for j in (2, 3, 4)}
    p = {j: sp.Symbol(f"p{j}") for j in (2, 3, 4)}
    dm = {j: sp.Symbol(f"d{j}") for j in (2, 3, 4)}
    a1 = sp.Symbol("a1")
    sym_of = {}

    def Dop(s, f):
        assert s == 1
        return dm[sym_of[f]] if f in sym_of else sp.Symbol("d1")      # D^+ omega_1 enters Omega_1 only

    W2 = {1: p[2], -1: m[2]}
    W3 = {1: p[3], -1: m[3]}
    W4 = {1: p[4], -1: m[4]}
    for j in (2, 3, 4):
        sym_of[m[j]] = j
    O = Omegas(1, a1, W2, W3, W4, Dop, Dn)
    G = {}
    for i in (2, 3, 4):
        e = sp.expand(O[i - 1] / X)
        # e = D^+ m_i + G_i: the coefficient of d_i must be 1 and no other d_j may appear
        ok = sp.simplify(sp.diff(e, dm[i]) - 1) == 0 and all(sp.diff(e, dm[j]) == 0 for j in (2, 3, 4) if j != i)
        print(f"  Omega_{i}^+ / xi = D^+ omega_{i}^- + G_{i}: {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"form of Omega_{i}")
        G[i] = sp.expand((e - dm[i]).subs(X, -1))
    order = (2, 4, 3)
    A = sp.Matrix(3, 3, lambda r, c: sp.simplify(sp.diff(G[order[r]], m[order[c]])))
    print("  A = dG/dm at xi = -1, rows and columns in the order (omega_2, omega_4, omega_3):")
    for r in range(3):
        print("    [" + ",  ".join(str(A[r, c]) for c in range(3)) + "]")
    tri = all(A[r, c] == 0 for r in range(3) for c in range(3) if c > r)
    print(f"  lower triangular: {'ok' if tri else 'FAIL'}")
    if not tri:
        fails.append("triangular")
    a2, a4, a3 = A[0, 0], A[1, 1], A[2, 2]
    if mutate == "a4":
        a4 = a4 * (1 + sp.Rational(1, 1000))
    ok3 = sp.simplify(a3 - mu) == 0
    print(f"  a_2 = {sp.factor(a2)}\n  a_4 = {sp.factor(a4)}\n  a_3 = {a3} ({'ok, lambda_3 = -1' if ok3 else 'FAIL'})")
    if not ok3:
        fails.append("a3")
    # the Frobenius operator at order n, on a series m_j = sum_n c_n s^n with D^+ c_n s^n = (c_n' + mu n c_n) s^n
    n = sp.Symbol("n", integer=True, nonnegative=True)
    print("  lambda_j = -(<a_j> + i k)/mu; order n of the series: (d_t + mu n + A) c_n = (lower orders)")
    # check the order-n statement on a series with K = 3 for omega_2's row (the triangular structure persists)
    ss = s_
    c = {j: [sp.Symbol(f"c{j}_{k}") for k in range(4)] for j in (2, 3, 4)}
    cd = {j: [sp.Symbol(f"cd{j}_{k}") for k in range(4)] for j in (2, 3, 4)}
    pc = {j: [sp.Symbol(f"q{j}_{k}") for k in range(4)] for j in (2, 3, 4)}
    ac = [sp.Symbol(f"b1_{k}") for k in range(4)]
    ser = lambda L: sum(L[k] * ss ** k for k in range(4))
    Ms = {j: ser(c[j]) for j in (2, 3, 4)}
    Dps = {j: sum((cd[j][k] + mu * k * c[j][k]) * ss ** k for k in range(4)) for j in (2, 3, 4)}
    sym2 = {}
    for j in (2, 3, 4):
        sym2[Ms[j]] = j
    Dop2 = lambda s, f: Dps[sym2[f]] if f in sym2 else sp.Symbol("dser1")
    O2s = Omegas(1, ser(ac), {1: ser(pc[2]), -1: Ms[2]}, {1: ser(pc[3]), -1: Ms[3]}, {1: ser(pc[4]), -1: Ms[4]},
                 Dop2, Dn)
    okser = True
    at0 = {m[jj]: c[jj][0] for jj in (2, 3, 4)}
    at0.update({p[jj]: pc[jj][0] for jj in (2, 3, 4)})
    at0[a1] = ac[0]
    for i, Oi in zip((2, 3, 4), O2s[1:]):
        R = sp.expand(sp.series(Oi.subs(X, ss - 1) / (ss - 1), ss, 0, 4).removeO())
        for k in range(1, 4):
            Rk = R.coeff(ss, k)
            for j in (2, 3, 4):
                want = mu * k * (1 if i == j else 0) + sp.diff(G[i], m[j]).subs(at0)
                bad = [sp.simplify(sp.diff(Rk, c[j][k]) - want) != 0, sp.diff(Rk, cd[j][k]) != (1 if i == j else 0)]
                bad += [sp.diff(Rk, c[j][kk]) != 0 or sp.diff(Rk, cd[j][kk]) != 0 for kk in range(k + 1, 4)]
                if any(bad):
                    okser = False
                    print(f"    order {k} of Omega_{i}, coefficient of omega_{j}: mismatch")
    print(f"  order-n operator d_t + mu n + A on the series (orders 1..3): {'ok' if okser else 'FAIL'}")
    if not okser:
        fails.append("series operator")
    if state:
        fails += compare_state(state, G, m, p, a1, Dn, a4)
    return fails, (a2, a4, m, p, a1, Dn)


def compare_state(path, G, m, p, a1, Dn, a4):
    """A at xi = -1 against rt_model's Jacobian entries there (a 16 x 41 resampling of the state)."""
    from rt_model import RT
    d = np.load(path)
    D = float(d["D"])
    src = RT(D, int(d["Nt"]), int(d["N"]))
    rt = RT(D, 16, 41)
    z = src.resample_to(d["zh"], rt)
    w1, w2, w3, w4, muv = rt.unpack(z)
    N, Nt = rt.N, rt.Nt
    order = (2, 4, 3)
    blocks = {2: 1, 3: 2, 4: 3}                       # field i's block in z (after omega_1's half block)
    off = lambda i: rt.n1 + (blocks[i] - 1) * rt.nf
    rowoff = {2: rt.n1, 3: rt.n1 + rt.nf, 4: rt.n1 + 2 * rt.nf}
    worst = 0.0
    for a in range(Nt):
        vals = {m[2]: w2[0, a, N], m[3]: w3[0, a, N], m[4]: w4[0, a, N], p[2]: -w2[0, a, 0], p[3]: -w3[0, a, 0],
                p[4]: -w4[0, a, 0], a1: w1[0, a, N], mu: muv[0, 0, 0], Dn: D}
        for i in order:
            for j in order:
                col = off(j) + a * (N + 1) + N
                E = np.zeros(rt.L)
                E[col] = 1.0
                Jij = ((rt.residual(z + E) - rt.residual(z - E)) / 2)[0, rowoff[i] + a * (N + 1) + N]
                sym = (a4 if (i, j) == (4, 4) else sp.diff(G[i], m[j]))
                Aij = float(sym.subs(vals))
                worst = max(worst, abs(Jij - Aij) / (1 + abs(Aij)))
    ok = worst < 1e-12
    print(f"  A against rt_model's Jacobian at xi = -1 ({path.rsplit('/', 1)[-1]}, 16x41, all t): max relative "
          f"difference {worst:.1e} (< 1e-12: {'ok' if ok else 'FAIL'})")
    return [] if ok else ["A vs Jacobian"]


def exponents(paths):
    from rt_model import RT
    print("| D | grid | mu | lambda_2 | lambda_4 | spread of a_2 over t | spread of a_4 over t |")
    print("|---|---|---|---|---|---|---|")
    for path in paths:
        d = np.load(path)
        z = d["zh"] if "zh" in d else d["z"]
        D, Nt, N = float(d["D"]), int(d["Nt"]), int(d["N"])
        rt = RT(D, Nt, N)
        w1, w2, w3, w4, muv = rt.unpack(z)
        mv = muv[0, 0, 0]
        m2, p2, W1 = w2[0, :, N], -w2[0, :, 0], w1[0, :, N]
        a2 = ((D - 3) * (m2 - W1) - (D - 1) * p2) / 2
        a4 = -mv * (D - 4) / 2 - (D - 2) * p2 / 2
        print(f"| {D:.6f} | {Nt}x{N} | {mv:.6f} | {-a2.mean() / mv:.4f} | {-a4.mean() / mv:.4f} | {np.ptp(a2):.3f} | "
              f"{np.ptp(a4):.3f} |")


# ------------------------------------------------------------------------------------------------ planted
def manufactured(rt, rng):
    """Closed-form fields (sympy) and their grid values: Chebyshev degree <= 9 in xi, harmonics <= 2 (5/2)."""
    xs = sp.Symbol("xi")
    exprs = {}
    for name, anti, odd in (("w1", False, True), ("w2", False, False), ("w3", False, False), ("w4", True, False)):
        e = 0
        ks = [sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(5, 2)] if anti else [0, 1, 2]
        for k in ks:
            for mm in range(10):
                if odd and mm % 2 == 0:
                    continue
                amp = sp.Rational(int(rng.integers(-60, 61)), 100) * sp.Rational(1, 2) ** (mm + 2 * k)
                trig = sp.cos(k * t) if rng.random() < 0.5 or k == 0 else sp.sin(k * t)
                e += amp * sp.chebyshevt(mm, xs) * trig
        exprs[name] = e
    mu_s = sp.Rational(13, 100)
    import mpmath
    import flint
    mpmath.mp.prec = 220
    tm = [2 * mpmath.pi * a / rt.Nt for a in range(rt.Nt)]
    xm = [mpmath.cos(mpmath.pi * j / rt.N) for j in range(rt.N + 1)]
    vals, valm = {}, {}
    for name, e in exprs.items():
        f = sp.lambdify((t, xs), e, "mpmath")
        g = np.array([[flint.arb(mpmath.nstr(f(ta, xj), 64)) for xj in xm] for ta in tm], dtype=object)
        valm[name] = g
        vals[name] = np.array([[float(v.mid()) for v in row] for row in g])
    return exprs, mu_s, xs, vals, valm


def planted(raw, mutate=None, out=None):
    import flint
    from rt_model import RT
    from rt_mp import RTmp
    import rt_mpsolve as mps
    arb = flint.arb
    flint.ctx.prec = 192
    tm = 1 + 1e-3 if mutate == "taylor" else 1.0
    K, D = 7, 50.007491
    rng = np.random.default_rng(163)
    rt = RT(D, 16, 41, horizon=K, taylor_mut=tm)
    exprs, mu_s, xs, vals, valm = manufactured(rt, rng)
    z = rt.pack(vals["w1"], vals["w2"], vals["w3"], vals["w4"], float(mu_s))
    # sympy: Omega_{2,3,4}^+ / xi of the closed forms, Taylor coefficients at xi = -1 on three t
    w = {nm: exprs[nm].subs(xs, X) for nm in exprs}
    refl = lambda e: -e.subs(X, -X)
    W2, W3, W4 = ({1: refl(w[nm]), -1: w[nm]} for nm in ("w2", "w3", "w4"))
    Dop = lambda s, f: s * sp.diff(f, t) + mu * (1 + s * X) * sp.diff(f, X)
    O = Omegas(1, w["w1"], W2, W3, W4, Dop, sp.Rational(D))           # the binary value of D, as arb(D)
    want = {}
    for i in (2, 3, 4):
        for a in (1, 6, 11):
            e = sp.expand(O[i - 1].subs({mu: mu_s, t: 2 * sp.pi * a / rt.Nt}).subs(X, s_ - 1))
            poly = sp.Poly(e, s_)
            co = [sp.N(poly.coeff_monomial(s_ ** k), 70) for k in range(K + 1)]
            want[(i, a)] = [-sum(co[:k + 1]) for k in range(K + 1)]          # / xi = -(1 + s + ...)
    # the code's rows: f64 and arb
    Rf = rt.horizon_rows(z[None, :])
    rm = RTmp(D, 16, 41, prec=192, horizon=K, taylor_mut=tm)
    zo = rt.pack(valm["w1"], valm["w2"], valm["w3"], valm["w4"], flint.arb(mu_s.p) / mu_s.q)
    w1m, w2m, w3m, w4m, mum = rm.unpack_m(zo)
    Rm = rm.horizon_rows_m(w1m, w2m, w3m, w4m, mum)
    ea = ef = 0.0
    for i in (2, 3, 4):
        for a in (1, 6, 11):
            for k in range(K + 1):
                ref = float(want[(i, a)][k])
                scale = max(1.0, abs(ref))
                refm = arb(str(want[(i, a)][k]))
                ea = max(ea, abs(float((Rm[i - 2][a, k] - refm).mid())) / scale)
                ef = max(ef, abs(Rf[i - 2][0, a, k] - ref) * rt.tscale[k] / rt.N ** 2)
    ca, cf = ea < 1e-30, ef < 1e-12
    print(f"(a) horizon rows against sympy's Taylor coefficients (orders 0..{K}, three t): arb {ea:.1e} (< 1e-30: {ca}),"
          f" f64 rows as scaled in the residual, relative to N^2 {ef:.1e} (< 1e-12: {cf})")
    # (c) order 0 equals the collocated row at xi = -1 (from the unchanged collocation formulas)
    r0 = RTmp(D, 16, 41, prec=192)
    F0 = r0.residual_m(zo)
    nf, n1, N = rt.nf, rt.n1, rt.N
    e0 = 0.0
    for i in (2, 3, 4):
        for a in range(rt.Nt):
            e0 = max(e0, abs(float((Rm[i - 2][a, 0] - F0[n1 + (i - 2) * nf + a * (N + 1) + N]).mid())))
    cc = e0 < 1e-40
    print(f"(c) order-0 row against the collocated row at xi = -1: {e0:.1e} (< 1e-40: {cc})")
    # (b) Newton through newton_mp on the forced system, from the D = 50 root resampled
    d = np.load(os.path.join(raw, "I145-mac", "cont", "cont_32x101_D50.0075_32x101.npz"))
    src = RT(float(d["D"]), int(d["Nt"]), int(d["N"]))
    zs = src.resample_to(d["zh"], rt)
    zso = mps.obj(zs)
    fstar = rm.residual_m(zso)
    resid = lambda x: rm.residual_m(x) - fstar
    co = RT(D, 8, 11)
    T8, X8 = np.meshgrid(co.t, co.xi, indexing="ij")
    pf = co.pack(np.sin(np.pi * X8 / 2) * np.cos(T8), np.cos(T8) * X8 ** 2, np.sin(T8) * (1 - X8),
                 np.cos(T8 / 2) * (1 + X8 ** 3), 0.3)
    pert = 1e-6 * co.resample_to(pf, rt)
    J = rt.jacobian(zs + pert)
    P = mps.LUdd(J, overwrite=True)
    x, ok, info = mps.newton_mp(None, mps.obj(zs + pert), P, tol=1e-50, it_max=10, residual=resid, max_products=120,
                                log=print, full_below=1e-4, stall_any=False)
    err = mps.amax(mps.mids(x - zso))
    steps = len(info["res"]) - 1
    cb = err < 1e-30 and steps <= 10
    print(f"(b) forced system at D = {D} on 16x41 with K = {K}: max|z - z*| {err:.1e} after {steps} Newton steps "
          f"(< 1e-30 within 10: {cb}); products {info['products']}")
    passed = ca and cf and cb and cc
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(dict(test="planted_horizon", mutate=mutate, K=K, err_arb=ea, err_f64=ef, err_order0=e0,
                                    err_newton=err, steps=steps, products=info["products"],
                                    crit=[bool(ca), bool(cf), bool(cb), bool(cc)])) + "\n")
    print("PASS" if passed else "FAIL")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["symbolic", "exponents", "planted"])
    ap.add_argument("states", nargs="*")
    ap.add_argument("--mutate", default=None, choices=["a4", "taylor"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None, help="planted: the raw-data repository (its I145-mac/cont state at D 50.01)")
    a = ap.parse_args()
    if a.what == "symbolic":
        fails, _ = symbolic(a.states[0] if a.states else None, a.mutate)
        print("exit", 1 if fails else 0, fails)
        sys.exit(1 if fails else 0)
    if a.what == "exponents":
        exponents(a.states)
        sys.exit(0)
    if not a.raw:
        ap.error("planted needs --raw")
    sys.exit(0 if planted(a.raw, a.mutate, a.out) else 1)


if __name__ == "__main__":
    main()
