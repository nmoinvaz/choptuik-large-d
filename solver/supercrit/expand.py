#!/usr/bin/env python3
"""1/n expansion of Ecker, Ecker & Grumiller's supercritical branch in this repository's variables, to any order K.

Variables (eqmap.py, lo_check.py): eps = 1/n, y = p - m, sig = p + m, u = (n-1)(b-1), c; d/dtau = omega d/dtheta.
Exact equations (README "Large D", in sum/difference form):
    E1  x c_x = c u
    E2  x u_x = (n-1+u)(P - u),  P = (y^2 + sig^2)/2
    (A) x^2 y_x + x y_tau - x c sig_x = c (n-1+u) sig
    (B) x^2 sig_x + x sig_tau - x c y_x = c (u-1) y
Branch variables: beta(tau) > 0, f(tau) > 0, z = beta^2 x^2 - 1 > 0, w = sqrt(z), x^2 = (z+1)/beta^2, and
    y = sqrt2 (x/w) Yh,   c = w Ch,   u = Uh,   sig = sqrt2 x^2 Sh,
with Yh = sum eps^k Y_k, etc. Leading order: Y_0 = beta, C_0 = f, U_0 = (z+1)/z, S_1 = -(beta + beta')/(f z^2).
In these variables the equations are rational in z (x d_x = 2(z+1) d_z; d_tau at fixed x = D + 2 (beta'/beta)(z+1) d_z,
with D = d_tau at fixed z, acting on beta^(j), f^(j) and the integration functions):
    (A) [-Y/z + 2(z+1)Y' - (b'/b)(z+1)Y/z + DY + 2(b'/b)(z+1)Y'] - z C (2S + 2(z+1)S') = z C (n-1+U) S
    (B) x^2 [2S + 2(z+1)S' + DS + 2(b'/b)(z+1)S'] - C(-Y/z + 2(z+1)Y') = C (U-1) Y
    E1  ((z+1)/z) C + 2(z+1) C' = C U
    E2  2(z+1) U' = (n-1+U)(P - U),   P = ((z+1)/(beta^2 z)) Y^2 + ((z+1)^2/beta^4) S^2
At order k >= 1: U_k from E2 (algebraic), then (B) reduces to  -2 f ((z+1)/z) (z Y_k)' = -R_B  (the C_k terms cancel
identically), so Y_k = (1/z)[int z R_B/(2f(z+1)) dz + A_k(tau)]; E1 reduces to 2(z+1) C_k' = R_1, so
C_k = int R_1/(2(z+1)) dz + B_k(tau); and (A) at order k+1 gives S_{k+1} algebraically. No step has a solvability
condition: A_k, B_k are free at every order. A_k/z is the first-order change of Y under beta -> beta - eps^k A_k.

Normalization ("--norm"):
    free  : keep A_k, B_k as symbols (default; shows the freedom). At k = 1 the script checks that the A_1, B_1 parts
            are exactly the first-order changes of the leading order under beta -> beta - eps A_1, f -> f + eps B_1.
    zero  : A_k = B_k = 0, i.e. z Y_k and C_k have no z^0 term (other than log terms) in their expansion at z = 0;
            this is the convention that defines beta(tau) and f(tau) beyond leading order.
The script prints each order, the large-z and small-z behaviour, and writes the series to logs/expand_K{K}_{norm}.pkl
for formal_check.py.

    python3 expand.py 2 --norm=zero         # NLO and NNLO (K = 2), general beta(tau), f(tau)
    python3 expand.py 2 --norm=zero --css   # the same for CSS (beta, f constant): much faster
"""
import pickle
import sys
import sympy as sp

args = [a for a in sys.argv[1:] if not a.startswith("--")]
K = int(args[0]) if args else 2
norm = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--norm=")), "free")
css = "--css" in sys.argv
NDER = 2 * K + 4
z, eps = sp.symbols("z epsilon", positive=True)
Bd = sp.symbols(f"b0:{NDER + 1}")            # beta, beta', beta'', ...
Fd = sp.symbols(f"f0:{NDER + 1}")
Ad = {k: sp.symbols(f"A{k}_0:{NDER + 1}") for k in range(1, K + 1)}
Cd = {k: sp.symbols(f"B{k}_0:{NDER + 1}") for k in range(1, K + 1)}
Lz, L1 = sp.symbols("Lz L1")                  # ln z, ln(z+1)
chains = [Bd, Fd] + list(Ad.values()) + list(Cd.values())


def D(e):
    """d/dtau at fixed z"""
    if css:
        return sp.Integer(0)
    out = 0
    for ch in chains:
        for j in range(NDER):
            out += sp.diff(e, ch[j]) * ch[j + 1]
    return sp.expand(out)


def dz(e):
    return sp.expand(sp.diff(e, z) + sp.diff(e, Lz) / z + sp.diff(e, L1) / (z + 1))


def is_zero(e):
    """exact zero test for expressions polynomial in Lz, L1 with rational coefficients (faster than simplify)"""
    e = sp.expand(e)
    if e == 0:
        return True
    P = sp.Poly(e, Lz, L1)
    return all(sp.cancel(sp.together(cf)) == 0 for cf in P.coeffs())


def _Iz(k, a):
    """int z^k Lz^a dz"""
    if k == -1:
        return Lz ** (a + 1) / (a + 1)
    if a == 0:
        return z ** (k + 1) / (k + 1)
    return z ** (k + 1) * Lz ** a / (k + 1) - sp.Rational(a, k + 1) * _Iz(k, a - 1)


def _Iw(m, a):
    """int (z+1)^(-m) Lz^a dz, m >= 1"""
    if a == 0:
        return L1 if m == 1 else -(z + 1) ** (1 - m) / (m - 1)
    if m == 1:
        raise NotImplementedError("dilogarithm")
    # by parts; (z+1)^(1-m)/z = 1/z - sum_{j=1}^{m-1} (z+1)^(-j)
    rest = Lz ** a / a - sum(_Iw(j, a - 1) for j in range(1, m))
    return -Lz ** a * (z + 1) ** (1 - m) / (m - 1) + sp.Rational(a, m - 1) * rest


def integrate_z(e):
    """antiderivative in z of an expression polynomial in Lz (and free of L1) with rational coefficients;
    terms z^k Lz^a and (z+1)^-m Lz^a are integrated exactly (no additive constants); anything else falls back to sympy"""
    e = sp.expand(e)
    out = 0
    try:
        P = sp.Poly(e, Lz, L1)
        for (a, bb), cf in P.terms():
            if bb:
                raise NotImplementedError("L1 in the integrand")
            R = sp.expand(sp.apart(sp.cancel(sp.together(cf)), z))
            for t in sp.Add.make_args(R):
                cst, zp = t.as_independent(z, as_Add=False)
                if zp == 1:
                    out += cst * _Iz(0, a)
                    continue
                base, ex = zp.as_base_exp()
                if base == z:
                    out += cst * _Iz(int(ex), a)
                elif sp.expand(base - z - 1) == 0 and ex < 0:
                    out += cst * _Iw(int(-ex), a)
                else:
                    raise NotImplementedError(f"term {zp}")
        r = sp.expand(out)
    except NotImplementedError as err:
        print("   (falling back to sympy.integrate:", err, ")", flush=True)
        ee = e.subs({Lz: sp.log(z), L1: sp.log(z + 1)})
        r = sp.expand(sp.expand_log(sp.integrate(ee, z), force=True)).subs({sp.log(z): Lz, sp.log(z + 1): L1})
        assert not r.has(sp.Integral), "integration failed"
    assert is_zero(dz(r) - e), "antiderivative check"
    return r


b, bp = Bd[0], (0 if css else Bd[1])
f = Fd[0]
x2 = (z + 1) / b ** 2
N = K + 2
Y = [b] + [0] * (N - 1)
C = [f] + [0] * (N - 1)
U = [(z + 1) / z] + [0] * (N - 1)
S = [0] * (N + 1)


def mul(a, c, k):
    return sp.expand(sum(a[i] * c[k - i] for i in range(k + 1) if i < len(a) and k - i < len(c)))


def LA(k):     # order-k part of (A)'s left side without the eps factor, minus the z C(2S + 2(z+1)S') part
    Yk = Y[k]
    t = -Yk / z + 2 * (z + 1) * dz(Yk) - (bp / b) * (z + 1) * Yk / z + D(Yk) + 2 * (bp / b) * (z + 1) * dz(Yk)
    Sq = [2 * S[i] + 2 * (z + 1) * dz(S[i]) for i in range(N)]
    return sp.expand(t - z * mul(C, Sq, k))


def P_ser(k):
    return sp.expand((z + 1) / (b ** 2 * z) * mul(Y, Y, k) + (z + 1) ** 2 / b ** 4 * mul(S, S, k))


# S_1 from (A) at order 1 (eps * LHS at order 1 = LHS_0)
S[1] = sp.together(LA(0) / (z * f))
assert is_zero(S[1] + (b + bp) / (f * z ** 2))
report = {}
for k in range(1, K + 1):
    # --- U_k (E2 times eps at order k): (P-U)_k = 2(z+1)U'_{k-1} - sum_{i>=1} q_i (P-U)_{k-i},  q = 1 - eps + eps U
    q = [sp.Integer(1), sp.Integer(-1) + U[0]] + [U[i - 1] for i in range(2, k + 1)]
    PmU = [sp.expand(P_ser(j) - U[j]) for j in range(k)]
    rhs = 2 * (z + 1) * dz(U[k - 1]) - sum(q[i] * PmU[k - i] for i in range(1, k + 1))
    Yk = sp.Symbol("Yk")
    U[k] = sp.expand(P_ser(k) - rhs)                       # contains Y[k] = 0 so far; add its part below
    Upart = 2 * (z + 1) / (b * z) * Yk                      # U_k's dependence on Y_k (2 (z+1) Y0 Yk / (beta^2 z))
    # --- (B) at order k with Y_k = C_k = 0 (U_k known part only)
    Sq = [2 * S[i] + 2 * (z + 1) * dz(S[i]) + D(S[i]) + 2 * (bp / b) * (z + 1) * dz(S[i]) for i in range(N)]
    Yq = [-Y[i] / z + 2 * (z + 1) * dz(Y[i]) for i in range(N)]
    Um1 = [U[0] - 1] + U[1:]
    RB = sp.expand(x2 * Sq[k] - mul(C, Yq, k) - sum(C[i] * mul(Um1, Y, k - i) for i in range(k + 1)))
    # (zY_k)' = z RB / (2 f (z+1))
    Ik = integrate_z(z * RB / (2 * f * (z + 1)))
    Y[k] = sp.expand((Ik + (Ad[k][0] if norm == "free" else 0)) / z)
    U[k] = sp.expand(U[k] + Upart.subs(Yk, Y[k]))
    # check (B) at order k
    Yq = [-Y[i] / z + 2 * (z + 1) * dz(Y[i]) for i in range(N)]
    Um1 = [U[0] - 1] + U[1:]
    assert is_zero(x2 * Sq[k] - mul(C, Yq, k) - sum(C[i] * mul(Um1, Y, k - i) for i in range(k + 1))), f"(B) at order {k}"
    # --- C_k from E1: 2(z+1)C_k' = [C U]_k - ((z+1)/z) C_k - ... with C_k = 0
    R1 = sp.expand(mul(C, U, k) - (z + 1) / z * C[k] - 2 * (z + 1) * dz(C[k]))
    C[k] = sp.expand(integrate_z(R1 / (2 * (z + 1))) + (Cd[k][0] if norm == "free" else 0))
    assert is_zero(mul(C, U, k) - (z + 1) / z * C[k] - 2 * (z + 1) * dz(C[k])), f"E1 at order {k}"
    # --- S_{k+1} from (A): LA(k) = [z C (1 - eps + eps U) S]_{k+1}
    qq = [sp.Integer(1), sp.Integer(-1) + U[0]] + [U[i - 1] for i in range(2, k + 2)]
    CqS = sum(mul(C, qq, i) * S[k + 1 - i] for i in range(1, k + 1))   # terms without S_{k+1}
    S[k + 1] = sp.expand(sp.together((LA(k) - z * CqS) / (z * f)))
    for name, e in (("Y", Y[k]), ("C", C[k]), ("U", U[k]), ("S", S[k + 1])):
        print(f"order {k}: {name}_{k if name != 'S' else k + 1} = {sp.factor(sp.together(e)) if not e.has(Lz, L1) else sp.collect(e, [Lz, L1])}",
              flush=True)

# --- meaning of A_1, B_1: shifts of beta and f (free run only)
if norm == "free":
    bb, ff = sp.symbols("bb ff", positive=True)
    # leading order in physical form as functions of (beta, f) at fixed x, written back in hat variables at the
    # unshifted z: y0 = sqrt2 x beta / sqrt(beta^2 x^2 - 1), c0 = f sqrt(beta^2 x^2 - 1), u0 = beta^2 x^2/(beta^2 x^2 - 1)
    xx = sp.Symbol("xx", positive=True)
    y0 = sp.sqrt(2) * xx * bb / sp.sqrt(bb ** 2 * xx ** 2 - 1)
    c0 = ff * sp.sqrt(bb ** 2 * xx ** 2 - 1)
    u0 = bb ** 2 * xx ** 2 / (bb ** 2 * xx ** 2 - 1)
    at = {bb: b, ff: f, xx: sp.sqrt(z + 1) / b}
    wv = sp.sqrt(z)
    dY = sp.simplify((sp.diff(y0, bb) * (-Ad[1][0])).subs(at) / (sp.sqrt(2) * sp.sqrt(z + 1) / (b * wv)))
    dC = sp.simplify((sp.diff(c0, bb) * (-Ad[1][0]) + sp.diff(c0, ff) * Cd[1][0]).subs(at) / wv)
    dU = sp.simplify((sp.diff(u0, bb) * (-Ad[1][0])).subs(at))
    okA = [sp.simplify(sp.diff(Y[1], Ad[1][0]) * Ad[1][0] - dY) == 0,
           sp.simplify(sp.expand(C[1]).coeff(Cd[1][0]) * Cd[1][0] + sp.diff(C[1], Ad[1][0]) * Ad[1][0] - dC
                       + (-Ad[1][0]) * f / b) == 0,
           sp.simplify(sp.diff(U[1], Ad[1][0]) * Ad[1][0] - dU) == 0]
    print("A_1 is the shift beta -> beta - eps A_1 in Y, C (up to a constant absorbed in B_1), U:", okA)
    assert all(okA)

# --- behaviour at the two ends
print("\nlarge z (x -> oo), leading terms:")
w_ = sp.Symbol("v", positive=True)       # v = 1/z
for name, ser in (("Y", Y), ("C", C), ("U", U), ("S", S)):
    for k in range(1, K + 1 + (name == "S")):
        e = ser[k]
        if e == 0:
            continue
        ee = e.subs({Lz: -sp.log(w_), L1: sp.log(1 + w_) - sp.log(w_)}).subs(z, 1 / w_)
        lead = sp.series(sp.expand(ee), w_, 0, 2).removeO()
        print(f"   {name}_{k}: {sp.collect(sp.expand(lead), [sp.log(w_), w_])}")
print("\nsmall z (beta^2 x^2 -> 1), most singular power:")
for name, ser in (("Y", Y), ("C", C), ("U", U), ("S", S)):
    for k in range(0, K + 1 + (name == "S")):
        e = ser[k]
        if e == 0:
            continue
        ee = sp.expand(e.subs({L1: sp.log(1 + z), Lz: sp.log(z)}))
        num, den = sp.fraction(sp.together(ee.subs(sp.log(z), 0).subs(sp.log(1 + z), 0)))
        pw = sp.Poly(den, z).monoms()[-1][0] if den.has(z) else 0
        lowest = sp.Poly(num, z).monoms()[-1][0] if num.has(z) else 0
        print(f"   {name}_{k} ~ z^{lowest - pw}  (+ log z terms: {ee.has(sp.log(z))})")
tag = f"K{K}_{norm}" + ("_css" if css else "")
with open(f"logs/expand_{tag}.pkl", "wb") as fh:
    pickle.dump({"K": K, "Y": Y, "C": C, "U": U, "S": S, "syms": (z, eps, Bd, Fd, Ad, Cd, Lz, L1)}, fh)
print("saved", f"logs/expand_{tag}.pkl")
