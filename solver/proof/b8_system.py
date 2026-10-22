#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 1): the reduced form of the M-harmonic outer system used in the proof,
checked symbolically against the self-similar equations of proof/C_selection.md section 1.

Variables (p = eps^(1/2) P, m = eps^(1/2) M, eps = 1/n):
    S = P + M (fast),   Delta = P - M,   v = P_M[c Delta] - x S (slow),   V = v / x,
    beta = (n - 1)(b - 1) (fast, the b-mode),   c (slow).
Claims checked here:
  (1) general M (any number of harmonics, c and b theta-dependent, products projected):  x v_x = v + omega x S_theta
      exactly, i.e. x V' = omega dS/dtheta.  Checked for the M = 3 Galerkin model with symbolic harmonics.
  (2) M = 1 (b, c harmonic 0; mu = +-1 the harmonic of p, m):
        x V_mu'                 = i mu omega S_mu
        x (x^2 - c^2) S_mu'     = [(n-1) c^2 b - x^2 (1 + 2 i mu omega - beta)] S_mu - (1 + i mu omega - beta) x^2 V_mu
        x beta' + (n-1) beta + beta^2 = (1 - eps) b H,   H = S_+ S_- + x^2 Y_+ Y_- / c^2,  Y = V + S,  b = 1 + beta/(n-1)
        x c'                    = c beta
  (3) the order-1 part of the fast rate: lambda = A/(x(x^2-c^2)) = n/(x(x^2-1)) + lambda1_0 + O(eps), with
      lambda1_0 = [-1 - x^2 (1 + 2 i mu omega - 2 <F^2>)] / (x (x^2 - 1)), when c = 1 - eps <F^2>(1 - x^2) + O(eps^2).
Exits non-zero if a check fails.  --mutate=sign flips the sign of the omega term in (2) and must make it fail.

    python3 b8_system.py [--mutate=sign]
"""
import sys
import sympy as sp

MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), "")
x, n, w = sp.symbols("x n omega")
eps = 1 / n
ok = True


def check(name, expr):
    global ok
    e = sp.simplify(sp.expand(expr))
    good = (e == 0)
    ok &= good
    print(f"  {name:62s} {'OK' if good else 'FAIL: ' + str(e)[:200]}")


# ---------------------------------------------------------------- (2) M = 1, harmonics mu = +-1, complexified
print("(2) M = 1 reduced system")
pp, pm, mp, mm, b, c = [sp.Function(nm)(x) for nm in ("pp", "pm", "mp", "mm", "b", "c")]
P = {1: pp, -1: pm}
Mf = {1: mp, -1: mm}
G = (n - 1) * c * (b - 1)
# original equations, harmonic mu (d/dtheta -> i mu)
E3 = {mu: x * (x - c) * sp.diff(P[mu], x) - (-w * x * sp.I * mu * P[mu] + G * P[mu] - c * P[mu]
                                             + n / 2 * c * (P[mu] + Mf[mu])) for mu in (1, -1)}
E4 = {mu: x * (x + c) * sp.diff(Mf[mu], x) - (-w * x * sp.I * mu * Mf[mu] - G * Mf[mu] + c * Mf[mu]
                                              - n / 2 * c * (P[mu] + Mf[mu])) for mu in (1, -1)}
q2 = 2 * (pp * pm + mp * mm)                       # harmonic 0 of p^2 + m^2 (unscaled fields)
E1 = x * sp.diff(c, x) - (n - 1) * c * (b - 1)
E2 = x * sp.diff(b, x) - (b * q2 - (n - 1) * b * (b - 1))
# new variables (scaled by sqrt(n): P = sqrt(n) p)
sq = sp.sqrt(n)
S = {mu: sq * (P[mu] + Mf[mu]) for mu in (1, -1)}
Dl = {mu: sq * (P[mu] - Mf[mu]) for mu in (1, -1)}
V = {mu: (c * Dl[mu] - x * S[mu]) / x for mu in (1, -1)}
beta = (n - 1) * (b - 1)
# solve the original system for the derivatives
sol = sp.solve([E3[1], E3[-1], E4[1], E4[-1], E1, E2],
               [sp.diff(f, x) for f in (pp, pm, mp, mm, c, b)], dict=True)[0]
sub = lambda e: sp.expand(e.doit().subs(sol))
sgn = -1 if MUT == "sign" else 1
for mu in (1, -1):
    check(f"x V_{mu:+d}' - i mu omega S", sub(x * sp.diff(V[mu], x)) - sgn * sp.I * mu * w * S[mu])
    A = (n - 1) * c**2 * b - x**2 * (1 + 2 * sp.I * mu * w - beta)
    B = 1 + sp.I * mu * w - beta
    check(f"x(x^2-c^2) S_{mu:+d}' - A S + B x^2 V", sub(x * (x**2 - c**2) * sp.diff(S[mu], x)) - A * S[mu]
          + B * x**2 * V[mu])
H = S[1] * S[-1] + x**2 * (V[1] + S[1]) * (V[-1] + S[-1]) / c**2
check("x beta' + (n-1) beta + beta^2 - (1-eps) b H", sub(x * sp.diff(beta, x)) + (n - 1) * beta + beta**2
      - (1 - eps) * b * H)
check("x c' - c beta", sub(x * sp.diff(c, x)) - c * beta)

# ---------------------------------------------------------------- (1) general M: the slow identity, M = 3 Galerkin
print("(1) slow identity x V' = omega S_theta for the M = 3 Galerkin model (c, b with harmonics 0, +-2)")
odd = (-3, -1, 1, 3)
even = (-2, 0, 2)
pf = {j: sp.Function(f"p{j}")(x) for j in odd}
mf = {j: sp.Function(f"m{j}")(x) for j in odd}
cf = {j: sp.Function(f"c{j}")(x) for j in even}
bf = {j: sp.Function(f"b{j}")(x) for j in even}


def prod(a, bb, keep):
    out = {k: 0 for k in keep}
    for i, ai in a.items():
        for j, bj in bb.items():
            if i + j in out:
                out[i + j] += ai * bj
    return out


Gf = {k: (n - 1) * v for k, v in prod(cf, {j: bf[j] - (1 if j == 0 else 0) for j in even}, even).items()}
cp_x = prod(cf, {j: sp.diff(pf[j], x) for j in odd}, odd)
cm_x = prod(cf, {j: sp.diff(mf[j], x) for j in odd}, odd)
Gp, Gm = prod(Gf, pf, odd), prod(Gf, mf, odd)
cp, cm = prod(cf, pf, odd), prod(cf, mf, odd)
cpm = prod(cf, {j: pf[j] + mf[j] for j in odd}, odd)
E3g = {j: x * x * sp.diff(pf[j], x) - x * cp_x[j] - (-w * x * sp.I * j * pf[j] + Gp[j] - cp[j] + n / 2 * cpm[j])
       for j in odd}
E4g = {j: x * x * sp.diff(mf[j], x) + x * cm_x[j] - (-w * x * sp.I * j * mf[j] - Gm[j] + cm[j] - n / 2 * cpm[j])
       for j in odd}
E1g = {k: x * sp.diff(cf[k], x) - Gf[k] for k in even}
# x v_x - v - omega x S_theta, with v = P_M[c Delta] - x S (no b equation needed)
Sg = {j: pf[j] + mf[j] for j in odd}
Dg = {j: pf[j] - mf[j] for j in odd}
cD = prod(cf, Dg, odd)
for j in (1, 3):
    vj = cD[j] - x * Sg[j]
    expr = x * sp.diff(vj, x) - vj - sgn * w * x * sp.I * j * Sg[j]
    # reduce with the equations: E3 + E4 gives P_M[x c Delta_x] in terms of S; E1 gives x c_x
    comb = (E3g[j] + E4g[j])
    # expr + comb should vanish identically after using x c_x = G (E1)
    tot = sp.expand(expr + comb)
    tot = tot.subs({sp.diff(cf[k], x): Gf[k] / x for k in even})
    check(f"M = 3, harmonic {j}: x v' - v - omega x S_theta (mod equations)", tot)

# ---------------------------------------------------------------- (3) order-1 part of the fast rate, M = 1
print("(3) order-1 part of the fast rate lambda = A/(x(x^2-c^2)), M = 1")
e, F2, mu = sp.symbols("e F2 mu")
cc = 1 - e * F2 * (1 - x**2)
A = (1 / e - 1) * cc**2 * 1 - x**2 * (1 + 2 * sp.I * mu * w)       # b = 1 + O(e^2), beta = O(e)
lam = A / (x * (x**2 - cc**2))
lam0 = sp.series(lam, e, 0, 1).removeO()
lam1 = sp.simplify(lam0 - 1 / (e * x * (x**2 - 1)))
target = (-1 - x**2 * (1 + 2 * sp.I * mu * w - 2 * F2)) / (x * (x**2 - 1))
check("lambda - n/(x(x^2-1)) at order eps^0 = lambda1_0", lam1 - sgn * target if MUT == "sign" else lam1 - target)

print("ALL OK" if ok else "SOME CHECK FAILED")
sys.exit(0 if ok else 1)
