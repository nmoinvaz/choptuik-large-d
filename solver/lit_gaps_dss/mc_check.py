"""How a constraint violation propagates in two first-order forms of the DSS equations (notes/LIT_GAPS_DSS.md, lead 1).

Variables of solver/dss.py and hp/src/model.rs: n = D - 2, x = r/(-t), tau = -ln(-t), fields p, m, a, c, and
  H_rhs = (1/2)[p^2 + m^2 + (n-1)(1 - a^2)],   flux = c (p^2 - m^2)/(2x).
Einstein's two constraints are  x a_x/a = H_rhs  (Hamiltonian, G^t_t)  and  (a_tau + x a_x)/a = flux  (momentum,
G_tr; solver/matter/struct_common.py's a_t/a = (kappa/n) r T_tr in these variables).

Form A (ours): x-equations for c, a (the Hamiltonian constraint), p, m; the momentum constraint is not imposed.
  Its residual E = (a_tau + x a_x)/a - flux obeys  x E_x = -(n-1) a^2 E  identically (check A), so a violation grows
  inward as about x^-(n-1): the mass mode of notes/LARGE_D_BASICS.md 3.3 is the momentum-constraint violation.
Form S (Gundlach, PRD 55, 695 (1997), Eq. (19); Martin-Garcia and Gundlach, PRD 68, 024011 (2003), Eq. (17), both at
  D = 4): x-equations for c, p, m only; a slaved at each x by the momentum constraint with a_x replaced by the
  Hamiltonian right-hand side,  a_tau/a = flux - H_rhs.  Its Hamiltonian residual H = x a_x/a - H_rhs obeys
  H_tau = (n-1) a^2 H  identically (check S), so a solution periodic in tau has H = 0 at every x.
Usage: python3 mc_check.py [--mutate=flux]   (flux flips the flux term in form S; check S must then fail).
Exits non-zero if a check fails.
"""
import sys

import sympy as sp

mut = "--mutate=flux" in sys.argv
n, x = sp.symbols("n x", positive=True)
p, m, a, c, pt, mt, H, at, E = sp.symbols("p m a c p_t m_t H a_t E")   # values; p_t etc. are d/dtau

RH = sp.Rational(1, 2) * (p**2 + m**2 + (n - 1) * (1 - a**2))
flux = c * (p**2 - m**2) / (2 * x)

cx = (n - 1) * c * (a**2 - 1) / x
px = (-pt + cx * p - c * p / x + n * c * (p + m) / (2 * x)) / (x - c)
mx = (-mt - cx * m + c * m / x - n * c * (p + m) / (2 * x)) / (x + c)


def ddx(expr, ax):
    """Total x-derivative along the x-equations (c_x, p_x, m_x above, a_x given)."""
    return (sp.diff(expr, x) + sp.diff(expr, p) * px + sp.diff(expr, m) * mx + sp.diff(expr, a) * ax
            + sp.diff(expr, c) * cx)


def ddt(expr, a_t):
    """Total tau-derivative of an expression in p, m, a (c_tau does not occur in the expressions differentiated)."""
    return sp.diff(expr, p) * pt + sp.diff(expr, m) * mt + sp.diff(expr, a) * a_t


# Form S: a_tau from the slaving, a_x = a (H + H_rhs)/x with H free; H_tau = x d_x(d_tau ln a) - d_tau H_rhs.
lna_t = (-flux if mut else flux) - RH
Ht = x * ddx(lna_t, a * (H + RH) / x) - ddt(RH, a * lna_t)
res_S = sp.simplify(sp.together(Ht - (n - 1) * a**2 * H))
# Form A: a_x = a H_rhs/x, a_tau free; x E_x = d_tau(x a_x/a) + x d_x(H_rhs - flux), then a_tau written through E.
xEx = ddt(RH, at) + x * ddx(RH - flux, a * RH / x)
xEx = sp.simplify(sp.together(xEx.subs(at, a * (E - RH + flux))))
coefA = sp.factor(sp.diff(xEx, E))
res_A = sp.simplify(xEx - sp.diff(xEx, E) * E)
print("check S: H_tau - (n-1) a^2 H =", res_S)
print("check A: x E_x = (%s) E + (%s)" % (coefA, res_A))
okS = res_S == 0
okA = res_A == 0 and sp.simplify(coefA + (n - 1) * a**2) == 0
print("S:", "holds" if okS else "FAILS", "| A:", "holds" if okA else "FAILS")
sys.exit(0 if (okS and okA) else 1)
