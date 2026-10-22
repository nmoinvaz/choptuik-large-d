"""(b), (d) The large-n inner problems (r ~ sqrt(n), s = r^2 / (2n), physical time t), derived with sympy from the exact
covariant polar-areal equations of struct_eqs.py, for each model.

    nice -n 10 python3 struct_inner.py            (about 1 minute)

Scalings (kappa = 2n normalization; eps = 1/n):
    multiplet, l fixed:   phi = eps^{1/2} vphi(s, t),   b = a^2 = 1 + eps beta,   c = alpha / a = O(1)
    wave map, eta fixed:  u = O(1),                     b = 1 + eps beta                (eta = kappa f^2 / 2)
    Yang-Mills U-system:  U = eps^{1/2} V                (W = 1 - (r / l_YM) U, r << l_YM)
Results (leading order in eps; every residual printed below must be 0):
    multiplet:  d_t(vphi_t / c) = c (vphi_s - l vphi / (2 s)),   c_s = 2 vphi_t^2 / c,   beta = 4 s vphi_t^2/c^2 + 2 l vphi^2
    wave map:   d_t(u_t / c)    = c (u_s - sin(2u) / (4 s)),     c_s = 2 eta u_t^2 / c,   beta = 4 eta s u_t^2/c^2 + 2 eta sin^2 u
    YM U-system: identical to the l = 1 multiplet with V = sqrt(2) vphi (the YM-specific terms are O(eps)).
For l = 0 the multiplet system is the potential form of GAMMA.md's heat system: Y = 2 sqrt2 vphi_t / c, S = 2 sqrt2 vphi_s
satisfy Y_t = c S, S_t = c (Y_s + Y^3/4), c_s = c Y^2 / 4.
Why these scalings: with phi = eps^sigma vphi, c_s = 2 n^{1 - 2 sigma} vphi_t^2 / c; sigma > 1/2 leaves a linear (gravity-
free) inner problem, whose multipliers cannot vanish (Proposition 2's C(n,s) != 0 has the same Weber/Hermite structure),
and sigma < 1/2 makes c vary on s = O(n^{2 sigma - 1}) << 1, i.e. moves the inner region. sigma = 1/2 is the
distinguished limit. For the wave map the field u is an angle, so its amplitude is not free: the coupling eta plays the
role of n^{1 - 2 sigma}, and eta fixed as n -> infinity is the distinguished limit (eta -> infinity is the l = 1 multiplet,
u = vphi / sqrt(eta)).

Also checked: the flat (c = 1) inner wave-map equation has the exact CSS solution u = 2 arctan(sqrt(2 s)/(-t)), the
D -> infinity limit of the flat blowup u = 2 arctan(x / sqrt(n - 1)) (itself checked exactly for every n), and the
flat inner Yang-Mills equation W_tt = W_s + W (1 - W^2)/(2 s). Section 7: an exact static (omega = 0) solution of the
real-scalar inner problem, c = (1 - xi^2)^(-1/2), y = sqrt2 xi c, sigma = sqrt2 xi^2 c^2 (a by-product of struct_css.py).
"""
import sympy as sp

ALL = []


def check(label, expr, simp=sp.simplify):
    z = simp(expr)
    ok = (z == 0)
    ALL.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + ("" if ok else f": residual {z}"))


eps = sp.Symbol("epsilon", positive=True)
s, t = sp.symbols("s t", positive=True)
l = sp.Symbol("l", nonnegative=True)
eta = sp.Symbol("eta", positive=True)


def lead(e, k=0):
    """Coefficient of eps^k in the expansion of e about eps = 0."""
    return sp.simplify(sp.series(sp.expand(e), eps, 0, k + 1).removeO().coeff(eps, k))


# Generic inner substitution. Functions of (s, t); r = sqrt(2 n s), d_r = (r / n) d_s.
def setup(field_scale):
    n = 1 / eps
    vp = sp.Function("vphi")(s, t)
    c = sp.Function("c")(s, t)
    beta = sp.Function("beta")(s, t)
    a = sp.sqrt(1 + eps * beta)
    al = c * a
    rr = sp.sqrt(2 * n * s)
    dr = lambda e: (rr / n) * sp.diff(e, s)
    psi = field_scale * vp
    Phi = dr(psi)
    Pi = a * sp.diff(psi, t) / al
    return n, vp, c, beta, a, al, rr, dr, psi, Phi, Pi


def inner_equations(K, V, Vpsi, field_scale):
    """Exact covariant equations (kappa = 2n, K const) in inner variables; K, V, Vpsi callables of (r, psi)."""
    n, vp, c, beta, a, al, rr, dr, psi, Phi, Pi = setup(field_scale)
    b = a ** 2
    rho = K * (Phi ** 2 + Pi ** 2) / (2 * b) + V(rr, psi)
    Pr = K * (Phi ** 2 + Pi ** 2) / (2 * b) - V(rr, psi)
    H = rr * dr(a) / a - (2 * rr ** 2 * b * rho + (n - 1) * (1 - b) / 2)          # r a_r/a equation
    S = rr * dr(al) / al - (2 * rr ** 2 * b * Pr + (n - 1) * (b - 1) / 2)          # r alpha_r/alpha equation
    # d_t(K Pi) = r^-n d_r(r^n c K Phi) - c b V_psi  with  r^-n d_r(r^n X) = (n/r) X + d_r X
    Wq = sp.diff(K * Pi, t) - ((n / rr) * (al / a) * K * Phi + dr((al / a) * K * Phi) - (al / a) * b * Vpsi(rr, psi))
    return H, S, Wq, vp, c, beta


print("1. l-multiplet (K = 1, V = L phi^2/(2 r^2), L = l (l + n - 1)), phi = eps^{1/2} vphi")
Lf = lambda: l * (l + 1 / eps - 1)
H, S, Wq, vp, c, beta = inner_equations(1, lambda rr, ps: Lf() * ps ** 2 / (2 * rr ** 2),
                                        lambda rr, ps: Lf() * ps / rr ** 2, sp.sqrt(eps))
vt, vs = sp.diff(vp, t), sp.diff(vp, s)
Xb = 4 * s * vt ** 2 / c ** 2 + 2 * l * vp ** 2
check("Hamiltonian, order 1:  beta = 4 s vphi_t^2 / c^2 + 2 l vphi^2", lead(H) - (beta - Xb) / 2)
check("slicing - Hamiltonian, order 1 (beta eliminated):  c_s = 2 vphi_t^2 / c",
      lead(S - H).subs(beta, Xb) - 2 * s * (sp.diff(c, s) / c - 2 * vt ** 2 / c ** 2))
check("field equation, order eps^{1/2}:  d_t(vphi_t/c) = c (vphi_s - l vphi/(2 s))",
      lead(sp.expand(Wq / sp.sqrt(eps))) - (sp.diff(vt / c, t) - c * (vs - l * vp / (2 * s))))
print("   (the n-dependence is gone: the D = infinity inner problem is n-free for every fixed l)")

print("\n2. l = 0: the potential form of GAMMA.md's autonomous heat system")
Yv = 2 * sp.sqrt(2) * vt / c
Sv = 2 * sp.sqrt(2) * vs
on_shell = {sp.diff(c, s): 2 * vt ** 2 / c}
# use the field equation (l = 0) to eliminate d_t(vphi_t/c): vphi_tt = c^2 vphi_s + vphi_t c_t / c
vtt = c ** 2 * vs + vt * sp.diff(c, t) / c
sub = lambda e: sp.simplify(e.subs(sp.diff(vp, t, 2), vtt).subs(on_shell).doit())
check("Y_t = c S", sub(sp.diff(Yv, t) - c * Sv))
check("S_t = c (Y_s + Y^3/4)   (identity, given c_s = 2 vphi_t^2/c)",
      (sp.diff(Sv, t) - c * (sp.diff(Yv, s) + Yv ** 3 / 4)).subs(on_shell))
check("c_s = c Y^2 / 4", sp.diff(c, s).subs(on_shell) - c * Yv ** 2 / 4)

print("\n3. wave map (K = eta / n, V = eta sin^2 u / (2 r^2); eta = kappa f^2 / 2 fixed), u = O(1)")
H, S, Wq, u, c, beta = inner_equations(eta * eps, lambda rr, ps: eta * sp.sin(ps) ** 2 / (2 * rr ** 2),
                                       lambda rr, ps: eta * sp.sin(2 * ps) / (2 * rr ** 2), 1)
ut, us = sp.diff(u, t), sp.diff(u, s)
Xb = 4 * eta * s * ut ** 2 / c ** 2 + 2 * eta * sp.sin(u) ** 2
check("Hamiltonian, order 1:  beta = 4 eta s u_t^2/c^2 + 2 eta sin^2 u", lead(H) - (beta - Xb) / 2)
check("slicing - Hamiltonian, order 1 (beta eliminated):  c_s = 2 eta u_t^2 / c",
      lead(S - H).subs(beta, Xb) - 2 * s * (sp.diff(c, s) / c - 2 * eta * ut ** 2 / c ** 2))
check("field equation, order eps:  d_t(u_t/c) = c (u_s - sin(2u)/(4 s))",
      lead(sp.expand(Wq / (eta * eps))) - (sp.diff(ut / c, t) - c * (us - sp.sin(2 * u) / (4 * s))))
print("   eta -> infinity with u = vphi / sqrt(eta): sin(2u)/(4s) -> vphi/(2 s sqrt(eta)), c_s -> 2 vphi_t^2/c: the l = 1 multiplet")
vv = sp.Function("vphi")(s, t)
uu = vv / sp.sqrt(eta)
wm_field = sp.diff(sp.diff(uu, t) / c, t) - c * (sp.diff(uu, s) - sp.sin(2 * uu) / (4 * s))
check("wave map -> l = 1 multiplet as eta -> infinity (field equation)",
      sp.limit(sp.expand(wm_field * sp.sqrt(eta)).subs(eta, 1 / sp.Symbol("h") ** 2), sp.Symbol("h"), 0)
      - (sp.diff(sp.diff(vv, t) / c, t) - c * (sp.diff(vv, s) - vv / (2 * s))))
check("wave map -> l = 1 multiplet as eta -> infinity (c_s)", 2 * eta * sp.diff(uu, t) ** 2 - 2 * sp.diff(vv, t) ** 2)

print("\n4. Yang-Mills U-system (struct_eqs.py, section 5): U = eps^{1/2} V")
n = 1 / eps
V = sp.Function("V")(s, t)
c = sp.Function("c")(s, t)
beta = sp.Function("beta")(s, t)
a = sp.sqrt(1 + eps * beta)
al = c * a
b = a ** 2
rr = sp.sqrt(2 * n * s)
dr = lambda e: (rr / n) * sp.diff(e, s)
U = sp.sqrt(eps) * V
PiU, PhiU = a * sp.diff(U, t) / al, dr(U)
# kappa rho_U, kappa P_r,U from struct_eqs.py section 5
krho = n * ((U + rr * PhiU) ** 2 / rr ** 2 + PiU ** 2) / (2 * b) + n * (n - 1) * U ** 2 / rr ** 2
kPr = n * ((U + rr * PhiU) ** 2 / rr ** 2 + PiU ** 2) / (2 * b) - n * (n - 1) * U ** 2 / rr ** 2
H = rr * dr(a) / a - ((rr ** 2 / n) * b * krho + (n - 1) * (1 - b) / 2)
S = rr * dr(al) / al - ((rr ** 2 / n) * b * kPr + (n - 1) * (b - 1) / 2)
cc = al / a
fieldU = sp.diff(PiU, t) - ((n / rr) * cc * PhiU + dr(cc * PhiU) - cc * b * n * U / rr ** 2) \
    + (-(n - 2) * (b - 1) * cc * U / rr ** 2 + dr(cc) * U / rr)
Vt, Vs = sp.diff(V, t), sp.diff(V, s)
Xb = 2 * s * Vt ** 2 / c ** 2 + 2 * V ** 2
check("Hamiltonian, order 1:  beta = 2 s V_t^2/c^2 + 2 V^2   (l = 1 multiplet at V = sqrt2 vphi: 2 s V_t^2/c^2 + V^2)",
      lead(H) - (beta - Xb) / 2)
check("slicing - Hamiltonian, order 1 (beta eliminated):  c_s = V_t^2 / c   (= 2 vphi_t^2 / c)",
      lead(S - H).subs(beta, Xb) - 2 * s * (sp.diff(c, s) / c - Vt ** 2 / c ** 2))
check("field equation, order eps^{1/2}:  d_t(V_t/c) = c (V_s - V/(2 s))   (the YM-specific terms are O(eps^{3/2}))",
      lead(sp.expand(fieldU / sp.sqrt(eps))) - (sp.diff(Vt / c, t) - c * (Vs - V / (2 * s))))
print("   (the YM-U stress tensor differs from the multiplet's in rho - P_r: 2n(n-1) vs n^2 in struct_eqs.py's")
print("    normalization; in the inner region that difference sits in beta, which first enters the equations at O(eps))")

print("\n5. Flat self-similar wave-map blowup and its inner limit")
x = sp.Symbol("x", positive=True)
nn = sp.Symbol("n", positive=True)
r_, t_ = sp.symbols("r t_", real=True)
u0 = 2 * sp.atan(r_ / (sp.sqrt(nn - 1) * (-t_)))
flat = sp.diff(u0, t_, 2) - (sp.diff(u0, r_, 2) + nn * sp.diff(u0, r_) / r_ - nn * sp.sin(2 * u0) / (2 * r_ ** 2))
check("u0 = 2 arctan(r / (sqrt(n-1)(-t))) solves the flat wave-map equation for every n (d = n + 1 >= 3)",
      sp.simplify(sp.expand_trig(flat)))
ui = 2 * sp.atan(sp.sqrt(2 * s) / (-t_))
inner_flat = sp.diff(ui, t_, 2) - (sp.diff(ui, s) - sp.sin(2 * ui) / (4 * s))
check("u = 2 arctan(sqrt(2 s)/(-t)) solves the flat inner equation u_tt = u_s - sin(2u)/(4 s)",
      sp.simplify(sp.expand_trig(inner_flat)))
print("   (x = sqrt(n) xi: 2 arctan(x/sqrt(n-1)) -> 2 arctan(xi), with xi = rho/(-t), rho = sqrt(2 s): the same solution)")
print("   With gravity the CSS ansatz u = U(xi), c = C(xi) is consistent with the inner wave-map system for every eta")
xi, T = sp.symbols("xi T", positive=True)            # T = -t > 0, d_t = -d_T
Uq, Cq = sp.Function("Uq")(xi), sp.Function("Cq")(xi)
uC, cC = Uq.subs(xi, sp.sqrt(2 * s) / T), Cq.subs(xi, sp.sqrt(2 * s) / T)
dt_ = lambda e: -sp.diff(e, T)
e1 = dt_(dt_(uC) / cC) - cC * (sp.diff(uC, s) - sp.sin(2 * uC) / (4 * s))
e2 = sp.diff(cC, s) - 2 * eta * dt_(uC) ** 2 / cC
# both scale like T^-2 at fixed xi: multiply by T^2, substitute s = xi^2 T^2 / 2, and check that T drops out
e1s = sp.simplify((e1 * T ** 2).subs(s, xi ** 2 * T ** 2 / 2))
e2s = sp.simplify((e2 * T ** 2).subs(s, xi ** 2 * T ** 2 / 2))
check("CSS reduction of the inner wave-map field equation is T-free (an ODE in xi)", sp.diff(e1s, T))
check("CSS reduction of c_s = 2 eta u_t^2 / c is T-free", sp.diff(e2s, T))

print("\n6. Flat Yang-Mills at large n: the inner equation")
W = sp.Function("W")(s, t)
rr = sp.sqrt(2 * s / eps)
dr = lambda e: (rr * eps) * sp.diff(e, s)
flatYM = sp.diff(W, t, 2) - (dr(dr(W)) + (1 / eps - 2) * dr(W) / rr + (1 / eps - 1) * W * (1 - W ** 2) / rr ** 2)
check("W_tt = W_rr + (n-2) W_r/r + (n-1) W(1-W^2)/r^2  ->  W_tt = W_s + W (1 - W^2)/(2 s) + O(eps)",
      lead(flatYM) - (sp.diff(W, t, 2) - sp.diff(W, s) - W * (1 - W ** 2) / (2 * s)))
print("   With gravity, an O(1) field has (kappa/n) r^2 rho_YM ~ n l_YM^2/(4 r^2) = l_YM^2/(8 s): strong for s <~ l_YM^2,")
print("   i.e. this flat inner problem is never reached at small scales (struct_eqs.py, section 5).")

print("\n7. An exact static (omega = 0) solution of the real-scalar inner problem (README form)")
# L y = c sigma, L sigma = c xi y_xi + c (y^2/2 - 1) y, xi c_xi = c y^2 / 2, with L = xi^2 d_xi at omega = 0.
# Found as the omega = 0 zero A = 1/sqrt2 of the CSS Stokes jump (struct_css.py): the inner series converges.
cS = 1 / sp.sqrt(1 - xi ** 2)
yS = sp.sqrt(2) * xi * cS
sS = sp.sqrt(2) * xi ** 2 * cS ** 2
check("c = (1 - xi^2)^(-1/2), y = sqrt2 xi c, sigma = sqrt2 xi^2 c^2:  L y = c sigma", xi ** 2 * sp.diff(yS, xi) - cS * sS)
check("  L sigma = c xi y_xi + c (y^2/2 - 1) y",
      xi ** 2 * sp.diff(sS, xi) - (cS * xi * sp.diff(yS, xi) + cS * (yS ** 2 / 2 - 1) * yS))
check("  xi c_xi = c y^2 / 2", xi * sp.diff(cS, xi) - cS * yS ** 2 / 2)
print("   (regular data y = 2 A xi + ..., A = 1/sqrt2; eta = n(b - 1) = y^2/2 = xi^2/(1 - xi^2) diverges at xi = 1)")

print("\nall ok" if all(ALL) else "\nSOME CHECKS FAILED")
