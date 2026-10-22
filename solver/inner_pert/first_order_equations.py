"""The O(1/n) inner equations, derived from the exact self-similar equations (sympy), in the forms used by
first_order_jet.py, and the symmetry-breaking term of the s-translation (notes/FIRST_ORDER.md, section 1).

1. Exact equations in inner variables x = sqrt(n) xi, y = p - m, sigma = sqrt(n)(p + m), b = 1 + eta/n, nu = n^(-1/2)
   (as solver/proof/inner_corrections.py), with the tau-derivative kept as d_tau (so that perturbations e^{lam tau} and
   non-self-similar solutions are covered). The b equation is solved for eta order by order (slaved: eta = y^2/2 +
   nu^2 eta1 + ...), and the three remaining equations are expanded to O(nu^2). Result, with E = y^2/2,
   Psi = sigma^2/2 - xi y y_xi, L = xi^2 d_xi + xi d_tau:
        L y      = c sigma - nu^2 c (sigma - xi sigma_xi - E sigma)
        L sigma  = c xi y_xi + (c E + nu^2 c Psi) y - c y
        xi c_xi  = c E + nu^2 c Psi
   and, given the c equation, L sigma = (xi d_xi - 1)(c y) exactly to this order.
2. Physical inner variables rho = r/sqrt(n) = xi (-t), t, s = rho^2/2, y = sqrt(s) Y, sigma = sqrt(2) s S
   (L = rho d_t, xi d_xi = rho d_rho): the heat form
        Y_t = c S + nu^2 c (S + 2 s S_s + s Y^2 S / 2),   S_t = (c Y)_s,   c_s = c (Y^2/4 + nu^2 (s S^2/2 - Y^2/2 - s Y Y_s)).
   The O(1) part has no explicit s (Proposition 5); the O(nu^2) part has, which breaks the s-translation.
3. The breaking term. For any fields (not only solutions), with the s-translation generator
        dy = (rho y_rho - y)/rho^2,  dsigma = (rho sigma_rho - 2 sigma)/rho^2,  dc = c_rho/rho,
   the linearized equations satisfy  DE_k . dZ - (rho d_rho - w_k) E_k / rho^2 = nu^2 R_k  with weights w = (2, 3, 2)
   for (E_y, E_sigma, E_c) and
        R_y = (2/rho^2) c (rho sigma_rho - 2 sigma + E sigma),   R_sigma = (2/rho^2) y c (Psi + y^2),
        R_c = (2/rho^2) c (Psi + y^2),                            Psi = sigma^2/2 - rho y y_rho
   (R_sigma = y R_c: zero in the form L sigma = (xi d_xi - 1)(c y)). The time reparametrization t -> T(t), c -> T' c is
   an exact symmetry of the exact equations (checked here on the outer equations), so it adds no breaking.

    python3 first_order_equations.py
"""
import sympy as sp

xi, th, w, nu, tau = sp.symbols("xi theta omega nu tau", positive=True)


def section1():
    y = sp.Function("y")(xi, tau)
    s = sp.Function("sigma")(xi, tau)
    eta = sp.Function("eta")(xi, tau)
    c = sp.Function("c")(xi, tau)
    n = nu ** -2
    x = xi / nu
    p = (y + nu * s) / 2
    m = (nu * s - y) / 2
    b = 1 + eta * nu ** 2
    G = (n - 1) * c * (b - 1)
    dx = lambda f: sp.diff(f, xi) * nu
    dt = lambda f: sp.diff(f, tau)
    # outer equations (C_selection.md section 1) with -omega x p_theta written as -x p_tau
    E_c = x * dx(c) - (n - 1) * c * (b - 1)
    E_b = x * dx(b) - b * (p ** 2 + m ** 2) + (n - 1) * b * (b - 1)
    E_p = x * (x - c) * dx(p) + x * dt(p) - G * p + c * p - n / 2 * c * (p + m)
    E_m = x * (x + c) * dx(m) + x * dt(m) + G * m - c * m + n / 2 * c * (p + m)
    E_y = sp.expand(nu * (E_p - E_m))          # p - m equation, times nu (leading order nu^0)
    E_s = sp.expand(E_p + E_m)                 # p + m equation
    E_c = sp.expand(E_c)
    E_b = sp.expand(E_b)
    # slaved eta: eta = y^2/2 + nu^2 eta1 + nu^4 eta2
    e1, e2 = sp.Function("e1")(xi, tau), sp.Function("e2")(xi, tau)
    sub = {eta: y ** 2 / 2 + nu ** 2 * e1 + nu ** 4 * e2}
    Eb = sp.expand(E_b.subs(sub).doit())
    Eb0 = sp.simplify(Eb.coeff(nu, 0))
    assert sp.simplify(Eb0) == 0, Eb0
    sol1 = sp.solve(Eb.coeff(nu, 2), e1)[0]
    print("slaved eta = y^2/2 + nu^2 eta1,  eta1 =", sp.simplify(sol1))
    sub1 = {eta: y ** 2 / 2 + nu ** 2 * sol1}
    E = y ** 2 / 2
    Psi = s ** 2 / 2 - xi * y * sp.diff(y, xi)
    L = lambda f: xi ** 2 * sp.diff(f, xi) + xi * sp.diff(f, tau)
    target = {
        "y": L(y) - (c * s - nu ** 2 * c * (s - xi * sp.diff(s, xi) - E * s)),
        "sigma": L(s) - (c * xi * sp.diff(y, xi) + (c * E + nu ** 2 * c * Psi) * y - c * y),
        "c": xi * sp.diff(c, xi) - (c * E + nu ** 2 * c * Psi),
    }
    for name, ex in (("y", E_y), ("sigma", E_s), ("c", E_c)):
        e = sp.expand(ex.subs(sub1).doit())
        trunc = sum(e.coeff(nu, k) * nu ** k for k in range(0, 3))
        # E_y, E_s, E_c are the target equations up to an overall sign / factor at each order
        diff = sp.simplify(sp.expand(trunc - target[name]))
        diff2 = sp.simplify(sp.expand(trunc + target[name]))
        ok = "equal" if diff == 0 else ("equal up to sign" if diff2 == 0 else f"DIFFERENT: {diff}")
        print(f"equation {name}: exact equation to O(nu^2) vs the jet form: {ok}")
        # no odd powers of nu
        odd = [k for k in (1, 3) if sp.simplify(e.coeff(nu, k)) != 0]
        print(f"   odd powers of nu present: {odd}")
    # the sigma equation in the form (xi d_xi - 1)(c y), given the c equation
    cxi = (c * E + nu ** 2 * c * Psi) / xi
    form = sp.expand(xi * (cxi * y + c * sp.diff(y, xi)) - c * y)
    rhs = sp.expand(c * xi * sp.diff(y, xi) + (c * E + nu ** 2 * c * Psi) * y - c * y)
    print("sigma equation = (xi d_xi - 1)(c y) given the c equation:", sp.simplify(form - rhs) == 0)
    return E_c, E_b, E_p, E_m


def section2():
    rho, t, ss = sp.symbols("rho t s", positive=True)
    Y = sp.Function("Y")(ss, t)
    S = sp.Function("S")(ss, t)
    c = sp.Function("c")(ss, t)
    # self-similar-free physical form: L = rho d_t, xi d_xi = rho d_rho = 2 s d_s
    y = sp.sqrt(ss) * Y
    sg = sp.sqrt(2) * ss * S
    rho_ = sp.sqrt(2 * ss)
    rdr = lambda f: 2 * ss * sp.diff(f, ss)
    Lf = lambda f: rho_ * sp.diff(f, t)
    E = y ** 2 / 2
    Psi = sg ** 2 / 2 - y * rdr(y)
    Ey = Lf(y) - (c * sg - nu ** 2 * c * (sg - rdr(sg) - E * sg))
    Es = Lf(sg) - (c * rdr(y) + (c * E + nu ** 2 * c * Psi) * y - c * y)
    Ec = rdr(c) - (c * E + nu ** 2 * c * Psi)
    heat = {
        "Y": sp.diff(Y, t) - (c * S + nu ** 2 * c * (S + 2 * ss * sp.diff(S, ss) + ss * Y ** 2 * S / 2)),
        "S": sp.diff(S, t) - (c * (sp.diff(Y, ss) + Y ** 3 / 4) + nu ** 2 * c * Y * (ss * S ** 2 / 2 - Y ** 2 / 2 - ss * Y * sp.diff(Y, ss))),
        "c": sp.diff(c, ss) - c * (Y ** 2 / 4 + nu ** 2 * (ss * S ** 2 / 2 - Y ** 2 / 2 - ss * Y * sp.diff(Y, ss))),
    }
    for name, ex, fac in (("Y", Ey, sp.sqrt(2) * ss), ("S", Es, 2 * ss ** sp.Rational(3, 2)), ("c", Ec, 2 * ss)):
        d = sp.simplify(sp.expand(ex / fac - heat[name]))
        print(f"heat form, {name} equation: {'ok' if d == 0 else d}")
    # S_t = (c Y)_s given the c equation
    cs = c * (Y ** 2 / 4 + nu ** 2 * (ss * S ** 2 / 2 - Y ** 2 / 2 - ss * Y * sp.diff(Y, ss)))
    d = sp.simplify(sp.expand(c * (sp.diff(Y, ss) + Y ** 3 / 4) + nu ** 2 * c * Y * (ss * S ** 2 / 2 - Y ** 2 / 2 - ss * Y * sp.diff(Y, ss))
                              - (cs * Y + c * sp.diff(Y, ss))))
    print("heat form: S_t = (c Y)_s given the c equation:", d == 0)


def section3():
    rho, t = sp.symbols("rho t", positive=True)
    y = sp.Function("y")(rho, t)
    sg = sp.Function("sigma")(rho, t)
    c = sp.Function("c")(rho, t)
    a = sp.symbols("a")
    Y, Sg, C = sp.Function("Yp")(rho, t), sp.Function("Sp")(rho, t), sp.Function("Cp")(rho, t)

    def eqs(y, sg, c):
        rdr = lambda f: rho * sp.diff(f, rho)
        E = y ** 2 / 2
        Psi = sg ** 2 / 2 - y * rdr(y)
        Ey = rho * sp.diff(y, t) - (c * sg - nu ** 2 * c * (sg - rdr(sg) - E * sg))
        Es = rho * sp.diff(sg, t) - (c * rdr(y) + (c * E + nu ** 2 * c * Psi) * y - c * y)
        Ec = rdr(c) - (c * E + nu ** 2 * c * Psi)
        return [Ey, Es, Ec]
    # s-translation generator
    dy = (rho * sp.diff(y, rho) - y) / rho ** 2
    ds = (rho * sp.diff(sg, rho) - 2 * sg) / rho ** 2
    dc = sp.diff(c, rho) / rho
    base = eqs(y, sg, c)
    pert = eqs(y + a * dy, sg + a * ds, c + a * dc)
    E = y ** 2 / 2
    Psi = sg ** 2 / 2 - rho * y * sp.diff(y, rho)
    R = [2 / rho ** 2 * c * (rho * sp.diff(sg, rho) - 2 * sg + E * sg), 2 / rho ** 2 * y * c * (Psi + y ** 2),
         2 / rho ** 2 * c * (Psi + y ** 2)]
    for k, (name, wgt) in enumerate((("y", 2), ("sigma", 3), ("c", 2))):
        lin = sp.diff(pert[k], a).subs(a, 0)
        cov = (rho * sp.diff(base[k], rho) - wgt * base[k]) / rho ** 2
        d = sp.simplify(sp.expand(lin - cov - nu ** 2 * R[k]))
        print(f"s-translation breaking, {name} equation: DE.dZ - (rho d_rho - {wgt}) E / rho^2 = nu^2 R: {d == 0}")
    # time reparametrization t -> t + a eta(t), c -> c + a (eta' c + eta c_t), fields + a eta f_t: exact symmetry
    eta = sp.Function("eta")(t)
    py = y + a * eta * sp.diff(y, t)
    ps = sg + a * eta * sp.diff(sg, t)
    pc = c + a * (sp.diff(eta, t) * c + eta * sp.diff(c, t))
    pr = eqs(py, ps, pc)
    for k, (name, cw) in enumerate((("y", 1), ("sigma", 1), ("c", 1))):
        lin = sp.diff(pr[k], a).subs(a, 0)
        cov = eta * sp.diff(base[k], t) + cw * sp.diff(eta, t) * base[k]
        d = sp.simplify(sp.expand(lin - cov))
        print(f"time reparametrization, {name} equation: DE.dZ = (eta d_t + {cw} eta') E: {d == 0}")


if __name__ == "__main__":
    print("== 1. O(1/n) inner equations from the exact equations")
    section1()
    print("\n== 2. heat form")
    section2()
    print("\n== 3. symmetry breaking by the O(1/n) terms")
    section3()
