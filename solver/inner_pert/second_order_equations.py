"""The inner equations to O(1/n^2), derived from the exact self-similar equations (sympy), in the form used by
second_order_jet.py (notes/SECOND_ORDER.md, section 2).

1. The exact equations in inner variables (x = sqrt(n) xi, y = p - m, sigma = sqrt(n)(p + m), b = 1 + u/(n - 1),
   e = 1/n), with the tau-derivative kept as d_tau and L = xi^2 d_xi + xi d_tau, are EXACTLY
        L y        = c sigma + e c (xi sigma_xi - sigma + u sigma)
        L sigma    = (xi d_xi - 1)(c y)                      (given the c equation)
        xi c_xi    = c u
        e xi u_xi  = (1 - e + e u)(P - u),   P = (y^2 + e sigma^2)/2.
   Only the last one is singular in e (the fast mode b ~ x^-n), so u is slaved, u = sum e^k u_k:
        u_0 = E = y^2/2,   u_1 = Psi = sigma^2/2 - xi y y_xi,   u_2 = Phi = -xi Psi_xi - (1 - E) xi E_xi.
   With it, the inner equations to O(e^2) are
        L y     = c sigma - e c (sigma - xi sigma_xi - E sigma) + e^2 c Psi sigma
        L sigma = (xi d_xi - 1)(c y)
        xi c_xi = c (E + e Psi + e^2 Phi).
   The O(e) part is notes/FIRST_ORDER.md section 1 (first_order_equations.py).
2. Checks: (a) the four exact equations are the outer equations (C_selection.md section 1) in inner variables, with no
   expansion; (b) the slaved u to O(e^2) solves the u equation to O(e^3); (c) the O(e^2) system above is the
   exact system expanded to O(nu^4), nu = n^(-1/2), with no odd powers of nu (only integer powers of 1/n, as Lemma
   B5.0 states); (d) the time reparametrization t -> T(t), c -> T' c is an exact symmetry of the exact system (so
   the gauge datum c(xi = 0) drops out of the selection in the full problem at every order).
   Exit status 1 if any check fails; --mutate=phi (the (1 - E) of Phi replaced by 1) must make (b) and (c) fail.

    python3 second_order_equations.py [--mutate=phi]
"""
import sys

import sympy as sp

MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
FAILS = []
xi, nu, tau = sp.symbols("xi nu tau", positive=True)
e = nu ** 2


def verdict(name, ok):
    if not ok:
        FAILS.append(name)
    print(f"   {name}: {'PASS' if ok else 'FAIL'}", flush=True)


y = sp.Function("y")(xi, tau)
s = sp.Function("sigma")(xi, tau)
c = sp.Function("c")(xi, tau)
u = sp.Function("u")(xi, tau)
L = lambda f: xi ** 2 * sp.diff(f, xi) + xi * sp.diff(f, tau)


def outer_in_inner(uu):
    """the outer equations E_c, E_b, E_p - E_m, E_p + E_m in inner variables with (n - 1)(b - 1) = uu"""
    n = nu ** -2
    x = xi / nu
    p = (y + nu * s) / 2
    m = (nu * s - y) / 2
    b = 1 + uu / (n - 1)
    G = (n - 1) * c * (b - 1)
    dx = lambda f: sp.diff(f, xi) * nu
    dt = lambda f: sp.diff(f, tau)
    E_c = x * dx(c) - (n - 1) * c * (b - 1)
    E_b = x * dx(b) - b * (p ** 2 + m ** 2) + (n - 1) * b * (b - 1)
    E_p = x * (x - c) * dx(p) + x * dt(p) - G * p + c * p - n / 2 * c * (p + m)
    E_m = x * (x + c) * dx(m) + x * dt(m) + G * m - c * m + n / 2 * c * (p + m)
    return E_c, E_b, E_p, E_m


def section1():
    print("(a) the exact equations in inner variables", flush=True)
    E_c, E_b, E_p, E_m = outer_in_inner(u)
    P = (y ** 2 + e * s ** 2) / 2
    ours = {
        "c": xi * sp.diff(c, xi) - c * u,
        "u": e * xi * sp.diff(u, xi) - (1 - e + e * u) * (P - u),
        "y": L(y) - (c * s + e * c * (xi * sp.diff(s, xi) - s + u * s)),
        "sigma": L(s) - (xi * sp.diff(c * y, xi) - c * y),
    }
    # E_c is ours exactly; (1 - e) E_b = ours (b = 1 + u/(n - 1)); nu (E_p - E_m) = ours; E_p + E_m = ours + y E_c
    chk = {
        "c": sp.simplify(E_c - ours["c"]) == 0,
        "u": sp.simplify(sp.expand(E_b * (1 - e) - ours["u"])) == 0,
        "y": sp.simplify(sp.expand(nu * (E_p - E_m) - ours["y"])) == 0,
        "sigma": sp.simplify(sp.expand((E_p + E_m) - ours["sigma"] - y * ours["c"])) == 0,
    }
    for k, ok in chk.items():
        verdict(f"{k} equation exact in inner variables", ok)
    return ours


def section2():
    print("(b) the slaved u to O(e^2)", flush=True)
    E = y ** 2 / 2
    Psi = s ** 2 / 2 - xi * y * sp.diff(y, xi)
    one = 1 if MUT == "phi" else (1 - E)
    Phi = -xi * sp.diff(Psi, xi) - one * xi * sp.diff(E, xi)
    us = E + e * Psi + e ** 2 * Phi
    P = (y ** 2 + e * s ** 2) / 2
    res = sp.expand(e * xi * sp.diff(us, xi) - (1 - e + e * us) * (P - us))
    low = [sp.simplify(res.coeff(nu, k)) for k in range(0, 6)]
    verdict("u = E + e Psi + e^2 Phi solves the u equation to O(e^3)", all(v == 0 for v in low))
    return us


def section3(us):
    print("(c) the O(e^2) system = the exact system expanded to O(nu^4), no odd powers", flush=True)
    E = y ** 2 / 2
    Psi = s ** 2 / 2 - xi * y * sp.diff(y, xi)
    Phi = -xi * sp.diff(Psi, xi) - (1 - E) * xi * sp.diff(E, xi)
    target = {
        "y": L(y) - (c * s - e * c * (s - xi * sp.diff(s, xi) - E * s) + e ** 2 * c * Psi * s),
        "c": xi * sp.diff(c, xi) - c * (E + e * Psi + e ** 2 * Phi),
    }
    # the exact equations with the slaved u substituted (u's own equation is then solved to O(e^3))
    E_c, E_b, E_p, E_m = outer_in_inner(us)
    exact = {"y": sp.expand(nu * (E_p - E_m)), "c": sp.expand(E_c)}
    for k in ("y", "c"):
        ex = exact[k]
        tr = sum(ex.coeff(nu, j) * nu ** j for j in range(0, 5))
        verdict(f"{k} equation to O(nu^4) equals the stated form", sp.simplify(sp.expand(tr - target[k])) == 0)
        odd = [j for j in (1, 3) if sp.simplify(ex.coeff(nu, j)) != 0]
        verdict(f"{k} equation: no odd powers of nu through nu^3 (found {odd})", not odd)
    # sigma: exactly (xi d_xi - 1)(c y) given xi c_xi = c u (no e-terms of its own)
    form = L(s) - (xi * sp.diff(c * y, xi) - c * y)
    ex = sp.expand(E_p + E_m)
    verdict("sigma equation = L sigma - (xi d_xi - 1)(c y) + y (xi c_xi - c u), exactly",
            sp.simplify(sp.expand(ex - form - y * (xi * sp.diff(c, xi) - c * us))) == 0)


def section4():
    """in the physical inner variables rho = r/sqrt(n), t (xi = rho/(-t), tau = -ln(-t)): L = rho d_t,
    xi d_xi = rho d_rho; the reparametrization t -> t + a eta(t) acts as f -> f + a eta f_t, c -> c + a (eta' c +
    eta c_t)"""
    print("(d) the time reparametrization is exact for the exact system", flush=True)
    a = sp.symbols("a")
    rho, t = sp.symbols("rho t", positive=True)
    eta = sp.Function("eta")(t)
    y, s, c, u = (sp.Function(k)(rho, t) for k in ("y", "sigma", "c", "u"))
    Lp = lambda f: rho * sp.diff(f, t)
    rdr = lambda f: rho * sp.diff(f, rho)

    def eqs(Y, S_, C, U):
        return [Lp(Y) - (C * S_ + e * C * (rdr(S_) - S_ + U * S_)),
                Lp(S_) - (rdr(C * Y) - C * Y),
                rdr(C) - C * U,
                e * rdr(U) - (1 - e + e * U) * ((Y ** 2 + e * S_ ** 2) / 2 - U)]
    base = eqs(y, s, c, u)
    tr = lambda f: f + a * eta * sp.diff(f, t)
    pert = eqs(tr(y), tr(s), c + a * (sp.diff(eta, t) * c + eta * sp.diff(c, t)), tr(u))
    for k, (name, cw) in enumerate((("y", 1), ("sigma", 1), ("c", 1), ("u", 0))):
        lin = sp.diff(pert[k], a).subs(a, 0)
        cov = eta * sp.diff(base[k], t) + cw * sp.diff(eta, t) * base[k]
        verdict(f"{name} equation covariant under t -> T(t), c -> T' c", sp.simplify(sp.expand(lin - cov)) == 0)


if __name__ == "__main__":
    print(f"mutation: {MUT}" if MUT else "no mutation", flush=True)
    section1()
    us = section2()
    section3(us)
    section4()
    print("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS))
    sys.exit(1 if FAILS else 0)
