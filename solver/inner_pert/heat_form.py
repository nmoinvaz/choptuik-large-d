"""The D = infinity inner problem as a nonlinear heat equation (sympy check).

Inner equations (README, "The inner problem at large x"), xi = x/sqrt(n), L = xi^2 d_xi + omega xi d_theta:
    L y = c sigma,   L sigma = c xi y_xi + c (y^2/2 - 1) y,   xi c_xi = c y^2/2.
In physical inner variables rho = r/sqrt(n) and time t (xi = rho/(-t), tau = -ln(-t), theta = omega tau + const) one has
L = rho d_t at fixed rho and xi d_xi = rho d_rho at fixed t. With s = rho^2/2 and
    y = sqrt(s) Y,   sigma = sqrt(2) s S,
the system becomes free of explicit s and t:
    Y_t = c S,   S_t = c (Y_s + Y^3/4),   c_s = c Y^2/4.
This script substitutes and prints the residuals (all zero). Symmetries of the autonomous system: t-translation (the
lambda = 1 mode), s-translation (lambda = 2: d_s at fixed t is e^{2 tau} xi^-2 xi d_xi), the scaling t -> a t, s -> a^2 s,
Y -> Y/a, S -> S/a^2 (self-similarity), and time reparametrization t -> T(t), c -> c T'(t) (gauge).
    python3 heat_form.py
"""
import sympy as sp

s, t = sp.symbols("s t", positive=True)
Y, S, c = (sp.Function(n)(s, t) for n in ("Y", "S", "c"))
rho = sp.sqrt(2 * s)
y = sp.sqrt(s) * Y
sigma = sp.sqrt(2) * s * S
rho_drho = lambda f: 2 * s * sp.diff(f, s)                 # rho d_rho = 2 s d_s
# original equations, written with L = rho d_t and xi d_xi = rho d_rho
E1 = rho * sp.diff(y, t) - c * sigma
E2 = rho * sp.diff(sigma, t) - (c * rho_drho(y) + c * (y ** 2 / 2 - 1) * y)
E3 = rho_drho(c) - c * y ** 2 / 2
# claimed autonomous form
A1 = sp.diff(Y, t) - c * S
A2 = sp.diff(S, t) - c * (sp.diff(Y, s) + Y ** 3 / 4)
A3 = sp.diff(c, s) - c * Y ** 2 / 4
print("E1 - sqrt(2) s A1        :", sp.simplify(E1 - sp.sqrt(2) * s * A1))
print("E2 - 2 s^1.5 A2          :", sp.simplify(E2 - 2 * s ** sp.Rational(3, 2) * A2))
print("E3 - 2 s A3              :", sp.simplify(E3 - 2 * s * A3))
