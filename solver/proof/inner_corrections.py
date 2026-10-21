"""The exact M-harmonic (or full) self-similar equations in inner variables: inner problem + corrections, and in which
powers of n the corrections come (B5: is the inner limit reached at rate 1/n or n^(-1/2)?).

Outer equations (proof/C_selection.md section 1), with tau-derivatives already written as omega d/dtheta:
    x c_x = (n-1) c (b-1),   x b_x = b (p^2+m^2) - (n-1) b (b-1),
    x (x-c) p_x = -omega x p_th + G p - c p + n/2 c (p+m),
    x (x+c) m_x = -omega x m_th - G m + c m - n/2 c (p+m),      G = (n-1) c (b-1).
Inner variables (README, "The inner problem at large x"): x = sqrt(n) xi, y = p - m, sigma = sqrt(n) (p+m),
b = 1 + eta/n. With nu = n^(-1/2) the equations are expanded in nu; the script prints the leading equations and checks
which powers of nu occur in the corrections.

    python3 inner_corrections.py
"""
import sympy as sp

xi, th, w, nu = sp.symbols("xi theta omega nu", positive=True)
y = sp.Function("y")(xi, th)
s = sp.Function("sigma")(xi, th)
eta = sp.Function("eta")(xi, th)
c = sp.Function("c")(xi, th)
n = nu ** -2
x = xi / nu
p = (y + nu * s) / 2
m = (nu * s - y) / 2
b = 1 + eta * nu ** 2
G = (n - 1) * c * (b - 1)
dx = lambda f: sp.diff(f, xi) * nu          # d/dx = nu d/dxi
dth = lambda f: sp.diff(f, th)
E_c = x * dx(c) - (n - 1) * c * (b - 1)
E_b = x * dx(b) - b * (p ** 2 + m ** 2) + (n - 1) * b * (b - 1)
E_p = x * (x - c) * dx(p) + w * x * dth(p) - G * p + c * p - n / 2 * c * (p + m)
E_m = x * (x + c) * dx(m) + w * x * dth(m) + G * m - c * m + n / 2 * c * (p + m)
E_diff = sp.expand((E_p - E_m))           # carries y
E_sum = sp.expand((E_p + E_m))            # carries sigma


def series(e, lead_pow):
    e = sp.expand(e * nu ** (-lead_pow))
    terms = sp.collect(e, nu, evaluate=False)
    return {k: sp.simplify(v) for k, v in terms.items()}


for name, e in (("c", E_c), ("b", E_b), ("p - m", E_diff), ("p + m", E_sum)):
    pw = sorted({sp.Poly(t, nu).degree() if t.has(nu) else 0 for t in sp.Add.make_args(sp.expand(e * nu ** 6))})
    lo = min(pw) - 6
    ser = series(e, lo)
    print(f"equation {name}: lowest power nu^{lo}")
    for k in sorted(ser, key=lambda t: sp.degree(t, nu) if t != 1 else 0):
        deg = sp.degree(k, nu) if k != 1 else 0
        print(f"   nu^{lo + deg} (relative nu^{deg}): {ser[k]}")


# Lemma B5.1: the outer fast mode in inner variables. For x > 1, Phi = w +- i pi/2 with w = -1/2 ln(1 - x^-2); the
# Proposition 3 amplitudes give p - m and sqrt(n)(p + m) factors -2x^2/(x^2-1) and -2 sqrt(n) x/(x^2-1); the gravity
# factor contains (x^2 - 1)^<F^2> = n^<F^2> u^<F^2> (1 - 1/(n u))^<F^2>. With x^2 = n u all of these are series in 1/n.
uu, nn, F2 = sp.symbols("u n F2", positive=True)
X2 = nn * uu
print("\nLemma B5.1 (x^2 = n u):")
print("  n w           =", sp.series(-nn / 2 * sp.log(1 - 1 / X2), nn, sp.oo, 4))
print("  y-factor      =", sp.series(-2 * X2 / (X2 - 1), nn, sp.oo, 3))
print("  sigma-factor  =", sp.series(-2 * nn * sp.sqrt(uu) / (X2 - 1), nn, sp.oo, 3))
print("  (x^2-1)^<F^2> = n^<F^2> u^<F^2> *", sp.series((1 - 1 / X2) ** F2, nn, sp.oo, 3))
