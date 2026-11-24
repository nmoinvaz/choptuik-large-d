"""Symbolic checks of the large-D structure of the DSS equations (#1086, notes/LARGE_D_BASICS.md).

The equations are those of hp/src/model.rs (n = D - 2, d/dtau = omega d/dtheta, horizon at x = 1 where c = 1):

  E1 = x c_x - (n-1) c (a^2 - 1)
  E2 = 2 x a_x - a (p^2 + m^2) - (n-1) a (1 - a^2)
  E3 = x (x - c) p_x + omega x p_th - G p + c p - (n/2) c (p + m)
  E4 = x (x + c) m_x + omega x m_th + G m - c m + (n/2) c (p + m),     G = (n-1) c (a^2 - 1)

Checks, each printed with PASS/FAIL (exit status 1 on any FAIL):
  1. centre indicial exponents: (p, m) ~ x^s gives s = 1 (p = -m) and s = 1 - n (p = m); a - 1 ~ x^s gives s = 1 - n
  2. SSH indicial exponent of p (theta-independent g = c_x(1)): sigma = (n/2 - 1 + g)/(1 - g)
  3. leading-order slow solution: v = p - m = sqrt(2K) x / sqrt(1 + K x^2), c = sqrt((1 + K x^2)/(1 + K)),
     p + m = O(1/n), (n-1)(a^2 - 1) = v^2/2 + O(1/n), for any K(theta) and any omega: the four residuals divided by
     their O(n) scale vanish as n -> infinity
  4. fast-mode eikonal: with (p, m) ~ exp(n S), (x^2 S_x + omega x S_th)^2 = x c S_x (x c S_x + c); for S_th = 0 the
     nonzero root is S_x = c^2 / (x (x^2 - c^2)), and on the slow solution its integral is
     -ln x + ((1 + K)/2) ln(1 - x^2)
  5. free field (a = c = 1): phi = e^{lambda tau} f(x) with f = 2F1(lambda/2, (lambda+1)/2; (n+1)/2; x^2) solves the
     flat wave equation (numerically, mpmath, at two points)

Usage: python3 derive.py [--mutate=N]   (N = 1..5 breaks check N on purpose; that check must FAIL)
"""
import sys

import mpmath as mp
import sympy as sp

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MUT = int(opts.get("mutate", 0))
results = []


def report(k, name, ok):
    results.append(ok)
    print("%d. %-60s %s" % (k, name, "PASS" if ok else "FAIL"))


x, n, w, th, s, g, u, sig = sp.symbols("x n omega theta s g u sigma")
K = sp.Function("K")(th)

# 1. centre exponents. Near x = 0 with c -> c0, a -> 1, G -> 0, dropping O(x) terms:
#    E3: -c x p_x - c p ... careful: x(x - c) p_x ~ -c x p_x
c0 = sp.symbols("c0", positive=True)
P, Mm = sp.symbols("P Mm")
e3 = -c0 * s * P + c0 * P - (n / 2) * c0 * (P + Mm)  # x^s coefficient of E3 at leading order
e4 = c0 * s * Mm - c0 * Mm + (n / 2) * c0 * (P + Mm)
Amat = sp.Matrix([[sp.diff(e3, P), sp.diff(e3, Mm)], [sp.diff(e4, P), sp.diff(e4, Mm)]])
roots = sp.solve(sp.factor(Amat.det()), s)
if MUT == 1:
    roots = [r + 1 for r in roots]
# a = 1 + alpha x^s: E2 linear in alpha: 2 s alpha - (n-1)(1 - 3) alpha ... (1 - a^2) a ~ -2 alpha
sa = sp.solve(2 * s + 2 * (n - 1), s)
ok1 = set(sp.simplify(r) for r in roots) == {sp.Integer(1), 1 - n} and sp.simplify(sa[0] - (1 - n)) == 0
report(1, "centre exponents s = 1, 1 - n (p, m) and 1 - n (a)", ok1)

# 2. SSH exponent: x = 1 - u, c = 1 - g u + O(u^2), p = u^sigma q, theta-independent
cx = 1 - g * u
xx = 1 - u
# x (x - c) p_x with p_x = -dp/du: x (x - c) = (1 - u)(-u + g u) = -u (1 - g) (1 - u)
lhs = sp.expand(xx * (xx - cx) * (-sig / u))  # times p = u^sigma q
# E3 / p at u -> 0 (omega p_th = 0 for the theta-independent exponent, G(1) = (n-1)(a_h^2-1) c = g)
e3s = sp.limit(lhs, u, 0) - g + 1 - n / 2
sig_sol = sp.solve(e3s, sig)[0]
target = (n / 2 - 1 + g) / (1 - g)
if MUT == 2:
    target = (n / 2 + g) / (1 - g)
report(2, "SSH exponent sigma = (n/2 - 1 + g)/(1 - g)", sp.simplify(sig_sol - target) == 0)

# 3. leading-order slow solution
v = sp.sqrt(2 * K) * x / sp.sqrt(1 + K * x ** 2)
c = sp.sqrt((1 + K * x ** 2) / (1 + K))
if MUT == 3:
    c = sp.sqrt((1 + 2 * K * x ** 2) / (1 + 2 * K))
wsum = (x ** 2 * sp.diff(v, x) + w * x * sp.diff(v, th)) / (n * c)  # p + m at O(1/n)
p = (v + wsum) / 2
m = (wsum - v) / 2
a2 = 1 + (v ** 2 / 2) / (n - 1)
a = sp.sqrt(a2)
G = (n - 1) * c * (a2 - 1)
E1 = x * sp.diff(c, x) - (n - 1) * c * (a2 - 1)
E2 = 2 * x * sp.diff(a, x) - a * (p ** 2 + m ** 2) - (n - 1) * a * (1 - a2)
E3 = x * (x - c) * sp.diff(p, x) + w * x * sp.diff(p, th) - G * p + c * p - (n / 2) * c * (p + m)
E4 = x * (x + c) * sp.diff(m, x) + w * x * sp.diff(m, th) + G * m - c * m + (n / 2) * c * (p + m)
ok3 = True
Kv = 0.3 + 0.2 * sp.cos(2 * th)  # a sample K(theta) with theta dependence, evaluated numerically
for E, scale in ((E1, 1), (E2, 1), (E3, 1), (E4, 1)):
    Ef = E.subs(K, Kv).doit()
    vals = []
    for nn in (1e3, 1e5):
        val = Ef.subs({n: nn, w: 3.7, x: 0.6, th: 0.4}).evalf(30)
        vals.append(abs(complex(val)))
    # each residual's O(n) and O(1) parts cancel: the residual falls at least like 1/n
    ok3 = ok3 and vals[1] < 0.03 * vals[0] + 1e-20
    print("   residual at n = 1e3, 1e5: %.3e %.3e" % tuple(vals))
report(3, "slow solution: residuals O(1/n) for any K(theta), omega", ok3)

# 4. eikonal and the fast exponent on the slow solution
Sx, Sth, cc = sp.symbols("S_x S_th c", positive=True)
T = x ** 2 * Sx + w * x * Sth
Q = x * cc * Sx
# u = p + m, v = p - m with (u, v) ~ exp(n S): T u = Q v, T v = (Q + c) u
eik = sp.expand(T ** 2 - Q * (Q + cc))
roots4 = sp.solve(eik.subs(Sth, 0), Sx)
lam = cc ** 2 / (x * (x ** 2 - cc ** 2))
ok4 = any(sp.simplify(r - lam) == 0 for r in roots4)
Kc = sp.symbols("Kc", positive=True)
lam_slow = sp.simplify(lam.subs(cc, sp.sqrt((1 + Kc * x ** 2) / (1 + Kc))))
prim = -sp.log(x) + (1 + Kc) / 2 * sp.log(1 - x ** 2)
if MUT == 4:
    prim = -sp.log(x) + Kc / 2 * sp.log(1 - x ** 2)
ok4 = ok4 and sp.simplify(sp.diff(prim, x) - lam_slow) == 0
report(4, "eikonal root c^2/(x(x^2-c^2)); integral -ln x + (1+K)/2 ln(1-x^2)", ok4)

# 5. free field: (1 - x^2) f'' + (n/x - 2 (lambda + 1) x) f' - lambda (lambda + 1) f = 0
mp.mp.dps = 40
ok5 = True
for nn, lamb in ((40, mp.mpc(0, "3.7")), (258, mp.mpc(0, "7.4"))):
    cpar = mp.mpf(nn + 1) / 2 + (mp.mpf("0.5") if MUT == 5 else 0)
    f = lambda xx: mp.hyp2f1(lamb / 2, (lamb + 1) / 2, cpar, xx ** 2)
    for xx in (mp.mpf("0.3"), mp.mpf("0.7")):
        r = (1 - xx ** 2) * mp.diff(f, xx, 2) + (nn / xx - 2 * (lamb + 1) * xx) * mp.diff(f, xx) \
            - lamb * (lamb + 1) * f(xx)
        ok5 = ok5 and abs(r) < mp.mpf(10) ** -25 * max(1, abs(f(xx)))
report(5, "free field: 2F1(l/2, (l+1)/2; (n+1)/2; x^2) solves the wave eq.", ok5)

sys.exit(0 if all(results) else 1)
