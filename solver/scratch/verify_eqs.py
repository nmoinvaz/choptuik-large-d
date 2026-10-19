"""Check that the reformulated large-D system follows exactly from the DSS equations.
Original (n = D-2, tau = -ln(-t), x = r/(-t), c = alpha/a, p = r(Phi+Pi), m = r(Phi-Pi)):
  a_x = a[(p^2+m^2) + (n-1)(1-a^2)]/(2x),  c_x = c(n-1)(a^2-1)/x
  p_tau + (x-c) p_x =  c_x p - c p/x + n c (p+m)/(2x)
  m_tau + (x+c) m_x = -c_x m + c m/x - n c (p+m)/(2x)
New variables: q=(p-m)/2, s=n(p+m)/2, A=n(a^2-1), Q=c q/x, eps=1/n."""
import sympy as sp
t, x, n = sp.symbols('tau x n', positive=True)
p, m, a, c = [sp.Function(f)(t, x) for f in 'pmac']
eps = 1 / n
cx = c * (n - 1) * (a**2 - 1) / x
ax = a * ((p**2 + m**2) + (n - 1) * (1 - a**2)) / (2 * x)
pt = -(x - c) * p.diff(x) + cx * p - c * p / x + n * c * (p + m) / (2 * x)
mt = -(x + c) * m.diff(x) - cx * m + c * m / x - n * c * (p + m) / (2 * x)
sub = {c.diff(x): cx, a.diff(x): ax, p.diff(t): pt, m.diff(t): mt}
q = (p - m) / 2; s = n * (p + m) / 2; A = n * (a**2 - 1); Q = c * q / x
D = lambda e, v: sp.diff(e, v).subs(sub)
E1 = x * D(Q, x) - eps * (D(s, t) + x * D(s, x))
E2 = D(q, t) + x * D(q, x) - (c * s / x + eps * (c * D(s, x) + cx * s - c * s / x))
E3 = cx - c * (1 - eps) * A / x
E4 = eps * D(A, x) - (1 + eps * A) * (2 * q**2 + 2 * eps**2 * s**2 - (1 - eps) * A) / x
for name, e in (("E1", E1), ("E2", E2), ("E3", E3), ("E4", E4)):
    print(name, "residual simplifies to:", sp.simplify(sp.expand(e)))
