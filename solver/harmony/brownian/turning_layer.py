"""Leading-order turning-point layer of the scaled SC02 equation, and the Hermite quantization.
eps (s'' + s'/xi + xi s s') + s'(1/xi - xi) + s^2 - alpha s = 0, xi = 1 + eta sqrt(eps).
--mutate=sign flips the drift sign (the layer equation must then differ from Hermite's)."""
import sys, sympy as sp
eta, e, al = sp.symbols('eta epsilon alpha', positive=True)
s = sp.Function('s')
xi = 1 + eta*sp.sqrt(e)
sgn = -1 if '--mutate=sign' in sys.argv else 1
d = lambda f: sp.diff(f, eta)/sp.sqrt(e)
S = s(eta)
expr = e*(d(d(S)) + d(S)/xi + xi*S*d(S)) + sgn*d(S)*(1/xi - xi) + S**2 - al*S
lead = sp.simplify(sp.limit(expr, e, 0))
print('layer equation:', lead, '= 0')
# linearize about the reaction's fixed point s = alpha: s = alpha - delta
dl = sp.Function('delta')
lin = sp.expand(lead.subs(s(eta), al - dl(eta)).doit())
lin = sp.simplify(lin.subs(dl(eta)**2, 0))
herm = -(sp.diff(dl(eta), eta, 2) - 2*eta*sp.diff(dl(eta), eta) + al*dl(eta))
ok = sp.simplify(sp.expand(lin) - sp.expand(herm) - dl(eta)**2) == 0 or sp.simplify(lin - herm) == 0
print('linearized about s = alpha: Hermite  delta\'\' - 2 eta delta\' + 2 nu delta = 0 with nu = alpha/2:', ok)
sys.exit(0 if ok else 1)
