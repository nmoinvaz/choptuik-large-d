"""Symbolic checks for the magnetism/geometric-flow comparison (sympy).

Part A: explicit solutions of the D = infinity inner problem (notes/DINF_ANALYTIC.md 1.1).
  A1 steady soliton of the gauge-fixed form (A): Y = 2k sec(ku), V = k tan(ku), i.e. tanh(phi) = sin(ku).
  A2 uniform expanding solution of Rozali-Way's potential form: phi = a t, c = sqrt(1 + 2 a^2 s).
  A3 singular homogeneous ("equator-like") solution of the inner form: y = sqrt2, sigma = 0, c = kappa xi.
Part B: wave maps R^{d+1} -> S^d, profile f(rho). In d = 5 the reflected gauge mode v = rho f'/(1-rho^2)
  solves the linearized equation at lambda = 3 for every solution f of the profile equation (BBM17's argument);
  it is smooth at rho = 1 iff f'(1) = 0. The same v fails in d = 4 and 6 and at lambda = 3.1.
Each part has a mutation that must make it fail. Exit 1 if any verdict is not as expected.
"""
import sys
import sympy as sp

u, s, t, k, a, kap, xi, th, w = sp.symbols('u s t k a kappa xi theta omega', real=True)
ok = True

def report(name, expr, expect_zero):
    global ok
    z = sp.simplify(expr) == 0
    tag = 'zero' if z else 'NONZERO'
    good = (z == expect_zero)
    ok &= good
    print(f'{name:58s} {tag:8s} {"as expected" if good else "UNEXPECTED"}')

def partA(qY=sp.Rational(1, 4), qc=1, qy=sp.Rational(1, 2)):
    # A1: (A) with s-independence: (V Y)_u = Y_uu, V_u = qY Y^2
    Y = 2*k*sp.sec(k*u); V = k*sp.tan(k*u)
    r1 = sp.diff(V*Y, u) - sp.diff(Y, u, 2)
    r2 = sp.diff(V, u) - qY*Y**2
    # A2: phi_s = c^-1 d_t(c^-1 phi_t), c_s = qc phi_t^2 / c
    phi = a*t; c = sp.sqrt(1 + 2*a**2*s)
    r3 = sp.diff(phi, s) - sp.diff(sp.diff(phi, t)/c, t)/c
    r4 = sp.diff(c, s) - qc*sp.diff(phi, t)**2/c
    # A3: L = xi^2 d_xi + w xi d_th; L y = c sig, L sig = c xi y_xi + c(qy y^2 - 1) y, xi c_xi = c y^2/2
    y = sp.sqrt(2); sig = sp.Integer(0); cc = kap*xi
    L = lambda F: xi**2*sp.diff(F, xi) + w*xi*sp.diff(F, th)
    r5 = L(y) - cc*sig
    r6 = L(sig) - cc*xi*sp.diff(y, xi) - cc*(qy*y**2 - 1)*y
    r7 = xi*sp.diff(cc, xi) - cc*y**2/2
    return [r1, r2], [r3, r4], [r5, r6, r7]

A1, A2, A3 = partA()
report('A1 steady soliton tanh(phi)=sin(ku) in (A)', sum(r**2 for r in A1), True)
report('A2 uniform expanding phi=at, c=sqrt(1+2a^2 s)', sum(r**2 for r in A2), True)
report('A3 homogeneous y=sqrt2, sigma=0, c=kappa*xi', sum(r**2 for r in A3), True)
M1, M2, M3 = partA(qY=sp.Rational(1, 2), qc=2, qy=1)   # mutations: Y^2/2, 2 phi_t^2/c, y^2 in place of y^2/2
report('A1 control: V_u = Y^2/2', sum(r**2 for r in M1), False)
report('A2 control: c_s = 2 phi_t^2/c', sum(r**2 for r in M2), False)
report('A3 control: cubic y^3 in place of y^3/2', sum(r**2 for r in M3), False)

# Part B
rho, d, lam = sp.symbols('rho d lambda', positive=True)
f = sp.Function('f')
F0, F1, F2 = sp.symbols('F0 F1 F2')
# profile ODE: (1-rho^2) f'' + ((d-1)/rho - 2 rho) f' - (d-1) sin(2f)/(2 rho^2) = 0, solved for f''
f2 = (-((d-1)/rho - 2*rho)*F1 + (d-1)*sp.sin(2*F0)/(2*rho**2))/(1 - rho**2)
def lin_residual(lmb, p):
    # v = rho f' (1-rho^2)^p; linearized: (1-rho^2)v'' + ((d-1)/rho - 2(l+1)rho)v' - [(d-1)cos 2f/rho^2 + l(l+1)]v
    fx = sp.Function('g')(rho)
    v = rho*sp.diff(fx, rho)*(1 - rho**2)**p
    expr = (1 - rho**2)*sp.diff(v, rho, 2) + ((d-1)/rho - 2*(lmb+1)*rho)*sp.diff(v, rho) \
        - ((d-1)*sp.cos(2*fx)/rho**2 + lmb*(lmb+1))*v
    # f''' from differentiating the profile ODE
    f3 = sp.diff(f2.subs({F0: fx, F1: sp.diff(fx, rho)}), rho)
    expr = expr.subs(sp.Derivative(fx, (rho, 3)), f3)
    expr = expr.subs(sp.Derivative(fx, (rho, 2)), f2.subs({F0: fx, F1: sp.diff(fx, rho)}))
    expr = expr.subs(sp.Derivative(fx, (rho, 3)), 0)  # none should remain
    return sp.simplify(expr/(1 - rho**2)**p)

def lin_residual_d(lmb, p, dd):
    fx = sp.Function('g')(rho)
    F2 = (-((dd-1)/rho - 2*rho)*sp.diff(fx, rho) + (dd-1)*sp.sin(2*fx)/(2*rho**2))/(1 - rho**2)
    v = rho*sp.diff(fx, rho)*(1 - rho**2)**p
    e = (1 - rho**2)*sp.diff(v, rho, 2) + ((dd-1)/rho - 2*(lmb+1)*rho)*sp.diff(v, rho) \
        - ((dd-1)*sp.cos(2*fx)/rho**2 + lmb*(lmb+1))*v
    e = e.subs(sp.Derivative(fx, (rho, 3)), sp.diff(F2, rho)).subs(sp.Derivative(fx, (rho, 2)), F2)
    e = e.subs(sp.Derivative(fx, (rho, 2)), F2)
    return sp.simplify(e)

report('B gauge mode v = rho f\', lambda = 1 (d = 3..6)', sum(lin_residual_d(1, 0, dd)**2 for dd in (3, 4, 5, 6)), True)
report('B d = 5 reflected gauge mode rho f\'/(1-rho^2), lambda = 3', lin_residual_d(3, -1, 5), True)
report('B control: same v, lambda = 3.1, d = 5', lin_residual_d(sp.Rational(31, 10), -1, 5), False)
report('B control: same construction in d = 4 (lambda = 2)', lin_residual_d(2, -1, 4), False)
report('B control: same construction in d = 6 (lambda = 4)', lin_residual_d(4, -1, 6), False)
print('ALL AS EXPECTED' if ok else 'SOME VERDICT UNEXPECTED')
sys.exit(0 if ok else 1)
