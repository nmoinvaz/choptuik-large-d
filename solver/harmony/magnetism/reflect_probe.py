import sympy as sp
rho, d, lam, p = sp.symbols('rho d lambda p')
g = sp.Function('g')(rho)
f2 = (-((d-1)/rho - 2*rho)*sp.diff(g, rho) + (d-1)*sp.sin(2*g)/(2*rho**2))/(1 - rho**2)
def res(lmb, pp, dd):
    v = rho*sp.diff(g, rho)*(1 - rho**2)**pp
    e = (1 - rho**2)*sp.diff(v, rho, 2) + ((dd-1)/rho - 2*(lmb+1)*rho)*sp.diff(v, rho) \
        - ((dd-1)*sp.cos(2*g)/rho**2 + lmb*(lmb+1))*v
    F2 = f2.subs(d, dd)
    e = e.subs(sp.Derivative(g, (rho, 3)), sp.diff(F2, rho)).subs(sp.Derivative(g, (rho, 2)), F2)
    e = e.subs(sp.Derivative(g, (rho, 2)), F2)
    return sp.simplify(e/(1 - rho**2)**pp)
for dd in [3, 4, 5, 6]:
    print(dd, 'lambda=d-2,p=-1:', sp.factor(res(dd-2, -1, dd)))
e = res(lam, p, d)
print('general:', sp.factor(sp.together(e)))
