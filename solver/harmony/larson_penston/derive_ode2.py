import sympy as sp
y = sp.symbols('y', positive=True)
R, w, Rp, wp = sp.symbols('R w Rp wp')
U = y * (w - 1); Up = (w - 1) + y * wp
cont = y * Rp + R * Up + 2 * R + U * Rp + 2 * R * U / y
mom = 2 * R * (y + U) + y * Up + U * Up + Rp / R          # 4 pi beta = 2, M = y^2 R (y + U)
sol = sp.solve([cont, mom], [Rp, wp], dict=True)[0]
B = 1 - y**2 * w**2
ghj_R = -2 * y * w * R * (R - w) / B
ghj_w = (1 - 3 * w) / y + 2 * y * w**2 * (R - w) / B
print('R  difference:', sp.simplify(sol[Rp] - ghj_R))
print('w  difference:', sp.simplify(sol[wp] - ghj_w))
# control: a wrong gravity normalisation (4 pi beta = 2.2) must give a nonzero difference
mom_bad = 2.2 * R * (y + U) + y * Up + U * Up + Rp / R
sb = sp.solve([cont, mom_bad], [Rp, wp], dict=True)[0]
print('control (4 pi beta = 2.2) R diff nonzero:', sp.simplify(sb[Rp] - ghj_R) != 0)
