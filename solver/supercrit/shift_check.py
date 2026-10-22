#!/usr/bin/env python3
"""The leading order of both large-D branches is one family, closed under the r^2-shift of Proposition 5.

1. With C(tau) real and k(tau) > 0, y = sqrt2 x/sqrt(x^2 + C), c = k sqrt(x^2 + C), (n-1)(b-1) = x^2/(x^2 + C) solves the
   leading-order equations (x y_x = (1 - y^2/2) y, x c_x = c y^2/2, u = y^2/2). C = 1/beta^2 is EEG's regular branch
   (k = f0 beta), C = -1/beta^2 the supercritical one (k = f beta).
2. In r = x e^{-tau}, t = -e^{-tau}: y and b depend only on r and rho(t) = C t^2, and c = K(t) sqrt(r^2 + rho) with
   K = k/(-t); a time reparametrization t -> T(t) (alpha -> alpha/T') changes K only, so f (or k) is gauge and the
   physical datum is rho(t) (the singular sphere r_s = sqrt(-rho) when rho < 0).
3. In the heat variables of Proposition 5 (s = r^2/(2n), y = sqrt(s) Y) the family is Y = sqrt2/sqrt(s + s_c(t)),
   s_c = rho/(2n), and solves Y_s = -Y^3/4, c_s = c Y^2/4 (the inner equations without t-derivatives). The
   s-translation s -> s + sigma0 maps s_c -> s_c + sigma0, i.e. C(tau) -> C(tau) + 2 n sigma0 e^{2 tau}: the lambda = 2
   direction. Its generator at sigma0 -> 0 equals Proposition 5's delta y = e^{2tau} (x y_x - y)/x^2 (in x = sqrt(n) xi).
4. A finite negative shift of a regular CSS member (C0 > 0): r_s(t)^2 = 2n|sigma0| - C0 t^2; the front is born at r = 0
   when C = 0 (there y = sqrt2 exactly: the stationary solution of y1_modes.py) and decelerates to
   r_BH = sqrt(2n|sigma0|) as t -> 0^-; the layer parameter kappa~ = r_s/|dr_s/dT| (dT = K dt) grows without bound.

    python3 shift_check.py
    python3 shift_check.py --mutate=rate    # C -> C + 2 n sigma0 e^{tau} (rate 1): must FAIL item 3
"""
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
x, tau = sp.symbols("x tau", real=True)
n, s, r, T = sp.symbols("n s r T", positive=True)
t = -T                                          # t < 0 (past patch)
C = sp.Function("C")(tau)
k = sp.Function("k")(tau)
ok = True

# 1. the family solves the leading-order equations, either sign of C
y = sp.sqrt(2) * x / sp.sqrt(x ** 2 + C)
c = k * sp.sqrt(x ** 2 + C)
u = x ** 2 / (x ** 2 + C)
r1 = [sp.simplify(x * sp.diff(y, x) - (1 - y ** 2 / 2) * y), sp.simplify(x * sp.diff(c, x) - c * y ** 2 / 2),
      sp.simplify(u - y ** 2 / 2)]
print("1. LO residuals (x y_x - (1 - y^2/2) y, x c_x - c u, u - y^2/2):", r1)
ok &= all(v == 0 for v in r1)
bt = sp.Symbol("beta", positive=True)
reg = sp.simplify(y.subs(C, 1 / bt ** 2) - sp.sqrt(2) * bt * x / sp.sqrt(1 + bt ** 2 * x ** 2))
sup = sp.simplify((y.subs(C, -1 / bt ** 2) - sp.sqrt(2) * bt * x / sp.sqrt(bt ** 2 * x ** 2 - 1)).subs(x, 2 / bt))
print("   C = +1/beta^2 is EEG's regular Pi_LO (difference):", reg, "; C = -1/beta^2 the supercritical one (at x = 2/beta):", sup)
ok &= reg == 0 and sup == 0

# 2. physical form: y(r, rho), c = K sqrt(r^2 + rho)
rho = sp.Symbol("rho", real=True)
K = sp.Symbol("K", positive=True)
xr = r / (-t)                                   # x = r/(-t), e^{tau} = 1/(-t)
q = sp.Symbol("q", positive=True)               # q = r^2 + rho > 0 on the domain
yphys = sp.simplify(y.subs(x, xr).subs(C, rho / t ** 2).subs(rho, q - r ** 2))
print("2. y in (r, t) with C = rho/t^2, q = r^2 + rho:", yphys, "(depends on r, rho only)")
ok &= yphys.free_symbols <= {r, q}
cphys = sp.simplify((c.subs(x, xr).subs(C, rho / t ** 2)).subs(k, K * (-t)).subs(rho, q - r ** 2) / sp.sqrt(q))
print("   c / sqrt(r^2 + rho) with k = K (-t):", cphys)

# 3. heat variables and the s-shift
sc = sp.Function("s_c")(T)
Yh = sp.sqrt(2) / sp.sqrt(s + sc)
ch = sp.Function("K")(T) * sp.sqrt(2 * n * s + 2 * n * sc)
h1 = sp.simplify(sp.diff(Yh, s) + Yh ** 3 / 4)
h2 = sp.simplify(sp.diff(ch, s) - ch * Yh ** 2 / 4)
q2 = sp.Symbol("q2", positive=True)             # q2 = s + s_c > 0
yh = sp.simplify((sp.sqrt(s) * Yh - sp.sqrt(2) * sp.sqrt(2 * n * s) / sp.sqrt(2 * n * s + 2 * n * sc)).subs(sc, q2 - s))
print("3. heat form: Y_s + Y^3/4 =", h1, "; c_s - c Y^2/4 =", h2, "; y = sqrt(s) Y matches (r^2 = 2ns, rho = 2n s_c):", yh)
ok &= h1 == 0 and h2 == 0 and yh == 0
sig0 = sp.Symbol("sigma0", real=True)
shifted = Yh.subs(s, s + sig0)
same = sp.simplify(shifted - Yh.subs(sc, sc + sig0))
print("   Y(s + sigma0) - Y with s_c -> s_c + sigma0:", same)
ok &= same == 0
# in tau: s_c = rho/(2n) = C t^2/(2n) = C e^{-2 tau}/(2n), so C -> C + 2n sigma0 e^{2 tau}
rate = 1 if mut == "rate" else 2
dC = 2 * n * sig0 * sp.exp(rate * tau)
gen = sp.diff(y.subs(C, C + dC), sig0).subs(sig0, 0)
prop5 = sp.exp(2 * tau) * (x * sp.diff(y, x) - y) / x ** 2 * n      # Prop. 5's s-translation part, xi = x/sqrt(n)
g = sp.simplify(gen - prop5)
print("   generator d/dsigma0 of the shifted family - Prop. 5's e^{2tau}(x y_x - y)/x^2 (times n):", g)
ok &= g == 0

# 4. a finite negative shift of a regular CSS member
C0, S0 = sp.symbols("C0 S0", positive=True)       # sigma0 = -S0
rs2 = sp.simplify(-(C0 - 2 * n * S0 * sp.exp(2 * tau)) * sp.exp(-2 * tau))
print("4. r_s^2 = -C t^2 =", rs2, "-> 2 n S0 as tau -> oo; zero at tau* = ln(C0/(2 n S0))/2:",
      sp.simplify(rs2.subs(tau, sp.log(C0 / (2 * n * S0)) / 2)))
ok &= sp.simplify(rs2.subs(tau, sp.log(C0 / (2 * n * S0)) / 2)) == 0
kk = sp.Symbol("k0", positive=True)                # k constant (the regular CSS member's lapse normalization)
rs = sp.sqrt(rs2)
# dT = K dt = k/(-t) dt = k dtau; kappa~ = r_s / |dr_s/dT|
kt = sp.simplify(kk * rs / sp.diff(rs, tau))
print("   kappa~ = r_s/|dr_s/dT| =", sp.factor(kt), "-> grows like e^{2 tau}")
ok &= sp.simplify(kt - kk * (2 * n * S0 * sp.exp(2 * tau) / C0 - 1)) == 0
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
