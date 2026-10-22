#!/usr/bin/env python3
"""Leading order of Ecker, Ecker & Grumiller's supercritical branch, in this repository's variables, substituted into the
exact equations (hp/src/model.rs; README "Large D"):

    x c_x = (n-1) c (b-1)                                   (E1)
    x b_x = b (p^2+m^2) - (n-1) b (b-1)                     (E2)
    x (x-c) p_x = -x p_tau + G p - c p + (n/2) c (p+m)      (E3)     G = (n-1) c (b-1),  d/dtau = omega d/dtheta
    x (x+c) m_x = -x m_tau - G m + c m - (n/2) c (p+m)      (E4)

EEG's branch (supplemental material, "More on CSS solutions": beta -> i beta, f0 -> i f0), with eps = 1/n, w = sqrt(beta^2 x^2 - 1),
beta(tau) > 0 and f(tau) > 0 arbitrary (no zeros), on beta^2 x^2 > 1. Through eqmap.py's map (y = p - m = sqrt2 x Pi,
p + m = sqrt2 x^2 Psi, (n-1)(b-1) = x^2 Omega, c = f_EEG):

    p - m = sqrt(2) beta x / w,     c = f w,     (n-1)(b-1) = beta^2 x^2 / w^2,
    p + m = eps * x (x d_x + d_tau)(p - m) / c = -eps sqrt(2) x^2 (beta + beta') / (f w^4).

The last line is the leading order of E3 - E4 (EEG's Eq. (8), Psi_LO = D_1 Pi_LO / f_LO). The script shows symbolically
that the O(1) part of every residual vanishes for arbitrary beta(tau), f(tau), and that the O(eps) part does not (so the
residual is exactly first order), then evaluates the residuals for one non-constant beta, f at n = 10^2..10^4 to show
the 1/n fall-off. Exit status 1 on failure.

    python3 lo_check.py
    python3 lo_check.py --mutate=regular   # c = f sqrt(beta^2 x^2 + 1) (the regular branch's lapse): must FAIL
    python3 lo_check.py --mutate=psisign   # p + m with the opposite sign: must FAIL
"""
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
x, tau, eps = sp.symbols("x tau epsilon", positive=True)
beta = sp.Function("beta")(tau)
f = sp.Function("f")(tau)
r2 = sp.sqrt(2)
w = sp.sqrt(beta ** 2 * x ** 2 - 1)

y0 = r2 * beta * x / w
c0 = f * (sp.sqrt(beta ** 2 * x ** 2 + 1) if mut == "regular" else w)
u0 = beta ** 2 * x ** 2 / w ** 2
s1 = x * (x * sp.diff(y0, x) + sp.diff(y0, tau)) / c0
s1_closed = -r2 * x ** 2 * (beta + sp.diff(beta, tau)) / (f * w ** 4)
if mut != "regular":
    assert sp.simplify(s1 - s1_closed) == 0, "closed form of p + m"
sgn = -1 if mut == "psisign" else 1
n = 1 / eps
sig = sgn * eps * s1
p, m = (sig + y0) / 2, (sig - y0) / 2
b = 1 + u0 / (n - 1)
c = c0
G = (n - 1) * c * (b - 1)
res = {
    "E1": x * sp.diff(c, x) - G,
    "E2": x * sp.diff(b, x) - (b * (p ** 2 + m ** 2) - (n - 1) * b * (b - 1)),
    "E3": x * (x - c) * sp.diff(p, x) + x * sp.diff(p, tau) - (G * p - c * p + n / 2 * c * (p + m)),
    "E4": x * (x + c) * sp.diff(m, x) + x * sp.diff(m, tau) - (-G * m + c * m - n / 2 * c * (p + m)),
}
ok = True
order0 = {}
for k, R in res.items():
    Rc = sp.cancel(sp.together(R.doit()))            # rational in eps: cancel before setting eps = 0
    R0 = sp.simplify(Rc.subs(eps, 0))
    R1 = sp.simplify(sp.diff(Rc, eps).subs(eps, 0))
    order0[k] = R0
    print(f"{k}: O(1) part = {R0}")
    print(f"    O(eps) part is {'zero' if R1 == 0 else 'non-zero'}" + ("" if R1 == 0 else f": {sp.sstr(R1)[:90]}"))
    ok &= R0 == 0
# numerical fall-off for a non-constant profile
prof = {beta: 1.7 + 0.4 * sp.sin(tau), f: 0.8 + 0.3 * sp.cos(2 * tau)}
fns = {k: sp.lambdify((x, tau, eps), R.subs(prof).doit(), "mpmath") for k, R in res.items()}
print("max |residual| over x in {1, 1.5, 3, 8}, tau in {0, 1, 2}, beta = 1.7 + 0.4 sin tau, f = 0.8 + 0.3 cos 2tau:")
for nn in (100, 1000, 10000):
    worst = {}
    for k, fn in fns.items():
        worst[k] = max(abs(fn(xv, tv, 1.0 / nn)) for xv in (1.0, 1.5, 3.0, 8.0) for tv in (0.0, 1.0, 2.0))
    print(f"   n = {nn:6d}: " + "  ".join(f"{k} {float(v):.3e}" for k, v in worst.items())
          + "   (x n: " + " ".join(f"{float(v) * nn:.3f}" for v in worst.values()) + ")")
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
