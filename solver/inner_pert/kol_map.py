"""Kol's correspondence at D = infinity: Emparan-Suzuki's U(1)-symmetric Ricci flow onto Proposition 5(i) (#768).

Emparan and Suzuki [ES19] reduce the large-D merger to the flow (1/rho) d_rho g_ab = 2 R_ab of a 2-metric (their
Eq. (3.20); Eqs. (3.2)-(3.3) with lambda = lambda_0 - rho^2/2), written for static configurations as
g = V(rho, xi) (-dt^2 + dxi^2) (Eq. (4.1)), where it becomes -(1/rho) V_rho = (ln V)_xixi (Eq. (4.2)). Kol's double
analytic continuation [K06, Eq. (3.7)] sends the tangential coordinate to i times the Choptuik time and the static
time to i times a new circle; his reduction over that circle (Eqs. (3.1)-(3.4)) gives the scalar A = (1/2) ln g_circle
and a Weyl factor e^{2A/(d-2)} that tends to 1 at d = infinity. Equation numbers are those of the arXiv sources
(1905.01062v3, hep-th/0502033v3, the latest versions), counted from their TeX.
Here: xi = i x, t = i z, sigma = rho^2/2. The checks (each prints PASS or FAIL; the exit code is 1 if any fails):
  1. the Ricci tensor of V(-dt^2 + dxi^2) turns (3.20) into +-ES's (4.2) in its two components;
  2. after xi = i x, t = i z the 2-metric is V^(-dx^2 + dz^2), V^(sigma, x) = V(rho, i x), and (3.20) becomes the
     forward logarithmic diffusion V^_sigma = (ln V^)_xx, the same equation as (4.2) with d_xi = -i d_x;
  3. the Weyl factor (V/n)^(1/(d-2)) -> 1 at fixed V (d = n + 2 after the reduction);
  4. the map: phi = (1/2) ln V^ on the base x-line, the gauge x = X(s, t) with X_s = -grad phi = -phi_x / V^, and
     c^2 = V^(s, X) X_t^2. The residuals of form L (DINF_ANALYTIC.md 1.1), phi_s - c^-1 (c^-1 phi_t)_t and
     (c^2)_s - 2 alpha phi_t^2 with alpha = 1, vanish identically on every solution of the diffusion (jet space);
  5. the same through Prop. 5(i) itself, Y = 2 phi_t / c, W = Y_t / c: W_t - c (Y_s + Y^3/4) and c_s - c Y^2/4 vanish;
  6. exact solutions: ES's cone (4.7) and King-Rosenau (4.22) solve (4.2), stay real under xi -> i x and solve the
     continued diffusion; the cone's gauge ODE integrates in closed form and gives the form-L solution
     phi = (1/2) ln(2s - mu + t^2), c^2 = (2s - mu)/(2s - mu + t^2), checked directly in form L and in Prop. 5(i).
     ES's single-black-hole solution (4.9), not even in xi, has no real continuation (reported, not a failure).
Mutations (each must make the script exit 1):
  --mutate=alpha2   the scalar term doubled: (c^2)_s - 4 phi_t^2 (List's normalization without phi = sqrt2 psi);
  --mutate=nocont   no continuation: ES's (4.2) itself (backward diffusion) used in place of the continued one;
  --mutate=gauge2   the gauge velocity doubled, X_s = -2 phi_x / V^.
References.
  [K06]  B. Kol, Choptuik scaling and the merger transition, JHEP 10 (2006) 017, hep-th/0502033.
  [ES19] R. Emparan, R. Suzuki, Topology-changing horizons at large D as Ricci flows, JHEP 07 (2019) 094,
         arXiv:1905.01062.
    python3 kol_map.py [--mutate=NAME]
"""
import sys

import sympy as sp

MUT = None
for arg in sys.argv[1:]:
    if arg.startswith("--mutate="):
        MUT = arg.split("=", 1)[1]
        if MUT not in ("alpha2", "nocont", "gauge2"):
            sys.exit(f"unknown mutation {MUT}")
FAILS = 0


def report(name, ok, detail=""):
    global FAILS
    FAILS += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)


def ricci_2d(g, coords):
    """Ricci tensor of a 2-metric (sympy Matrix) in coordinates coords"""
    n = 2
    gi = g.inv()
    Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], coords[c]) + sp.diff(g[d, c], coords[b]) - sp.diff(g[b, c], coords[d]))
                 for d in range(n)) / 2 for c in range(n)] for b in range(n)] for a in range(n)]
    Ric = sp.zeros(n, n)
    for b in range(n):
        for c in range(n):
            Ric[b, c] = sp.simplify(sum(sp.diff(Gam[a][b][c], coords[a]) - sp.diff(Gam[a][b][a], coords[c])
                                        + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
                                              for d in range(n)) for a in range(n)))
    return Ric


rho, xi, t, x, z, s, sig, mu, n = sp.symbols("rho xi t x z s sigma mu n", positive=True)

# ---- 1. ES's flow (3.20) on V (-dt^2 + dxi^2) gives (4.2) -------------------------------------------------------
print("1. Emparan-Suzuki (3.20): (1/rho) d_rho g = 2 Ric on g = V(rho, xi)(-dt^2 + dxi^2)")
V = sp.Function("V")(rho, xi)
g = sp.diag(-V, V)
Ric = ricci_2d(g, [t, xi])
target = -sp.diff(V, rho) / rho - sp.diff(sp.log(V), xi, 2)          # (4.2): this vanishes
res_tt = sp.simplify((sp.diff(g[0, 0], rho) / rho - 2 * Ric[0, 0]) - target)
res_xx = sp.simplify((sp.diff(g[1, 1], rho) / rho - 2 * Ric[1, 1]) + target)
report("the tt and xi-xi components of (3.20) are +(4.2) and -(4.2) (R_ab = R g_ab / 2), off-diagonal zero",
       res_tt == 0 and res_xx == 0 and sp.simplify(Ric[0, 1]) == 0, f"residuals {res_tt}, {res_xx}")

# ---- 2. the double continuation ------------------------------------------------------------------------------
print("2. Kol's continuation (3.7): xi = i x, t = i z")
Vh = sp.Function("Vh")(sig, x)
gh = sp.diag(-Vh, Vh)                                          # coordinates (x, z): V(-dt^2 + dxi^2) -> Vh(-dx^2+dz^2)
Rh = ricci_2d(gh, [x, z])
drho = lambda e: sp.diff(e, sig) * sp.sqrt(2 * sig)             # d/d rho = rho d/d sigma, rho = sqrt(2 sigma)
rho_s = sp.sqrt(2 * sig)
diff_eq = sp.diff(Vh, sig) - sp.diff(sp.log(Vh), x, 2)          # forward diffusion: this vanishes
r_xx = sp.simplify((drho(gh[0, 0]) / rho_s - 2 * Rh[0, 0]) + diff_eq * 1)
r_zz = sp.simplify((drho(gh[1, 1]) / rho_s - 2 * Rh[1, 1]) - diff_eq * 1)
report("(3.20) on Vh(-dx^2 + dz^2) is the forward diffusion Vh_sigma = (ln Vh)_xx in both components",
       r_xx == 0 and r_zz == 0)
# the same from (4.2) by the chain rule: V(rho, xi) = Vh(rho^2/2, -i xi); d_xi = -i d_x
W2 = sp.Function("W2")
Vc = W2(rho ** 2 / 2, -sp.I * xi)
e42 = sp.simplify(-sp.diff(Vc, rho) / rho - sp.diff(sp.log(Vc), xi, 2))
X0 = sp.Symbol("X0")
e42 = e42.subs(-sp.I * xi, X0).doit()
e_cont = (sp.diff(W2(sig, x), sig) - sp.diff(sp.log(W2(sig, x)), x, 2))
chk = sp.simplify(e42.subs({rho: sp.sqrt(2 * sig), X0: x}).doit() + e_cont)
report("ES's (4.2) with xi = i x is minus the forward diffusion (same solutions)", chk == 0, f"difference {chk}")

# ---- 3. the Weyl factor --------------------------------------------------------------------------------------
print("3. Kol's Weyl rescaling e^{2A/(d-2)}, e^{2A} = V/n, d = n + 2")
Vs = sp.Symbol("V", positive=True)
lim = sp.limit((Vs / n) ** (1 / n), n, sp.oo)
report("(V/n)^(1/n) -> 1 as n -> infinity at fixed V", lim == 1, f"limit {lim}")

# ---- 4. the map onto form L, in jet space -------------------------------------------------------------------
print("4. reduction over z and the gauge X_s = -grad phi: form L, alpha = 1" + (f"  [mutate={MUT}]" if MUT else ""))
K = 6
Vj = sp.symbols("V0:%d" % (K + 3), positive=True)                 # Vh and its x-derivatives at x = X(s, t)
Pj = sp.symbols("P1:%d" % (K + 2), positive=True)                 # X_t, X_tt, ...


def Dx(e):
    """total x-derivative on Vh-jets"""
    return sum(sp.diff(e, Vj[k]) * Vj[k + 1] for k in range(K + 2))


sgn = -1 if MUT == "nocont" else 1
E0 = sgn * (Vj[2] / Vj[0] - Vj[1] ** 2 / Vj[0] ** 2)              # Vh_sigma = (ln Vh)_xx (continued; nocont: minus)
Es = [E0]
for k in range(K):
    Es.append(sp.expand(Dx(Es[-1])))                              # d_x^k Vh_sigma
phi_x = Vj[1] / (2 * Vj[0])
Wv = -(2 if MUT == "gauge2" else 1) * phi_x / Vj[0]               # X_s = -g^{xx} phi_x


def Dt(e):
    out = sum(sp.diff(e, Vj[k]) * Vj[k + 1] * Pj[0] for k in range(K + 2))
    return out + sum(sp.diff(e, Pj[k]) * Pj[k + 1] for k in range(K))


Ws = [Wv]                                                         # d_t^k (X_s) as jets: X_{t^k s}
for k in range(K):
    Ws.append(sp.expand(Dt(Ws[-1])))


def Ds(e):
    out = sum(sp.diff(e, Vj[k]) * (Es[k] + Vj[k + 1] * Wv) for k in range(K + 1))
    return out + sum(sp.diff(e, Pj[k]) * Ws[k + 1] for k in range(K))


phi = sp.log(Vj[0]) / 2
c2 = Vj[0] * Pj[0] ** 2
c = sp.sqrt(Vj[0]) * Pj[0]
alpha = 2 if MUT == "alpha2" else 1
phi_t = Dt(phi)
R1 = sp.simplify(Ds(phi) - (1 / c) * Dt(phi_t / c))
R2 = sp.simplify(Ds(c2) - 2 * alpha * phi_t ** 2)
report("phi_s - c^-1 (c^-1 phi_t)_t = 0", R1 == 0, f"residual {R1}")
report("(c^2)_s - 2 alpha phi_t^2 = 0 with alpha = %d" % alpha, R2 == 0, f"residual {R2}")

# ---- 5. through Prop. 5(i) ------------------------------------------------------------------------------------
print("5. Prop. 5(i): Y_t = c W, W_t = c (Y_s + Y^3/4), c_s = c Y^2/4 with Y = 2 phi_t / c")
Y = 2 * phi_t / c
Wp = Dt(Y) / c
R3 = sp.simplify(Dt(Wp) - c * (Ds(Y) + Y ** 3 / 4))
R4 = sp.simplify(Ds(c) - c * Y ** 2 / 4 * (alpha if MUT == "alpha2" else 1))
report("W_t - c (Y_s + Y^3/4) = 0", R3 == 0, f"residual {R3}")
report("c_s - c Y^2/4 = 0", R4 == 0, f"residual {R4}")

# ---- 6. exact solutions -------------------------------------------------------------------------------------
print("6. exact solutions of ES and their counterparts")
es42 = lambda Vx: sp.simplify(-sp.diff(Vx, rho) / rho - sp.diff(sp.log(Vx), xi, 2))
cone = (rho ** 2 - mu) / sp.cosh(xi) ** 2
kr = sp.sinh(2 * (rho ** 2 - mu)) / (sp.cosh(2 * (rho ** 2 - mu)) + sp.cosh(2 * xi))
bh = 1 / (1 + sp.exp(-2 * (xi + rho ** 2)))
for name, sol in (("cone (4.7)", cone), ("King-Rosenau (4.22)", kr), ("single black hole (4.9)", bh)):
    r42 = es42(sol)
    cont = sp.simplify(sol.subs({xi: sp.I * x, rho: sp.sqrt(2 * sig)}))
    im = sp.simplify(sp.im(sp.expand_complex(cont.subs({x: sp.Rational(3, 10), sig: sp.Rational(7, 10),
                                                         mu: sp.Rational(1, 5)}))))
    rdiff = sp.simplify(sp.diff(cont, sig) - sp.diff(sp.log(cont), x, 2)) if im == 0 else None
    real = im == 0
    if name.startswith("single"):
        print(f"  info  {name}: solves (4.2) {r42 == 0}; continuation real: {real} (Im at a sample point {im})")
        continue
    report(f"{name} solves (4.2), is real after xi = i x ({sp.simplify(cont)}) and solves the continued diffusion",
           r42 == 0 and real and rdiff == 0, f"(4.2) {r42}, continued {rdiff}")
# the cone's gauge ODE: X_sigma = -phi_x / Vh = -sin(2X) / (2 (2 sigma - mu)) -> tan X = t / sqrt(2 sigma - mu)
u = 2 * s - mu
Xc = sp.atan(t / sp.sqrt(u))
Vhc = u / sp.cos(x) ** 2
ode = sp.simplify(sp.diff(Xc, s) - (-sp.diff(sp.log(Vhc), x) / 2 / Vhc).subs(x, Xc))
phic = sp.simplify((sp.log(Vhc) / 2).subs(x, Xc))
c2c = sp.simplify(Vhc.subs(x, Xc) * sp.diff(Xc, t) ** 2)
cc = sp.sqrt(c2c)
L1 = sp.simplify(sp.diff(phic, s) - sp.diff(sp.diff(phic, t) / cc, t) / cc)
L2 = sp.simplify(sp.diff(c2c, s) - 2 * sp.diff(phic, t) ** 2)
Yc = sp.simplify(2 * sp.diff(phic, t) / cc)
P1 = sp.simplify(sp.diff(sp.diff(Yc, t) / cc, t) - cc * (sp.diff(Yc, s) + Yc ** 3 / 4))
P2 = sp.simplify(sp.diff(cc, s) - cc * Yc ** 2 / 4)
report("the cone's gauge ODE is solved by tan X = t / sqrt(2s - mu)", ode == 0, f"residual {ode}")
report("the cone's counterpart phi = %s, c^2 = %s solves form L" % (phic, c2c), L1 == 0 and L2 == 0,
       f"residuals {L1}, {L2}")
report("and Prop. 5(i), with Y = %s" % Yc, P1 == 0 and P2 == 0, f"residuals {P1}, {P2}")

print(f"\n{FAILS} check(s) failed" + (f" under --mutate={MUT}" if MUT else ""))
sys.exit(1 if FAILS else 0)
