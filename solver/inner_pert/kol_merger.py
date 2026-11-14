"""Kol's double-cone mode at D = infinity and the weight-2 generator (sympy; issue #429).

Kol [K06, Eqs. (2.3)-(2.5)] perturbs the Ricci-flat double cone over S^2 x S^{D-3},
    ds^2 = drho^2 + rho^2/(D-2) [ e^{eps} dOmega_2^2 + (D-4) e^{-2 eps/(D-3)} dOmega_{D-3}^2 ],
and finds eps = rho^s with s_pm = ((D-2)/2)(-1 +- i sqrt(8/(D-2) - 1)).
Here n = D - 2. The checks, in order (each prints PASS or FAIL; the exit code is the number of FAILs):
  0. the warped-product Ricci formula against a brute-force Ricci tensor at D = 5 and 6 (lapse included), and
     O'Neill's formula for a 3-dimensional base times S^k against brute force at k = 2;
  1. Kol's ansatz: the cone is Ricci flat; linearized, rho-rho vanishes identically and both sphere components are
     multiples of P(s) = s^2 + (D-2) s + 2(D-2), whose roots are Eq. (2.5), symbolically and at D = 5, 7, 13 to 1e-20;
  2. a second gauge (the areal radius r of S^{D-3} as coordinate, lapse N): every nonzero 2x2 minor of the
     linearized system is P(s) times a factor that does not vanish at Eq. (2.5);
  3. D = infinity in the areal gauge, at fixed s = r^2/(2n) and A = a^2: N = 1 and A_s = 2, with no explicit s. The
     s-translation applied to the cone A = 2s gives dA/A = 1/s, i.e. eps ~ rho^-2, and P(s)/n -> s + 2;
  4. the root near -2 is -2 - 4/n - 16/n^2 - 80/n^3 + O(n^-4);
  5. Emparan-Suzuki's large-D system [ES19, Eqs. (4.2), (4.5)]: with N^2 = 1 + rho V_rho/(n V) every leading component
     vanishes on the cone V = (rho^2 - mu) sech^2 xi and on the King-Rosenau solution [ES19, Eq. (4.22)], and the
     xi-xi component is -(1/2)[(1/rho) V_rho + (ln V)_xixi]; a mutated profile sech^2(2 xi) fails;
  6. the linearization of (1/rho) V_rho + (ln V)_xixi = 0 about the cone, V = rho^2 sech^2 xi (1 + c rho^s P_l(tanh xi)),
     gives s = l(l+1) - 2 for l = 0..4. That is the D -> infinity limit of Asnin-Kol-Smolkin's scalar family [AKS06,
     Eq. (spectrum)] s^s_+ = (2 - D + sqrt((D-2)(4l^2+4l+D-10)))/2; l = 0 is the lambda-translation.
Mutations (each must make the script exit non-zero):
  --mutate=bigsphere   S^{D-3} factor e^{-2 eps/(D-2)} in place of e^{-2 eps/(D-3)};
  --mutate=prefactor   cone prefactor (D-3) in place of (D-4);
  --mutate=twosphere   S^2 factor e^{2 eps} in place of e^{eps};
  --mutate=kr          King-Rosenau profile with cosh(4 xi) in place of cosh(2 xi) in check 5.
References.
  [K06]   B. Kol, Choptuik scaling and the merger transition, JHEP 10 (2006) 017, hep-th/0502033.
  [AKS06] V. Asnin, B. Kol, M. Smolkin, Analytic evidence for continuous self similarity of the critical merger
          solution, Class. Quant. Grav. 23 (2006) 6805, hep-th/0607129.
  [ES19]  R. Emparan, R. Suzuki, Topology-changing horizons at large D as Ricci flows, JHEP 07 (2019) 094,
          arXiv:1905.01062.
    python3 kol_merger.py [--mutate=NAME]
"""
import sys

import mpmath as mp
import sympy as sp

MUT = None
for arg in sys.argv[1:]:
    if arg.startswith("--mutate="):
        MUT = arg.split("=", 1)[1]
        if MUT not in ("bigsphere", "prefactor", "twosphere", "kr"):
            sys.exit(f"unknown mutation {MUT}")
FAILS = 0


def verdict(name, ok, detail=""):
    global FAILS
    FAILS += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def ricci_brute(g, X):
    """Coordinate Ricci tensor R_mn = d_l G^l_mn - d_n G^l_ml + G^l_ls G^s_mn - G^l_ns G^s_ml."""
    d = len(X)
    gi = g.inv()
    G = [[[sp.simplify(sum(gi[l, k] * (sp.diff(g[k, m], X[nn]) + sp.diff(g[k, nn], X[m]) - sp.diff(g[m, nn], X[k]))
                           for k in range(d)) / 2) for nn in range(d)] for m in range(d)] for l in range(d)]
    R = sp.zeros(d, d)
    for m in range(d):
        for nn in range(m, d):
            e = sum(sp.diff(G[l][m][nn], X[l]) - sp.diff(G[l][m][l], X[nn]) for l in range(d))
            e += sum(G[l][l][s] * G[s][m][nn] - G[l][nn][s] * G[s][m][l] for l in range(d) for s in range(d))
            R[m, nn] = R[nn, m] = sp.simplify(e)
    return R, G


def warped_frame(N, radii, dims, x):
    """Frame Ricci of N(x)^2 dx^2 + sum_i radii_i^2 (unit S^{dims_i}); returns (R_xx coordinate, [R_i frame])."""
    D1 = lambda f: sp.diff(f, x) / N                       # d/d(proper distance)
    Rxx = -N ** 2 * sum(m * D1(D1(a)) / a for a, m in zip(radii, dims))
    Rs = []
    for i, (a, m) in enumerate(zip(radii, dims)):
        e = -D1(D1(a)) / a - (m - 1) * D1(a) ** 2 / a ** 2 + (m - 1) / a ** 2
        e -= sum(mj * D1(a) * D1(b) / (a * b) for j, (b, mj) in enumerate(zip(radii, dims)) if j != i)
        Rs.append(e)
    return Rxx, Rs


rho, r, D, s, c, n = sp.symbols("rho r D s c n", positive=True)
th1, ph1, th2, ph2, psi = sp.symbols("theta1 phi1 theta2 phi2 psi")

# ---- 0a. warped-product formula against brute force -------------------------------------------------------------
Nf, af, bf = (sp.Function(q)(rho) for q in ("N", "a", "b"))
for k, X, g in (
        (2, [rho, th1, ph1, th2, ph2],
         sp.diag(Nf ** 2, af ** 2, af ** 2 * sp.sin(th1) ** 2, bf ** 2, bf ** 2 * sp.sin(th2) ** 2)),
        (3, [rho, th1, ph1, psi, th2, ph2],
         sp.diag(Nf ** 2, af ** 2, af ** 2 * sp.sin(th1) ** 2, bf ** 2, bf ** 2 * sp.sin(psi) ** 2,
                 bf ** 2 * sp.sin(psi) ** 2 * sp.sin(th2) ** 2))):
    R, _ = ricci_brute(g, X)
    Rxx, (R2, Rk) = warped_frame(Nf, [af, bf], [2, k], rho)
    res = [sp.simplify(R[0, 0] - Rxx), sp.simplify(R[1, 1] / g[1, 1] - R2), sp.simplify(R[3, 3] / g[3, 3] - Rk),
           sp.simplify(R[0, 1]), sp.simplify(R[1, 3])]
    verdict(f"0a warped-product Ricci = brute force, D = {k + 3}", all(e == 0 for e in res), f"residuals {res}")

# ---- 1. Kol's ansatz (proper distance rho) ----------------------------------------------------------------------
eps = c * rho ** s
pref = (D - 3) if MUT == "prefactor" else (D - 4)
e2 = sp.exp(2 * eps) if MUT == "twosphere" else sp.exp(eps)          # [K06] writes e^{2 eps/2}
ek = sp.exp(-2 * eps / (D - 2)) if MUT == "bigsphere" else sp.exp(-2 * eps / (D - 3))
a_K = rho * sp.sqrt(e2 / (D - 2))
b_K = rho * sp.sqrt(pref * ek / (D - 2))
Rxx, (R2, Rk) = warped_frame(sp.Integer(1), [a_K, b_K], [2, D - 3], rho)
comps = {"rho-rho": Rxx, "S^2": R2, "S^(D-3)": Rk}
bg = {k_: sp.simplify(v.subs(c, 0)) for k_, v in comps.items()}
verdict("1a the cone is Ricci flat", all(v == 0 for v in bg.values()), f"{bg}")
lin = {k_: sp.factor(sp.simplify(sp.diff(v, c).subs(c, 0) * rho ** (2 - s))) for k_, v in comps.items()}
P = s ** 2 + (D - 2) * s + 2 * (D - 2)
print("     linearized components times rho^(2-s):", lin)
quot = {k_: sp.simplify(v / P) for k_, v in lin.items()}
verdict("1b rho-rho vanishes identically", lin["rho-rho"] == 0, f"{lin['rho-rho']}")
verdict("1c both sphere components are s-independent multiples of P(s)",
        all(q != 0 and sp.diff(q, s) == 0 for k_, q in quot.items() if k_ != "rho-rho"), f"{quot}")
s_kol = [(D - 2) / 2 * (-1 + sg * sp.I * sp.sqrt(8 / (D - 2) - 1)) for sg in (1, -1)]
verdict("1d Eq. (2.5) solves P symbolically", all(sp.simplify(sp.expand(P.subs(s, z))) == 0 for z in s_kol))
mp.mp.dps = 40
worst = mp.mpf(0)
for Dv in (5, 7, 13):
    kol = [mp.mpf(Dv - 2) / 2 * (-1 + sg * 1j * mp.sqrt(mp.mpf(8) / (Dv - 2) - 1)) for sg in (1, -1)]
    for k_, v in lin.items():                       # every component must vanish at both of Kol's exponents
        f = sp.lambdify(s, v.subs(D, Dv), "mpmath")
        worst = max(worst, max(abs(f(z)) for z in kol))
    roots = mp.polyroots([1, Dv - 2, 2 * (Dv - 2)], extraprec=60)
    worst = max(worst, min(abs(roots[0] - kol[0]) + abs(roots[1] - kol[1]),
                           abs(roots[0] - kol[1]) + abs(roots[1] - kol[0])))
verdict("1e at D = 5, 7, 13 all components vanish at Eq. (2.5) to 1e-20", worst < mp.mpf("1e-20"),
        f"max {mp.nstr(worst, 3)}")

# ---- 2. areal gauge: N(r)^2 dr^2 + A(r) dOmega_2^2 + r^2 dOmega_{D-3}^2 ------------------------------------------
al, nu = sp.symbols("alpha nu")
N0, A0 = sp.sqrt((D - 2) / (D - 4)), r ** 2 / (D - 4)               # the cone in this gauge
Na = N0 * (1 + c * nu * r ** s)
Aa = A0 * (1 + c * al * r ** s)
Rxx, (R2, Rk) = warped_frame(Na, [sp.sqrt(Aa), r], [2, D - 3], r)
bg2 = [sp.simplify(e.subs(c, 0)) for e in (Rxx, R2, Rk)]
rows = [sp.expand(sp.powsimp(sp.simplify(sp.diff(e, c).subs(c, 0) * r ** (2 - s)), force=True)) for e in (Rxx, R2, Rk)]
Mlin = sp.Matrix([[e.coeff(al), e.coeff(nu)] for e in rows])
ok2 = all(v == 0 for v in bg2) and not Mlin.has(r)
minors = [sp.factor(sp.simplify(Mlin.extract([i, j], [0, 1]).det())) for i, j in ((0, 1), (0, 2), (1, 2))]
print("     areal-gauge 2x2 minors:", minors)
nonzero = [m_ for m_ in minors if m_ != 0]
ok2 &= len(nonzero) > 0
for mnr in nonzero:
    q, rem = sp.div(sp.Poly(sp.numer(sp.together(mnr)), s), sp.Poly(P, s))
    ok2 &= all(sp.simplify(cf) == 0 for cf in rem.all_coeffs())
    ok2 &= all(sp.simplify(q.as_expr().subs(s, z)) != 0 for z in s_kol)
g_ = sp.Poly(sp.numer(sp.together(nonzero[0])), s)
for mnr in nonzero[1:]:
    g_ = sp.gcd(g_, sp.Poly(sp.numer(sp.together(mnr)), s))
ok2 &= g_.degree() == 2 and sp.simplify(g_.as_expr() / g_.LC() - P) == 0   # the common factor is exactly P
verdict("2  areal gauge: cone Ricci flat and every 2x2 minor is P(s) times a factor nonzero at Eq. (2.5)", ok2)

# ---- 3. D = infinity at fixed s = r^2/(2n), n = D - 2 ----------------------------------------------------------
S_ = sp.symbols("s_", positive=True)
Af, Nh = sp.Function("A")(S_), sp.Function("Nh")(S_)
h = sp.symbols("h", positive=True)                    # h = 1/n
Nr, Ar = sp.Function("Nr")(r), sp.Function("Ar")(r)
Rxx, (R2, Rk) = warped_frame(Nr, [sp.sqrt(Ar), r], [2, D - 3], r)


def to_s(expr):
    """Substitute r = sqrt(2 n s), A(r) = A(s), N(r) = N(s), with d/dr = (r/n) d/ds."""
    rr = sp.sqrt(2 * S_ / h)
    rep = {}
    for F, G in ((Ar, Af), (Nr, Nh)):
        for k in (2, 1):
            dk = G
            for _ in range(k):
                dk = rr * h * sp.diff(dk, S_)
            rep[sp.diff(F, r, k)] = dk
        rep[F] = G
    return sp.simplify(expr.xreplace(rep).subs(D, 1 / h + 2).subs(r, rr))


# rows normalized to O(1): R_rr as is, R_S2 times A, R_S(D-3) times r^2/n
lead = {}
for name, e, mult in (("rho-rho", Rxx, 1), ("S^2", R2, Ar), ("S^(D-3)", Rk, r ** 2)):
    E = to_s(e * mult) * (h if name == "S^(D-3)" else 1)
    lead[name] = sp.simplify(sp.limit(sp.simplify(E), h, 0))
print("     leading order at D = infinity:", lead)
target = {"rho-rho": Nh.diff(S_) / Nh, "S^2": 1 - Af.diff(S_) / (2 * Nh ** 2), "S^(D-3)": 1 - 1 / Nh ** 2}
ok3 = all(sp.simplify(lead[k_] - target[k_]) == 0 for k_ in target)
A0s, Ap, N0s, Np = sp.symbols("A0s Ap N0s Np")
frozen = [e.subs({Af.diff(S_): Ap, Nh.diff(S_): Np}).subs({Af: A0s, Nh: N0s}) for e in lead.values()]
ok3 &= all(f.free_symbols <= {A0s, Ap, N0s, Np} for f in frozen)     # autonomous: no explicit s, r or n
verdict("3a D = inf: N_s = 0, N = 1 and A_s = 2 N^2, with no explicit s, r or n", ok3, f"{lead}")
cone = 2 * S_
gen = sp.diff(cone, S_)                                 # s-translation: dA = A_s (here, d/ds of the cone)
verdict("3b s-translation of the cone: dA/A = 1/s, i.e. eps ~ rho^-2",
        sp.simplify(gen / cone - 1 / S_) == 0 and lead["S^2"].subs({Nh: 1}).subs(Af, cone).doit() == 0)
lim = sp.limit(P.subs(D, 1 / h + 2) * h, h, 0)
verdict("3c P(s)/n -> s + 2", sp.simplify(lim - (s + 2)) == 0, f"{lim}")

# ---- 4. the root near -2 ---------------------------------------------------------------------------------------
root = (-(1 / h) + sp.sqrt((1 / h) ** 2 - 8 / h)) / 2   # s_+ with D - 2 = 1/h, real for n >= 8
ser = sp.series(sp.simplify(root), h, 0, 4).removeO()
verdict("4  s_+ = -2 - 4/n - 16/n^2 - 80/n^3 + O(n^-4)",
        sp.simplify(ser - (-2 - 4 * h - 16 * h ** 2 - 80 * h ** 3)) == 0, f"{sp.expand(ser)}")

# ---- 0b, 5. O'Neill's formula and Emparan-Suzuki's leading-order system ----------------------------------------
xi, t = sp.symbols("xi t", real=True)
mu = sp.symbols("mu", real=True)
Nb, Wb = sp.Function("N")(rho, xi), sp.Function("W")(rho, xi)
XB = [rho, xi, t]
gB = sp.diag(Nb ** 2, Wb, Wb)
RB, GB = ricci_brute(gB, XB)
hess = sp.Matrix(3, 3, lambda i, j: sp.diff(rho, XB[i], XB[j]) - sum(GB[l][i][j] * sp.diff(rho, XB[l])
                                                                    for l in range(3)))
grad2 = sum(gB.inv()[i, j] * sp.diff(rho, XB[i]) * sp.diff(rho, XB[j]) for i in range(3) for j in range(3))
lap = sum(sp.diff(sp.sqrt(gB.det()) * gB.inv()[i, j] * sp.diff(rho, XB[j]), XB[i]) for i in range(3)
          for j in range(3)) / sp.sqrt(gB.det())


def oneill(k):
    """Ricci of g_B + rho^2 (unit S^k): horizontal block and the fibre's frame value."""
    return RB - k / rho * hess, (k - 1) / rho ** 2 - (lap / rho + (k - 1) * grad2 / rho ** 2)


Rfull, _ = ricci_brute(sp.diag(Nb ** 2, Wb, Wb, rho ** 2, rho ** 2 * sp.sin(th2) ** 2), XB + [th2, ph2])
H2, F2 = oneill(2)
res0b = [sp.simplify(Rfull[i, j] - H2[i, j]) for i in range(3) for j in range(i, 3)]
res0b.append(sp.simplify(Rfull[3, 3] / rho ** 2 - F2))
verdict("0b O'Neill's formula = brute force at k = 2 (base N(rho,xi)^2 drho^2 + W(rho,xi)(dxi^2 + dt^2))",
        all(e == 0 for e in res0b))

Vf = sp.Function("V")(rho, xi)
Hk, Fk = oneill(n)
sub_es = {Wb: Vf / n, Nb: sp.sqrt(1 + rho * sp.diff(Vf, rho) / (n * Vf))}


def jet(expr):
    """Replace V(rho, xi) and its derivatives by plain symbols (limit() cannot collect mixed derivatives)."""
    ds = sorted(expr.atoms(sp.Derivative), key=lambda d_: -sum(k for _, k in d_.variable_count))
    rep = {d_: sp.Symbol("V_" + "".join(str(v) * k for v, k in d_.variable_count)) for d_ in ds}
    return expr.xreplace(rep).xreplace({Vf: sp.Symbol("V_")}), {v: k for k, v in rep.items()}


def es_leading(expr):
    e, back = jet(expr.subs(sub_es).doit().subs(n, 1 / h))
    lim_ = sp.limit(sp.cancel(sp.together(e)), h, 0)
    return sp.simplify(lim_.xreplace(back).xreplace({sp.Symbol("V_"): Vf}))


comp5 = {"rho-rho": Hk[0, 0], "rho-xi": Hk[0, 1], "xi-xi": Hk[1, 1], "t-t": Hk[2, 2], "fibre": Fk * rho ** 2}
lead5 = {k_: es_leading(v) for k_, v in comp5.items()}
pde = sp.diff(Vf, rho) / rho + sp.diff(sp.log(Vf), xi, 2)
verdict("5a xi-xi and t-t leading = -(1/2)[(1/rho) V_rho + (ln V)_xixi]",
        all(sp.simplify(lead5[k_] + pde / 2) == 0 for k_ in ("xi-xi", "t-t")), f"{lead5['xi-xi']}")
verdict("5b rho-xi and the fibre vanish identically at leading order", lead5["rho-xi"] == 0 and lead5["fibre"] == 0,
        f"{lead5['rho-xi']}, {lead5['fibre']}")
kr_arg = 4 * xi if MUT == "kr" else 2 * xi
sols = {"cone": (rho ** 2 - mu) / sp.cosh(xi) ** 2,
        "King-Rosenau": sp.sinh(2 * (rho ** 2 - mu)) / (sp.cosh(2 * (rho ** 2 - mu)) + sp.cosh(kr_arg))}
for name, Vs in sols.items():
    vals = [sp.simplify(sp.expand_trig(v.subs(Vf, Vs).doit().rewrite(sp.exp))) for v in lead5.values()]
    verdict(f"5c every leading component vanishes on the {name} solution", all(v == 0 for v in vals), f"{vals}")
Vbad = (rho ** 2 - mu) / sp.cosh(2 * xi) ** 2
bad = sp.simplify(pde.subs(Vf, Vbad).doit())
verdict("5d control: the profile sech^2(2 xi) does not solve it", bad != 0, f"{bad}")

# ---- 6. linearization about the cone: s = l(l+1) - 2 ----------------------------------------------------------
ok6, rows6 = True, []
for l in range(5):
    Pl = sp.legendre(l, sp.tanh(xi))
    Vl = rho ** 2 / sp.cosh(xi) ** 2 * (1 + c * rho ** s * Pl)
    e = sp.diff(pde.subs(Vf, Vl).doit(), c).subs(c, 0)
    e = sp.powsimp(sp.expand(e * rho ** (-s) * sp.cosh(xi) ** 2), force=True)
    sl = sp.simplify((-e.subs(s, 0) / sp.diff(e, s)).rewrite(sp.exp))      # e is linear in s
    aks = (2 - D + sp.sqrt((D - 2) * (4 * l ** 2 + 4 * l + D - 10))) / 2
    lim_aks = sp.limit(aks, D, sp.oo)
    rows6.append((l, sl, lim_aks))
    ok6 &= sp.simplify(sl - (l * (l + 1) - 2)) == 0 and lim_aks == l * (l + 1) - 2
print("     (l, exponent from the linearized flow, D -> inf of AKS s^s_+):", rows6)
verdict("6  linearized flow gives s = l(l+1) - 2 = lim AKS s^s_+ for l = 0..4", ok6)
es_quoted = sp.limit((2 - D + sp.sqrt((D - 2) * (4 * 2 ** 2 + 12 * 2 + D + 6))) / 2, D, sp.oo)
print(f"     note: ES19 App. A prints 4l^2+12l+D+6 (AKS's tensor family s^t); its l = 2 limit is {es_quoted}, not 4")

print(f"\n{FAILS} check(s) failed" + (f" under --mutate={MUT}" if MUT else ""))
sys.exit(1 if FAILS else 0)
