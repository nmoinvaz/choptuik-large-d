"""The Reiterer-Trubowitz (RT) gauge equations in D dimensions, checked with sympy against the covariant
Einstein-scalar equations.

    nice -n 10 python3 solver/rt/rt_eqs_check.py                 (about a minute; exits 0 when every check passes)
    nice -n 10 python3 solver/rt/rt_eqs_check.py --mutate=coef   (one coefficient of Omega_3 scaled; must exit 1)
    nice -n 10 python3 solver/rt/rt_eqs_check.py --mutate=warp   (the warped-product formula broken; must exit 1)

Gauge (RT Sec. 2.1 at D = 4; EEG arXiv:2611.07007v1 Eq. (2) in D = n + 2):
  ds^2 = e^{-2 zeta} ( -(1 - xi^2) dt^2 - (2/mu) xi dt dxi + dxi^2/mu^2 + xi^2/(mu + Q xi^2)^2 dOmega_n^2 ),
  D^s = s d_t + mu (1 + s xi) d_xi  (s = +-),  Ric = 2 dphi dphi,  box phi = 0.
Unknowns (RT Eq. (8), EEG Eq. (4)): omega_1 = 2 xi Q + omega_2^+ + omega_2^-, omega_2^s = D^s(zeta + ln(mu + Q xi^2)) - s mu,
omega_3^s = D^s zeta - s mu, omega_4^s = D^s phi.
Equations: EEG Eq. (5a)-(5e) (Omega_5 is RT's Omega_2#), transcribed from the arXiv HTML of 2611.07007v1.

1. The warped-product curvature R_ab = K g_ab - (n/r) nabla_a nabla_b r, R_ij = [(n-1)(1 - |dr|^2) - r box r] ghat_ij
   and box phi = box_2 phi + (n/r) <dr, dphi>, against brute-force Ricci of the D = 4 and D = 5 metric in angles.
2. Each of the ten Omega^s_i, written in zeta, Q, phi, is a combination of the five covariant components
   E_{++}, E_{--}, E_{+-} (E_ab = R_ab - 2 d_a phi d_b phi along D^s), E_S (the sphere component) and W = box phi,
   with coefficients free of second derivatives: the coefficients are fixed by the second derivatives (a rank-5
   linear system), and the remainder must vanish identically, as a rational function of the jets and of n.
   Five combinations are identities (RT: Omega_1 - Omega_2 - Omega_2#, and Omega_i^+ - Omega_i^- for i = 2, 3, 4);
   their coefficients come out 0.
3. At D = 4 the D-dimensional equations reduce to RT's Eq. (11a)-(11e) (read from arXiv:1203.3766v1).
"""
import sys
import sympy as sp

MUT = None
for a in sys.argv[1:]:
    if a.startswith("--mutate="):
        MUT = a.split("=", 1)[1]
        assert MUT in ("coef", "warp"), MUT

t, X, mu, n = sp.symbols("t xi mu n")
E0 = sp.symbols("E0", positive=True)       # e^{zeta} at the point; zeta enters only through it and its derivatives
FAIL = []


def report(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        FAIL.append(label)


# ---------------------------------------------------------------- 1. warped-product formulas
def ricci(X_, g):
    N = len(X_)
    gi = g.inv()
    Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], X_[c]) + sp.diff(g[d, c], X_[b]) - sp.diff(g[b, c], X_[d]))
                 for d in range(N)) / 2 for c in range(N)] for b in range(N)] for a in range(N)]
    R = sp.zeros(N, N)
    for b in range(N):
        for c in range(N):
            s = 0
            for a in range(N):
                s += sp.diff(Gam[a][b][c], X_[a]) - sp.diff(Gam[a][b][a], X_[c])
                for d in range(N):
                    s += Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
            R[b, c] = s
    return R, Gam


def warped(nn, g2, rr, X2):
    """R_ab (2D block) and the sphere coefficient by the general-n formulas."""
    R2, Gam = ricci(X2, g2)
    gi = g2.inv()
    Rs2 = sp.simplify(sum(gi[a, b] * R2[a, b] for a in range(2) for b in range(2)))
    K = Rs2 / 2
    dr = [sp.diff(rr, x) for x in X2]
    hess = sp.Matrix(2, 2, lambda a, b: sp.diff(rr, X2[a], X2[b]) - sum(Gam[c][a][b] * dr[c] for c in range(2)))
    boxr = sum(gi[a, b] * hess[a, b] for a in range(2) for b in range(2))
    dr2 = sum(gi[a, b] * dr[a] * dr[b] for a in range(2) for b in range(2))
    k_hess = nn if MUT != "warp" else nn + 1
    Rab = sp.Matrix(2, 2, lambda a, b: K * g2[a, b] - k_hess * hess[a, b] / rr)
    RS = (nn - 1) * (1 - dr2) - rr * boxr
    return Rab, RS, gi, dr, boxr, dr2


print("1. warped-product curvature against brute force (D = 4, 5)")
A_, B_, C_, R_ = [sp.Function(s)(t, X) for s in ("A", "B", "C", "R")]
for nn in (2, 3):
    th = sp.symbols(f"th1:{nn + 1}")
    X_ = [t, X] + list(th)
    g = sp.zeros(nn + 2, nn + 2)
    g[0, 0], g[0, 1], g[1, 0], g[1, 1] = A_, B_, B_, C_
    for k in range(nn):
        g[2 + k, 2 + k] = R_ ** 2 * sp.prod([sp.sin(th[j]) ** 2 for j in range(k)])
    Rfull, _ = ricci(X_, g)
    Rab, RS, *_ = warped(nn, sp.Matrix([[A_, B_], [B_, C_]]), R_, [t, X])
    ok = all(sp.simplify(Rfull[a, b] - Rab[a, b]) == 0 for a in range(2) for b in range(2))
    ok &= sp.simplify(Rfull[2, 2] / R_ ** 2 - RS / R_ ** 2) == 0
    ok &= all(sp.simplify(Rfull[a, 2 + k]) == 0 for a in range(2) for k in range(nn))
    report(f"n = {nn}: R_ab, R_th th, R_a th", ok)

# ---------------------------------------------------------------- 2. RT gauge, D dimensions
print("2. the ten Omega^s_i as combinations of the covariant equations, general n")
zeta, Q, phi = [sp.Function(s)(t, X) for s in ("zeta", "Q", "phi")]
H = mu + Q * X ** 2
g2 = sp.exp(-2 * zeta) * sp.Matrix([[-(1 - X ** 2), -X / mu], [-X / mu, 1 / mu ** 2]])
rr = X * sp.exp(-zeta) / H
Rab, RS, gi, dr, boxr, dr2 = warped(n, g2, rr, [t, X])
dphi = [sp.diff(phi, t), sp.diff(phi, X)]
_, Gam = ricci([t, X], g2)
hphi = sp.Matrix(2, 2, lambda a, b: sp.diff(phi, [t, X][a], [t, X][b]) - sum(Gam[c][a][b] * dphi[c] for c in range(2)))
box2phi = sum(gi[a, b] * hphi[a, b] for a in range(2) for b in range(2))
W = box2phi + n * sum(gi[a, b] * dr[a] * dphi[b] for a in range(2) for b in range(2)) / rr


def Dv(s):
    return (s, mu * (1 + s * X))


def Dop(s, f):
    return s * sp.diff(f, t) + mu * (1 + s * X) * sp.diff(f, X)


def covariant(kappa):
    """E_{++}, E_{--}, E_{+-}, E_S, W for Ric = kappa dphi dphi."""
    Eab = sp.Matrix(2, 2, lambda a, b: Rab[a, b] - kappa * dphi[a] * dphi[b])
    con = lambda u, v: sum(u[a] * v[b] * Eab[a, b] for a in range(2) for b in range(2))
    return [con(Dv(1), Dv(1)), con(Dv(-1), Dv(-1)), con(Dv(1), Dv(-1)), RS, W]


Enames = ["E++", "E--", "E+-", "E_S", "W"]

w2 = {s: Dop(s, zeta + sp.log(H)) - s * mu for s in (1, -1)}
w3 = {s: Dop(s, zeta) - s * mu for s in (1, -1)}
w4 = {s: Dop(s, phi) for s in (1, -1)}
w1 = 2 * X * Q + w2[1] + w2[-1]
Dn = n + 2   # spacetime dimension D

k3 = sp.Rational(1, 2) * (1 + sp.Rational(1, 1000) if MUT == "coef" else 1)


def Omegas(s, w1, w2, w3, w4, Dop, Dn):
    """EEG Eq. (5a)-(5e); s = +1 or -1; Dop(s, f) the null derivative."""
    p, q = s, -s
    O1 = (X * (Dop(s, w1) + s * mu * w1) + mu * (Dn - 2) * w1
          + X / 4 * ((Dn - 3) * (w1 ** 2 + w2[1] ** 2 + w2[-1] ** 2 - 2 * w2[q] * (w2[p] + w1))
                     - 2 * (Dn - 1) * w1 * w2[p] + 4 * w3[p] * (w1 + w2[p] - w2[q]) - 4 * w4[p] ** 2))
    O2 = (X * (Dop(s, w2[q]) + s * mu * w2[q]) + mu * ((Dn - 3) * w1 + w2[1] + w2[-1])
          + X / 4 * ((Dn - 3) * (w1 ** 2 + w2[1] ** 2 + w2[-1] ** 2 - 2 * w1 * (w2[1] + w2[-1]))
                     - 2 * (Dn - 1) * w2[1] * w2[-1]))
    O3 = (X * (Dop(s, w3[q]) + s * mu * w3[q]) - mu / 2 * (Dn - 3) * (Dn - 2) * w1
          + X / 4 * (Dn - 2) * ((Dn - 3) * ((w2[1] + w2[-1]) * w1 - k3 * w1 ** 2 - sp.Rational(1, 2) * (w2[1] - w2[-1]) ** 2)
                                - 2 * w4[1] * w4[-1]))
    O4 = (X * (Dop(s, w4[q]) + s * mu * w4[q]) + mu / 2 * (Dn - 2) * (w4[1] + w4[-1])
          - X / 2 * (Dn - 2) * (w4[1] * w2[-1] + w4[-1] * w2[1]))
    O5 = (X * (Dop(s, w2[p]) + s * mu * w2[p]) + 2 * mu * (w2[p] - w3[p])
          + X * (2 * w3[p] * w2[p] - w2[p] ** 2 - w4[p] ** 2))
    return [O1, O2, O3, O4, O5]


Om = {s: Omegas(s, w1, w2, w3, w4, Dop, Dn) for s in (1, -1)}

# jets: replace derivatives by symbols, zeta by log(E0)
J = {}
for f, nm in ((zeta, "z"), (Q, "q"), (phi, "f")):
    for (i, j) in ((2, 0), (1, 1), (0, 2), (1, 0), (0, 1)):
        d = f
        for _ in range(i):
            d = sp.diff(d, t)
        for _ in range(j):
            d = sp.diff(d, X)
        J[d] = sp.Symbol(f"{nm}_{'t' * i}{'x' * j}")
SEC = [J[k] for k in J if str(J[k]).count("t") + str(J[k]).count("x") == 2]
q0 = sp.Symbol("q")


def jet(e):
    e = e.xreplace(J).xreplace({zeta: sp.log(E0), Q: q0})
    return sp.cancel(sp.together(e))


csym = sp.symbols("c0:5")
Omj = {s: [jet(e) for e in Om[s]] for s in (1, -1)}


def combos(kappa, verbose):
    """For each Omega: coefficients from the second derivatives, and the remainder (0 when the equation holds)."""
    Ej = [jet(e) for e in covariant(kappa)]
    Bmat = sp.Matrix([[sp.cancel(sp.diff(e, v)) for e in Ej] for v in SEC])     # 9 x 5
    if verbose:
        print(f"  rank of the second-derivative matrix of the covariant equations: {Bmat.rank(simplify=True)}")
    out = []
    for s in (1, -1):
        for i in range(5):
            lab = f"Omega_{i + 1}^{'+' if s > 0 else '-'}"
            e = Omj[s][i]
            avec = sp.Matrix([sp.cancel(sp.diff(e, v)) for v in SEC])
            sol = sp.solve(list(Bmat * sp.Matrix(csym) - avec), csym, dict=True)
            if not sol:
                out.append((lab, None, None))
                continue
            cs = [sp.cancel(sol[0].get(c, c)) for c in csym]
            rem = sp.factor(sp.cancel(e - sum(c * ej for c, ej in zip(cs, Ej))))
            out.append((lab, [nm for c, nm in zip(cs, Enames) if c != 0], rem))
    return out


print("  coupling Ric = (D - 2) dphi dphi, the normalization under which EEG Eq. (5) holds as printed:")
for lab, nz, rem in combos(n, True):
    if nz is None:
        report(f"{lab}: second derivatives not a combination of the covariant equations", False)
    else:
        report(f"{lab} = combination of {', '.join(nz) if nz else 'nothing (an identity)'}; remainder {rem}", rem == 0)
print("  coupling Ric = 2 dphi dphi (EEG Eq. (1) with 4 pi G = 1): the printed Eq. (5) fails off D = 4, by")
bad = combos(2, False)
for lab, nz, rem in bad:
    if rem is not None and rem != 0:
        print(f"    {lab}: remainder {rem}")
report("with Ric = 2 dphi dphi every nonzero remainder carries the factor (n - 2) = (D - 4)",
       all(rem is not None and (rem == 0 or sp.cancel(rem.subs(n, 2)) == 0) for _, _, rem in bad)
       and any(rem not in (None, 0) for _, _, rem in bad))

# ---------------------------------------------------------------- 3. D = 4 against RT Eq. (11)
print("3. at D = 4 the equations are RT's Eq. (11)")
a1, a2m, a2p, a3m, a3p, a4m, a4p, dw = sp.symbols("w1 w2m w2p w3m w3p w4m w4p dw")
W2 = {1: a2p, -1: a2m}
W3 = {1: a3p, -1: a3m}
W4 = {1: a4p, -1: a4m}
Dz = lambda s, f: dw            # the derivative terms are identical in form; compare the rest
ok = True
for s in (1, -1):
    p, q = s, -s
    E = Omegas(s, a1, W2, W3, W4, Dz, 4)
    RT = [X * (dw + s * mu * a1) + mu * 2 * a1 + X / 4 * (a1 * a1 - 2 * W2[q] * a1 - 6 * W2[p] * a1 + 4 * W3[p] * a1
                                                            + W2[q] ** 2 - 2 * W2[p] * W2[q] - 4 * W3[p] * W2[q]
                                                            + W2[p] ** 2 + 4 * W3[p] * W2[p] - 4 * W4[p] ** 2),
          X * (dw + s * mu * W2[q]) + mu * (a1 + a2m + a2p) + X / 4 * (a1 * a1 - 2 * a2m * a1 - 2 * a2p * a1 + a2m ** 2
                                                                       - 6 * a2p * a2m + a2p ** 2),
          X * (dw + s * mu * W3[q]) - mu * a1 + X / 4 * (-a1 * a1 + 2 * a2m * a1 + 2 * a2p * a1 - a2m ** 2
                                                         + 2 * a2p * a2m - a2p ** 2 - 4 * a4p * a4m),
          X * (dw + s * mu * W4[q]) + mu * (a4m + a4p) + X / 4 * (-4 * a4p * a2m - 4 * a4m * a2p),
          X * (dw + s * mu * W2[p]) + mu * (2 * W2[p] - 2 * W3[p]) + X / 4 * (-4 * W2[p] ** 2 + 8 * W3[p] * W2[p]
                                                                         - 4 * W4[p] ** 2)]
    for i in range(5):
        d = sp.expand(E[i] - RT[i])
        ok &= d == 0
        if d != 0:
            print(f"    Omega_{i + 1}^{s:+d}: EEG(D = 4) - RT = {d}")
report("EEG Eq. (5) at D = 4 equals RT Eq. (11) (the Omega_5 of EEG is RT's Omega_2#)", ok)

print(f"\n{'ALL CHECKS PASS' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED'}" + (f" (mutation {MUT})" if MUT else ""))
sys.exit(1 if FAIL else 0)
