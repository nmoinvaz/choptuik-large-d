#!/usr/bin/env python3
"""Independent symbolic check of L2_MODE.md section 9's identity (issue #1010's review).

Written for notes/L2_C2_REVIEW.md before section 9 and l2_c2*.py were read. It shares no code with l2_mode.py or
l2_c2_forcing.py. What it shares with them is listed in the review note, section 4.

Frame. The l = 2 mode is taken as a metric perturbation, not as a deformation of the fields. The linear map
x -> (1 + a H)x of space, H traceless, applied to the spherical solution is a diffeomorphism. It gives the fields that
follow S = s + e s Y2 and the spatial metric delta + k, with k = 2 a H constant in Cartesian coordinates. So the
deformation of L2_MODE.md section 1 (fields follow S, flat metric kept) is that diffeomorphism minus k, and its residual
in the finite-n equations is minus the linearised equations applied to k. For p = q, k = e (dy.dy - dz.dz).

Curvature. Warped products over the base (t, s, u), u = tan(chi/2), with two sphere fibres (O'Neill 1983, ch. 7;
the formulas are re-derived in first-order perturbation theory here and tested on planted cases):
  R_ab = Rb_ab - sum_k m_k (D_a D_b U_k + d_a U_k d_b U_k),   box f = boxb f + sum_k m_k dU_k . df .

Checks (each exits non-zero on failure; --mutate=NAME breaks one input, and the check must then fail):
  A  planted: Schwarzschild-Tangherlini is Ricci flat in this frame, at n = 3, 4 and 7      (--mutate=fibre)
  B  planted: a diffeomorphism's first-order rows are the Lie derivative of the zeroth-order ones, off shell,
     exactly in n                                                                            (--mutate=lin)
  C  the spherical equations to O(1/n) equal the review's hand-derived ones, and FIRST_ORDER.md section 1's heat
     system through C = c/a, Y = 2 Phi_t/C, S = 2 Phi_s                                     (--mutate=bg)
  D  the l = 2 residual: zero at O(1), and at O(1/n) it is s Y2 times twice the radial second-derivative part Q
     of the spherical equations                                                              (--mutate=k)
  E  the identity: F2/(s Y2) + F0 = 0 with the l = 0 mode d_s(c, Phi) of the lapse c, and
     = L0[(-c_s, 0)] = L0[(0, Phi_s)] with FIRST_ORDER.md's mode, which translates C = c/a   (--mutate=sign)
  F  F2's lapse row equals the expression printed in L2_MODE.md section 8.3                  (--mutate=k)
  G  conjugation: the O(1) rows on (s Y2 gamma, s Y2 psi) are s Y2 times the l = 0 rows; control cos^2 chi
  H  the slicing family ((eps c)_t, eps Phi_t), eps(t) free, is in the kernel of the l = 0 rows
  I  which metric perturbations reach the two rows at O(1): the fibre warps only, through V1 + V2
Added after section 9 was read:
  B3 the deformation of L2_MODE.md 8.3, with its metric corrections, is exactly Lie_xi(background) - k
  J  every other row (momentum constraints, spatial rows, fibres) vanishes on k at its natural order
  J2 the chi-chi row at its natural order holds the second t-derivative of the metric function C
  P  the p = 1 frame: F2 = 2 s cos^2(chi) Q, the same F2/(s Y2) as p = q                      (--mutate=k)
  K  the stored forcings of data/l2_c2_forcing.json equal this derivation                    (--mutate=sign)

Run: python3 l2_c2_review_sym.py [--mutate=NAME]      (about 1 min, one process)
"""
import sys
import sympy as sp

MUT = None
for a in sys.argv[1:]:
    if a.startswith('--mutate='):
        MUT = a.split('=', 1)[1]

t, s, u, n, nu = sp.symbols('t s u n nu', positive=True)
X = (t, s, u)
cF = sp.Function('c')(s, t)
muF = sp.Function('mu')(s, t)
PhF = sp.Function('Phi')(s, t)

# chi = 2 atan u keeps every expression rational
cosx = (1 - u**2) / (1 + u**2)
sinx = 2 * u / (1 + u**2)
cos2 = sp.cancel(cosx**2 - sinx**2)
sin2 = sp.cancel(2 * sinx * cosx)
dchi_du = 2 / (1 + u**2)

FAILS = []
import time
T0 = time.time()


def report(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (('   ' + detail) if detail else '') + '   [%.0f s]' % (time.time() - T0),
          flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------------------
# first-order perturbation theory of the warped-product curvature
# ---------------------------------------------------------------------------------------------------------------
def rows(g0, h, U0, dU, m, phi0, dphi):
    """Zeroth- and first-order parts of the Ricci tensor R_ab (base indices) and of box(phi).

    g = g0 + e h on the base, U_k = U0_k + e dU_k, phi = phi0 + e dphi. Returns (R0, R1, box0, box1, ginv0, ginv1).
    """
    N = 3
    gi0 = g0.inv()
    gi1 = -gi0 * h * gi0

    def low(g):
        return [[[sp.Rational(1, 2) * (sp.diff(g[d, c], X[b]) + sp.diff(g[d, b], X[c]) - sp.diff(g[b, c], X[d]))
                  for c in range(N)] for b in range(N)] for d in range(N)]

    L0, L1 = low(g0), low(h)
    G0 = [[[sum(gi0[a, d] * L0[d][b][c] for d in range(N)) for c in range(N)] for b in range(N)] for a in range(N)]
    G1 = [[[sum(gi0[a, d] * L1[d][b][c] + gi1[a, d] * L0[d][b][c] for d in range(N)) for c in range(N)]
           for b in range(N)] for a in range(N)]
    if MUT == 'lin':  # drops the inverse metric's variation from the connection
        G1 = [[[sum(gi0[a, d] * L1[d][b][c] for d in range(N)) for c in range(N)] for b in range(N)]
              for a in range(N)]

    R0 = sp.zeros(N, N)
    R1 = sp.zeros(N, N)
    for b in range(N):
        for c in range(b, N):
            r0 = r1 = 0
            for a in range(N):
                r0 += sp.diff(G0[a][b][c], X[a]) - sp.diff(G0[a][b][a], X[c])
                r1 += sp.diff(G1[a][b][c], X[a]) - sp.diff(G1[a][b][a], X[c])
                for d in range(N):
                    r0 += G0[a][a][d] * G0[d][b][c] - G0[a][c][d] * G0[d][b][a]
                    r1 += (G1[a][a][d] * G0[d][b][c] + G0[a][a][d] * G1[d][b][c]
                           - G1[a][c][d] * G0[d][b][a] - G0[a][c][d] * G1[d][b][a])
            for k in range(len(m)):
                hess0 = sp.diff(U0[k], X[b], X[c]) - sum(G0[a][b][c] * sp.diff(U0[k], X[a]) for a in range(N))
                hess1 = (sp.diff(dU[k], X[b], X[c]) - sum(G0[a][b][c] * sp.diff(dU[k], X[a]) for a in range(N))
                         - sum(G1[a][b][c] * sp.diff(U0[k], X[a]) for a in range(N)))
                dd0 = sp.diff(U0[k], X[b]) * sp.diff(U0[k], X[c])
                dd1 = sp.diff(dU[k], X[b]) * sp.diff(U0[k], X[c]) + sp.diff(U0[k], X[b]) * sp.diff(dU[k], X[c])
                mk = m[k]
                if MUT == 'fibre' and k == 1:
                    mk = m[k] + 1
                r0 -= mk * (hess0 + dd0)
                r1 -= mk * (hess1 + dd1)
            R0[b, c] = R0[c, b] = r0
            R1[b, c] = R1[c, b] = r1

    box0 = box1 = 0
    for b in range(N):
        for c in range(N):
            cd0 = sp.diff(phi0, X[b], X[c]) - sum(G0[a][b][c] * sp.diff(phi0, X[a]) for a in range(N))
            cd1 = (sp.diff(dphi, X[b], X[c]) - sum(G0[a][b][c] * sp.diff(dphi, X[a]) for a in range(N))
                   - sum(G1[a][b][c] * sp.diff(phi0, X[a]) for a in range(N)))
            for k in range(len(m)):
                cd0 += m[k] * sp.diff(U0[k], X[b]) * sp.diff(phi0, X[c])
                cd1 += m[k] * (sp.diff(dU[k], X[b]) * sp.diff(phi0, X[c]) + sp.diff(U0[k], X[b]) * sp.diff(dphi, X[c]))
            box0 += gi0[b, c] * cd0
            box1 += gi0[b, c] * cd1 + gi1[b, c] * cd0
    FIB.clear()
    for k in range(len(m)):
        def dotp(f0, f1, h0, h1):
            d0 = sum(gi0[a, b] * sp.diff(f0, X[a]) * sp.diff(h0, X[b]) for a in range(N) for b in range(N))
            d1 = sum(gi1[a, b] * sp.diff(f0, X[a]) * sp.diff(h0, X[b])
                     + gi0[a, b] * (sp.diff(f1, X[a]) * sp.diff(h0, X[b]) + sp.diff(f0, X[a]) * sp.diff(h1, X[b]))
                     for a in range(N) for b in range(N))
            return d0, d1
        bx0 = sum(gi0[b, c] * (sp.diff(U0[k], X[b], X[c]) - sum(G0[a][b][c] * sp.diff(U0[k], X[a]) for a in range(N)))
                  for b in range(N) for c in range(N))
        bx1 = sum(gi1[b, c] * (sp.diff(U0[k], X[b], X[c]) - sum(G0[a][b][c] * sp.diff(U0[k], X[a]) for a in range(N)))
                  + gi0[b, c] * (sp.diff(dU[k], X[b], X[c]) - sum(G0[a][b][c] * sp.diff(dU[k], X[a])
                                                                 + G1[a][b][c] * sp.diff(U0[k], X[a]) for a in range(N)))
                  for b in range(N) for c in range(N))
        f0 = (m[k] - 1) * sp.exp(-2 * U0[k]) - bx0
        f1 = (m[k] - 1) * sp.exp(-2 * U0[k]) * (-2 * dU[k]) - bx1
        for l in range(len(m)):
            d0, d1 = dotp(U0[k], dU[k], U0[l], dU[l])
            f0 -= m[l] * d0
            f1 -= m[l] * d1
        FIB.append((f0, f1))
    return R0, R1, box0, box1, gi0, gi1


FIB = []          # the fibre rows of the last call of rows()


def two_rows(g0, h, U0, dU, m, phi0, dphi, cc):
    """(W, E) at zeroth and first order: W = box(phi), E = c (R^t_t + Phi_t^2/c^2) at kappa = 1.

    The first-order E is taken at fixed background c in the prefactor (the prefactor multiplies a row that vanishes
    on shell, so its variation does not matter there; off shell it is added by the caller where needed).
    """
    R0, R1, b0, b1, gi0, gi1 = rows(g0, h, U0, dU, m, phi0, dphi)
    Rtt0 = sum(gi0[0, b] * R0[b, 0] for b in range(3))
    Rtt1 = sum(gi0[0, b] * R1[b, 0] + gi1[0, b] * R0[b, 0] for b in range(3))
    # R^t_t = kappa g^{tb} d_b phi d_t phi
    T0 = sum(gi0[0, b] * sp.diff(phi0, X[b]) for b in range(3)) * sp.diff(phi0, t)
    T1 = (sum(gi1[0, b] * sp.diff(phi0, X[b]) + gi0[0, b] * sp.diff(dphi, X[b]) for b in range(3)) * sp.diff(phi0, t)
          + sum(gi0[0, b] * sp.diff(phi0, X[b]) for b in range(3)) * sp.diff(dphi, t))
    return (b0, b1, cc * (Rtt0 - T0), cc * (Rtt1 - T1), R0, R1)


def background(cc, mu, phi, nn=n):
    """Base metric, warps and fibre dimensions of the spherical background in the p = q frame."""
    g0 = sp.diag(-cc**2, (nn + mu) / (2 * s), 2 * nn * s * dchi_du**2)
    U0 = [sp.log(2 * nn * s) / 2 + sp.log(cosx), sp.log(2 * nn * s) / 2 + sp.log(sinx)]
    m = [(nn - 1) / 2, (nn - 1) / 2]
    return g0, U0, m


def k_pert(nn=n):
    """k = dy.dy - dz.dz in (s, chi), and its warps: k_NN = cos 2 chi."""
    h = sp.zeros(3, 3)
    h[1, 1] = nn * cos2 / (2 * s)
    h[1, 2] = h[2, 1] = -nn * sin2 * dchi_du
    h[2, 2] = -2 * nn * s * cos2 * dchi_du**2
    if MUT == 'k':  # a trace part: k is no longer traceless
        h[1, 1] = nn * (cos2 + sp.Rational(1, 10)) / (2 * s)
    dU = [sp.Rational(1, 2), -sp.Rational(1, 2)]
    return h, dU


# ---------------------------------------------------------------------------------------------------------------
# 1/n expansion and on-shell reduction
# ---------------------------------------------------------------------------------------------------------------
def atoms_to_symbols(expr):
    reps = {}
    for a in sorted(expr.atoms(sp.Derivative, sp.Function), key=str):
        if a.func in (sp.log,):
            continue
        reps[a] = sp.Dummy('a%d' % len(reps))
    return expr.xreplace(reps), {v: k for k, v in reps.items()}


def nseries(expr, order=1):
    """Coefficients of n^0, n^-1, ..., n^-order of an expression that is regular at n = infinity."""
    ex, back = atoms_to_symbols(sp.sympify(expr))
    ex = sp.cancel(sp.together(ex.subs(n, 1 / nu)))
    out = []
    cur = ex
    for j in range(order + 1):
        cj = sp.cancel(cur.subs(nu, 0))
        out.append(cj.xreplace(back))
        cur = sp.cancel((cur - cj) / nu)
    return out


def leading(expr, power=0):
    """Coefficient of n^power and of the next order down."""
    return nseries(sp.sympify(expr) / n**power, 1)


# leading rules, kappa = 1, with s as the evolution variable
def reduce_s(expr, extra=None):
    """Eliminates every s-derivative of c and Phi with c_s = Phi_t^2/c and Phi_s = (Phi_t/c)_t/c."""
    expr = sp.sympify(expr).doit()
    rc = sp.diff(PhF, t)**2 / cF
    rp = sp.diff(sp.diff(PhF, t) / cF, t) / cF
    for _ in range(12):
        ders = [d for d in expr.atoms(sp.Derivative) if s in d.variables and d.expr in (cF, PhF)]
        if not ders:
            break
        # lowest s-order first is not needed: replace each by differentiating the rule
        reps = {}
        for d in ders:
            ks = sum(1 for v in d.variables if v == s)
            kt = sum(1 for v in d.variables if v == t)
            rule = rc if d.expr == cF else rp
            val = rule
            if ks > 1:
                val = sp.diff(val, s, ks - 1)
            if kt:
                val = sp.diff(val, t, kt)
            reps[d] = val
        expr = expr.xreplace(reps).doit()
    return expr


def is_zero(expr, onshell=True):
    e2 = reduce_s(expr) if onshell else sp.sympify(expr).doit()
    e3, _ = atoms_to_symbols(e2)
    return sp.cancel(sp.together(e3)) == 0


def L0_rows(Lp, Pp):
    """The l = 0 leading rows on a lapse perturbation Lp and a scalar perturbation Pp (hand-linearised)."""
    c, P = cF, PhF
    W = (sp.diff(Pp, s) - sp.diff(Pp, t, 2) / c**2 + sp.diff(c, t) * sp.diff(Pp, t) / c**3
         + 2 * sp.diff(P, t, 2) * Lp / c**3 + sp.diff(P, t) * sp.diff(Lp, t) / c**3
         - 3 * sp.diff(c, t) * sp.diff(P, t) * Lp / c**4)
    E = -sp.diff(Lp, s) + 2 * sp.diff(P, t) * sp.diff(Pp, t) / c - sp.diff(P, t)**2 * Lp / c**2
    return W, E


def main():
    half = sp.Rational(1, 2)

    # ---- A: Schwarzschild-Tangherlini at n = 3, 4 and 7 ------------------------------------------------------
    r0 = sp.symbols('r0', positive=True)
    okA = True
    for nv in (sp.Integer(3), sp.Integer(4), sp.Integer(7)):
        f = 1 - r0 * (2 * nv * s)**(-((nv - 1) / 2))
        g0, U0, m = background(sp.sqrt(f), nv / f - nv, sp.Integer(0), nn=nv)
        R0, _, _, _, _, _ = rows(g0, sp.zeros(3, 3), U0, [0, 0], m, sp.Integer(0), sp.Integer(0))
        okA = okA and all(sp.simplify(R0[a, b]) == 0 for a in range(3) for b in range(a, 3))
    report('A  Tangherlini is Ricci flat in the p = q frame (R_ab on the base, n = 3, 4, 7)', okA)

    # ---- B: a diffeomorphism, off shell, exact in n ---------------------------------------------------------
    g0, U0, m = background(cF, muF, PhF)
    xi = [0, s * cos2, -half * sin2 / dchi_du]  # the linear map's generator: xi^s = s cos 2chi, xi^chi = -sin 2chi/2
    hL = sp.zeros(3, 3)
    for a in range(3):
        for b in range(3):
            hL[a, b] = sum(xi[d] * sp.diff(g0[a, b], X[d]) + g0[d, b] * sp.diff(xi[d], X[a])
                           + g0[a, d] * sp.diff(xi[d], X[b]) for d in range(3))
    dUL = [sum(xi[d] * sp.diff(U0[k], X[d]) for d in range(3)) for k in range(2)]
    dPL = sum(xi[d] * sp.diff(PhF, X[d]) for d in range(3))
    W0, W1, E0, E1, R0, R1 = two_rows(g0, hL, U0, dUL, m, PhF, dPL, cF)
    lieW = sum(xi[d] * sp.diff(W0, X[d]) for d in range(3))
    # E = c (R^t_t - T^t_t): the prefactor c is held fixed in E1, so compare with c xi(E0/c)
    lieE = cF * sum(xi[d] * sp.diff(E0 / cF, X[d]) for d in range(3))
    okB = is_zero(W1 - lieW, onshell=False) and is_zero(E1 - lieE, onshell=False)
    report('B  diffeomorphism: first-order rows = Lie derivative of the zeroth-order rows (off shell, exact n)', okB)
    # the diffeomorphism is the linear map x -> (1 + (e/2) diag(1, -1)) x: on flat space (mu = 0) its Lie derivative
    # of the spatial metric is k, and the warps shift by +-e/2
    hk, dUk = k_pert()
    gf = sp.diag(0, n / (2 * s), 2 * n * s * dchi_du**2)
    hF = sp.zeros(3, 3)
    for a in range(3):
        for b in range(3):
            hF[a, b] = sum(xi[d] * sp.diff(gf[a, b], X[d]) + gf[d, b] * sp.diff(xi[d], X[a])
                           + gf[a, d] * sp.diff(xi[d], X[b]) for d in range(3))
    okB2 = (sp.cancel(dUL[0] - dUk[0]) == 0 and sp.cancel(dUL[1] - dUk[1]) == 0
            and all(sp.cancel(hF[a, b] - hk[a, b]) == 0 for a in range(3) for b in range(3)))
    okB3 = sp.cancel(dPL - s * cos2 * sp.diff(PhF, s)) == 0
    report('B2 that diffeomorphism is the linear map: Lie(flat metric) = k, warps shift by +-e/2, Phi by s Y2 Phi_s',
           okB2 and okB3)

    # ---- C: the spherical equations to O(1/n) ---------------------------------------------------------------
    Z = sp.zeros(3, 3)
    W0, _, E0, _, R0, _ = two_rows(g0, Z, U0, [0, 0], m, PhF, sp.Integer(0), cF)
    c, P, mu = cF, PhF, muF
    cs, ct, Ps, Pt = sp.diff(c, s), sp.diff(c, t), sp.diff(P, s), sp.diff(P, t)
    # momentum row and the chi-chi row give mu
    Ets = leading(R0[0, 1] - Pt * Ps)
    Exx = leading((R0[2, 2] / (2 * n * s * dchi_du**2)), -1)
    mu_t_rule = sp.solve(Ets[0], sp.diff(mu, t))
    mu_rule = sp.solve(Exx[0], mu)
    okC0 = (len(mu_t_rule) == 1 and sp.cancel(mu_t_rule[0] - 4 * s * Ps * Pt) == 0
            and len(mu_rule) == 1 and sp.cancel(mu_rule[0] - 2 * s * cs / c) == 0)
    report('C0 background: mu = 2 s c_s/c from the chi-chi row, mu_t = 4 s Phi_s Phi_t from the t-s row', okC0,
           'mu = %s, mu_t = %s' % (mu_rule, mu_t_rule))
    Wn = leading(W0)
    En = leading(E0)
    # hand-derived (review note, section 1.2), mu still free
    W0h = Ps - sp.diff(Pt / c, t) / c
    W1h = Ps + 2 * s * sp.diff(P, s, 2) + 2 * s * Ps * cs / c - mu * Ps - sp.diff(mu, t) * Pt / (2 * c**2)
    E0h = -cs + Pt**2 / c
    E1h = -cs - 2 * s * sp.diff(c, s, 2) + mu * cs + sp.diff(sp.diff(mu, t) / c, t) / 2
    if MUT == 'bg':
        W1h = W1h + Ps
    okC = all(is_zero(a - b, onshell=False) for a, b in ((Wn[0], W0h), (Wn[1], W1h), (En[0], E0h), (En[1], E1h)))
    report('C  spherical rows to O(1/n) = the hand-derived ones (mu free, off shell)', okC)

    musub = {sp.diff(mu, t): 4 * s * Ps * Pt, sp.diff(mu, t, 2): sp.diff(4 * s * Ps * Pt, t),
             sp.diff(mu, s, t): sp.diff(4 * s * Ps * Pt, s)}

    def slave(expr):
        expr = sp.sympify(expr).doit().xreplace(musub)
        return expr.xreplace({mu: 2 * s * cs / c}).doit()

    W1m, E1m = slave(Wn[1]), slave(En[1])

    # FIRST_ORDER.md section 1's heat system through C = c/a, Y = 2 Phi_t/C, S = 2 Phi_s (kappa = 1), to O(1/n).
    # Its O(1/n) terms need the fields at leading order only, and its O(1) terms need a = 1 + mu/(2n).
    eps = sp.symbols('epsilon')  # 1/n
    a_ = 1 + eps * mu / 2
    Cc = c / a_
    Yy = 2 * Pt / Cc
    Ss = 2 * Ps
    FO_Y = sp.diff(Yy, t) - Cc * Ss - eps * Cc * (Ss + 2 * s * sp.diff(Ss, s) + half * s * Yy**2 * Ss)
    FO_C = sp.diff(Cc, s) - Cc * (Yy**2 / 4 + eps * (half * s * Ss**2 - half * Yy**2 - s * Yy * sp.diff(Yy, s)))

    def eps1(expr):
        e0 = expr.subs(eps, 0).doit()
        e1 = sp.diff(expr, eps).subs(eps, 0).doit()
        return e0, e1

    y0, y1_ = eps1(FO_Y)
    c0, c1_ = eps1(FO_C)
    # my rows: W = W0h + W1/n, E = E0h + E1/n. FO_Y = -(2c)(W) and FO_C = -E hold at O(1); at O(1/n) the two
    # differ by multiples of the leading rows, which vanish on shell.
    okC2 = (is_zero(y0 + 2 * c * W0h, onshell=False) and is_zero(c0 + E0h, onshell=False)
            and is_zero(slave(y1_) + 2 * c * W1m) and is_zero(slave(c1_) + E1m))
    report('C2 FIRST_ORDER.md section 1: FO_Y = -2c W and FO_C = -E to O(1/n) on the leading shell', okC2)

    # ---- D: the l = 2 residual ------------------------------------------------------------------------------
    hk, dUk = k_pert()
    # the mode is the diffeomorphism minus k: residual = -(rows on k)
    _, W1k, _, E1k, _, _ = two_rows(g0, -hk, U0, [-dUk[0], -dUk[1]], m, PhF, sp.Integer(0), cF)
    Wk = leading(W1k)
    Ek = leading(E1k)
    okD0 = is_zero(Wk[0], onshell=False) and is_zero(Ek[0], onshell=False)
    report('D0 l = 2: the O(1) parts of both rows vanish off shell', okD0)
    F2W, F2E = slave(Wk[1]), slave(Ek[1])
    # Q: the part of the O(1/n) spherical rows that multiplies 2s
    QW = sp.diff(P, s, 2) - Ps * Pt**2 / c**2
    QE = -(sp.diff(c, s, 2) - cs**2 / c - sp.diff(Ps * Pt / c, t))
    okD = is_zero(F2W - 2 * s * cos2 * QW) and is_zero(F2E - 2 * s * cos2 * QE)
    report('D  l = 2: F2 = 2 s Y2 Q in both rows (on the leading shell)', okD)
    okD1 = is_zero(W1m - (Ps + 2 * s * QW)) and is_zero(E1m - (-cs + 2 * s * QE))
    report('D1 the spherical O(1/n) rows are (Phi_s, -c_s) + 2 s Q', okD1)

    # ---- E: the identity ------------------------------------------------------------------------------------
    # F0 = n L_n[d_s Z_n]: the Gateaux derivative of the O(1/n) rows along d_s(c, Phi) minus their s-derivative.
    # Done on the rows as functionals of (c, Phi), mu slaved: replace c, Phi by c + h c_s, Phi + h Phi_s.
    hh = sp.symbols('h')

    def gateaux(expr, dc, dP):
        e2 = expr.subs({c: c + hh * dc, P: P + hh * dP}, simultaneous=True).doit()
        return sp.diff(e2, hh).subs(hh, 0).doit()

    F0W = gateaux(W1m, cs, Ps) - sp.diff(W1m, s)
    F0E = gateaux(E1m, cs, Ps) - sp.diff(E1m, s)
    okE0 = is_zero(F0W + 2 * QW, onshell=False) and is_zero(F0E + 2 * QE, onshell=False)
    report('E0 l = 0: F0 = -2 Q for the mode d_s(c, Phi) of the lapse c (off shell)', okE0)
    sg = -1 if MUT == 'sign' else 1
    okE = is_zero(F2W / (s * cos2) + sg * F0W) and is_zero(F2E / (s * cos2) + sg * F0E)
    report('E  identity: F2/(s Y2) + F0 = 0 in both rows, l = 0 mode d_s(c, Phi)', okE)
    # FIRST_ORDER.md's mode translates (C, Y, S), C = c/a. Its lapse part is c_s - c_s/n, so F0_FO = F0 + L0[(-c_s, 0)].
    LW1, LE1 = L0_rows(-cs, sp.Integer(0))
    LW2, LE2 = L0_rows(sp.Integer(0), Ps)
    okE1 = is_zero(LW1 - LW2) and is_zero(LE1 - LE2)
    report('E1 L0[(-c_s, 0)] = L0[(0, Phi_s)] on shell (their difference is L0 on the translation mode)', okE1)
    F0W_FO, F0E_FO = F0W + LW1, F0E + LE1
    # FIRST_ORDER.md section 4's forcing in the heat variables: -(d_s explicit) of the O(1/n) terms, i.e.
    # FO_Y row: -C (2 S_s + Y^2 S/2), FO_C row: -C (S^2/2 - Y Y_s); in my rows W = -FO_Y/(2c), E = -FO_C,
    # and n L_n[mode] in the FO rows is +(d_s explicit of what is subtracted) -> see the note, section 1.4.
    Y_, S_ = 2 * Pt / c, 2 * Ps
    FOfW = -(2 * sp.diff(S_, s) + half * Y_**2 * S_) / 2
    FOfE = -c * (half * S_**2 - Y_ * sp.diff(Y_, s))
    okE2 = is_zero(F0W_FO - FOfW) and is_zero(F0E_FO - FOfE)
    report("E2 the l = 0 forcing of FIRST_ORDER.md's mode is its section 4 forcing in these rows", okE2)
    okE3 = is_zero(F2W / (s * cos2) + sg * F0W_FO - LW2) and is_zero(F2E / (s * cos2) + sg * F0E_FO - LE2)
    report("E3 identity with FIRST_ORDER.md's l = 0 forcing: F2/(s Y2) + F0_FO = L0[(0, Phi_s)]", okE3)

    # ---- F: L2_MODE.md section 8.3's printed F[E^t_t] -------------------------------------------------------
    Ptt, Pttt, ctt = sp.diff(P, t, 2), sp.diff(P, t, 3), sp.diff(c, t, 2)
    F83 = (2 * s * cos2 / c**6) * (2 * c**2 * Pt**4 - c**2 * Pt * Pttt + c**2 * Ptt**2 + c * Pt**2 * ctt
                                   + c * Pt * Ptt * ct - 2 * Pt**2 * ct**2)
    okF = is_zero(F2E / c - F83)
    report('F  F2[E]/c = the F[E^t_t] printed in L2_MODE.md section 8.3', okF)

    # ---- G: conjugation -------------------------------------------------------------------------------------
    gam = sp.Function('gamma')(s, t)
    psi = sp.Function('psi')(s, t)
    for name, ang, want in (('G  conjugation by s Y2 at O(1)', cos2, True),
                            ('G1 control: cos^2 chi in place of cos 2chi must not conjugate', cosx**2, False)):
        Lp, Pp = s * ang * gam, s * ang * psi
        hl = sp.zeros(3, 3)
        hl[0, 0] = -2 * cF * Lp
        _, W1l, _, E1l, _, _ = two_rows(g0, hl, U0, [0, 0], m, PhF, Pp, cF)
        # E1 holds the prefactor c fixed; the variation of c in c (R^t_t - T) multiplies E0/c, zero on shell
        Wl, El = leading(W1l), leading(E1l)
        LW, LE = L0_rows(gam, psi)
        ok = is_zero(Wl[0] - s * ang * LW) and is_zero(El[0] - s * ang * LE)
        report(name, ok == want)

    # ---- H: slicing family ----------------------------------------------------------------------------------
    ep = sp.Function('varepsilon')(t)
    LW, LE = L0_rows(sp.diff(ep * c, t), ep * Pt)
    report('H  ((eps c)_t, eps Phi_t) is in the kernel of the l = 0 rows for every eps(t)', is_zero(LW) and is_zero(LE))

    # ---- I: which metric perturbations reach the two rows at O(1) -------------------------------------------
    Af = sp.Function('A')(s, t, u)
    names = []
    for label, hmat, dUs in (
            ('g_ss, relative order 1/n', {(1, 1): Af / (2 * s)}, [0, 0]),
            ('g_s chi, relative order 1/n', {(1, 2): Af * dchi_du}, [0, 0]),
            ('g_chi chi, relative order 1/n', {(2, 2): 2 * s * Af * dchi_du**2}, [0, 0]),
            ('warp V1 = -V2', {}, [Af / n, -Af / n]),
            ('warp V1 = V2', {}, [Af / n, Af / n])):
        hm = sp.zeros(3, 3)
        for (a, b), v in hmat.items():
            hm[a, b] = hm[b, a] = v
        _, W1a, _, E1a, _, _ = two_rows(g0, hm, U0, dUs, m, PhF, sp.Integer(0), cF)
        z = is_zero(leading(W1a)[0], onshell=False) and is_zero(leading(E1a)[0], onshell=False)
        names.append((label, z))
    okI = [z for _, z in names] == [True, True, True, True, False]
    report('I  at O(1) the two rows see only V1 + V2 among the metric perturbations of relative order 1/n', okI,
           '; '.join('%s: %s' % (l, 'absent' if z else 'present') for l, z in names))

    # ---- checks added after reading section 9 (the review note, section 2) -----------------------------------
    # B3: the Lie derivative of the background's metric along xi is the lapse part s Y2 c_s, plus k, plus the
    #     corrections A = (s mu_s + mu) Y2, B = -mu sin 2chi, C = 0 that L2_MODE.md 8.3 gives for the deformation
    dm = (hL - hF).applyfunc(sp.cancel)
    okB4 = (sp.cancel(dm[0, 0] + 2 * cF * s * cos2 * sp.diff(cF, s)) == 0
            and sp.cancel(dm[1, 1] - (s * sp.diff(muF, s) + muF) * cos2 / (2 * s)) == 0
            and sp.cancel(dm[1, 2] + muF * sin2 * dchi_du) == 0 and sp.cancel(dm[2, 2]) == 0
            and sp.cancel(dm[0, 1]) == 0 and sp.cancel(dm[0, 2]) == 0)
    report("B3 L2_MODE.md 8.3's deformation is exactly Lie_xi(background) - k: L = s Y2 c_s, A = (s mu_s + mu) Y2, "
           "B = -mu sin 2chi, C = V = 0", okB4)

    # J: every other row on k at its natural order: the mixed components E^a_b = g^{ac} E_cb and the fibre rows
    _, _, _, _, R0k, R1k = two_rows(g0, -hk, U0, [-dUk[0], -dUk[1]], m, PhF, sp.Integer(0), cF)
    fib = list(FIB)
    gi0 = g0.inv()
    okJ, worstJ = True, []
    for (a, b, pw, nm) in ((0, 1, 0, 'E^t_s'), (0, 2, 0, 'E^t_chi'), (1, 1, -1, 'E^s_s'), (1, 2, -1, 'E^s_chi'),
                           (2, 2, -1, 'E^chi_chi')):
        ex = gi0[a, a] * R1k[a, b]      # the background rows vanish on shell, so only g0^{aa} dE_ab is left
        co = leading(ex, pw + 1)        # coefficients of n^(pw+1) and n^pw: nothing above the natural order
        z = is_zero(slave(co[0])) and is_zero(slave(co[1]))
        okJ = okJ and z
        if not z:
            worstJ.append(nm)
    for kf in range(2):
        co = leading(fib[kf][1], 0)
        z = is_zero(slave(co[0])) and is_zero(slave(co[1]))
        okJ = okJ and z
        if not z:
            worstJ.append('F%d' % kf)
    report('J  every other row vanishes on k at its natural order (so k/n is a second kernel element of the '
           'leading problem)', okJ, 'nonzero: %s' % worstJ if worstJ else '')

    # J2: the spatial rows are evolution equations for the metric functions: C_tt is in the chi-chi row at O(1)
    Cf = sp.Function('Cm')(s, t, u)
    hm = sp.zeros(3, 3)
    hm[2, 2] = 2 * s * Cf * dchi_du**2
    _, _, _, _, _, R1c = two_rows(g0, hm, U0, [0, 0], m, PhF, sp.Integer(0), cF)
    lead_cc = leading(R1c[2, 2] / (2 * n * s * dchi_du**2), -1)[0]
    has_tt = sp.diff(Cf, t, 2) in sp.expand(lead_cc.doit()).atoms(sp.Derivative)
    report('J2 the chi-chi row at its natural order contains the second t-derivative of the metric function C', has_tt)

    # P: the p = 1 frame (y in R, z in R^n, one fibre S^(n-1)): k = e (dy^2 - dz.dz/n), k_NN = cos^2 chi - sin^2 chi/n
    g1 = sp.diag(-cF**2, (n + muF) / (2 * s), 2 * n * s * dchi_du**2)
    U1 = [sp.log(2 * n * s) / 2 + sp.log(sinx)]
    m1 = [n - 1]
    h1 = sp.zeros(3, 3)
    h1[1, 1] = n * (cosx**2 - sinx**2 / n) / (2 * s)
    h1[1, 2] = h1[2, 1] = -n * sinx * cosx * (1 + 1 / n) * dchi_du
    h1[2, 2] = 2 * n * s * (sinx**2 - cosx**2 / n) * dchi_du**2
    if MUT == 'k':
        h1[1, 1] = n * (cosx**2 - sinx**2 / n + sp.Rational(1, 10)) / (2 * s)
    W0p, W1p, E0p, E1p, _, _ = two_rows(g1, -h1, U1, [1 / (2 * n)], m1, PhF, sp.Integer(0), cF)
    okP0 = is_zero(W0p - W0, onshell=False) and is_zero(E0p - E0, onshell=False)
    Wp, Ep = leading(W1p), leading(E1p)
    okP1 = is_zero(Wp[0], onshell=False) and is_zero(Ep[0], onshell=False)
    okP2 = is_zero(slave(Wp[1]) - 2 * s * cosx**2 * QW) and is_zero(slave(Ep[1]) - 2 * s * cosx**2 * QE)
    report('P  p = 1 frame: the spherical rows are those of p = q exactly in n; on k the O(1) parts vanish and '
           'F2 = 2 s cos^2(chi) Q, so F2/(s Y2) is the p = q one', okP0 and okP1 and okP2)

    # K: the stored forcings of l2_c2_forcing.py (data/l2_c2_forcing.json) against this derivation, both rows
    import json
    import os
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'l2_c2_forcing.json')
    if os.path.exists(fn):
        d = json.load(open(fn))
        NPs = 8
        Pk = sp.symbols('P1:%d' % (NPs + 1))
        Ck = sp.symbols('C0:%d' % NPs)
        back = {Pk[i]: sp.diff(PhF, t, i + 1) for i in range(NPs)}
        back.update({Ck[i]: (sp.diff(cF, t, i) if i else cF) for i in range(NPs)})
        mine = {'l2': (F2W / (s * cos2), F2E / (cF * s * cos2)), 'l0': (F0W_FO, F0E_FO / cF), 'G': (LW2, LE2 / cF)}
        okK = True
        for nm, (mw, me) in mine.items():
            tw = sp.sympify(d[nm]['W']).xreplace(back)
            te = sp.sympify(d[nm]['E^t_t']).xreplace(back)
            okK = okK and is_zero(sg * mw - tw) and is_zero(sg * me - te)
        report('K  data/l2_c2_forcing.json: its l2, l0 and G rows equal this derivation', okK)

    print()
    if MUT:
        print('mutation %s: %d check(s) failed: %s' % (MUT, len(FAILS), ', '.join(f.split()[0] for f in FAILS)))
        sys.exit(1 if FAILS else 0)
    print('all passed' if not FAILS else '%d FAILED' % len(FAILS))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
