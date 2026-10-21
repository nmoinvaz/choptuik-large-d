#!/usr/bin/env python3
"""Validated integration (step ii of proof/A_certify.md) of the truncated M = 1 inner system, in ball arithmetic.

The system is the one solver/inner_stokes/lateral_np.py integrates (Dd, sigma on mu = +-1; c on nu = 0, +-2, every
product projected), 7 complex unknowns, here with omega and A carried as constant states (9 in all) so that parameter
boxes are propagated with their derivatives. The path is the one lateral_np uses: the ray xi = t e^{3 i pi/8},
t: r0 -> r1, then the arc xi = r1 e^{i(3 pi/8 - t)}, t: 0 -> 3 pi/8, to the real axis.

Method: a QR-Lohner Taylor integrator (Lohner 1987; Nedialkov-Jackson), written with python-flint acb / acb_series:
  * the set is X = c + Q q (c a point, Q a point matrix, q a box), so linear propagation does not wrap;
  * per step, an a-priori enclosure Y~ of all trajectories from X over [t0, t0 + h] (interval Picard, with the time
    as a ball), the Taylor polynomial of order N of the trajectory from c and of the variational matrix over X (both by
    Picard iteration on power series), and Lagrange-type remainders h^{N+1} coef_{N+1} evaluated on Y~ (for the
    variational part, Phi_{N+1}(Y~) V~ with V~ an a-priori enclosure of the variational matrix);
  * mean value form: phi(X) in phi(c) + J (X - c), J the enclosure of the step's Jacobian over X; then
    c' = mid phi(c), Q' = the Q of a QR factorization of mid(J Q) (columns pivoted by norm), q' = (Q'^-1 J Q) q
    + Q'^-1 (phi(c) - c').
The stiffness (fast modes decaying by about e^-21 to e^-26 along each leg) is harmless here: it only shrinks q in the
fast directions; nothing wraps by exp(int |lambda|) because no box is ever pushed through a Taylor polynomial.

  * C^1 (krawczyk mode): the parameter derivatives d(state)/d(A, wf) are carried as extra Lohner sets cV + QV qV and
    updated with the same enclosed step Jacobian J (valid over the whole state set), v' = J v; A and w are constant
    states, so J contains d/dw and the derivative columns start from (dy/dA, 0, 1) and (dy/dwf, W0, 0).

Start data. The krawczyk mode is fully rigorous: it starts from fixed_point.start_data (step (i), agent FK, proof/
A_certify.md 4.4-4.5): y+(xi0) at the centre to ~5e-32 and dy/d(A, wf) over the box, Cauchy-bounded. The older modes
(point, boxtest, miranda) start from the truncated series plus an assumed ball eps0 instead.

    python3 validated_int.py krawczyk h [A wf] [--xi1=0.3] [--procs=2] [--only=point|box] [--out=f.json]
                                         [--point=f.json] [--box=f.json]
        step (iv) of A_certify.md: point run at x^ (D+_{+-1}(xi1)), box run over x^ + [-h, h]^2 with the variational
        columns (dD+_{+-1}(xi1)/d(A, wf) over X), then krawczyk.normalization -> g_from_states -> krawczyk
    python3 validated_int.py linear              linear (no gravity) check against the exact Kummer-U jump (lin_check)
    python3 validated_int.py point [A wf eps0]   G = R_1 xi1^{-A^2} at one (A, w/w_inf), as a ball (xi1 = 0.2)
    python3 validated_int.py boxtest A wf dA dwf G over a parameter box in C^0 mean value form (xi1 = 0.2)
"""
import sys
import time
T0 = time.time()
import numpy as np
from flint import acb, arb, acb_mat, arb_mat, acb_series, ctx

PREC = 256
ctx.prec = PREC
I = acb(0, 1)
PI = arb.pi()
W0 = 2 * PI / arb("1.64389375")
PHI0 = 3 * PI / 8
ODD = (-1, 1)


def ball(x, r):
    """acb with real part ball(x, r) (x float or arb), imaginary part 0"""
    return acb(arb(x, r))


def pr(X, Y, out, zero):
    """projected product: {k: sum_{a+b=k} X_a Y_b} for k in out"""
    r = {}
    for k in out:
        acc = zero
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc = acc + xa * yb
        r[k] = acc
    return r


class Sys:
    """state: D[mu], S[mu] (mu = -1, 1), C[nu] (nu = -2, 0, 2) and w; packed as a list of 8 (7 complex + w)"""

    def __init__(self, gravity=True):
        self.grav = gravity
        self.EVEN = (-2, 0, 2) if gravity else (0,)
        self.keys = [("D", m) for m in ODD] + [("S", m) for m in ODD] + [("C", n) for n in self.EVEN]
        self.nd = len(self.keys)           # dynamic components
        self.n = self.nd + 2               # + w, A (constants; A enters only the start values and the jump)

    def unpack(self, Y):
        D = {m: Y[i] for i, m in enumerate(ODD)}
        S = {m: Y[2 + i] for i, m in enumerate(ODD)}
        C = {n: Y[4 + i] for i, n in enumerate(self.EVEN)}
        return D, S, C, Y[self.nd]

    def pack(self, D, S, C, w):
        return [D[m] for m in ODD] + [S[m] for m in ODD] + [C[n] for n in self.EVEN] + [w, w]

    def rhs(self, Y, xi, ixi, zero):
        """dY/dxi (w constant) and the value-level products needed by the linearization"""
        D, S, C, w = self.unpack(Y)
        ixi2 = ixi * ixi
        CS = pr(C, S, ODD, zero)
        dD = {m: (CS[m] - I * m * w * xi * D[m]) * ixi2 for m in ODD}
        if self.grav:
            E = {k: v / 2 for k, v in pr(D, D, self.EVEN, zero).items()}
            cE = pr(C, E, self.EVEN, zero)
            xdD = {m: xi * dD[m] for m in ODD}
            a, b, c = pr(C, xdD, ODD, zero), pr(cE, D, ODD, zero), pr(C, D, ODD, zero)
            r2 = {m: a[m] + b[m] - c[m] for m in ODD}
            dC = {n: cE[n] * ixi for n in self.EVEN}
        else:
            E = cE = None
            r2 = {m: xi * dD[m] - D[m] for m in ODD}
            dC = {0: zero}
        dS = {m: (r2[m] - I * m * w * xi * S[m]) * ixi2 for m in ODD}
        return self.pack(dD, dS, dC, zero), (D, S, C, w, E, cE, dD)

    def lin(self, cache, V, xi, ixi, zero):
        """derivative of rhs at the cached point in the direction V (8 components, V[-1] = dw)"""
        D, S, C, w, E, cE, dD = cache
        dD_, dS_, dC_, dw = self.unpack(V)
        ixi2 = ixi * ixi
        dCS = {m: a + b for (m, a), b in zip(pr(dC_, S, ODD, zero).items(), pr(C, dS_, ODD, zero).values())}
        ddD = {m: (dCS[m] - I * m * (dw * xi * D[m] + w * xi * dD_[m])) * ixi2 for m in ODD}
        if self.grav:
            dE = pr(D, dD_, self.EVEN, zero)
            p1, p2 = pr(dC_, E, self.EVEN, zero), pr(C, dE, self.EVEN, zero)
            dcE = {n: p1[n] + p2[n] for n in self.EVEN}
            xdD = {m: xi * dD[m] for m in ODD}
            xddD = {m: xi * ddD[m] for m in ODD}
            t = [pr(dC_, xdD, ODD, zero), pr(C, xddD, ODD, zero), pr(dcE, D, ODD, zero), pr(cE, dD_, ODD, zero),
                 pr(dC_, D, ODD, zero), pr(C, dD_, ODD, zero)]
            dr2 = {m: t[0][m] + t[1][m] + t[2][m] + t[3][m] - t[4][m] - t[5][m] for m in ODD}
            ddC = {n: dcE[n] * ixi for n in self.EVEN}
        else:
            dr2 = {m: xi * ddD[m] - dD_[m] for m in ODD}
            ddC = {0: zero}
        ddS = {m: (dr2[m] - I * m * (dw * xi * S[m] + w * xi * dS_[m])) * ixi2 for m in ODD}
        return self.pack(ddD, ddS, ddC, zero)


class Leg:
    """xi(t), t in [0, 1] exactly: 'ray' xi = (r0 + (r1 - r0) t) e^{i phi0}; 'arc' xi = r1 e^{i phi0 (1 - t)};
    'real' xi = r0 + (r1 - r0) t (the real axis)"""

    def __init__(self, kind, r0, r1):
        self.kind, self.r0, self.r1 = kind, arb(r0), arb(r1)
        self.u = (I * PHI0).exp() if kind == "ray" else acb(1)

    def xi(self, t):                       # t: acb (point or ball)
        if self.kind in ("ray", "real"):
            return (acb(self.r0) + acb(self.r1 - self.r0) * t) * self.u
        return acb(self.r1) * (I * PHI0 * (1 - t)).exp()

    def dxi(self, xi):
        return acb(self.r1 - self.r0) * self.u if self.kind in ("ray", "real") else -I * PHI0 * xi

    def series(self, t0, L):
        """xi(t0 + tau) and xi'(t0 + tau) as series in tau of length L"""
        x0 = self.xi(t0)
        if self.kind in ("ray", "real"):
            v = acb(self.r1 - self.r0) * self.u
            return acb_series([x0, v], prec=L), acb_series([v], prec=L)
        e = acb_series([0, -I * PHI0], prec=L).exp()      # e^{-i phi0 tau}
        xs = x0 * e
        return xs, -I * PHI0 * xs


def taylor_Y(sys_, leg, t0, Y0, N):
    """Taylor coefficients 0..N of the trajectory through Y0 at t0 (Picard on series); Y0 entries acb (maybe balls).
    Returns the list of series (length N + 1) and the value-level cache for the variational equation."""
    L = N + 1
    ctx.cap = L
    xs, dxs = leg.series(t0, L)
    ixs = 1 / xs
    zero = acb_series([0], prec=L)
    nd = sys_.nd
    Ys = [acb_series([y], prec=L) for y in Y0]
    for it in range(N + 1):
        F, cache = sys_.rhs(Ys, xs, ixs, zero)
        new = [acb_series([Y0[i]], prec=L) + (F[i] * dxs).integral() for i in range(nd)]
        Ys = new + Ys[nd:]
    F, cache = sys_.rhs(Ys, xs, ixs, zero)
    return Ys, (cache, xs, ixs, dxs, zero)


def taylor_V(sys_, ctxv, N, V0=None):
    """Taylor coefficients 0..N of the fundamental matrix V' = A(tau) V, V(0) = I, along the series trajectory in ctxv,
    by the exact coefficient recursion V_{k+1} = (1/(k+1)) sum_{j <= k} A_{k-j} V_j, with A(tau) = F_Y(Y(tau)) xi'(tau)
    built from the linearization applied to the unit directions. Returns the list of coefficient matrices V_0..V_N."""
    cache, xs, ixs, dxs, zero = ctxv
    L = N + 1
    ctx.cap = L
    n, nd = sys_.n, sys_.nd
    Acols = []
    for j in range(n):
        e = [acb_series([1 if i == j else 0], prec=L) for i in range(n)]
        F = sys_.lin(cache, e, xs, ixs, zero)
        Acols.append([(F[i] * dxs).coeffs() for i in range(nd)])
    Ak = []
    for k in range(N + 1):
        M = acb_mat(n, n)
        for j in range(n):
            for i in range(nd):
                cs = Acols[j][i]
                M[i, j] = cs[k] if k < len(cs) else acb(0)
        Ak.append(M)
    V = [acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])]
    for k in range(N):
        acc = Ak[k] * V[0]
        for j in range(1, k + 1):
            acc = acc + Ak[k - j] * V[j]
        V.append(acc * acb(arb(1) / (k + 1)))
    return V


def evalpoly(s, h, N):
    cs = s.coeffs()
    acc = acb(0)
    for k in range(min(N, len(cs) - 1), -1, -1):
        acc = acc * h + cs[k]
    return acc


def coef(s, k):
    cs = s.coeffs()
    return cs[k] if k < len(cs) else acb(0)


def inflate(z, fac, add):
    re, im = z.real, z.imag
    return acb(arb(re.mid(), re.rad() * fac + add), arb(im.mid(), im.rad() * fac + add))


def apriori(sys_, leg, t0, h, Xbox):
    """Y~ with Xbox + [0,h] F(Y~, xi(t0+[0,h])) xi' contained in Y~ (componentwise), or None"""
    T = acb(t0) + acb(arb(h / 2, h / 2))
    xi = leg.xi(T)
    dxi = leg.dxi(xi)
    ixi = 1 / xi
    H = acb(arb(h / 2, h / 2))
    zero = acb(0)
    nd = sys_.nd
    F, _ = sys_.rhs(Xbox, xi, ixi, zero)
    Yt = [Xbox[i] + H * F[i] * dxi for i in range(nd)] + Xbox[nd:]
    Yt = [inflate(y, 2.0, 1e-60) for y in Yt[:nd]] + Xbox[nd:]
    for it in range(30):
        F, _ = sys_.rhs(Yt, xi, ixi, zero)
        Z = [Xbox[i] + H * F[i] * dxi for i in range(nd)] + Xbox[nd:]
        if all(Yt[i].contains(Z[i]) for i in range(nd)):
            return Z, T
        Yt = [inflate(Yt[i].union(Z[i]), 1.5, 1e-60) for i in range(nd)] + Xbox[nd:]
    return None, T


def apriori_V(sys_, leg, T, Yt, h):
    """V~ (8x8 acb_mat) with I + [0,h] xi' F_Y(Y~) V~ contained in V~"""
    n = sys_.n
    xi = leg.xi(T)
    dxi = leg.dxi(xi)
    ixi = 1 / xi
    zero = acb(0)
    _, cache = sys_.rhs(Yt, xi, ixi, zero)
    Fy = acb_mat(n, n)
    for j in range(n):
        e = [acb(1) if i == j else acb(0) for i in range(n)]
        col = sys_.lin(cache, e, xi, ixi, zero)
        for i in range(sys_.nd):
            Fy[i, j] = col[i] * dxi
    H = acb(arb(h / 2, h / 2))
    Id = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    V = Id + H * (Fy * Id)
    V = acb_mat([[inflate(V[i, j], 2.0, 1e-60) for j in range(n)] for i in range(n)])
    for it in range(30):
        Z = Id + H * (Fy * V)
        if all(V[i, j].contains(Z[i, j]) for i in range(n) for j in range(n)):
            return Z
        V = acb_mat([[inflate(V[i, j].union(Z[i, j]), 1.5, 1e-60) for j in range(n)] for i in range(n)])
    return None


def qr_pivot(Am, wts=None):
    """orthogonal Q (numpy) from QR of A with columns sorted by decreasing norm * weight. With the weights = the
    sizes of the q components, Q's leading columns follow the largest axes of the set (Lohner's pivoting rule),
    which keeps the wrapping in (Q'^-1 J Q) q small where the flow is strongly non-normal (start of the arc)."""
    nrm = np.linalg.norm(Am, axis=0)
    if wts is not None:
        nrm = nrm * (np.asarray(wts) + 1e-300)
    order = np.argsort(-nrm)
    Q, R = np.linalg.qr(Am[:, order])
    return Q


def to_np(M):
    n, m = M.nrows(), M.ncols()
    return np.array([[complex(M[i, j].mid()) for j in range(m)] for i in range(n)])


def from_np(A):
    n, m = A.shape
    return acb_mat([[acb(complex(A[i, j])) for j in range(m)] for i in range(n)])


def realify(M):
    """complex n x m acb_mat -> the real 2n x 2m arb_mat of the same R-linear map: [[Re, -Im], [Im, Re]]"""
    n, m = M.nrows(), M.ncols()
    R = arb_mat(2 * n, 2 * m)
    for i in range(n):
        for j in range(m):
            a, b = M[i, j].real, M[i, j].imag
            R[i, j], R[i, m + j], R[n + i, j], R[n + i, m + j] = a, -b, b, a
    return R


def stack(M):
    """complex n x m acb_mat (columns = vectors) -> real 2n x m arb_mat [Re; Im]"""
    n, m = M.nrows(), M.ncols()
    R = arb_mat(2 * n, m)
    for i in range(n):
        for j in range(m):
            R[i, j], R[n + i, j] = M[i, j].real, M[i, j].imag
    return R


def unstack(R):
    n2, m = R.nrows(), R.ncols()
    n = n2 // 2
    return acb_mat([[acb(R[i, j], R[n + i, j]) for j in range(m)] for i in range(n)])


def to_np_r(M):
    return np.array([[float(M[i, j].mid()) for j in range(M.ncols())] for i in range(M.nrows())])


def from_np_r(A):
    return arb_mat([[arb(float(A[i, j])) for j in range(A.shape[1])] for i in range(A.shape[0])])


class Lohner:
    """QR-Lohner set c + Q q in REAL coordinates: c a complex point (n), Q a real 2n x 2n matrix, q a real box (2n),
    the state being c + (Q q)[:n] + i (Q q)[n:]. Real coordinates matter: a complex ball is a rectangle, and every
    product with a complex number of non-zero phase inflates it (by up to sqrt 2); on the arc, where the solution
    rotates, that wrapping grew the enclosures by about e^18 at xi1 = 0.3. Real Q absorbs the rotations exactly.
    The constructor accepts a complex (Q, q) (acb_mat, list of acb) and converts it."""

    def __init__(self, sys_, c, Q, q, N=24, var=None):
        self.s, self.c, self.N = sys_, c, N
        if isinstance(Q, acb_mat):
            qs = stack(acb_mat([[x] for x in q]))
            Q, q = realify(Q), [qs[i, 0] for i in range(2 * len(q))]
        self.Q, self.q = Q, q                      # Q: arb_mat 2n x 2n; q: list of 2n arb
        # optional C^1 part: columns V of d(state)/d(parameters), as a Lohner set cV + QV qV (cV: n x m acb_mat of
        # points; QV: real 2n x 2n; qV: real 2n x m)
        if var is not None and isinstance(var[1], acb_mat):
            cV, QV, qV = var
            var = (cV, realify(QV), stack(qV))
        self.var = var
        # dx: parameter offsets x - x^ over the box (list of acb). If set, (c, Q, q) is the set of the single
        # trajectory from x^ and the state set over the box is c + Qq + V dx (mean value theorem with the enclosed
        # derivative columns V): no second-order wrapping in the box directions.
        self.dx = None

    def var_box(self):
        cV, QV, qV = self.var
        return cV + unstack(QV * qV)

    def box(self):
        n = self.s.n
        Qq = self.Q * arb_mat([[x] for x in self.q])
        X = [self.c[i] + acb(Qq[i, 0], Qq[n + i, 0]) for i in range(n)]
        if self.dx is not None:
            V = self.var_box()
            X = [X[i] + sum((V[i, j] * self.dx[j] for j in range(len(self.dx))), acb(0)) for i in range(n)]
        return X

    def step(self, leg, t0, h):
        s, N, n = self.s, self.N, self.s.n
        X = self.box()
        Yt, T = apriori(s, leg, t0, h, X)
        if Yt is None:
            return False
        Vt = apriori_V(s, leg, T, Yt, h)
        if Vt is None:
            return False
        hh = acb(h)
        hN1 = hh ** (N + 1)
        # point trajectory and its remainder
        Yc, _ = taylor_Y(s, leg, acb(t0), self.c, N)
        Yr, ctr = taylor_Y(s, leg, T, Yt, N + 1)
        nd = s.nd
        phic = [evalpoly(Yc[i], hh, N) + hN1 * coef(Yr[i], N + 1) for i in range(nd)] + self.c[nd:]
        # Jacobian over X: Taylor of V along the trajectory from the box, plus remainder Phi_{N+1}(Y~) V~
        _, ctxX = taylor_Y(s, leg, acb(t0), X, N)
        Vk = taylor_V(s, ctxX, N)
        VR = taylor_V(s, ctr, N + 1)
        J = Vk[N]
        for k in range(N - 1, -1, -1):
            J = J * hh + Vk[k]
        PhiN1 = VR[N + 1]
        J = J + hN1 * (PhiN1 * Vt)
        JR = realify(J)
        # Lohner update, real coordinates
        cnew = [acb(v.mid()) for v in phic]
        e = stack(acb_mat([[phic[i] - cnew[i]] for i in range(n)]))
        A = JR * self.Q
        Qn = from_np_r(qr_pivot(to_np_r(A), [float(abs(x).upper()) for x in self.q]))
        Qi = Qn.inv()
        qn = (Qi * A) * arb_mat([[x] for x in self.q]) + Qi * e
        if self.var is not None:
            # C^1: every derivative column v of a trajectory in X obeys v' = Dphi v with Dphi in J (J encloses the
            # step Jacobian over the whole set X), so v_new in J cV + (J QV) qV; same QR-Lohner update, no remainder
            cV, QV, qV = self.var
            m = cV.ncols()
            JC = J * cV
            cVn = acb_mat([[acb(JC[i, j].mid()) for j in range(m)] for i in range(n)])
            AV = JR * QV
            QVn = from_np_r(qr_pivot(to_np_r(AV), [max(float(abs(qV[i, j]).upper()) for j in range(m))
                                                   for i in range(2 * n)]))
            QVi = QVn.inv()
            self.var = (cVn, QVn, (QVi * AV) * qV + QVi * stack(JC - cVn))
        self.c, self.Q, self.q = cnew, Qn, [qn[i, 0] for i in range(2 * n)]
        return True

    def run(self, leg, h0, verbose=False):
        """integrate t: 0 -> 1 exactly; steps are multiples of 2^-30 so every t is exact in binary"""
        q = 2.0 ** -30
        hmax = getattr(self, "hmax", 1.0)
        t, h, steps = 0.0, min(h0, hmax), 0
        while t < 1.0:
            hs = min(max(round(h / q), 1) * q, 1.0 - t)
            if not self.step(leg, t, hs):
                h = hs / 2
                if h < 1e-9:
                    raise RuntimeError(f"step size underflow at t = {t}")
                continue
            t = t + hs                    # exact: both multiples of 2^-30, below 1
            steps += 1
            h = min(hs * 1.25, hmax)
            if verbose and steps % 20 == 0:
                X = self.box()
                rad = max(max(float(x.real.rad()), float(x.imag.rad())) for x in X[:self.s.nd])
                print(f"   {leg.kind} t {t:.6f} h {hs:.2e}  max state radius {rad:.2e}  [{time.time() - T0:.0f} s]", flush=True)
        return steps


# ---------------------------------------------------------------- start values, path, jump

def series_generic(A, w, J):
    """the formal series recursion of series_arb.SeriesArb (M = 1, gravity), for any number type: acb, or acb_series
    'jets' in a dummy variable (then the coefficients carry derivatives with respect to A or w)"""
    zero, one = A * 0, A * 0 + 1
    odd, even = ODD, (-2, 0, 2)
    Fh = {1: A * acb(0, -0.5), -1: A * acb(0, 0.5)}

    def mul(X, Y, keep):
        out = {k: zero for k in keep}
        for i, xi in X.items():
            for j, yj in Y.items():
                if i + j in out:
                    out[i + j] = out[i + j] + xi * yj
        return out
    d = [{mu: 2 * Fh[mu] for mu in odd}]
    sl, g = [], [{nu: (one if nu == 0 else zero) for nu in even}]
    e = [{nu: zero for nu in even}]
    ce = [{nu: zero for nu in even}]
    for j in range(J):
        sj = {mu: (2 * j + 1 + I * mu * w) * d[j][mu] for mu in odd}
        for i in range(1, j + 1):
            t = mul(g[i], sl[j - i], odd)
            for mu in odd:
                sj[mu] = sj[mu] - t[mu]
        sl.append(sj)
        m = j + 1
        em = {nu: zero for nu in even}
        for i in range(m):
            t = mul(d[i], d[m - 1 - i], even)
            for nu in even:
                em[nu] = em[nu] + t[nu] / 2
        e.append(em)
        cm = {nu: zero for nu in even}
        for i in range(m):
            t = mul(g[i], e[m - i], even)
            for nu in even:
                cm[nu] = cm[nu] + t[nu]
        ce.append(cm)
        g.append({nu: cm[nu] / (2 * m) for nu in even})
        num = {mu: (2 * j + 2 + I * mu * w) * sl[j][mu] for mu in odd}
        for i in range(1, j + 2):
            t = mul(g[i], d[j + 1 - i], odd)
            for mu in odd:
                num[mu] = num[mu] - 2 * (j + 1 - i) * t[mu]
        for mm in range(1, j + 2):
            t = mul(ce[mm], d[j + 1 - mm], odd)
            for mu in odd:
                num[mu] = num[mu] - t[mu]
        d.append({mu: num[mu] / (2 * j + 2) for mu in odd})
    return d, sl, g


def series_value(A, w, r0, n):
    """the first n terms of the formal series at xi0 = r0 e^{3 i pi/8}: [D_-1, D_1, S_-1, S_1, C_-2, C_0, C_2]"""
    d, sl, g = series_generic(A, w, n)
    xi0 = acb(arb(r0)) * (I * PHI0).exp()
    x2 = xi0 * xi0
    D, S, C = {}, {}, {}
    for mu in ODD:
        acc_d, acc_s, p = A * 0, A * 0, xi0
        for j in range(n):
            acc_d = acc_d + d[j][mu] * p
            acc_s = acc_s + sl[j][mu] * p * xi0
            p = p * x2
        D[mu], S[mu] = acc_d, acc_s
    for nu in (-2, 0, 2):
        acc, p = A * 0, acb(1)
        for j in range(n):
            acc = acc + g[j][nu] * p
            p = p * x2
        C[nu] = acc
    return [D[m] for m in ODD] + [S[m] for m in ODD] + [C[k] for k in (-2, 0, 2)]


def optimal_n(A, w, r0):
    J = int(1 / (2 * float(r0) ** 2)) + 30
    d, _, _ = series_generic(acb(A), acb(w), J)
    x2 = arb(r0) ** 2
    mags = [max(abs(d[j][mu]) for mu in ODD) * x2 ** j for j in range(J)]
    return min(range(1, J), key=lambda j: float(mags[j].mid()))


def start_lohner(sys_, A, wf, r0, eps0, N=24, n=None):
    """Lohner set at the start point for the parameter box A (arb ball), wf (arb ball), w = W0 wf:
    c = series at the box midpoint; Q columns: identity (residual), d/dw and d/dA of the series (midpoints);
    q = [residual box incl. eps0 (the assumed step-(i) enclosure radius) and second-order terms, dw, dA]"""
    Am, wm = arb(A.mid()), W0 * arb(wf.mid())
    wbox = W0 * wf
    n = n or optimal_n(Am, wm, r0)
    Y0 = series_value(acb(Am), acb(wm), r0, n)
    nd = sys_.nd
    c = [acb(y.mid()) for y in Y0] + [acb(wm.mid()), acb(Am.mid())]
    dA, dw = A - Am, wbox - acb(wm.mid()).real
    epsb = acb(arb(0, eps0), arb(0, eps0))
    res = [Y0[k] - c[k] + epsb for k in range(nd)]
    cols = {}
    if A.rad() > 0 or wf.rad() > 0:
        ctx.cap = 2
        jA = series_value(acb_series([acb(A), 1], prec=2), acb(wbox), r0, n)
        jw = series_value(acb(A), acb_series([acb(wbox), 1], prec=2), r0, n)
        dYdA = [coef(v, 1) for v in jA]
        dYdw = [coef(v, 1) for v in jw]
        mA = [acb(v.mid()) for v in dYdA]
        mw = [acb(v.mid()) for v in dYdw]
        res = [res[k] + (dYdA[k] - mA[k]) * acb(dA) + (dYdw[k] - mw[k]) * acb(dw) for k in range(nd)]
        # the midpoint value Y0(Am, wm) differs from c only by rounding; the mean value form uses the derivatives over
        # the whole box, so Y0(p) in Y0(pm) + dYdA(box) dA + dYdw(box) dw
    else:
        mA = mw = [acb(0)] * nd
    n_ = sys_.n
    Q = acb_mat(n_, n_)
    for i in range(n_):
        for j in range(n_):
            if j < nd:
                Q[i, j] = acb(1) if i == j else acb(0)
            elif j == nd:                     # w direction
                Q[i, j] = mw[i] if i < nd else (acb(1) if i == nd else acb(0))
            else:                             # A direction
                Q[i, j] = mA[i] if i < nd else (acb(1) if i == nd + 1 else acb(0))
    q = res + [acb(dw), acb(dA)]
    return Lohner(sys_, c, Q, q, N), n


def fk_start(A, wf, r0="0.1", cauchy="1e-3"):
    """step (i) start data from fixed_point.start_data (agent FK, proof/A_certify.md 4.4-4.5), rigorous: values and
    d/dA, d/dwf at xi0 = r0 e^{3 i pi/8}, valid over the real box (A, wf), state order [D-1, D+1, S-1, S+1, C-2, C0, C2]"""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixed_point as FP
    ctx.prec = 320
    _, vals, dAs, dws, eta = FP.start_data(A, wf, r0, +1, arb(cauchy))
    ctx.prec = PREC
    return vals, dAs, dws, eta


def start_lohner_fk(sys_, Ahat, wfhat, A, wf, r0="0.1", var=False, N=24, cauchy="1e-3", centre_set=False):
    """Lohner set at xi0 containing (y+(xi0; x), w(x), A(x)) for every x = (A, wf) in the real box X (A, wf balls that
    contain the centre (Ahat, wfhat)), in mean value form:
        y(x) in y(x^) + dy/dA(X) (A - A^) + dy/dwf(X) (wf - wf^)
    c = mid y(x^); Q columns: identity, then mid dy/dw and mid dy/dA (w = W0 wf); q = [residual, w - c_w, A - c_A].
    With var=True also the C^1 part: the columns d(state)/dA, d(state)/dwf over X (rows w, A: (0, W0), (1, 0))."""
    nd, n = sys_.nd, sys_.n
    yc, _, _, etac = fk_start(Ahat, wfhat, r0, cauchy)
    if A.rad() > 0 or wf.rad() > 0:
        _, dAX, dwX, etaX = fk_start(A, wf, r0, cauchy)
    else:
        _, dAX, dwX, etaX = fk_start(Ahat, wfhat, r0, cauchy)
    cY = [acb(y.mid()) for y in yc]
    cw, cA = acb((W0 * wfhat).mid()), acb(Ahat.mid())
    mA = [acb(v.mid()) for v in dAX]
    mw = [acb((v / W0).mid()) for v in dwX]
    dA_, dwf_ = acb(A - Ahat), acb(wf - wfhat)
    res = [yc[k] - cY[k] + (dAX[k] - mA[k]) * dA_ - mA[k] * (acb(Ahat) - cA)
           + (dwX[k] - acb(W0) * mw[k]) * dwf_ - mw[k] * (acb(W0 * wfhat) - cw) for k in range(nd)]
    Q = acb_mat(n, n)
    for i in range(n):
        for j in range(n):
            if j < nd:
                Q[i, j] = acb(1) if i == j else acb(0)
            elif j == nd:
                Q[i, j] = mw[i] if i < nd else (acb(1) if i == nd else acb(0))
            else:
                Q[i, j] = mA[i] if i < nd else (acb(1) if i == nd + 1 else acb(0))
    q = res + [acb(W0 * wf) - cw, acb(A) - cA]
    if centre_set:                       # the set of the trajectory from x^ only; the box enters through V dx
        res0 = [yc[k] - cY[k] - mA[k] * (acb(Ahat) - cA) - mw[k] * (acb(W0 * wfhat) - cw) for k in range(nd)]
        q = res0 + [acb(W0 * wfhat) - cw, acb(Ahat) - cA]
    vs = None
    if var:
        V0 = acb_mat(n, 2)
        for i in range(nd):
            V0[i, 0], V0[i, 1] = dAX[i], dwX[i]
        V0[nd, 0], V0[nd, 1] = acb(0), acb(W0)
        V0[nd + 1, 0], V0[nd + 1, 1] = acb(1), acb(0)
        cV = acb_mat([[acb(V0[i, j].mid()) for j in range(2)] for i in range(n)])
        Id = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
        vs = (cV, Id, V0 - cV)
    lo = Lohner(sys_, [*cY, cw, cA], Q, q, N, vs)
    if centre_set:
        lo.dx = [dA_, dwf_]
    return lo, (etac, etaX)


def rad_of(z):
    return max(float(z.real.rad()), float(z.imag.rad()))


def kpoint_job(args):
    """point run at the centre x^ for the Krawczyk test: enclosures of D+_{-1}(xi1), D+_{+1}(xi1)"""
    global PATH_RHO
    As, wfs, r0, r1, verbose, PATH_RHO = args
    ctx.prec = PREC
    t0 = time.time()
    s = Sys(gravity=True)
    ctx.prec = 320
    Ahat, wfhat = arb(As), arb(wfs)
    ctx.prec = PREC
    lo, eta = start_lohner_fk(s, Ahat, wfhat, Ahat, wfhat, r0)
    n1, n2 = integrate_lohner(lo, r0, r1, verbose)
    X = lo.box()
    return {"Dm1": [X[0].real.str(600, radius=True, more=True), X[0].imag.str(600, radius=True, more=True)],
            "Dp1": [X[1].real.str(600, radius=True, more=True), X[1].imag.str(600, radius=True, more=True)],
            "rad": [rad_of(X[0]), rad_of(X[1])], "state_rad": max(rad_of(x) for x in X[:s.nd]),
            "steps": n1 + n2, "sec": time.time() - t0, "eta0": float(eta[0][0].upper())}


def kbox_job(args):
    """box run over X = x^ + [-h, h]^2 with the variational columns: enclosures of dD+_{+-1}(xi1)/d(A, wf) over X"""
    global PATH_RHO
    As, wfs, h, r0, r1, verbose, PATH_RHO = args
    t0 = time.time()
    s = Sys(gravity=True)
    ctx.prec = 320
    Ahat, wfhat = arb(As), arb(wfs)
    A, wf = Ahat + arb(0, arb(h)), wfhat + arb(0, arb(h))
    ctx.prec = PREC
    lo, eta = start_lohner_fk(s, Ahat, wfhat, A, wf, r0, var=True, centre_set=True)
    V0 = lo.var_box()
    n1, n2 = integrate_lohner(lo, r0, r1, verbose)
    V = lo.var_box()
    X = lo.box()
    out = {}
    for name, i in (("dDm1", 0), ("dDp1", 1)):
        out[name] = [[V[i, j].real.str(600, radius=True, more=True), V[i, j].imag.str(600, radius=True, more=True)] for j in range(2)]
        out[name + "_rad"] = [rad_of(V[i, j]) for j in range(2)]
    out.update({"V0_rad": max(rad_of(V0[i, j]) for i in range(s.nd) for j in range(2)),
                "state_rad": max(rad_of(x) for x in X[:s.nd]), "steps": n1 + n2, "sec": time.time() - t0,
                "etaX": float(eta[1][0].upper()), "etaX_der": float(eta[1][1].upper())})
    return out


def parse_acb(p):
    return acb(arb(p[0]), arb(p[1]))


def krawczyk_assemble(As, wfs, h, r1, pt, bx):
    """K(X) of krawczyk.py from the point run (pt) and the box run (bx)"""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import krawczyk as KR
    ctx.prec = 320
    xhat = [arb(As), arb(wfs)]
    Dm1, Dp1 = parse_acb(pt["Dm1"]), parse_acb(pt["Dp1"])
    dDm1 = [parse_acb(p) for p in bx["dDm1"]]
    dDp1 = [parse_acb(p) for p in bx["dDp1"]]
    N, xs = KR.normalization(xhat, r1, prec=320)
    g_hat, JX = KR.g_from_states(Dm1, Dp1, dDm1, dDp1, N, xs)
    ok, K, margins = KR.krawczyk(xhat, g_hat, JX, [arb(h), arb(h)])
    JD = Dp1 - Dm1.conjugate()
    dJD = [dDp1[k] - dDm1[k].conjugate() for k in range(2)]
    return ok, K, margins, g_hat, JX, JD, dJD


PATH_RHO = None          # if set (e.g. "0.2"): ray to rho, arc at rho, then the real axis from rho to xi1


def integrate_lohner(lo, r0, r1, verbose=False):
    if PATH_RHO is None:
        n1 = lo.run(Leg("ray", r0, r1), 1e-3, verbose)
        n2 = lo.run(Leg("arc", r0, r1), 1e-3, verbose)
        return n1, n2
    n1 = lo.run(Leg("ray", r0, PATH_RHO), 1e-3, verbose)
    n2 = lo.run(Leg("arc", r0, PATH_RHO), 1e-3, verbose)
    n3 = lo.run(Leg("real", PATH_RHO, r1), 1e-3, verbose)
    return n1, n2 + n3


def kappa(A, w, r1):
    """G = J_1 kappa, kappa = r1^{-A^2} / ref, ref = 2 pi i K'(s) f(r1) 2 F_1 (lin_check), F_1 = -i A / 2"""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lin_check
    _, _, _, ref = lin_check.jump_exact(w, arb(r1), mu=1, Fmu=acb(0, -A / 2) if isinstance(A, arb) else A * acb(0, -0.5))
    return acb(arb(r1)) ** (-(A * A)) / ref


def H_enclosure(lo, r1, P):
    """H = P [Re G, Im G] over the final Lohner set c + Q q, in mean value form (first order in q kept linear)"""
    sys_ = lo.s
    nd, n = sys_.nd, sys_.n
    c, Q, q = lo.c, lo.Q, lo.q
    X = lo.box()
    i_m1, i_p1 = ODD.index(-1), ODD.index(1)
    Abox, wbox = X[nd + 1].real, X[nd].real
    kap_box = kappa(Abox, wbox, r1)
    kap_mid = kappa(c[nd + 1].real, c[nd].real, r1)
    J0c = c[i_p1] - c[i_m1].conjugate()
    G0 = J0c * kap_mid
    Pm = [[arb(P[0][0]), arb(P[0][1])], [arb(P[1][0]), arb(P[1][1])]]

    def app(re, im):
        return [Pm[0][0] * re + Pm[0][1] * im, Pm[1][0] * re + Pm[1][1] * im]
    H = app(G0.real, G0.imag)
    # J0c (kappa(q) - kappa(c)): kappa over the box
    t = J0c * (kap_box - kap_mid)
    tt = app(t.real, t.imag)
    H = [H[0] + tt[0], H[1] + tt[1]]
    # linear part: sum_k [z_{+1,k} - conj(z_{-1,k})] q_k kappa_box, z_{i,k} = Q[i,k] + i Q[n+i,k], q_k real
    for k in range(2 * n):
        zp, zm = acb(Q[i_p1, k], Q[n + i_p1, k]), acb(Q[i_m1, k], Q[n + i_m1, k])
        al = (zp - zm.conjugate()) * kap_box
        tt = app(al.real * q[k], al.imag * q[k])
        H = [H[0] + tt[0], H[1] + tt[1]]
    return H, G0


def jump_ball(sys_, lo, A, w, r1):
    """J_1 = D_1 - conj(D_-1) from the final box; R_1 = J_1 / ref; G = R_1 r1^{-A^2}"""
    X = lo.box()
    D = {m: X[i] for i, m in enumerate(ODD)}
    J1 = D[1] - D[-1].conjugate()
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lin_check
    _, _, _, ref = lin_check.jump_exact(w, arb(r1), mu=1, Fmu=acb(0, -A / 2))
    R1 = J1 / ref
    G = R1 * acb(arb(r1)) ** (-(A * A))
    return J1, ref, R1, G


def linear_start(sys_, A, w, r0, eps0):
    """linear (no gravity) start: Dd = 2F xi 2F0(a,b;;2xi^2) per harmonic, sigma = xi^2 Dd' + i mu w xi Dd, c = 1,
    optimally truncated, plus eps0"""
    Fh = {1: acb(0, -A / 2), -1: acb(0, A / 2)}
    xi0 = acb(arb(r0)) * (I * PHI0).exp()
    Y, Sv = [], {}
    for m in ODD:
        s_ = I * m * w
        a, b = (1 + s_) / 2, (2 + s_) / 2
        z = 2 * xi0 * xi0
        terms, t = [], acb(1)
        for k in range(400):
            terms.append(t)
            t = t * (a + k) * (b + k) / (k + 1) * z
        kt = min(range(1, 400), key=lambda k: float(abs(terms[k]).mid()))
        P = sum(terms[:kt], acb(0))
        dP = sum((terms[k] * 2 * k / xi0 for k in range(1, kt)), acb(0))
        Dd = 2 * Fh[m] * xi0 * P
        Y.append(Dd)
        Sv[m] = xi0 * xi0 * 2 * Fh[m] * (P + xi0 * dP) + I * m * w * xi0 * Dd
    Y += [Sv[m] for m in ODD] + [acb(1)]
    epsb = acb(arb(0, eps0), arb(0, eps0))
    Y = [y + epsb for y in Y[:-1]] + [Y[-1]] + [acb(w), acb(A)]
    n = sys_.n
    c = [acb(y.mid()) for y in Y]
    q = [Y[k] - c[k] for k in range(n)]
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    return Lohner(sys_, c, Q, q, 24)


def precond(A, wf, r1, hfd=1e-6):
    """J0^{-1} for H = J0^{-1} G, from the double-precision lateral solver (not rigorous; any matrix is allowed)"""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_stokes"))
    from lateral_np import jump, Fharm, W0 as W0f

    def G(a, f):
        R, *_ = jump(1, W0f * f, Fharm(a), r1=r1)
        z = R[1] * r1 ** (-a * a)
        return np.array([z.real, z.imag])
    g0 = G(A, wf)
    J0 = np.array([(G(A + hfd, wf) - g0) / hfd, (G(A, wf + hfd) - g0) / hfd]).T
    return np.linalg.inv(J0), J0, g0


def face_job(args):
    """one Poincare-Miranda face: (index i, sign, A*, wf*, h, eps0, P) -> H enclosure over the face"""
    i, sgn, As, wfs, h, eps0, P, r0, r1 = args
    ctx.prec = PREC
    if i == 0:
        A, wf = arb(As) + sgn * arb(h), arb(wfs, h)
    else:
        A, wf = arb(As, h), arb(wfs) + sgn * arb(h)
    s = Sys(gravity=True)
    t0 = time.time()
    lo, n = start_lohner(s, A, wf, r0, eps0)
    n1, n2 = integrate_lohner(lo, r0, r1)
    H, G0 = H_enclosure(lo, float(r1), P)
    return i, sgn, [x.str(6, radius=True) for x in H], [float(x.lower()) for x in H], [float(x.upper()) for x in H], \
        n1 + n2, time.time() - t0


def main_krawczyk(argv):
    """validated_int.py krawczyk h [A wf] [--xi1=0.3] [--procs=2] [--out=f.json] [--only=point|box] [--sub=k]
    [--point=f.json] [--box=f.json]. With --sub=k the box X is covered by k x k sub-boxes of half-width h/k, one box run
    each (the derivative excess grows with the box), and J(X) is the union of their enclosures."""
    import json
    import multiprocessing as mpr
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    h = pos[0]
    As, wfs = (pos[1], pos[2]) if len(pos) > 2 else ("2.5421480475", "0.7769940136")
    r0, r1 = "0.1", opts.get("xi1", "0.3")
    procs = int(opts.get("procs", "2"))
    only = opts.get("only")
    rho = opts.get("rho")                        # path: ray to rho, arc at rho, real axis to xi1 (default: ray to xi1)
    loaded = {}
    for key in ("point", "box"):                 # --point=file.json / --box=file.json: reuse a saved run
        if key in opts:
            with open(opts[key]) as fh:
                d = json.load(fh)
            assert d["A"] == As and d["wf"] == wfs and d["xi1"] == r1 and (key == "point" or d["h"] == h)
            assert d.get("rho") == rho or key == "point"
            loaded[key] = d[key]
    jobs = []
    if only in (None, "point") and "point" not in loaded:
        jobs.append((kpoint_job, (As, wfs, r0, r1, only == "point", rho)))
    sub = int(opts.get("sub", "1"))
    if only in (None, "box") and "box" not in loaded:
        if sub == 1:
            jobs.append((kbox_job, (As, wfs, h, r0, r1, only == "box", rho)))
        else:
            from decimal import Decimal as Dec, getcontext
            getcontext().prec = 50
            hs = Dec(h) / sub
            for i in range(sub):
                for j in range(sub):
                    a = Dec(As) + hs * (2 * i + 1 - sub)
                    b = Dec(wfs) + hs * (2 * j + 1 - sub)
                    jobs.append((kbox_job, (str(a), str(b), str(hs), r0, r1, False, rho)))
    if procs > 1 and len(jobs) > 1:
        with mpr.get_context("spawn").Pool(min(procs, len(jobs))) as pool:
            res = [pool.apply_async(f, (a,)) for f, a in jobs]
            res = [r.get() for r in res]
    else:
        res = [f(a) for f, a in jobs]
    out = {"A": As, "wf": wfs, "h": h, "xi0": "0.1 e^{3 i pi/8}", "xi1": r1, "rho": rho}
    out.update(loaded)
    subs = []
    for f_a, r in zip(jobs, res):
        if f_a[0] is kpoint_job:
            out["point"] = r
        else:
            subs.append(r)
    if len(subs) == 1:
        out["box"] = subs[0]
    elif subs:                                   # union of the sub-box enclosures = J over the whole box
        ctx.prec = 320
        un = {}
        for nm in ("dDm1", "dDp1"):
            cols = []
            for jcol in range(2):
                u = parse_acb(subs[0][nm][jcol])
                for r in subs[1:]:
                    u = u.union(parse_acb(r[nm][jcol]))
                cols.append(u)
            un[nm] = [[c.real.str(600, radius=True, more=True), c.imag.str(600, radius=True, more=True)] for c in cols]
            un[nm + "_rad"] = [rad_of(c) for c in cols]
        un.update({"V0_rad": max(r["V0_rad"] for r in subs), "state_rad": max(r["state_rad"] for r in subs),
                   "steps": sum(r["steps"] for r in subs), "sec": sum(r["sec"] for r in subs), "sub": sub,
                   "sub_rad": [[r[nm + "_rad"] for nm in ("dDm1", "dDp1")] for r in subs]})
        out["box"] = un
    if "out" in opts:
        with open(opts["out"], "w") as fh:
            json.dump(out, fh, indent=1)
    if "point" in out:
        p = out["point"]
        print(f"point run at x^ = ({As}, {wfs}), xi1 = {r1}: {p['steps']} steps, {p['sec']:.0f} s; start-value eta bound "
              f"{p['eta0']:.1e}; final state radius {p['state_rad']:.2e}")
        print(f"   D+_-1(xi1) = {parse_acb(p['Dm1'])}\n   D+_+1(xi1) = {parse_acb(p['Dp1'])}")
    if "box" in out:
        b = out["box"]
        print(f"box run over x^ + [-{h}, {h}]^2 with the variational columns: {b['steps']} steps, {b['sec']:.0f} s; "
              f"start derivative radius {b['V0_rad']:.1e}; final state-set radius {b['state_rad']:.2e}")
        for nm in ("dDm1", "dDp1"):
            print(f"   {nm}/d(A, wf) = " + ", ".join(str(parse_acb(x)) for x in b[nm]) +
                  f"   radii {b[nm + '_rad'][0]:.2e}, {b[nm + '_rad'][1]:.2e}")
    if "point" in out and "box" in out:
        ok, K, margins, g_hat, JX, JD, dJD = krawczyk_assemble(As, wfs, h, r1, out["point"], out["box"])
        print(f"   J_D(x^) = {JD.str(8, radius=True)}  (radius {rad_of(JD):.1e})")
        print(f"   dJ_D/d(A, wf) over X = {dJD[0].str(6, radius=True)}, {dJD[1].str(6, radius=True)}  "
              f"(radii {rad_of(dJD[0]):.2e}, {rad_of(dJD[1]):.2e})")
        print(f"   g(x^) = [{g_hat[0].str(6, radius=True)}, {g_hat[1].str(6, radius=True)}]")
        print("   J(X) = [[" + ", ".join(JX[0][j].str(6, radius=True) for j in range(2)) + "], [" +
              ", ".join(JX[1][j].str(6, radius=True) for j in range(2)) + "]]")
        print(f"   K(X) = [{K[0].str(12, radius=True)}, {K[1].str(12, radius=True)}]")
        print(f"Krawczyk on |A - {As}| <= {h}, |w/w_inf - {wfs}| <= {h} (xi1 = {r1}): K(X) in int X: {ok}; "
              f"margins {margins[0]:.3e}, {margins[1]:.3e}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "point"
    if mode == "krawczyk":
        return main_krawczyk(sys.argv[2:])
    r0, r1 = "0.1", "0.2"
    if mode == "linear":
        s = Sys(gravity=False)
        A, wf = arb(1), arb("0.7769940136")
        w = W0 * wf
        lo = linear_start(s, A, w, r0, 1e-30)
        n1, n2 = integrate_lohner(lo, r0, r1, verbose=True)
        J1, ref, R1, G = jump_ball(s, lo, A, w, r1)
        import lin_check
        Dp, Dm, Jx, refx = lin_check.jump_exact(w, arb(r1), mu=1, Fmu=acb(0, -A / 2))
        X = lo.box()
        print(f"linear: steps {n1} + {n2}, {time.time() - T0:.0f} s")
        print(f"   D_1(xi1) validated {X[1].str(15, radius=True)};  exact {Dp.str(15, radius=True)};  overlap {X[1].overlaps(Dp)}")
        print(f"   R_1 = {R1.str(15, radius=True)}  (exact: 1)  contains 1: {R1.contains(acb(1))}")
        return
    s = Sys(gravity=True)
    if mode == "point":
        A = arb(sys.argv[2]) if len(sys.argv) > 2 else arb("2.5421480475")
        wf = arb(sys.argv[3]) if len(sys.argv) > 3 else arb("0.7769940136")
        eps0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-30
        lo, n = start_lohner(s, A, wf, r0, eps0)
        n1, n2 = integrate_lohner(lo, r0, r1, verbose=True)
        J1, ref, R1, G = jump_ball(s, lo, A, W0 * wf, r1)
        X = lo.box()
        rad = max(max(float(x.real.rad()), float(x.imag.rad())) for x in X[:s.nd])
        print(f"point: A {A.str(12)}, w/w_inf {wf.str(12)}, eps0 {eps0:.0e}, series terms {n}: steps {n1} + {n2}, "
              f"{time.time() - T0:.0f} s; final state radius {rad:.2e}")
        print(f"   J_1 = {J1.str(12, radius=True)}")
        print(f"   R_1 = {R1.str(12, radius=True)}")
        print(f"   G   = {G.str(12, radius=True)}")
        return
    if mode == "boxtest":
        # H enclosure over a small parameter box around (A, wf): how the enclosure width scales with the box
        A = arb(sys.argv[2], float(sys.argv[4]))
        wf = arb(sys.argv[3], float(sys.argv[5]))
        P, J0, g0 = precond(float(A.mid()), float(wf.mid()), float(r1))
        lo, n = start_lohner(s, A, wf, r0, 1e-30)
        n1, n2 = integrate_lohner(lo, r0, r1)
        H, G0 = H_enclosure(lo, float(r1), P)
        print(f"box A {A.str(12, radius=True)}, wf {wf.str(12, radius=True)}: H = [{H[0].str(6, radius=True)}, "
              f"{H[1].str(6, radius=True)}]; radii {float(H[0].rad()):.2e} {float(H[1].rad()):.2e} "
              f"({n1 + n2} steps, {time.time() - T0:.0f} s)")
        return
    if mode == "miranda":
        As, wfs = sys.argv[2], sys.argv[3]
        h = float(sys.argv[4])
        eps0 = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-30
        procs = int(sys.argv[6]) if len(sys.argv) > 6 else 2
        P, J0, g0 = precond(float(As), float(wfs), float(r1))
        print(f"preconditioner J0^-1 from the double-precision solver: J0 = {np.round(J0, 3).tolist()}, G(center) ~ {g0}")
        jobs = [(0, -1), (0, 1), (1, -1), (1, 1)]
        args = [(i, sg, As, wfs, h, eps0, P.tolist(), r0, r1) for i, sg in jobs]
        import multiprocessing as mpr
        with mpr.get_context("spawn").Pool(procs) as pool:
            out = pool.map(face_job, args)
        ok = True
        for i, sg, Hs, lo_, hi_, nst, sec in out:
            name = ("A" if i == 0 else "w/w_inf") + (" = c + h" if sg > 0 else " = c - h")
            val_lo, val_hi = lo_[i], hi_[i]
            good = (val_lo > 0) if sg > 0 else (val_hi < 0)
            ok &= good
            print(f"face {name}: H = [{Hs[0]}, {Hs[1]}]  need sign {'+' if sg > 0 else '-'} on H_{i + 1}: {good} "
                  f"({nst} steps, {sec:.0f} s)")
        print(f"Poincare-Miranda on the box |A - {As}| <= {h}, |w/w_inf - {wfs}| <= {h}: {'HOLDS' if ok else 'not established'} "
              f"(conditional on the start enclosure eps0 = {eps0:.0e}, step i)")
        return


if __name__ == "__main__":
    main()
