"""
Choptuik's critical solution as an exact periodic (discretely self-similar) solution,
in D = n + 2 spacetime dimensions.

Metric (polar-areal):  ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2,  t < 0.
Self-similar coordinates:  tau = -ln(-t),  x = r/(-t),  and the time coordinate t is chosen
so that the past light cone of the singularity (the self-similarity horizon, SSH) sits at x = 1.
That is the gauge condition  c(tau, 1) = 1  with  c = alpha / a.

Fields (dimensionless):  p = r (Phi + Pi),  m = r (Phi - Pi),  a,  c,
with Phi = d(phi)/dr, Pi = (a/alpha) d(phi)/dt, and the scalar normalised so that
G_ab = 2n (d_a phi d_b phi - 1/2 g_ab (d phi)^2)   [any normalisation gives the same Delta, gamma].

Equations (derivation in notes.md):
  a_x = a [ (p^2 + m^2) + (n-1)(1 - a^2) ] / (2x)
  c_x = c (n-1)(a^2 - 1) / x
  p_tau + (x - c) p_x =  c_x p - c p / x + n c (p + m) / (2x)
  m_tau + (x + c) m_x = -c_x m + c m / x - n c (p + m) / (2x)

The critical solution is periodic in tau with period Delta, with p, m odd and a, c even under
tau -> tau + Delta/2. We solve it as a boundary-value problem: Fourier in tau, ODE in x, shooting
from the regular centre x = 0 and from the SSH x = 1 (where regularity fixes p), matching at x_m.
Unknowns: p_1(tau) = lim p/x at the centre, c(tau, 0), m(tau, 1), a(tau, 1), and Delta.

Perturbations ~ exp(lambda tau) are handled by the same code: every tau-derivative of a batch member
is replaced by D f + lambda (f - f_0), where member 0 of the batch is the background.
"""
import numpy as np


class DSS:
    def __init__(self, n=2.0, N=64, Kc=8, Ks=8, x0=0.03, u0=0.03, xm=0.35, steps_c=300, steps_s=600):
        self.n, self.N, self.Kc, self.Ks = float(n), int(N), Kc, Ks
        self.x0, self.u0, self.xm = x0, u0, xm
        self.steps_c, self.steps_s = steps_c, steps_s
        self.lam = 0.0
        self.k = np.arange(self.N // 2 + 1)
        H = self.N // 2
        # dense spectral differentiation matrix for omega = 1 (used in periodic linear ODE solves)
        E = np.eye(self.N)
        self.Dmat1 = self._D_unit(E)            # rows: d/dtau of basis vectors (omega = 1)

    # ---------- tau derivative ----------
    def _D_unit(self, f):
        F = np.fft.rfft(f, axis=-1)
        F = F * (1j * self.k)
        F[..., -1] = 0.0
        return np.fft.irfft(F, n=self.N, axis=-1)

    def D(self, f, omega):
        """d/dtau of batch arrays f (B, N); omega (B,). Adds lambda (f - f[0]) for perturbations."""
        d = self._D_unit(f) * omega[:, None]
        if self.lam != 0.0:
            d = d + self.lam * (f - f[0:1])
        return d

    def solve_periodic(self, kappa, rhs, omega, bg=None):
        """Solve D f + kappa f = rhs for periodic f, batchwise. kappa, rhs: (B, N)."""
        B, N = rhs.shape
        out = np.empty_like(rhs)
        Dm = self.Dmat1.T            # f_tau = Dm @ f  (Dmat1 rows are D(e_i) -> columns after transpose)
        for b in range(B):
            Mtx = omega[b] * Dm + np.diag(kappa[b])
            r = rhs[b]
            if self.lam != 0.0 and b > 0:
                Mtx = Mtx + self.lam * np.eye(N)
                r = r + self.lam * out[0]
            out[b] = np.linalg.solve(Mtx, r)
        return out

    # ---------- symmetry helpers ----------
    @staticmethod
    def odd(h):
        return np.concatenate([h, -h], axis=-1)

    @staticmethod
    def even(h):
        return np.concatenate([h, h], axis=-1)

    # ---------- series helpers ----------
    @staticmethod
    def cs(A, B, k, jmin=0):
        """k-th coefficient of the product of series A, B (lists of arrays)."""
        s = 0.0
        for j in range(jmin, k + 1):
            s = s + A[j] * B[k - j]
        return s

    # ---------- centre series ----------
    def centre_series(self, p1, c0, omega):
        n, K = self.n, self.Kc
        Z = np.zeros_like(p1)
        P = [Z.copy() for _ in range(K + 1)]; M = [Z.copy() for _ in range(K + 1)]
        A = [Z.copy() for _ in range(K + 1)]; C = [Z.copy() for _ in range(K + 1)]
        A[0] = np.ones_like(p1); C[0] = c0.copy(); P[1] = p1.copy(); M[1] = -p1.copy()
        for k in range(2, K + 1):
            PM = [P[j] + M[j] for j in range(K + 1)]
            Q2 = [self.cs(P, P, j) + self.cs(M, M, j) for j in range(k + 1)]
            s = self.cs(A, Q2, k)
            A2 = [self.cs(A, A, j) for j in range(k + 1)]
            t3 = self.cs(A, A2, k)                      # with A[k] = 0
            A[k] = (0.5 * s - 0.5 * (n - 1) * t3) / (k + n - 1)
            A2 = [self.cs(A, A, j) for j in range(k + 1)]
            A2m1 = [A2[j] - (1.0 if j == 0 else 0.0) for j in range(k + 1)]
            C[k] = (n - 1) * self.cs(C, A2m1, k) / k
            # residuals of x*E3 and x*E4 at order k with P[k] = M[k] = 0
            Dp = self.D(P[k - 1], omega); Dm = self.D(M[k - 1], omega)
            X = [Z, -C[0]] + [(1.0 if j == 2 else 0.0) - C[j - 1] for j in range(2, k + 2)]
            Y = [Z, C[0]] + [(1.0 if j == 2 else 0.0) + C[j - 1] for j in range(2, k + 2)]
            xpx = sum(X[j] * (k - j + 1) * P[k - j + 1] for j in range(1, k + 1))
            xmx = sum(Y[j] * (k - j + 1) * M[k - j + 1] for j in range(1, k + 1))
            xcxp = sum(j * C[j] * P[k - j] for j in range(1, k + 1))
            xcxm = sum(j * C[j] * M[k - j] for j in range(1, k + 1))
            cp = self.cs(C, P, k); cm = self.cs(C, M, k); cpm = self.cs(C, PM, k)
            r3 = Dp + xpx - xcxp + cp - 0.5 * n * cpm
            r4 = Dm + xmx + xcxm - cm + 0.5 * n * cpm
            c = C[0]
            m11 = c * (1 - k) - 0.5 * n * c; m12 = -0.5 * n * c
            m21 = 0.5 * n * c;               m22 = c * (k - 1) + 0.5 * n * c
            det = m11 * m22 - m12 * m21
            P[k] = (-r3 * m22 + r4 * m12) / det
            M[k] = (-r4 * m11 + r3 * m21) / det
        return P, M, A, C

    # ---------- SSH series (u = 1 - x) ----------
    def ssh_series(self, m0, a0, omega):
        n, K = self.n, self.Ks
        Z = np.zeros_like(m0)
        P = [Z.copy() for _ in range(K + 2)]; M = [Z.copy() for _ in range(K + 2)]
        A = [Z.copy() for _ in range(K + 2)]; C = [Z.copy() for _ in range(K + 2)]
        A[0] = a0.copy(); C[0] = np.ones_like(m0); M[0] = m0.copy()

        def cstep(Kk):        # c_{K+1} from E2 at order K
            A2m1 = [self.cs(A, A, j) - (1.0 if j == 0 else 0.0) for j in range(Kk + 1)]
            C[Kk + 1] = (Kk * C[Kk] - (n - 1) * self.cs(C, A2m1, Kk)) / (Kk + 1)

        def pstep(Kk):        # p_K from E3 at order K (periodic linear ODE)
            w = [1.0 - C[0]] + [-1.0 - C[1]] + [-C[j] for j in range(2, Kk + 2)]
            W = [w[0]] + [w[j] - w[j - 1] for j in range(1, Kk + 2)]
            V = [(j + 1) * C[j + 1] - j * C[j] for j in range(Kk + 1)]
            PM = [P[j] + M[j] for j in range(Kk + 1)]
            tau_term = -self.D(P[Kk - 1], omega) if Kk >= 1 else 0.0
            wpu = sum(W[j] * (Kk - j + 1) * P[Kk - j + 1] for j in range(1, Kk + 2) if Kk - j + 1 >= 0)
            r = tau_term - wpu + self.cs(V, P, Kk) + self.cs(C, P, Kk) - 0.5 * n * self.cs(C, PM, Kk)
            kappa = Kk * (1.0 + C[1]) + C[1] + 1.0 - 0.5 * n
            P[Kk] = self.solve_periodic(kappa + 0 * r, -r, omega)

        for Kk in range(K + 1):
            cstep(Kk)
            pstep(Kk)
            if Kk == K:
                break
            # a_{K+1}
            Q2 = [self.cs(P, P, j) + self.cs(M, M, j) for j in range(Kk + 1)]
            A2 = [self.cs(A, A, j) for j in range(Kk + 1)]
            A3 = self.cs(A, A2, Kk)
            aG = self.cs(A, Q2, Kk) + (n - 1) * (A[Kk] - A3)
            A[Kk + 1] = (Kk * A[Kk] - 0.5 * aG) / (Kk + 1)
            # m_{K+1}
            z = [1.0 + C[0], -1.0 + C[1]] + [C[j] for j in range(2, Kk + 2)]
            Zc = [z[0]] + [z[j] - z[j - 1] for j in range(1, Kk + 2)]
            V = [(j + 1) * C[j + 1] - j * C[j] for j in range(Kk + 1)]
            PM = [P[j] + M[j] for j in range(Kk + 1)]
            tau_term = self.D(M[Kk], omega) - (self.D(M[Kk - 1], omega) if Kk >= 1 else 0.0)
            zmu = sum(Zc[j] * (Kk - j + 1) * M[Kk - j + 1] for j in range(1, Kk + 2) if Kk - j + 1 >= 0)
            rest = tau_term - zmu - self.cs(V, M, Kk) - self.cs(C, M, Kk) + 0.5 * n * self.cs(C, PM, Kk)
            M[Kk + 1] = rest / (Zc[0] * (Kk + 1))
        return P[:K + 1], M[:K + 1], A[:K + 1], C[:K + 1]

    @staticmethod
    def eval_series(S, x):
        out = 0.0
        for k in range(len(S) - 1, -1, -1):
            out = out * x + S[k]
        return out

    # ---------- ODE in x ----------
    def rhs(self, x, s, omega):
        p, m, a, c = s
        n = self.n
        a2 = a * a
        cx = c * (n - 1) * (a2 - 1) / x
        ax = a * ((p * p + m * m) + (n - 1) * (1 - a2)) / (2 * x)
        pc = n * c * (p + m) / (2 * x)
        px = (-self.D(p, omega) + cx * p - c * p / x + pc) / (x - c)
        mx = (-self.D(m, omega) - cx * m + c * m / x - pc) / (x + c)
        return np.stack([px, mx, ax, cx])

    def integrate(self, s, xa, xb, steps, omega):
        h = (xb - xa) / steps
        x = xa
        for _ in range(steps):
            k1 = self.rhs(x, s, omega)
            k2 = self.rhs(x + h / 2, s + h / 2 * k1, omega)
            k3 = self.rhs(x + h / 2, s + h / 2 * k2, omega)
            k4 = self.rhs(x + h, s + h * k3, omega)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x += h
        return s

    # ---------- shooting ----------
    def shoot(self, p1, c0, mS, aS, omega):
        P, M, A, C = self.centre_series(p1, c0, omega)
        x0 = self.x0
        sc = np.stack([self.eval_series(S, x0) for S in (P, M, A, C)])
        sc = self.integrate(sc, x0, self.xm, self.steps_c, omega)
        P, M, A, C = self.ssh_series(mS, aS, omega)
        u0 = self.u0
        ss = np.stack([self.eval_series(S, u0) for S in (P, M, A, C)])
        ss = self.integrate(ss, 1 - u0, self.xm, self.steps_s, omega)
        return sc, ss

    def unpack(self, Zb):
        """Zb: (B, 2N+1) -> full-period fields and omega."""
        H = self.N // 2
        p1 = self.odd(Zb[:, 0:H]); c0 = self.even(Zb[:, H:2 * H])
        mS = self.odd(Zb[:, 2 * H:3 * H]); aS = self.even(Zb[:, 3 * H:4 * H])
        omega = 2 * np.pi / Zb[:, 4 * H]
        return p1, c0, mS, aS, omega

    def residual(self, Zb):
        H = self.N // 2
        p1, c0, mS, aS, omega = self.unpack(Zb)
        sc, ss = self.shoot(p1, c0, mS, aS, omega)
        d = sc - ss                                     # (4, B, N)
        mis = np.concatenate([d[0, :, :H], d[1, :, :H], d[2, :, :H], d[3, :, :H]], axis=1)
        phase = np.fft.rfft(p1, axis=-1)[:, 1].imag / self.N
        return np.concatenate([mis, phase[:, None]], axis=1)

    def jacobian(self, z, eps=1e-7):
        nz = len(z)
        h = eps * np.maximum(1.0, np.abs(z))
        Zb = np.repeat(z[None, :], nz + 1, axis=0)
        Zb[1:] += np.diag(h)
        R = self.residual(Zb)
        return R[0], (R[1:] - R[0]).T / h

    def newton(self, z, tol=1e-12, maxit=30, verbose=True, damping=1.0):
        for it in range(maxit):
            F, J = self.jacobian(z)
            nF = np.linalg.norm(F)
            if verbose:
                print(f"  it {it:2d}  |F| = {nF:.3e}  Delta = {z[-1]:.12f}", flush=True)
            if nF < tol:
                return z, nF, True
            dz = np.linalg.lstsq(J, -F, rcond=None)[0]
            # backtracking line search
            t = damping
            for _ in range(12):
                zt = z + t * dz
                Ft = self.residual(zt[None, :])[0]
                if np.linalg.norm(Ft) < nF * (1 - 0.1 * t) or t < 1e-3:
                    break
                t *= 0.5
            z = zt
        F = self.residual(z[None, :])[0]
        return z, np.linalg.norm(F), np.linalg.norm(F) < tol

    # ---------- perturbations ----------
    def mode_matrix(self, z, lam, eps=1e-6, sector="same"):
        """Linear map from perturbation data (dp1, dc0, dmS, daS) to mismatch, for growth rate lam.
        Central differences around the background z, Delta fixed."""
        H = self.N // 2
        nv = 4 * H
        self.lam = lam
        try:
            Zb = np.repeat(z[None, :], 2 * nv + 1, axis=0)
            for j in range(nv):
                Zb[1 + j, j] += eps
                Zb[1 + nv + j, j] -= eps
            R = self.residual(Zb)[:, :4 * H]
        finally:
            self.lam = 0.0
        return (R[1:1 + nv] - R[1 + nv:]).T / (2 * eps)


class DSSMulti(DSS):
    """Multiple shooting: unknown full states at interior nodes, short integrations between them.
    Robust against rough initial guesses."""

    def __init__(self, nodes, x0=1e-4, Kc=4, log_steps=120, seg_h=0.002, **kw):
        super().__init__(x0=x0, Kc=Kc, **kw)
        self.nodes = list(nodes)
        self.log_steps, self.seg_h = log_steps, seg_h

    def integrate_log(self, s, xa, xb, steps, omega):
        """Integrate in ln x (for the approach from the centre)."""
        h = (np.log(xb) - np.log(xa)) / steps
        y = np.log(xa)
        f = lambda y, s: np.exp(y) * self.rhs(np.exp(y), s, omega)
        for _ in range(steps):
            k1 = f(y, s); k2 = f(y + h / 2, s + h / 2 * k1); k3 = f(y + h / 2, s + h / 2 * k2); k4 = f(y + h, s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y += h
        return s

    def nseg(self, xa, xb, omega=None):
        h = self.seg_h
        if omega is not None:      # keep the stiff transport term stable: |c - x| >= ~0.1 away from the SSH
            h = min(h, 0.15 / (0.5 * self.N * float(np.max(omega))))
        return max(4, int(np.ceil(abs(xb - xa) / h)))

    def integrate_u(self, s, xa, xb, omega):
        """Integrate in y = ln(1 - x): steps shrink towards the SSH where the p-equation is stiff."""
        ya, yb = np.log(1 - xa), np.log(1 - xb)
        hy = min(0.01, 0.45 / (0.5 * self.N * float(np.max(omega))))
        steps = max(4, int(np.ceil(abs(yb - ya) / hy)))
        h = (yb - ya) / steps
        y = ya
        f = lambda y, s: -np.exp(y) * self.rhs(1 - np.exp(y), s, omega)
        for _ in range(steps):
            k1 = f(y, s); k2 = f(y + h / 2, s + h / 2 * k1); k3 = f(y + h / 2, s + h / 2 * k2); k4 = f(y + h, s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y += h
        return s

    def seg(self, s, xa, xb, omega):
        if min(xa, xb) >= 0.4:
            return self.integrate_u(s, xa, xb, omega)
        return self.integrate(s, xa, xb, self.nseg(xa, xb, omega), omega)

    def layout(self):
        H = self.N // 2
        K = len(self.nodes)
        return H, K, 2 * H + 4 * K * H + 2 * H + 1

    def unpack_ms(self, Zb):
        H, K, L = self.layout()
        o = 0
        p1 = self.odd(Zb[:, o:o + H]); o += H
        c0 = self.even(Zb[:, o:o + H]); o += H
        states = []
        for i in range(K):
            ps = self.odd(Zb[:, o:o + H]); o += H
            ms = self.odd(Zb[:, o:o + H]); o += H
            as_ = self.even(Zb[:, o:o + H]); o += H
            cs_ = self.even(Zb[:, o:o + H]); o += H
            states.append(np.stack([ps, ms, as_, cs_]))
        mS = self.odd(Zb[:, o:o + H]); o += H
        aS = self.even(Zb[:, o:o + H]); o += H
        omega = 2 * np.pi / Zb[:, o]
        return p1, c0, states, mS, aS, omega

    def residual(self, Zb):
        H, K, L = self.layout()
        p1, c0, states, mS, aS, omega = self.unpack_ms(Zb)
        blocks = []
        P, M, A, C = self.centre_series(p1, c0, omega)
        s = np.stack([self.eval_series(S, self.x0) for S in (P, M, A, C)])
        s = self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, omega)
        blocks.append(s - states[0])
        for i in range(K - 1):
            xa, xb = self.nodes[i], self.nodes[i + 1]
            s = self.seg(states[i], xa, xb, omega)
            blocks.append(s - states[i + 1])
        P, M, A, C = self.ssh_series(mS, aS, omega)
        s = np.stack([self.eval_series(S, self.u0) for S in (P, M, A, C)])
        xa = 1 - self.u0
        s = self.seg(s, xa, self.nodes[-1], omega)
        blocks.append(s - states[-1])
        mis = np.concatenate([np.concatenate([b[0, :, :H], b[1, :, :H], b[2, :, :H], b[3, :, :H]], axis=1) for b in blocks], axis=1)
        phase = np.fft.rfft(p1, axis=-1)[:, 1].imag / self.N
        return np.concatenate([mis, phase[:, None]], axis=1)

    def mode_matrix(self, z, lam, eps=1e-6):
        H, K, L = self.layout()
        nv = L - 1
        self.lam = lam
        try:
            Zb = np.repeat(z[None, :], 2 * nv + 1, axis=0)
            for j in range(nv):
                Zb[1 + j, j] += eps
                Zb[1 + nv + j, j] -= eps
            R = self.residual(Zb)[:, :nv]
        finally:
            self.lam = 0.0
        return (R[1:1 + nv] - R[1 + nv:]).T / (2 * eps)


def _structured_jacobian(self, z, eps=1e-7):
    """Block-structured Jacobian for multiple shooting: each segment is integrated only with the
    perturbations of its own starting data (plus Delta), ~K times cheaper than the dense version."""
    H, K, L = self.layout()
    N = self.N
    F0 = self.residual(z[None, :])[0]
    J = np.zeros((L, L))
    D = z[-1]; hD = eps * max(1.0, abs(D))
    blk = 4 * H                      # rows per block
    # offsets of unknowns
    o_c = 0                          # p1, c0: [0, 2H)
    o_nodes = 2 * H                  # node i state: [2H + 4H i, 2H + 4H (i+1))
    o_s = 2 * H + 4 * H * K          # mS, aS
    o_D = L - 1

    def to_rows(s, ref):
        d = s - ref
        return np.concatenate([d[0, :, :H], d[1, :, :H], d[2, :, :H], d[3, :, :H]], axis=1)

    def batch_from(base_vec, nfree):
        """base_vec: 4H (or 2H) half-sample data; returns (1 + nfree + 1) x len batch with unit perturbations + Delta"""
        B = 1 + nfree + 1
        X = np.repeat(base_vec[None, :], B, axis=0)
        h = eps * np.maximum(1.0, np.abs(base_vec))
        X[1:1 + nfree] += np.diag(h)
        om = np.full(B, 2 * np.pi / D); om[-1] = 2 * np.pi / (D + hD)
        return X, h, om

    # centre block: unknowns p1 (odd), c0 (even)
    X, h, om = batch_from(z[o_c:o_c + 2 * H], 2 * H)
    p1 = self.odd(X[:, :H]); c0 = self.even(X[:, H:])
    P, M, A, C = self.centre_series(p1, c0, om)
    s = np.stack([self.eval_series(S_, self.x0) for S_ in (P, M, A, C)])
    s = self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, om)
    R = to_rows(s, s[:, 0:1])                     # differences w.r.t. the unperturbed member
    J[0:blk, o_c:o_c + 2 * H] = (R[1:1 + 2 * H] / h[:, None]).T
    J[0:blk, o_D] += R[-1] / hD
    J[0:blk, o_nodes:o_nodes + 4 * H] -= np.eye(4 * H)
    # interior segments
    for i in range(K - 1):
        oi = o_nodes + 4 * H * i
        X, h, om = batch_from(z[oi:oi + 4 * H], 4 * H)
        st = np.stack([self.odd(X[:, :H]), self.odd(X[:, H:2 * H]), self.even(X[:, 2 * H:3 * H]), self.even(X[:, 3 * H:])])
        s = self.seg(st, self.nodes[i], self.nodes[i + 1], om)
        R = to_rows(s, s[:, 0:1])
        r0 = blk * (i + 1)
        J[r0:r0 + blk, oi:oi + 4 * H] = (R[1:1 + 4 * H] / h[:, None]).T
        J[r0:r0 + blk, o_D] += R[-1] / hD
        J[r0:r0 + blk, oi + 4 * H:oi + 8 * H] -= np.eye(4 * H)
    # SSH block
    X, h, om = batch_from(z[o_s:o_s + 2 * H], 2 * H)
    mS = self.odd(X[:, :H]); aS = self.even(X[:, H:])
    P, M, A, C = self.ssh_series(mS, aS, om)
    s = np.stack([self.eval_series(S_, self.u0) for S_ in (P, M, A, C)])
    s = self.seg(s, 1 - self.u0, self.nodes[-1], om)
    R = to_rows(s, s[:, 0:1])
    r0 = blk * K
    J[r0:r0 + blk, o_s:o_s + 2 * H] = (R[1:1 + 2 * H] / h[:, None]).T
    J[r0:r0 + blk, o_D] += R[-1] / hD
    J[r0:r0 + blk, o_nodes + 4 * H * (K - 1):o_nodes + 4 * H * K] -= np.eye(4 * H)
    # phase row: Im(rfft(odd(p1))[1]) / N is linear in the half samples
    E = np.eye(H)
    J[-1, o_c:o_c + H] = np.fft.rfft(self.odd(E), axis=-1)[:, 1].imag / N
    return F0, J

DSSMulti.jacobian = _structured_jacobian


def _structured_mode_matrix(self, z, lam, eps=1e-6):
    """Linearised mismatch for perturbations ~ exp(lam tau), Delta fixed. Central differences, block-structured."""
    H, K, L = self.layout()
    nv = L - 1
    M = np.zeros((nv, nv))
    D = z[-1]
    blk = 4 * H
    o_c, o_nodes, o_s = 0, 2 * H, 2 * H + 4 * H * K

    def batch_from(base_vec):
        nf = len(base_vec)
        X = np.repeat(base_vec[None, :], 1 + 2 * nf, axis=0)
        X[1:1 + nf] += eps * np.eye(nf); X[1 + nf:] -= eps * np.eye(nf)
        return X, np.full(1 + 2 * nf, 2 * np.pi / D)

    def rows(s):
        d = np.concatenate([s[0, :, :H], s[1, :, :H], s[2, :, :H], s[3, :, :H]], axis=1)
        nf = (d.shape[0] - 1) // 2
        return ((d[1:1 + nf] - d[1 + nf:]) / (2 * eps)).T

    self.lam = lam
    try:
        X, om = batch_from(z[o_c:o_c + 2 * H])
        P, Mm, A, C = self.centre_series(self.odd(X[:, :H]), self.even(X[:, H:]), om)
        s = np.stack([self.eval_series(S_, self.x0) for S_ in (P, Mm, A, C)])
        s = self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, om)
        M[0:blk, o_c:o_c + 2 * H] = rows(s)
        M[0:blk, o_nodes:o_nodes + 4 * H] -= np.eye(4 * H)
        for i in range(K - 1):
            oi = o_nodes + 4 * H * i
            X, om = batch_from(z[oi:oi + 4 * H])
            st = np.stack([self.odd(X[:, :H]), self.odd(X[:, H:2 * H]), self.even(X[:, 2 * H:3 * H]), self.even(X[:, 3 * H:])])
            s = self.seg(st, self.nodes[i], self.nodes[i + 1], om)
            r0 = blk * (i + 1)
            M[r0:r0 + blk, oi:oi + 4 * H] = rows(s)
            M[r0:r0 + blk, oi + 4 * H:oi + 8 * H] -= np.eye(4 * H)
        X, om = batch_from(z[o_s:o_s + 2 * H])
        P, Mm, A, C = self.ssh_series(self.odd(X[:, :H]), self.even(X[:, H:]), om)
        s = np.stack([self.eval_series(S_, self.u0) for S_ in (P, Mm, A, C)])
        s = self.seg(s, 1 - self.u0, self.nodes[-1], om)
        r0 = blk * K
        M[r0:r0 + blk, o_s:o_s + 2 * H] = rows(s)
        M[r0:r0 + blk, o_nodes + 4 * H * (K - 1):o_nodes + 4 * H * K] -= np.eye(4 * H)
    finally:
        self.lam = 0.0
    return M

DSSMulti.mode_matrix = _structured_mode_matrix
