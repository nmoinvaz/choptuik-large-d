"""
DSS (echoing) critical solution of the self-gravitating hedgehog sigma-model (wave map into S^{D-1}, coupling
eta = 4 pi G f^2) in D = n + 2 dimensions, as a periodic boundary-value problem.

Copied and adapted from the l-multiplet solver of another agent (worktree agent-a5bec5ffdf3db4330,
solver/matter/dss_ell.py, class DSSEll), which is itself built on solver/spec.py (DSSSpec) of this repository.
Nothing is imported from that worktree.

Equations (matter/SIGMA.md section 2, checked in solver/matter/sigma_eqs.py), in the field normalisation
psi = sqrt(eta/n) u, p, m = sqrt(eta/n) r (Phi +- Pi), and with kappa^2 = n/eta = n*eps (eps = 1/eta):

  sigma(psi) = sin(kappa psi)/kappa,   gam(psi) = cos(kappa psi)      [eta sin^2 u = n sigma^2, sqrt(eta n) sin 2u/2 = n sigma gam]

  x a_x   = a [p^2 + m^2 + 2 n a^2 sigma^2 + (n-1)(1 - a^2)] / 2
  x c_x   = c [(n-1)(a^2 - 1) - 2 n a^2 sigma^2]
  x psi_x = (p + m) / 2
  p_tau + (x - c) p_x =  c_x p - c p/x + n c (p + m)/(2x) - n c a^2 sigma gam / x
  m_tau + (x + c) m_x = -c_x m + c m/x - n c (p + m)/(2x) + n c a^2 sigma gam / x

At eps = 0 (eta = infinity) sigma = psi, gam = 1, and these are exactly the l = 1 multiplet's equations (L = n):
the hedgehog becomes a triplet (D - 1 components) of free scalars.

Symmetry: the equations are invariant under (p, m, psi) -> -(p, m, psi) (u -> -u), and the solution is sought with
p, m, psi odd and a, c even under tau -> tau + Delta/2 (checked in sigdss_parity.py).

Boundary data (as for l = 1): centre psi ~ q(tau) x, p ~ m ~ q x, free q(tau) and c0(tau); horizon (x = 1, c = 1)
free m, a, psi, and p from regularity. The Taylor series of sigma(psi) and gam(psi) are generated order by order from
d sigma = gam d psi, d gam = -kappa^2 sigma d psi.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from spec import DSSSpec                      # noqa: E402  (this repository's solver/spec.py)

NODES_STD = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PAR = ("odd", "odd", "even", "even", "odd")      # p, m, a, c, psi


def _mm(f, M):
    """f @ M for real M: numpy has no BLAS path for complex @ real (it is ~20x slower), so split complex f."""
    if np.iscomplexobj(f):
        return (f.real @ M) + 1j * (f.imag @ M)
    return f @ M


def _rfft_op(N, mult):
    """Dense matrix of the linear map f -> irfft(rfft(f) * mult) acting on rows: f @ Mat."""
    E = np.eye(N)
    F = np.fft.rfft(E, axis=-1) * mult
    return np.fft.irfft(F, n=N, axis=-1)


class SigDSS(DSSSpec):
    def __init__(self, kc, nodes, n, eps, Delta_ref, Kc=9, Ks=16, u0=0.01, x0=1e-4, seg_h=0.002,
                 log_steps=150, rk=1, N=None, margin=1.04, full=False):
        """full=True: no half-period symmetry imposed (every field carries all harmonics 0..kc), for the parity check."""
        N = N or int(4 * np.ceil((3 * kc + 2) / 4))
        k = np.arange(N // 2 + 1)
        Dmult = 1j * k; Dmult[-1] = 0.0
        self._Dm = _rfft_op(N, Dmult)                      # d/dtau, omega = 1
        self._Tm = _rfft_op(N, (k <= kc).astype(float))    # truncation to k <= kc
        super().__init__(kc=kc, nodes=nodes, n=n, N=N, Ks=Ks, u0=u0, x0=x0, Kc=Kc, seg_h=seg_h, log_steps=log_steps)
        self.eps = eps                                      # 1/eta
        self.rk = int(rk)
        self.full = bool(full)
        self.PAR = ("all",) * 5 if self.full else PAR
        self.size = {"odd": self.n_odd, "even": self.n_even, "all": 1 + 2 * kc}
        self.set_delta_ref(Delta_ref, margin)
        self._B = {p: self._phys_fft(np.eye(self.size[p]), p) for p in ("odd", "even", "all")}
        self._S = {p: self._spec_fft(np.eye(self.N), p) for p in ("odd", "even", "all")}
        # phase condition: Im of harmonic 1 of the centre datum q
        self.phase_index = (self.kc + 1) if self.full else len(self.kodd)

    # ---------- coupling ----------
    @property
    def L(self):
        return self.n                                       # l = 1: L = l (l + n - 1) = n

    @property
    def k2(self):
        return self.n * self.eps                            # kappa^2 = n / eta

    def sg(self, q):
        """sigma(psi) = sin(kappa psi)/kappa and gam(psi) = cos(kappa psi), as entire functions of
        w = kappa^2 psi^2 (Taylor series for |w| < 0.5, so that complex steps in eps or n at eps = 0 survive)."""
        k2 = self.k2
        if k2 == 0:
            return q, np.ones_like(q)
        w = k2 * q * q
        small = np.abs(w) < 0.5
        # Taylor: sin(s)/s = sum (-w)^k/(2k+1)!, cos(s) = sum (-w)^k/(2k)!, 12 terms (|w|^12/24! ~ 1e-27)
        fs = np.zeros_like(w); fc = np.zeros_like(w)
        term_s = np.ones_like(w); term_c = np.ones_like(w)
        for k in range(12):
            fs = fs + term_s; fc = fc + term_c
            term_s = -term_s * w / ((2 * k + 2) * (2 * k + 3))
            term_c = -term_c * w / ((2 * k + 1) * (2 * k + 2))
        if not np.all(small):
            sw = np.sqrt(np.where(small, 1.0, w))
            fs = np.where(small, fs, np.sin(sw) / sw)
            fc = np.where(small, fc, np.cos(sw))
        return q * fs, fc

    # ---------- linear maps ----------
    def _phys_fft(self, c, parity):
        if parity == "all":
            kc = self.kc
            F = np.zeros((c.shape[0], self.N // 2 + 1), complex)
            F[:, 0] = c[:, 0]
            F[:, 1:kc + 1] = c[:, 1:kc + 1] + 1j * c[:, kc + 1:2 * kc + 1]
            return np.fft.irfft(F * self.N, n=self.N, axis=-1)
        return DSSSpec.phys(self, c, parity)

    def _spec_fft(self, f, parity):
        if parity == "all":
            kc = self.kc
            F = np.fft.rfft(f, axis=-1) / self.N
            return np.concatenate([F[:, :1].real, F[:, 1:kc + 1].real, F[:, 1:kc + 1].imag], axis=1)
        return DSSSpec.spec(self, f, parity)

    def phys(self, c, parity):
        return _mm(c, self._B[parity])

    def spec(self, f, parity):
        return _mm(f, self._S[parity])

    def _D_unit(self, f):
        return _mm(f, self._Dm)

    def trunc(self, f):
        return _mm(f, self._Tm)

    def solve_periodic(self, kappa, rhs, omega, bg=None):
        """(omega d/dtau + kappa) f = rhs, batchwise; members b > 0 get + lam (f - f_0) when self.lam != 0."""
        B, N = rhs.shape
        dt = np.result_type(kappa, rhs, omega, self.lam)
        Mt = omega[:, None, None] * self._Dm.T[None, :, :].astype(dt)
        idx = np.arange(N)
        Mt[:, idx, idx] += kappa
        if self.lam == 0.0:
            return np.linalg.solve(Mt, rhs[..., None])[..., 0]
        out = np.empty(rhs.shape, dtype=dt)
        out[0] = np.linalg.solve(Mt[0], rhs[0])
        if B > 1:
            Mt1 = Mt[1:].copy(); Mt1[:, idx, idx] += self.lam
            out[1:] = np.linalg.solve(Mt1, (rhs[1:] + self.lam * out[0])[..., None])[..., 0]
        return out

    # ---------- RK step counts frozen at a reference Delta ----------
    def set_delta_ref(self, Delta_ref, margin=1.04):
        self.Delta_ref = float(Delta_ref)
        self.om_ref = margin * 2 * np.pi / self.Delta_ref

    def refreeze(self, z, kappa=0.5, tol=0.15):
        old = getattr(self, "steps_map", None)
        new = self.freeze_from(z, kappa)
        if old is not None and set(old) == set(new) and all(abs(new[k] - old[k]) <= tol * old[k] for k in old):
            self.steps_map = old
        return self.steps_map

    def freeze_from(self, z, kappa=0.5):
        """RK4 step count of every segment from the stiffness kc omega/|x - c| of the transport terms of solution z
        (as in dss_ell.py). Frozen while Newton iterates."""
        f = self.fields(z)
        om = 2 * np.pi / f["Delta"] * 1.02
        kw = self.kc * om
        xs = [0.0] + list(self.nodes)
        cs = [f["c0"]] + [st[3] for st in f["states"]]

        def gap(i):
            return max(1e-3, float(np.min(np.abs(xs[i] - cs[i]))))
        st = {}
        g = max(self.x0 * kw / max(1e-3, float(np.min(np.abs(self.x0 - cs[0])))), self.nodes[0] * kw / gap(1))
        st["log"] = max(20, int(np.ceil(np.log(self.nodes[0] / self.x0) * g / kappa * self.rk)))
        for i in range(1, len(self.nodes)):
            xa, xb = self.nodes[i - 1], self.nodes[i]
            if min(xa, xb) >= 0.4:
                g = max((1 - xa) * kw / gap(i), (1 - xb) * kw / gap(i + 1))
                L = abs(np.log(1 - xb) - np.log(1 - xa))
            else:
                g = max(kw / gap(i), kw / gap(i + 1))
                L = xb - xa
            st[(round(xa, 9), round(xb, 9))] = max(4, int(np.ceil(L * g / kappa * self.rk)))
        a2 = f["aS"] ** 2
        s_, _ = self.sg(f["qS"])
        c1 = -(self.n - 1) * (a2 - 1) + 2 * self.L * a2 * s_ ** 2
        g = max(kw / max(0.05, float(np.min(1 + c1))), (1 - self.nodes[-1]) * kw / gap(len(self.nodes)))
        L = abs(np.log(1 - self.nodes[-1]) - np.log(self.u0))
        st[(round(1 - self.u0, 9), round(self.nodes[-1], 9))] = max(4, int(np.ceil(L * g / kappa * self.rk)))
        self.steps_map = st
        return st

    def _steps(self, xa, xb):
        m = getattr(self, "steps_map", None)
        if m is None:
            return None
        return m.get((round(xa, 9), round(xb, 9)))

    def seg(self, s, xa, xb, omega):
        k = self._steps(xa, xb)
        if min(xa, xb) >= 0.4:
            return self.integrate_u(s, xa, xb, omega, steps=k)
        return self.integrate(s, xa, xb, k if k else self.nseg(xa, xb, omega), omega)

    def nseg(self, xa, xb, omega=None):
        h = min(self.seg_h, 0.15 / (0.5 * self.N * self.om_ref)) / self.rk
        return max(4, int(np.ceil(abs(xb - xa) / h)))

    def integrate_u(self, s, xa, xb, omega, steps=None):
        ya, yb = np.log(1 - xa), np.log(1 - xb)
        if steps is None:
            hy = min(0.01, 0.45 / (0.5 * self.N * self.om_ref)) / self.rk
            steps = max(4, int(np.ceil(abs(yb - ya) / hy)))
        h = (yb - ya) / steps; y = ya
        f = lambda y, s: -np.exp(y) * self.rhs(1 - np.exp(y), s, omega)
        for _ in range(steps):
            k1 = f(y, s); k2 = f(y + h / 2, s + h / 2 * k1); k3 = f(y + h / 2, s + h / 2 * k2); k4 = f(y + h, s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y += h
        return s

    def integrate_log(self, s, xa, xb, steps, omega):
        m = getattr(self, "steps_map", None)
        steps = m["log"] if m is not None else steps * self.rk
        h = (np.log(xb) - np.log(xa)) / steps
        y = np.log(xa)
        f = lambda y, s: np.exp(y) * self.rhs(np.exp(y), s, omega)
        for _ in range(steps):
            k1 = f(y, s); k2 = f(y + h / 2, s + h / 2 * k1); k3 = f(y + h / 2, s + h / 2 * k2); k4 = f(y + h, s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y += h
        return s

    # ---------- equations ----------
    def rhs(self, x, s, omega):
        p, m, a, c, q = s
        n, L = self.n, self.L
        a2 = a * a
        sq, gq = self.sg(q)
        g = a2 * sq * sq
        cx = c * ((n - 1) * (a2 - 1) - 2 * L * g) / x
        ax = a * ((p * p + m * m) + 2 * L * g + (n - 1) * (1 - a2)) / (2 * x)
        qx = (p + m) / (2 * x)
        pc = n * c * (p + m) / (2 * x)
        lt = L * c * a2 * sq * gq / x
        px = (-self.D(p, omega) + cx * p - c * p / x + pc - lt) / (x - c)
        mx = (-self.D(m, omega) - cx * m + c * m / x - pc + lt) / (x + c)
        return self.trunc(np.stack([px, mx, ax, cx, qx]))

    @staticmethod
    def _cs(X, Y, k, j0=0):
        s = 0.0
        for j in range(j0, k + 1):
            s = s + X[j] * Y[k - j]
        return s

    def centre_series(self, d, c0, omega):
        """Taylor series at x = 0, psi ~ q x (d = q). Returns P, M, A, C, Q lists."""
        n, L, K, k2 = self.n, self.L, self.Kc, self.k2
        cs = self._cs
        Z = np.zeros_like(d * c0)
        P = [Z.copy() for _ in range(K + 1)]; M = [Z.copy() for _ in range(K + 1)]
        A = [Z.copy() for _ in range(K + 1)]; C = [Z.copy() for _ in range(K + 1)]
        Q = [Z.copy() for _ in range(K + 1)]
        Sn = [Z.copy() for _ in range(K + 1)]; Co = [Z.copy() for _ in range(K + 1)]   # sigma(psi), gam(psi)
        A[0] = np.ones_like(Z); C[0] = c0 + 0 * Z; Co[0] = np.ones_like(Z)
        P[1] = d + 0 * Z; M[1] = d + 0 * Z; Q[1] = d + 0 * Z
        Sn[1] = Q[1] * Co[0]; Co[1] = 0 * Z                 # sigma_0 = 0
        for k in range(1, K + 1):
            if k >= 2:
                # gam_k does not involve Q_k (sigma_0 = 0); sigma_k provisional (Q_k = 0)
                Co[k] = -k2 * sum(j * Q[j] * Sn[k - j] for j in range(1, k)) / k
                Sn[k] = sum(j * Q[j] * Co[k - j] for j in range(1, k)) / k
            SS = [cs(Sn, Sn, j) for j in range(k + 1)]          # sigma^2 (sigma_0 = 0: no sigma_k at order k)
            Q2 = [cs(P, P, j) + cs(M, M, j) for j in range(k + 1)]
            s = cs(A, Q2, k)
            A2 = [cs(A, A, j) for j in range(k + 1)]
            t3 = cs(A, A2, k)                                   # with A[k] = 0
            A2S = [cs(A2, SS, j) for j in range(k + 1)]
            u = cs(A, A2S, k)                                   # [a^3 sigma^2]_k: A[k] does not enter
            A[k] = (0.5 * s + L * u - 0.5 * (n - 1) * t3) / (k + n - 1)
            A2 = [cs(A, A, j) for j in range(k + 1)]
            A2m1 = [A2[j] - (1.0 if j == 0 else 0.0) for j in range(k + 1)]
            G = [cs(A2, SS, j) for j in range(k + 1)]
            C[k] = ((n - 1) * cs(C, A2m1, k) - 2 * L * cs(C, G, k)) / k
            if k <= 1:
                continue                                        # order 1 is the free datum
            PM = [P[j] + M[j] for j in range(K + 1)]
            Dp = self.D(P[k - 1], omega); Dmm = self.D(M[k - 1], omega)
            X = [Z, -C[0]] + [(1.0 if j == 2 else 0.0) - C[j - 1] for j in range(2, k + 2)]
            Y = [Z, C[0]] + [(1.0 if j == 2 else 0.0) + C[j - 1] for j in range(2, k + 2)]
            xpx = sum(X[j] * (k - j + 1) * P[k - j + 1] for j in range(1, k + 1))
            xmx = sum(Y[j] * (k - j + 1) * M[k - j + 1] for j in range(1, k + 1))
            xcxp = sum(j * C[j] * P[k - j] for j in range(1, k + 1))
            xcxm = sum(j * C[j] * M[k - j] for j in range(1, k + 1))
            cp = cs(C, P, k); cm = cs(C, M, k); cpm = cs(C, PM, k)
            SG = [cs(Sn, Co, j) for j in range(k + 1)]          # sigma gam, order k provisional (Q_k = 0)
            H = [cs(A2, SG, j) for j in range(k + 1)]           # a^2 sigma gam
            ch = cs(C, H, k)
            r3 = Dp + xpx - xcxp + cp - 0.5 * n * cpm + L * ch
            r4 = Dmm + xmx + xcxm - cm + 0.5 * n * cpm - L * ch
            c = C[0]
            al = 0.5 * n - L / (2.0 * k)                        # d[sigma gam]_k / dQ_k = 1 at the centre
            m11 = c * (1 - k - al); m12 = -c * al
            m21 = c * al;           m22 = c * (k - 1 + al)
            det = m11 * m22 - m12 * m21
            P[k] = (-r3 * m22 + r4 * m12) / det
            M[k] = (-r4 * m11 + r3 * m21) / det
            Q[k] = (P[k] + M[k]) / (2.0 * k)
            Sn[k] = Sn[k] + Q[k] * Co[0]
        return P, M, A, C, Q

    def ssh_series(self, m0, a0, q0, omega):
        """Taylor series in u = 1 - x at the horizon (c = 1 there). Returns P, M, A, C, Q lists (orders 0..Ks)."""
        n, L, K, k2 = self.n, self.L, self.Ks, self.k2
        B, N = m0.shape
        dt = np.result_type(m0, a0, q0, omega, self.lam, float, k2)
        shp = (K + 2, B, N)
        P = np.zeros(shp, dt); M = np.zeros(shp, dt); A = np.zeros(shp, dt); C = np.zeros(shp, dt)
        Q = np.zeros(shp, dt); Sn = np.zeros(shp, dt); Co = np.zeros(shp, dt)
        A2 = np.zeros(shp, dt); Q2 = np.zeros(shp, dt)
        G = np.zeros(shp, dt); H = np.zeros(shp, dt)                 # a^2 sigma^2, a^2 sigma gam
        SS = np.zeros(shp, dt); SG = np.zeros(shp, dt)
        DP = np.zeros(shp, dt); DM = np.zeros(shp, dt)
        cs = lambda X, Y, k: np.einsum("jbn,jbn->bn", X[:k + 1], Y[k::-1] if k > 0 else Y[:1])
        A[0] = a0; C[0] = 1.0; M[0] = m0; Q[0] = q0
        Sn[0], Co[0] = self.sg(q0 + 0 * Q[0])
        A2[0] = A[0] * A[0]; SS[0] = Sn[0] * Sn[0]; SG[0] = Sn[0] * Co[0]
        G[0] = A2[0] * SS[0]; H[0] = A2[0] * SG[0]
        DM[0] = self.D(M[0], omega)
        for Kk in range(K + 1):
            C[Kk + 1] = (Kk * C[Kk] - (n - 1) * (cs(C, A2, Kk) - C[Kk]) + 2 * L * cs(C, G, Kk)) / (Kk + 1)
            j = np.arange(Kk + 2)
            w = -C[:Kk + 2].copy(); w[0] += 1.0; w[1] -= 1.0
            W = np.empty_like(w); W[0] = w[0]; W[1:] = w[1:] - w[:-1]
            V = (j[1:, None, None] * C[1:Kk + 2]) - (j[:-1, None, None] * C[:Kk + 1])
            PM = P[:Kk + 1] + M[:Kk + 1]
            tau_term = -DP[Kk - 1] if Kk >= 1 else 0.0
            i = j[:-1]
            wpu = np.einsum("ibn,ibn->bn", i[:, None, None] * P[:Kk + 1], W[Kk + 1:0:-1])
            r = tau_term - wpu + cs(V, P, Kk) + cs(C, P, Kk) - 0.5 * n * cs(C, PM, Kk) + L * cs(C, H, Kk)
            kappa = Kk * (1.0 + C[1]) + C[1] + 1.0 - 0.5 * n
            P[Kk] = self.solve_periodic(kappa + 0 * r, -r, omega)
            DP[Kk] = self.D(P[Kk], omega)
            Q2[Kk] = cs(P, P, Kk) + cs(M, M, Kk)
            if Kk == K:
                break
            aG = cs(A, Q2, Kk) + (n - 1) * (A[Kk] - cs(A, A2, Kk)) + 2 * L * cs(A, G, Kk)
            A[Kk + 1] = (Kk * A[Kk] - 0.5 * aG) / (Kk + 1)
            A2[Kk + 1] = cs(A, A, Kk + 1)
            zc = C[:Kk + 2].copy(); zc[0] += 1.0; zc[1] -= 1.0
            Zc = np.empty_like(zc); Zc[0] = zc[0]; Zc[1:] = zc[1:] - zc[:-1]
            tau_term = DM[Kk] - (DM[Kk - 1] if Kk >= 1 else 0.0)
            PM = P[:Kk + 1] + M[:Kk + 1]
            zmu = np.einsum("ibn,ibn->bn", i[:, None, None] * M[:Kk + 1], Zc[Kk + 1:0:-1])
            rest = tau_term - zmu - cs(V, M, Kk) - cs(C, M, Kk) + 0.5 * n * cs(C, PM, Kk) - L * cs(C, H, Kk)
            M[Kk + 1] = rest / (Zc[0] * (Kk + 1))
            DM[Kk + 1] = self.D(M[Kk + 1], omega)
            K1 = Kk + 1
            Q[K1] = (Kk * Q[Kk] - 0.5 * (P[Kk] + M[Kk])) / K1
            jj = np.arange(1, K1 + 1)[:, None, None]
            Sn[K1] = np.einsum("jbn,jbn->bn", jj * Q[1:K1 + 1], Co[K1 - 1::-1][:K1]) / K1
            Co[K1] = -k2 * np.einsum("jbn,jbn->bn", jj * Q[1:K1 + 1], Sn[K1 - 1::-1][:K1]) / K1
            SS[K1] = cs(Sn, Sn, K1); SG[K1] = cs(Sn, Co, K1)
            G[K1] = cs(A2, SS, K1)
            H[K1] = cs(A2, SG, K1)
        return P[:K + 1], M[:K + 1], A[:K + 1], C[:K + 1], Q[:K + 1]

    # ---------- shooting pieces ----------
    def _centre_to_node0(self, d, c0, omega):
        S = self.centre_series(d, c0, omega)
        s = self.trunc(np.stack([self.eval_series(S_, self.x0) for S_ in S]))
        return self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, omega)

    def _ssh_to_last(self, mS, aS, qS, omega):
        S = self.ssh_series(mS, aS, qS, omega)
        s = self.trunc(np.stack([self.eval_series(S_, self.u0) for S_ in S]))
        return self.seg(s, 1 - self.u0, self.nodes[-1], omega)

    # ---------- layout ----------
    def _cpar(self):
        """parities of the centre data (q, c0) and horizon data (m, a, psi)."""
        if self.full:
            return ["all", "all"], ["all", "all", "all"]
        return ["odd", "even"], ["odd", "even", "odd"]

    def layout(self):
        K = len(self.nodes)
        sizes = [self.size[p] for p in self.parities()]
        return sizes, K, sum(sizes) + 1

    def parities(self):
        K = len(self.nodes)
        cp, sp = self._cpar()
        return cp + list(self.PAR) * K + sp

    def tospec(self, s):
        return np.concatenate([self.spec(s[f], self.PAR[f]) for f in range(5)], axis=-1)

    def residual(self, Zb):
        sizes, K, L = self.layout()
        parts, Dl = self.split(Zb)
        pars = self.parities()
        F = [self.phys(c, p) for c, p in zip(parts, pars)]
        omega = 2 * np.pi / Dl
        states = [np.stack(F[2 + 5 * i:7 + 5 * i]) for i in range(K)]
        blocks = [self._centre_to_node0(F[0], F[1], omega) - states[0]]
        for i in range(K - 1):
            blocks.append(self.seg(states[i], self.nodes[i], self.nodes[i + 1], omega) - states[i + 1])
        blocks.append(self._ssh_to_last(F[-3], F[-2], F[-1], omega) - states[-1])
        mis = np.concatenate([self.tospec(b) for b in blocks], axis=1)
        phase = parts[0][:, self.phase_index]
        return np.concatenate([mis, phase[:, None]], axis=1)

    def _blk(self):
        return sum(self.size[p] for p in self.PAR)

    def _block_list(self):
        sizes, K, L = self.layout()
        offs = np.concatenate([[0], np.cumsum(sizes)])
        cp, sp = self._cpar()
        c1 = self.size[cp[0]]
        out = []

        def centre(X, om):
            return self._centre_to_node0(self.phys(X[:, :c1], cp[0]), self.phys(X[:, c1:], cp[1]), om)
        out.append((0, c1 + self.size[cp[1]], centre, 0))
        nb = self._blk()
        cut = np.cumsum([0] + [self.size[p] for p in self.PAR])
        for i in range(K - 1):
            def seg(X, om, i=i):
                st = np.stack([self.phys(X[:, cut[f]:cut[f + 1]], self.PAR[f]) for f in range(5)])
                return self.seg(st, self.nodes[i], self.nodes[i + 1], om)
            out.append((offs[2 + 5 * i], nb, seg, i + 1))
        s1, s2 = self.size[sp[0]], self.size[sp[0]] + self.size[sp[1]]

        def ssh(X, om):
            return self._ssh_to_last(self.phys(X[:, :s1], sp[0]), self.phys(X[:, s1:s2], sp[1]),
                                     self.phys(X[:, s2:], sp[2]), om)
        out.append((offs[2 + 5 * K], s2 + self.size[sp[2]], ssh, K - 1))
        return out, offs

    def jacobian(self, z, h=1e-20):
        """Residual and exact (complex-step) Jacobian."""
        sizes, K, L = self.layout()
        F0 = self.residual(z[None, :])[0]
        J = np.zeros((L, L))
        blocks, offs = self._block_list()
        blk = self._blk()
        node0 = offs[2]
        D = z[-1]
        for b, (c0, nf, fun, tgt) in enumerate(blocks):
            X = np.repeat(z[None, c0:c0 + nf], nf + 1, axis=0).astype(complex)
            X[:nf] += 1j * h * np.eye(nf)
            om = np.full(nf + 1, 2 * np.pi / D, dtype=complex); om[-1] = 2 * np.pi / (D + 1j * h)
            R = self.tospec(fun(X, om)).imag / h
            r0 = b * blk
            J[r0:r0 + blk, c0:c0 + nf] = R[:nf].T
            J[r0:r0 + blk, -1] = R[nf]
            t0 = node0 + blk * tgt
            J[r0:r0 + blk, t0:t0 + blk] -= np.eye(blk)
        J[-1, self.phase_index] = 1.0
        return F0, J

    def mode_matrix(self, z, lam, h=1e-20):
        """Linearised mismatch for perturbations ~ exp(lam tau), lam real, Delta fixed (complex step)."""
        sizes, K, L = self.layout()
        nv = L - 1
        Mm = np.zeros((nv, nv))
        blocks, offs = self._block_list()
        blk = self._blk()
        node0 = offs[2]
        D = z[-1]
        self.lam = float(lam)
        try:
            for b, (c0, nf, fun, tgt) in enumerate(blocks):
                X = np.repeat(z[None, c0:c0 + nf], nf + 1, axis=0).astype(complex)
                X[1:] += 1j * h * np.eye(nf)
                om = np.full(nf + 1, 2 * np.pi / D, dtype=complex)
                R = self.tospec(fun(X, om))[1:].imag / h
                r0 = b * blk
                Mm[r0:r0 + blk, c0:c0 + nf] = R.T
                t0 = node0 + blk * tgt
                Mm[r0:r0 + blk, t0:t0 + blk] -= np.eye(blk)
        finally:
            self.lam = 0.0
        return Mm

    def mode_matrix_c(self, z, lam, eps=1e-7):
        """As mode_matrix, for complex lam, by central differences (the perturbed members are complex)."""
        sizes, K, L = self.layout()
        nv = L - 1
        Mm = np.zeros((nv, nv), complex)
        blocks, offs = self._block_list()
        blk = self._blk()
        node0 = offs[2]
        D = z[-1]
        self.lam = complex(lam)
        try:
            for b, (c0, nf, fun, tgt) in enumerate(blocks):
                base = z[c0:c0 + nf]
                hh = eps * np.maximum(1.0, np.abs(base))
                X = np.repeat(base[None, :], 2 * nf + 1, axis=0).astype(complex)
                X[1:nf + 1] += np.diag(hh); X[nf + 1:] -= np.diag(hh)
                om = np.full(2 * nf + 1, 2 * np.pi / D, dtype=complex)
                R = self.tospec(fun(X, om))
                Rd = (R[1:nf + 1] - R[nf + 1:]) / (2 * hh[:, None])
                r0 = b * blk
                Mm[r0:r0 + blk, c0:c0 + nf] = Rd.T
                t0 = node0 + blk * tgt
                Mm[r0:r0 + blk, t0:t0 + blk] -= np.eye(blk)
        finally:
            self.lam = 0.0
        return Mm

    # ---------- conversions ----------
    def regrid_from(self, z, other):
        """Solution vector of another SigDSS (other kc and/or N, same nodes) on this one's harmonics."""
        parts, Dl = other.split(z[None, :])
        out = []
        for c, par in zip(parts, other.parities()):
            c = c[0]
            ko = other.kodd if par == "odd" else other.keven
            kn = self.kodd if par == "odd" else self.keven
            if par == "odd":
                mo = len(ko); re, im = dict(zip(ko, c[:mo])), dict(zip(ko, c[mo:]))
                out.append(np.concatenate([[re.get(k, 0.0) for k in kn], [im.get(k, 0.0) for k in kn]]))
            else:
                mo = len(ko) - 1; re = dict(zip(ko[1:], c[1:1 + mo])); im = dict(zip(ko[1:], c[1 + mo:]))
                out.append(np.concatenate([[c[0]], [re.get(k, 0.0) for k in kn[1:]], [im.get(k, 0.0) for k in kn[1:]]]))
        return np.concatenate(out + [Dl])

    def to_full(self, z, other):
        """Solution vector of a symmetric SigDSS `other` (same kc, N, nodes) in this full (no-parity) layout."""
        parts, Dl = other.split(z[None, :])
        out = [self.spec(other.phys(c, par), "all")[0] for c, par in zip(parts, other.parities())]
        return np.concatenate(out + [Dl])

    def wrong_parity_norm(self, z):
        """For a full-layout vector: norm of the harmonics that the half-period symmetry forbids (even harmonics of
        p, m, psi, q, mS, qS; odd harmonics of a, c, c0, aS), and the norm of the allowed ones."""
        parts, Dl = self.split(z[None, :])
        kc = self.kc
        sym = ["odd", "even"] + list(PAR) * len(self.nodes) + ["odd", "even", "odd"]
        bad = good = 0.0
        k = np.concatenate([[0], np.arange(1, kc + 1), np.arange(1, kc + 1)])
        for c, par in zip(parts, sym):
            c = c[0]
            wrong = (k % 2 == 0) if par == "odd" else (k % 2 == 1)
            bad += float(np.sum(c[wrong] ** 2)); good += float(np.sum(c[~wrong] ** 2))
        return np.sqrt(bad), np.sqrt(good)

    def fields(self, z):
        parts, Dl = self.split(z[None, :])
        F = [self.phys(c, p)[0] for c, p in zip(parts, self.parities())]
        K = len(self.nodes)
        return dict(d=F[0], c0=F[1], states=[np.stack(F[2 + 5 * i:7 + 5 * i]) for i in range(K)],
                    mS=F[-3], aS=F[-2], qS=F[-1], Delta=float(Dl[0]))

    def constraint(self, z):
        """max over nodes and tau of |psi_tau + (p + m)/2 - c (p - m)/(2x)| relative to max |p| (zero for an exact
        solution; a resolution diagnostic)."""
        f = self.fields(z)
        om = np.array([2 * np.pi / f["Delta"]])
        worst = 0.0
        for x, st in zip(self.nodes, f["states"]):
            p, m, a, c, q = st
            K = self.D(q[None, :], om)[0] + 0.5 * (p + m) - c * (p - m) / (2 * x)
            worst = max(worst, float(np.max(np.abs(K))))
        return worst / float(max(np.max(np.abs(s[0])) for s in f["states"]))

    def dF_dpar(self, z, which="n", h=1e-20):
        """d(residual)/d(n or eps) at fixed z, by complex step."""
        n0, e0 = self.n, self.eps
        if which == "n":
            self.n = n0 + 1j * h
        else:
            self.eps = e0 + 1j * h
        try:
            F = self.residual(z[None, :].astype(complex))[0]
        finally:
            self.n, self.eps = n0, e0
        return F.imag / h

    def gauge_lambda(self, z):
        """Growth rate of the horizon-relabelling (gauge) mode: mean over tau of 1 - c_x(tau, 1)."""
        f = self.fields(z)
        a2 = f["aS"] ** 2
        s_, _ = self.sg(f["qS"])
        return float(np.mean(1 - (self.n - 1) * (a2 - 1) + 2 * self.L * a2 * s_ ** 2))

    def u_amplitude(self, z):
        """max over tau and nodes of |u| = kappa |psi| (u = psi at eps = 0 is not meaningful; returns kappa max|psi|)."""
        f = self.fields(z)
        mx = max(float(np.max(np.abs(st[4]))) for st in f["states"])
        mx = max(mx, float(np.max(np.abs(f["qS"]))))
        return float(np.sqrt(self.k2)) * mx, mx


# ---------------- solvers (as in dss_ell.py) ----------------
def harmonic_weights(S, p=4):
    w = []
    for par in S.parities():
        if par == "all":
            ks = np.arange(S.kc + 1)
            k = np.concatenate([ks[:1], ks[1:], ks[1:]])
            w.append((k / S.kc) ** p)
            continue
        ks = S.kodd if par == "odd" else S.keven
        k = np.concatenate([ks, ks]) if par == "odd" else np.concatenate([ks[:1], ks[1:], ks[1:]])
        w.append((k / S.kc) ** p)
    return np.concatenate(w + [[0.0]])


def lm(S, z, tol=1e-12, maxjac=30, ncorr=2, verbose=True, mus=None, stall=3, prep=None, reg=0.0, regp=4):
    """Levenberg-Marquardt: one SVD per (complex-step) Jacobian, all damping trials (with chord corrections)
    evaluated as one batched residual call. Returns z, |F|, seconds. reg > 0: smoothness prior as in dss_ell.py."""
    if reg:
        return _lm_reg(S, z, tol, maxjac, ncorr, verbose, mus, prep, reg, regp)
    mus = mus if mus is not None else np.concatenate([[0.0], 10.0 ** np.arange(-12, 0.5, 1.0)])
    t0 = time.time()
    F = S.residual(z[None, :])[0]; nF = np.linalg.norm(F); bad = 0
    for it in range(maxjac):
        if prep is not None:
            prep(z)
            F = S.residual(z[None, :])[0]; nF = np.linalg.norm(F)
        if nF < tol:
            break
        _, J = S.jacobian(z)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        filt = np.stack([s / (s * s + (mu * s[0]) ** 2) for mu in mus])
        step = lambda Fb: -((U.T @ Fb.T).T * filt) @ Vt
        Zt = z[None, :] + step(np.repeat(F[None, :], len(mus), 0))
        best = (nF, None, None, 0)
        for c in range(ncorr + 1):
            Ft = S.residual(Zt); nt = np.linalg.norm(Ft, axis=1)
            for q in range(len(mus)):
                if np.isfinite(nt[q]) and nt[q] < best[0]:
                    best = (nt[q], Zt[q].copy(), mus[q], c)
            if c == ncorr:
                break
            Ft[~np.isfinite(nt)] = 0.0
            Zt = Zt + step(Ft)
        if best[1] is None:
            if verbose:
                print(f"   jac {it}: no decrease (|F| {nF:.3e}, cond {s[0] / s[-1]:.1e})", flush=True)
            break
        prev = nF
        z = best[1]; F = S.residual(z[None, :])[0]; nF = np.linalg.norm(F)
        bad = bad + 1 if nF > 0.5 * prev else 0
        if verbose:
            print(f"   jac {it}: |F| {nF:.3e} Delta {z[-1]:.12f} mu {best[2]:.0e} corr {best[3]} "
                  f"smin/smax {s[-1] / s[0]:.1e} ({time.time() - t0:.0f} s)", flush=True)
        if bad >= stall:
            break
    return z, nF, time.time() - t0


def _lm_reg(S, z, tol, maxjac, ncorr, verbose, mus, prep, reg, regp):
    mus = mus if mus is not None else np.concatenate([[0.0], 10.0 ** np.arange(-12, 0.5, 1.0)])
    t0 = time.time()
    w = harmonic_weights(S, regp)
    mu_r = None

    def R(Zb):
        Fb = S.residual(Zb)
        return np.concatenate([Fb, mu_r * w[None, :] * Zb], axis=1)
    nR = np.inf; small = 0
    for it in range(maxjac):
        if prep is not None:
            prep(z)
        F, J = S.jacobian(z)
        if mu_r is None:
            mu_r = reg * np.linalg.norm(J, 2)
        Rz = np.concatenate([F, mu_r * w * z]); nR = np.linalg.norm(Rz)
        Ja = np.vstack([J, mu_r * np.diag(w)])
        U, s, Vt = np.linalg.svd(Ja, full_matrices=False)
        filt = np.stack([s / (s * s + (mu * s[0]) ** 2) for mu in mus])
        step = lambda Rb: -((U.T @ Rb.T).T * filt) @ Vt
        Zt = z[None, :] + step(np.repeat(Rz[None, :], len(mus), 0))
        best = (nR, None, None, 0)
        for c in range(ncorr + 1):
            Rt = R(Zt); nt = np.linalg.norm(Rt, axis=1)
            for q in range(len(mus)):
                if np.isfinite(nt[q]) and nt[q] < best[0]:
                    best = (nt[q], Zt[q].copy(), mus[q], c)
            if c == ncorr:
                break
            Rt[~np.isfinite(nt)] = 0.0
            Zt = Zt + step(Rt)
        if best[1] is None:
            break
        gain = (nR - best[0]) / nR
        z = best[1]
        if verbose:
            nF = np.linalg.norm(S.residual(z[None, :])[0])
            print(f"   jac {it}: |R| {best[0]:.3e} |F| {nF:.3e} Delta {z[-1]:.12f} mu {best[2]:.0e} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        small = small + 1 if gain < 1e-4 else 0
        if small >= 2 or best[0] < tol:
            break
    nF = np.linalg.norm(S.residual(z[None, :])[0])
    return z, nF, time.time() - t0


def lam_root(S, z, guess, widths=(0.02, 0.06, 0.15, 0.35), xtol_rel=2e-9):
    """Real root of det M(lam) near guess: sign change of slogdet on guess*(1 +- w), refined by brentq."""
    from scipy.optimize import brentq
    cache = {}

    def sd(l):
        if l not in cache:
            cache[l] = np.linalg.slogdet(S.mode_matrix(z, l))
        return cache[l]
    for w in widths:
        grid = guess * (1 + w * np.array([-1.0, 0.0, 1.0]))
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(2) if sg[i] != sg[i + 1]]
        if ch:
            i = ch[0] if len(ch) == 1 else min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - guess))
            ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            r = brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=xtol_rel * guess,
                       rtol=1e-13)
            return r, len(cache), len(ch)
    return None, len(cache), 0


def real_roots(S, z, lo, hi, ngrid=9, xtol=1e-10):
    """All sign changes of det M(lam) on a grid in [lo, hi], refined by brentq."""
    from scipy.optimize import brentq
    cache = {}

    def sd(lam):
        if lam not in cache:
            cache[lam] = np.linalg.slogdet(S.mode_matrix(z, lam))
        return cache[lam]
    grid = np.linspace(lo, hi, ngrid)
    sg = [sd(l)[0] for l in grid]
    roots = []
    for i in range(ngrid - 1):
        if sg[i] != sg[i + 1]:
            ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            g = lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref)
            roots.append(brentq(g, grid[i], grid[i + 1], xtol=xtol, rtol=1e-14))
    return roots, len(cache)


def make(D, eps, z, kc=15, Ks=16, u0=0.01, kappa=0.7, rk=1, N=None, nodes=None):
    """A SigDSS for (D, eps) with RK steps frozen from z."""
    S = SigDSS(kc=kc, nodes=nodes or NODES_STD, n=D - 2, eps=eps, Delta_ref=z[-1], Ks=Ks, u0=u0, rk=rk, N=N)
    S.freeze_from(z, kappa)
    return S
