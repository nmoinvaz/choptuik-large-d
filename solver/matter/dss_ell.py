"""
DSS critical solution of the massless scalar l-multiplet in D = n + 2 dimensions (periodic BVP).

Model and equations: matter/ELL.md and solver/matter/ell_equations.py (sympy). Fields p = r(Phi + Pi), m = r(Phi - Pi),
a, c = alpha/a and the field psi itself (phi_m = psi Y_m, sum_m Y_m^2 = 1, L = l(l + n - 1)):

  a_x   = a [p^2 + m^2 + 2 L a^2 psi^2 + (n-1)(1 - a^2)] / (2x)
  c_x   = c [(n-1)(a^2 - 1) - 2 L a^2 psi^2] / x
  psi_x = (p + m) / (2x)
  p_tau + (x - c) p_x =  c_x p - c p/x + n c (p + m)/(2x) - L c a^2 psi / x
  m_tau + (x + c) m_x = -c_x m + c m/x - n c (p + m)/(2x) + L c a^2 psi / x

At L = 0 the (p, m, a, c) part is solver/dss.py exactly (psi decouples). Parities under tau -> tau + Delta/2: p, m,
psi odd, a, c even.

Boundary data:
  centre (x = 0): psi ~ q(tau) x^l, p ~ m ~ l q x^l (l >= 1), free q(tau) and c0(tau) = c(tau, 0);
                  for l = 0 the free datum is p1(tau) (p ~ p1 x, m ~ -p1 x) as in dss.py, and psi(tau, 0) is fixed by
                  the constraint psi_tau = c Pi at the centre.
  horizon (x = 1, c = 1): free m(tau, 1), a(tau, 1), psi(tau, 1); p(tau, 1) from the regularity condition
                  p_tau + (1 - c_x(1) - n/2) p = (n/2) m - L a^2 psi.
The constraint K = psi_tau + (p + m)/2 - c (p - m)/(2x) is x-independent (sympy); it vanishes on the centre side by
construction and is enforced on the horizon side by the matching of psi, so the system is square.

Discretisation as in solver/spec.py (dealiased Fourier in tau up to harmonic kc, multiple shooting in x, RK4), with
frozen RK step counts (independent of Delta) and every derivative (Jacobian, growth-mode matrix) by complex step.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from spec import DSSSpec                      # noqa: E402

NODES_STD = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
PAR = ("odd", "odd", "even", "even", "odd")      # p, m, a, c, psi


def _rfft_op(N, mult):
    """Dense matrix of the linear map f -> irfft(rfft(f) * mult) acting on rows: f @ Mat."""
    E = np.eye(N)
    F = np.fft.rfft(E, axis=-1) * mult
    return np.fft.irfft(F, n=N, axis=-1)


class DSSEll(DSSSpec):
    def __init__(self, kc, nodes, n, ell, Delta_ref, Kc=None, Ks=8, u0=0.03, long=False, x0=1e-4, seg_h=0.002,
                 log_steps=150, rk=1, N=None, margin=1.04):
        N = N or int(4 * np.ceil((3 * kc + 2) / 4))
        k = np.arange(N // 2 + 1)
        Dmult = 1j * k; Dmult[-1] = 0.0
        self._Dm = _rfft_op(N, Dmult)                      # d/dtau, omega = 1
        self._Tm = _rfft_op(N, (k <= kc).astype(float))    # truncation to k <= kc
        self.ell = int(ell)
        Kc = Kc if Kc is not None else (4 if self.ell == 0 else self.ell + 8)
        if long:
            u0 = 1.0 - nodes[-1]
        super().__init__(kc=kc, nodes=nodes, n=n, N=N, Ks=Ks, u0=u0, x0=x0, Kc=Kc, seg_h=seg_h, log_steps=log_steps)
        self.long = long
        self.L = self.ell * (self.ell + self.n - 1)
        self.rk = int(rk)
        self.set_delta_ref(Delta_ref, margin)
        # spectral <-> physical as matrices (so complex-step perturbations pass through)
        self._B = {p: self._phys_fft(np.eye(self.n_odd if p == "odd" else self.n_even), p) for p in ("odd", "even")}
        self._S = {p: self._spec_fft(np.eye(self.N), p) for p in ("odd", "even")}
        inv = np.zeros(N // 2 + 1, complex)
        inv[1:-1] = 1.0 / (1j * k[1:-1])                     # 1/(ik); mean and Nyquist removed
        self._Dpinv = _rfft_op(N, inv)                      # (a pinv of _Dm would invert its round-off null values)

    # ---------- linear maps ----------
    def _phys_fft(self, c, parity):
        return DSSSpec.phys(self, c, parity)

    def _spec_fft(self, f, parity):
        return DSSSpec.spec(self, f, parity)

    def phys(self, c, parity):
        return c @ self._B[parity]

    def spec(self, f, parity):
        return f @ self._S[parity]

    def _D_unit(self, f):
        return f @ self._Dm

    def trunc(self, f):
        return f @ self._Tm

    def solve_periodic(self, kappa, rhs, omega, bg=None):
        """(omega d/dtau + kappa) f = rhs, batchwise; members b > 0 get + lam (f - f_0) when self.lam != 0."""
        B, N = rhs.shape
        dt = np.result_type(kappa, rhs, omega)
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

    def antideriv(self, rhs, omega):
        """psi with (omega d/dtau [+ lam]) psi = rhs, zero mean (rhs odd): the l = 0 centre value of psi."""
        out = (rhs @ self._Dpinv) / omega[:, None]          # psi @ Dm * omega = rhs (rows convention)
        if self.lam != 0.0 and rhs.shape[0] > 1:
            N = self.N
            dt = np.result_type(rhs, omega)
            Mt = omega[1:, None, None] * self._Dm.T[None, :, :].astype(dt)
            idx = np.arange(N); Mt[:, idx, idx] += self.lam
            out = out.astype(dt)
            out[1:] = np.linalg.solve(Mt, (rhs[1:] + self.lam * out[0])[..., None])[..., 0]
        return out

    # ---------- RK step counts frozen at a reference Delta ----------
    def set_delta_ref(self, Delta_ref, margin=1.04):
        self.Delta_ref = float(Delta_ref)
        self.om_ref = margin * 2 * np.pi / self.Delta_ref

    def refreeze(self, z, kappa=0.5, tol=0.15):
        """freeze_from, but keep the current step counts unless one of them would change by more than tol (each
        change of the counts moves the discrete residual by ~1e-8, which stalls Newton if done every iteration)."""
        old = getattr(self, "steps_map", None)
        new = self.freeze_from(z, kappa)
        if old is not None and set(old) == set(new) and all(abs(new[k] - old[k]) <= tol * old[k] for k in old):
            self.steps_map = old
        return self.steps_map

    def freeze_from(self, z, kappa=0.5):
        """Freeze the RK4 step count of every segment from the stiffness of the transport terms of solution z:
        lambda = kc omega / |x - c| (in d/dx; times x in ln x, times 1 - x in ln(1 - x)), steps = length * max
        lambda / kappa (times rk). Needed for l >= 1, where c(tau, 0) dips far below 0.1 and the omega-only rule
        of solver/spec.py under-resolves the centre. The counts stay fixed while Newton iterates (no Delta jumps)."""
        f = self.fields(z)
        om = 2 * np.pi / f["Delta"] * 1.02
        kw = self.kc * om
        xs = [0.0] + list(self.nodes)
        cs = [f["c0"]] + [st[3] for st in f["states"]]

        def gap(i):
            return max(1e-3, float(np.min(np.abs(xs[i] - cs[i]))))
        st = {}
        # centre series start -> node 0 in ln x
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
        if not self.long:
            # horizon segment 1 - u0 -> last node, in ln(1 - x); near x = 1, |x - c| ~ u (1 + c_1)
            a2 = f["aS"] ** 2
            c1 = -(self.n - 1) * (a2 - 1) + 2 * self.L * a2 * f["qS"] ** 2
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
        g = a2 * q * q
        cx = c * ((n - 1) * (a2 - 1) - 2 * L * g) / x
        ax = a * ((p * p + m * m) + 2 * L * g + (n - 1) * (1 - a2)) / (2 * x)
        qx = (p + m) / (2 * x)
        pc = n * c * (p + m) / (2 * x)
        lt = L * c * a2 * q / x
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
        """Taylor series at x = 0. d = q (psi ~ q x^l) for l >= 1, d = p1 for l = 0. Returns P, M, A, C, Q lists."""
        n, L, l, K = self.n, self.L, self.ell, self.Kc
        cs = self._cs
        Z = np.zeros_like(d * c0)
        P = [Z.copy() for _ in range(K + 1)]; M = [Z.copy() for _ in range(K + 1)]
        A = [Z.copy() for _ in range(K + 1)]; C = [Z.copy() for _ in range(K + 1)]
        Q = [Z.copy() for _ in range(K + 1)]
        A[0] = np.ones_like(Z); C[0] = c0 + 0 * Z
        if l == 0:
            P[1] = d + 0 * Z; M[1] = -d + 0 * Z
            Q[0] = self.antideriv(C[0] * P[1], omega)        # psi_tau(tau, 0) = c Pi(tau, 0)
            kstart = 2
        else:
            P[l] = l * d + 0 * Z; M[l] = l * d + 0 * Z; Q[l] = d + 0 * Z
            kstart = 1
        A2 = [None] * (K + 1); Q2 = [None] * (K + 1); QQ = [None] * (K + 1)
        A2[0] = A[0] * A[0]
        for k in range(kstart, K + 1):
            if l >= 1 and k == l:
                # free order; still need the metric at this order
                pass
            # a_k
            Q2 = [cs(P, P, j) + cs(M, M, j) for j in range(k + 1)]
            QQ = [cs(Q, Q, j) for j in range(k + 1)]
            s = cs(A, Q2, k)
            A2 = [cs(A, A, j) for j in range(k + 1)]
            t3 = cs(A, A2, k)                                   # with A[k] = 0
            A2Q = [cs(A2, QQ, j) for j in range(k + 1)]
            u = cs(A, A2Q, k)                                   # [a^3 psi^2]_k (A[k] does not enter: psi^2 = O(x^2l))
            A[k] = (0.5 * s + L * u - 0.5 * (n - 1) * t3) / (k + n - 1)
            A2 = [cs(A, A, j) for j in range(k + 1)]
            A2m1 = [A2[j] - (1.0 if j == 0 else 0.0) for j in range(k + 1)]
            G = [cs(A2, QQ, j) for j in range(k + 1)]
            C[k] = ((n - 1) * cs(C, A2m1, k) - 2 * L * cs(C, G, k)) / k
            if l >= 1 and k <= l:
                continue                                        # P, M, Q zero below l; order l is the free datum
            # residuals of x*E3, x*E4 at order k with P[k] = M[k] = Q[k] = 0
            PM = [P[j] + M[j] for j in range(K + 1)]
            Dp = self.D(P[k - 1], omega); Dmm = self.D(M[k - 1], omega)
            X = [Z, -C[0]] + [(1.0 if j == 2 else 0.0) - C[j - 1] for j in range(2, k + 2)]
            Y = [Z, C[0]] + [(1.0 if j == 2 else 0.0) + C[j - 1] for j in range(2, k + 2)]
            xpx = sum(X[j] * (k - j + 1) * P[k - j + 1] for j in range(1, k + 1))
            xmx = sum(Y[j] * (k - j + 1) * M[k - j + 1] for j in range(1, k + 1))
            xcxp = sum(j * C[j] * P[k - j] for j in range(1, k + 1))
            xcxm = sum(j * C[j] * M[k - j] for j in range(1, k + 1))
            cp = cs(C, P, k); cm = cs(C, M, k); cpm = cs(C, PM, k)
            H = [cs(A2, Q, j) for j in range(k + 1)]            # a^2 psi
            ch = cs(C, H, k)
            r3 = Dp + xpx - xcxp + cp - 0.5 * n * cpm + L * ch
            r4 = Dmm + xmx + xcxm - cm + 0.5 * n * cpm - L * ch
            c = C[0]
            al = 0.5 * n - L / (2.0 * k)
            m11 = c * (1 - k - al); m12 = -c * al
            m21 = c * al;           m22 = c * (k - 1 + al)
            det = m11 * m22 - m12 * m21
            P[k] = (-r3 * m22 + r4 * m12) / det
            M[k] = (-r4 * m11 + r3 * m21) / det
            Q[k] = (P[k] + M[k]) / (2.0 * k)
        return P, M, A, C, Q

    def ssh_series(self, m0, a0, q0, omega):
        """Taylor series in u = 1 - x at the horizon (c = 1 there). Returns P, M, A, C, Q lists (orders 0..Ks)."""
        n, L, K = self.n, self.L, self.Ks
        B, N = m0.shape
        dt = np.result_type(m0, a0, q0, omega, float)
        shp = (K + 2, B, N)
        P = np.zeros(shp, dt); M = np.zeros(shp, dt); A = np.zeros(shp, dt); C = np.zeros(shp, dt)
        Q = np.zeros(shp, dt)
        A2 = np.zeros(shp, dt); Q2 = np.zeros(shp, dt); QQ = np.zeros(shp, dt)
        G = np.zeros(shp, dt); H = np.zeros(shp, dt)                 # a^2 psi^2, a^2 psi
        DP = np.zeros(shp, dt); DM = np.zeros(shp, dt)
        cs = lambda X, Y, k: np.einsum("jbn,jbn->bn", X[:k + 1], Y[k::-1] if k > 0 else Y[:1])
        A[0] = a0; C[0] = 1.0; M[0] = m0; Q[0] = q0
        A2[0] = A[0] * A[0]; QQ[0] = Q[0] * Q[0]; G[0] = A2[0] * QQ[0]; H[0] = A2[0] * Q[0]
        DM[0] = self.D(M[0], omega)
        for Kk in range(K + 1):
            # c_{K+1}
            C[Kk + 1] = (Kk * C[Kk] - (n - 1) * (cs(C, A2, Kk) - C[Kk]) + 2 * L * cs(C, G, Kk)) / (Kk + 1)
            # p_K
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
            # a_{K+1}
            aG = cs(A, Q2, Kk) + (n - 1) * (A[Kk] - cs(A, A2, Kk)) + 2 * L * cs(A, G, Kk)
            A[Kk + 1] = (Kk * A[Kk] - 0.5 * aG) / (Kk + 1)
            A2[Kk + 1] = cs(A, A, Kk + 1)
            # m_{K+1}
            zc = C[:Kk + 2].copy(); zc[0] += 1.0; zc[1] -= 1.0
            Zc = np.empty_like(zc); Zc[0] = zc[0]; Zc[1:] = zc[1:] - zc[:-1]
            tau_term = DM[Kk] - (DM[Kk - 1] if Kk >= 1 else 0.0)
            PM = P[:Kk + 1] + M[:Kk + 1]
            zmu = np.einsum("ibn,ibn->bn", i[:, None, None] * M[:Kk + 1], Zc[Kk + 1:0:-1])
            rest = tau_term - zmu - cs(V, M, Kk) - cs(C, M, Kk) + 0.5 * n * cs(C, PM, Kk) - L * cs(C, H, Kk)
            M[Kk + 1] = rest / (Zc[0] * (Kk + 1))
            DM[Kk + 1] = self.D(M[Kk + 1], omega)
            # psi_{K+1}
            Q[Kk + 1] = (Kk * Q[Kk] - 0.5 * (P[Kk] + M[Kk])) / (Kk + 1)
            QQ[Kk + 1] = cs(Q, Q, Kk + 1)
            G[Kk + 1] = cs(A2, QQ, Kk + 1)
            H[Kk + 1] = cs(A2, Q, Kk + 1)
        return P[:K + 1], M[:K + 1], A[:K + 1], C[:K + 1], Q[:K + 1]

    # ---------- shooting pieces ----------
    def _centre_to_node0(self, d, c0, omega):
        S = self.centre_series(d, c0, omega)
        s = self.trunc(np.stack([self.eval_series(S_, self.x0) for S_ in S]))
        return self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, omega)

    def _ssh_to_last(self, mS, aS, qS, omega):
        S = self.ssh_series(mS, aS, qS, omega)
        s = self.trunc(np.stack([self.eval_series(S_, self.u0) for S_ in S]))
        if self.long:
            return s
        return self.seg(s, 1 - self.u0, self.nodes[-1], omega)

    # ---------- layout ----------
    def layout(self):
        K = len(self.nodes)
        no, ne = self.n_odd, self.n_even
        sizes = [no, ne] + [no, no, ne, ne, no] * K + [no, ne, no]
        return sizes, K, sum(sizes) + 1

    def parities(self):
        K = len(self.nodes)
        return ["odd", "even"] + list(PAR) * K + ["odd", "even", "odd"]

    def tospec(self, s):
        return np.concatenate([self.spec(s[f], PAR[f]) for f in range(5)], axis=-1)

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
        phase = parts[0][:, len(self.kodd)]
        return np.concatenate([mis, phase[:, None]], axis=1)

    def _block_list(self):
        """(input offset, input size, function(X, omega) -> end state, target node index) per shooting block."""
        sizes, K, L = self.layout()
        offs = np.concatenate([[0], np.cumsum(sizes)])
        no, ne = self.n_odd, self.n_even
        out = []

        def centre(X, om):
            return self._centre_to_node0(self.phys(X[:, :no], "odd"), self.phys(X[:, no:], "even"), om)
        out.append((0, no + ne, centre, 0))
        nb = 3 * no + 2 * ne
        cut = np.cumsum([0, no, no, ne, ne, no])
        for i in range(K - 1):
            def seg(X, om, i=i):
                st = np.stack([self.phys(X[:, cut[f]:cut[f + 1]], PAR[f]) for f in range(5)])
                return self.seg(st, self.nodes[i], self.nodes[i + 1], om)
            out.append((offs[2 + 5 * i], nb, seg, i + 1))

        def ssh(X, om):
            return self._ssh_to_last(self.phys(X[:, :no], "odd"), self.phys(X[:, no:no + ne], "even"),
                                     self.phys(X[:, no + ne:], "odd"), om)
        out.append((offs[2 + 5 * K], 2 * no + ne, ssh, K - 1))
        return out, offs

    def jacobian(self, z, h=1e-20):
        """Residual and exact (complex-step) Jacobian."""
        sizes, K, L = self.layout()
        F0 = self.residual(z[None, :])[0]
        J = np.zeros((L, L))
        blocks, offs = self._block_list()
        blk = 3 * self.n_odd + 2 * self.n_even
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
        J[-1, len(self.kodd)] = 1.0
        return F0, J

    def mode_matrix(self, z, lam, h=1e-20):
        """Linearised mismatch for perturbations ~ exp(lam tau), Delta fixed (complex step)."""
        sizes, K, L = self.layout()
        nv = L - 1
        Mm = np.zeros((nv, nv))
        blocks, offs = self._block_list()
        blk = 3 * self.n_odd + 2 * self.n_even
        node0 = offs[2]
        D = z[-1]
        self.lam = lam
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

    # ---------- conversions ----------
    def regrid_from(self, z, other):
        """Solution vector of another DSSEll (other kc and/or N, same nodes) on this one's harmonics."""
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

    def from_dss4(self, z4, S4):
        """l = 0 seed in the 4-field DSSSpec layout (same nodes, kc) -> this layout with psi = 0 (it decouples at L = 0)."""
        parts, Dl = S4.split(z4[None, :]); parts = [p[0] for p in parts]
        K = len(self.nodes)
        zo = np.zeros(self.n_odd)
        out = [parts[0], parts[1]]
        for i in range(K):
            out += parts[2 + 4 * i:6 + 4 * i] + [zo]
        out += [parts[-2], parts[-1], zo]
        return np.concatenate(out + [Dl])

    def fields(self, z):
        """Physical-space boundary data and node states of solution z."""
        parts, Dl = self.split(z[None, :])
        F = [self.phys(c, p)[0] for c, p in zip(parts, self.parities())]
        K = len(self.nodes)
        return dict(d=F[0], c0=F[1], states=[np.stack(F[2 + 5 * i:7 + 5 * i]) for i in range(K)],
                    mS=F[-3], aS=F[-2], qS=F[-1], Delta=float(Dl[0]))

    def constraint(self, z):
        """max over nodes and tau of |psi_tau + (p + m)/2 - c (p - m)/(2x)| (zero for an exact solution), relative
        to max |p|."""
        f = self.fields(z)
        om = np.array([2 * np.pi / f["Delta"]])
        worst = 0.0
        for x, st in zip(self.nodes, f["states"]):
            p, m, a, c, q = st
            K = self.D(q[None, :], om)[0] + 0.5 * (p + m) - c * (p - m) / (2 * x)
            worst = max(worst, float(np.max(np.abs(K))))
        return worst / float(max(np.max(np.abs(s[0])) for s in f["states"]))

    def dF_dn(self, z, h=1e-20):
        """d(residual)/dn at fixed z, by complex step in the dimension (the continuation tangent's right side)."""
        n0, L0 = self.n, self.L
        self.n = n0 + 1j * h
        self.L = self.ell * (self.ell + self.n - 1)
        try:
            F = self.residual(z[None, :].astype(complex))[0]
        finally:
            self.n, self.L = n0, L0
        return F.imag / h

    def gauge_lambda(self, z):
        """Growth rate of the horizon-relabelling (gauge) mode: mean over tau of 1 - c_x(tau, 1)."""
        f = self.fields(z)
        a2 = f["aS"] ** 2
        return float(np.mean(1 - (self.n - 1) * (a2 - 1) + 2 * self.L * a2 * f["qS"] ** 2))


# ---------------- solvers ----------------
def harmonic_weights(S, p=4):
    """(k/kc)^p on every spectral unknown, 0 for Delta (as solver/newton_tsvd.py, for the 5-field layout)."""
    w = []
    for par in S.parities():
        ks = S.kodd if par == "odd" else S.keven
        k = np.concatenate([ks, ks]) if par == "odd" else np.concatenate([ks[:1], ks[1:], ks[1:]])
        w.append((k / S.kc) ** p)
    return np.concatenate(w + [[0.0]])


def lm(S, z, tol=1e-12, maxjac=30, ncorr=2, verbose=True, mus=None, stall=3, prep=None, reg=0.0, regp=4,
       return_jac=False):
    """Levenberg-Marquardt: one SVD per (complex-step) Jacobian, all damping trials (with chord corrections)
    evaluated as one batched residual call. Returns z, |F|, seconds.
    reg > 0: weighted damping, steps minimise |F + J dz|^2 + mu^2 |W dz|^2 with W = (k/kc)^regp and
    mu = reg * sigma_max(J): moves along the near-null high harmonics of the centre data (the 'junk' directions of
    solver/notes.md) are suppressed, and the iteration still converges to an exact zero of F."""
    if reg:
        return _lm_reg(S, z, tol, maxjac, ncorr, verbose, mus, prep, reg, regp, return_jac)
    mus = mus if mus is not None else np.concatenate([[0.0], 10.0 ** np.arange(-12, 0.5, 1.0)])
    t0 = time.time()
    F = S.residual(z[None, :])[0]; nF = np.linalg.norm(F); bad = 0
    for it in range(maxjac):
        if prep is not None:                     # e.g. re-freeze the RK steps from the current iterate
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


def _lm_reg(S, z, tol, maxjac, ncorr, verbose, mus, prep, reg, regp, return_jac=False):
    """Weighted-damping LM: each step minimises |F + J dz|^2 + mu^2 |W dz|^2 (W = (k/kc)^regp, mu = reg sigma_max),
    i.e. it penalises moving the near-null high harmonics ('junk' directions) instead of their values, so the
    iteration converges to an exact zero of F without drifting along them."""
    mus = mus if mus is not None else np.array([0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    ncorr = min(ncorr, 1)
    Jlast = None
    t0 = time.time()
    w = harmonic_weights(S, regp)
    mu_r = None
    nF = np.inf; small = 0
    for it in range(maxjac):
        if prep is not None:
            prep(z)
        F, J = S.jacobian(z)
        Jlast = J
        nF = np.linalg.norm(F)
        if nF < tol:
            break
        if mu_r is None:
            mu_r = reg * np.linalg.norm(J, 2)
        Ja = np.vstack([J, mu_r * np.diag(w)])
        U, s, Vt = np.linalg.svd(Ja, full_matrices=False)
        filt = np.stack([s / (s * s + (mu * s[0]) ** 2) for mu in mus])
        pad = np.zeros(len(w))
        step = lambda Fb: -((U.T @ np.concatenate([Fb, np.repeat(pad[None, :], len(Fb), 0)], axis=1).T).T * filt) @ Vt
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
            break
        gain = (nF - best[0]) / nF
        z = best[1]; nF = best[0]
        if verbose:
            print(f"   jac {it}: |F| {nF:.3e} Delta {z[-1]:.12f} mu {best[2]:.0e} ({time.time() - t0:.0f} s)", flush=True)
        small = small + 1 if gain < 0.1 else 0
        if small >= 3 or nF < tol:
            break
    nF = np.linalg.norm(S.residual(z[None, :])[0])
    if return_jac:
        return z, nF, time.time() - t0, Jlast
    return z, nF, time.time() - t0


def growth_rate(S, z, lo, hi, ngrid=9, xtol=1e-11, verbose=False):
    """Real roots of det M(lam) in [lo, hi]: sign changes of slogdet on a grid, refined by brentq."""
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
            if verbose:
                print(f"   root in [{grid[i]:.4f}, {grid[i + 1]:.4f}]: {roots[-1]:.10f}", flush=True)
    return roots, len(cache)


def smin_curve(S, z, lams):
    """Smallest singular value of M(lam) / largest, on a list of lam (to see near-roots without sign change)."""
    out = []
    for l in lams:
        s = np.linalg.svd(S.mode_matrix(z, l), compute_uv=False)
        out.append(s[-1] / s[0])
    return np.array(out)
