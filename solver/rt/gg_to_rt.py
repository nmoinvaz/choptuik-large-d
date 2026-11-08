"""The map from our gauge (hp/src/model.rs: x, tau, p, m, a, c, Delta) to the Reiterer-Trubowitz gauge (t, xi, mu,
omega_i), applied to an hp state.

Our metric (README, solver/dss.py): ds^2 = e^{-2 tau} [a^2 ((x^2 - c^2) dtau^2 - 2 x dtau dx + dx^2) + x^2 dOmega_n^2],
null directions dx/dtau = x - c (ingoing, holding the horizon x = 1) and x + c (outgoing). RT's null coordinates are
u^s = -(1 + s xi) e^{-mu t} with u^s o Theta = e^{-2 pi mu} u^s for the self-similarity map Theta: tau -> tau + Delta/2,
and u^- analytic with du^- != 0 across the horizon (RT Sec. 1.4). Writing u^s = -e^{-lam tau} w^s(tau, x) with w^s
periodic in tau, lam = 4 pi mu / Delta, the conditions are the linear transport equations

    w^-_tau + (x - c) w^-_x = lam w^-,   regular at x = 1 (so w^-(tau, 1) = 0),
    w^+_tau + (x + c) w^+_x = lam w^+,   w^+(tau, 0) = w^-(tau, 0)  (u^+ = u^- on the centre).

At x = 1, w^- = (1 - x) h with h_tau = (lam - 1 + c_x(tau, 1)) h, so a periodic h != 0 needs
lam = 1 - <c_x(1)> = 1 - (n - 1) <a_h^2 - 1>, i.e.

    Delta = 4 pi mu / (1 - (n - 1) <a_h^2 - 1>)        (EEG Eq. (3) has <psi_{-1}^2> in place of <c_x(1)>).

Then xi = (w^+ - w^-)/(w^+ + w^-), t = (4 pi / Delta) tau - ln((w^+ + w^-)/2)/mu, and with A^+ = -mu (w^+ + w^-)/(2 c w^-_x),
A^- = -mu (w^+ + w^-)/(2 c w^+_x), D^s = A^s (d_tau + (x + s c) d_x):
    e^{-zeta} = mu e^{-tau} a ((w^+ + w^-)/2) / sqrt(-w^+_x w^-_x),   mu + Q xi^2 = xi e^{-zeta} / (e^{-tau} x),
    omega_2^s = mu / xi - s A^s c / x,   omega_3^s = D^s zeta - s mu,   omega_1 = 2 xi Q + omega_2^+ + omega_2^-,
    omega_4^+ = sqrt(2) A^+ c p / x,   omega_4^- = -sqrt(2) A^- c m / x
(our scalar has Ric = 2n dphi dphi, EEG's Eq. (5) Ric = n dphi dphi: phi_EEG = sqrt(2) phi; p = (x/c)(d_tau + (x + c) d_x)phi,
m = -(x/c)(d_tau + (x - c) d_x)phi).

    python3 gg_to_rt.py profile STATE.npz OUT.npz [Ng]      (venv with hpkernel: dense profiles at Chebyshev x)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cheb01(Ng):
    """Lobatto points on [0, 1], x_0 = 0, x_Ng = 1, and d/dx."""
    from rt_model import cheb
    y, Dy = cheb(Ng)
    return (1 - y) / 2, -2 * Dy


def dump_profile(state, out, Ng=48, h=0.002):
    """p, m, a, c of an hp state at the Chebyshev points of [0, 1]: the centre series below the first node, the horizon
    series above the last, Taylor steps of at most h from the node below in between (hp/largeD_profiles.py's 0.01
    diverged near x = 0.15 at D = 4). Each node is also reached from the one below and compared, as a check."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
    from largeD_profiles import solver_for
    x, _ = cheb01(Ng)
    s, zh, zl, meta = solver_for(state)
    M, nodes = meta["M"], meta["nodes"]
    K = len(nodes)
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    hh, hl = s.series(zh, zl, 1)
    Hs = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
    node = lambda b: (zh[2 * M + 4 * M * b:2 * M + 4 * M * (b + 1)], zl[2 * M + 4 * M * b:2 * M + 4 * M * (b + 1)])

    def step(b, xe):
        sh, sl = node(b)
        pts = list(np.linspace(nodes[b], xe, max(2, int(np.ceil((xe - nodes[b]) / h)) + 1)))
        rh, rl = s.integrate(zh, zl, sh.copy(), sl.copy(), pts)
        return (np.asarray(rh) + np.asarray(rl)).reshape(4, M)

    hor = lambda xx: np.einsum("k,kqm->qm", (1 - xx) ** np.arange(Hs.shape[0]), Hs)
    F = []
    for xx in x:
        if xx <= nodes[0]:
            F.append(np.einsum("k,kqm->qm", xx ** np.arange(C.shape[0]), C))
        elif xx >= nodes[-1]:
            F.append(hor(xx))
        else:
            F.append(step(max(i for i in range(K) if nodes[i] <= xx), xx))
    F = np.array(F)
    gaps = [float(np.abs(step(b, nodes[b + 1]).ravel() - sum(node(b + 1))).max()) for b in range(K - 1)]
    gaps.append(float(np.abs(hor(nodes[-1]).ravel() - sum(node(K - 1))).max()))
    assert np.all(np.isfinite(F)), "profile not finite"
    print(f"  profile {state}: D {meta['D']} M {M}, node gaps max {max(gaps):.2e}")
    z = zh + zl
    np.savez(out, D=meta["D"], Delta=float(z[-1]), M=M, x=x, p=F[:, 0].T, m=F[:, 1].T, a=F[:, 2].T, c=F[:, 3].T,
             state=state, gaps=np.array(gaps))


# ---------------------------------------------------------------- spectral helpers on the half tau grid
def tau_k(M, T, odd):
    k = np.fft.fftfreq(M, 1.0 / M)
    if odd:
        return 2 * np.pi * (k + 0.5) / T
    kk = 2 * np.pi * k / T
    return kk


def dtau_mat(M, T, odd):
    """Dense d/dtau on M points over [0, T) (period T, or antiperiod T if odd)."""
    E = np.eye(M)
    return np.array([dtau(E[i], T, odd) for i in range(M)]).T


def dtau(f, T, odd, axis=0):
    f = np.asarray(f, dtype=float)
    M = f.shape[axis]
    tt = np.arange(M) * T / M
    sh = [1] * f.ndim
    sh[axis] = M
    k = tau_k(M, T, odd)
    if not odd:
        k = k.copy()
        k[M // 2] = 0.0 if M % 2 == 0 else k[M // 2]
        return np.fft.ifft(np.fft.fft(f, axis=axis) * (1j * k).reshape(sh), axis=axis).real
    ph = np.exp(-1j * np.pi * tt / T).reshape(sh)
    return (np.fft.ifft(np.fft.fft(f * ph, axis=axis) * (1j * k).reshape(sh), axis=axis) / ph).real


class Interp2:
    """Fourier (half tau grid) x Chebyshev (x in [0, 1]) interpolant with first derivatives."""

    def __init__(self, f, T, odd, Ng):
        M = f.shape[0]
        self.T, self.odd, self.M, self.Ng = T, odd, M, Ng
        tt = np.arange(M) * T / M
        g = f.astype(complex)
        if odd:
            g = g * np.exp(-1j * np.pi * tt / T)[:, None]
        G = np.fft.fft(g, axis=0) / M
        k = np.fft.fftfreq(M, 1.0 / M)
        if not odd and M % 2 == 0:
            ny = M // 2
            G = np.vstack([G, 0.5 * G[ny:ny + 1]])
            G[ny] *= 0.5
            k = np.r_[k, -k[ny]]
        self.k = 2 * np.pi * (k + (0.5 if odd else 0.0)) / T
        # Chebyshev coefficients in y = 1 - 2x from the Lobatto values (y_j = cos(pi j / Ng))
        j = np.arange(Ng + 1)
        C = np.cos(np.pi * np.outer(j, j) / Ng)
        w = np.ones(Ng + 1)
        w[0] = w[-1] = 0.5
        A = (2.0 / Ng) * (C * w[None, :])
        A[0] *= 0.5
        A[-1] *= 0.5
        self.C = G @ A.T                               # (Mk, Ng+1): coefficient of e^{i k tau} T_l(y)

    def __call__(self, tau, x, der=(0, 0)):
        y = 1 - 2 * np.asarray(x, dtype=float)
        tau = np.asarray(tau, dtype=float)
        Ng = self.Ng
        T = np.zeros((Ng + 1,) + y.shape)
        U = np.zeros((Ng + 1,) + y.shape)
        T[0] = 1.0
        T[1] = y
        U[0] = 1.0
        U[1] = 2 * y
        for l in range(2, Ng + 1):
            T[l] = 2 * y * T[l - 1] - T[l - 2]
            U[l] = 2 * y * U[l - 1] - U[l - 2]
        if der[1] == 1:
            dT = np.zeros_like(T)
            dT[1:] = np.arange(1, Ng + 1).reshape((-1,) + (1,) * y.ndim) * U[:-1]
            B = -2 * dT
        else:
            B = T
        E = np.exp(1j * np.multiply.outer(self.k, tau))  # (Mk, P)
        if der[0] == 1:
            E = E * (1j * self.k).reshape((-1,) + (1,) * tau.ndim)
        return np.einsum("kl,kp,lp->p", self.C, E.reshape(len(self.k), -1), B.reshape(Ng + 1, -1)).real.reshape(tau.shape)


# ---------------------------------------------------------------- the map
class GGMap:
    def __init__(self, prof, lam_mut=0.0, verbose=True):
        d = np.load(prof, allow_pickle=True)
        self.D, self.Delta, self.M = float(d["D"]), float(d["Delta"]), int(d["M"])
        self.x = d["x"]
        p, m, a, c = d["p"], d["m"], d["a"], d["c"]
        n = self.D - 2
        M, Ng = p.shape[0], p.shape[1] - 1
        self.Ng = Ng
        T = self.Delta / 2
        self.T = T
        x, Dx = cheb01(Ng)
        tt = np.arange(M) * T / M
        X = x[None, :]
        cx = np.zeros_like(c)
        cx[:, 1:] = (n - 1) * c[:, 1:] * (a[:, 1:] ** 2 - 1) / X[:, 1:]
        lam = 1 - cx[:, -1].mean() + lam_mut           # lam_mut: the control of the map's checks
        mu = lam * self.Delta / (4 * np.pi)
        self.lam, self.mu = lam, mu
        Dt = dtau_mat(M, T, False)
        I_t, I_x = np.eye(M), np.eye(Ng + 1)
        K1 = np.kron(Dt, I_x)
        K2 = np.kron(I_t, Dx)
        Lm = K1 + (X - c).ravel()[:, None] * K2 - lam * np.eye(M * (Ng + 1))
        # w^- = (1 - x) h with h(tau, 1) = exp(int (lam - 1 + c_x(1))) (periodic, mean of the exponent 0); the rows
        # w^-_x(tau, 1) = -h(tau, 1) fix the scale and exclude the spurious regular solution of the Nyquist mode
        g = lam - 1 + cx[:, -1]
        G = np.fft.fft(g)
        kk = 2 * np.pi * np.fft.fftfreq(M, 1.0 / M) / T
        G[0] = 0.0
        G[1:] = G[1:] / (1j * kk[1:])
        if M % 2 == 0:
            G[M // 2] = 0.0
        h1 = np.exp(np.fft.ifft(G).real)
        rows = np.kron(np.eye(M), Dx[-1][None, :])
        A = np.vstack([Lm, rows])
        b = np.r_[np.zeros(M * (Ng + 1)), -h1]
        wm = np.linalg.lstsq(A, b, rcond=None)[0].reshape(M, Ng + 1)
        self.res_wm = np.abs(Lm @ wm.ravel()).max()
        Lp = K1 + (X + c).ravel()[:, None] * K2 - lam * np.eye(M * (Ng + 1))
        rhs = np.zeros(M * (Ng + 1))
        i0 = np.arange(M) * (Ng + 1)
        Lp[i0] = 0.0
        Lp[i0, i0] = 1.0
        rhs[i0] = wm[:, 0]
        wp = np.linalg.solve(Lp, rhs).reshape(M, Ng + 1)
        wpx, wmx = wp @ Dx.T, wm @ Dx.T
        ws = wp + wm
        self.check = dict(wm_at_horizon=float(np.abs(wm[:, -1]).max()), wm_min_inside=float(wm[:, :-1].min()),
                          wp_min=float(wp.min()), wpx_min=float(wpx.min()), wmx_max=float(wmx.max()),
                          centre_symmetry=float(np.abs(wpx[:, 0] + wmx[:, 0]).max()), res_wm=float(self.res_wm))
        xi = (wp - wm) / ws
        tp = -np.log(ws / 2) / mu
        zp = -np.log(mu) - np.log(a) - np.log(ws / 2) + 0.5 * np.log(-wpx * wmx)
        zp_t = dtau(zp, T, False)
        zp_x = zp @ Dx.T
        mk = lambda f, odd=False: Interp2(f, T, odd, Ng)
        self.I = dict(wp=mk(wp), wm=mk(wm), wpx=mk(wpx), wmx=mk(wmx), a=mk(a), c=mk(c), p=mk(p, True), m=mk(m, True),
                      zp_t=mk(zp_t), zp_x=mk(zp_x), zp=mk(zp), xi=mk(xi), tp=mk(tp))
        if verbose:
            print(f"  map: D {self.D} Delta {self.Delta:.15f} lam {lam:.15f} mu {mu:.15f}  {self.check}")

    def invert(self, tstar, xistar, it=40):
        """(tau, x) with t(tau, x) = tstar, xi(tau, x) = xistar."""
        I, w = self.I, 4 * np.pi / self.Delta
        x = np.clip(np.asarray(xistar, float).copy(), 0.0, 1.0)
        tau = tstar / w
        for _ in range(3):
            tau = (tstar - I["tp"](tau, x)) / w
        for k in range(it):
            g1 = w * tau + I["tp"](tau, x) - tstar
            g2 = I["xi"](tau, x) - xistar
            a11 = w + I["tp"](tau, x, (1, 0))
            a12 = I["tp"](tau, x, (0, 1))
            a21 = I["xi"](tau, x, (1, 0))
            a22 = I["xi"](tau, x, (0, 1))
            det = a11 * a22 - a12 * a21
            dtau_ = (g1 * a22 - g2 * a12) / det
            dx_ = (a11 * g2 - a21 * g1) / det
            tau = tau - dtau_
            x = np.clip(x - dx_, 0.0, 1.0)
            if max(np.abs(g1).max(), np.abs(g2).max()) < 1e-14:
                break
        return tau, x, max(np.abs(g1).max(), np.abs(g2).max())

    def omegas_at(self, tau, x, xi):
        I, mu = self.I, self.mu
        wp, wm, wpx, wmx = I["wp"](tau, x), I["wm"](tau, x), I["wpx"](tau, x), I["wmx"](tau, x)
        a, c, p, m = I["a"](tau, x), I["c"](tau, x), I["p"](tau, x), I["m"](tau, x)
        ws = wp + wm
        Ap = -mu * ws / (2 * c * wmx)
        Am = -mu * ws / (2 * c * wpx)
        zt = 1 + I["zp_t"](tau, x)
        zx = I["zp_x"](tau, x)
        o3p = Ap * (zt + (x + c) * zx) - mu
        o3m = Am * (zt + (x - c) * zx) + mu
        o2p = mu / xi - Ap * c / x
        o2m = mu / xi + Am * c / x
        o4p = np.sqrt(2) * Ap * c * p / x
        o4m = -np.sqrt(2) * Am * c * m / x
        H = xi * np.exp(-I["zp"](tau, x)) / x
        o1 = 2 * (H - mu) / xi + o2p + o2m
        return o1, o2p, o2m, o3p, o3m, o4p, o4m

    def to_rt(self, rt, t_off=0.0):
        """Initial RT state on the grid of rt (an RT instance), with t shifted by t_off."""
        pos = rt.pos
        TT, XX = np.meshgrid(rt.t, rt.xi[pos], indexing="ij")
        tau, x, err = self.invert(TT + t_off, XX)
        o1, o2p, o2m, o3p, o3m, o4p, o4m = self.omegas_at(tau, x, XX)
        Nt, N1 = rt.Nt, rt.N + 1
        w = [np.zeros((Nt, N1)) for _ in range(4)]
        neg = rt.rev[pos]
        w[0][:, pos], w[0][:, neg] = o1, -o1
        for f, (om, op) in zip(w[1:], ((o2m, o2p), (o3m, o3p), (o4m, o4p))):
            f[:, pos], f[:, neg] = om, -op
        return rt.pack(w[0], w[1], w[2], w[3], self.mu), err

    def to_rt_phased(self, rt):
        """As to_rt, with the t origin chosen so that Re (omega_4)_{1,0} = 0 (the phase row)."""
        z, _ = self.to_rt(rt, 0.0)
        w4 = rt.unpack(z)[3][0]
        cw = rt.phase_w[0] / np.cos(0.0)
        a = np.sum(w4 * np.cos(rt.t / 2)[:, None] * cw[None, :])
        b = np.sum(w4 * np.sin(rt.t / 2)[:, None] * cw[None, :])
        # w4(t + s): its cos(t/2) projection is a cos(s/2) + b sin(s/2) (to first harmonic); zero it
        s = 2 * np.arctan2(-a, b)
        z, err = self.to_rt(rt, s)
        return z, err, s


if __name__ == "__main__":
    if sys.argv[1] == "profile":
        dump_profile(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 48)
