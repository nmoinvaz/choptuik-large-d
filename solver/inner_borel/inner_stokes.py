"""Fast-type Stokes data of the inner problem, from the late orders of the formal series (inner_series.py).

Borel variable: u = xi^2, B_mu(zeta) = sum_j d_{j,mu} zeta^j / j!. The fast mode e^{-1/(2u)} puts the Borel singularity at
zeta = 1/2. Fast modes (README Prop. 3 analogue in the inner problem): Dd = e^{-1/(2u)} a, with the transport
    xi a_xi = (-2 i w diag(nu) + 2 T(F^2)) a        (T: Toeplitz/convolution matrix of F^2 on the kept harmonics)
so a mode with eigenvalue kappa_k, eigenvector v_k gives a Borel singularity (1 - 2 zeta)^(-gamma_k), gamma_k = 1/2 - kappa_k/2,
and late orders  d_{j,mu} ~ sum_k S_k v_{k,mu} 2^j Gamma(j + gamma_k) (1 + c_{k,1}/(j+gamma_k-1) + ...).
Without gravity kappa = -2 i nu w, gamma = s + 1/2 and S = d_0 / (Gamma(alpha) Gamma(beta)) exactly.

    python3 inner_stokes.py linear            # Darboux + Dlog-Pade test against the exact 2F0
    python3 inner_stokes.py grav A Mmax J [omega_factor] [B3]
"""
import sys
import mpmath as mp
from flint import acb
from inner_series import series

mp.mp.dps = 120
W0 = 2 * mp.pi / mp.mpf("1.64389375")


def to_mp(a):
    m = a.mid()
    return mp.mpc(mp.mpf(m.real.str(110, radius=False)), mp.mpf(m.imag.str(110, radius=False)))


def transport_modes(Fh, w, M):
    """eigen-decomposition of -2 i w diag(nu) + 2 T(F^2) on odd nu in [-M, M]; Fh: {mu: F_mu}"""
    nus = list(range(-M, M + 1, 2))
    F2 = {}
    for a, fa in Fh.items():
        for b, fb in Fh.items():
            F2[a + b] = F2.get(a + b, 0) + mp.mpc(fa) * mp.mpc(fb)
    T = mp.matrix(len(nus), len(nus))
    for i, nu in enumerate(nus):
        for k, nk in enumerate(nus):
            T[i, k] = 2 * F2.get(nu - nk, 0)
        T[i, i] += -2j * w * nu
    E, V = mp.eig(T)
    return nus, E, V


def fit(dmu, gam, j0, j1, nc):
    """least squares d_j / 2^j = sum_k sum_m S_{k,m} Gamma(j + gam_k - m), j in [j0, j1]; returns S_{k,0}"""
    cols = [(k, m) for k in range(len(gam)) for m in range(nc + 1)]
    ref = lambda j: mp.gamma(j + mp.re(gam[0]))
    rows, rhs = [], []
    for j in range(j0, j1 + 1):
        sc = 1 / ref(j)
        rows.append([mp.gamma(j + gam[k] - m) * sc for k, m in cols])
        rhs.append(dmu[j] / mp.mpf(2) ** j * sc)
    A = mp.matrix(rows); b = mp.matrix(rhs)
    x = mp.qr_solve(A, b)[0]
    return [x[cols.index((k, 0))] for k in range(len(gam))], x


def linear():
    w = W0
    J = 200
    d, _, _ = series({1: -0.5j, -1: 0.5j}, w, J, M=2, prec=700, gravity=False)
    dm = [to_mp(d[j][1 + 2]) for j in range(J + 1)]
    s = 1j * w
    al, be = (1 + s) / 2, (2 + s) / 2
    gam = s + mp.mpf(1) / 2
    exact = dm[0] / (mp.gamma(al) * mp.gamma(be))
    for nc in (6, 12, 20):
        S, _ = fit(dm, [gam], J - 80, J, nc)
        print(f"linear, Darboux fit with {nc} corrections: S = {mp.nstr(S[0], 15)}  exact d0/(G(a)G(b)) = {mp.nstr(exact, 15)}"
              f"  rel err {mp.nstr(abs(S[0] / exact - 1), 3)}")
    # Dlog-Pade: B'/B has a simple pole at 1/2 with residue -gamma (B ~ (1 - 2 zeta)^(-gamma))
    for N in (10, 20, 30):
        Bc = [dm[j] / mp.factorial(j) for j in range(2 * N + 3)]
        inv = [1 / Bc[0]]
        for k in range(1, 2 * N + 2):
            inv.append(-sum(Bc[i] * inv[k - i] for i in range(1, k + 1)) / Bc[0])
        Bd = [(k + 1) * Bc[k + 1] for k in range(2 * N + 2)]
        L = [sum(Bd[i] * inv[k - i] for i in range(k + 1)) for k in range(2 * N + 1)]
        try:
            p, q = mp.pade(L, N, N)
        except ZeroDivisionError:
            print(f"Dlog-Pade [{N}/{N}]: singular"); continue
        roots = mp.polyroots(q[::-1], maxsteps=400, extraprec=600)
        r0 = min(roots, key=lambda r: abs(r - 0.5))
        pv = mp.polyval(p[::-1], r0)
        dq = mp.polyval([c * (len(q) - 1 - i) for i, c in enumerate(q[::-1])][:-1], r0)
        print(f"Dlog-Pade [{N}/{N}]: pole at {mp.nstr(r0, 15)}, residue {mp.nstr(pv / dq, 12)}"
              f"  (expected 1/2 and -gamma/2 = {mp.nstr(-gam / 2, 12)})")


def grav(A, Mmax, J, wf=1, B3=0):
    w = W0 * wf
    Fh = {1: -0.5j * A, -1: 0.5j * A}
    if B3:
        Fh.update({3: -0.5j * B3, -3: 0.5j * B3})
    d, _, _ = series(Fh, w, J, M=Mmax + 1, prec=900)
    Mst = Mmax + 1
    nus, E, V = transport_modes(Fh, w, Mmax)
    gam = [mp.mpf(1) / 2 - E[k] / 2 for k in range(len(nus))]
    order = sorted(range(len(nus)), key=lambda k: mp.im(gam[k]))
    out = {}
    for mu in (1, 3):
        if mu > Mmax:
            continue
        dm = [to_mp(d[j][mu + Mst]) for j in range(J + 1)]
        res = {}
        for nc in (4, 8):
            S, _ = fit(dm, [gam[k] for k in order], J - 70, J, nc)
            res[nc] = S
        out[mu] = res
        print(f"A {A} Mmax {Mmax} J {J} w {mp.nstr(w, 6)}: harmonic {mu}")
        for i, k in enumerate(order):
            vk = V[nus.index(mu), k]
            print(f"   mode gamma {mp.nstr(gam[k], 8):>26s}  S_k v_k,mu (nc 4) {mp.nstr(res[4][i], 8):>28s}  (nc 8) {mp.nstr(res[8][i], 8):>28s}"
                  f"   S_k = {mp.nstr(res[8][i] / vk, 8)}")
    return nus, gam, order, out


if __name__ == "__main__":
    if sys.argv[1] == "linear":
        linear()
    else:
        A = mp.mpf(sys.argv[2]); Mmax = int(sys.argv[3]); J = int(sys.argv[4])
        wf = mp.mpf(sys.argv[5]) if len(sys.argv) > 5 else 1
        B3 = mp.mpf(sys.argv[6]) if len(sys.argv) > 6 else 0
        grav(A, Mmax, J, wf, B3)
