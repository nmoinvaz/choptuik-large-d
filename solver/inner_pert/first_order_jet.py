"""Jets of the D = infinity inner problem with its O(1/n) corrections: lateral (Stokes-jump) integration of the
background and of any number of first-order (and mixed second-order) linear responses at once, M-harmonic
Galerkin truncation, double precision. Used by the first_order_*.py scripts (notes/FIRST_ORDER.md).

Equations (notes/FIRST_ORDER.md, section 1; derived and checked in first_order_equations.py). With
L = xi^2 d_xi + xi (omega d_theta + lambda), E = y^2/2, Psi = sigma^2/2 - xi y y_xi and eps = 1/n:
    L y           = c sigma - eps c (sigma - xi sigma_xi - E sigma)
    L sigma       = c xi y_xi + (c E + eps c Psi) y - c y          ( = (xi d_xi - 1)(c y) given the c equation)
    xi c_xi       = c E + eps c Psi
eps = 0 is the inner problem of the README; lambda is the growth rate of a perturbation (0 for the background and
for parameter derivatives). The model is the one of solver/inner_stokes/inner_np.py: y, sigma on odd harmonics
|mu| <= M, c, E, cE (and Psi, cPsi) on even |mu| <= M + 1, every product truncated, same product order.

Jets. A field is an array (nc, 2L+1): component 0 is the value, the others are indexed by bit masks over "base
directions" (first-order, nilpotent: eps_b^2 = 0); a product of components a, b with a & b = 0 lands on a | b if that
mask is kept. A base direction carries any of:
    lam     growth rate added to the components containing it (a perturbation e^{lam tau});
    omega   d omega / d eps_b (omega is a jet scalar);
    q1      eps_b multiplies the O(1/n) terms (eps above);
    dlam    the derivative d/d lambda of the perturbation components (those containing a 'lam' direction);
    force   a callable giving extra forcing (y, c equations) from the value component (symmetry-breaking terms).
Initial data per kept component: d0 (coefficient of xi in y, i.e. 2 x profile) and g0 (c at xi = 0).

    from first_order_jet import JetModel
"""
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp

W0 = float(2 * mp.pi / mp.mpf("1.64389375"))
DELTA = np.pi / 8

# Mutations for the checks (first_order_checks.py --mutate=...): each must make some check fail.
#   q1y   flips the sign of the O(1/n) term of the y equation
#   q1c   drops the O(1/n) term (c Psi) of the c equation (and so of the sigma equation)
#   force flips the sign of the s-translation breaking forcing (SymForce)
#   rhs   flips the O(1/n) term of the y equation in the right-hand side only (not in the series)
MUTATE = set()
SGN_Y = lambda: -1.0 if "q1y" in MUTATE else 1.0
SGN_YR = lambda: -1.0 if ("q1y" in MUTATE or "rhs" in MUTATE) else 1.0
SGN_C = lambda: 0.0 if "q1c" in MUTATE else 1.0
SGN_F = lambda: -1.0 if "force" in MUTATE else 1.0


def Kp(s):
    a, b = (1 + s) / 2, (2 + s) / 2
    return complex(mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b)))


class Dir:
    def __init__(self, name, lam=0.0, omega=0.0, q1=False, dlam=False, force=None):
        self.name, self.lam, self.omega, self.q1, self.dlam, self.force = name, lam, omega, q1, dlam, force


class JetModel:
    def __init__(self, M, w, Fh, dirs, masks, data):
        """M: harmonics; w: omega (value); Fh: {mu: F_mu} background profile; dirs: list of Dir (bit i <-> dirs[i]);
        masks: kept component masks (0 first, closed under subsets); data: {mask: (Fh_k, g0h_k)} initial data of each
        non-zero component: profile harmonics (d0 = 2 Fh_k) and c(0) harmonics (dict, even mu)."""
        self.M, self.w = M, w
        self.L = L = M + 2
        k = np.arange(-L, L + 1)
        self.k = k
        self.odd = (np.abs(k) <= M) & (k % 2 == 1)
        self.even = (np.abs(k) <= M + 1) & (k % 2 == 0)
        self.dirs = dirs
        self.masks = list(masks)
        assert self.masks[0] == 0
        self.idx = {m: i for i, m in enumerate(self.masks)}
        self.nc = len(self.masks)
        self.pairs = [(i, j, self.idx[a | b]) for i, a in enumerate(self.masks) for j, b in enumerate(self.masks)
                      if a & b == 0 and (a | b) in self.idx]
        self.PI = np.array([p[0] for p in self.pairs]); self.PJ = np.array([p[1] for p in self.pairs])
        self.PK = np.array([p[2] for p in self.pairs])
        N = 2 * L + 1
        mm, nn = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        self.TIDX = np.clip(mm - nn + L, 0, N - 1)
        self.TVAL = (np.abs(mm - nn) <= L).astype(float)
        self.maxpop = max(bin(m).count("1") for m in self.masks)
        # per-component growth rate
        self.lamc = np.array([sum(d.lam for b, d in enumerate(dirs) if m >> b & 1) for m in self.masks])
        self.pertbits = sum(1 << b for b, d in enumerate(dirs) if d.lam != 0 or d.name.startswith("pert"))
        # omega jet: component index -> value
        self.wj = {0: w}
        for b, d in enumerate(dirs):
            if d.omega and (1 << b) in self.idx:
                self.wj[self.idx[1 << b]] = d.omega
        self.qbit = next((1 << b for b, d in enumerate(dirs) if d.q1), 0)
        self.lbit = next((1 << b for b, d in enumerate(dirs) if d.dlam), 0)
        self.fdirs = [(1 << b, d.force) for b, d in enumerate(dirs) if d.force is not None]
        # shift maps: component i -> component of mask | bit
        self.qmap = self._shiftmap(self.qbit)
        self.lmap = [(i, j) for i, j in self._shiftmap(self.lbit) if self.masks[i] & self.pertbits] if self.lbit else []
        self.imu = 1j * k
        # initial data
        self.F = self._harm(Fh) * self.odd
        self.d0 = np.zeros((self.nc, 2 * L + 1), complex)
        self.g0 = np.zeros((self.nc, 2 * L + 1), complex)
        self.d0[0] = 2 * self.F
        self.g0[0][L] = 1.0
        for m, (fk, gk) in data.items():
            if m in self.idx:
                self.d0[self.idx[m]] = 2 * self._harm(fk) * self.odd
                self.g0[self.idx[m]] = self._harm(gk) * self.even

    def _shiftmap(self, bit):
        if not bit:
            return []
        return [(i, self.idx[m | bit]) for i, m in enumerate(self.masks) if not m & bit and (m | bit) in self.idx]

    def _harm(self, h):
        out = np.zeros(2 * self.L + 1, complex)
        for mu, v in (h or {}).items():
            if abs(mu) <= self.L:
                out[mu + self.L] = v
        return out

    # ---- jet arithmetic on arrays (nc, 2L+1)
    def mul(self, a, b, space):
        """truncated product: for every pair of components, the convolution of the harmonic lists (Toeplitz
        matrix-vector products, the same sums as np.convolve), kept on |mu| <= L and then projected on `space`"""
        Ta = a[:, self.TIDX] * self.TVAL
        prod = np.einsum("pmn,pn->pm", Ta[self.PI], b[self.PJ])
        out = np.zeros_like(a)
        np.add.at(out, self.PK, prod)
        return out * space

    def shift(self, a, smap):
        out = np.zeros_like(a)
        for i, j in smap:
            out[j] += a[i]
        return out

    def wmul(self, a):
        """omega_hat * i mu * a (jet scalar omega times theta-derivative)"""
        if not hasattr(self, "_wp"):
            wp = [(j, self.idx[self.masks[ci] | mj], wv) for ci, wv in self.wj.items() for j, mj in enumerate(self.masks)
                  if not self.masks[ci] & mj and (self.masks[ci] | mj) in self.idx]
            self._wp = (np.array([t[0] for t in wp]), np.array([t[1] for t in wp]), np.array([t[2] for t in wp]))
        J, K, Wv = self._wp
        out = np.zeros_like(a)
        np.add.at(out, K, Wv[:, None] * (self.imu * a[J]))
        return out

    def lmul(self, a):
        """Lambda_hat * a: per-component growth rate, plus the d/d lambda direction"""
        out = self.lamc[:, None] * a
        for i, j in self.lmap:
            out[j] += a[i]
        return out

    def ginv_solve(self, g0, rhs, space):
        """solve [g0 * x] = rhs for x, g0 = 1 + (nilpotent part)"""
        gn = g0.copy()
        gn[0] = 0
        x = rhs.copy()
        for _ in range(self.maxpop):
            x = rhs - self.mul(gn, x, space)
        return x

    # ---- formal series at xi0 (scaled: D_j = d_j xi0^(2j), ...), optimally truncated on the value component
    def series_at(self, xi0, J, stop=True):
        q = xi0 * xi0
        o, e = self.odd, self.even
        for b, fn in self.fdirs:
            fn.prepare(self, xi0, J + 3)
        D, S, G, E, CE, PS, CP, OM = [self.d0.copy()], [], [self.g0.copy()], [None], [None], [None], [None], []
        Z = np.zeros_like(self.d0)
        mags = [np.abs(D[0][0]).max()]
        for j in range(J):
            # (a) s_j
            R = (2 * j + 1) * D[j] + self.lmul(D[j]) + self.wmul(D[j])
            for i in range(1, j + 1):
                R = R - self.mul(G[i], S[j - i], o)
            for b, fn in self.fdirs:
                R = R + fn.series_y(j, self)
            sj = self.ginv_solve(G[0], R * o, o)
            if self.qbit:
                Om_j = (2 * j + 1) * sj
                for m in range(1, j + 1):
                    Om_j = Om_j + self.mul(E[m], S[j - m] if m > 0 else sj, o)
                OMj = OM + [Om_j]
                Qy = Z.copy()
                for i in range(0, j + 1):
                    Qy = Qy + self.mul(G[i], OMj[j - i], o)
                sj = self.ginv_solve(G[0], (R - SGN_Y() * self.shift(Qy, self.qmap)) * o, o)
                Om_j = (2 * j + 1) * sj
                for m in range(1, j + 1):
                    Om_j = Om_j + self.mul(E[m], S[j - m], o)
                OM.append(Om_j)
            S.append(sj * o)
            # (b) e_{j+1}, g_{j+1}
            m = j + 1
            em = Z.copy()
            for i in range(m):
                em = em + self.mul(D[i], D[m - 1 - i], e)
            em = q * em / 2
            E.append(em)
            cm = Z.copy()
            for i in range(m):
                cm = cm + self.mul(G[i], E[m - i], e)
            CE.append(cm)
            gm = cm
            if self.qbit:
                pm = -2 * m * em
                ss = Z.copy()
                for i in range(0, m - 1):
                    ss = ss + self.mul(S[i], S[m - 2 - i], e)
                pm = pm + q * q * ss / 2
                PS.append(pm)
                cp = Z.copy()
                for i in range(m):
                    cp = cp + self.mul(G[i], PS[m - i], e)
                CP.append(cp)
                gm = gm + SGN_C() * self.shift(cp, self.qmap)
            for b, fn in self.fdirs:
                gm = gm + fn.series_c(m, self)
            G.append(gm / (2 * m))
            # (c) d_{j+1}
            num = q * ((2 * j + 2) * S[j] + self.lmul(S[j]) + self.wmul(S[j]))
            for i in range(1, j + 2):
                num = num - (2 * j + 2) * self.mul(G[i], D[j + 1 - i], o)
            dn = self.ginv_solve(G[0], num * o / (2 * j + 2), o)
            D.append(dn * o)
            mags.append(np.abs(D[-1][0]).max())
            if stop and j > 10 and mags[-1] > 10 * min(mags):
                break
        jt = int(np.argmin(mags[1:])) + 1
        Dd = xi0 * sum(D[:jt]); Sg = q * sum(S[:jt]); Cc = sum(G[:jt])
        self.ser = dict(D=D, S=S, G=G, E=E, jt=jt, mags=mags)
        return self.pack(Dd, Sg, Cc), jt, mags[jt] / mags[0]

    # ---- packing
    def pack(self, D, S, C):
        return np.concatenate([D[:, self.odd].ravel(), S[:, self.odd].ravel(), C[:, self.even].ravel()])

    def unpack(self, Y):
        no, ne, nc = self.odd.sum(), self.even.sum(), self.nc
        D = np.zeros((nc, 2 * self.L + 1), complex); S = np.zeros_like(D); C = np.zeros_like(D)
        D[:, self.odd] = Y[:nc * no].reshape(nc, no)
        S[:, self.odd] = Y[nc * no:2 * nc * no].reshape(nc, no)
        C[:, self.even] = Y[2 * nc * no:].reshape(nc, ne)
        return D, S, C

    def rhs(self, xi, Y):
        o, e = self.odd, self.even
        D, S, C = self.unpack(Y)
        trans = lambda X: xi * (self.wmul(X) + self.lmul(X))
        dD = (self.mul(C, S, o) - trans(D)) / (xi * xi)
        E = self.mul(D, D, e) / 2
        cE = self.mul(C, E, e)
        ext_y, ext_c = 0, 0
        if self.qbit or self.fdirs:
            r2 = self.mul(C, xi * dD, o) + self.mul(cE, D, o) - self.mul(C, D, o)
            dS0 = (r2 - trans(S)) / (xi * xi)
        for b, fn in self.fdirs:
            fy, fc = fn.rhs(xi, self, D, S, C, E, dD, dS0)
            ext_y, ext_c = ext_y + fy, ext_c + fc
        if self.qbit:
            Om = S - xi * dS0 - self.mul(E, S, o)
            dD = dD - SGN_YR() * self.shift(self.mul(C, Om, o), self.qmap) / (xi * xi)
            Psi = self.mul(S, S, e) / 2 - self.mul(D, xi * dD, e)
            cE = cE + SGN_C() * self.shift(self.mul(C, Psi, e), self.qmap)
        dD = dD + ext_y / (xi * xi)
        cE = cE + ext_c
        r2 = self.mul(C, xi * dD, o) + self.mul(cE, D, o) - self.mul(C, D, o)
        dS = (r2 - trans(S)) / (xi * xi)
        dC = cE / xi
        return self.pack(dD, dS * o, dC)

    def integrate(self, Y, kind, a, b, par, rtol=1e-12, atol=1e-16):
        if kind == "ray":
            u = np.exp(1j * par)
            f = lambda t, Z: self.rhs(t * u, Z) * u
        else:
            r = par
            f = lambda t, Z: self.rhs(r * np.exp(1j * t), Z) * 1j * r * np.exp(1j * t)
        sol = solve_ivp(f, (a, b), Y, method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]

    def r0_for(self, margin=12.0):
        sd = np.sin(DELTA)
        return float(np.sqrt(sd / (2 * (self.M * self.w * (np.pi / 2 + DELTA) + margin))))

    def jumps(self, r1s=(0.2,), r0=None, rtol=1e-12):
        """jump J = y+ - y- of every component at each r1 on the real axis (y- = conjugate of y+ for real data);
        returns {r1: array (nc, 2L+1)}, info"""
        phi = (np.pi / 2 + DELTA) / 2
        r0 = r0 or self.r0_for()
        Y, jt, terr = self.series_at(r0 * np.exp(1j * phi), J=int(1.2 / (2 * r0 * r0)) + 40)
        out, ends = {}, {}
        rprev = r0
        for r1 in sorted(r1s):
            Y = self.integrate(Y, "ray", rprev, r1, phi, rtol)
            rprev = r1
            Ye = self.integrate(Y, "arc", phi, 0.0, r1, rtol)
            D, S, C = self.unpack(Ye)
            out[r1] = D - np.conj(D[:, ::-1])
            ends[r1] = (D, S, C)
        self.ends = ends
        return out, (r0, jt, terr)


class SymForce:
    """Forcing of the O(1/n) part W of the lambda = 2 mode (the s-translation symmetry broken by the O(1/n) terms;
    first_order_equations.py, section 3). With the mode written dZ = D Z_n + W/n, W solves the perturbation equations
    at lambda = 2 with
        L_2 W_y   = ... - (2/xi^2) c (xi sigma_xi - 2 sigma + E sigma)            (= F_y)
        xi W_c,xi = ... - (2/xi^2) c (Psi + 2 E),  Psi = sigma^2/2 - xi y y_xi     (= F_c)
    and the sigma equation forced by y F_c (included through cE), all from the background (value component).
    The series needs the background one order ahead, so the background series is computed first."""

    def __init__(self, bit):
        self.bit = bit

    def prepare(self, model, xi0, J):
        bg = JetModel(model.M, model.w, None, [], [0], {})
        bg.d0[0] = model.d0[0]
        bg.series_at(xi0, J, stop=False)
        self.q = xi0 * xi0
        self.bg = {k: [v[0] if v is not None else None for v in bg.ser[k]] for k in ("D", "S", "G", "E")}
        self.slot = model.idx[self.bit]
        self.L = model.L
        o, e = model.odd, model.even
        S, E, G = self.bg["S"], self.bg["E"], self.bg["G"]
        q = self.q
        cv = lambda a, b, sp: np.convolve(a, b)[self.L:3 * self.L + 1] * sp
        nS = len(S)
        self.Th = [np.zeros_like(S[0])]
        for k in range(1, nS):
            t = 2 * k * S[k]
            for m in range(1, k + 1):
                t = t + cv(E[m], S[k - m], o)
            self.Th.append(t)
        self.PE = [np.zeros_like(S[0]), np.zeros_like(S[0])]
        for k in range(2, min(len(E), nS + 2)):
            t = -(2 * k - 2) * E[k]
            for i in range(0, k - 1):
                t = t + q * q * cv(S[i], S[k - 2 - i], e) / 2
            self.PE.append(t)
        self.cv, self.o, self.e = cv, o, e

    def _put(self, model, arr):
        out = np.zeros((model.nc, 2 * model.L + 1), complex)
        out[self.slot] = arr
        return out

    def series_y(self, j, model):
        """+(2/q) sum_i G_i Theta_{j+1-i}  (the contribution -F_y to [g0 s_j])"""
        G = self.bg["G"]
        t = np.zeros(2 * self.L + 1, complex)
        for i in range(0, j + 1):
            t = t + self.cv(G[i], self.Th[j + 1 - i], self.o)
        return self._put(model, SGN_F() * 2 * t / self.q)

    def series_c(self, m, model):
        """F_c at order m: -(2/q) sum_i G_i (Psi + 2E)_{m+1-i}"""
        G = self.bg["G"]
        t = np.zeros(2 * self.L + 1, complex)
        for i in range(0, m + 1):
            t = t + self.cv(G[i], self.PE[m + 1 - i], self.e)
        return self._put(model, -SGN_F() * 2 * t / self.q)

    def rhs(self, xi, model, D, S, C, E, dD, dS):
        o, e = self.o, self.e
        cv = self.cv
        D0, S0, C0, E0 = D[0], S[0], C[0], E[0]
        Th = xi * dS[0] - 2 * S0 + cv(E0, S0, o)
        Fy = -SGN_F() * 2 * cv(C0, Th, o) / (xi * xi)
        Psi = cv(S0, S0, e) / 2 - cv(D0, xi * dD[0], e)
        Fc = -SGN_F() * 2 * cv(C0, Psi + 2 * E0, e) / (xi * xi)
        return self._put(model, Fy), self._put(model, Fc)


# ---------------- background parametrization (as inner_np: F = A sin t + sum_nu (B_nu sin nu t + C_nu cos nu t))
def Fh_from(x, M):
    A = x[0]
    h = {1: A / 2j, -1: -A / 2j}
    for i, nu in enumerate(range(3, M + 1, 2)):
        B, C = x[2 + 2 * i], x[3 + 2 * i]
        h[nu] = B / 2j + C / 2
        h[-nu] = np.conj(h[nu])
    return h


def dFh(p, M):
    """dF/dx_p as harmonics (p = 0: A; p = 1: omega -> no profile; p >= 2: B_nu, C_nu)"""
    if p == 0:
        return {1: 1 / 2j, -1: -1 / 2j}
    if p == 1:
        return {}
    nu = 3 + 2 * ((p - 2) // 2)
    if (p - 2) % 2 == 0:
        return {nu: 1 / 2j, -nu: -1 / 2j}
    return {nu: 0.5, -nu: 0.5}


def F2mean(Fh):
    return sum(abs(v) ** 2 for v in Fh.values())


def F2h(Fh, M):
    """harmonics of F^2 truncated to even |mu| <= M + 1 (the c(xi^2) coefficient g_1 of the model)"""
    out = {}
    for a, fa in Fh.items():
        for b, fb in Fh.items():
            if abs(a + b) <= M + 1:
                out[a + b] = out.get(a + b, 0) + fa * fb
    return out


def demix_matrix(x, M, r1, NTH=64):
    """linear map from the raw jump (harmonics -M..M, odd) to the de-mixed normalized residual r_nu (nu = 1..M odd),
    as inner_np.residual_demixed; returns a function J (2L+1 array, L = M + 2) -> complex array over nu"""
    w = W0 * x[1]
    Fh = Fh_from(x, M)
    L = M + 2
    th = 2 * np.pi * np.arange(NTH) / NTH
    kk = np.fft.fftfreq(NTH, 1 / NTH)
    Fth = sum(v * np.exp(1j * mu * th) for mu, v in Fh.items()).real
    F2 = np.fft.fft(Fth ** 2) / NTH
    Pi = np.real(np.fft.ifft(np.where(kk != 0, F2 / (1j * kk + (kk == 0)), 0)) * NTH) / w
    lsc = 1 / (2 * r1 * r1) - 2 * F2mean(Fh) * np.log(r1)
    nus = list(range(1, M + 1, 2))
    fac = np.array([r1 ** (2j * nu * w) * np.exp(lsc - np.log(2 * np.pi * abs(Kp(1j * nu * w)))) for nu in nus])
    E = np.exp(-Pi)

    def f(Jd):
        Jth = sum(Jd[k + L] * np.exp(1j * k * th) for k in range(-M, M + 1) if k % 2)
        qh = np.fft.fft(Jth * E) / NTH
        return np.array([qh[nu % NTH] for nu in nus]) * fac
    return f


def ri(z):
    """complex array -> stacked (Re, Im) real vector, as inner_np's residuals"""
    z = np.asarray(z)
    return np.ravel(np.column_stack([z.real, z.imag]))
