#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 9): numerical checks of the M = 1 bootstrap on actual solutions, at moderate
n (double precision, not rigorous), on the complex paths of the proof.

  (A) The remainder R = Z - Z^K of the centre-regular solution (gauge c(0) = c^K(0)) from the remainder equation
      R' = J(x) R + tau(x), J = dF/dZ at Z^K (central differences, numpy), tau = F(x, Z^K) - Z^K' (mpmath, 40 digits:
      the cancellation is exact there), F the reduced right-hand side (b8m_model.reduced_rhs, re-coded in numpy and
      checked against it).  Paths: the ray arg x = alpha from x_0 = 1e-3 e^(i alpha) (R(x_0) = 0 up to O(x_0^2 eps^(K+1)))
      to R'_n, then arcs |x| = r from alpha down to 0.
      Cross-checks: (A1) against the next formal orders eps^(K+1) Z_(K+1) + eps^(K+2) Z_(K+2) (relative difference
      O(rho)); (A2) against Z - Z^K from an independent integration of the ORIGINAL model in (p, m, b, c) (model_rhs
      re-coded in numpy), where Z - Z^K is resolvable in double precision.
  (B) The bootstrap bounds: max over the sample points of |R_V|/(h_V rho^(K+1)), |R_S|/(h_S theta rho^(K+1)),
      |R_beta|/(h_b theta rho^(K+1)), |R_c|/(h_c rho^(K+1)) (l1 norms) with h from a closing b8m_scan.py log line.
      These must be < 1.  (The lemma is proved for n >= n_0; at moderate n the check tests the shape of the bound.)
  (C) Lemma K on an arc: the propagator E of the homogeneous fast equation x R_S' = Lam R_S (Lam = (D^K)^-1 A^K) along
      |x| = r from alpha down to arg x, against the bound exp(int (Re lam_scalar + |lam| ||H'||) du).

Mutations: --mutate=tau  (tau of the S-components scaled by 1.01: (A1) and (A2) must fail);
           --mutate=weight (the bound with rho^(K+2) instead of rho^(K+1): (B) must fail at the larger n);
           --mutate=kernel (||H'|| dropped from the Lemma K exponent: (C) must fail).

    python3 b8m_numcheck.py n [K] [--log=b8m_scan_K4_box.log] [--L0=73.89] [--mutate=...]
"""
import json
import sys
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from flint import ctx
import b8m_formal as FM
import b8m_model as BM

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
mp.mp.dps = 40
O, E = [-1, 1], [-2, 0, 2]
KAPPA, KAPPA1, DELTA = 0.3, 0.6, 0.05
ALPHA = np.pi / 4 + KAPPA / 2


# ------------------------------------------------------------------ numpy forms of the M = 1 model
def Cmat(c):
    return np.array([[c[1], c[0]], [c[2], c[1]]])          # c = (c_-2, c_0, c_2) acting on (f_-1, f_1)


def oo(f, g):
    return np.array([f[0] * g[0], f[0] * g[1] + f[1] * g[0], f[1] * g[1]])


def ee(a, b):
    return np.array([a[0] * b[1] + a[1] * b[0], a[1] * b[1] + a[0] * b[2] + a[2] * b[0], a[2] * b[1] + a[1] * b[2]])


def F_np(x, Z, n, w):
    """reduced right-hand side; Z = (V_-1, V_1, S_-1, S_1, beta_-2, beta_0, beta_2, c_-2, c_0, c_2)"""
    V, S, be, c = Z[0:2], Z[2:4], Z[4:7], Z[7:10]
    eps = 1 / n
    Cm = Cmat(c)
    Bb = Cmat(ee(c, be))
    Om = np.diag([-1j * w, 1j * w])
    I = np.eye(2)
    Ci = np.linalg.inv(Cm)
    B = I - Bb @ Ci + Cm @ Om @ Ci
    A = (n - 1) * Cm @ Cm + Cm @ Bb - x**2 * Om - x**2 * B
    dS = np.linalg.solve(x * (x**2 * I - Cm @ Cm), A @ S - x**2 * B @ V)
    dV = Om @ S / x
    D = x * Ci @ (V + S)
    H = (oo(S, S) + oo(D, D)) / 2
    dbe = (-(n - 1) * be - ee(be, be) + (1 - eps) * (H + ee(be, H) / (n - 1))) / x
    dc = ee(c, be) / x
    return np.concatenate([dV, dS, dbe, dc])


def model_np(x, U, n, w):
    """the ORIGINAL model (b8m_model.model_rhs) in U = (p_-1, p_1, m_-1, m_1, b_-2, b_0, b_2, c_-2, c_0, c_2)"""
    p, m, b, c = U[0:2], U[2:4], U[4:7], U[7:10]
    bm1 = b - np.array([0, 1, 0])
    G = (n - 1) * ee(c, bm1)
    p2m2 = oo(p, p) + oo(m, m)
    dc = G / x
    db = (ee(b, p2m2) - (n - 1) * ee(b, bm1)) / x
    Cm = Cmat(c)
    Gm = Cmat(G)
    mu = np.array([-1, 1])
    cpm = Cm @ (p + m)
    rp = -w * x * 1j * mu * p + Gm @ p - Cm @ p + n / 2 * cpm
    rm = -w * x * 1j * mu * m - Gm @ m + Cm @ m - n / 2 * cpm
    I = np.eye(2)
    dp = np.linalg.solve(x * x * I - x * Cm, rp)
    dm = np.linalg.solve(x * x * I + x * Cm, rm)
    return np.concatenate([dp, dm, db, dc])


def to_model(Z, n):
    V, S, be, c = Z[0:2], Z[2:4], Z[4:7], Z[7:10]
    x = Z[10]
    D = x * np.linalg.solve(Cmat(c), V + S)
    h = 1 / np.sqrt(n)
    return np.concatenate([h * (S + D) / 2, h * (S - D) / 2, np.array([0, 1, 0]) + be / (n - 1), c])


def from_model(U, x, n):
    p, m, b, c = U[0:2], U[2:4], U[4:7], U[7:10]
    sq = np.sqrt(n)
    S, D = sq * (p + m), sq * (p - m)
    V = (Cmat(c) @ D - x * S) / x
    return np.concatenate([V, S, (n - 1) * (b - np.array([0, 1, 0])), c])


# ------------------------------------------------------------------ the truncated series in mpmath
class Series:
    def __init__(self, K, n, extra=2):
        ctx.prec = 200
        F, om = FM.params()
        self.fs = FM.formal(K + extra, F, om)
        self.K, self.n = K, mp.mpf(n)
        self.w = mp.mpf(om.real.mid().str(50, radius=False))
        tm = lambda z: mp.mpc(mp.mpf(z.real.mid().str(45, radius=False)), mp.mpf(z.imag.mid().str(45, radius=False)))
        self.co = {}
        for nm, keys in (("V", O), ("S", O), ("be", E), ("c", E)):
            for h in keys:
                self.co[(nm, h)] = [[tm(c) for c in self.fs[nm][k][h].coeffs()] for k in range(K + extra + 1)]

    def ev(self, x, kmin, kmax, d=0):
        out = []
        for nm, keys in (("V", O), ("S", O), ("be", E), ("c", E)):
            for h in keys:
                tot = mp.mpc(0)
                for k in range(kmin, kmax + 1):
                    cs = self.co[(nm, h)][k]
                    if d == 0:
                        v = mp.polyval(cs[::-1], x) if cs else 0
                    else:
                        v = mp.polyval([j * cs[j] for j in range(len(cs) - 1, 0, -1)], x) if len(cs) > 1 else 0
                    tot += v / self.n ** k
                out.append(tot)
        return out

    def tau(self, x):
        x = mp.mpc(x)
        Z = self.ev(x, 0, self.K)
        dZ = self.ev(x, 0, self.K, 1)
        V, S = {O[i]: Z[i] for i in range(2)}, {O[i]: Z[2 + i] for i in range(2)}
        be, c = {E[i]: Z[4 + i] for i in range(3)}, {E[i]: Z[7 + i] for i in range(3)}
        dV, dS, dbe, dc = BM.reduced_rhs(1, x, self.n, self.w, V, S, be, c)
        Fv = [dV[m] for m in O] + [dS[m] for m in O] + [dbe[k] for k in E] + [dc[k] for k in E]
        t = np.array([complex(Fv[i] - dZ[i]) for i in range(10)])
        if MUT == "tau":
            t[2:4] *= 1.01
        return t

    def ZK(self, x):
        return np.array([complex(v) for v in self.ev(mp.mpc(x), 0, self.K)])

    def nxt(self, x):
        return np.array([complex(v) for v in self.ev(mp.mpc(x), self.K + 1, self.K + 2)])


def jac(x, Z, n, w):
    J = np.zeros((10, 10), dtype=complex)
    for k in range(10):
        hk = 1e-6 * (1 + abs(Z[k]))
        e = np.zeros(10, dtype=complex)
        e[k] = hk
        J[:, k] = (F_np(x, Z + e, n, w) - F_np(x, Z - e, n, w)) / (2 * hk)
    return J


def cplx_block(Mc):
    return np.block([[Mc.real, -Mc.imag], [Mc.imag, Mc.real]])


class ZKfast:
    """Z^K(x) in double precision from the mpmath coefficients (Horner per component)"""

    def __init__(self, ser):
        self.cs = []
        for nm, keys in (("V", O), ("S", O), ("be", E), ("c", E)):
            for h in keys:
                lst = ser.co[(nm, h)]
                deg = max(len(lst[k]) for k in range(ser.K + 1))
                tab = np.zeros((ser.K + 1, deg), dtype=complex)
                for k in range(ser.K + 1):
                    for m, c in enumerate(lst[k]):
                        tab[k, m] = complex(c)
                self.cs.append(tab)
        self.n = float(ser.n)

    def __call__(self, x):
        out = np.empty(10, dtype=complex)
        for i, tab in enumerate(self.cs):
            coef = (tab * (self.n ** -np.arange(tab.shape[0]))[:, None]).sum(axis=0)
            out[i] = np.polyval(coef[::-1], x)
        return out


def integrate_R(ser, zk, n, w, path, R0, ts, tgrid, rtol=1e-9):
    """dR/dt = (J R + tau) x'(t) along x = path(t); returns R at the times ts.  tau (mpmath) is sampled on tgrid and
    interpolated by a cubic spline (its relative interpolation error is printed); J is evaluated directly."""
    from scipy.interpolate import CubicSpline
    tv = np.array([ser.tau(path(t)[0]) for t in tgrid])
    sp = CubicSpline(tgrid, tv)
    mid = 0.5 * (tgrid[:-1] + tgrid[1:])
    sel = mid[:: max(1, len(mid) // 6)]
    err = max(float(np.max(np.abs(sp(t) - ser.tau(path(t)[0]))) / (1e-300 + np.max(np.abs(ser.tau(path(t)[0])))))
              for t in sel)

    def f(t, y):
        x, dx = path(t)
        r = y[:10] + 1j * y[10:]
        d = (jac(x, zk(x), n, w) @ r + sp(t)) * dx
        return np.concatenate([d.real, d.imag])

    def jf(t, y):
        x, dx = path(t)
        return cplx_block(jac(x, zk(x), n, w) * dx)

    sol = solve_ivp(f, (ts[0], ts[-1]), np.concatenate([R0.real, R0.imag]), method="Radau", t_eval=ts,
                    jac=jf, rtol=rtol, atol=1e-30)
    if not sol.success:
        raise RuntimeError(sol.message)
    return [sol.y[:10, i] + 1j * sol.y[10:, i] for i in range(len(ts))], err


def integrate_model(n, w, path, U0, ts, rtol=1e-13):
    def f(t, y):
        x, dx = path(t)
        u = y[:10] + 1j * y[10:]
        d = model_np(x, u, n, w) * dx
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(f, (ts[0], ts[-1]), np.concatenate([U0.real, U0.imag]), method="Radau", t_eval=ts, rtol=rtol,
                    atol=1e-18)
    if not sol.success:
        raise RuntimeError(sol.message)
    return [sol.y[:10, i] + 1j * sol.y[10:, i] for i in range(len(ts))]


def l1(v):
    return float(np.sum(np.abs(v)))


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = float(pos[0])
    K = int(pos[1]) if len(pos) > 1 else 4
    ser = Series(K, n)
    w = float(ser.w)
    # F_np against b8m_model.reduced_rhs (mpmath) at random points
    rng = np.random.default_rng(1)
    dev = 0.0
    for _ in range(3):
        x = complex(rng.uniform(0.5, 3), rng.uniform(0.2, 1))
        Z = ser.ZK(x) + 1e-3 * (rng.standard_normal(10) + 1j * rng.standard_normal(10))
        V, S = {O[i]: mp.mpc(Z[i]) for i in range(2)}, {O[i]: mp.mpc(Z[2 + i]) for i in range(2)}
        be, c = {E[i]: mp.mpc(Z[4 + i]) for i in range(3)}, {E[i]: mp.mpc(Z[7 + i]) for i in range(3)}
        r = BM.reduced_rhs(1, mp.mpc(x), mp.mpf(n), mp.mpf(w), V, S, be, c)
        ref = np.array([complex(r[0][m]) for m in O] + [complex(r[1][m]) for m in O] + [complex(r[2][k]) for k in E]
                       + [complex(r[3][k]) for k in E])
        dev = max(dev, float(np.max(np.abs(F_np(x, Z, n, w) - ref)) / np.max(np.abs(ref))))
    print(f"n = {n:g}, K = {K}: numpy reduced rhs vs b8m_model.reduced_rhs: {dev:.1e}", flush=True)
    Rout = np.sqrt(KAPPA1 * n / (np.cos(KAPPA) * np.log(n)))
    e_a = np.exp(1j * ALPHA)
    x0 = 1e-3
    zk = ZKfast(ser)
    ray = lambda s: (np.exp(s) * e_a, np.exp(s) * e_a)          # s = ln |x|
    rs = np.unique(np.concatenate([np.geomspace(0.02, Rout, 28), [0.5, 2.0, 5.0, Rout]]))
    rs = rs[rs > x0]
    ts = np.concatenate([[x0], rs])
    Rray, e1 = integrate_R(ser, zk, n, w, ray, np.zeros(10, dtype=complex), np.log(ts),
                           np.linspace(np.log(x0), np.log(Rout), 161))
    pts = [(r * e_a, R) for r, R in zip(ts[1:], Rray[1:])]
    arcs = [r for r in (0.5, 2.0, 5.0, Rout) if r < Rout + 1e-9]
    spl = [e1]
    for r in arcs:
        R0 = Rray[list(ts).index(r)]
        ang = np.linspace(0, ALPHA, 13)          # t = angle travelled from alpha
        arc = lambda t, r=r: (r * np.exp(1j * (ALPHA - t)), -1j * r * np.exp(1j * (ALPHA - t)))
        Rarc, e2 = integrate_R(ser, zk, n, w, arc, R0, ang, np.linspace(0, ALPHA, 31))
        spl.append(e2)
        pts += [(r * np.exp(1j * (ALPHA - t)), R) for t, R in zip(ang[1:], Rarc[1:])]
    print(f"tau spline: max relative interpolation error {max(spl):.1e}", flush=True)
    # (A1) next orders
    a1 = 0.0
    for x, R in pts:
        pred = ser.nxt(x)
        if l1(pred) > 0:
            a1 = max(a1, l1(R - pred) / l1(pred))
    rho_end = (1 + Rout) ** 2 / n
    print(f"(A1) remainder vs eps^(K+1) Z_(K+1) + eps^(K+2) Z_(K+2) on {len(pts)} points of the ray and the arcs: "
          f"max relative difference {a1:.2e} (expected O(rho^2) <= {rho_end ** 2:.1e}... O(rho) {rho_end:.1e})",
          flush=True)
    # (A2) independent integration of the original model along the ray and the outer arc
    U0 = to_model(np.concatenate([ser.ZK(x0 * e_a), [x0 * e_a]]), n)
    Ur = integrate_model(n, w, ray, U0, np.log(ts))
    a2, cnt = 0.0, 0
    for (x, R), U in zip(pts[:len(ts) - 1], Ur[1:]):
        Zd = from_model(U, x, n)
        diff = Zd - ser.ZK(x)
        if l1(R) > 1e-7 * l1(ser.ZK(x)):
            a2 = max(a2, l1(diff - R) / l1(R))
            cnt += 1
    print(f"(A2) Z - Z^K from the original model vs R, at {cnt} ray points where resolvable (|R| > 1e-7 |Z|): "
          f"max relative difference {a2:.2e}", flush=True)
    # (B) the bootstrap bounds
    hyp = None
    if "log" in OPTS:
        for line in open(OPTS["log"]):
            rr = json.loads(line)
            if rr["K"] == K and rr.get("closes") and abs(rr["L0"] - float(OPTS.get("L0", "73.89"))) < 1e-9:
                hyp = rr["hyp"]
    if hyp:
        kw = K + 2 if MUT == "weight" else K + 1
        worst = dict(V=0.0, S=0.0, b=0.0, c=0.0)
        for x, R in pts:
            X = 1 + abs(x)
            rho, th = X * X / n, abs(x) / X
            worst["V"] = max(worst["V"], l1(R[0:2]) / (hyp["V"] * rho ** kw))
            worst["S"] = max(worst["S"], l1(R[2:4]) / (hyp["S"] * th * rho ** kw))
            worst["b"] = max(worst["b"], l1(R[4:7]) / (hyp["b"] * th * rho ** kw))
            worst["c"] = max(worst["c"], l1(R[7:10]) / (hyp["c"] * rho ** kw))
        okB = all(v < 1 for v in worst.values())
        print(f"(B) max |R|/bound (h of the ln n0 = {OPTS.get('L0', '73.89')} certificate, weight rho^{kw}): "
              + ", ".join(f"{s} {v:.2e}" for s, v in worst.items()) + f"  {'OK' if okB else 'FAIL'}", flush=True)
    else:
        okB = True
        print("(B) skipped (no closing hypothesis given)")
    # (C) Lemma K along the arc |x| = r_C: fundamental matrix of the homogeneous fast equation vs the log-norm bound
    rC = min(5.0, Rout)
    worstC = 0.0
    th = np.linspace(0, ALPHA, 200)
    Eprop = np.eye(2, dtype=complex)
    bound = 0.0

    def lam_parts(x):
        Z = zk(x)
        c, be = Z[7:10], Z[4:7]
        Cm = Cmat(c)
        Bb = Cmat(ee(c, be))
        Om = np.diag([-1j * w, 1j * w])
        I = np.eye(2)
        Ci = np.linalg.inv(Cm)
        B = I - Bb @ Ci + Cm @ Om @ Ci
        A = (n - 1) * Cm @ Cm + Cm @ Bb - x**2 * Om - x**2 * B
        D = x**2 * I - Cm @ Cm
        Lam = np.linalg.solve(D, A)
        lam_s = (n - 1) / (x * (x * x - 1))                     # scalar part: Lam / x = lam_s (I + H')
        Hp = (Lam / x) / lam_s - I
        return Lam / x, lam_s, Hp

    for i in range(len(th) - 1):
        # the step from angle alpha - th[i] to alpha - th[i+1] (midpoint rule for the bound, exact exponential map
        # of the frozen generator for E)
        tm = 0.5 * (th[i] + th[i + 1])
        x = rC * np.exp(1j * (ALPHA - tm))
        dx = -1j * x * (th[i + 1] - th[i])
        G, lam_s, Hp = lam_parts(x)
        from scipy.linalg import expm
        Eprop = expm(G * dx) @ Eprop
        hn = 0.0 if MUT == "kernel" else float(np.max(np.sum(np.abs(Hp), axis=0)))
        bound += (lam_s * dx).real + abs(lam_s * dx) * hn
        worstC = max(worstC, np.log(np.max(np.sum(np.abs(Eprop), axis=0))) - bound)
    okC = worstC < 1e-6
    print(f"(C) Lemma K on the arc |x| = {rC:g}: max over the arc of ln||E|| - (bound exponent) = {worstC:.3e} "
          f"(must be <= 0); final ln||E|| = {np.log(np.max(np.sum(np.abs(Eprop), axis=0))):.3f}, bound {bound:.3f}  "
          f"{'OK' if okC else 'FAIL'}", flush=True)
    okA = a1 < 0.1 and a2 < 1e-3
    print(f"(A) {'OK' if okA else 'FAIL'}")
    print("ALL CHECKS OK" if (okA and okB and okC) else "SOME CHECK FAILED")
    return 0 if (okA and okB and okC) else 1


if __name__ == "__main__":
    sys.exit(main())
