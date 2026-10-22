#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md): numerical form of the one-harmonic (M = 1) outer problem on the complex domain
Sigma^+_n, used to check every estimate of the proof on actual solutions.

What it computes, at the Theorem A(M = 1) parameters (A*, omega*), for given n and truncation order K:
  * the exact centre-regular solution Z = (V_+, V_-, S_+, S_-, beta, c) near x = 0 as a power series in x^2
    (order-by-order recursion at the regular singular point, mpmath, 50 digits), with the gauge of the notes:
    c(0) = c^[K](0), V_mu(0) = 2 f c(0);
  * the remainder R = Z - Z^[K] along the paths of Lemma B8.1 (ray arg x = alpha from the centre, then arcs), from
    the remainder ODE  R' = (int_0^1 J(x, Z^K + t R) dt) R + tau(x),  tau = F(x, Z^K) - Z^K'  (tau in mpmath from the
    truncated polynomials of b8_formal.py at 256 bits; J the analytic Jacobian of the reduced system of b8_system.py),
    integrated with Radau on the real/imaginary split;
  * independently, the full system Z' = F(x, Z) from the Taylor start, integrated the same way, and Z - Z^K compared
    with R where R is resolvable in double precision (|R| > 1e-9 |Z|).  The two share only F (lambdified from the
    symbolically checked equations) and the Taylor start.

    python3 b8_numcheck.py n K [kappa1] [--mutate=...]
"""
import os
import sys
import numpy as np
import mpmath as mp
import sympy as sp
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flint
flint.ctx.prec = 256
import b8_formal as BF

mp.mp.dps = 50

# ------------------------------------------------------------------ reduced system, symbolic -> numeric
_x, _n, _w = sp.symbols("x n omega")
_Z = sp.symbols("Vp Vm Sp Sm be c")


def _rhs_sym():
    Vp, Vm, Sp, Sm, be, c = _Z
    eps = 1 / _n
    b = 1 + be / (_n - 1)
    out = []
    S = {1: Sp, -1: Sm}
    V = {1: Vp, -1: Vm}
    for mu in (1, -1):
        out.append(sp.I * mu * _w * S[mu] / _x)
    for mu in (1, -1):
        A = (_n - 1) * c**2 * b - _x**2 * (1 + 2 * sp.I * mu * _w - be)
        B = 1 + sp.I * mu * _w - be
        out.append((A * S[mu] - B * _x**2 * V[mu]) / (_x * (_x**2 - c**2)))
    H = Sp * Sm + _x**2 * (Vp + Sp) * (Vm + Sm) / c**2
    out.append((-(_n - 1) * be - be**2 + (1 - eps) * b * H) / _x)
    out.append(c * be / _x)
    return sp.Matrix(out)


_F = _rhs_sym()
_J = _F.jacobian(sp.Matrix(_Z))
F_np = sp.lambdify((_x, _n, _w) + _Z, list(_F), "numpy")
J_np = sp.lambdify((_x, _n, _w) + _Z, _J.tolist(), "numpy")
F_mp = sp.lambdify((_x, _n, _w) + _Z, list(_F), "mpmath")


def Fv(x, n, w, Z):
    return np.array(F_np(x, n, w, *Z), dtype=complex)


def Jv(x, n, w, Z):
    return np.array(J_np(x, n, w, *Z), dtype=complex)


# ------------------------------------------------------------------ parameters and the truncated formal series
def setup(n, K):
    f, P, om = BF.params()
    T, raw = BF.truncated(K, f, P, om)
    V, S, be, c = raw
    to_mp = lambda z: mp.mpc(mp.mpf(z.real.mid().str(60, radius=False)), mp.mpf(z.imag.mid().str(60, radius=False)))
    eps = mp.mpf(1) / n

    def combine(bpoly):
        """sum_j eps^j p_j(x) -> list of mp coefficients in x"""
        deg = max(q.degree() for q in bpoly.values()) + 1
        cc = [mp.mpc(0)] * deg
        for j, q in bpoly.items():
            for m, cm in enumerate(q.coeffs()):
                cc[m] += eps**j * to_mp(cm)
        return cc

    comb = [combine(T["V"][1]), combine(T["V"][-1]), combine(T["S"][1]), combine(T["S"][-1]), combine(T["be"]),
            combine(T["c"])]
    Rs = BF.residuals(K, f, P, om, T)
    res = {k: combine(BF.structural(v, K + 1 if k.startswith("rho4") else K, 2)[0]) for k, v in Rs.items()}
    npc = lambda cc: np.array([complex(z) for z in cc][::-1])      # numpy polyval order
    st = dict(n=n, K=K, f=to_mp(f), w=to_mp(om), comb=comb, wf=complex(to_mp(om)),
              npZ=[npc(cc) for cc in comb], npdZ=[np.polyder(npc(cc)) for cc in comb],
              npres={k: npc(v) for k, v in res.items()})
    return st


def ZK_np(st, x):
    return np.array([np.polyval(p, x) for p in st["npZ"]])


def tau_np(st, x):
    """F(x, Z^K) - Z^K' from the residual polynomials: (0, 0, -rho2_+/(x D^K), -rho2_-/(x D^K), -rho3t/(x c^2), -rho4/x)"""
    Z = ZK_np(st, x)
    c = Z[5]
    D = x * x - c * c
    r = st["npres"]
    return np.array([0, 0, -np.polyval(r["rho2_+1"], x) / (x * D), -np.polyval(r["rho2_-1"], x) / (x * D),
                     -np.polyval(r["rho3t"], x) / (x * c * c), -np.polyval(r["rho4"], x) / x])


def ZK(st, x, deriv=False):
    out = []
    for cc in st["comb"]:
        if deriv:
            out.append(sum(m * cc[m] * x ** (m - 1) for m in range(1, len(cc))))
        else:
            out.append(sum(cc[m] * x**m for m in range(len(cc))))
    return out


def tau(st, x):
    """F(x, Z^K) - Z^K'(x) in mpmath"""
    x = mp.mpc(x)
    Z = ZK(st, x)
    Fx = F_mp(x, mp.mpf(st["n"]), st["w"], *Z)
    dZ = ZK(st, x, True)
    return np.array([complex(Fx[i] - dZ[i]) for i in range(6)])


# ------------------------------------------------------------------ exact Taylor series at the centre (series in x^2)
def taylor(st, J=90):
    n = mp.mpf(st["n"])
    w = st["w"]
    eps = 1 / n
    c0 = st["comb"][5][0]
    f = st["f"]
    V = {1: [2 * f * c0], -1: [2 * f * c0]}
    S = {1: [mp.mpc(0)], -1: [mp.mpc(0)]}
    be, c = [mp.mpc(0)], [c0]

    def cv(a, b, j):
        return sum(a[i] * b[j - i] for i in range(j + 1) if i < len(a) and j - i < len(b))

    for j in range(1, J + 1):
        c2 = [cv(c, c, i) for i in range(j + 1)]
        bb = [1 + be[i] / (n - 1) if i == 0 else be[i] / (n - 1) for i in range(len(be))]
        c2b = [cv(c2, bb, i) for i in range(j + 1)]
        for mu in (1, -1):
            imw = mp.mpc(0, mu) * w
            Sd = [2 * i * S[mu][i] for i in range(len(S[mu]))]          # coefficients of x S'
            rhs_rest = (sum((n - 1) * c2b[j - i] * S[mu][i] for i in range(1, j))
                        - (1 + 2 * imw) * S[mu][j - 1] + cv(be, S[mu], j - 1)
                        - (1 + imw) * V[mu][j - 1] + cv(be, V[mu], j - 1))
            lhs_rest = (Sd[j - 1] if j - 1 < len(Sd) else 0) - sum(c2[j - i] * Sd[i] for i in range(1, j))
            S[mu].append((rhs_rest - lhs_rest) / (-2 * j * c2[0] - (n - 1) * c2b[0]))
        Y = {mu: [V[mu][i] + S[mu][i] for i in range(j)] for mu in (1, -1)}
        ic2 = [1 / c2[0]]
        for i in range(1, j):
            ic2.append(-sum(c2[k] * ic2[i - k] for k in range(1, i + 1)) / c2[0])
        YY = [cv(Y[1], Y[-1], i) for i in range(j)]
        H = [mp.mpc(0)] + [cv(S[1], S[-1], i) + cv(YY, ic2, i - 1) for i in range(1, j + 1)]
        bH = sum(be[i] * H[j - i] for i in range(1, j))
        bb2 = cv(be, be, j)
        be.append(((1 - eps) * H[j] + eps * bH - bb2) / (2 * j + n - 1))
        c.append(cv(c, be, j) / (2 * j))
        for mu in (1, -1):
            V[mu].append(mp.mpc(0, mu) * w * S[mu][j] / (2 * j))
    return [V[1], V[-1], S[1], S[-1], be, c]


def taylor_eval(ser, x, deriv=False):
    x2 = mp.mpc(x) ** 2
    out = []
    for s in ser:
        if deriv:
            out.append(sum(2 * j * s[j] * x2**j for j in range(1, len(s))) / mp.mpc(x))
        else:
            out.append(sum(s[j] * x2**j for j in range(len(s))))
    return out


# ------------------------------------------------------------------ paths and integration
GL_T = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
GL_W = np.array([5 / 18, 8 / 18, 5 / 18])


class Interp:
    """Z^K and tau on a path, by direct evaluation (mpmath) at the solver's request; cached on a grid of t"""

    def __init__(self, st):
        self.st = st
        self.cache = {}

    def at(self, x):
        return ZK_np(self.st, x), tau_np(self.st, x)


def path_fun(kind, a, b, r=None):
    """x(t), x'(t) for t in [0, 1]: kind 'line' from a to b, 'arc' at radius r from angle a to angle b"""
    if kind == "line":
        return (lambda t: a + (b - a) * t), (lambda t: (b - a))
    return (lambda t: r * np.exp(1j * (a + (b - a) * t))), (lambda t: 1j * (b - a) * r * np.exp(1j * (a + (b - a) * t)))


def integrate_R(st, ip, R0, xf, dxf, ts, rtol=1e-9, atol=1e-40):
    n, w = st["n"], st["wf"]

    def rhs(t, y):
        R = y[:6] + 1j * y[6:]
        x = xf(t)
        Zk, ta = ip.at(x)
        Jm = sum(GL_W[q] * Jv(x, n, w, Zk + GL_T[q] * R) for q in range(3))
        d = (Jm @ R + ta) * dxf(t)
        return np.concatenate([d.real, d.imag])

    def jac(t, y):
        R = y[:6] + 1j * y[6:]
        x = xf(t)
        Zk, _ = ip.at(x)
        Jm = Jv(x, n, w, Zk + 0.5 * R) * dxf(t)
        return np.block([[Jm.real, -Jm.imag], [Jm.imag, Jm.real]])

    y0 = np.concatenate([R0.real, R0.imag])
    sol = solve_ivp(rhs, (0.0, 1.0), y0, method="Radau", jac=jac, t_eval=ts, rtol=rtol, atol=atol)
    if sol.status != 0:
        raise RuntimeError(sol.message)
    return sol.y[:6].T + 1j * sol.y[6:].T


def integrate_Z(st, Z0, xf, dxf, ts, rtol=1e-12, atol=1e-16):
    n, w = st["n"], st["wf"]

    def rhs(t, y):
        Z = y[:6] + 1j * y[6:]
        d = Fv(xf(t), n, w, Z) * dxf(t)
        return np.concatenate([d.real, d.imag])

    def jac(t, y):
        Z = y[:6] + 1j * y[6:]
        Jm = Jv(xf(t), n, w, Z) * dxf(t)
        return np.block([[Jm.real, -Jm.imag], [Jm.imag, Jm.real]])

    sol = solve_ivp(rhs, (0.0, 1.0), np.concatenate([Z0.real, Z0.imag]), method="Radau", jac=jac, t_eval=ts,
                    rtol=rtol, atol=atol)
    if sol.status != 0:
        raise RuntimeError(sol.message)
    return sol.y[:6].T + 1j * sol.y[6:].T


def geometry(n, kappa1=1.0, kappa=np.pi / 4 * 0.9, delta=0.2):
    alpha = np.pi / 4 + kappa / 2
    Rn = np.sqrt(kappa1 * n / np.log(n))
    return dict(alpha=alpha, Rn=Rn, delta=delta, kappa=kappa, kappa1=kappa1)


def run(n, K, kappa1=1.0, nr=24, narc=12, verbose=True):
    st = setup(n, K)
    ser = taylor(st)
    g = geometry(n, kappa1)
    al, Rn, de = g["alpha"], g["Rn"], g["delta"]
    ip = Interp(st)
    x0 = 0.25 * np.exp(1j * al)
    Zt = taylor_eval(ser, x0)
    R0 = np.array([complex(Zt[i] - ZK(st, mp.mpc(x0))[i]) for i in range(6)])
    radii = np.unique(np.concatenate([np.geomspace(0.26, Rn, nr), [1 - de / 2, 1 + de / 2]]))
    radii = radii[radii > 0.25]
    xf, dxf = path_fun("line", x0, Rn * np.exp(1j * al))
    ts = (radii - 0.25) / (Rn - 0.25)
    Rray = integrate_R(st, ip, R0, xf, dxf, ts)
    pts = []          # (x, R)
    for r, Rc in zip(radii, Rray):
        pts.append((r * np.exp(1j * al), Rc))
        # arc down from alpha to the lowest angle in Sigma at this radius
        if abs(r - 1) < de / 2:
            phimin = np.arccos((r * r + 1 - de * de / 4) / (2 * r))
        else:
            phimin = 0.0
        tsa = np.linspace(0, 1, narc + 1)[1:]
        xa, dxa = path_fun("arc", al, phimin, r)
        Ra = integrate_R(st, ip, Rc, xa, dxa, tsa)
        for t, Rv in zip(tsa, Ra):
            pts.append((xa(t), Rv))
    return st, ser, g, pts


if __name__ == "__main__":
    n = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    k1 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    st, ser, g, pts = run(n, K, k1)
    rho = lambda x: (1 + abs(x)) ** 2 / n
    th = lambda x: abs(x) / (1 + abs(x))
    names = ["V+", "V-", "S+", "S-", "beta", "c"]
    wts = [lambda x: rho(x) ** (K + 1)] * 2 + [lambda x: th(x) * rho(x) ** (K + 1)] * 3 + [lambda x: rho(x) ** (K + 1)]
    mx = np.zeros(6)
    for x, R in pts:
        for i in range(6):
            mx[i] = max(mx[i], abs(R[i]) / wts[i](x))
    print(f"n = {n:g}, K = {K}, R_n = {g['Rn']:.3f}, alpha = {g['alpha']:.4f}: sup |R_i| / w_i over the grid:")
    print("  " + "  ".join(f"{names[i]} {mx[i]:.4g}" for i in range(6)))
