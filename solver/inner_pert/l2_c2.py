#!/usr/bin/env python3
"""c2 of the l = 2 mode, lambda_2 = c2/n + O(1/n^2), in the M-harmonic truncation (notes/L2_MODE.md 9, #1010).

On the lapse and the scalar the l = 2 problem is the l = 0 problem conjugated by sY (L2_MODE.md 8.3), so the l = 2
mode D2 Z_n + W2/n is sY times the l = 0 perturbation d_s Z_n + w/n at lambda = 2, with w forced by Ft = F/(sY)
(l2_c2_forcing.py). The s-translation d_s Z_0 acts harmonic by harmonic, so it is an exact mode of every truncated
model at lambda = 2, with profile d1 (the xi^3 coefficient of y) and c(0) = 2 g1 (twice the xi^2 coefficient of c);
the time-gauge modes, which would remove that c(0), are not exact in a truncated model. The slice

    c(0) = 2 g1 (phi . f)        (phi . f2 = 1, phi on the harmonic-1 profile components)

contains d_s Z_0 and is a gauge slice of the full problem whenever phi . f_D != 0. Solvability on it:

    c2 = - phi . A^-1 S(W^F) / phi . A^-1 S(W^lam),

A = S(2) the profile-to-multiplier map at c(0) = 0, W^F the response to -Ft with zero data, W^lam the d/dlambda
component of the mode. A is nearly singular when the model's own lambda* is near 2, and then c2 does not depend on
phi; the spread over phi (and the left singular vector of A's Schur complement) is reported as a truncation measure.
The l = 0 forcing (SymForce of first_order_jet.py, and the same terms from l2_mode's reduction) through the same code
gives lambda_1 (planted). Deflation for M = 5 as first_order_hier.py.

The forcing enters the jets through a small expression graph in the model's own truncated-product algebra. Its
xi-derivatives come from the model's ODEs (Leibniz rule through truncated products), so the forcing's series at
xi0 and its right-hand side along the integration are one analytic function of xi.

    python3 l2_c2.py --checks                 the graph checks (seconds): evaluator against SymForce, the l = 0
                                              identity, series against right-hand side; exit 0 iff all pass
    python3 l2_c2.py --checks --mutate=ident  a dropped term in the converted l = 0 forcing: the identity must fail
    python3 l2_c2.py M [--xi1=0.2] [--forcing=l2,l2_ctrl,l0,sym,G] [--lamcheck]
                                              c2 (and lambda_1 for l0, sym) at the stored zero of M (data/
                                              first_order_hier_M{M}_xi{xi1}.json); results to data/l2_c2_M{M}_xi{xi1}_{forcings}.json
    python3 l2_c2.py 5 --forcing=l0,sym --mutate=gsign    every graph forcing's sign flipped: the planted lambda_1
                                              must leave its 1% bar (it changes sign), exit 1. At M = 5 a run with
                                              l0 or sym exits 1 whenever their lambda_1 is more than 1% from 3.0324
    python3 l2_c2.py 1 --planted-lam          the planted lambda* of M = 1 through first_order_hier.find_lam
"""
import json
import os
import sys
import time

import numpy as np
import sympy as sp
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import first_order_jet as FJ                            # noqa: E402
import first_order_run as R                             # noqa: E402
import first_order_hier as H                            # noqa: E402
from first_order_jet import JetModel, Dir, Fh_from, W0, SymForce  # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
LAM1_FIT = 3.0324                                       # FIRST_ORDER.md 6, the double-double fit of lambda_1
FJ.MUTATE.update(m for m in MUT.split(",") if m)
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


# ---------------------------------------------------------------- the expression graph
class Node:
    __slots__ = ("op", "args", "par", "id")

    def __init__(self, op, args, par, nid):
        self.op, self.args, self.par, self.id = op, args, par, nid


class Graph:
    """Nodes over the model's fields Yh = y/xi, Sh = sigma/xi^2, C = c (odd, odd, even). Products are the model's
    truncated products (convolution kept on |mu| <= L, then projected on odd |mu| <= M or even |mu| <= M + 1), so
    the association of every product is fixed by the graph. E is Euler's xi d/dxi, from the model's ODEs at
    lambda = 0 (the background): E Yh = [C Sh] - (1 + w d_theta) Yh, E C = [C Ec], Ec = xi^2 [Yh Yh]/2,
    E Sh = (E[C Yh]) / xi^2 - (2 + w d_theta) Sh."""

    def __init__(self, w):
        self.w = w
        self.memo, self.Emem = {}, {}
        self.Y, self.S, self.C = self.mk("Y", (), 1), self.mk("S", (), 1), self.mk("C", (), 0)

    def mk(self, op, args, par):
        key = (op, args)
        n = self.memo.get(key)
        if n is None:
            n = Node(op, args, par, len(self.memo))
            self.memo[key] = n
        return n

    def add(self, *terms):
        """terms (coef, node | None); like nodes combined, zeros dropped; returns None for zero"""
        acc = {}
        for c, nd in terms:
            if nd is None or c == 0:
                continue
            if nd.op == "add":
                for c2, n2 in nd.args:
                    acc[n2.id] = (acc.get(n2.id, (0, n2))[0] + c * c2, n2)
            else:
                acc[nd.id] = (acc.get(nd.id, (0, nd))[0] + c, nd)
        items = tuple(sorted(((c, nd) for c, nd in acc.values() if c != 0), key=lambda p: p[1].id))
        if not items:
            return None
        pars = {nd.par for _, nd in items}
        assert len(pars) == 1, "a sum of odd and even nodes"
        if len(items) == 1 and items[0][0] == 1:
            return items[0][1]
        return self.mk("add", items, pars.pop())

    def scale(self, c, a):
        return self.add((c, a))

    def mul(self, a, b):
        if a is None or b is None:
            return None
        ka = a.args[1] if a.op == "xi2" else 0
        kb = b.args[1] if b.op == "xi2" else 0
        a0 = a.args[0] if a.op == "xi2" else a
        b0 = b.args[0] if b.op == "xi2" else b
        if a0.id > b0.id:
            a0, b0 = b0, a0
        return self.xi2(self.mk("mul", (a0, b0), a0.par ^ b0.par), ka + kb)

    def dth(self, a):
        if a is None:
            return None
        if a.op == "xi2":
            return self.xi2(self.dth(a.args[0]), a.args[1])
        return self.mk("dth", (a,), a.par)

    def xi2(self, a, k):
        if a is None:
            return None
        if a.op == "xi2":
            a, k = a.args[0], a.args[1] + k
        return a if k == 0 else self.mk("xi2", (a, k), a.par)

    def inv(self, c):
        return self.mk("inv", (c,), 0)

    def solve(self, c, r):
        if r is None:
            return None
        return self.mk("solve", (c, r), r.par)

    def E(self, a):
        if a is None:
            return None
        hit = self.Emem.get(a.id)
        if hit is not None:
            return hit
        op = a.op
        if op == "Y":
            out = self.add((1, self.mul(self.C, self.S)), (-self.w, self.dth(self.Y)), (-1, self.Y))
        elif op == "C":
            out = self.mul(self.C, self.Ec())
        elif op == "S":
            out = self.add((1, self.xi2(self.E(self.mul(self.C, self.Y)), -1)), (-2, self.S), (-self.w, self.dth(self.S)))
        elif op == "add":
            out = self.add(*[(c, self.E(nd)) for c, nd in a.args])
        elif op == "mul":
            x, y = a.args
            out = self.add((1, self.mul(self.E(x), y)), (1, self.mul(x, self.E(y))))
        elif op == "dth":
            out = self.dth(self.E(a.args[0]))
        elif op == "xi2":
            x, k = a.args
            out = self.add((2 * k, a), (1, self.xi2(self.E(x), k)))
        elif op == "inv":
            c = a.args[0]
            out = self.solve(c, self.scale(-1, self.mul(self.E(c), a)))
        elif op == "solve":
            c, r = a.args
            out = self.solve(c, self.add((1, self.E(r)), (-1, self.mul(self.E(c), a))))
        else:
            raise ValueError(op)
        self.Emem[a.id] = out
        return out

    def Ec(self):
        """E = y^2/2 = xi^2 [Yh Yh]/2 (the model's E, even)"""
        return self.xi2(self.scale(0.5, self.mul(self.Y, self.Y)), 1)

    def size(self, roots):
        return len(topo(roots))


def children(n):
    if n.op == "add":
        return [p[1] for p in n.args]
    if n.op in ("mul", "solve"):
        return list(n.args)
    if n.op in ("dth", "inv"):
        return [n.args[0]]
    if n.op == "xi2":
        return [n.args[0]]
    return []


def topo(roots):
    order, seen = [], set()
    for r in roots:
        if r is None:
            continue
        st = [(r, False)]
        while st:
            n, done = st.pop()
            if done:
                order.append(n)
                continue
            if n.id in seen:
                continue
            seen.add(n.id)
            st.append((n, True))
            for ch in children(n):
                if ch.id not in seen:
                    st.append((ch, False))
    return order


class Space:
    """harmonic layout -L..L and the two projections: the model's (odd |mu| <= M, even |mu| <= M + 1, L = M + 2), or
    with full=True every harmonic up to L (an untruncated test)"""

    def __init__(self, M, L=None, full=False):
        self.M = M
        self.L = L = L if L is not None else M + 2
        k = np.arange(-L, L + 1)
        self.k = k
        lo, le = (L, L) if full else (M, M + 1)
        self.mask = {1: ((np.abs(k) <= lo) & (k % 2 == 1)).astype(float),
                     0: ((np.abs(k) <= le) & (k % 2 == 0)).astype(float)}
        self.imu = 1j * k

    def conv(self, a, b, par):
        L = self.L
        return np.convolve(a, b)[L:3 * L + 1] * self.mask[par]

    def cmat(self, c):
        """the matrix of z -> [c z] on the even harmonics"""
        N = 2 * self.L + 1
        Mx = np.zeros((N, N), complex)
        for j in range(N):
            ej = np.zeros(N, complex)
            ej[j] = 1
            Mx[:, j] = self.conv(c, ej, 0)
        ev = self.mask[0] > 0
        return Mx[np.ix_(ev, ev)], ev


def eval_point(order, sp_, Yh, Sh, Cc, xi):
    """values (harmonic vectors) of the nodes in `order` at one point xi"""
    v = {}
    cm = {}
    for n in order:
        op = n.op
        if op == "Y":
            x = Yh
        elif op == "S":
            x = Sh
        elif op == "C":
            x = Cc
        elif op == "add":
            x = sum(c * v[nd.id] for c, nd in n.args)
        elif op == "mul":
            x = sp_.conv(v[n.args[0].id], v[n.args[1].id], n.par)
        elif op == "dth":
            x = sp_.imu * v[n.args[0].id]
        elif op == "xi2":
            x = v[n.args[0].id] * xi ** (2 * n.args[1])
        elif op in ("inv", "solve"):
            c = n.args[0]
            if c.id not in cm:
                cm[c.id] = sp_.cmat(v[c.id])
            Mx, ev = cm[c.id]
            rhs = np.zeros(2 * sp_.L + 1, complex)
            if op == "inv":
                rhs[sp_.L] = 1
            else:
                rhs = v[n.args[1].id]
            x = np.zeros(2 * sp_.L + 1, complex)
            x[ev] = np.linalg.solve(Mx, rhs[ev])
            x = x * sp_.mask[n.par]
        v[n.id] = x
    return v


def eval_series(order, sp_, Ys, Ss, Cs, q):
    """series in u = xi^2/q of the nodes (arrays (K, 2L+1)); returns values and the largest coefficient a negative
    xi-power dropped (relative to the node's largest), which must be at rounding level"""
    K = Ys.shape[0]
    v = {}
    dropped = 0.0
    for n in order:
        op = n.op
        if op == "Y":
            x = Ys
        elif op == "S":
            x = Ss
        elif op == "C":
            x = Cs
        elif op == "add":
            x = sum(c * v[nd.id] for c, nd in n.args)
        elif op == "mul":
            a, b = v[n.args[0].id], v[n.args[1].id]
            L = sp_.L
            x = convolve2d(a, b)[:K, L:3 * L + 1] * sp_.mask[n.par][None, :]
        elif op == "dth":
            x = sp_.imu[None, :] * v[n.args[0].id]
        elif op == "xi2":
            a, k = v[n.args[0].id], n.args[1]
            x = np.zeros_like(a)
            if k > 0:
                x[k:] = a[:K - k] * q ** k
            else:
                m = -k
                x[:K - m] = a[m:] * q ** k
                scale = np.abs(a[:m + 4]).max() or 1.0
                dropped = max(dropped, np.abs(a[:m]).max() / scale)
        elif op in ("inv", "solve"):
            assert n.par == 0, "an odd solve"
            c = v[n.args[0].id]
            Mx, ev = sp_.cmat(c[0])
            r = np.zeros_like(c)
            if op == "inv":
                r[0, sp_.L] = 1
            else:
                r = v[n.args[1].id]
            x = np.zeros_like(c)
            for m in range(K):
                acc = r[m].copy()
                for i in range(1, m + 1):
                    acc -= sp_.conv(c[i], x[m - i], n.par)
                x[m, ev] = np.linalg.solve(Mx, acc[ev])
                x[m] *= sp_.mask[n.par]
        v[n.id] = x
    return v, dropped


# ---------------------------------------------------------------- forcings as graphs
def sym_graph(g):
    """first_order_jet.SymForce's forcing, in its own association: F_y = -2 [C Th]/xi^2, Th = xi s_xi - 2 s + [E s];
    F_c = -2 [C (Psi + 2E)]/xi^2, Psi = [s s]/2 - xi^2 [Yh (Yh + E Yh)]"""
    sig = g.xi2(g.S, 1)
    Ee = g.Ec()
    Th = g.add((1, g.E(sig)), (-2, sig), (1, g.mul(Ee, sig)))
    Fy = g.scale(-2, g.xi2(g.mul(g.C, Th), -1))
    Psi = g.add((0.5, g.mul(sig, sig)), (-1, g.xi2(g.mul(g.Y, g.add((1, g.Y), (1, g.E(g.Y)))), 1)))
    Fc = g.scale(-2, g.xi2(g.mul(g.C, g.add((1, Psi), (2, Ee))), -1))
    return Fy, Fc


NP = 8
Psym = sp.symbols(f"P1:{NP + 1}")
Csym = sp.symbols(f"C0:{NP}")
tsym = sp.Symbol("t", positive=True)
ssym = sp.Symbol("s", positive=True)


class PhysQ:
    """(T-power, node) of Phi^(k) (k >= 1) and c^(k) (k >= 0), t-derivatives at fixed s, T = -t, built on demand.
    Phi_t = C Y/2 with Y = sqrt(2) Yh / T; D_t (p, q) = (p + 1, p q + E q + w d_theta q)"""

    def __init__(self, g):
        self.g = g
        self.Ph = [None, (1, g.scale(1 / np.sqrt(2), g.mul(g.C, g.Y)))]
        self.Cq = [(0, g.C)]

    def Dt(self, pq):
        g = self.g
        return (pq[0] + 1, g.add((pq[0], pq[1]), (1, g.E(pq[1])), (g.w, g.dth(pq[1]))))

    def phi(self, k):
        while len(self.Ph) <= k:
            self.Ph.append(self.Dt(self.Ph[-1]))
        return self.Ph[k]

    def c(self, k):
        while len(self.Cq) <= k:
            self.Cq.append(self.Dt(self.Cq[-1]))
        return self.Cq[k]


def expr_graph(g, expr, cache):
    """a rational function of P_k, C_k (and t) with a monomial denominator -> (T-power, node)"""
    num, den = sp.fraction(sp.together(sp.sympify(expr)))
    gens = list(Psym) + list(Csym) + [tsym, ssym]
    dp = sp.Poly(den, *gens)
    if len(dp.terms()) != 1:
        raise ValueError(f"denominator not a monomial: {den}")
    dmon, dcoef = dp.terms()[0]
    out, tp = [], None
    invC = g.inv(g.C)
    for mon, coef in sp.Poly(sp.expand(num), *gens).terms():
        e = [a - b for a, b in zip(mon, dmon)]
        cf = complex(coef / dcoef)
        p = 0
        nd = None
        facs = []
        for k in range(NP):
            facs += [cache.phi(k + 1)] * e[k]
        for k in range(1, NP):
            if e[NP + k] < 0:
                raise ValueError("a negative power of a derivative of c")
            facs += [cache.c(k)] * e[NP + k]
        facs += [(0, g.C)] * max(e[NP], 0) + [(0, invC)] * max(-e[NP], 0)
        et = e[2 * NP]                                  # t^et = (-1)^et T^et: T-power -et
        es = e[2 * NP + 1]                              # s^es = (T^2 xi^2 / 2)^es: T-power -2 es, xi^(2 es)
        if es < 0:
            raise ValueError("a negative power of s")
        cf *= (-1) ** et * 0.5 ** es
        p = -et - 2 * es
        for pp, qq in facs:
            p += pp
            nd = qq if nd is None else g.mul(nd, qq)
        nd = g.xi2(nd, es)
        if tp is None:
            tp = p
        elif tp != p:
            raise ValueError(f"inhomogeneous in T: {tp} and {p}")
        out.append((cf, nd))
    return tp, g.add(*out)


def converted_graph(g, Fd, scale=1.0):
    """F_y = sqrt(2) xi^2 [C Ft_W], F_c = xi^2 [C Ft_E] (with T^2 xi^2 = rho^2; the T-powers checked: 4 -> 2)"""
    cache = PhysQ(g)
    pw, nW = expr_graph(g, Fd["W"], cache)
    pe, nE = expr_graph(g, Fd["E^t_t"], cache)
    if "ident" in MUT and nE.op == "add":             # the identity check's mutation: the last term of Ft_E dropped
        nE = g.add(*nE.args[:-1])
    if pw != 4 or pe != 4:
        raise ValueError(f"T-powers {pw}, {pe}, not 4")
    Fy = g.scale(scale * np.sqrt(2), g.xi2(g.mul(g.C, nW), 1))
    Fc = g.scale(scale, g.xi2(g.mul(g.C, nE), 1))
    return Fy, Fc


def load_forcings():
    fn = os.path.join(HERE, "data", "l2_c2_forcing.json")
    d = json.load(open(fn))
    return {k: v for k, v in d.items() if k != "symbols"}


class GraphForce:
    """the jets' forcing interface (as first_order_jet.SymForce) for a graph builder build(g) -> (F_y, F_c)"""

    def __init__(self, bit, build):
        self.bit, self.build = bit, build

    def prepare(self, model, xi0, J):
        g = Graph(model.w)
        self.Fy, self.Fc = self.build(g)
        self.order = topo([self.Fy, self.Fc])
        self.sp = Space(model.M)
        self.q = xi0 * xi0
        K = J + 16
        bg = JetModel(model.M, model.w, None, [], [0], {})
        bg.d0[0] = model.d0[0]
        bg.series_at(xi0, K, stop=False)
        ser = bg.ser
        Ys = np.array([d[0] for d in ser["D"][:K]])
        Ss = np.array([x[0] for x in ser["S"][:K]])
        Cs = np.array([x[0] for x in ser["G"][:K]])
        Kc = min(len(Ys), len(Ss), len(Cs))
        v, self.dropped = eval_series(self.order, self.sp, Ys[:Kc], Ss[:Kc], Cs[:Kc], self.q)
        self.sy, self.sc = v[self.Fy.id], v[self.Fc.id]
        if "series" in MUT:                              # the series check's mutation: the series off by 1e-6
            self.sy = self.sy * (1 + 1e-6)
        self.sgn = -1.0 if "gsign" in MUT else 1.0       # the planted lambda_1's mutation: every graph forcing flipped
        self.sy, self.sc = self.sgn * self.sy, self.sgn * self.sc
        self.slot = model.idx[self.bit]
        self.L = model.L
        self.reg = (np.abs(self.sy[0]).max() / np.abs(self.sy[1:5]).max(),        # F_y, F_c at u = 0: must vanish
                    np.abs(self.sc[0]).max() / np.abs(self.sc[1:5]).max())

    def _put(self, model, arr):
        out = np.zeros((model.nc, 2 * model.L + 1), complex)
        out[self.slot] = arr
        return out

    def series_y(self, j, model):
        return self._put(model, -self.sy[j + 1] / self.q)

    def series_c(self, m, model):
        return self._put(model, self.sc[m])

    def point(self, xi, D0, S0, C0):
        v = eval_point(self.order, self.sp, D0 / xi, S0 / (xi * xi), C0, xi)
        return self.sgn * v[self.Fy.id], self.sgn * v[self.Fc.id]

    def rhs(self, xi, model, D, S, C, E, dD, dS):
        fy, fc = self.point(xi, D[0], S[0], C[0])
        return self._put(model, fy), self._put(model, fc)


# ---------------------------------------------------------------- checks of the graph
def bg_series(x, M, xi0, K):
    bg = JetModel(M, W0 * x[1], Fh_from(x, M), [], [0], {})
    bg.series_at(xi0, K, stop=False)
    return bg


def checks():
    ok = True

    def say(label, good, detail):
        nonlocal ok
        ok &= good
        print(f"{label}: {detail}  ({'as required' if good else '** NOT AS REQUIRED **'})", flush=True)
    Fd = load_forcings()
    dat = json.load(open(os.path.join(HERE, "data", "first_order_hier_M3_xi0.2.json")))
    x, M = np.array(dat["x"]), 3
    w = W0 * x[1]
    # 1. the graph evaluator against SymForce itself (series and right-hand side), M = 3 background
    xi0 = 0.06 * np.exp(1j * (np.pi / 2 + FJ.DELTA) / 2)
    J = 60
    dirs = [Dir("W", lam=2.0, force=SymForce(1))]
    jm = JetModel(M, w, Fh_from(x, M), dirs, [0, 1], {})
    sf = SymForce(1)
    sf.prepare(jm, xi0, J + 3)
    gf = GraphForce(1, sym_graph)
    gf.prepare(jm, xi0, J + 3)
    ds = max(np.abs(sf.series_y(j, jm) - gf.series_y(j, jm)).max() / (np.abs(sf.series_y(j, jm)).max() + 1e-300)
             for j in range(J))
    dc = max(np.abs(sf.series_c(m, jm) - gf.series_c(m, jm)).max() / (np.abs(sf.series_c(m, jm)).max() + 1e-300)
             for m in range(1, J))
    # right-hand side at a point on the ray, from the background series summed there
    jm2 = JetModel(M, w, Fh_from(x, M), dirs, [0, 1], {})
    Y, jt, terr = jm2.series_at(xi0, 200)
    D, S, Cc = jm2.unpack(Y)
    xi = xi0
    o, e = jm2.odd, jm2.even
    dD = (jm2.mul(Cc, S, o) - xi * (jm2.wmul(D) + jm2.lmul(D))) / (xi * xi)
    Ee = jm2.mul(D, D, e) / 2
    cE = jm2.mul(Cc, Ee, e)
    r2 = jm2.mul(Cc, xi * dD, o) + jm2.mul(cE, D, o) - jm2.mul(Cc, D, o)
    dS0 = (r2 - xi * (jm2.wmul(S) + jm2.lmul(S))) / (xi * xi)
    fy0, fc0 = sf.rhs(xi, jm2, D, S, Cc, Ee, dD, dS0)
    fy1, fc1 = gf.rhs(xi, jm2, D, S, Cc, Ee, dD, dS0)
    dr = max(np.abs(fy0 - fy1).max() / np.abs(fy0).max(), np.abs(fc0 - fc1).max() / np.abs(fc0).max())
    say("1. graph evaluator = SymForce (series orders < 60, and right-hand side at xi0)", max(ds, dc, dr) < 1e-11,
        f"series y {ds:.1e}, series c {dc:.1e}, rhs {dr:.1e} (< 1e-11)")
    # 2. the identity: l2_mode's l = 0 forcing, converted, = SymForce's terms, untruncated (L = 40, random fields)
    rng = np.random.default_rng(7)
    spB = Space(3, L=40, full=True)
    kB = spB.k

    def rnd(par, top, amp):
        z = (rng.normal(size=kB.size) + 1j * rng.normal(size=kB.size)) * amp
        return z * ((np.abs(kB) <= top) & (kB % 2 == par)) * np.exp(-0.3 * np.abs(kB))
    worst = 0.0
    for trial in range(3):
        Yh, Sh = rnd(1, 3, 1.0), rnd(1, 3, 1.0)
        Cc = rnd(0, 2, 0.1)
        Cc[spB.L] = 1.0
        xi = 0.3 * np.exp(0.4j * (trial + 1))
        g1, g2 = Graph(w), Graph(w)
        a = sym_graph(g1)
        b = converted_graph(g2, Fd["l0"])
        va = eval_point(topo(list(a)), spB, Yh, Sh, Cc, xi)
        vb = eval_point(topo(list(b)), spB, Yh, Sh, Cc, xi)
        for na, nb in zip(a, b):
            ref = np.abs(va[na.id]).max()
            worst = max(worst, np.abs(va[na.id] - vb[nb.id]).max() / ref)
    say("2. l2_mode's l = 0 forcing, converted = FIRST_ORDER's s-translation breaking (untruncated, random data)",
        worst < 1e-9, f"max relative difference {worst:.1e} (< 1e-9)")
    # 3. series = right-hand side for the converted forcings on the M = 3 background, and regularity at xi = 0
    for name in ("l0", "l2", "l2_ctrl", "l2_lam", "G"):
        sc = -1.0 if name == "l2_lam" else 1.0
        gf = GraphForce(1, lambda g, d=Fd[name], sc=sc: converted_graph(g, d, sc))
        jm = JetModel(M, w, Fh_from(x, M), [Dir("W", lam=2.0, force=gf)], [0, 1], {})
        xs = 0.05 * np.exp(1j * (np.pi / 2 + FJ.DELTA) / 2)
        bg = JetModel(M, w, Fh_from(x, M), [], [0], {})
        _, jt, terr = bg.series_at(xs, 400)                # optimal truncation of the background at xs
        gf.prepare(jm, xs, jt + 4)
        D0 = xs * sum(d[0] for d in bg.ser["D"][:jt])
        S0 = xs * xs * sum(d[0] for d in bg.ser["S"][:jt])
        C0 = sum(d[0] for d in bg.ser["G"][:jt])
        fy, fc = gf.point(xs, D0, S0, C0)
        sy = sum(gf.sy[:jt + 1])
        scs = sum(gf.sc[:jt + 1])
        d = max(np.abs(fy - sy).max() / np.abs(fy).max(), np.abs(fc - scs).max() / np.abs(fc).max())
        reg = max(gf.reg)
        nn = Graph(w)
        size = nn.size(list(converted_graph(nn, Fd[name], sc)))
        say(f"3. {name}: summed series = right-hand side at |xi| = 0.05; F(0) = 0; dropped negative powers",
            d < 1e-10 and reg < 1e-12 and gf.dropped < 1e-12,
            f"{d:.1e} (< 1e-10); |F(0)| {reg:.1e}; dropped {gf.dropped:.1e} (< 1e-12); {size} nodes")
    # diagnostic (not a check): the same comparison at four radii. Rounding in the nested divisions by xi^2 of the
    # Euler derivatives grows like a power of 1/|xi|; a mismatch of the series would not fall with |xi|
    for name in ("l2", "l2_ctrl"):
        line = []
        for r in (0.04, 0.05, 0.07, 0.1):
            gf = GraphForce(1, lambda g, d=Fd[name]: converted_graph(g, d, 1.0))
            jm = JetModel(M, w, Fh_from(x, M), [Dir("W", lam=2.0, force=gf)], [0, 1], {})
            xs = r * np.exp(1j * (np.pi / 2 + FJ.DELTA) / 2)
            bg = JetModel(M, w, Fh_from(x, M), [], [0], {})
            _, jt, terr = bg.series_at(xs, 600)
            gf.prepare(jm, xs, jt + 4)
            D0 = xs * sum(dd[0] for dd in bg.ser["D"][:jt])
            S0 = xs * xs * sum(dd[0] for dd in bg.ser["S"][:jt])
            C0 = sum(dd[0] for dd in bg.ser["G"][:jt])
            fy, fc = gf.point(xs, D0, S0, C0)
            sy, scs = sum(gf.sy[:jt + 1]), sum(gf.sc[:jt + 1])
            d = max(np.abs(fy - sy).max() / np.abs(fy).max(), np.abs(fc - scs).max() / np.abs(fc).max())
            line.append(f"|xi| {r:.2f}: {d:.1e} (series truncation {terr:.0e})")
        print(f"diagnostic, {name}: summed series - right-hand side by radius: " + "; ".join(line), flush=True)
    return ok


# ---------------------------------------------------------------- c2 by the slice
def mode_data(x, M):
    """d_s Z_0 at lambda = 2: profile d1 (d0 = 2 d1) and c(0) = 2 g1"""
    bg = bg_series(x, M, 1.0, 3)
    L = bg.L
    d1, g1 = bg.ser["D"][1][0], bg.ser["G"][1][0]
    prof = {int(k): d1[k + L] for k in range(-M, M + 1) if k % 2}
    gk = {int(k): 2 * g1[k + L] for k in range(-(M + 1), M + 2) if k % 2 == 0}
    return prof, gk


def prof_vec(prof, M):
    out = []
    for nu in range(1, M + 1, 2):
        cc = prof.get(nu, 0)
        out += [-2 * cc.imag, 2 * cc.real]
    return np.array(out)


def builders(Fd):
    b = {"sym": lambda g: sym_graph(g)}
    for name in Fd:
        sc = -1.0 if name == "l2_lam" else 1.0
        b[name] = (lambda d, sc: (lambda g: converted_graph(g, d, sc)))(Fd[name], sc)
    return b


def c2_run(x, M, xi1, names, lamcheck=False):
    Fd = load_forcings()
    bld = builders(Fd)
    w = W0 * x[1]
    prof, gk = mode_data(x, M)
    Sch, Cmat, Lp, resH = H.sch_lam(x, M, xi1, 2.0)
    log(f"M {M}: Sch(2) {Sch.tolist()}  singular values {np.linalg.svd(Sch, compute_uv=False)}  (deflated r_H {resH:.1e})")
    basis = R.pert_basis(M)
    nb = len(basis)
    names = list(names) + (["l2_lam"] if lamcheck and "l2_lam" not in names else [])
    e, l = 1, 2
    dirs = [Dir("pert", lam=2.0), Dir("l", dlam=True)]
    masks = [0, e, e | l]
    fb = {}
    for i, name in enumerate(names):
        b = 1 << (2 + i)
        dirs.append(Dir(f"W_{name}", lam=2.0, force=GraphForce(b, bld[name])))
        masks.append(b)
        fb[name] = b

    def run(Cin):
        data = {e: (prof, gk)}
        for j, m in enumerate([e | l] + [fb[nm] for nm in names]):
            wp = {}
            for bb, cc in zip(basis[2:], Cin[j]):
                for k, v in bb.items():
                    wp[k] = wp.get(k, 0) + cc * v
            data[m] = (wp, {})
        jm = JetModel(M, w, Fh_from(x, M), dirs, masks, data)
        Rr, info = R.residuals(jm, x, M, xi1)
        run.last = (Rr[jm.idx[e]], Rr[0], jm)
        return [Rr[jm.idx[m]] for m in [e | l] + [fb[nm] for nm in names]]
    C0 = [np.zeros(nb - 2) for _ in range(1 + len(names))]
    rows, Cin, hist = H.refine(run, C0, Lp, nit=6, label="deflation of e|l and the forced directions")
    Je, r0, jm = run.last
    corr = {"lam": Lp.corr(rows[0])}
    for i, nm in enumerate(names):
        corr[nm] = Lp.corr(rows[1 + i])
    f2 = prof_vec(prof, M)
    U, sv, Vt = np.linalg.svd(Sch)
    ll = U[:, -1]
    phis = {"e1": np.array([1.0, 0.0]), "e2": np.array([0.0, 1.0]), "f2": f2[:2] / np.linalg.norm(f2[:2])}
    SchI = np.linalg.inv(Sch)
    out = dict(M=M, xi1=xi1, x=x.tolist(), Sch2=Sch.tolist(), sv=sv.tolist(), f2=f2.tolist(),
               gk={str(k): [v.real, v.imag] for k, v in gk.items()}, deflation_hist=hist,
               mode_jump=float(np.linalg.norm(Lp.corr(Je))), mode_jump_raw=float(np.linalg.norm(Je)),
               mode_scale=float(np.linalg.norm(Sch) * np.linalg.norm(f2[:2])))
    log(f"   mode d_s Z0: corrected jump {out['mode_jump']:.2e} against |Sch| |f2_1| {out['mode_scale']:.2e}")
    if lamcheck:
        # l2_mode's F_lambda = L0[ln(-t) d_s Z0] in (Phi, c) maps to the solver's ln(-t) d_s Z0 plus the local field
        # dy = -sigma/(xi c) (from Y = 2 Phi_t / C, at T = 1), whose multiplier vanishes; the two responses differ by
        # the homogeneous solution with that field's data, profile -s0/2 (d0 = -s0), c(0) = 0: corr difference
        # = -Sch f_x (harmonic 1). Required: that relation to truncation level, and equal ell projections.
        bg = bg_series(x, M, 1.0, 3)
        s0 = bg.ser["S"][0][0]
        fx = prof_vec({int(k): -s0[k + bg.L] / 2 for k in range(-M, M + 1) if k % 2}, M)
        dlt = corr["l2_lam"] - corr["lam"]
        rel = float(np.linalg.norm(dlt + Sch @ fx[:2]) / np.linalg.norm(corr["lam"]))
        relp = float(abs(ll @ dlt) / abs(ll @ corr["lam"]))
        out.update(lamcheck=rel, lamcheck_raw=float(np.linalg.norm(dlt) / np.linalg.norm(corr["lam"])), lamcheck_ell=relp,
                   corr_l2_lam=corr["l2_lam"].tolist())
        log(f"   F_lambda forcing against the d/dlambda component: raw difference {out['lamcheck_raw']:.2e}; after the "
            f"local field's homogeneous part {rel:.2e}; ell projections differ by {relp:.2e}")
    for nm in names:
        if nm == "l2_lam":
            continue
        val = {"ell": float(-(ll @ corr[nm]) / (ll @ corr["lam"]))}
        for pk, ph in phis.items():
            val[pk] = float(-(ph @ SchI @ corr[nm]) / (ph @ SchI @ corr["lam"]))
        out[nm] = val
        out[f"corr_{nm}"] = corr[nm].tolist()
        lab = "lambda_1" if nm in ("l0", "sym") else "c2"
        log(f"   {nm}: {lab} = {val['ell']:.6f} (left singular vector); phi e1 {val['e1']:.6f}, e2 {val['e2']:.6f}, "
            f"f2 {val['f2']:.6f}")
    out["corr_lam"] = corr["lam"].tolist()
    return out


def main():
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    if "--checks" in opts:
        return checks()
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    tag = f"M{M}_xi{xi1:g}"
    dat = json.load(open(os.path.join(HERE, "data", f"first_order_hier_{tag}.json")))
    x = np.array(dat["x"])
    if "--planted-lam" in opts:
        lam = H.find_lam(x, M, xi1, 2.0)
        good = abs(lam - 1.769122008) < 1e-8
        log(f"planted: lambda* of M = {M} from 2.0: {lam:.10f} against 1.769122008 (|difference| < 1e-8)")
        json.dump(dict(M=M, xi1=xi1, lam=lam, want=1.769122008, good=bool(good)),
                  open(os.path.join(HERE, "data", f"l2_c2_{tag}_planted_lam.json"), "w"), indent=1)
        return good
    names = opts.get("--forcing", "l2,l2_ctrl,l0,sym").split(",")
    out = c2_run(x, M, xi1, names, lamcheck="--lamcheck" in opts)
    fn = os.path.join(HERE, "data", f"l2_c2_{tag}_{'-'.join(names)}{'_' + MUT if MUT else ''}.json")
    json.dump(out, open(fn, "w"), indent=1)
    log(f"wrote {fn}")
    good = True
    if M == 5:                         # the planted lambda_1 (notes/L2_MODE.md 9.5): the l = 0 forcings within 1% of the fit
        for nm in ("l0", "sym"):
            if nm in out:
                rel = abs(out[nm]["ell"] - LAM1_FIT) / LAM1_FIT
                ok = rel <= 0.01
                good &= ok
                log(f"planted lambda_1 (M = 5), {nm}: {out[nm]['ell']:.4f} against {LAM1_FIT}, {100 * rel:.2f}%: "
                    f"{'OK' if ok else 'BAD'} (bar 1%)")
    return good


if __name__ == "__main__":
    good = main()
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
