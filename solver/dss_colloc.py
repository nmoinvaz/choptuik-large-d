#!/usr/bin/env python3
"""Route 1's horizon block as a collocation in u = 1 - x, in the reference solver's framework (dss.py, DSSMulti):
the prototype of notes/QUAD_DOUBLE.md section 13 (13.2 the formulation, 13.4 the tests T1-T4).

The series block (DSSMulti) holds m_h, a_h at the horizon, builds the Taylor series in u, sums it at u0 = 1 - x_last
and matches the last node. DSSColloc instead holds the four fields at Nu Chebyshev-Gauss-Lobatto points u_i on
[0, u0] (u_0 = 0 the horizon, u_{Nu-1} = u0 the last node) and writes the field equations there, with the
u-derivative by the Chebyshev differentiation matrix and the tau-derivative by the FFT as everywhere else:

    (x - c) p_x = -p_tau + c_x p - c p/x + n c (p + m)/(2x)     at every point; at u = 0, where x - c -> 0, this is
                                                               the periodic relation that fixes p at the horizon
    (x + c) m_x = -m_tau - c_x m + c m/x - n c (p + m)/(2x)     at the points u > 0
    a_x = a [(p^2 + m^2) + (n - 1)(1 - a^2)] / (2x),  c_x = c (n - 1)(a^2 - 1) / x   at the points u > 0;  c(0) = 1

with p_x = -p_u. Rows: p at Nu points, m and a at Nu - 1, c at Nu - 1 plus the gauge: 4 H Nu - 2 H per block (H
half-samples in tau), against 4 H Nu unknowns, the same net count, -2H, as the series block (2H unknowns, no rows
of its own). The last point's values are matched to the last node as the series' sum was. The kernel's stored
double-double states (hp/data/states, the raw-data repository) load as DSSMulti states (same layout) and are the
starting points; `from_series` samples their horizon series at the u_i for the collocation values.

    python3 dss_colloc.py STATE.npz [--Nu 32] [--mutate p0sign|nogauge|dut]     T1 on one state (the consistency test)
"""
import json
import sys

import numpy as np

from dss import DSSMulti


def cheb_lobatto(Nu, u0):
    """Chebyshev-Gauss-Lobatto points on [0, u0] (u_0 = 0, u_{Nu-1} = u0) and the differentiation matrix d/du"""
    j = np.arange(Nu)
    t = np.cos(np.pi * j / (Nu - 1))                      # 1 ... -1
    cj = np.where((j == 0) | (j == Nu - 1), 2.0, 1.0) * (-1.0) ** j
    T = np.tile(t, (Nu, 1)).T
    dT = T - T.T
    Dt = np.outer(cj, 1.0 / cj) / (dT + np.eye(Nu))      # Trefethen, Spectral Methods in MATLAB, cheb.m
    Dt = Dt - np.diag(Dt.sum(axis=1))
    u = u0 / 2 * (1 - t)                                  # u increases with j: u_0 = 0, u_{Nu-1} = u0
    Du = Dt * (-2.0 / u0)                                 # du = -(u0/2) dt
    return u, Du


class DSSColloc(DSSMulti):
    def __init__(self, nodes, Nu=32, mutate=None, **kw):
        kw["u0"] = 1 - nodes[-1]
        super().__init__(nodes=nodes, **kw)
        self.Nu = Nu
        self.chunk = 512                      # columns per finite-difference batch of the block's Jacobian
        self.u, self.Du = cheb_lobatto(Nu, self.u0)
        self.mutate = mutate
        if mutate == "dut":
            self.Du = self.Du.T.copy()

    # ---------- layout ----------
    def layout(self):
        H = self.N // 2
        K = len(self.nodes)
        return H, K, 2 * H + 4 * K * H + 4 * H * self.Nu + 1

    def offsets(self):
        H, K, L = self.layout()
        return dict(c=0, nodes=2 * H, coll=2 * H + 4 * H * K, D=L - 1)

    def unpack_colloc(self, Zb):
        H, K, L = self.layout()
        o = 0
        p1 = self.odd(Zb[:, o:o + H]); o += H
        c0 = self.even(Zb[:, o:o + H]); o += H
        states = []
        for i in range(K):
            st = np.stack([self.odd(Zb[:, o:o + H]), self.odd(Zb[:, o + H:o + 2 * H]),
                           self.even(Zb[:, o + 2 * H:o + 3 * H]), self.even(Zb[:, o + 3 * H:o + 4 * H])])
            states.append(st); o += 4 * H
        coll = self.unpack_coll_block(Zb[:, o:o + 4 * H * self.Nu]); o += 4 * H * self.Nu
        omega = 2 * np.pi / Zb[:, o]
        return p1, c0, states, coll, omega

    def unpack_coll_block(self, X):
        """X: (B, 4 H Nu), point-major (p, m, a, c half-samples at u_0, then u_1, ...) -> (Nu, 4, B, N) full arrays"""
        H = self.N // 2
        out = np.empty((self.Nu, 4, X.shape[0], self.N))
        for i in range(self.Nu):
            o = 4 * H * i
            out[i, 0] = self.odd(X[:, o:o + H]); out[i, 1] = self.odd(X[:, o + H:o + 2 * H])
            out[i, 2] = self.even(X[:, o + 2 * H:o + 3 * H]); out[i, 3] = self.even(X[:, o + 3 * H:o + 4 * H])
        return out

    def pack_coll(self, vals):
        """(Nu, 4, N) full arrays -> the 4 H Nu half-sample vector"""
        H = self.N // 2
        return np.concatenate([vals[i, q, :H] for i in range(self.Nu) for q in range(4)])

    # ---------- the block ----------
    def coll_rows(self, coll, omega):
        """the collocation block's rows for coll (Nu, 4, B, N): (B, 4 H Nu - 2 H)"""
        H = self.N // 2
        n = self.n
        Fu = np.einsum("ij,jqbn->iqbn", self.Du, coll)          # d/du of each field at each point
        rows = []
        for i in range(self.Nu):
            x = 1.0 - self.u[i]
            p, m, a, c = coll[i]
            pu, mu, au, cu = Fu[i]
            a2 = a * a
            cx = c * (n - 1) * (a2 - 1) / x
            pc = n * c * (p + m) / (2 * x)
            if self.mutate == "p0sign" and i == 0:
                pc = -pc
            # (x - c) p_x - [...] with p_x = -p_u
            rp = -(x - c) * pu - (-self.D(p, omega) + cx * p - c * p / x + pc)
            rows.append(rp[:, :H])
            if i == 0:
                if self.mutate != "nogauge":
                    rows.append((c - 1.0)[:, :H])
                else:
                    rows.append(np.zeros_like(c[:, :H]))
                continue
            rm = -(x + c) * mu - (-self.D(m, omega) - cx * m + c * m / x - pc)
            ra = -au - a * ((p * p + m * m) + (n - 1) * (1 - a2)) / (2 * x)
            rc = -cu - cx
            rows += [rm[:, :H], ra[:, :H], rc[:, :H]]
        return np.concatenate(rows, axis=1)

    def residual(self, Zb):
        H, K, L = self.layout()
        p1, c0, states, coll, omega = self.unpack_colloc(Zb)
        blocks = []
        P, M, A, C = self.centre_series(p1, c0, omega)
        s = np.stack([self.eval_series(S, self.x0) for S in (P, M, A, C)])
        if self.x0 < self.nodes[0]:
            s = self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, omega)
        blocks.append(s - states[0])
        for i in range(K - 1):
            blocks.append(self.seg(states[i], self.nodes[i], self.nodes[i + 1], omega) - states[i + 1])
        blocks.append(coll[self.Nu - 1] - states[-1])            # the last point is the last node
        mis = np.concatenate([np.concatenate([b[0, :, :H], b[1, :, :H], b[2, :, :H], b[3, :, :H]], axis=1) for b in blocks], axis=1)
        phase = np.fft.rfft(p1, axis=-1)[:, 1].imag / self.N
        return np.concatenate([mis, self.coll_rows(coll, omega), phase[:, None]], axis=1)

    # ---------- Jacobian, block-structured ----------
    def jacobian(self, z, eps=1e-7):
        H, K, L = self.layout()
        N, Nu = self.N, self.Nu
        o = self.offsets()
        F0 = self.residual(z[None, :])[0]
        J = np.zeros((L, L))
        D = z[-1]; hD = eps * max(1.0, abs(D))
        blk = 4 * H

        def to_rows(s, ref):
            d = s - ref
            return np.concatenate([d[0, :, :H], d[1, :, :H], d[2, :, :H], d[3, :, :H]], axis=1)

        def batch_from(base_vec, nfree):
            B = 1 + nfree + 1
            X = np.repeat(base_vec[None, :], B, axis=0)
            h = eps * np.maximum(1.0, np.abs(base_vec))
            X[1:1 + nfree] += np.diag(h)
            om = np.full(B, 2 * np.pi / D); om[-1] = 2 * np.pi / (D + hD)
            return X, h, om

        # centre block
        X, h, om = batch_from(z[o["c"]:o["c"] + 2 * H], 2 * H)
        P, M, A, C = self.centre_series(self.odd(X[:, :H]), self.even(X[:, H:]), om)
        s = np.stack([self.eval_series(S_, self.x0) for S_ in (P, M, A, C)])
        if self.x0 < self.nodes[0]:
            s = self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, om)
        R = to_rows(s, s[:, 0:1])
        J[0:blk, o["c"]:o["c"] + 2 * H] = (R[1:1 + 2 * H] / h[:, None]).T
        J[0:blk, o["D"]] += R[-1] / hD
        J[0:blk, o["nodes"]:o["nodes"] + 4 * H] -= np.eye(4 * H)
        # interior segments
        for i in range(K - 1):
            oi = o["nodes"] + 4 * H * i
            X, h, om = batch_from(z[oi:oi + 4 * H], 4 * H)
            st = np.stack([self.odd(X[:, :H]), self.odd(X[:, H:2 * H]), self.even(X[:, 2 * H:3 * H]), self.even(X[:, 3 * H:])])
            s = self.seg(st, self.nodes[i], self.nodes[i + 1], om)
            R = to_rows(s, s[:, 0:1])
            r0 = blk * (i + 1)
            J[r0:r0 + blk, oi:oi + 4 * H] = (R[1:1 + 4 * H] / h[:, None]).T
            J[r0:r0 + blk, o["D"]] += R[-1] / hD
            J[r0:r0 + blk, oi + 4 * H:oi + 8 * H] -= np.eye(4 * H)
        # the last point against the last node: linear
        r0 = blk * K
        ol = o["coll"] + 4 * H * (Nu - 1)
        J[r0:r0 + blk, ol:ol + 4 * H] = np.eye(4 * H)
        J[r0:r0 + blk, o["nodes"] + 4 * H * (K - 1):o["nodes"] + 4 * H * K] -= np.eye(4 * H)
        # the collocation block: finite differences over its own unknowns and Delta, in chunks of columns
        nc = 4 * H * Nu
        base = z[o["coll"]:o["coll"] + nc]
        h = eps * np.maximum(1.0, np.abs(base))
        r0 = blk * (K + 1)
        om0 = 2 * np.pi / D
        R0 = self.coll_rows(self.unpack_coll_block(base[None, :]), np.array([om0]))[0]
        nr = len(R0)
        for a in range(0, nc, self.chunk):
            b = min(nc, a + self.chunk)
            X = np.repeat(base[None, :], b - a, axis=0)
            X[np.arange(b - a), np.arange(a, b)] += h[a:b]
            R = self.coll_rows(self.unpack_coll_block(X), np.full(b - a, om0))
            J[r0:r0 + nr, o["coll"] + a:o["coll"] + b] = ((R - R0) / h[a:b, None]).T
        RD = self.coll_rows(self.unpack_coll_block(base[None, :]), np.array([2 * np.pi / (D + hD)]))[0]
        J[r0:r0 + nr, o["D"]] = (RD - R0) / hD
        # phase row
        E = np.eye(H)
        J[-1, o["c"]:o["c"] + H] = np.fft.rfft(self.odd(E), axis=-1)[:, 1].imag / N
        return F0, J

    # ---------- starting values from a series-block state ----------
    def from_series(self, z_ms, s_ms):
        """the DSSColloc state whose collocation values are s_ms's horizon series (from the DSSMulti state z_ms)
        sampled at the u_i; everything else copied"""
        H, K, L = self.layout()
        p1, c0, states, mS, aS, omega = s_ms.unpack_ms(z_ms[None, :])
        P, M, A, C = s_ms.ssh_series(mS, aS, omega)
        vals = np.empty((self.Nu, 4, self.N))
        for i in range(self.Nu):
            for q, S in enumerate((P, M, A, C)):
                vals[i, q] = self.eval_series(S, self.u[i])[0]
        o = self.offsets()
        z = np.empty(L)
        z[:o["coll"]] = z_ms[:o["coll"]]
        z[o["coll"]:o["coll"] + 4 * H * self.Nu] = self.pack_coll(vals)
        z[-1] = z_ms[-1]
        return z


def load_state(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d["zh"], m


def solvers(meta, Nu=32, Ks=None, mutate=None, nodes=None):
    """(DSSMulti, DSSColloc) for a stored state's layout: the series summed at the first node (x0 = nodes[0]), the
    horizon series to Ks orders, summed at the last node (u0 = 1 - x_last)"""
    D, M = meta["D"], meta["M"]
    nodes = list(meta["nodes"]) if nodes is None else list(nodes)
    ks = Ks or min(meta.get("ks", 130), 120)
    common = dict(n=D - 2, N=2 * M, Kc=120, Ks=ks, x0=nodes[0], log_steps=400)
    sm = DSSMulti(nodes=nodes, u0=1 - nodes[-1], **common)
    sc = DSSColloc(nodes=nodes, Nu=Nu, mutate=mutate, **common)
    return sm, sc


def main():
    args = list(sys.argv[1:])
    Nu, mutate = 32, None
    if "--Nu" in args:
        i = args.index("--Nu"); Nu = int(args[i + 1]); del args[i:i + 2]
    if "--mutate" in args:
        i = args.index("--mutate"); mutate = args[i + 1]; del args[i:i + 2]
    zh, meta = load_state(args[0])
    sm, sc = solvers(meta, Nu=Nu, mutate=mutate)
    H, K, L = sc.layout()
    Fm = sm.residual(zh[None, :])[0]
    zc = sc.from_series(zh, sm)
    Fc = sc.residual(zc[None, :])[0]
    blk = 4 * H
    print(f"D {meta['D']:.4f}, M {meta['M']}, nodes {meta['nodes']}, Nu {Nu}, mutate {mutate}: unknowns series {len(zh)}, collocation {L}")
    print(f"  series block state: |F| {np.linalg.norm(Fm):.2e}, horizon rows max {np.abs(Fm[blk * K:blk * (K + 1)]).max():.2e}")
    cr = Fc[blk * (K + 1):-1]
    print(f"  collocation state from the series' samples: |F| {np.linalg.norm(Fc):.2e}; matching rows max "
          f"{np.abs(Fc[:blk * (K + 1)]).max():.2e}; collocation rows max {np.abs(cr).max():.2e} (p rows at u = 0: "
          f"{np.abs(cr[:H]).max():.2e}, gauge: {np.abs(cr[H:2 * H]).max():.2e})")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- double-double samples from the kernel's series
def kernel_samples(path, u, threads=2):
    """the horizon series of the kernel (double-double, the stored state's own) summed at the points u in 40-digit
    arithmetic, rounded to f64: (len(u), 4, M) full-period values at the state's M tau points"""
    from decimal import Decimal, getcontext
    from hpkernel.driver import load, make_solver
    getcontext().prec = 40
    zh, zl, m = load(path)
    D, M, nodes = m["D"], m["M"], list(m["nodes"])
    kf = 13 if D >= 35 else None
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    hh, hl = s.series(zh, zl, 1, True)
    sig = Decimal(s.series_scale(1))
    nk = len(hh) // (4 * M)
    ch = np.asarray(hh).reshape(nk, 4, M); cl = np.asarray(hl).reshape(nk, 4, M)
    out = np.empty((len(u), 4, M))
    for q in range(4):
        for t in range(M):
            coef = [Decimal(float(ch[k, q, t])) + Decimal(float(cl[k, q, t])) for k in range(nk)]   # c_k sigma^k
            for i, ui in enumerate(u):
                r = Decimal(float(ui)) / sig
                acc = Decimal(0)
                for k in range(nk - 1, -1, -1):
                    acc = acc * r + coef[k]
                out[i, q, t] = float(acc)
    return out


def colloc_state_from_kernel(path, sc, z_ms, threads=2):
    """the DSSColloc state with the kernel's double-double series sampled at sc.u (f64-rounded); the rest from z_ms"""
    H, K, L = sc.layout()
    vals = kernel_samples(path, sc.u, threads=threads)
    full = np.empty((sc.Nu, 4, sc.N))
    for q, ext in enumerate((sc.odd, sc.odd, sc.even, sc.even)):
        full[:, q] = ext(vals[:, q, :H])
    o = sc.offsets()
    z = np.empty(L)
    z[:o["coll"]] = z_ms[:o["coll"]]
    z[o["coll"]:o["coll"] + 4 * H * sc.Nu] = sc.pack_coll(full)
    z[-1] = z_ms[-1]
    return z
