#!/usr/bin/env python3
"""The strong-row term S of the reduced cubic, and the reduced map on the strong-solved manifold (#528).

    cd hp && python3 reduced_cubic.py run RAWDIR DZ0 BASE [--k=4] [--its=16] [--threads=2] [--json=OUT] [--mutate=M]
    cd hp && python3 reduced_cubic.py planted [--seed=5] [--json=OUT] [--mutate=M]

`run` rebuilds a corrector attempt's pinned system as hp/weak_corrector.py does (DZ0 is anm_run.py --save_dz0's file,
BASE the state it stepped from, both relative to RAWDIR; the held offset c = F_bare(z0) + J dz0 on the horizon block's
p rows, J dz0 summed by columns in eight parts). `planted` builds a synthetic system with a known reduced map. Both go
through the same functions below; only the residual F and the solves with J = F'(z0) differ.

Notation (#499, #527). z0 is the start, X = V the k weakest right singular vectors of J (near_null.subspace, refined
by three steps of block inverse iteration at 140 digits, as dz0_weak.py), U = orth(J^-T V) with J^-T V = U R, so
B = U^T J V = R^-T. Then U^T J w = 0 for every w orthogonal to V, and J^-1 maps the complement of U onto that of V:
Kuznetsov's bordered solve A^INV P_X is J^-1 (I - U U^T) here. a = V^T dz0 (Newton's first move).

The strong solve. At fixed c, w(c), orthogonal to V, solves (I - U U^T) F(z0 + V c + w) = 0 by the chord iteration
w <- w - J^-1 (I - U U^T) F with z0's LU (each step stays orthogonal to V), from w(0) = w0. The reduced map is
g(c) = U^T F(z0 + V c + w(c)) (Keller 1987, Method III; Decker, Keller and Kelley 1983, Thm 2.6).

The tensors at z0, by differences of F at two steps each (the second a tenth of the first):
  Q_mjl = u_m . F''(v_j, v_l), central differences at h_Q;
  T_mijl = u_m . F'''(v_i, v_j, v_l), antisymmetric four-point differences and polarization at h_T (dz0_weak.cubic);
  w_jl = J^-1 (I - U U^T) F''(v_j, v_l), central differences at h_F, h_F / 10 and h_F / 100;
  S_m,i,jl = u_m . F''(v_i, w_jl), mixed central differences with the step h_S along v_i and along w_jl / |w_jl|,
      symmetrized, S_mijl = (S_m,i,jl + S_m,j,il + S_m,l,ij) / 3;
  L_mj = u_m . F''(v_j, w0) and q00 = U^T F''(w0, w0), the coupling through the start's own strong correction w0.
With w(c) = w0 - 1/2 sum_jl w_jl c_j c_l + ..., the reduced map is
  g(c) = g(0) + (B + L) c + 1/2 Q(c, c) + 1/6 (T - 3 S)(c, c, c) + ...
(Kuznetsov 1999, sec. 5, c = (1/6) <p, C(q,q,q) - 3 B(q, A^INV B(q,q))>, for one null vector; here four). The
map's own linear and quadratic coefficients, B_LS and Q_LS, come from central differences of g at h_LS, each point a
strong solve.

Models, each g_model(c) = g(0) + B' c + 1/2 Q'(c, c) + 1/6 C'(c, c, c): "T" (#499: B, Q = 0, T, on z0 + V c, its
g(0) = -B a), "T,Q" (adds Q), "T-3S" (B, Q, T - 3S), "T-3S,L" (B + L, Q, T - 3S), and "T-3S,LS" (B_LS, Q_LS, T - 3S).
Their real roots are found by Newton at 140 digits from c = a, 0 and random starts; at each root g (the reduced map,
with the strong solve) is evaluated, and the root with the smallest |g| is the model's c**. c* is the root of "T"
whose affine weak residual U^T F(z0 + V c) is smallest (#499's root, recomputed here).

Controls at c*:
  - S control (judged): dw = w(c*) - w0 replaced by 2 dw changes g by -1/2 S(c*, c*, c*), to 1e-2 of it;
  - w control (diagnostic): w(c*) replaced by 2 w(c*) changes g by L c* - 1/2 S(c*, c*, c*) + 3/2 q00.

Checks of both modes (exit 1 on failure): w_jl at h_F / 10 against h_F / 100 to 1e-2 (2-norm, the worst pair; w_jl is
taken from h_F / 100, and h_F against h_F / 10 is printed for the h^2 pattern); S at h_S
against h_S / 10 to 1e-2 of max|S| (or, on real data, reported as bounded below its truncation when the two differ by
the factor 100 of an h^2 term); the S control. `planted` also
judges the recovery of every tensor (rotated into the plant's basis) to 1e-6 of its largest entry, and that the model
"T-3S,L" has a root within 1e-3 relative of the planted root with |g| < 1e-2 |B a| there, and that the plant shows
(|g(c*)| >= 0.1 |B a| on the strong-solved manifold). Mutations: --mutate=w0 skips the strong solve (w(c) = 0, the
map on z0 + V c), and the root check and the S control must fail; --mutate=Ssign flips S's sign in the models and in
the control's prediction, and both must fail; --mutate=Sh steps 10 h along w_jl / |w_jl| at h_S / 10 but divides by
that h squared, so that S(h_S / 10) is ten times too large, and the h check of S must fail; --mutate=hF forms w_jl at
h_F = 1e-24, where its truncation dominates, and the h check of w_jl must fail. notes/REDUCED_CUBIC.md has the
protocol and the results.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.mp.dps = 140

# the first step of each difference; the second is /10 (w_jl: /10 and /100, judged on the last two). h_F is small
# because J^-1 (I - U U^T) multiplies the second difference's h^2 truncation along the weakest strong directions
# (sigma_5 ~ 1e-19) by up to 1e19: at the refused D 242.5 start the truncation was all of w_jl at h_F = 1e-24 (0.99
# between h and h/10) and 1.4e-2 of it at 1e-32
H = dict(Q=1e-32, T=1e-23, F=1e-32, S=1e-32, LS=1e-33)
H_MUT_F = 1e-24                                             # --mutate=hF: w_jl from that step, whose h check must fail


def mgs(W):
    """modified Gram-Schmidt of the mpf vectors W in order: (Q, R) with W_j = sum_i R_ij Q_i (weak_curv.mgs)"""
    Q, k = [], len(W)
    R = [[mp.mpf(0)] * k for _ in range(k)]
    for j, w in enumerate(W):
        v = list(w)
        for i, q in enumerate(Q):
            R[i][j] = mp.fsum(a * b for a, b in zip(q, v))
            v = [a - R[i][j] * b for a, b in zip(v, q)]
        R[j][j] = mp.sqrt(mp.fsum(a * a for a in v))
        Q.append([a / R[j][j] for a in v])
    return Q, R


def dot(x, y):
    return mp.fsum(a * b for a, b in zip(x, y))


def nrm(x):
    return mp.sqrt(mp.fsum(a * a for a in x))


def amax(x):
    return max(abs(a) for a in x)


def lin(terms):
    """sum of coefficient * vector over terms, at 140 digits"""
    n = len(terms[0][1])
    return [mp.fsum(c * v[r] for c, v in terms) for r in range(n)]


def subspace64(solve64, L, k, its, seed=1):
    """near_null.subspace: the k weakest right singular vectors by subspace iteration on (J^T J)^-1 in f64"""
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((L, k)))[0]
    hist = []
    for _ in range(its):
        Y = np.column_stack([solve64(solve64(V[:, i], True)) for i in range(k)])
        hist.append(1 / np.sqrt(np.linalg.norm(Y, axis=0)))
        V = np.linalg.qr(Y)[0]
    return V, hist


# ------------------------------------------------------------------ the systems
class Real:
    """an attempt's pinned system, rebuilt as weak_corrector.py does; F(p) is the pinned residual at z0 + p (p an mpf
    list or None), solve(r) is J^-1 r through z0's octo-double LU"""

    def __init__(self, raw, dzf, base, threads, log):
        from hpkernel.driver import make_solver
        from multidouble import od_axpy
        from octo_corrector import PinnedO, load_o, o8, omax
        from freeze_round import mpv, parts8
        self.mpv, self.parts8, self.od_axpy, self.o8 = mpv, parts8, od_axpy, o8
        f = np.load(os.path.join(raw, dzf))
        z0 = o8([f[f"z{q}"] for q in range(8)])
        d0 = o8([f[f"d{q}"] for q in range(8)])
        md = json.loads(str(f["meta"]))
        _, mb = load_o(os.path.join(raw, base))
        M, nodes, kf = int(mb["M"]), list(mb["nodes"]), int(mb.get("kfreeze", 3))
        K, L = len(nodes), len(z0[0])
        # the attempt's D as anm_run.py forms it, the base's D (its string) plus a, exact (dz0_weak.py)
        D = mp.mpf(mb["D_str"]) + mp.mpf(md["a"])
        if abs(D - mp.mpf(md["D"])) > mp.mpf("1e-9"):
            sys.exit(f"{dzf}: the base's D plus a, {mp.nstr(D, 15)}, is not the file's D {md['D']}")
        t0 = time.time()
        s = make_solver(str(D), M, nodes=nodes, kt=mb.get("kt", 36), kc=mb.get("kc", 146), ks=mb.get("ks", 130),
                        threads=threads, kfreeze=kf)
        s.set_segs(mb["segs"])
        # the held offset c = F_bare(z0) + J dz0 on the horizon block's p rows; J dz0 by columns in eight parts
        J = [np.asarray(p) for p in s.jacobian_o(list(z0))]
        JT = [np.ascontiguousarray(p.T) for p in J]
        zero = np.zeros(L)
        acc = [zero] * 8
        for col in range(L):
            for q in range(8):
                x = float(d0[q][col])
                if x != 0.0:
                    acc = o8(od_axpy(x, *[JT[p][col] for p in range(8)], *acc))
        coff = mpv(o8(od_axpy(1.0, *acc, *o8(s.residual_o(list(z0))))), range(L))
        hp = set(range(4 * M * K, 4 * M * K + M))
        on, off = max(abs(coff[r]) for r in hp), max(abs(coff[r]) for r in range(L) if r not in hp)
        self.offset_ok = bool(on > 0 and off < mp.mpf("1e-6") * on)
        c8 = parts8([coff[r] if r in hp else mp.mpf(0) for r in range(L)])
        self.so = PinnedO(s, c8[0], c8[1], c8=c8)
        self.s, self.z0, self.L = s, z0, L
        st = s.newton_o(list(z0))                     # z0's LU, kept for every solve below
        self.F0max = omax(self.so.residual_o(z0))
        self.dz0 = mpv(d0, range(L))
        self.label = f"{dzf}: D {mp.nstr(D, 12)} (a {md.get('a')}), from {base}"
        log(f"{self.label}; L {L}; held offset on its rows {mp.nstr(on, 3)}, elsewhere {mp.nstr(off, 3)} (bound 1e-6 "
            f"of the first): {'pass' if self.offset_ok else 'FAIL'}; the recovered system's start max|F| "
            f"{self.F0max:.4e}; octo-double LU pivots {st[3]:.1e}..{st[4]:.1e} ({time.time() - t0:.0f} s)")
        self.meta = dict(dz0=dzf, base=base, D=float(D), D_str=mp.nstr(D, 30), a_step=md.get("a"), L=L,
                         offset=dict(on=float(on), off=float(off), ok=self.offset_ok), F0max=self.F0max,
                         pivots=[st[3], st[4]])

    def F(self, p=None):
        z = self.z0 if p is None else self.o8(self.od_axpy(1.0, *self.parts8(p), *self.z0))
        return self.mpv(self.so.residual_o(z), range(self.L))

    def solve(self, r, transpose=False):
        return self.mpv(self.o8(self.s.olu_solve(list(self.parts8(r)), transpose)), range(self.L))

    def solve64(self, b, transpose=False):
        x = self.s.olu_solve([np.ascontiguousarray(b)] + [np.zeros(self.L)] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])


def sym_cubic(rng, k, scale, rank=6):
    """a symmetric trilinear map R^k^3 -> R^k as a sum of rank-one cubes, [m][i][j][l]"""
    t = rng.standard_normal((k, rank))
    b = rng.standard_normal((rank, k))
    return [[[[mp.mpf(scale) * mp.fsum(mp.mpf(t[m, r]) * mp.mpf(b[r, i]) * mp.mpf(b[r, j]) * mp.mpf(b[r, l])
                                       for r in range(rank)) for l in range(k)] for j in range(k)] for i in range(k)]
            for m in range(k)]


class Planted:
    """a synthetic system with a known reduced map. Canonical unknowns (y in R^k, x in R^m), rotated on both sides by
    random orthogonal P_L, P_R at 140 digits: F(z) = P_L G(P_R^T z), z0 = 0, with
        G_w = beta + Sigma y + 1/2 Q(y, y) + 1/6 T(y, y, y) + 1/24 P4(y, y, y, y) + K(y, x)
        G_s = gamma + (Lambda + M(y)) x + 1/2 C(y, y) + 1/24 c4 (b4 . y)^4 e_0.
    J = P_L diag(Sigma, Lambda) P_R^T at z0. The strong rows are linear in x at fixed y, so
    x(y) = -(Lambda + M(y))^-1 (gamma + 1/2 C(y, y)) and the reduced map is G_w(y, x(y)) exactly. Its truth: B = Sigma,
    w0 = -Lambda^-1 gamma, L = K(., w0), Q, T, S_m,i,jl = K_m(e_i, Lambda^-1 C(e_j, e_l)), q00 = 0. beta is set so that
    G_w(y*, x(y*)) = 0 at a chosen y* (|y*| 4.5e-30), the planted root. The scales are the D 242.5 starts' (#499,
    #529): sigma 2.1e-27..7.1e-26, |T| ~ 1.6e33, |S| ~ |T| / 3, |L| ~ |B|, |Q| ~ 1e3, strong singular values 1e-19..3e34."""

    def __init__(self, seed, log, k=4, m=8):
        rng = np.random.default_rng(seed)
        n = k + m
        self.L, self.k, self.m = n, k, m
        rnd = lambda: [mp.mpf(float(x)) for x in rng.standard_normal(n)]   # noqa: E731
        self.PL = mgs([rnd() for _ in range(n)])[0]     # columns
        self.PR = mgs([rnd() for _ in range(n)])[0]
        self.sig = [mp.mpf(x) for x in ("2.1e-27", "2.2e-27", "3.2e-26", "7.1e-26")]
        self.lam = [mp.mpf(x) for x in ("1e-19", "1e-10", "1e-3", "1", "1e20", "1e25", "1e30", "3e34")]
        self.Q = [[[mp.mpf(0)] * k for _ in range(k)] for _ in range(k)]
        qr = rng.standard_normal((k, k, k))
        for mm in range(k):
            for i in range(k):
                for j in range(k):
                    self.Q[mm][i][j] = mp.mpf(1e3) * mp.mpf((qr[mm, i, j] + qr[mm, j, i]) / 2)
        self.T = sym_cubic(rng, k, 1.6e33 / 3)
        a4 = rng.standard_normal((k, k))
        self.P4 = [(mp.mpf(2e58) * mp.mpf(float(rng.standard_normal())), [mp.mpf(float(x)) for x in a4[mm]])
                   for mm in range(k)]                  # P4_m(y^4) = p_m (a_m . y)^4, ~1e-4 |B a| at the root
        # the start's strong residual lives in rows 0-2, where w0 ~ 4e-33 per component
        w0c = [mp.mpf(float(x)) * mp.mpf("4e-33") for x in rng.uniform(0.5, 1.0, 3) * rng.choice([-1, 1], 3)] + \
              [mp.mpf(0)] * (m - 3)
        self.gam = [-self.lam[s] * w0c[s] for s in range(m)]
        # C(y, y) in the strong rows 4-7: Lambda^-1 C ~ 3e3 in rows 4-6, C ~ 1e33 in row 7
        self.C = []
        for s in range(m):
            cr = rng.standard_normal((k, k))
            scale = 0 if s < 4 else (min(self.lam[s] * 3e3, mp.mpf(1e33)))
            self.C.append([[mp.mpf(scale) * mp.mpf((cr[i, j] + cr[j, i]) / 2) for j in range(k)] for i in range(k)])
        # a quartic in the weakest strong row, 1/24 c4 (b4 . y)^4 with c4 1e43: J^-1 multiplies its share of the
        # second differences' truncation by 1e19, so w_jl from h_F = 1e-24 is that truncation (as at the D 242.5 starts)
        # while h_F = 1e-32 resolves w_jl (--mutate=hF)
        self.c4 = (mp.mpf(1e43), [mp.mpf(float(x)) for x in rng.standard_normal(k)])
        # K(y, x): 1.5e6 on rows 0-3 of x (L ~ |B|), 1e29 on rows 4-6 (S ~ |T| / 3), 1e30 on row 7
        kr = rng.standard_normal((k, k, m))
        ks = [1.5e6] * 4 + [1e29] * 3 + [1e30]
        self.K = [[[mp.mpf(ks[s]) * mp.mpf(kr[mm, j, s]) for s in range(m)] for j in range(k)] for mm in range(k)]
        # M(y) x in the strong rows, within the blocks 0-3 and 4-7 (so that M(y) w0 stays out of K's heavy rows):
        # Lambda^-1 M(y) ~ 1e25 |y|, a chord contraction ~ 5e-5 at |y| 5e-30
        mr = rng.standard_normal((m, k, m))
        self.Mt = [[[self.lam[s] * mp.mpf(1e25) * mp.mpf(mr[s, j, t]) / 4 if (s < 4) == (t < 4) else mp.mpf(0)
                     for t in range(m)] for j in range(k)] for s in range(m)]
        # the planted root y*: of 20 random directions at |y*| 4.5e-30, the one whose root is best conditioned,
        # kappa = sigma_min(g'(y*)) |y*| / |beta| largest (a root in a cluster of near roots tests their conditioning,
        # not the recovery); beta is then set so that g(y*) = 0
        best = None
        for _ in range(20):
            yr = rng.standard_normal(k)
            y = [mp.mpf(float(x)) * mp.mpf("4.5e-30") / mp.mpf(float(np.linalg.norm(yr))) for x in yr]
            self.beta = [mp.mpf(0)] * k
            beta = [-x for x in self.Gw(y, self.xsolve(y))]
            hh = mp.mpf("1e-36")
            Jg = mp.matrix(k, k)
            for j in range(k):
                yp, ym = list(y), list(y)
                yp[j] += hh
                ym[j] -= hh
                gp, gm = self.Gw(yp, self.xsolve(yp)), self.Gw(ym, self.xsolve(ym))
                for mm in range(k):
                    Jg[mm, j] = (gp[mm] - gm[mm]) / (2 * hh)
            kap = min(mp.svd_r(Jg, compute_uv=False)) * nrm(y) / nrm(beta)
            if best is None or kap > best[0]:
                best = (kap, y, beta)
        self.kappa, self.yroot, self.beta = best
        self.w0c = w0c
        self.F0max = float(amax(self.F()))
        self.dz0 = [-x for x in self.solve(self.F())]
        self.label = f"planted (seed {seed}): n {n}, k {k}"
        self.meta = dict(seed=seed, n=n, k=k, F0max=self.F0max, yroot=[float(x) for x in self.yroot],
                         kappa=float(self.kappa))
        log(f"{self.label}; start max|F| {self.F0max:.3e}, |w0| {mp.nstr(nrm(w0c), 3)}, planted root |y*| "
            f"{mp.nstr(nrm(self.yroot), 3)}, its conditioning sigma_min(g') |y*| / |B a| {mp.nstr(self.kappa, 3)}")

    def xsolve(self, y):
        """x(y), the strong rows' solution at fixed y"""
        m, k = self.m, self.k
        A = mp.matrix(m, m)
        for s in range(m):
            for t in range(m):
                A[s, t] = (self.lam[s] if s == t else 0) + mp.fsum(self.Mt[s][j][t] * y[j] for j in range(k))
        rhs = mp.matrix([-(self.gam[s] + mp.fsum(self.C[s][i][j] * y[i] * y[j] for i in range(k) for j in range(k)) / 2
                           + self.quart(s, y)) for s in range(m)])
        x = mp.lu_solve(A, rhs)
        return [x[s] for s in range(m)]

    def quart(self, s, y):
        """the strong row's quartic term, in row 0 only"""
        return self.c4[0] * dot(self.c4[1], y) ** 4 / 24 if s == 0 else mp.mpf(0)

    def Gw(self, y, x):
        k, m = self.k, self.m
        out = []
        for mm in range(k):
            v = self.beta[mm] + self.sig[mm] * y[mm]
            v += mp.fsum(self.Q[mm][i][j] * y[i] * y[j] for i in range(k) for j in range(k)) / 2
            v += mp.fsum(self.T[mm][i][j][l] * y[i] * y[j] * y[l] for i in range(k) for j in range(k)
                         for l in range(k)) / 6
            p, av = self.P4[mm]
            v += p * dot(av, y) ** 4 / 24
            v += mp.fsum(self.K[mm][j][s] * y[j] * x[s] for j in range(k) for s in range(m))
            out.append(v)
        return out

    def Gs(self, y, x):
        k, m = self.k, self.m
        return [self.gam[s] + self.lam[s] * x[s] + mp.fsum(self.Mt[s][j][t] * y[j] * x[t] for j in range(k)
                                                             for t in range(m))
                + mp.fsum(self.C[s][i][j] * y[i] * y[j] for i in range(k) for j in range(k)) / 2 + self.quart(s, y)
                for s in range(m)]

    def F(self, p=None):
        n, k = self.L, self.k
        zeta = [mp.mpf(0)] * n if p is None else [dot(self.PR[i], p) for i in range(n)]
        y, x = zeta[:k], zeta[k:]
        G = self.Gw(y, x) + self.Gs(y, x)
        return [mp.fsum(G[i] * self.PL[i][r] for i in range(n)) for r in range(n)]

    def solve(self, r, transpose=False):
        """J^-1 r = P_R D^-1 P_L^T r, J^-T r = P_L D^-1 P_R^T r, D = diag(Sigma, Lambda)"""
        n = self.L
        d = self.sig + self.lam
        A, Bm = (self.PR, self.PL) if transpose else (self.PL, self.PR)
        u = [dot(A[i], r) / d[i] for i in range(n)]
        return [mp.fsum(u[i] * Bm[i][rr] for i in range(n)) for rr in range(n)]

    def solve64(self, b, transpose=False):
        return np.array([float(x) for x in self.solve([mp.mpf(float(v)) for v in b], transpose)])

    def truth(self, X, Um):
        """the planted tensors in the code's bases: with X = V_true O and U = U_true O', B_X = O'^T B O, and each
        tensor's weak-row index rotated by O', its directions by O"""
        k = self.k
        O = [[dot(self.PR[i], X[j]) for j in range(k)] for i in range(k)]      # O_ij = v_true_i . x_j
        Op = [[dot(self.PL[i], Um[j]) for j in range(k)] for i in range(k)]
        w0x = self.w0c
        Lc = [[mp.fsum(self.K[mm][j][s] * w0x[s] for s in range(self.m)) for j in range(k)] for mm in range(k)]
        lw = [[[self.C[s][j][l] / self.lam[s] for s in range(self.m)] for l in range(k)] for j in range(k)]
        Sc = [[[[mp.fsum(self.K[mm][i][s] * lw[j][l][s] for s in range(self.m)) for l in range(k)] for j in range(k)]
               for i in range(k)] for mm in range(k)]
        Sc = symmetrize(Sc)
        Bc = [[self.sig[mm] if mm == j else mp.mpf(0) for j in range(k)] for mm in range(k)]

        def r2(A):
            return [[mp.fsum(Op[mp_][mm] * A[mp_][jp] * O[jp][j] for mp_ in range(k) for jp in range(k))
                     for j in range(k)] for mm in range(k)]

        def r3(A):
            # rotate index by index: the weak row, then the two directions
            A1 = [[[mp.fsum(Op[mp_][mm] * A[mp_][i][j] for mp_ in range(k)) for j in range(k)] for i in range(k)]
                  for mm in range(k)]
            A2 = [[[mp.fsum(A1[mm][ip][j] * O[ip][i] for ip in range(k)) for j in range(k)] for i in range(k)]
                  for mm in range(k)]
            return [[[mp.fsum(A2[mm][i][jp] * O[jp][j] for jp in range(k)) for j in range(k)] for i in range(k)]
                    for mm in range(k)]

        def r4(A):
            A1 = [[[[mp.fsum(Op[mp_][mm] * A[mp_][i][j][l] for mp_ in range(k)) for l in range(k)] for j in range(k)]
                   for i in range(k)] for mm in range(k)]
            for ax in range(1, 4):
                A2 = [[[[mp.mpf(0)] * k for _ in range(k)] for _ in range(k)] for _ in range(k)]
                for mm in range(k):
                    for i in range(k):
                        for j in range(k):
                            for l in range(k):
                                idx = [i, j, l]
                                acc = []
                                for p in range(k):
                                    q = list(idx)
                                    q[ax - 1] = p
                                    acc.append(A1[mm][q[0]][q[1]][q[2]] * O[p][idx[ax - 1]])
                                A2[mm][i][j][l] = mp.fsum(acc)
                A1 = A2
            return A1
        yr = self.yroot
        croot = [mp.fsum(O[i][j] * yr[i] for i in range(k)) for j in range(k)]     # c = O^T y
        return dict(B=r2(Bc), L=r2(Lc), Q=r3(self.Q), T=r4(self.T), S=r4(Sc), root=croot)


# ------------------------------------------------------------------ tensors at z0
def symmetrize(S):
    """the symmetric part of S[m][i][j][l], symmetric in (j, l) already: (S_i,jl + S_j,il + S_l,ij) / 3"""
    k = len(S)
    return [[[[(S[m][i][j][l] + S[m][j][i][l] + S[m][l][i][j]) / 3 for l in range(k)] for j in range(k)]
             for i in range(k)] for m in range(k)]


def proj_U(Um, v):
    return [dot(u, v) for u in Um]


def strong_part(Um, v):
    """(I - U U^T) v"""
    c = proj_U(Um, v)
    return [v[r] - mp.fsum(c[m] * Um[m][r] for m in range(len(Um))) for r in range(len(v))]


def second_diffs(sysm, X, h):
    """F''(v_j, v_l) as full vectors by central differences at step h: three points on the diagonal, four off it"""
    k = len(X)
    F0 = sysm.F()
    at = lambda c: sysm.F(lin([(mp.mpf(cj) * h, X[j]) for j, cj in c.items()]))   # noqa: E731
    out = {}
    for j in range(k):
        Fp, Fm = at({j: 1}), at({j: -1})
        out[(j, j)] = [(a - 2 * b + c) / mp.mpf(h) ** 2 for a, b, c in zip(Fp, F0, Fm)]
        for l in range(j + 1, k):
            pp, pm, mp_, mm = at({j: 1, l: 1}), at({j: 1, l: -1}), at({j: -1, l: 1}), at({j: -1, l: -1})
            out[(j, l)] = out[(l, j)] = [(a - b - c + d) / (4 * mp.mpf(h) ** 2) for a, b, c, d in zip(pp, pm, mp_, mm)]
    return out


def third_tensor(sysm, X, Um, h):
    """T[m][i][j][l] = u_m . F'''(v_i, v_j, v_l) by antisymmetric four-point differences of F along v and
    polarization (dz0_weak.cubic)"""
    k = len(X)
    D = {}

    def D3(key):
        if key not in D:
            c = dict(key)
            acc = None
            for wgt, t in ((1, 2), (-2, 1), (2, -1), (-1, -2)):
                Fv = proj_U(Um, sysm.F(lin([(mp.mpf(t) * h * cj, X[j]) for j, cj in c.items()])))
                acc = [wgt * x for x in Fv] if acc is None else [a + wgt * x for a, x in zip(acc, Fv)]
            D[key] = [x / (2 * mp.mpf(h) ** 3) for x in acc]
        return D[key]
    T = {}
    for j in range(k):
        T[(j, j, j)] = D3(((j, 1),))
    for j in range(k):
        for q in range(k):
            if q != j:
                plus, minus, yy = D3(((j, 1), (q, 1))), D3(((j, 1), (q, -1))), D3(((q, 1),))
                T[(j, j, q)] = [(p - mm - 2 * y) / 6 for p, mm, y in zip(plus, minus, yy)]
    for j in range(k):
        for q in range(j + 1, k):
            for w in range(q + 1, k):
                acc = [mp.mpf(0)] * k
                for e2 in (1, -1):
                    for e3 in (1, -1):
                        v = D3(((j, 1), (q, e2), (w, e3)))
                        acc = [x + e2 * e3 * y for x, y in zip(acc, v)]
                T[(j, q, w)] = [x / 24 for x in acc]
    full = [[[[None] * k for _ in range(k)] for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            for l in range(k):
                a, b, c = sorted((i, j, l))
                key = (a, a, a) if a == b == c else (a, a, c) if a == b else (c, c, a) if b == c else (a, b, c)
                for m in range(k):
                    full[m][i][j][l] = T[key][m]
    return full, 4 * len(D)


def mixed(sysm, Um, x, w, h, mutate=""):
    """U^T F''(x, w) by mixed central differences, step h along x (a unit vector) and h along w / |w|. --mutate=Sh
    (the control of S's h check) steps 10 h along w / |w| and divides by 4 h^2 all the same"""
    hw = 10 * h if mutate == "Sh" else h
    nw = nrm(w)
    e = [t / nw for t in w]
    acc = None
    for sx, sw, wgt in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        Fv = proj_U(Um, sysm.F(lin([(mp.mpf(sx) * h, x), (mp.mpf(sw) * hw, e)])))
        acc = [wgt * t for t in Fv] if acc is None else [a + wgt * t for a, t in zip(acc, Fv)]
    return [t * nw / (4 * mp.mpf(h) ** 2) for t in acc]


def S_tensor(sysm, X, Um, wjl, h, mutate=""):
    """S[m][i][j][l] = u_m . F''(v_i, w_jl) by mixed differences (symmetric in j, l; not symmetrized)"""
    k = len(X)
    S = [[[[None] * k for _ in range(k)] for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            for l in range(j, k):
                v = mixed(sysm, Um, X[i], wjl[(j, l)], h, mutate)
                for m in range(k):
                    S[m][i][j][l] = S[m][i][l][j] = v[m]
    return S


def tmax(A):
    """max |entry| of a nested list"""
    if isinstance(A, list):
        return max(tmax(a) for a in A)
    return abs(A)


def tdiff(A, B):
    if isinstance(A, list):
        return max(tdiff(a, b) for a, b in zip(A, B))
    return abs(A - B)


def tmap(f, A, B):
    if isinstance(A, list):
        return [tmap(f, a, b) for a, b in zip(A, B)]
    return f(A, B)


# ------------------------------------------------------------------ the strong solve and the reduced map
def strong_solve(sysm, X, Um, c, w_init, wref, mutate="", it_max=30):
    """w(c) by the chord iteration (docstring), judged by its corrections |dw| (2-norm), not by the strong rows'
    residual, whose rows differ in scale by 1e53 and whose max norm can grow while w converges. Converged when
    |dw| < 1e-40 wref (wref = |w0|); stalled (|dw| above 0.3 of its value two steps back) counts as converged only
    below 1e-30 wref; a growing |dw| is a failure. Returns (w, F at z0 + V c + w, the steps' (max|strong residual|,
    |dw|), ok)"""
    pv = lin([(mp.mpf(cj), X[j]) for j, cj in enumerate(c)])
    if mutate == "w0":                                  # the control: no strong solve, the map on z0 + V c
        Fv = sysm.F(pv)
        return [mp.mpf(0)] * len(pv), Fv, [(float(amax(strong_part(Um, Fv))), 0.0)], True
    w = list(w_init)
    hist = []
    tol, stall = mp.mpf("1e-40") * wref, mp.mpf("1e-30") * wref
    for _ in range(it_max):
        Fv = sysm.F([a + b for a, b in zip(pv, w)])
        r = strong_part(Um, Fv)
        dw = sysm.solve(r)
        nd = nrm(dw)
        hist.append((float(amax(r)), float(nd)))
        if nd < tol:                                    # converged; F is that of the current w
            return w, Fv, hist, True
        if len(hist) >= 3 and nd > 0.3 * hist[-3][1]:   # stalled
            return w, Fv, hist, bool(nd < stall)
        if len(hist) >= 2 and nd > hist[-2][1]:         # diverging
            return w, Fv, hist, False
        w = [a - b for a, b in zip(w, dw)]
    return w, Fv, hist, False


class Map:
    """g(c) on the strong-solved manifold, with every evaluation logged"""

    def __init__(self, sysm, X, Um, w0, mutate, log):
        self.sysm, self.X, self.Um, self.w0, self.mutate, self.log = sysm, X, Um, w0, mutate, log
        self.wref = max(nrm(w0), mp.mpf("1e-300"))
        self.calls = 0
        self.steps = []

    def __call__(self, c, w_init=None):
        w, Fv, hist, ok = strong_solve(self.sysm, self.X, self.Um, c, self.w0 if w_init is None else w_init,
                                       self.wref, self.mutate)
        self.calls += 1
        self.steps.append(len(hist))
        if not ok:
            self.log(f"  strong solve at |c| {mp.nstr(nrm(c), 3)} did not converge: (max|strong residual|, |dw|) per "
                     "step " + ", ".join(f"({x:.1e}, {y:.1e})" for x, y in hist))
        return proj_U(self.Um, Fv), w, Fv, hist, ok


def model(g0, Bm, Qm, Cm):
    """g(c) = g0 + B c + 1/2 Q(c, c) + 1/6 C(c, c, c) and its Jacobian, C symmetric"""
    k = len(g0)

    def g(c):
        return [g0[m] + mp.fsum(Bm[m][j] * c[j] for j in range(k))
                + (mp.fsum(Qm[m][j][l] * c[j] * c[l] for j in range(k) for l in range(k)) / 2 if Qm else 0)
                + mp.fsum(Cm[m][i][j][l] * c[i] * c[j] * c[l] for i in range(k) for j in range(k) for l in range(k)) / 6
                for m in range(k)]

    def jac(c):
        return mp.matrix([[Bm[m][j] + (mp.fsum(Qm[m][j][l] * c[l] for l in range(k)) if Qm else 0)
                           + mp.fsum(Cm[m][j][i][l] * c[i] * c[l] for i in range(k) for l in range(k)) / 2
                           for j in range(k)] for m in range(k)])
    return g, jac


def model_roots(g, jac, starts, scale, nBa):
    roots = []
    for c in starts:
        c = list(c)
        ng = mp.inf
        for _ in range(200):
            gv = g(c)
            ng = nrm(gv)
            if ng < mp.mpf("1e-40") * nBa:
                break
            try:
                dc = mp.lu_solve(jac(c), mp.matrix(gv))
            except ZeroDivisionError:
                break
            c = [c[j] - dc[j] for j in range(len(c))]
        if ng < mp.mpf("1e-40") * nBa and not any(amax([x - y for x, y in zip(c, r)]) < mp.mpf("1e-8") * scale
                                                  for r in roots):
            roots.append(c)
    return roots


# ------------------------------------------------------------------ the measurement
def measure(sysm, opts, mutate, log, rec, dump):
    k, its = int(opts.get("k", 4)), int(opts.get("its", 16))
    L = sysm.L
    t0 = time.time()
    V, hist = subspace64(sysm.solve64, L, k, its)
    X = mgs([[mp.mpf(float(x)) for x in V[:, j]] for j in range(k)])[0]
    for _ in range(3):
        X = mgs([sysm.solve(sysm.solve(x, True)) for x in X])[0]
    Um, Rm = mgs([sysm.solve(x, True) for x in X])
    Bm_ = (mp.matrix(Rm) ** -1).T
    B = [[Bm_[i, j] for j in range(k)] for i in range(k)]
    sv = sorted(float(x) for x in mp.svd_r(Bm_, compute_uv=False))
    dz0 = sysm.dz0
    a = [dot(x, dz0) for x in X]
    rest = [y - mp.fsum(a[j] * X[j][r] for j in range(k)) for r, y in enumerate(dz0)]
    Ba = [mp.fsum(B[m][j] * a[j] for j in range(k)) for m in range(k)]
    nBa, na = nrm(Ba), nrm(a)
    F0 = sysm.F()
    gF0 = proj_U(Um, F0)
    dz0_re = [-x for x in sysm.solve(F0)]
    redz = nrm([x - y for x, y in zip(dz0_re, dz0)]) / nrm(dz0)
    w0 = [-x for x in sysm.solve(strong_part(Um, F0))]
    log(f"  basis: sigma of B {', '.join(f'{x:.4e}' for x in sv)} (iteration {', '.join(f'{x:.4e}' for x in hist[-1])}); "
        f"|dz0| {mp.nstr(nrm(dz0), 4)}, a = V^T dz0 {', '.join(mp.nstr(x, 4) for x in a)}, |a| {mp.nstr(na, 4)}, the "
        f"rest {mp.nstr(nrm(rest), 3)}; |B a| {mp.nstr(nBa, 4)}, |U^T F(z0) + B a| / |B a| "
        f"{mp.nstr(nrm([x + y for x, y in zip(gF0, Ba)]) / nBa, 3)}; -J^-1 F(z0) against dz0 {mp.nstr(redz, 3)}; "
        f"w0 = -J^-1 (I - U U^T) F(z0): |w0| {mp.nstr(nrm(w0), 4)}, |V^T w0| {mp.nstr(nrm(proj_U(X, w0)), 3)} "
        f"({time.time() - t0:.0f} s)")
    rec.update(sigma_B=sv, sigma_iter=[float(x) for x in hist[-1]], a=[float(x) for x in a], a_norm=float(na),
               Ba=float(nBa), rest=float(nrm(rest)), dz0_re=float(redz), w0=float(nrm(w0)))
    dump()

    # Q, T, w_jl, S, L, q00 at z0
    t0 = time.time()
    hQ = H["Q"]
    Qs = []
    for hh in (hQ, hQ / 10):
        F2 = second_diffs(sysm, X, hh)
        Qs.append([[[dot(Um[m], F2[(j, l)]) for l in range(k)] for j in range(k)] for m in range(k)])
    Q = Qs[1]
    dQ = tdiff(Qs[0], Qs[1]) / tmax(Qs[1])
    log(f"  Q = U^T F'': max|Q| {mp.nstr(tmax(Q), 4)} at h {hQ / 10:.0e}, against h {hQ:.0e} {mp.nstr(dQ, 3)} "
        f"({time.time() - t0:.0f} s)")
    rec.update(Q=dict(max=float(tmax(Q)), h_h10=float(dQ)))
    dump()
    t0 = time.time()
    hF = H_MUT_F if mutate == "hF" else H["F"]
    Ws = []
    for hh in (hF, hF / 10, hF / 100):
        F2 = second_diffs(sysm, X, hh)
        Ws.append({key: sysm.solve(strong_part(Um, v)) for key, v in F2.items() if key[0] <= key[1]})
    # w_jl from the smallest step; the judged pair is the last two (h_F / 10 against h_F / 100), the first pair is
    # printed for the h^2 pattern (a truncation falls by 100 per decade of h, a rounding grows)
    dWs = [max(nrm([x - y for x, y in zip(Wa[key], Wb[key])]) / nrm(Wb[key]) for key in Wb)
           for Wa, Wb in ((Ws[0], Ws[1]), (Ws[1], Ws[2]))]
    wjl = Ws[2]
    dW = dWs[1]
    W_ok = bool(dW < 1e-2)
    for key in list(wjl):
        wjl[(key[1], key[0])] = wjl[key]
    F2max = max(amax(v) for v in F2.values())
    log(f"  w_jl = J^-1 (I - U U^T) F''(v_j, v_l): max|F''| {mp.nstr(F2max, 3)}, |w_jl| "
        f"{mp.nstr(min(nrm(v) for v in wjl.values()), 3)}..{mp.nstr(max(nrm(v) for v in wjl.values()), 3)} at h "
        f"{hF / 100:.0e}, |V^T w_jl| / |w_jl| up to {mp.nstr(max(nrm(proj_U(X, v)) / nrm(v) for v in wjl.values()), 3)}; "
        f"h {hF:.0e} against {hF / 10:.0e} {mp.nstr(dWs[0], 3)}, {hF / 10:.0e} against {hF / 100:.0e} {mp.nstr(dW, 3)} "
        f"(bound 1e-2): {'pass' if W_ok else 'FAIL'} ({time.time() - t0:.0f} s)")
    rec.update(w=dict(F2max=float(F2max), wmax=float(max(nrm(v) for v in wjl.values())),
                      wmin=float(min(nrm(v) for v in wjl.values())), h=hF / 100, diffs=[float(x) for x in dWs],
                      ok=W_ok))
    dump()
    t0 = time.time()
    Ts = [third_tensor(sysm, X, Um, hh) for hh in (H["T"], H["T"] / 10)]
    T = Ts[1][0]
    dT = tdiff(Ts[0][0], Ts[1][0]) / tmax(T)
    log(f"  T = U^T F''': max|T| {mp.nstr(tmax(T), 4)} at h {H['T'] / 10:.0e}, against h {H['T']:.0e} {mp.nstr(dT, 3)} "
        f"({Ts[0][1]} residuals per h, {time.time() - t0:.0f} s)")
    rec.update(T=dict(max=float(tmax(T)), h_h10=float(dT)))
    dump()
    t0 = time.time()
    Sr = [S_tensor(sysm, X, Um, wjl, hh, mutate if hh != H["S"] else "") for hh in (H["S"], H["S"] / 10)]
    Ss = [symmetrize(S) for S in Sr]
    S = Ss[1]
    dS = tdiff(Ss[0], Ss[1]) / tmax(S)
    ratio = tmax(Ss[0]) / tmax(Ss[1])
    S_ok = bool(dS < 1e-2)
    S_trunc = bool(not S_ok and 50 < ratio < 200)
    asym = tdiff(Sr[1], S) / tmax(S)
    log(f"  S = U^T F''(v_i, w_jl), symmetrized: max|S| {mp.nstr(tmax(S), 4)} at h {H['S'] / 10:.0e}, against h "
        f"{H['S']:.0e} {mp.nstr(dS, 3)} (bound 1e-2): {'pass' if S_ok else 'FAIL'}"
        + (f"; the ratio {mp.nstr(ratio, 3)} is an h^2 truncation's, so |S| < {mp.nstr(tmax(S), 3)}" if S_trunc else "")
        + f"; its unsymmetric part {mp.nstr(asym, 3)} of it; 3 max|S| / max|T| {mp.nstr(3 * tmax(S) / tmax(T), 4)} "
        f"({time.time() - t0:.0f} s)")
    rec.update(S=dict(max=float(tmax(S)), h_h10=float(dS), ok=S_ok, trunc_bound=S_trunc, ratio=float(ratio),
                      asym=float(asym), threeS_T=float(3 * tmax(S) / tmax(T))))
    dump()
    t0 = time.time()
    if nrm(w0) > 0:
        Ls = [[mixed(sysm, Um, X[j], w0, hh) for j in range(k)] for hh in (H["S"], H["S"] / 10)]
        Lm = [[Ls[1][j][m] for j in range(k)] for m in range(k)]
        dL = tdiff(Ls[0], Ls[1]) / tmax(Ls[1])
        nw = nrm(w0)
        e = [t / nw for t in w0]
        hh = H["S"]
        Fp, Fm = sysm.F([hh * t for t in e]), sysm.F([-hh * t for t in e])
        q00 = [(a - 2 * b + c) * nw ** 2 / mp.mpf(hh) ** 2 for a, b, c in zip(proj_U(Um, Fp), gF0, proj_U(Um, Fm))]
    else:
        Lm, dL, q00 = [[mp.mpf(0)] * k for _ in range(k)], mp.mpf(0), [mp.mpf(0)] * k
    sL = sorted(float(x) for x in mp.svd_r(mp.matrix(Lm), compute_uv=False)) if nrm(w0) > 0 else [0.0] * k
    BL = [[B[m][j] + Lm[m][j] for j in range(k)] for m in range(k)]
    sBL = sorted(float(x) for x in mp.svd_r(mp.matrix(BL), compute_uv=False))
    log(f"  L = U^T F''(v_j, w0): max|L| {mp.nstr(tmax(Lm), 4)}, sigma {', '.join(f'{x:.3e}' for x in sL)}, h against "
        f"h/10 {mp.nstr(dL, 3)}; sigma of B + L {', '.join(f'{x:.3e}' for x in sBL)}; |q00| = |U^T F''(w0, w0)| "
        f"{mp.nstr(nrm(q00), 3)} ({time.time() - t0:.0f} s)")
    rec.update(L=dict(max=float(tmax(Lm)), sigma=sL, h_h10=float(dL), sigma_BL=sBL), q00=float(nrm(q00)))
    dump()

    # the reduced map
    g = Map(sysm, X, Um, w0, mutate, log)
    t0 = time.time()
    g00, w00, F00, h00, ok00 = g([mp.mpf(0)] * k)
    w00 = list(w00)
    log(f"  g(0): |g(0) + B a| / |B a| {mp.nstr(nrm([x + y for x, y in zip(g00, Ba)]) / nBa, 3)}, |w(0) - w0| "
        f"{mp.nstr(nrm([x - y for x, y in zip(w00, w0)]), 3)}, (max|strong residual|, |dw|) per chord step "
        + ", ".join(f"({x:.1e}, {y:.1e})" for x, y in h00) + f" ({time.time() - t0:.0f} s)")
    rec.update(g0=dict(rel=float(nrm([x + y for x, y in zip(g00, Ba)]) / nBa), chord=h00, ok=ok00))
    # B_LS and Q_LS from central differences of g at h_LS
    t0 = time.time()
    hl = mp.mpf(H["LS"])
    e = lambda d: [hl * d.get(j, 0) for j in range(k)]       # noqa: E731
    gat = {}
    for j in range(k):
        for sgn in (1, -1):
            gat[((j, sgn),)] = g(e({j: sgn}), w00)[0]
        for l in range(j + 1, k):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    gat[((j, s1), (l, s2))] = g(e({j: s1, l: s2}), w00)[0]
    BLS = [[(gat[((j, 1),)][m] - gat[((j, -1),)][m]) / (2 * hl) for j in range(k)] for m in range(k)]
    QLS = [[[None] * k for _ in range(k)] for _ in range(k)]
    for m in range(k):
        for j in range(k):
            QLS[m][j][j] = (gat[((j, 1),)][m] - 2 * g00[m] + gat[((j, -1),)][m]) / hl ** 2
            for l in range(j + 1, k):
                QLS[m][j][l] = QLS[m][l][j] = (gat[((j, 1), (l, 1))][m] - gat[((j, 1), (l, -1))][m]
                                               - gat[((j, -1), (l, 1))][m] + gat[((j, -1), (l, -1))][m]) / (4 * hl ** 2)
    dBL = tdiff(BLS, BL) / tmax(BL)
    dQL = tdiff(QLS, Q) / tmax(Q)
    log(f"  the map's own coefficients at h {H['LS']:.0e}: |B_LS - (B + L)| / max|B + L| {mp.nstr(dBL, 3)}; max|Q_LS| "
        f"{mp.nstr(tmax(QLS), 4)}, |Q_LS - Q| / max|Q| {mp.nstr(dQL, 3)} ({g.calls} solves, {time.time() - t0:.0f} s)")
    rec.update(LS=dict(B_vs_BL=float(dBL), Qmax=float(tmax(QLS)), Q_vs_Q=float(dQL)))
    dump()

    # c* (#499's root of the affine cubic), and every model's roots
    t0 = time.time()
    scale = mp.cbrt(6 * nBa / tmax(T))
    rng = np.random.default_rng(int(opts.get("seed_roots", 3)))
    starts = [list(a), [mp.mpf(0)] * k] + [[mp.mpf(float(x)) * scale for x in rng.standard_normal(k)] for _ in range(24)]
    Cm = tmap(lambda t, s: t - 3 * s * (-1 if mutate == "Ssign" else 1), T, S)
    mB = [-x for x in Ba]
    models = {
        "T": (mB, B, None, T),
        "T,Q": (mB, B, Q, T),
        "T-3S": (g00, B, Q, Cm),
        "T-3S,L": (g00, BL, Q, Cm),
        "T-3S,LS": (g00, BLS, QLS, Cm),
    }
    out = {}
    cstar = None
    for name, (g0m, Bm, Qm, Cmm) in models.items():
        gm, jm = model(g0m, Bm, Qm, Cmm)
        rts = model_roots(gm, jm, starts, scale, nBa)
        rts.sort(key=lambda r: nrm([x - y for x, y in zip(r, a)]))
        evals = []
        for r in rts[:int(opts.get("nroots", 6))]:
            gv, wv, Fv, hv, okv = g(r, w00)
            # the affine map too (U^T F(z0 + V c)), #499's root check
            gaff = proj_U(Um, sysm.F(lin([(cj, X[j]) for j, cj in enumerate(r)]))) if name == "T" else None
            evals.append(dict(c=r, g=nrm(gv) / nBa, gaff=(nrm(gaff) / nBa) if gaff else None, ok=okv,
                              Fmax=float(amax(Fv)), w=wv, chord=hv))
        conv = [d for d in evals if d["ok"]]
        best = min(conv, key=lambda d: (d["gaff"] if name == "T" else d["g"])) if conv else None
        if name == "T" and best is not None:
            cstar = best
        out[name] = dict(roots=len(rts), evals=evals, best=best)
        log(f"  model {name}: {len(rts)} real roots from {len(starts)} starts; nearest a: " + "; ".join(
            f"|c| {mp.nstr(nrm(d['c']), 3)} ({', '.join(mp.nstr(x, 3) for x in d['c'])}): |g| / |B a| "
            f"{mp.nstr(d['g'], 3)}" + (f", affine {mp.nstr(d['gaff'], 3)}" if d["gaff"] is not None else "")
            + f", max|F| {d['Fmax']:.2e}" + ("" if d["ok"] else " (strong solve NOT converged)") for d in evals))
    rec["models"] = {n: dict(roots=d["roots"], evals=[dict(c=[float(x) for x in e["c"]], g=float(e["g"]),
                                                           gaff=None if e["gaff"] is None else float(e["gaff"]),
                                                           ok=e["ok"], Fmax=e["Fmax"], chord=e["chord"])
                                                      for e in d["evals"]]) for n, d in out.items()}
    ga = g(a, w00)
    log(f"  at c = a: |g| / |B a| {mp.nstr(nrm(ga[0]) / nBa, 3)}, max|F| {float(amax(ga[2])):.2e}; models' roots "
        f"{time.time() - t0:.0f} s")
    rec.update(g_at_a=float(nrm(ga[0]) / nBa))
    dump()

    # the controls at c*
    ok = (S_ok or S_trunc) and W_ok
    if cstar is None:
        log("  no root of model T: no c*, the controls cannot run")
        rec["controls"] = None
        return False, rec
    cs = cstar["c"]
    gcs, wcs = g(cs, w00)[:2]
    dw = [x - y for x, y in zip(wcs, w00)]
    Sc = [mp.fsum(S[m][i][j][l] * cs[i] * cs[j] * cs[l] for i in range(k) for j in range(k) for l in range(k))
          for m in range(k)]
    Tc = [mp.fsum(T[m][i][j][l] * cs[i] * cs[j] * cs[l] for i in range(k) for j in range(k) for l in range(k))
          for m in range(k)]
    sgnS = -1 if mutate == "Ssign" else 1
    pv = lin([(cj, X[j]) for j, cj in enumerate(cs)])
    F1 = proj_U(Um, sysm.F([p + x + y for p, x, y in zip(pv, w00, dw)]))
    F2 = proj_U(Um, sysm.F([p + x + 2 * y for p, x, y in zip(pv, w00, dw)]))
    dmeas = [y - x for x, y in zip(F1, F2)]
    dpred = [-sgnS * x / 2 for x in Sc]
    pred_n = nrm(dpred)
    errS = nrm([x - y for x, y in zip(dmeas, dpred)]) / max(pred_n, nrm(dmeas))
    resolvable = bool(max(pred_n, nrm(dmeas)) > mp.mpf("1e-85"))
    ok_S = bool(resolvable and errS < 1e-2)
    # the dw's quadratic model, -1/2 sum w_jl c_j c_l
    dwq = lin([(-cs[j] * cs[l] / 2, wjl[(j, l)]) for j in range(k) for l in range(k)])
    edw = nrm([x - y for x, y in zip(dw, dwq)]) / max(nrm(dw), mp.mpf("1e-300"))
    F3 = proj_U(Um, sysm.F([p + 2 * x + 2 * y for p, x, y in zip(pv, w00, dw)]))
    wm = [y - x for x, y in zip(F1, F3)]
    Lc = [mp.fsum(Lm[m][j] * cs[j] for j in range(k)) for m in range(k)]
    wp = [lc + pd + 3 * q / 2 for lc, pd, q in zip(Lc, dpred, q00)]
    errW = nrm([x - y for x, y in zip(wm, wp)]) / max(nrm(wp), nrm(wm))
    log(f"  c* = the root of model T (|c*| {mp.nstr(nrm(cs), 4)}): |g(c*)| / |B a| {mp.nstr(nrm(gcs) / nBa, 3)} on the "
        f"strong-solved manifold, affine {mp.nstr(cstar['gaff'], 3)}; |S(c*,c*,c*)| {mp.nstr(nrm(Sc), 3)}, "
        f"|T(c*,c*,c*)| {mp.nstr(nrm(Tc), 3)}; dw = w(c*) - w(0): |dw| {mp.nstr(nrm(dw), 3)}, against its quadratic "
        f"model {mp.nstr(edw, 3)}")
    log(f"  S control: dw -> 2 dw changes g by {mp.nstr(nrm(dmeas), 4)}, predicted -1/2 S(c*,c*,c*) "
        f"{mp.nstr(pred_n, 4)}, difference {mp.nstr(errS, 3)} of it (bound 1e-2"
        + ("" if resolvable else ", but both below 1e-85: not resolved") + f"): {'pass' if ok_S else 'FAIL'}")
    log(f"  w control (diagnostic): w(c*) -> 2 w(c*) changes g by {mp.nstr(nrm(wm), 4)}, predicted L c* - 1/2 S + "
        f"3/2 q00 {mp.nstr(nrm(wp), 4)} (L c* {mp.nstr(nrm(Lc), 3)}, 3/2 q00 {mp.nstr(nrm(q00) * 3 / 2, 3)}), "
        f"difference {mp.nstr(errW, 3)} of it")
    rec["controls"] = dict(cstar=[float(x) for x in cs], g_cstar=float(nrm(gcs) / nBa), gaff_cstar=float(cstar["gaff"]),
                           Sc=float(nrm(Sc)), Tc=float(nrm(Tc)), dw=float(nrm(dw)), dw_vs_quad=float(edw),
                           S_meas=float(nrm(dmeas)), S_pred=float(pred_n), S_err=float(errS), S_ok=ok_S,
                           S_resolvable=resolvable, w_meas=float(nrm(wm)), w_pred=float(nrm(wp)), w_err=float(errW),
                           Lc=float(nrm(Lc)), q00=float(nrm(q00)))
    # the verdict of the issue
    small_S = bool(3 * tmax(S) < mp.mpf("1e-2") * tmax(T))
    small_g = bool(nrm(gcs) < mp.mpf("1e-2") * nBa)
    full = out["T-3S,L"]["best"]
    rec["verdict"] = dict(small_S=small_S, small_g_cstar=small_g,
                          model_root_g=None if full is None else float(full["g"]))
    if small_S and small_g:
        log("  verdict: 3|S| < 1e-2 |T| and |g(c*)| < 1e-2 |B a|: the coupling is not this term")
    else:
        for name in ("T-3S", "T-3S,L", "T-3S,LS"):
            b = out[name]["best"]
            log(f"  verdict, model {name}: " + ("no real root" if b is None else
                f"root |c**| {mp.nstr(nrm(b['c']), 4)} with |g(c**)| / |B a| {mp.nstr(b['g'], 3)} "
                f"({'below' if b['g'] < 1e-2 else 'NOT below'} 1e-2), max|F| there {b['Fmax']:.2e}"))
    dump()
    rec["_internal"] = dict(X=X, Um=Um, B=B, BL=BL, Q=Q, T=T, S=S, Lm=Lm, out=out, BLS=BLS, QLS=QLS, cstar=cs)
    return ok and ok_S, rec


def strip(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    mutate = opts.get("mutate", "")
    if mutate not in ("", "w0", "Ssign", "Sh", "hF"):
        sys.exit(f"--mutate={mutate}: w0, Ssign, Sh or hF")
    if not args or args[0] not in ("run", "planted") or (args[0] == "run" and len(args) != 4):
        sys.exit(__doc__)
    out = opts.get("json", "")
    rec = dict(mode=args[0], mutate=mutate, h=H, started=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    def log(s):
        print(s, flush=True)

    def dump():
        if out:
            with open(out, "w") as fh:
                json.dump(strip(rec), fh, indent=1, default=float)
    if args[0] == "run":
        sysm = Real(*args[1:4], int(opts.get("threads", 2)), log)
        rec.update(sysm.meta)
        if not sysm.offset_ok:
            log("check: FAIL (the held offset is not on its rows only)")
            dump()
            return 1
    else:
        sysm = Planted(int(opts.get("seed", 5)), log)
        rec.update(sysm.meta)
    ok, rec = measure(sysm, opts, mutate, log, rec, dump)
    if args[0] == "planted":
        I = rec["_internal"]
        tr = sysm.truth(I["X"], I["Um"])
        errs = dict(B=tdiff(I["B"], tr["B"]) / tmax(tr["B"]), L=tdiff(I["Lm"], tr["L"]) / tmax(tr["L"]),
                    Q=tdiff(I["Q"], tr["Q"]) / tmax(tr["Q"]), T=tdiff(I["T"], tr["T"]) / tmax(tr["T"]),
                    S=tdiff(I["S"], tr["S"]) / tmax(tr["S"]))
        ok_t = all(e < 1e-6 for e in errs.values())
        croot = tr["root"]
        evs = [d for d in I["out"]["T-3S,L"]["evals"] if d["ok"]]
        best = min(evs, key=lambda d: nrm([x - y for x, y in zip(d["c"], croot)])) if evs else None
        dr = (nrm([x - y for x, y in zip(best["c"], croot)]) / nrm(croot)) if best else mp.inf
        ok_r = bool(best is not None and dr < 1e-3 and best["g"] < 1e-2)
        shows = bool(I["out"]["T"]["best"] is not None and rec["controls"]["g_cstar"] >= 0.1)
        log("  planted: recovered against planted, relative to the largest entry: "
            + ", ".join(f"{n} {mp.nstr(e, 3)}" for n, e in errs.items()) + f" (bound 1e-6): {'pass' if ok_t else 'FAIL'}")
        log(f"  planted: model T-3S,L's root against the planted root {mp.nstr(dr, 3)} (bound 1e-3), |g| / |B a| there "
            f"{mp.nstr(best['g'], 3) if best else 'none'} (bound 1e-2): {'pass' if ok_r else 'FAIL'}; the plant shows "
            f"(|g(c*)| / |B a| = {rec['controls']['g_cstar']:.3g} >= 0.1 at model T's root): {'yes' if shows else 'NO'}")
        rec["planted_check"] = dict(errors={n: float(e) for n, e in errs.items()}, tensors_ok=ok_t, root_err=float(dr),
                                    root_ok=ok_r, shows=shows)
        ok = ok and ok_t and ok_r and shows
    rec["passed"] = bool(ok)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    dump()
    print("check:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
