#!/usr/bin/env python3
"""The strong-row term S of the reduced cubic, and the reduced map on the strong-solved manifold (#528).

    cd hp && python3 reduced_cubic.py run RAWDIR DZ0 BASE [--k=4] [--its=16] [--threads=2] [--json=OUT] [--mutate=M]
    cd hp && python3 reduced_cubic.py planted [--seed=5] [--json=OUT] [--mutate=M]
    cd hp && python3 reduced_cubic.py ray RAWDIR DZ0 BASE [--tensors=1] [--threads=2] [--json=OUT] [--mutate=w0|B]
    cd hp && python3 reduced_cubic.py ray planted [--seed=5] [--json=OUT] [--mutate=w0|B]
    cd hp && python3 reduced_cubic.py raysum OUT.json [OUT.json ...]     (the ray runs' summary, raysum's docstring)
    cd hp && python3 reduced_cubic.py rows RAWDIR DZ0 BASE [--threads=2] [--json=OUT]
    cd hp && python3 reduced_cubic.py rows planted [--seed=5] [--json=OUT] [--mutate=quadA|parity]
    cd hp && python3 reduced_cubic.py rowsum OUT.json [OUT.json ...]     (the rows runs' summary)

`rows` (#694) splits the weak left vectors U over the residual's rows, expanded on harmonics within each field of each
block (row_harm: block.field.harmonic, e.g. horizon.p.k5c, and the phase row), and measures each row's second
derivative along the weak right directions, q_r = max over j <= l of |F''_r(v_j, v_l)| (central differences at
H_ROWS, the second judged). A row is affine to 1e-10 when q_r is below 1e-10 of the largest q and below 1e-10 of its
own |J_r|. Printed: the share of |U| on the affine rows (the verdict: above 0.99 supports the conjecture of affine weak
rows, below 0.5 refutes it), by block and field, the rows that carry the rest, V by unknowns, and along e = a / |a| the
rows' parts of U^T F''(e, e), U^T T(e) = U^T F'''(e, e, e) and U^T 3S(e) = 3 U^T F''(e, w_ee), each against the size
it would have without cancellation between rows. Checks on real data (exit 1): the rows linear by construction (the
horizon block's p rows above kfreeze and the phase row) are classified affine; the change of rows is orthonormal; w_ee,
T(e) and S(e) at two steps agree to 1e-2; and |U^T (T - 3S)(e)| < 1e-12 |U^T T(e)|. Its plant (RowPlanted) has five
planted affine rows, four of which carry U with the planted share 0.96625; it must recover the affine set, the share
and every row's share to 1e-6, q_r, F''(e, e), T(e) and S(e) per row to 1e-6. Its controls --mutate=quadA (a
quadratic term in the planted affine rows that carry U) and --mutate=parity (each field expanded on the other parity's
harmonics) must fail.

`ray` (#677) scans the reduced map g on the strong-solved manifold along rays c = +-t e, e = a / |a| and two random
unit vectors, at t = 1e-30 .. 1e-24 (four per decade; 1e-29 .. 1e-26 judged), and fits the order at which it departs
from its affine part, dep(c) = g(c) - g(0) - B c, in its even and odd parts (ray_scan). With --tensors=1 it first
runs the whole of `run`'s measurement, whose h checks are then judged. Its plant (RayPlanted) has T = 3S, Q = L = 0
and a planted departure P4/24 + P5/120; it must recover each point to 1e-3, the orders 4 and 5 to 0.05 and the
crossing of 1e-2 |B a| to 1e-2. Its controls --mutate=w0 (no strong solve) and --mutate=B (B scaled by 1 + 1e-12)
must fail.

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
        self.s, self.z0, self.L, self.M, self.kf = s, z0, L, M, kf
        self.J0 = J[0]                                  # the Jacobian's leading part, for the rows' scales (#694)
        # the rows linear in z by construction: the horizon block's p rows above kfreeze (Pi' p1 minus the held
        # offset, multishoot's projected rows) and the phase row (sum_j w_j p1_j)
        self.frozen = [f"horizon.p.k{h}{cs}" for h in range(kf + 2, M, 2) for cs in "cs"] + ["phase"]
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

    def row_norms(self, RH):
        """|J_r|, the 2-norm of each row of J in RH's rows (f64, from the Jacobian's leading part)"""
        return [float(x) for x in np.linalg.norm(RH.matrix64() @ self.J0, axis=1)]


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


class RayPlanted(Planted):
    """the ray mode's plant (#677): Planted with Q = 0, L = 0 (K zero on x's rows 0-3, where w0 lives), M = 0 and
    T = 3 S, so that the reduced map is exactly g(y) = beta + Sigma y + 1/24 P4(y^4) + 1/120 P5(y^5), with
    P4_m(y^4) = p_m (a_m . y)^4 and P5_m(y^5) = q_m (b_m . y)^5. Its departure from the affine part is even of order 4
    and odd of order 5, with constants set so that it reaches 1e-2 |B a| near |c| = 20 |a| and P5 equals P4 near
    |c| = 1e-26. beta = -Sigma y_a with |y_a| 8.3e-30 (REF's |a|), so a = V^T dz0 is y_a rotated. The strong rows keep
    Planted's quartic c4 in row 0, where K is now zero, so it reaches w_jl but not g."""

    def __init__(self, seed, log, k=4, m=8):
        self.P5 = [(mp.mpf(0), [mp.mpf(0)] * k)] * k        # Planted's construction calls Gw
        super().__init__(seed, lambda s: None, k, m)
        rng = np.random.default_rng(seed + 677)
        zero = mp.mpf(0)
        self.Q = [[[zero] * k for _ in range(k)] for _ in range(k)]
        for mm in range(k):
            for j in range(k):
                for s in range(4):
                    self.K[mm][j][s] = zero
        self.Mt = [[[zero] * m for _ in range(k)] for _ in range(m)]
        lw = [[[self.C[s][j][l] / self.lam[s] for s in range(m)] for l in range(k)] for j in range(k)]
        Sc = symmetrize([[[[mp.fsum(self.K[mm][i][s] * lw[j][l][s] for s in range(m)) for l in range(k)]
                           for j in range(k)] for i in range(k)] for mm in range(k)])
        self.T = [[[[3 * Sc[mm][i][j][l] for l in range(k)] for j in range(k)] for i in range(k)] for mm in range(k)]
        yr = rng.standard_normal(k)
        self.ya = [mp.mpf(float(x)) * mp.mpf("8.3e-30") / mp.mpf(float(np.linalg.norm(yr))) for x in yr]
        self.beta = [-self.sig[mm] * self.ya[mm] for mm in range(k)]
        nBa = nrm([self.sig[mm] * self.ya[mm] for mm in range(k)])
        p4 = 24 * mp.mpf("1e-2") * nBa / (20 * nrm(self.ya)) ** 4
        q5 = 5 * p4 / mp.mpf("1e-26")
        self.P4 = [(p4 * mp.mpf(float(rng.standard_normal())), [mp.mpf(float(x)) for x in rng.standard_normal(k)])
                   for _ in range(k)]
        self.P5 = [(q5 * mp.mpf(float(rng.standard_normal())), [mp.mpf(float(x)) for x in rng.standard_normal(k)])
                   for _ in range(k)]
        self.F0max = float(amax(self.F()))
        self.dz0 = [-x for x in self.solve(self.F())]
        # the plant's own check: its reduced map minus beta + Sigma y is P4/24 + P5/120 (T - 3S cancels exactly)
        self.selfcheck = mp.mpf(0)
        for scale in ("1e-29", "1e-27"):
            y = [mp.mpf(float(x)) * mp.mpf(scale) for x in rng.standard_normal(k)]
            g0 = self.Gw([zero] * k, self.xsolve([zero] * k))
            gy = self.Gw(y, self.xsolve(y))
            dep = [a_ - b_ - s_ * yy for a_, b_, s_, yy in zip(gy, g0, self.sig, y)]
            want = self.dep_canon(y)
            self.selfcheck = max(self.selfcheck, nrm([x - w for x, w in zip(dep, want)]) / nrm(want))
        self.label = f"ray plant (seed {seed}): n {self.L}, k {k}"
        self.meta = dict(seed=seed, n=self.L, k=k, F0max=self.F0max, ya=[float(x) for x in self.ya],
                         selfcheck=float(self.selfcheck))
        log(f"{self.label}; start max|F| {self.F0max:.3e}, |w0| {mp.nstr(nrm(self.w0c), 3)}, |y_a| "
            f"{mp.nstr(nrm(self.ya), 3)}, |B a| {mp.nstr(nBa, 4)}; its reduced map minus beta + Sigma y against "
            f"P4/24 + P5/120 {mp.nstr(self.selfcheck, 3)} (bound 1e-20): {'pass' if self.selfcheck < 1e-20 else 'FAIL'}")

    def Gw(self, y, x):
        return [v + q * dot(b, y) ** 5 / 120 for v, (q, b) in zip(super().Gw(y, x), self.P5)]

    def dep_canon(self, y):
        """the planted departure P4(y^4)/24 + P5(y^5)/120 in the plant's own weak rows"""
        return [p * dot(av, y) ** 4 / 24 + q * dot(b, y) ** 5 / 120 for (p, av), (q, b) in zip(self.P4, self.P5)]

    def dep_true(self, X, Um):
        """the planted departure in the code's coordinates: c -> U^T-rotated P4/24 + P5/120 at y = O c"""
        k = self.k
        O = [[dot(self.PR[i], X[j]) for j in range(k)] for i in range(k)]
        Op = [[dot(self.PL[i], Um[j]) for j in range(k)] for i in range(k)]

        def f(c):
            y = [mp.fsum(O[i][j] * c[j] for j in range(k)) for i in range(k)]
            d = self.dep_canon(y)
            return [mp.fsum(Op[i][j] * d[i] for i in range(k)) for j in range(k)]
        return f


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
def strong_solve(sysm, X, Um, c, w_init, wref, gref, mutate="", it_max=30):
    """w(c) by the chord iteration (docstring), judged by its corrections |dw| (2-norm), not by the strong rows'
    residual, whose rows differ in scale by 1e53 and whose max norm can grow while w converges. Converged when
    |dw| < 1e-40 wref (wref = |w0|). The chord uses z0's J, and at c != 0 it differs from J(z) in the softest strong
    directions (sigma_5 ~ 1e-19). There its second correction can undo its first (at REF's c*: |dw| 5.44e-56, 5.44e-56,
    then 5e-82, with w(c*) - w(0) = 1.3e-60), so one step whose |dw| does not fall is allowed. A solve stalls when two
    steps in a row fail to cut |dw| by 0.3, or after it_max steps; it counts as converged when its last correction
    cannot change g: |U^T [F(w - dw) - F(w)]| < 1e-6 gref (gref = |B a|), one more residual. A |dw| that grows by more
    than 2 is a failure. Returns (w, F at z0 + V c + w, the steps' (max|strong residual|, |dw|), ok, the last
    correction's change of g or None)"""
    pv = lin([(mp.mpf(cj), X[j]) for j, cj in enumerate(c)])
    if mutate == "w0":                                  # the control: no strong solve, the map on z0 + V c
        Fv = sysm.F(pv)
        return [mp.mpf(0)] * len(pv), Fv, [(float(amax(strong_part(Um, Fv))), 0.0)], True, None
    w = list(w_init)
    hist = []
    tol = mp.mpf("1e-40") * wref

    def judge(w, Fv, dw):
        F2 = sysm.F([a + b - d for a, b, d in zip(pv, w, dw)])
        dg = nrm([x - y for x, y in zip(proj_U(Um, F2), proj_U(Um, Fv))])
        return w, Fv, hist, bool(dg < mp.mpf("1e-6") * gref), float(dg / gref)
    for _ in range(it_max):
        Fv = sysm.F([a + b for a, b in zip(pv, w)])
        r = strong_part(Um, Fv)
        dw = sysm.solve(r)
        nd = nrm(dw)
        hist.append((float(amax(r)), float(nd)))
        if nd < tol:                                    # converged; F is that of the current w
            return w, Fv, hist, True, None
        if len(hist) >= 2 and nd > 2 * hist[-2][1]:     # diverging
            return w, Fv, hist, False, None
        if len(hist) >= 3 and nd > 0.3 * hist[-2][1] and hist[-2][1] > 0.3 * hist[-3][1]:     # stalled
            return judge(w, Fv, dw)
        w = [a - b for a, b in zip(w, dw)]
    return judge(w, Fv, dw)


class Map:
    """g(c) on the strong-solved manifold, with every evaluation logged"""

    def __init__(self, sysm, X, Um, w0, gref, mutate, log):
        self.sysm, self.X, self.Um, self.w0, self.gref, self.mutate, self.log = sysm, X, Um, w0, gref, mutate, log
        self.wref = max(nrm(w0), mp.mpf("1e-300"))
        self.calls = 0
        self.stalls = []                                # the stalled solves' change of g, relative to |B a|

    def __call__(self, c, w_init=None):
        w, Fv, hist, ok, dg = strong_solve(self.sysm, self.X, self.Um, c, self.w0 if w_init is None else w_init,
                                           self.wref, self.gref, self.mutate)
        self.calls += 1
        if dg is not None:
            self.stalls.append(dg)
        if not ok:
            self.log(f"  strong solve at |c| {mp.nstr(nrm(c), 3)} did not converge: (max|strong residual|, |dw|) per "
                     "step " + ", ".join(f"({x:.1e}, {y:.1e})" for x, y in hist)
                     + ("" if dg is None else f"; its last correction changes g by {dg:.1e} of |B a|"))
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
def basis(sysm, opts, log, rec, dump):
    """V (X), U, B, a, w0 at z0, as the docstring says; the line it prints is the run's 'basis:' line"""
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
    return dict(k=k, X=X, Um=Um, B=B, a=a, Ba=Ba, nBa=nBa, na=na, rest=rest, F0=F0, gF0=gF0, w0=w0)


def measure(sysm, opts, mutate, log, rec, dump, bas=None):
    bas = bas or basis(sysm, opts, log, rec, dump)
    k, X, Um, B, a, Ba, nBa = (bas[n] for n in ("k", "X", "Um", "B", "a", "Ba", "nBa"))
    rest, gF0, w0 = bas["rest"], bas["gF0"], bas["w0"]

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
    C3 = tmap(lambda t, s: t - 3 * s, T, S)
    C3_0 = tmap(lambda t, s: t - 3 * s, Ts[0][0], Ss[0])
    dC3 = tdiff(C3_0, C3) / tmax(T)
    rec["_internal"] = dict(T=T, S=S, C3=C3, C3_0=C3_0)
    log(f"  T - 3S: max|T - 3S| / max|T| {mp.nstr(tmax(C3) / tmax(T), 4)}; T - 3S at the larger steps against the "
        f"smaller, relative to max|T|, {mp.nstr(dC3, 3)}")
    rec.update(S=dict(max=float(tmax(S)), h_h10=float(dS), ok=S_ok, trunc_bound=S_trunc, ratio=float(ratio),
                      asym=float(asym), threeS_T=float(3 * tmax(S) / tmax(T)), C3_T=float(tmax(C3) / tmax(T)),
                      C3_h=float(dC3)))
    rec["tensors"] = {n: tmap(lambda x, _: mp.nstr(x, 30), A, A) for n, A in
                      (("B", B), ("Q", Q), ("T", T), ("S", S), ("T_minus_3S", C3))}
    rec["tensors"]["a"] = [mp.nstr(x, 30) for x in a]
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
    g = Map(sysm, X, Um, w0, nBa, mutate, log)
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
    # #499's step split at c* (its z1 = z0 + V c* + (dz0 - V a)): U^T F with w the first step's rest, w0, w(0), w(c*)
    split = {}
    for name, wv in (("rest", rest), ("w0", w0), ("w(0)", w00), ("w(c*)", wcs)):
        split[name] = float(nrm(proj_U(Um, sysm.F([p + x for p, x in zip(pv, wv)]))) / nBa)
    log("  #499's split at c*, |U^T F(z0 + V c* + w)| / |B a| for w = " + ", ".join(f"{n} {v:.4g}"
                                                                                 for n, v in split.items())
        + f"; |w(0) - w0| {mp.nstr(nrm([x - y for x, y in zip(w00, w0)]), 3)}, |rest - w0| "
        f"{mp.nstr(nrm([x - y for x, y in zip(rest, w0)]), 3)}")
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
                           Lc=float(nrm(Lc)), q00=float(nrm(q00)), split=split)
    dump()
    # diagnostic of the S control (not judged): the control along c* at f c*, f = 1, 1/2, 1/4, where a next-order term
    # of the control falls as f relative to -1/2 S; and u.F''(V c, dw) by a mixed difference with the actual dw, which
    # separates dw's departure from its quadratic model (dw_quad) from the higher orders of the 2 dw replacement
    t0 = time.time()
    scan = []
    for f in ((mp.mpf(1), mp.mpf(1) / 2, mp.mpf(1) / 4) if mutate != "w0" else ()):     # w0: dw = 0, nothing to scan
        cf = [f * x for x in cs]
        wf = g(cf, w00)[1]
        dwf = [x - y for x, y in zip(wf, w00)]
        pvf = lin([(cj, X[j]) for j, cj in enumerate(cf)])
        G1 = proj_U(Um, sysm.F([p + x + y for p, x, y in zip(pvf, w00, dwf)]))
        G2 = proj_U(Um, sysm.F([p + x + 2 * y for p, x, y in zip(pvf, w00, dwf)]))
        dm = [y - x for x, y in zip(G1, G2)]
        dp = [-sgnS * f ** 3 * x / 2 for x in Sc]
        nc = nrm(cf)
        ec = [x / nc for x in pvf]
        mx = [t * nc for t in mixed(sysm, Um, ec, dwf, H["S"] / 10)]          # u.F''(V cf, dwf)
        dwqf = lin([(-cf[j] * cf[l] / 2, wjl[(j, l)]) for j in range(k) for l in range(k)])
        mq = [t * nc for t in mixed(sysm, Um, ec, dwqf, H["S"] / 10)]         # u.F''(V cf, dw_quad), = -1/2 S(cf^3)
        rel = lambda u, v: float(nrm([x - y for x, y in zip(u, v)]) / nrm(v))   # noqa: E731
        scan.append(dict(f=float(f), meas_pred=rel(dm, dp), meas_mixed=rel(dm, mx), mixed_pred=rel(mx, dp),
                         quad_pred=rel(mq, dp), dw_quad=rel(dwf, dwqf)))
    log("  S control along c* (diagnostic), f: (2dw change against -1/2 S(f c*), against u.F''(V f c*, dw); "
        "u.F''(V f c*, dw) against -1/2 S; u.F''(V f c*, dw_quad) against -1/2 S; dw against dw_quad): "
        + "; ".join(f"{d['f']:g}: ({d['meas_pred']:.3g}, {d['meas_mixed']:.3g}; {d['mixed_pred']:.3g}; "
                    f"{d['quad_pred']:.3g}; {d['dw_quad']:.3g})" for d in scan) + f" ({time.time() - t0:.0f} s)")
    rec["controls"]["scan"] = scan
    rec["stalls"] = dict(n=len(g.stalls), max_dg=max(g.stalls) if g.stalls else None)
    log(f"  strong solves: {g.calls}, of which {len(g.stalls)} stalled; their last correction changes g by at most "
        f"{(max(g.stalls) if g.stalls else 0):.1e} of |B a|")
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
    rec["_internal"].update(X=X, Um=Um, B=B, BL=BL, Q=Q, T=T, S=S, Lm=Lm, out=out, BLS=BLS, QLS=QLS, cstar=cs)
    return ok and ok_S, rec


# ------------------------------------------------------------------ the ray mode (#677)
RAY_T = [mp.mpf(10) ** (mp.mpf(e) / 4) for e in range(-120, -95)]    # |c| from 1e-30 to 1e-24, four per decade
RAY_JUDGED = (mp.mpf("0.999e-29"), mp.mpf("1.001e-26"))             # the issue's range; the rest is diagnostic
RESOLVED = 100                       # a part is resolved where it exceeds 100 times its floor
LEVEL = mp.mpf("1e-2")               # the departure the verdict locates, in units of |B a|


def ray_solve(sysm, X, Um, c, w_init, wref, mutate="", it_max=40):
    """w(c) for the ray mode: the chord of strong_solve, run until |dw| < 1e-60 wref, or until the smallest |dw| so far
    has not fallen by 0.3 in three steps (the chord's floor), and the iterate with the smallest |dw| is kept. A step may
    grow |dw|: at |c| >> |a| the chord's error map N = J^-1 (I - U U^T) F''(V c, .) can have a norm above 1 and a
    square far below it (#528 at c*: |dw| 5.44e-56, 5.44e-56, then 5e-82). The kept iterate counts as converged when
    its |dw| is below strong_solve's 1e-40 wref. Failure: a non-finite |dw| or one above 1e-3 max(|c|, wref), or no
    iterate below 1e-40 wref. Returns (g, w, F, the steps' (max|strong residual|, |dw|), ok, err), with
    err = |U^T [F(w - dw) - F(w)]| the change of g that the kept iterate's correction, not applied, would make: the
    point's error estimate (an estimate, since the next correction can be larger by the norm of N)."""
    pv = lin([(mp.mpf(cj), X[j]) for j, cj in enumerate(c)])
    if mutate == "w0":                                  # the control: no strong solve, the map on z0 + V c
        Fv = sysm.F(pv)
        return proj_U(Um, Fv), [mp.mpf(0)] * len(pv), Fv, [(float(amax(strong_part(Um, Fv))), 0.0)], True, mp.mpf(0)
    w = list(w_init)
    hist = []
    tol = mp.mpf("1e-60") * wref
    big = mp.mpf("1e-3") * max(nrm(c), wref)
    best, since = None, 0
    for _ in range(it_max):
        Fv = sysm.F([x + y for x, y in zip(pv, w)])
        r = strong_part(Um, Fv)
        dw = sysm.solve(r)
        nd = nrm(dw)
        hist.append((float(amax(r)), float(nd)))
        if not mp.isfinite(nd) or nd > big:
            return proj_U(Um, Fv), w, Fv, hist, False, None
        if best is None or nd < 0.3 * best[0]:
            since = 0
        else:
            since += 1
        if best is None or nd < best[0]:
            best = (nd, w, Fv, dw)
        if nd < tol or since >= 3:
            break
        w = [x - y for x, y in zip(w, dw)]
    nd, w, Fv, dw = best
    F2 = sysm.F([x + y - d for x, y, d in zip(pv, w, dw)])
    gv = proj_U(Um, Fv)
    return gv, w, Fv, hist, bool(nd < mp.mpf("1e-40") * wref), nrm([x - y for x, y in zip(proj_U(Um, F2), gv)])


def fit_part(ts, vals, floors, na, n_lead=4):
    """the leading order p and constant K of one part of the departure, vals ~ K (t / |a|)^p in units of |B a|: a least
    squares line through log vals against log t at the first n_lead resolved points (vals > RESOLVED floors, from the
    smallest t), and the local slope between every two neighbouring resolved points"""
    ok = [bool(v > RESOLVED * f) for v, f in zip(vals, floors)]
    res = [i for i, o in enumerate(ok) if o]
    local = [dict(t0=float(ts[i]), t1=float(ts[i + 1]),
                  p=float(mp.log(vals[i + 1] / vals[i]) / mp.log(ts[i + 1] / ts[i])))
             for i in range(len(ts) - 1) if ok[i] and ok[i + 1]]
    if len(res) < 3:
        return dict(n=len(res), p=None, K=None, local=local)
    lead = res[:n_lead]
    x = [mp.log(ts[i] / na) for i in lead]
    y = [mp.log(vals[i]) for i in lead]
    xm, ym = mp.fsum(x) / len(x), mp.fsum(y) / len(y)
    p = mp.fsum((u - xm) * (v - ym) for u, v in zip(x, y)) / mp.fsum((u - xm) ** 2 for u in x)
    rms = mp.sqrt(mp.fsum((v - ym - p * (u - xm)) ** 2 for u, v in zip(x, y)) / len(x))
    return dict(n=len(res), p=float(p), K=float(mp.exp(ym - p * xm)), rms=float(rms), t_from=float(ts[lead[0]]),
                t_to=float(ts[lead[-1]]), local=local)


def crossing(ts, vals, level=LEVEL):
    """the first t at which vals reaches level, by log-log interpolation between neighbouring points; else, from the
    last two points if both are positive and rising, a power-law extrapolation; else None"""
    for i, (t, v) in enumerate(zip(ts, vals)):
        if v >= level:
            if i == 0 or vals[i - 1] <= 0:
                return dict(t=float(t), how="at or below the first point")
            t1, v1 = ts[i - 1], vals[i - 1]
            lt = mp.log(t1) + (mp.log(level) - mp.log(v1)) * (mp.log(t) - mp.log(t1)) / (mp.log(v) - mp.log(v1))
            return dict(t=float(mp.exp(lt)), how="interpolated")
    if len(ts) >= 2 and 0 < vals[-2] < vals[-1]:
        p = mp.log(vals[-1] / vals[-2]) / mp.log(ts[-1] / ts[-2])
        return dict(t=float(ts[-1] * (level / vals[-1]) ** (1 / p)), how=f"extrapolated (local order {float(p):.2f})")
    return dict(t=None, how="not reached")


def true_crossing(dep, e, sgn, nBa, level=LEVEL):
    """the plant's crossing along sgn t e, from its planted departure: a scan at 40 points per decade, then bisection"""
    f = lambda t: nrm(dep([sgn * t * x for x in e])) / nBa      # noqa: E731
    grid = [mp.mpf(10) ** (mp.mpf(j) / 40) for j in range(-31 * 40, -23 * 40)]
    for t0, t1 in zip(grid, grid[1:]):
        if f(t0) < level <= f(t1):
            for _ in range(80):
                tm = mp.sqrt(t0 * t1)
                t0, t1 = (tm, t1) if f(tm) < level else (t0, tm)
            return t1
    return None


def ray_scan(sysm, bas, opts, mutate, log, rec, dump, truth=None, tensors=None):
    """g on the strong-solved manifold along the rays c = +-t e, e = a / |a| (judged) and two random unit vectors
    (diagnostic), at t in RAY_T. The departure from the affine part is dep(c) = g(c) - g(0) - B c; its even part
    (dep(t e) + dep(-t e)) / 2 holds the orders 2, 4, ..., its odd part the orders 1, 3, .... The floor of each point
    is the larger of its strong solve's error estimate (ray_solve) and the difference of g(0) between two solves from
    different starts. truth (the plant): c -> the planted departure. tensors: T - 3S at two step pairs, for the cubic
    the tensors predict along e. --mutate=B scales B by 1 + 1e-12 in dep (an order-1 term, which the plant's checks
    must catch); --mutate=w0 drops the strong solve (ray_solve)."""
    k, X, Um, B, a, Ba, nBa, na, w0 = (bas[n] for n in ("k", "X", "Um", "B", "a", "Ba", "nBa", "na", "w0"))
    wref = max(nrm(w0), mp.mpf("1e-300"))
    Bm = [[x * (1 + mp.mpf("1e-12")) for x in row] for row in B] if mutate == "B" else B
    nsolve = [0]

    def solve(c, w_init):
        nsolve[0] += 1
        return ray_solve(sysm, X, Um, c, w_init, wref, mutate)

    def dep_of(gv, g0, c):
        return [gv[m] - g0[m] - mp.fsum(Bm[m][j] * c[j] for j in range(k)) for m in range(k)]

    t0 = time.time()
    zero = [mp.mpf(0)] * k
    g00, w00, _, h00, ok00, e00 = solve(zero, w0)
    rng = np.random.default_rng(int(opts.get("seed_rays", 7)))
    # the second start is w0 (1 + 1e-5), below the failure bound 1e-3 wref. A random xi orthogonal to V failed at REF
    # (the pilot), since its square reaches the strong rows through F'' and J^-1 multiplies that along the softest
    # strong direction by 1e19
    g0b, _, _, h0b, ok0b, e0b = solve(zero, [x * (1 + mp.mpf("1e-5")) for x in w0])
    if not (ok00 and ok0b):
        log("  ray: the strong solve at c = 0 did not converge; the scan cannot run. (max|strong residual|, |dw|) per "
            "step, from w0: " + ", ".join(f"({x:.1e}, {y:.1e})" for x, y in h00) + "; from w0 (1 + 1e-5): "
            + ", ".join(f"({x:.1e}, {y:.1e})" for x, y in h0b))
        rec["ray"] = dict(ok=False, h00=h00, h0b=h0b)
        return False, rec
    floor0 = max(nrm([x - y for x, y in zip(g00, g0b)]), e00, e0b)
    eps0 = nrm([x + y for x, y in zip(g00, Ba)]) / nBa
    ga, _, Fa, ha, oka, ea = solve(a, w00)
    dep_a = dep_of(ga, g00, a)
    log(f"  ray: g(0) from w0 and from w0 (1 + 1e-5): |difference| / |B a| "
        f"{mp.nstr(nrm([x - y for x, y in zip(g00, g0b)]) / nBa, 3)}, their error estimates {mp.nstr(e00 / nBa, 3)}, "
        f"{mp.nstr(e0b / nBa, 3)}: floor0 {mp.nstr(floor0 / nBa, 3)} of |B a|; chord steps {len(h00)}, {len(h0b)}; "
        f"|g(0) + B a| / |B a| {mp.nstr(eps0, 3)}")
    log(f"  ray: at c = a: |g(a)| / |B a| {mp.nstr(nrm(ga) / nBa, 3)}, |dep(a)| / |B a| {mp.nstr(nrm(dep_a) / nBa, 3)}, "
        f"its error estimate {mp.nstr((ea or 0) / nBa, 3)}, max|F| {float(amax(Fa)):.2e}, chord steps {len(ha)} "
        f"({'converged' if oka else 'NOT converged'}) ({time.time() - t0:.0f} s)")
    out = dict(floor0=float(floor0 / nBa), eps0=float(eps0), g_a=float(nrm(ga) / nBa), dep_a=float(nrm(dep_a) / nBa),
               err_a=None if ea is None else float(ea / nBa), Fmax_a=float(amax(Fa)), ok_a=oka, rays={})
    rec["ray"] = out
    dump()
    ehat = [x / na for x in a]
    rays = [("a", ehat)]
    for name in ("r1", "r2"):
        v = rng.standard_normal(k)
        rays.append((name, [mp.mpf(float(x)) / mp.mpf(float(np.linalg.norm(v))) for x in v]))
    checks = dict(points=True, orders=True, crossings=True)
    for name, e in rays:
        t0 = time.time()
        pts = {}
        for sgn in (1, -1):
            pts[sgn] = []
            w_prev, t_prev = w00, None
            for t in RAY_T:
                c = [sgn * t * x for x in e]
                w_init = w00 if t_prev is None else [x0 + (t / t_prev) ** 2 * (wp - x0) for x0, wp in zip(w00, w_prev)]
                gv, wv, Fv, hv, okv, ev = solve(c, w_init)
                if not okv:
                    log(f"  ray {name}, sign {sgn:+d}: the strong solve at |c| {mp.nstr(t, 3)} did not converge, "
                        "(max|strong residual|, |dw|) per step " + ", ".join(f"({x:.1e}, {y:.1e})" for x, y in hv)
                        + "; the ray stops there")
                    pts[sgn].append(dict(t=t, ok=False, hist=hv))
                    break
                pts[sgn].append(dict(t=t, ok=True, dep=dep_of(gv, g00, c), err=ev, Fmax=float(amax(Fv)), hist=hv))
                w_prev, t_prev = wv, t
        rows = []
        for p, m in zip(pts[1], pts[-1]):
            if not (p["ok"] and m["ok"]):
                break
            E = [(x + y) / 2 for x, y in zip(p["dep"], m["dep"])]
            O = [(x - y) / 2 for x, y in zip(p["dep"], m["dep"])]
            row = dict(t=p["t"], dp=nrm(p["dep"]) / nBa, dm=nrm(m["dep"]) / nBa, E=nrm(E) / nBa, O=nrm(O) / nBa,
                       fl=max(p["err"], m["err"], floor0) / nBa, steps=(len(p["hist"]), len(m["hist"])),
                       Fmax=max(p["Fmax"], m["Fmax"]))
            if truth is not None:
                dpt, dmt = truth([p["t"] * x for x in e]), truth([-m["t"] * x for x in e])
                Et = [(x + y) / 2 for x, y in zip(dpt, dmt)]
                Ot = [(x - y) / 2 for x, y in zip(dpt, dmt)]
                for part, meas, tru in (("E", E, Et), ("O", O, Ot)):
                    nt = nrm(tru) / nBa
                    err = nrm([x - y for x, y in zip(meas, tru)]) / nBa
                    row[part + "_true"] = nt
                    row[part + "_err"] = err / max(nt, RESOLVED * row["fl"])
            rows.append(row)
        ts = [r["t"] for r in rows]
        fits = {part: fit_part(ts, [r[part] for r in rows], [r["fl"] for r in rows], na) for part in ("E", "O")}
        cross = {sgn: crossing([q["t"] for q in pts[sgn] if q["ok"]], [nrm(q["dep"]) / nBa for q in pts[sgn] if q["ok"]])
                 for sgn in (1, -1)}
        # the same for g(c) - B (c - a) = dep(c) + g(0) + B a, which differs from dep by the start's constant eps0
        e0v = [x + y for x, y in zip(g00, Ba)]
        cross_full = {sgn: crossing([q["t"] for q in pts[sgn] if q["ok"]],
                                    [nrm([x + y for x, y in zip(q["dep"], e0v)]) / nBa for q in pts[sgn] if q["ok"]])
                      for sgn in (1, -1)}
        judged = [r for r in rows if RAY_JUDGED[0] <= r["t"] <= RAY_JUDGED[1]]
        stop = {sgn: next((float(q["t"]) for q in pts[sgn] if not q["ok"]), None) for sgn in (1, -1)}
        log(f"  ray {name} (e . a / |a| {mp.nstr(dot(e, ehat), 3)}): t, t/|a|, |dep(+t e)|, |dep(-t e)|, even part, odd "
            "part, floor (of |B a|), chord steps (+, -), max|F|" + (", even and odd against the planted" if truth else ""))
        for r in rows:
            log(f"    {mp.nstr(r['t'], 3):>8} {float(r['t'] / na):9.3g} {float(r['dp']):10.3e} {float(r['dm']):10.3e} "
                f"{float(r['E']):10.3e} {float(r['O']):10.3e} {float(r['fl']):9.2e} {r['steps'][0]:>2},{r['steps'][1]:<2} "
                f"{r['Fmax']:9.2e}" + (f" {float(r['E_err']):8.1e} {float(r['O_err']):8.1e}" if truth else ""))
        for part, f in fits.items():
            log(f"    {'even' if part == 'E' else 'odd'} part: " + (
                f"leading order {f['p']:.3f} (rms {f['rms']:.1e} in ln), K {f['K']:.3e} |B a| (t/|a|)^p, from the first "
                f"{min(4, f['n'])} of {f['n']} resolved points, t {f['t_from']:.2e}..{f['t_to']:.2e}" if f["p"] is not None
                else f"not resolved ({f['n']} points above {RESOLVED} times the floor)")
                + ("; local orders " + ", ".join(f"{d['p']:.2f}" for d in f["local"]) if f["local"] else ""))
        for sgn in (1, -1):
            cr, cf = cross[sgn], cross_full[sgn]
            log(f"    sign {sgn:+d}: |dep| reaches {mp.nstr(LEVEL, 1)} |B a| at " + (
                f"|c| {cr['t']:.3e}, |c|/|a| {cr['t'] / float(na):.4g} ({cr['how']})" if cr["t"] is not None
                else cr["how"]) + "; |g(c) - B (c - a)| at " + (
                f"|c|/|a| {cf['t'] / float(na):.4g} ({cf['how']})" if cf["t"] is not None else cf["how"])
                + ("" if stop[sgn] is None else f"; the ray stopped at |c| {stop[sgn]:.2e}"))
        rr = dict(e=[float(x) for x in e], e_dot_ahat=float(dot(e, ehat)), stop=stop,
                  rows=[{kk: (float(v) if isinstance(v, mp.mpf) else v) for kk, v in r.items()} for r in rows],
                  fits=fits, cross={str(s): c for s, c in cross.items()},
                  cross_full={str(s): c for s, c in cross_full.items()},
                  judged_ok=bool(len(judged) == len([t for t in RAY_T if RAY_JUDGED[0] <= t <= RAY_JUDGED[1]])),
                  points=[dict(sgn=sgn, t=float(q["t"]), ok=q["ok"], dep=[float(x) for x in q.get("dep", [])],
                               err=None if q.get("err") is None else float(q["err"] / nBa), hist=q["hist"])
                          for sgn in (1, -1) for q in pts[sgn]])
        if tensors is not None:
            # the cubic the tensors predict along e: (T - 3S)(e, e, e) / 6 |a|^3 / |B a|, at both step pairs
            pr = [nrm([mp.fsum(C[m][i][j][l] * e[i] * e[j] * e[l] for i in range(k) for j in range(k)
                               for l in range(k)) for m in range(k)]) / 6 * na ** 3 / nBa for C in tensors]
            rr["K3_tensors"] = [float(x) for x in pr]
            log(f"    the tensors' cubic along e, |(T - 3S)(e, e, e)| / 6 |a|^3 / |B a|: {mp.nstr(pr[0], 3)} (smaller "
                f"steps), {mp.nstr(pr[1], 3)} (larger)")
        if truth is not None:
            pe = all(r["E_err"] < 1e-3 and r["O_err"] < 1e-3 for r in rows)
            po = bool(fits["E"]["p"] is not None and abs(fits["E"]["p"] - 4) < 0.05 and fits["O"]["p"] is not None
                      and abs(fits["O"]["p"] - 5) < 0.05)
            tc = {sgn: true_crossing(truth, e, sgn, nBa) for sgn in (1, -1)}
            pc = all(tc[s] is not None and cross[s]["t"] is not None and abs(cross[s]["t"] / float(tc[s]) - 1) < 1e-2
                     for s in (1, -1))
            checks["points"] &= pe
            checks["orders"] &= po
            checks["crossings"] &= pc
            rr["planted"] = dict(points=pe, orders=po, crossings=pc, tc={str(s): None if v is None else float(v)
                                                                       for s, v in tc.items()},
                                 worst=float(max(max(r["E_err"], r["O_err"]) for r in rows)) if rows else None)
            log(f"    planted: worst point error {rr['planted']['worst']:.1e} (bound 1e-3): {'pass' if pe else 'FAIL'}; "
                f"orders 4 and 5 to 0.05: {'pass' if po else 'FAIL'}; crossings against the planted "
                + ", ".join(f"{float(tc[s] / na) if tc[s] else float('nan'):.4g}" for s in (1, -1))
                + f" |a| (bound 1e-2): {'pass' if pc else 'FAIL'}")
        log(f"    ({time.time() - t0:.0f} s)")
        out["rays"][name] = rr
        dump()
    out["solves"] = nsolve[0]
    a_ray = out["rays"]["a"]
    out["verdict"] = dict(judged_converged=a_ray["judged_ok"],
                          order_fitted=bool(any(a_ray["fits"][p]["p"] is not None for p in ("E", "O"))))
    if truth is not None:
        out["planted_checks"] = checks
        ok = all(checks.values())
    else:
        ok = out["verdict"]["judged_converged"] and out["verdict"]["order_fitted"]
    log(f"  ray: {nsolve[0]} strong solves; along a / |a|, every judged point converged: "
        f"{'yes' if out['verdict']['judged_converged'] else 'NO'}; an order fitted: "
        f"{'yes' if out['verdict']['order_fitted'] else 'NO'}"
        + ("" if truth is None else "; planted checks " + ", ".join(f"{n} {'pass' if v else 'FAIL'}"
                                                                    for n, v in checks.items())))
    dump()
    return ok, rec


def raysum(paths):
    """the summary of ray runs (their --json files): per start and ray, the fitted orders and constants, the crossing,
    and estimates of the next orders. If the odd part is l t + s t^3 + ..., the relative change rho of |O|/t from the
    smallest to the largest resolved t estimates |s| t_max^2 / |l| as rho for s along l, and as sqrt(2 rho) for s
    orthogonal to l (the norm then changes at second order); the larger is taken. Likewise |E|/t^2 for the even
    part's quartic. The norms |E| and |O| are the run's, formed at 140 digits and stored with 16 (the stored dep
    vectors are not used: where one part exceeds the other by 16 digits their half sums cancel to rounding). So rho
    has a floor near 1e-16, and sqrt(2 rho) near 1e-8. These are estimates, not bounds, and they assume the leading
    orders are 1 and 2. In units of |B a| (t/|a|)^p: K3 for the cubic, K4 for the quartic; K3 also as
    |(T - 3S)(e, e, e)| / max|T| where the run measured T (with 6 K3 |B a| / |a|^3)."""
    for path in paths:
        with open(path) as fh:
            r = json.load(fh)
        ray = r.get("ray") or {}
        na, nBa = r["a_norm"], r["Ba"]
        print(f"{os.path.basename(path)}: {r.get('label', r.get('dz0', 'planted'))}; |a| {na:.4e}, |B a| {nBa:.4e}, "
              f"eps0 {ray.get('eps0', float('nan')):.3e}, |g(a)| {ray.get('g_a', float('nan')):.3e}, |dep(a)| "
              f"{ray.get('dep_a', float('nan')):.3e}, max|F| at a {ray.get('Fmax_a', float('nan')):.2e}, floor0 "
              f"{ray.get('floor0', float('nan')):.2e} (of |B a|)"
              + (f"; max|L| |a| / |B a| {r['L']['max'] * na / nBa:.2g} (L's h against h/10 {r['L']['h_h10']:.2g})"
                 if isinstance(r.get("L"), dict) else ""))       # an int L is the system's size
        for name, rr in (ray.get("rays") or {}).items():
            rows = rr["rows"]
            nb = {}
            for part, p0 in (("O", 1), ("E", 2)):
                rt = [row for row in rows if row[part] > RESOLVED * row["fl"]]
                if len(rt) >= 2:
                    r0, r1 = rt[0], rt[-1]
                    k0, k1 = r0[part] / r0["t"] ** p0, r1[part] / r1["t"] ** p0
                    rho = max(abs(k1 / k0 - 1), 2.2e-16)            # not below the stored digits' rounding
                    # |next coefficient| in |B a| (t/|a|)^(p0+2): max(rho, sqrt(2 rho)) |lead| / t_max^2
                    nb[part] = max(rho, (2 * rho) ** 0.5) * k1 / r1["t"] ** 2 * na ** (p0 + 2)
            fits = rr["fits"]
            line = f"  ray {name}:"
            for part in ("E", "O"):
                f = fits[part]
                line += (f" {'even' if part == 'E' else 'odd'} order {f['p']:.3f}, K {f['K']:.3e}" if f["p"] is not None
                         else f" {'even' if part == 'E' else 'odd'} not resolved") + ";"
            if "O" in nb:
                line += f" cubic K3 <~ {nb['O']:.1e}"
                if r.get("T"):
                    line += f" (|(T - 3S)(e,e,e)| / max|T| <~ {6 * nb['O'] * nBa / na ** 3 / r['T']['max']:.1e})"
            if "E" in nb:
                line += f", quartic K4 <~ {nb['E']:.1e}"
            cr = rr["cross"]["1"]
            line += (f"; 1e-2 |B a| at |c|/|a| {cr['t'] / na:.3g} ({cr['how']})" if cr["t"] is not None
                     else f"; 1e-2 |B a| {cr['how']}")
            if fits["E"]["p"] is not None:
                # where the even part alone, K (t/|a|)^p, would reach 1e-2 (an extrapolation of its fit)
                line += f", the even part alone at {(1e-2 / fits['E']['K']) ** (1 / fits['E']['p']):.2g}"
            if rr.get("K3_tensors"):
                line += f"; tensors' K3 {rr['K3_tensors'][0]:.2e} ({rr['K3_tensors'][1]:.2e} at the larger steps)"
            if any(rr["stop"].values()):
                line += f"; stopped at {rr['stop']}"
            print(line)


# ------------------------------------------------------------------ the weak rows (#694)
ROW_FIELDS = (("p", 1), ("m", 1), ("a", 0), ("c", 0))     # the fields of each residual block, with parity (1: odd)
H_ROWS = (1e-32, 1e-33)          # F''(v_j, v_l) per row at these steps (#528's Q steps); the second is judged
AFFINE = mp.mpf("1e-10")         # affine to 1e-10: q_r below 1e-10 of the largest q and of the row's own |J_r|
SUPPORT, REFUTE = 0.99, 0.5      # the issue's verdict on the share of |U| on the affine rows


def harm_basis(M, odd):
    """the orthonormal harmonic basis of a field's M values on the half grid theta_j = pi j / M (multishoot's
    Parity::Odd and Parity::Even grids): an odd field has cos and sin of k = 1, 3, ..., M - 1, each times sqrt(2 / M);
    an even one cos of k = 0 and of the Nyquist harmonic k = M, each times 1 / sqrt(M), and cos and sin of
    k = 2, 4, ..., M - 2, times sqrt(2 / M). [(label, M mpf values)]"""
    th = [mp.pi * j / M for j in range(M)]
    s2, s1 = mp.sqrt(mp.mpf(2) / M), 1 / mp.sqrt(M)
    pair = lambda k: [(f"k{k}c", [s2 * mp.cos(k * t) for t in th]), (f"k{k}s", [s2 * mp.sin(k * t) for t in th])]  # noqa: E731
    if odd:
        return [b for k in range(1, M, 2) for b in pair(k)]
    return [("k0c", [s1] * M)] + [b for k in range(2, M, 2) for b in pair(k)] + [(f"k{M}c", [s1 * (-1) ** j
                                                                                             for j in range(M)])]


def harm_ortho(M):
    """max |H H^T - I| of harm_basis at 140 digits, over both parities"""
    worst = mp.mpf(0)
    for odd in (0, 1):
        b = [v for _, v in harm_basis(M, odd)]
        for i in range(M):
            for j in range(i, M):
                worst = max(worst, abs(dot(b[i], b[j]) - (1 if i == j else 0)))
    return worst


class Harm:
    """an orthogonal change of rows (or of columns) to (field, harmonic): the fields of M values each, in order, every
    one expanded on harm_basis of its parity, then one single entry (the phase row, or the column Delta). A row here is
    labelled block.field.harmonic, e.g. horizon.p.k5c. --mutate=parity (a control of the plant) expands each field on
    the basis of the other parity, which keeps the change orthogonal but labels every row wrongly."""

    def __init__(self, fields, single, M, mutate=""):
        bases = {odd: harm_basis(M, odd != (mutate == "parity")) for odd in (0, 1)}
        self.labels, self.rows, self.M = [], [], M
        for f, (name, odd) in enumerate(fields):
            for lab, vec in bases[odd]:
                self.labels.append(f"{name}.{lab}")
                self.rows.append((M * f, vec))
        self.labels.append(single)
        self.rows.append((M * len(fields), None))
        self.n = len(self.rows)

    def __call__(self, v):
        return [v[s] if vec is None else mp.fsum(x * v[s + j] for j, x in enumerate(vec)) for s, vec in self.rows]

    def matrix64(self):
        H = np.zeros((self.n, self.n))
        for i, (s, vec) in enumerate(self.rows):
            if vec is None:
                H[i, s] = 1.0
            else:
                H[i, s:s + self.M] = [float(x) for x in vec]
        return H


def row_harm(L, M, mutate=""):
    """the residual's rows (hp/src/model.rs): blocks b = 0..K of 4M rows, each the fields p, m, a, c on the half grid
    (block 0 the centre series at node 0 minus node 0, block b the Taylor step node b-1 -> node b minus node b, block K
    the horizon series minus node K-1), then the phase row"""
    K1 = (L - 1) // (4 * M)
    assert 4 * M * K1 + 1 == L, f"L {L} is not 4 M (K + 1) + 1 with M {M}"
    names = ["centre"] + [f"node{b}" for b in range(1, K1 - 1)] + ["horizon"]
    return Harm([(f"{bn}.{fn}", odd) for bn in names for fn, odd in ROW_FIELDS], "phase", M, mutate)


def col_harm(L, M):
    """the unknowns (near_null.layout): p1, c0 | node_i: p, m, a, c | m_h, a_h | Delta"""
    K = (L - 1) // (4 * M) - 1
    return Harm([("p1", 1), ("c0", 0)] + [(f"node{i}.{fn}", odd) for i in range(K) for fn, odd in ROW_FIELDS]
                + [("m_h", 1), ("a_h", 0)], "Delta", M)


class RowPlanted:
    """the rows mode's plant (#694): a system on the residual's row layout with M = 4 and K = 1 (the centre and horizon
    blocks of four fields each, and the phase row; n = 33), whose rows are given in its own harmonic basis (written
    here independently of harm_basis) and whose unknowns are rotated by a random orthogonal P_R at 140 digits; z0 = 0.
    In the harmonic rows,
        G(zeta) = beta + D zeta + sum over the nonlinear rows r of e_r [1/2 q_r (b_r . zeta)^2 + 1/6 t_r (b'_r . zeta)^3],
    zeta = P_R^T z, with D = U_c Sigma E_w^T + U_c' Lambda E_s^T and [U_c U_c'] orthogonal. The planted weak left
    vectors are u_m = sqrt(1 - eps_m^2) a_m + eps_m n_m, a_m an orthonormal basis of the four affine rows centre.m.k1c,
    horizon.p.k3c, horizon.p.k3s and phase (randomly rotated), n_m orthonormal on the nonlinear rows, eps 0.1, 0.15,
    0.2, 0.25: their share on the affine rows is 1 - mean(eps^2) = 0.96625. A fifth affine row, horizon.a.k0c, carries
    no U. Sigma is the D 242.5 starts' weak sigma, Lambda 29 strong singular values log-spaced from 1e-19 to 3e34;
    |q_r| from 1e28 to 1e33 and |t_r| from 1e30 to 1e34, log-uniform. The affine rows have no nonlinear term, so they
    are affine in z exactly; --mutate=quadA adds 1/2 q_A (b . zeta)^2, q_A = 1e-6 max|q|, to the four that carry U."""

    A_U = ("centre.m.k1c", "horizon.p.k3c", "horizon.p.k3s", "phase")
    A_0 = ("horizon.a.k0c",)
    EPS = (0.1, 0.15, 0.2, 0.25)

    def __init__(self, seed, log, mutate=""):
        M, K1, k = 4, 2, 4
        n = 4 * M * K1 + 1
        self.M, self.L, self.k = M, n, k
        rng = np.random.default_rng(seed + 694)
        rnd = lambda: [mp.mpf(float(x)) for x in rng.standard_normal(n)]   # noqa: E731
        # the plant's own grid functions: Hg[i] is harmonic row i on the grid
        self.labels, self.Hg = [], []
        for b, bn in enumerate(("centre", "horizon")):
            for f, (fn, odd) in enumerate(ROW_FIELDS):
                ks = ([(h, cs) for h in range(1, M, 2) for cs in "cs"] if odd else
                      [(0, "c")] + [(h, cs) for h in range(2, M, 2) for cs in "cs"] + [(M, "c")])
                for h, cs in ks:
                    wt = mp.sqrt((1 if h in (0, M) else 2) / mp.mpf(M))
                    v = [mp.mpf(0)] * n
                    for j in range(M):
                        v[4 * M * b + M * f + j] = wt * (mp.cos(h * mp.pi * j / M) if cs == "c" else mp.sin(h * mp.pi * j / M))
                    self.labels.append(f"{bn}.{fn}.k{h}{cs}")
                    self.Hg.append(v)
        self.labels.append("phase")
        self.Hg.append([mp.mpf(0)] * (n - 1) + [mp.mpf(1)])
        iA = [self.labels.index(x) for x in self.A_U]
        self.affine = set(self.A_U + self.A_0)
        iN = [i for i, x in enumerate(self.labels) if x not in self.affine]
        unit = lambda i: [mp.mpf(1 if r == i else 0) for r in range(n)]   # noqa: E731
        Q4 = mgs([[mp.mpf(float(x)) for x in rng.standard_normal(k)] for _ in range(k)])[0]
        aa = [lin([(Q4[m][i], unit(iA[i])) for i in range(k)]) for m in range(k)]
        nn = mgs([[x if r in iN else mp.mpf(0) for r, x in enumerate(rnd())] for _ in range(k)])[0]
        self.Uc = [[mp.sqrt(1 - mp.mpf(e) ** 2) * x + mp.mpf(e) * y for x, y in zip(aa[m], nn[m])]
                   for m, e in enumerate(self.EPS)]
        self.Ucp = mgs(self.Uc + [rnd() for _ in range(n - k)])[0][k:]
        self.sig = [mp.mpf(x) for x in ("2.1e-27", "2.2e-27", "3.2e-26", "7.1e-26")]
        self.lam = [mp.mpf(10) ** (mp.mpf(-19) + (mp.log10(mp.mpf("3e34")) + 19) * s / (n - k - 1))
                    for s in range(n - k)]
        self.PR = mgs([rnd() for _ in range(n)])[0]          # z = sum_i zeta_i PR[i]
        self.nl = {}
        for i in iN:
            self.nl[i] = (mp.mpf(float(rng.choice([-1, 1]) * 10 ** rng.uniform(28, 33))), rnd(),
                          mp.mpf(float(rng.choice([-1, 1]) * 10 ** rng.uniform(30, 34))), rnd())
        qmax = max(abs(v[0]) for v in self.nl.values())
        if mutate == "quadA":
            for i in iA:
                self.nl[i] = (qmax * mp.mpf("1e-6"), rnd(), mp.mpf(0), [mp.mpf(0)] * n)
        # the start: beta = -D zeta_a, so that the first Newton step is P_R zeta_a, |a| 8.3e-30, |w0| ~ 1e-33
        ya = [mp.mpf(float(x)) for x in rng.standard_normal(k)]
        ya = [x * mp.mpf("8.3e-30") / nrm(ya) for x in ya]
        za = ya + [mp.mpf(float(x)) * mp.mpf("1e-33") for x in rng.standard_normal(n - k)]
        self.beta = [mp.mpf(0)] * n
        self.beta = [-x for x in self.Dz(za)]
        self.F0max = float(amax(self.F()))
        self.dz0 = [-x for x in self.solve(self.F())]
        self.share_aff_true = 1 - mp.fsum(mp.mpf(e) ** 2 for e in self.EPS) / k
        self.label = f"rows plant (seed {seed}): n {n}, k {k}" + (f", --mutate={mutate}" if mutate else "")
        self.meta = dict(seed=seed, n=n, k=k, F0max=self.F0max, share_aff_true=float(self.share_aff_true))
        log(f"{self.label}; start max|F| {self.F0max:.3e}; planted affine rows {', '.join(sorted(self.affine))}; the "
            f"planted share of |U| on them {mp.nstr(self.share_aff_true, 8)}")

    def Dz(self, zeta):
        k = self.k
        return [mp.fsum([self.sig[m] * zeta[m] * self.Uc[m][r] for m in range(k)]
                        + [self.lam[s] * zeta[k + s] * self.Ucp[s][r] for s in range(self.L - k)])
                for r in range(self.L)]

    def G(self, zeta):
        g = [b + d for b, d in zip(self.beta, self.Dz(zeta))]
        for i, (q, b, t, b2) in self.nl.items():
            g[i] += q * dot(b, zeta) ** 2 / 2 + t * dot(b2, zeta) ** 3 / 6
        return g

    def zeta(self, p):
        return [dot(c, p) for c in self.PR]

    def F(self, p=None):
        g = self.G([mp.mpf(0)] * self.L if p is None else self.zeta(p))
        return [mp.fsum(g[i] * self.Hg[i][r] for i in range(self.L)) for r in range(self.L)]

    def solve(self, r, transpose=False):
        """J^-1 r = P_R D^-1 Hg^T r and J^-T r = Hg D^-T P_R^T r, with D^-1 y = [Sigma^-1 U_c^T y; Lambda^-1 U_c'^T y]
        and D^-T x = U_c Sigma^-1 x_w + U_c' Lambda^-1 x_s"""
        k, n = self.k, self.L
        d = self.sig + self.lam
        W = self.Uc + self.Ucp
        if transpose:
            x = self.zeta(r)
            y = lin([(x[i] / d[i], W[i]) for i in range(n)])
            return lin([(y[i], self.Hg[i]) for i in range(n)])
        y = [dot(h, r) for h in self.Hg]
        x = [dot(W[i], y) / d[i] for i in range(n)]
        return lin([(x[i], self.PR[i]) for i in range(n)])

    def solve64(self, b, transpose=False):
        return np.array([float(x) for x in self.solve([mp.mpf(float(v)) for v in b], transpose)])

    def row_norms(self, RH):
        n, k = self.L, self.k
        Hg = np.array([[float(x) for x in v] for v in self.Hg]).T            # grid <- harmonic rows
        D = np.array([[float(s * u[r]) for s, u in zip(self.sig + self.lam, self.Uc + self.Ucp)] for r in range(n)])
        PR = np.array([[float(x) for x in v] for v in self.PR]).T
        return [float(x) for x in np.linalg.norm(RH.matrix64() @ (Hg @ D @ PR.T), axis=1)]

    # the truth, in the plant's harmonic rows (the code's must equal them, row for row)
    def share_true(self):
        return [mp.fsum(u[r] ** 2 for u in self.Uc) / self.k for r in range(self.L)]

    def q_true(self, X):
        z = [self.zeta(x) for x in X]
        out = [mp.mpf(0)] * self.L
        for i, (q, b, _, _) in self.nl.items():
            bv = [dot(b, zz) for zz in z]
            out[i] = abs(q) * max(abs(bv[j] * bv[l]) for j in range(self.k) for l in range(j, self.k))
        return out

    def along_true(self, ez):
        """per row: F''(e, e), F'''(e, e, e) and F''(e, w_ee), w_ee = J^-1 (I - U U^T) F''(e, e) (U the planted)"""
        ze = self.zeta(ez)
        n, k = self.L, self.k
        Fee, T = [mp.mpf(0)] * n, [mp.mpf(0)] * n
        for i, (q, b, t, b2) in self.nl.items():
            Fee[i] = q * dot(b, ze) ** 2
            T[i] = t * dot(b2, ze) ** 3
        zw = [mp.mpf(0)] * k + [dot(self.Ucp[s], Fee) / self.lam[s] for s in range(n - k)]
        S = [mp.mpf(0)] * n
        for i, (q, b, _, _) in self.nl.items():
            S[i] = q * dot(b, ze) * dot(b, zw)
        return Fee, T, S


def d3_full(sysm, ev, h):
    """F'''(e, e, e) as a full vector, the antisymmetric four-point difference at step h (third_tensor's D3)"""
    acc = None
    for wgt, t in ((1, 2), (-2, 1), (2, -1), (-1, -2)):
        Fv = sysm.F([mp.mpf(t) * h * x for x in ev])
        acc = [wgt * x for x in Fv] if acc is None else [a + wgt * x for a, x in zip(acc, Fv)]
    return [x / (2 * mp.mpf(h) ** 3) for x in acc]


def mixed_full(sysm, ev, w, h):
    """F''(e, w) as a full vector: mixed central differences with the step h along e (a unit vector) and along
    w / |w| (mixed's, without the projection on U)"""
    nw = nrm(w)
    ew = [t / nw for t in w]
    acc = None
    for sx, sw, wgt in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        Fv = sysm.F(lin([(mp.mpf(sx) * h, ev), (mp.mpf(sw) * h, ew)]))
        acc = [wgt * t for t in Fv] if acc is None else [a + wgt * t for a, t in zip(acc, Fv)]
    return [t * nw / (4 * mp.mpf(h) ** 2) for t in acc]


def rows_scan(sysm, bas, opts, mutate, log, rec, dump, truth=None):
    """where the weak left vectors U live, and whether those rows are affine (docstring, `rows`). truth: the plant"""
    k, X, Um, a, na, nBa = (bas[n] for n in ("k", "X", "Um", "a", "na", "nBa"))
    t0 = time.time()
    M, L = sysm.M, sysm.L
    RH, CH = row_harm(L, M, mutate), col_harm(L, M)
    n = RH.n
    orth = harm_ortho(M)
    Uh = [RH(u) for u in Um]
    share = [mp.fsum(Uh[m][r] ** 2 for m in range(k)) / k for r in range(n)]
    Vh = [CH(x) for x in X]
    vshare = [mp.fsum(Vh[m][r] ** 2 for m in range(k)) / k for r in range(n)]
    jn = sysm.row_norms(RH)
    log(f"  rows: {n} rows in (block, field, harmonic), the change of rows orthonormal to {mp.nstr(orth, 3)}; shares "
        f"of |U| sum to {mp.nstr(mp.fsum(share), 20)}, of |V| to {mp.nstr(mp.fsum(vshare), 20)} ({time.time() - t0:.0f} s)")
    # F''(v_j, v_l) per row at the two steps; q_r the largest over j <= l
    qs = []
    for hh in H_ROWS:
        t1 = time.time()
        F2 = second_diffs(sysm, X, hh)
        rr = [RH(v) for key, v in F2.items() if key[0] <= key[1]]
        qs.append([max(abs(x[r]) for x in rr) for r in range(n)])
        log(f"  F''(v_j, v_l) per row at h {hh:.0e}: max q {mp.nstr(max(qs[-1]), 4)} ({time.time() - t1:.0f} s)")
    q = qs[-1]

    def classify(qv):
        qm = max(qv)
        glob = [bool(qv[r] <= AFFINE * qm) for r in range(n)]
        own = [bool(qv[r] <= AFFINE * jn[r]) for r in range(n)]
        return [g and o for g, o in zip(glob, own)], glob, own
    aff, glob, own = classify(q)
    aff_h = classify(qs[0])[0]
    sh = lambda sel: mp.fsum(share[r] for r in range(n) if sel[r])    # noqa: E731
    share_aff, share_glob, share_own, share_aff_h = sh(aff), sh(glob), sh(own), sh(aff_h)
    G = mp.matrix(k, k)
    for m in range(k):
        for m2 in range(k):
            G[m, m2] = mp.fsum(Uh[m][r] * Uh[m2][r] for r in range(n) if aff[r])
    lam_min = min(mp.eigsy(G)[0])
    verdict = lambda s: "supports" if s > SUPPORT else ("refutes" if s < REFUTE else "between")     # noqa: E731
    frozen = list(getattr(sysm, "frozen", [])) if truth is None else sorted(truth.affine)
    fidx = [RH.labels.index(x) if x in RH.labels else None for x in frozen]
    frozen_ok = bool(frozen and all(i is not None and aff[i] for i in fidx))
    share_frozen = mp.fsum(share[i] for i in fidx if i is not None)
    log(f"  share of |U| on the rows affine to 1e-10 (q_r <= 1e-10 max q and <= 1e-10 |J_r|, at h {H_ROWS[-1]:.0e}): "
        f"{mp.nstr(share_aff, 10)} ({sum(aff)} of {n} rows): {verdict(share_aff)}; at h {H_ROWS[0]:.0e} "
        f"{mp.nstr(share_aff_h, 10)} ({verdict(share_aff_h)}); against the largest q alone {mp.nstr(share_glob, 10)}, "
        f"against |J_r| alone {mp.nstr(share_own, 10)}; the worst weak left vector's share (smallest eigenvalue of "
        f"U_A^T U_A) {mp.nstr(lam_min, 6)}")
    log(f"  the rows linear by construction ({len(frozen)}: {frozen[0]} .. {frozen[-1]}): share of |U| "
        f"{mp.nstr(share_frozen, 10)}; all classified affine: {'yes' if frozen_ok else 'NO'}; their largest q "
        f"{mp.nstr(max((q[i] for i in fidx if i is not None), default=mp.mpf(0)), 3)}")
    # by block and field
    groups = {}
    for r, lab in enumerate(RH.labels):
        g = ".".join(lab.split(".")[:2])
        d = groups.setdefault(g, dict(share=mp.mpf(0), share_aff=mp.mpf(0), qmax=mp.mpf(0), qJ=mp.mpf(0), naff=0, n=0))
        d["share"] += share[r]
        d["share_aff"] += share[r] if aff[r] else 0
        d["qmax"] = max(d["qmax"], q[r])
        d["qJ"] = max(d["qJ"], q[r] / jn[r])
        d["naff"] += aff[r]
        d["n"] += 1
    log("  by block and field: share of |U|, of it on affine rows, max q, max q/|J_r|, affine rows / rows")
    for g, d in groups.items():
        log(f"    {g:<12} {float(d['share']):10.3e} {float(d['share_aff']):10.3e} {float(d['qmax']):10.3e} "
            f"{float(d['qJ']):10.3e} {d['naff']:>3}/{d['n']}")

    def row_line(r):
        rat = qs[0][r] / q[r] if q[r] > 0 else mp.inf
        return (f"    {RH.labels[r]:<18} share {float(share[r]):.3e}, q {float(q[r]):.3e}, q/|J_r| "
                f"{float(q[r] / jn[r]):.2e}, |J_r| {jn[r]:.2e}, q(1e-32)/q(1e-33) {float(rat):.3g}"
                f"{'' if aff[r] else ', NOT affine'}")
    order = sorted(range(n), key=lambda r: -share[r])
    log("  the rows of U, largest share first:")
    for r in order[:12]:
        log(row_line(r))
    rest = [r for r in order if not aff[r]]
    log(f"  the rest, on rows not affine ({len(rest)} rows, share {mp.nstr(1 - share_aff, 4)}), largest share first:")
    for r in rest[:12]:
        log(row_line(r))
    rest_groups = {}
    for r in rest:
        g = ".".join(RH.labels[r].split(".")[:2])
        rest_groups[g] = rest_groups.get(g, mp.mpf(0)) + share[r]
    log("  the rest by block and field: " + ", ".join(f"{g} {float(v):.2e}" for g, v in
                                                       sorted(rest_groups.items(), key=lambda x: -x[1])[:10]))
    vorder = sorted(range(n), key=lambda r: -vshare[r])
    log("  V by unknowns (diagnostic), largest share first: " + ", ".join(f"{CH.labels[r]} {float(vshare[r]):.3f}"
                                                                         for r in vorder[:10]))
    out = dict(n=n, orth=float(orth), labels=RH.labels, share=[float(x) for x in share],
               q=[[float(x) for x in qq] for qq in qs], Jr=jn, affine=aff, affine_h32=aff_h,
               share_aff=float(share_aff), share_aff_h32=float(share_aff_h), share_glob=float(share_glob),
               share_own=float(share_own), lam_min=float(lam_min), verdict=verdict(share_aff),
               verdict_h32=verdict(share_aff_h), frozen=frozen, frozen_ok=frozen_ok, share_frozen=float(share_frozen),
               groups={g: {kk: (float(v) if isinstance(v, mp.mpf) else v) for kk, v in d.items()} for g, d in groups.items()},
               rest_groups={g: float(v) for g, v in rest_groups.items()},
               vlabels=CH.labels, vshare=[float(x) for x in vshare])
    rec["rows"] = out
    dump()

    # 3S = T along e = a / |a|: per row F''(e, e), T(e) = F'''(e, e, e) and S(e) = F''(e, w_ee)
    t1 = time.time()
    e = [x / na for x in a]
    ev = lin([(e[j], X[j]) for j in range(k)])
    F0 = sysm.F()
    Tf = [d3_full(sysm, ev, hh) for hh in (H["T"], H["T"] / 10)]
    Fee = []
    for hh in (H["F"] / 10, H["F"] / 100):
        Fp, Fm = sysm.F([hh * x for x in ev]), sysm.F([-hh * x for x in ev])
        Fee.append([(p_ - 2 * z_ + m_) / mp.mpf(hh) ** 2 for p_, z_, m_ in zip(Fp, F0, Fm)])
    wee = [sysm.solve(strong_part(Um, f)) for f in Fee]
    dwee = nrm([x - y for x, y in zip(wee[0], wee[1])]) / nrm(wee[1])
    Sf = [mixed_full(sysm, ev, wee[1], hh) for hh in (H["S"], H["S"] / 10)]
    rel = lambda u, v: nrm([x - y for x, y in zip(u, v)]) / nrm(v)    # noqa: E731
    dT = rel(proj_U(Um, Tf[0]), proj_U(Um, Tf[1]))
    dS = rel(proj_U(Um, Sf[0]), proj_U(Um, Sf[1]))
    Th, Sh, Qh = RH(Tf[1]), RH(Sf[1]), RH(Fee[1])
    C3h = [t - 3 * s for t, s in zip(Th, Sh)]
    UT = proj_U(Um, Tf[1])
    UT_rows = [mp.fsum(Uh[m][r] * Th[r] for r in range(n)) for m in range(k)]
    # relative to the size U^T T(e) would have without cancellation between rows
    tr_err = nrm([x - y for x, y in zip(UT_rows, UT)]) / nrm([mp.fsum(abs(Uh[m][r] * Th[r]) for r in range(n))
                                                             for m in range(k)])
    UC3 = [mp.fsum(Uh[m][r] * C3h[r] for r in range(n)) for m in range(k)]
    c3rel = nrm(UC3) / nrm(UT)
    h_ok = bool(dwee < 1e-2 and dT < 1e-2 and dS < 1e-2)
    log(f"  along e = a / |a|: w_ee = J^-1 (I - U U^T) F''(e, e), |w_ee| {mp.nstr(nrm(wee[1]), 3)}, h {H['F'] / 10:.0e} "
        f"against {H['F'] / 100:.0e} {mp.nstr(dwee, 3)}; U^T T(e) at h {H['T']:.0e} against {H['T'] / 10:.0e} "
        f"{mp.nstr(dT, 3)}; U^T S(e) at h {H['S']:.0e} against {H['S'] / 10:.0e} {mp.nstr(dS, 3)} (bound 1e-2 each): "
        f"{'pass' if h_ok else 'FAIL'}; U^T x by rows against by grid points {mp.nstr(tr_err, 3)}; "
        f"|U^T (T - 3S)(e)| / |U^T T(e)| {mp.nstr(c3rel, 3)} (bound 1e-12) ({time.time() - t1:.0f} s)")

    def contrib(x):
        """U^T x, its parts on the affine rows and on the rest, and the sizes they would have without cancellation
        between rows, sqrt(sum_m (sum_r |U_mr x_r|)^2)"""
        part = lambda sel, ab: nrm([mp.fsum((abs(Uh[m][r] * x[r]) if ab else Uh[m][r] * x[r])   # noqa: E731
                                            for r in range(n) if sel(r)) for m in range(k)])
        allr, A_, N_ = (lambda r: True), (lambda r: aff[r]), (lambda r: not aff[r])
        d = dict(UTx=part(allr, False), abs_all=part(allr, True), A=part(A_, False), absA=part(A_, True),
                 N=part(N_, False), absN=part(N_, True), xmax=amax(x), xA=max((abs(x[r]) for r in range(n) if aff[r]),
                                                                              default=mp.mpf(0)))
        top = sorted(range(n), key=lambda r: -mp.sqrt(share[r]) * abs(x[r]))[:5]
        d["top"] = [(RH.labels[r], float(mp.sqrt(share[r] * k) * abs(x[r]))) for r in top]
        return d
    rel_tab = {}
    log("  per row, along e: |U^T x|; its part on the affine rows and on the rest; the sum of |U_r x_r| over all rows "
        "(no cancellation), on the affine rows and on the rest; max |x_r| over all rows and over the affine rows")
    for name, x in (("F''(e,e)", Qh), ("T(e)", Th), ("3S(e)", [3 * s for s in Sh]), ("T-3S", C3h)):
        d = contrib(x)
        rel_tab[name] = d
        log(f"    {name:<9} {mp.nstr(d['UTx'], 3):>10}; {mp.nstr(d['A'], 3):>10} {mp.nstr(d['N'], 3):>10}; "
            f"{mp.nstr(d['abs_all'], 3):>10} {mp.nstr(d['absA'], 3):>10} {mp.nstr(d['absN'], 3):>10}; "
            f"{mp.nstr(d['xmax'], 3):>10} {mp.nstr(d['xA'], 3):>10}; largest |U_r| |x_r|: "
            + ", ".join(f"{lab} {v:.2e}" for lab, v in d["top"][:3]))
    # diagnostics added after the REF pilot (not judged): each block's part of U^T x against its no-cancellation size;
    # the pairing of U's rows in the last two blocks, which both hold minus node K-1's state; w_ee by unknowns; and
    # how much of the axis Delta lies in span(V)
    bl = [lab.split(".")[0] for lab in RH.labels]
    byblock = {}
    log("  each block's part of U^T x, |sum| (sum of |U_r x_r|), the three largest by the latter:")
    for name, x in (("F''(e,e)", Qh), ("T(e)", Th), ("3S(e)", [3 * s for s in Sh]), ("T-3S", C3h)):
        d = {}
        for b in dict.fromkeys(bl):
            idx = [r for r in range(n) if bl[r] == b]
            d[b] = (float(nrm([mp.fsum(Uh[m][r] * x[r] for r in idx) for m in range(k)])),
                    float(nrm([mp.fsum(abs(Uh[m][r] * x[r]) for r in idx) for m in range(k)])))
        byblock[name] = d
        log(f"    {name:<9} " + "; ".join(f"{b} {v[0]:.2e} ({v[1]:.2e})"
                                          for b, v in sorted(d.items(), key=lambda t: -t[1][1])[:3]))
    K1 = (L - 1) // (4 * M)
    o1, o2 = 4 * M * (K1 - 2), 4 * M * (K1 - 1)
    pair = max(nrm([u[o1 + i] + u[o2 + i] for i in range(4 * M)]) / nrm(u) for u in Um)
    wsh = [x ** 2 for x in CH(wee[1])]
    ws = mp.fsum(wsh)
    worder = sorted(range(n), key=lambda r: -wsh[r])
    pdel = mp.fsum(x[L - 1] ** 2 for x in X)
    log(f"  U on the last two blocks' rows: max over the weak vectors of |u(block {K1 - 2}) + u(block {K1 - 1})| / |u| "
        f"{mp.nstr(pair, 3)}; |P_V e_Delta|^2 {mp.nstr(pdel, 8)}; w_ee by unknowns: "
        + ", ".join(f"{CH.labels[r]} {float(wsh[r] / ws):.3f}" for r in worder[:8]))
    out["blocks"] = byblock
    out["pair"] = float(pair)
    out["PV_Delta"] = float(pdel)
    out["wee_top"] = [(CH.labels[r], float(wsh[r] / ws)) for r in worder[:12]]
    out["along"] = dict(wee=float(nrm(wee[1])), dwee=float(dwee), dT=float(dT), dS=float(dS), h_ok=h_ok,
                        tr_err=float(tr_err), c3rel=float(c3rel),
                        contrib={nm: {kk: (v if kk == "top" else float(v)) for kk, v in d.items()} for nm, d in rel_tab.items()},
                        rows=dict(Fee=[float(x) for x in Qh], T=[float(x) for x in Th], S=[float(x) for x in Sh]))
    dump()

    if truth is not None:
        lab_ok = RH.labels == truth.labels
        got = set(RH.labels[r] for r in range(n) if aff[r])
        pr1 = bool(lab_ok and got == truth.affine)
        st = truth.share_true()
        e_share = abs(share_aff - truth.share_aff_true) / truth.share_aff_true
        e_rows = max(abs(share[r] - st[r]) for r in range(n)) / max(st)
        pr2 = bool(e_share < 1e-6 and e_rows < 1e-6)
        qt = truth.q_true(X)
        nl = [r for r in range(n) if qt[r] > 0]
        e_q = max(abs(q[r] - qt[r]) / qt[r] for r in nl)
        pr3 = bool(e_q < 1e-6)
        Ft, Tt, St = truth.along_true(ev)
        e_al = [max(abs(x - y) for x, y in zip(xm, xt)) / amax(xt) for xm, xt in ((Qh, Ft), (Th, Tt), (Sh, St))]
        partA = max(rel_tab[nm]["A"] / rel_tab[nm]["abs_all"] for nm in ("F''(e,e)", "T(e)", "3S(e)"))
        pr4 = bool(max(e_al) < 1e-6 and partA < 1e-10)
        log(f"  planted (PR1) the affine rows found, {sorted(got) if len(got) < 9 else str(len(got)) + ' rows'}, against "
            f"the planted {sorted(truth.affine)}, labels {'equal' if lab_ok else 'NOT equal'}: {'pass' if pr1 else 'FAIL'}")
        log(f"  planted (PR2) share on them {mp.nstr(share_aff, 10)} against {mp.nstr(truth.share_aff_true, 10)}: "
            f"{mp.nstr(e_share, 3)}; every row's share {mp.nstr(e_rows, 3)} of the largest (bound 1e-6): "
            f"{'pass' if pr2 else 'FAIL'}")
        log(f"  planted (PR3) q_r on the nonlinear rows against the planted {mp.nstr(e_q, 3)} (bound 1e-6): "
            f"{'pass' if pr3 else 'FAIL'}")
        log(f"  planted (PR4) F''(e,e), T(e), S(e) per row against the planted, of each one's largest: "
            + ", ".join(mp.nstr(x, 3) for x in e_al) + f"; the affine rows' part of U^T x, of the no-cancellation size, "
            f"{mp.nstr(partA, 3)} (bounds 1e-6, 1e-10): {'pass' if pr4 else 'FAIL'}")
        out["planted"] = dict(pr1=pr1, pr2=pr2, pr3=pr3, pr4=pr4, e_share=float(e_share), e_rows=float(e_rows),
                              e_q=float(e_q), e_along=[float(x) for x in e_al], partA=float(partA), found=sorted(got))
        dump()
        return pr1 and pr2 and pr3 and pr4 and bool(orth < 1e-120), rec
    tr_ok = bool(orth < 1e-120 and tr_err < 1e-100)
    c3_ok = bool(c3rel < 1e-12)
    out["checks"] = dict(W1_frozen=frozen_ok, W2_transform=tr_ok, W3_h=h_ok, W3_c3=c3_ok)
    log(f"  checks: (W1) the rows linear by construction classified affine {'pass' if frozen_ok else 'FAIL'}; (W2) the "
        f"change of rows {'pass' if tr_ok else 'FAIL'}; (W3) h checks {'pass' if h_ok else 'FAIL'}, 3S = T along e "
        f"{'pass' if c3_ok else 'FAIL'}; verdict (W4): the share {mp.nstr(share_aff, 6)} {verdict(share_aff)} the "
        f"conjecture (at h {H_ROWS[0]:.0e}: {verdict(share_aff_h)})")
    dump()
    return frozen_ok and tr_ok and h_ok and c3_ok, rec


def rowsum(paths):
    """the summary of rows runs (their --json files): per start the verdict, U's shares by block, the largest rows,
    the pairing and the cancellations, each as the run stored it"""
    for path in paths:
        with open(path) as fh:
            r = json.load(fh)
        w = r["rows"]
        lab, sh, q, Jr = w["labels"], w["share"], w["q"][-1], w["Jr"]
        blk = {}
        for x, s in zip(lab, sh):
            b = x.split(".")[0]
            blk[b] = blk.get(b, 0.0) + s
        print(f"{os.path.basename(path)}: {r.get('label', r.get('dz0', 'planted'))}")
        print(f"  share of |U| on affine rows {w['share_aff']:.4g} ({w['verdict']}; at h 1e-32 {w['share_aff_h32']:.4g}, "
              f"{w['verdict_h32']}); against the largest q alone {w['share_glob']:.4g}, against |J_r| alone "
              f"{w['share_own']:.4g}; worst weak vector {w['lam_min']:.3g}; rows linear by construction: share "
              f"{w['share_frozen']:.3g}, all affine {w['frozen_ok']}")
        print("  U by block: " + ", ".join(f"{b} {s:.4g}" for b, s in sorted(blk.items(), key=lambda t: -t[1])[:4]))
        top = sorted(range(len(lab)), key=lambda i: -sh[i])[:8]
        print("  largest rows: " + ", ".join(f"{lab[i]} {sh[i]:.3f} (q/|J_r| {q[i] / Jr[i]:.1e})" for i in top))
        uw = [i for i in range(len(lab)) if sh[i] > 1e-6]
        qj = [q[i] / Jr[i] for i in uw]
        print(f"  rows with share > 1e-6: {len(uw)}, q/|J_r| {min(qj):.1e}..{max(qj):.1e}, q {min(q[i] for i in uw):.2e}.."
              f"{max(q[i] for i in uw):.2e}")
        if "pair" in w:
            print(f"  pairing of the last two blocks {w['pair']:.2e}; |P_V e_Delta|^2 {w['PV_Delta']:.6f}; w_ee: "
                  + ", ".join(f"{a} {b:.3f}" for a, b in w["wee_top"][:5]))
        al = w.get("along")
        if al:
            for nm, d in al["contrib"].items():
                line = (f"  {nm:<9} |U^T x| {d['UTx']:.3e}, no-cancellation {d['abs_all']:.3e}, ratio "
                        f"{d['UTx'] / d['abs_all']:.2e}")
                if "blocks" in w and nm in w["blocks"]:
                    hb = w["blocks"][nm].get("horizon")
                    if hb:
                        line += f"; horizon block {hb[0]:.3e} of {hb[1]:.3e} ({hb[0] / hb[1]:.2e})"
                    lb = [b for b in w["blocks"][nm] if b.startswith("node")]
                    if lb:
                        nb = w["blocks"][nm][lb[-1]]
                        line += f"; {lb[-1]} {nb[0]:.3e} of {nb[1]:.3e}"
                print(line)
            print(f"  |U^T (T - 3S)(e)| / |U^T T(e)| {al['c3rel']:.2e}; w_ee h check {al['dwee']:.1e}, T {al['dT']:.1e}, "
                  f"S {al['dS']:.1e}")
        if "checks" in w:
            print("  checks: " + ", ".join(f"{n} {'pass' if v else 'FAIL'}" for n, v in w["checks"].items()))
        if "planted" in w:
            print("  planted: " + ", ".join(f"{n} {'pass' if w['planted'][n] else 'FAIL'}" for n in ("pr1", "pr2", "pr3", "pr4")))


def main_rows(args, opts, mutate, log, rec, dump):
    """rows RAWDIR DZ0 BASE or rows planted: the basis, then rows_scan"""
    if args[1:] == ["planted"]:
        sysm = RowPlanted(int(opts.get("seed", 5)), log, mutate)
        rec.update(sysm.meta)
        bas = basis(sysm, opts, log, rec, dump)
        ok, rec = rows_scan(sysm, bas, opts, mutate, log, rec, dump, truth=sysm)
        return ok
    sysm = Real(*args[1:4], int(opts.get("threads", 2)), log)
    rec.update(sysm.meta)
    if not sysm.offset_ok:
        log("check: FAIL (the held offset is not on its rows only)")
        return False
    bas = basis(sysm, opts, log, rec, dump)
    ok, rec = rows_scan(sysm, bas, opts, mutate, log, rec, dump)
    return ok


def strip(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def main_ray(args, opts, mutate, log, rec, dump):
    """ray RAWDIR DZ0 BASE [--tensors=1] or ray planted: the scan of ray_scan, after the basis (and, with --tensors=1,
    the whole of #528's measurement, whose h checks are then judged too)"""
    if args[1:] == ["planted"]:
        sysm = RayPlanted(int(opts.get("seed", 5)), log)
        rec.update(sysm.meta)
        bas = basis(sysm, opts, log, rec, dump)
        ok, rec = ray_scan(sysm, bas, opts, mutate, log, rec, dump, truth=sysm.dep_true(bas["X"], bas["Um"]))
        ok = ok and bool(sysm.selfcheck < 1e-20)
    else:
        sysm = Real(*args[1:4], int(opts.get("threads", 2)), log)
        rec.update(sysm.meta)
        if not sysm.offset_ok:
            log("check: FAIL (the held offset is not on its rows only)")
            return False
        bas = basis(sysm, opts, log, rec, dump)
        tens, hok = None, True
        if opts.get("tensors", "0") == "1":
            measure(sysm, opts, "", log, rec, dump, bas)
            I = rec["_internal"]
            tens = [I["C3"], I["C3_0"]]
            hok = bool((rec["S"]["ok"] or rec["S"]["trunc_bound"]) and rec["w"]["ok"])
            log(f"  #528's h checks (judged here): w_jl {'pass' if rec['w']['ok'] else 'FAIL'}, S "
                f"{'pass' if rec['S']['ok'] else ('bounded by its truncation' if rec['S']['trunc_bound'] else 'FAIL')}"
                "; its S control is reported, not judged (notes/REDUCED_CUBIC.md 5.4, 9)")
        ok, rec = ray_scan(sysm, bas, opts, mutate, log, rec, dump, tensors=tens)
        rec["h_checks_ok"] = hok
        ok = ok and hok
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    mutate = opts.get("mutate", "")
    if args and args[0] == "raysum":
        raysum(args[1:])
        return 0
    if args and args[0] == "rowsum":
        rowsum(args[1:])
        return 0
    modes = dict(run=("", "w0", "Ssign", "Sh", "hF"), planted=("", "w0", "Ssign", "Sh", "hF"), ray=("", "w0", "B"),
                 rows=("", "quadA", "parity"))
    if not args or args[0] not in modes or (args[0] == "run" and len(args) != 4) or \
            (args[0] in ("ray", "rows") and len(args) != 4 and args[1:] != ["planted"]):
        sys.exit(__doc__)
    if mutate not in modes[args[0]]:
        sys.exit(f"--mutate={mutate}: in mode {args[0]}, one of {', '.join(modes[args[0]][1:])}")
    out = opts.get("json", "")
    rec = dict(mode=args[0], mutate=mutate, h=H, started=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    def log(s):
        print(s, flush=True)

    def dump():
        if out:
            with open(out, "w") as fh:
                json.dump(strip(rec), fh, indent=1, default=float)
    if args[0] in ("ray", "rows"):
        if args[0] == "ray":
            rec["ray_t"] = [float(t) for t in RAY_T]
            ok = main_ray(args, opts, mutate, log, rec, dump)
        else:
            ok = main_rows(args, opts, mutate, log, rec, dump)
        rec["passed"] = bool(ok)
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        dump()
        print("check:", "pass" if ok else "FAIL")
        return 0 if ok else 1
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
