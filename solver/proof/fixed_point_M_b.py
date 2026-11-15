#!/usr/bin/env python3
"""Start-value contraction for the M-harmonic inner problem: a second implementation (issue #427).

It proves Lemma FP(M) of proof/A_lemmas_M.md, section 2 (the start-value fixed point, diag splitting, on a ray) with
code that shares nothing with fixed_point.py, fixed_point_M.py, series_arb.py or validated_int.py. It imports only
python-flint (arb/acb ball arithmetic) and the standard library. Notes: proof/A_lemmas_M.md, section 2.5.

System (solver/inner_stokes/inner_np.py's rhs, D, S on odd |mu| <= M, C on even |nu| <= M + 1, [.] the projection of a
harmonic product onto its result's set), with z = xi^2, D = xi P, S = z Q, ' = d/dz, s_mu = i mu w:
    2 z P' = [C Q] - (1 + s) P,      2 z Q' = -(2 + s) Q + 2 [C P'] + [G P],      C' = G / 2,
    G = [C E],  E = [P P] / 2.
With ct = C - 1, V = Q - (1 + s) P and W = V + [ct Q] this is
    P' = W / (2z),
    V' = (1/(2z^2) - (3 + 2s)/(2z)) V + Phi_V,
         Phi_V = ([ct Q] + [ct W]) / (2z^2) + (-(2 + s)(1 + s) P - (1 + s) [ct Q] + [G P]) / (2z),
    C' = G / 2.
Formal solution P = sum d_j z^j, Q = sum q_j z^j, C = sum g_j z^j, d_0 = 2 F, g_0 = 1:
    q_j = (2j + 1 + s) d_j - sum_{i=1..j} [g_i q_{j-i}],
    g_{j+1} = G_j / (2 (j + 1)),   G_j = sum_i [g_i E_{j-i}],   E_j = sum_i [d_i d_{j-i}] / 2,
    d_{j+1} = ((2j + 2 + s) q_j - 2 sum_{i=1..j} (j + 1 - i) [g_i d_{j+1-i}] - sum_{m=0..j} [G_m d_{j-m}]) / (2 (j + 1)).

The fixed point. Y = Y_N + eta, Y_N the series truncated at N terms, eta = (p, v, c). On a ray arg z = phi with
a = -cos(phi)/2 > 0, for 0 < |z| <= R:
    p(z) = int_0^z (Phi_P(Y_N + eta) - P_N') dt,          c(z) = int_0^z (Phi_C(Y_N + eta) - C_N') dt,
    v(z) = int_0^z K_mu(t, z) (Phi_V(Y_N + eta) + lambda V_N - V_N') dt,   K_mu = e^{1/(2t) - 1/(2z)} (t/z)^{(3 + 2s)/2}.
On the ray |K_mu| = e^{-a(1/rho - 1/r)} (rho/r)^{e_mu}, e_mu = 3/2 - mu Im w, and for e >= 0, m >= -2
    int_0^r e^{-a(1/rho - 1/r)} (rho/r)^e rho^m d rho  <=  r^{m+2} / (a + (m + 1 + e) r)
(substitute 1/rho = 1/r + tau and integrate by parts once: 1 = a I_{n-1} + (n - 1) r I_n >= (a + (n - 1) r) I_n).
Norms: sup r^{-k} |p|, sup r^{-k} |v|, sup r^{-k-1} |c|. Majorants are polynomials in t = |z| with nonnegative
coefficients, valid in every direction, carried with their derivatives in the 2(M + 1) + (M + 2) components of eta.

The test is componentwise. With F_i >= ||T(0)_i|| and L_ij(w) >= the norm of d(T eta)_i / d eta_j over the set
X_w = {||eta_i|| <= w_i}, a vector w > 0 with F + L(w) w < w in every component gives T(X_w) in X_w and a contraction
in the norm max_i ||eta_i|| / w_i with constant max_i (L w)_i / w_i < 1 (Banach's fixed point theorem).

    python3 fixed_point_M_b.py --M=5 --x=A,wf,B3,C3,B5,C5 [--r0=0.07] [--arg=3pi/8] [--sector=d] [--k=K] [--N=N]
                               [--cbox=c | --box=h]
        the contraction at a point, on a real box x +- h or on the complex box x +- (h + c) + i [-c, c]
    python3 fixed_point_M_b.py --M=5 --start --rho0=405 --h=1e-18 --cauchy=1e-3 --compare=ray.json [--x=...]
        the start set at s0 = rho0 e^{-i 2 arg} in the u-system's variables (P, V, kappa (C - 1)) with its derivative
        columns over x +- h, compared with the stored C^1 start set of lemma6b1_cover_M.py (sets["0"] of ray.json; --x
        defaults to the file's centre). Exit status 0 only if every contraction holds and every component overlaps.
    python3 fixed_point_M_b.py --planted [--M=5]
        the linear problem (no gravity), whose lateral solution is 2 F x^alpha U(alpha, 1/2, x), x = -1/(2z),
        alpha = (1 + s)/2 (Kummer's U, evaluated by arb): the enclosure must contain it
    mutations (each must make its check fail; the exit status is then 3):
        --mutate=radius   r0 = 0.3: the contraction must fail
        --mutate=coef     one series coefficient off by a relative 1e-6: the low orders of the residual no longer vanish
        --mutate=sign     the sign of [G P], in the series and the residual alike: a contraction for another system,
                          whose start set must miss the stored one
        --mutate=scale    [G P] times 1 + 1e-9, likewise
        --mutate=rho0     the start point at |u| = rho0 - 1: the start set must miss the stored one
        --mutate=truth    (planted) Kummer's alpha times 1 + 1e-12: the enclosure must miss it
        --mutate=forcing  (--two) the forcing of the shorter truncation divided by 1000: its enclosure must miss
    python3 fixed_point_M_b.py --M=5 --two=95,105 --compare=ray.json
        two truncations of the full problem at the file's centre: the two enclosures must overlap at the start
        point and at |z| = R; the ratio of Y_105 - Y_95 to the sum of the radii measures how tight the bounds are
"""
import json
import math
import sys
import time

from flint import arb, acb, acb_poly, ctx
import flint

PERIOD = "1.64389375"          # w_inf = 2 pi / 1.64389375, the solvers' constant
KAPPA = 100                    # the u-system scales its c components by 100 (lemma6_usys_M.KAPPA)
I = acb(0, 1)


def harmonics(M):
    return list(range(-M, M + 1, 2)), list(range(-(M + 1), M + 2, 2))


def angle(s):
    """'3pi/8' -> arb"""
    s = str(s).replace(" ", "")
    if "pi" not in s:
        return arb(s)
    num, _, den = s.partition("pi")
    num = arb(1) if num in ("", "+") else arb(num)
    den = arb(den.lstrip("/")) if den else arb(1)
    return num * arb.pi() / den


def up(x):
    """exact upper bound of a real ball, as a ball of radius 0"""
    return x.upper()


# ---------------------------------------------------------------- first-order jets in the parameters
class Jet:
    """a value and its derivatives in m parameters"""
    __slots__ = ("v", "d")

    def __init__(self, v, d):
        self.v, self.d = v, d

    def __add__(self, o):
        if isinstance(o, Jet):
            return Jet(self.v + o.v, [a + b for a, b in zip(self.d, o.d)])
        return Jet(self.v + o, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, [-a for a in self.d])

    def __sub__(self, o):
        if isinstance(o, Jet):
            return Jet(self.v - o.v, [a - b for a, b in zip(self.d, o.d)])
        return Jet(self.v - o, self.d)

    def __mul__(self, o):
        if isinstance(o, Jet):
            u, w = self.v, o.v
            return Jet(u * w, [a * w + u * b for a, b in zip(self.d, o.d)])
        return Jet(self.v * o, [a * o for a in self.d])

    __rmul__ = __mul__

    def __truediv__(self, o):
        return Jet(self.v / o, [a / o for a in self.d])


# ---------------------------------------------------------------- the profile and the formal series
def profile(x, M):
    """F's harmonics, w and the labels s_mu from x = (A, w/w_inf, B3, C3, ..); x: acb or Jet entries"""
    odd, _ = harmonics(M)
    F = {1: x[0] * (-I / 2), -1: x[0] * (I / 2)}
    for i, nu in enumerate(range(3, M + 1, 2)):
        B, C = x[2 + 2 * i], x[3 + 2 * i]
        F[nu] = B * (-I / 2) + C * acb(0.5)
        F[-nu] = B * (I / 2) + C * acb(0.5)
    w = x[1] * acb(2 * arb.pi() / arb(PERIOD))
    return F, w, {mu: w * (I * mu) for mu in odd}


def triples(left, right, target):
    """(i, j, i + j) with i in left, j in right, i + j in target: the terms of a projected harmonic product"""
    tg = set(target)
    return [(i, j, i + j) for i in left for j in right if i + j in tg]


def formal_series(F, s, M, N, zero, one, gravity=True, gp=None):
    """d_j, q_j, g_j for j < N (dicts over harmonics). gp: the factor of [G P] (None = 1; the mutations use it)."""
    odd, even = harmonics(M)
    t_eo, t_oo, t_ee = triples(even, odd, odd), triples(odd, odd, even), triples(even, even, even)
    d = [{mu: F[mu] * 2 for mu in odd}]
    g = [{nu: (one if nu == 0 else zero) for nu in even}]
    q, E, G = [], [], []
    for j in range(N):
        qj = {mu: d[j][mu] * (2 * j + 1) + s[mu] * d[j][mu] for mu in odd}
        if gravity:
            for i in range(1, j + 1):
                gi, qm = g[i], q[j - i]
                for (nu, m, mu) in t_eo:
                    qj[mu] = qj[mu] - gi[nu] * qm[m]
        q.append(qj)
        if j == N - 1:
            break
        num = {mu: qj[mu] * (2 * j + 2) + s[mu] * qj[mu] for mu in odd}
        if gravity:
            ej = {nu: zero for nu in even}
            for i in range(j + 1):
                di, dm = d[i], d[j - i]
                for (m1, m2, nu) in t_oo:
                    ej[nu] = ej[nu] + di[m1] * dm[m2]
            E.append({nu: ej[nu] / 2 for nu in even})
            Gj = {nu: zero for nu in even}
            for i in range(j + 1):
                gi, em = g[i], E[j - i]
                for (n1, n2, nu) in t_ee:
                    Gj[nu] = Gj[nu] + gi[n1] * em[n2]
            G.append(Gj)
            g.append({nu: Gj[nu] / (2 * (j + 1)) for nu in even})
            for i in range(1, j + 1):
                gi, dm, c = g[i], d[j + 1 - i], 2 * (j + 1 - i)
                for (nu, m, mu) in t_eo:
                    num[mu] = num[mu] - gi[nu] * dm[m] * c
            for m_ in range(j + 1):
                Gm, dm = G[m_], d[j - m_]
                for (nu, m, mu) in t_eo:
                    term = Gm[nu] * dm[m]
                    num[mu] = num[mu] - (term if gp is None else term * gp)
        else:
            g.append({nu: zero for nu in even})
        d.append({mu: num[mu] / (2 * (j + 1)) for mu in odd})
    return d, q, g


# ---------------------------------------------------------------- polynomials in z and the residual
def pconv(A, B, left, right, target):
    out = {k: acb_poly([0]) for k in target}
    for (i, j, k) in triples(left, right, target):
        out[k] = out[k] + A[i] * B[j]
    return out


def shift(p, n=1):
    """z^n p"""
    return acb_poly([0] * n + p.coeffs())


def residual_numerators(M, s, P, Q, C, V, gravity=True, gp=None):
    """N_P = 2z (Phi_P - P'), N_V = 2z^2 (Phi_V + lambda V - V'), N_C = Phi_C - C' at Y_N, as polynomials in z"""
    odd, even = harmonics(M)
    one = acb_poly([1])
    ct = {nu: C[nu] - (one if nu == 0 else acb_poly([0])) for nu in even}
    if gravity:
        cQ = pconv(ct, Q, even, odd, odd)
        W = {mu: V[mu] + cQ[mu] for mu in odd}
        cW = pconv(ct, W, even, odd, odd)
        PP = pconv(P, P, odd, odd, even)
        E = {nu: PP[nu] * acb(0.5) for nu in even}
        cE = pconv(ct, E, even, even, even)
        G = {nu: E[nu] + cE[nu] for nu in even}
        GP = pconv(G, P, even, odd, odd)
        if gp is not None:
            GP = {mu: GP[mu] * gp for mu in odd}
    else:
        zp = acb_poly([0])
        cQ = {mu: zp for mu in odd}
        W, cW, GP = dict(V), dict(cQ), dict(cQ)
        G = {nu: zp for nu in even}
    NP = {mu: W[mu] - 2 * shift(P[mu].derivative()) for mu in odd}
    NV = {}
    for mu in odd:
        sm = s[mu]
        lin = V[mu] - (3 + 2 * sm) * shift(V[mu])
        low = cQ[mu] + cW[mu]
        mid = -((2 + sm) * (1 + sm)) * P[mu] - (1 + sm) * cQ[mu] + GP[mu]
        NV[mu] = lin + low + shift(mid) - 2 * shift(V[mu].derivative(), 2)
    NC = {nu: G[nu] * acb(0.5) - C[nu].derivative() for nu in even}
    return NP, NV, NC


def absup(p, R, start=0, drop=0):
    """sum_{j >= start} |p_j| R^(j - drop), an exact upper bound"""
    tot = arb(0)
    cs = p.coeffs()
    for j in range(start, len(cs)):
        tot = up(tot + cs[j].abs_upper() * R ** (j - drop))
    return tot


# ---------------------------------------------------------------- majorants: sparse polynomials in t, coefficients >= 0
def sp_add(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = up(out[e] + c) if e in out else c
    return out


def sp_mul(a, b):
    out = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            c = up(c1 * c2)
            e = e1 + e2
            out[e] = up(out[e] + c) if e in out else c
    return out


class MJ:
    """|f| <= v(t) on the set X_w, and |d f / d eta_slot| <= d[slot](t); t = |z|"""
    __slots__ = ("v", "d")

    def __init__(self, v, d=None):
        self.v, self.d = v, (d or {})

    def __add__(self, o):
        d = dict(self.d)
        for sl, p in o.d.items():
            d[sl] = sp_add(d[sl], p) if sl in d else p
        return MJ(sp_add(self.v, o.v), d)

    def __mul__(self, o):
        d = {sl: sp_mul(p, o.v) for sl, p in self.d.items()}
        for sl, p in o.d.items():
            t = sp_mul(self.v, p)
            d[sl] = sp_add(d[sl], t) if sl in d else t
        return MJ(sp_mul(self.v, o.v), d)

    def scale(self, c):
        return MJ({e: up(v * c) for e, v in self.v.items()}, {sl: {e: up(v * c) for e, v in p.items()}
                                                              for sl, p in self.d.items()})

    def shift(self, n):
        return MJ({e + n: v for e, v in self.v.items()}, {sl: {e + n: v for e, v in p.items()}
                                                          for sl, p in self.d.items()})


def mconv(A, B, left, right, target):
    out = {}
    for (i, j, k) in triples(left, right, target):
        t = A[i] * B[j]
        out[k] = out[k] + t if k in out else t
    for k in target:
        out.setdefault(k, MJ({}))
    return out


def slow_norm(sp, kj, ki, R):
    """sup_{r <= R} r^-ki int_0^r sp(t) t^kj dt"""
    tot = arb(0)
    for m, c in sp.items():
        e = m + kj + 1
        if e < 1 or e < ki:
            raise ValueError(f"slow row: exponent {e} below the weight {ki}")
        tot = up(tot + c * R ** (e - ki) / e)
    return tot


def fast_norm(sp, kj, ki, R, a, emin, kernel="sharp"):
    """sup_{r <= R} r^-ki int_0^r e^{-a(1/rho - 1/r)} (rho/r)^e sp(rho) rho^kj d rho, e >= emin >= 0"""
    tot = arb(0)
    for m, c in sp.items():
        mm = m + kj
        pw = mm + 2 - ki
        if mm < -1 or pw < 0:
            raise ValueError(f"fast row: exponent {mm} below the weight {ki} - 2")
        if pw == 0:
            b = 1 / a
        elif kernel == "sharp":
            b = R ** pw / (a + (mm + 1 + emin) * R)
        else:                                   # the two bounds of A_certify.md 4.4: r^{m+2}/a and r^{m+1}/(m+1)
            b1, b2 = up(R ** pw / a), up(R ** (pw - 1) / (mm + 1))
            b = b1 if b1 < b2 else b2 if b2 < b1 else b1.union(b2)
        tot = up(tot + c * b)
    return tot


# ---------------------------------------------------------------- the contraction
class Setup:
    """series, truncation, residual and the constants that do not depend on the set X_w"""

    def __init__(self, x, M, r0="0.07", arg="3pi/8", sector=None, N=None, k=None, gravity=True, kernel="sharp",
                 mutate=None, prec=512, verbose=False):
        ctx.prec = prec
        self.M, self.gravity, self.kernel, self.mutate = M, gravity, kernel, mutate
        self.odd, self.even = harmonics(M)
        if mutate == "radius":
            r0 = "0.3"
        self.r0 = arb(r0)
        self.R = self.r0 ** 2
        self.x = [acb(v) for v in x]
        self.F, self.w, self.s = profile(self.x, M)
        self.gp = {"sign": acb(-1), "scale": 1 + acb("1e-9")}.get(mutate)
        # the ray: arg z = 2 arg xi; rate a = -cos(arg z)/2, on a sector of rays its minimum (at an end, -cos being
        # concave on (pi/2, 3pi/2))
        th = 2 * angle(arg)
        self.arg = angle(arg)
        if sector is None:
            a = -th.cos() / 2
        else:
            lo, hi = th - 2 * arb(sector), th + 2 * arb(sector)
            if not (lo > arb.pi() / 2 and hi < 3 * arb.pi() / 2):
                raise ValueError("the sector leaves pi/2 < arg z < 3pi/2")
            a1, a2 = -lo.cos() / 2, -hi.cos() / 2
            a = a1 if a1 < a2 else a2 if a2 < a1 else a1.union(a2)
        self.a = arb(a.lower())
        if not self.a > 0:
            raise ValueError("no decay of the fast kernel on this ray")
        # |(t/z)^{(3 + 2 s_mu)/2}| = (rho/r)^{3/2 - mu Im w}: the smallest exponent over the labels and the box
        self.emin = arb((arb(1.5) - M * self.w.imag.abs_upper()).lower())
        if not self.emin >= 0:
            raise ValueError("Im w too large for the kernel bound")
        Rf = float(self.R.mid())
        J = max(int(1 / (2 * Rf)) + 30, (N or 0) + 1) if N is None else N
        t0 = time.time()
        d, q, g = formal_series(self.F, self.s, M, J, acb(0), acb(1), gravity, self.gp)
        if N is None:
            mags = [max(d[j][mu].abs_upper() for mu in self.odd) * self.R ** j for j in range(J)]
            N = min(range(4, J), key=lambda j: float(mags[j].mid()))
        self.N = N
        if mutate == "coef":
            d[5][M] = d[5][M] * (1 + acb("1e-6"))
        self.P = {mu: acb_poly([d[j][mu] for j in range(N)]) for mu in self.odd}
        self.Q = {mu: acb_poly([q[j][mu] for j in range(N)]) for mu in self.odd}
        self.C = {nu: acb_poly([g[j][nu] for j in range(N)]) for nu in self.even}
        # V = Q - (1 + s) P; its order 0 is q_0 - (1 + s) d_0 = 0 identically (the recursion's first line)
        self.V = {mu: acb_poly([0] + [q[j][mu] - (1 + self.s[mu]) * d[j][mu] for j in range(1, N)]) for mu in self.odd}
        self.k = N - 2 if k is None else k
        if self.k > N - 2 or self.k < 2:
            raise ValueError(f"weight exponent {self.k} outside 2..N - 2")
        self.comps = [("p", mu) for mu in self.odd] + [("v", mu) for mu in self.odd]
        if gravity:
            self.comps += [("c", nu) for nu in self.even]
        self.kw = {cp: (self.k + 1 if cp[0] == "c" else self.k) for cp in self.comps}
        self.t_series = time.time() - t0
        self._forcing()
        self._sups()
        if verbose:
            print(f"  series {self.t_series:.1f} s, N = {N}, k = {self.k}, a = {float(self.a):.5f}, "
                  f"e_min = {float(self.emin):.4f}", flush=True)

    def _forcing(self):
        """F_i >= ||T(0)_i||, after checking that the residual vanishes below its structural orders"""
        N, k, R = self.N, self.k, self.R
        NP, NV, NC = residual_numerators(self.M, self.s, self.P, self.Q, self.C, self.V, self.gravity, self.gp)
        self.low_ok = True
        self.low_max = 0.0
        self.Fv = {}

        def low(p, n):
            # the coefficients below order n vanish identically; their balls must contain 0 (low_max: the largest
            # |ball| R^j among them, the size at |z| = R of what rounding left there)
            for j, c in enumerate(p.coeffs()[:n]):
                if not c.contains(0):
                    self.low_ok = False
                self.low_max = max(self.low_max, float((c.abs_upper() * R ** j).upper()))
        for mu in self.odd:
            low(NP[mu], N)
            low(NV[mu], N)
            # int_0^z N_P/(2t) dt = sum_j n_j z^j / (2j), exactly
            tot = arb(0)
            for j, c in enumerate(NP[mu].coeffs()):
                if j >= N:
                    tot = up(tot + c.abs_upper() * R ** (j - k) / (2 * j))
            self.Fv[("p", mu)] = tot
            sp = {j - 2: up(c.abs_upper() / 2) for j, c in enumerate(NV[mu].coeffs()) if j >= N}
            self.Fv[("v", mu)] = fast_norm(sp, 0, k, R, self.a, self.emin, self.kernel)
        if self.gravity:
            for nu in self.even:
                low(NC[nu], N - 1)
                tot = arb(0)
                for j, c in enumerate(NC[nu].coeffs()):
                    if j >= N - 1:
                        tot = up(tot + c.abs_upper() * R ** (j - k) / (j + 1))
                self.Fv[("c", nu)] = tot

    def _sups(self):
        R = self.R
        self.Ph = {mu: absup(self.P[mu], R) for mu in self.odd}
        self.Qh = {mu: absup(self.Q[mu], R) for mu in self.odd}
        self.V1 = {mu: absup(self.V[mu], R, 1, 1) for mu in self.odd}
        self.c1 = {nu: absup(self.C[nu], R, 1, 1) for nu in self.even}
        self.s1 = {mu: (1 + self.s[mu]).abs_upper() for mu in self.odd}
        self.s2 = {mu: ((2 + self.s[mu]) * (1 + self.s[mu])).abs_upper() for mu in self.odd}

    def lipschitz(self, w):
        """L_ij(w): the norm of d(T eta)_i / d eta_j over X_w, w = {component: radius}"""
        odd, even, k, R = self.odd, self.even, self.k, self.R
        one, zero = arb(1), arb(0)
        g = self.gravity
        wp = {mu: w[("p", mu)] for mu in odd}
        wv = {mu: w[("v", mu)] for mu in odd}
        P = {mu: MJ({0: self.Ph[mu], k: wp[mu]}, {("p", mu): {0: one}}) for mu in odd}
        V = {mu: MJ({1: self.V1[mu], k: wv[mu]}, {("v", mu): {0: one}}) for mu in odd}
        Q = {mu: MJ({0: self.Qh[mu], k: up(wv[mu] + self.s1[mu] * wp[mu])},
                    {("v", mu): {0: one}, ("p", mu): {0: self.s1[mu]}}) for mu in odd}
        half = arb(0.5)
        if g:
            ct = {nu: MJ({1: self.c1[nu], k + 1: w[("c", nu)]}, {("c", nu): {0: one}}) for nu in even}
            cQ = mconv(ct, Q, even, odd, odd)
            W = {mu: V[mu] + cQ[mu] for mu in odd}
            cW = mconv(ct, W, even, odd, odd)
            E = {nu: m.scale(half) for nu, m in mconv(P, P, odd, odd, even).items()}
            cE = mconv(ct, E, even, even, even)
            G = {nu: E[nu] + cE[nu] for nu in even}
            GP = mconv(G, P, even, odd, odd)
            if self.gp is not None:
                GP = {mu: m.scale(self.gp.abs_upper()) for mu, m in GP.items()}
        else:
            cQ = {mu: MJ({}) for mu in odd}
            W, cW, GP = dict(V), dict(cQ), dict(cQ)
        rows = {}
        for mu in odd:
            rows[("p", mu)] = W[mu].shift(-1).scale(half)
            mid = P[mu].scale(self.s2[mu]) + cQ[mu].scale(self.s1[mu]) + GP[mu]
            rows[("v", mu)] = (cQ[mu] + cW[mu]).shift(-2).scale(half) + mid.shift(-1).scale(half)
        if g:
            for nu in even:
                rows[("c", nu)] = G[nu].scale(half)
        L = []
        for ci in self.comps:
            row = []
            for cj in self.comps:
                sp = rows[ci].d.get(cj)
                if not sp:
                    row.append(zero)
                elif ci[0] == "v":
                    row.append(fast_norm(sp, self.kw[cj], self.kw[ci], R, self.a, self.emin, self.kernel))
                else:
                    row.append(slow_norm(sp, self.kw[cj], self.kw[ci], R))
            L.append(row)
        return L


def matvec(L, v):
    return [up(sum((L[i][j] * v[j] for j in range(len(v))), arb(0))) for i in range(len(v))]


def prove(st, verbose=True, label=""):
    """the componentwise contraction test for a Setup. Returns a dict; ok only if everything is proved."""
    comps, n = st.comps, len(st.comps)
    Fv = [st.Fv[cp] for cp in comps]
    out = {"M": st.M, "N": st.N, "k": st.k, "comps": comps, "low_ok": st.low_ok, "ok": False, "st": st}
    L0 = st.lipschitz({cp: arb(0) for cp in comps})
    # a positive vector u with L0 u ~ lambda u (power iteration on L0 + 1 in floats, which also converges for a
    # cyclic matrix; any positive vector is admissible)
    Lf = [[float(v.mid()) + 1e-13 for v in row] for row in L0]
    u = [1.0] * n
    for _ in range(3000):
        nu_ = [u[i] + sum(Lf[i][j] * u[j] for j in range(n)) for i in range(n)]
        m = max(nu_)
        u = [v / m for v in nu_]
    ua = [arb(v) for v in u]
    lam0 = max(float((matvec(L0, ua)[i] / ua[i]).upper()) for i in range(n))
    out["lam0"] = lam0
    if not lam0 < 0.999:
        out["why"] = f"the Lipschitz matrix at 0 has weighted row sums up to {lam0:.4f}"
        if verbose:
            report(out, label)
        return out
    # the big set X_w1: w1 = tau u with F <= (1 - lam0) tau u / 1.5
    tau = max((Fv[i] * 1.5 / ((1 - arb(lam0)) * ua[i])).upper() for i in range(n))
    w1 = [up(tau * ua[i]) for i in range(n)]
    L1 = st.lipschitz(dict(zip(comps, w1)))
    Lw = matvec(L1, w1)
    lam = max((Lw[i] / w1[i]).upper() for i in range(n))
    self1 = all(bool(Fv[i] + Lw[i] < w1[i]) for i in range(n))
    out.update(lam=lam, w1=w1, self1=self1, L=L1, F=Fv)
    # the refined set: w2 ~ (1 + 1e-3) (I - L1)^-1 F2, inside X_w1; T maps X_w2 into itself, so the fixed point is in
    # it. F2 = F + 1e-30 w1 >= F keeps every component positive (a row's forcing can vanish identically).
    F2 = [up(Fv[i] + w1[i] * arb("1e-30")) for i in range(n)]
    w2 = list(F2)
    for _ in range(600):
        t = matvec(L1, w2)
        w2 = [up(F2[i] + t[i]) for i in range(n)]
    w2 = [up(v * arb("1.001")) for v in w2]
    t = matvec(L1, w2)
    self2 = all(bool(F2[i] + t[i] <= w2[i]) and bool(w2[i] <= w1[i]) for i in range(n))
    out.update(w2=w2, self2=self2)
    out["ok"] = bool(st.low_ok and lam < 1 and self1 and self2)
    if verbose:
        report(out, label)
    return out


def eta_radius(out, cp, t):
    """|eta_cp(z)| at |z| = t <= R"""
    st = out["st"]
    i = out["comps"].index(cp)
    return up(out["w2"][i] * arb(t) ** st.kw[cp])


def report(out, label=""):
    st = out["st"]
    print(f"{label}M = {st.M}, r0 = {float(st.r0.mid()):.6g}, arg xi = {float(st.arg.mid()):.6f}, N = {st.N}, "
          f"k = {st.k}, a = {float(st.a):.5f}, e_min = {float(st.emin):.4f}, kernel bound {st.kernel}")
    print(f"  residual below its structural orders contains 0: {st.low_ok} (largest |ball| R^j there {st.low_max:.1e})")
    if "lam" not in out:
        print(f"  NO CONTRACTION: {out.get('why')}")
        return
    L, comps = out["L"], out["comps"]
    big = sorted(((float(L[i][j]), comps[i], comps[j]) for i in range(len(L)) for j in range(len(L))),
                 reverse=True)[:6]
    print("  largest Lipschitz entries: " + ", ".join(f"{b[1][0]}{b[1][1]:+d}<-{b[2][0]}{b[2][1]:+d} {b[0]:.4g}"
                                                         for b in big))
    print(f"  contraction constant max_i (L w)_i / w_i <= {float(out['lam']):.4f}; F + L w < w: {out['self1']}; "
          f"refined set inside and mapped into itself: {out['self2']}")
    if out["ok"]:
        rad = {t: max(float(eta_radius(out, cp, st.R)) for cp in comps if cp[0] == t) for t in "pvc" if
               any(cp[0] == t for cp in comps)}
        print("  => unique fixed point; |eta| at |z| = R: " + ", ".join(f"{t} {v:.2e}" for t, v in rad.items()))
    else:
        print("  NOT PROVED")


# ---------------------------------------------------------------- the start set at s0
def disc(e):
    r = arb(0, e)
    return acb(r, r)


def horner(cs, z, zero):
    acc = zero
    for c in reversed(cs):
        acc = acc * z + c
    return acc


def start_set(xs, M, h, cauchy, rho0, r0="0.07", arg="3pi/8", sector=None, N=None, k=None, mutate=None,
              kernel="sharp", prec=512, verbose=True):
    """values at the centre and derivative columns over X = xs +- h at u0 = rho0 e^{-2 i arg}, in the u-system's
    variables [P_mu, V_mu, KAPPA (C_nu - delta), w, A]. Returns (ok, vals, cols, info)."""
    ctx.prec = prec
    odd, even = harmonics(M)
    hh, cc = arb(h), arb(cauchy)
    xp = [acb(arb(v)) for v in xs]
    xb = [acb(arb(v) + arb(0, hh)) for v in xs]
    xc = [acb(arb(v) + arb(0, hh + cc), arb(0, cc)) for v in xs]
    stP = Setup(xp, M, r0, arg, sector, N, k, True, kernel, mutate, prec)
    outP = prove(stP, verbose, "centre: ")
    stC = Setup(xc, M, r0, arg, sector, stP.N, stP.k, True, kernel, mutate, prec)
    outC = prove(stC, verbose, f"complex box +-({h} + {cauchy})(1 + i): ")
    info = {"N": stP.N, "k": stP.k, "point": outP, "cbox": outC}
    if not (outP["ok"] and outC["ok"]):
        return False, None, None, info
    if mutate == "rho0":
        rho0 = float(rho0) - 1
    u0 = arb(rho0) * (I * (-2 * stP.arg)).exp()
    z0 = 1 / u0
    az = up(abs(z0).upper() if hasattr(abs(z0), "upper") else arb(abs(z0)))
    if not az <= stP.R:
        raise ValueError("the start point lies outside |z| <= R")
    vals = [stP.P[mu](z0) + disc(eta_radius(outP, ("p", mu), az)) for mu in odd]
    vals += [stP.V[mu](z0) + disc(eta_radius(outP, ("v", mu), az)) for mu in odd]
    vals += [KAPPA * (stP.C[nu](z0) - (1 if nu == 0 else 0) + disc(eta_radius(outP, ("c", nu), az))) for nu in even]
    # derivatives of Y_N over X by jets; of eta by Cauchy's estimate on the polydisc of radius c inside the box
    m = M + 1
    t0 = time.time()
    xj = [Jet(xb[i], [acb(1 if i == j else 0) for j in range(m)]) for i in range(m)]
    Fj, wj, sj = profile(xj, M)
    zero, one = Jet(acb(0), [acb(0)] * m), Jet(acb(1), [acb(0)] * m)
    d, q, g = formal_series(Fj, sj, M, stP.N, zero, one, True, stP.gp)
    Pj = {mu: horner([d[j][mu] for j in range(stP.N)], z0, zero) for mu in odd}
    Qj = {mu: horner([q[j][mu] for j in range(stP.N)], z0, zero) for mu in odd}
    Cj = {nu: horner([g[j][nu] for j in range(stP.N)], z0, zero) for nu in even}
    Vj = {mu: Qj[mu] - (sj[mu] + 1) * Pj[mu] for mu in odd}
    cols = [[] for _ in range(m)]
    for i in range(m):
        cols[i] += [Pj[mu].d[i] + disc(up(eta_radius(outC, ("p", mu), az) / cc)) for mu in odd]
        cols[i] += [Vj[mu].d[i] + disc(up(eta_radius(outC, ("v", mu), az) / cc)) for mu in odd]
        cols[i] += [KAPPA * (Cj[nu].d[i] + disc(up(eta_radius(outC, ("c", nu), az) / cc))) for nu in even]
    # the u-system's last two states, w and A, and their derivatives (exact)
    winf = acb(2 * arb.pi() / arb(PERIOD))
    vals += [stP.w, xp[0]]
    for i in range(m):
        cols[i] += [winf if i == 1 else acb(0), acb(1 if i == 0 else 0)]
    info.update(z0=z0, t_jets=time.time() - t0)
    if verbose:
        print(f"jets over the real box: {info['t_jets']:.1f} s", flush=True)
    return True, vals, cols, info


def rad(z):
    return max(float(z.real.rad()), float(z.imag.rad()))


def stored_set(path, key="0"):
    """the stored C^1 start set of lemma6b1_cover_M.py: point values c + Q q and derivative columns cV + QV qV"""
    with open(path) as fh:
        d = json.load(fh)
    st = d["sets"][key]
    n = len(st["c"])

    def pa(p):
        return acb(arb(p[0]), arb(p[1]))
    c = [pa(p) for p in st["c"]]
    q = [arb(v) for v in st["q"]]
    Qq = [sum((arb(st["Q"][i][j]) * q[j] for j in range(2 * n)), arb(0)) for i in range(2 * n)]
    vals = [c[i] + acb(Qq[i], Qq[n + i]) for i in range(n)]
    m = len(st["cV"][0])
    cols = []
    for jj in range(m):
        qv = [arb(st["qV"][i][jj]) for i in range(2 * n)]
        Qv = [sum((arb(st["QV"][i][j]) * qv[j] for j in range(2 * n)), arb(0)) for i in range(2 * n)]
        cols.append([pa(st["cV"][i][jj]) + acb(Qv[i], Qv[n + i]) for i in range(n)])
    return d, vals, cols


def compare(M, vals, cols, path, verbose=True):
    """overlap of every component (real and imaginary part) of the values and the columns with the stored set"""
    odd, even = harmonics(M)
    names = [f"P{mu:+d}" for mu in odd] + [f"V{mu:+d}" for mu in odd] + [f"c{nu:+d}" for nu in even] + ["w", "A"]
    d, svals, scols = stored_set(path)
    if len(svals) != len(names) or len(scols) != len(cols):
        raise ValueError("the stored set has another dimension")
    miss, worst, inside = [], 0.0, 0
    rows = []
    for i, nm in enumerate(names):
        a, b = vals[i], svals[i]
        ov = a.real.overlaps(b.real) and a.imag.overlaps(b.imag)
        inside += b.contains(a)
        dist = float(abs(a.mid() - b.mid()).upper())
        if not ov:
            miss.append(nm)
        rows.append((nm, ov, dist, rad(a), rad(b)))
    cmiss, cworst = [], 0.0
    for jj in range(len(cols)):
        for i, nm in enumerate(names):
            a, b = cols[jj][i], scols[jj][i]
            ov = a.real.overlaps(b.real) and a.imag.overlaps(b.imag)
            if not ov:
                cmiss.append(f"d{nm}/dx{jj}")
            if rad(a) + rad(b) > 0:
                cworst = max(cworst, float(abs(a.mid() - b.mid()).upper()) / (rad(a) + rad(b)))
    if verbose:
        print(f"stored set {path}: centre {d['x'][0][:14]}.., h = {d['h']}")
        print("  component, overlap, |mid - stored mid|, radius, stored radius")
        for r in rows:
            print(f"  {r[0]:4s} {str(r[1]):5s} {r[2]:.2e} {r[3]:.2e} {r[4]:.2e}")
        print(f"  values: {len(names) - len(miss)} of {len(names)} overlap; {inside} of this run's balls lie inside "
              f"the stored ones")
        print(f"  derivative columns: {len(names) * len(cols) - len(cmiss)} of {len(names) * len(cols)} overlap; "
              f"largest |mid difference| / (sum of radii) {cworst:.3g}")
        if miss or cmiss:
            print("  MISSES: " + ", ".join((miss + cmiss)[:12]) + (" ..." if len(miss + cmiss) > 12 else ""))
    return not (miss or cmiss), {"rows": rows, "value_misses": miss, "column_misses": cmiss, "inside": inside,
                                 "column_worst": cworst}


# ---------------------------------------------------------------- planted truth: the linear problem
def planted(M=5, xs=None, r0="0.07", arg="3pi/8", rho0=405, mutate=None, prec=512, verbose=True):
    """no gravity: P_mu = 2 F_mu x^alpha U(alpha, 1/2, x) and V_mu = 2 z P_mu', x = -1/(2z), alpha = (1 + s_mu)/2"""
    ctx.prec = prec
    odd, _ = harmonics(M)
    xs = xs or {1: ["2.5421480475", "0.7769940136"],
                3: ["2.726475886", "1.015603095", "0.144419478488", "0.011087425887"],
                5: ["2.718579189", "0.999228404", "0.149203574672", "0.014719052460", "0.008405346879",
                    "0.001470345176"]}[M]
    st = Setup([acb(arb(v)) for v in xs], M, r0, arg, None, None, None, False, "sharp",
               mutate if mutate in ("radius", "coef") else None, prec)
    out = prove(st, verbose, "linear problem: ")
    if not out["ok"]:
        return False
    z0 = 1 / (arb(rho0) * (I * (-2 * st.arg)).exp())
    az = up(arb(1) / arb(rho0))
    x = -1 / (2 * z0)
    ok = True
    worst = 0.0
    for mu in odd:
        al = (1 + st.s[mu]) / 2
        if mutate == "truth":
            al = al * (1 + acb("1e-12"))
        U0 = x.hypgeom_u(al, acb(0.5))
        U1 = x.hypgeom_u(al + 1, acb(1.5))
        Pt = 2 * st.F[mu] * x ** al * U0
        Vt = 2 * st.F[mu] * al / z0 * x ** (al - 1) * (U0 - x * U1)
        Pm = st.P[mu](z0) + disc(eta_radius(out, ("p", mu), az))
        Vm = st.V[mu](z0) + disc(eta_radius(out, ("v", mu), az))
        for nm, mine, truth in (("P", Pm, Pt), ("V", Vm, Vt)):
            ov = mine.real.overlaps(truth.real) and mine.imag.overlaps(truth.imag)
            ok = ok and ov
            dist = float(abs(mine.mid() - truth.mid()).upper())
            worst = max(worst, dist)
            if verbose:
                print(f"  {nm}{mu:+d}: overlap {ov}, |mid - Kummer mid| {dist:.2e}, radius {rad(mine):.2e}, Kummer's "
                      f"radius {rad(truth):.2e}")
    print(f"PLANTED (linear problem, M = {M}): {'PASS' if ok else 'FAIL'}; largest midpoint distance {worst:.2e}")
    return ok


# ---------------------------------------------------------------- two truncations of the full problem
def two_truncations(xs, M, N1, N2, r0="0.07", arg="3pi/8", rho0=405, mutate=None, prec=512, verbose=True):
    """the fixed points for N1 < N2 terms are the same solution, so eta_N1 = Y_N2 - Y_N1 + eta_N2 is known to
    eta_N2's radius: at the start point and at |z| = R the two enclosures must overlap, and the ratio of the real and
    imaginary parts of Y_N2 - Y_N1 to the sum of the radii (at most 1 for an overlap) says how tight the bounds are.
    With --mutate=forcing the N1 forcing is divided by 1e3 first, and the check must fail."""
    ctx.prec = prec
    x = [acb(arb(v)) for v in xs]
    outs = []
    for N in (N1, N2):
        st = Setup(x, M, r0, arg, None, N, None, True, "sharp", None, prec)
        if mutate == "forcing" and N == N1:
            st.Fv = {cp: up(v / 1000) for cp, v in st.Fv.items()}
        outs.append(prove(st, verbose, f"N = {N}: "))
        if not outs[-1]["ok"]:
            print("TWO TRUNCATIONS: FAIL (no contraction)")
            return False
    o1, o2 = outs
    s1, s2 = o1["st"], o2["st"]
    ok, tight = True, 0.0
    for name, t in (("the start point", arb(1) / arb(rho0)), ("|z| = R", s1.R)):
        z = t * (I * (2 * s1.arg)).exp()
        tt = up(t)
        worst = 0.0
        for cp in s1.comps:
            pol1 = {"p": s1.P, "v": s1.V, "c": s1.C}[cp[0]][cp[1]]
            pol2 = {"p": s2.P, "v": s2.V, "c": s2.C}[cp[0]][cp[1]]
            a = pol1(z) + disc(eta_radius(o1, cp, tt))
            b = pol2(z) + disc(eta_radius(o2, cp, tt))
            o = a.real.overlaps(b.real) and a.imag.overlaps(b.imag)
            ok = ok and o
            dl = pol2(z) - pol1(z)
            ratio = max(float(abs(dl.real).upper()), float(abs(dl.imag).upper())) / float(
                eta_radius(o1, cp, tt) + eta_radius(o2, cp, tt))
            worst = max(worst, ratio)
            if verbose and not o:
                print(f"  {cp} at {name}: no overlap, ratio {ratio:.3g}")
        print(f"  at {name}: largest |Re, Im (Y_N2 - Y_N1)| / (radius_N1 + radius_N2) over the components "
              f"{worst:.3f} (an overlap needs <= 1)")
        tight = max(tight, worst)
    print(f"TWO TRUNCATIONS (N = {N1} against N = {N2}): {'PASS' if ok else 'FAIL'}; largest ratio {tight:.3f}")
    return ok


# ---------------------------------------------------------------- command line
def sball(z, n=50):
    return [z.real.str(n, radius=True, more=True), z.imag.str(n, radius=True, more=True)]


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts.get("M", "5"))
    mutate = opts.get("mutate")
    prec = int(opts.get("prec", "512"))
    N = int(opts["N"]) if "N" in opts else None
    k = int(opts["k"]) if "k" in opts else None
    kernel = opts.get("kernel", "sharp")
    print(f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}, {prec} bits"
          + (f", MUTATION {mutate}" if mutate else ""))
    t0 = time.time()
    if "planted" in opts:
        ok = planted(M, opts["x"].split(",") if "x" in opts else None, opts.get("r0", "0.07"),
                     opts.get("arg", "3pi/8"), float(opts.get("rho0", "405")), mutate, prec)
        sys.exit(0 if ok else 3)
    if "two" in opts:
        N1, N2 = (int(v) for v in opts["two"].split(","))
        with open(opts["compare"]) as fh:
            xs = opts["x"].split(",") if "x" in opts else json.load(fh)["x"]
        ok = two_truncations(xs, M, N1, N2, opts.get("r0", "0.07"), opts.get("arg", "3pi/8"),
                             float(opts.get("rho0", "405")), mutate, prec)
        sys.exit(0 if ok else 3)
    if "start" in opts:
        path = opts.get("compare")
        if "x" in opts:
            xs = opts["x"].split(",")
        else:
            with open(path) as fh:
                xs = json.load(fh)["x"]
        ok, vals, cols, info = start_set(xs, M, opts.get("h", "1e-18"), opts.get("cauchy", "1e-3"),
                                         float(opts.get("rho0", "405")), opts.get("r0", "0.07"),
                                         opts.get("arg", "3pi/8"), opts.get("sector"), N, k, mutate, kernel, prec)
        res = {"M": M, "x": xs, "h": opts.get("h", "1e-18"), "cauchy": opts.get("cauchy", "1e-3"),
               "rho0": opts.get("rho0", "405"), "mutate": mutate, "contraction": ok, "N": info["N"], "k": info["k"],
               "python_flint": flint.__version__, "flint": flint.__FLINT_VERSION__, "prec": prec}
        for nm in ("point", "cbox"):
            o = info[nm]
            res[nm] = {"ok": o["ok"], "low_ok": o["low_ok"], "lam0": o.get("lam0"),
                       "lam": float(o["lam"]) if "lam" in o else None}
        if ok:
            res["vals"] = [sball(v) for v in vals]
            res["cols"] = [[sball(v) for v in col] for col in cols]
            res["val_rad"] = max(rad(v) for v in vals)
            res["col_rad"] = max(rad(v) for col in cols for v in col)
            print(f"start set at |u| = {opts.get('rho0', '405')}: value radius {res['val_rad']:.2e}, derivative "
                  f"radius {res['col_rad']:.2e}")
            if path:
                okc, cmp_ = compare(M, vals, cols, path)
                res["compare"] = {k_: v for k_, v in cmp_.items() if k_ != "rows"}
                res["compare"]["rows"] = [[r[0], r[1], r[2], r[3], r[4]] for r in cmp_["rows"]]
                ok = ok and okc
        res["ok"] = ok
        res["sec"] = time.time() - t0
        if "json" in opts:
            with open(opts["json"], "w") as fh:
                json.dump(res, fh, indent=1)
        print(f"START SET: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 3)
    xs = opts["x"].split(",")
    if "cbox" in opts:
        c, h = arb(opts["cbox"]), arb(opts.get("box", "0"))
        x = [acb(arb(v) + arb(0, h + c), arb(0, c)) for v in xs]
    elif "box" in opts:
        x = [acb(arb(v) + arb(0, arb(opts["box"]))) for v in xs]
    else:
        x = [acb(arb(v)) for v in xs]
    st = Setup(x, M, opts.get("r0", "0.07"), opts.get("arg", "3pi/8"), opts.get("sector"), N, k, True, kernel, mutate,
               prec)
    out = prove(st)
    print(f"CONTRACTION: {'PROVED' if out['ok'] else 'NOT PROVED'} ({time.time() - t0:.0f} s)")
    sys.exit(0 if out["ok"] else 3)


if __name__ == "__main__":
    main()
