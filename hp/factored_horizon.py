#!/usr/bin/env python3
"""Issue #1035: the factored horizon segment, stage 1 of notes/FACTORED_HORIZON.md section 5. A prototype in mpmath of
the horizon block alone, with the mass mode factored at the last node (section 3), its loop gain, the sweeps to
quad-double precision and P1 (equivalence).

The layout is #693's two-node layout, the rung layout of hp/reseed.py (`Reseeder.refine_o`, `rung_solver`): the centre
data, node 0, the chain's last node as node 1, and Delta. Its horizon block is the horizon series summed at
u0 = 1 - x_K minus node 1. It depends only on (m_h, a_h, Delta) and node 1, which are read from a stored state.

The old series (`series` with a0) transcribes model.rs `horizon_series` order by order at mp.prec bits, with hp's
constants: n = D - 2 and the folded odd d/dtheta rounded to double-double as hpkernel.driver makes them (make_solver,
dmat_odd), omega = 2 pi / Delta with 2 pi rounded to double-double (DD::TWO_PI), u0 = 1 - f64(x_K) exactly. It works
in u unscaled (hp's scaling by a power of two is exact). Each periodic solve (omega D + diag kappa_k) p_k = -r is
exact at the working precision, as hp refines its double-double LU to the lane's precision (`psolve`). The old rows
are the series at u0 minus node 1 (`horizon_state`). `horizon_lus` is not needed: it only preconditions those solves.

The factored segment (`series` with B, FACTORED_HORIZON 3) carries B = 1/a^2 and C = c a^2. With B given, one forward
pass forms C (x C_x = C q), c by its own recursion with the coupling (n - 1) C (1 - B) (stable: c = C B directly would
need B one order ahead, and c's recursion with c(a^2 - 1) = C - c carries the mass mode), and p, m as hp does, hence
q = p^2 + m^2 at orders 0..K. From q (`update`):
  B_sl  the minimal solution of B's rows k = 0..K with b_(K+1) = 0, the lower-Hessenberg system of 3.3 per tau point,
        solved by eliminating the superdiagonal from the last row up (backward recurrence in boundary-value form);
  chi   the mass mode, forward from chi_0 = 1, divided by its sum at u0;
  gamma per tau point, so that B = B_sl + gamma chi has its datum a_K at the node. --gamma a (the default) fixes it by
        a's sum at u0 equal to a_K, with a = B^(-1/2) from B's coefficients (J. C. P. Miller's power recurrence) and
        Newton's method per point. That makes the factored rows hp's rows composed with the change a_h -> a_K, exactly
        (3.3). --gamma B fixes it by B's sum at u0 = 1/a_K^2, another discrete problem at the truncation's level.
One sweep is that forward pass and that update. The sweeps repeat to the fixed point, plainly or with Anderson mixing
(Anderson 1965; type II as in Walker and Ni 2011) on y_k = b_k u0^k. The factored rows are a_K - a(node) and the p, m,
c series at u0 minus the node.

Commands, from hp/ (STATE an .npz of the chain, read with numpy; parts zh, zl, z2, ... summed exactly):
    python3 factored_horizon.py selftest [--mutate=nh|bsl|sign|miller|nofp]
    python3 factored_horizon.py check STATE --bound B [--mutate=nh]
    python3 factored_horizon.py gain STATE --ref REF.json [--start slaved|naive|warm] [--accel plain|anderson]
        [--prec 512] [--mem 5] [--warm 1e-20] [--maxit 40]
    python3 factored_horizon.py p1 STATE [--ref REF.json] [--prec 512] [--mutate=sign] [--gamma a|B] [--recip dd|exact]
        [--move 1e-8]
    python3 factored_horizon.py loop STATE [--prec 512] [--its 5] [--h 1e-100] [--orders 60,100,125]
    python3 factored_horizon.py curvjac STATE --prec 448 --cache CACHE.json
    python3 factored_horizon.py curv STATE --prec 448 --cache CACHE.json --move D|m1|m3 [--plant CURV_D.json]
        [--plant_c C] [--mutate=r0|plant1]
    python3 factored_horizon.py curvx STATE --prec 288 --ref REF.json --t 0|1e-20|-1e-20
    python3 factored_horizon.py curvxq X_t.json X_minus_t.json X_0.json --t 1e-20 --curv CURV_D.json
    python3 factored_horizon.py raisecheck STATE --ks 315 --prec 512 --cache CACHE.json [--mutate=linear]
--out FILE.json writes the command's record. Exit 1 when a check fails.

selftest, planted truth on a small problem (n = 100, M = 8, K = 40, x_K = 0.7, a_h the slaved value moved by
1e-3 / G, nodes at the old series' own sums, 256 bits). Criteria fixed before the first run; the planted problem was amended
once (n = 30 with generic a_h: its series diverge at u0, the coefficients reach 1e80, and S3 read rounding):
  S0  the old series solves model.rs's header equations at u0/8: residual below 1e-20 (truncation about 1e-29).
      Control --mutate=nh (n/2 scaled by 1 + 1e-8 in the p and m equations) must fail.
  S1  B_sl solves B's rows 0..K with b_(K+1) = 0 to 2^-(prec-24) of the rows' largest term. Control --mutate=bsl
      (q_0 dropped from the diagonal) must fail.
  S2  chi's coefficients equal those of (1 - u)^-(n-1) exp(int_0^u q/(1 - s) ds), formed independently by series
      exp and log, to 2^-(prec-24) relative. Control --mutate=sign (the transport's sign flipped) must fail.
  S3  a = B^(-1/2) from Miller's recurrence: (a a B)_k = delta_k0 to 2^-(prec-24). Control --mutate=miller (weights
      j - k, the recurrence of B^(-1)) must fail.
  S4  P1 on the planted problem: with a_K the old series' a at u0, the factored rows equal the old rows to
      100 2^-prec G. Control --mutate=sign must fail.
  S5  the plain sweeps and Anderson's both reach the fixed point to 2^-(prec-24). Control --mutate=nofp (one sweep)
      must fail.
  S6  the closure b_(K+1) = 0 is immaterial: b_(K+1) = 1 gives the same rows to 100 2^-prec G. Control --mutate=sign
      must fail.

check, the transcription against a stored state: at a converged state the old rows (m, a and c, and p projected on
the harmonics k <= kfreeze) vanish to the state's own floor. Pass: their largest |row| is below --bound, ten times
the state's recorded rows: #693's octo-double rows of #603's states (raw I693-mac/onsol.jsonl; D 225.13 3.2e-94,
D 241.96 2.0e-64), T81's max|F| at D 73.0 (2.52e-53, conv_pin5_D73.0000.jsonl), the corrector's tolerance at
D 242.36 (6.99e-20, I461-win2/crawlJ/anm_run.jsonl step 8). Control --mutate=nh must fail. --recip exact uses an
exact 1/k in place of hp's recip_int (the rows then miss by the mass mode's response to that rounding).

gain: the reference fixed point (Anderson from the slaved start, or with --ref_start old from B* = 1/a^2 of the old
series, to |g(y) - y| <= 1e-8 eps_qd, cached in --ref FILE with its state, precision, gamma and 1/k, and refused for any
other; --start none stops there); then one run, --accel plain or anderson (memory --mem), from --start slaved (beta = 0,
gamma = 0), naive (q = 0 and gamma from the datum) or warm (the fixed point of data moved by --warm relative, which must
converge to 1e-8 eps_qd within --maxit, as its last iterate is its own reference), to 1e-4 eps_qd or --maxit sweeps. Per
sweep: the largest error of the rows and of y against the reference. Reported: the error ratio per sweep and the sweeps
to eps_qd = 2^-212. Stage 1 passes on the sweeps if 5 or fewer reach eps_qd.

p1: P1 of FACTORED_HORIZON 5, at --prec bits. a_K is the old series' a at u0, the fixed point Anderson's to
2^-(prec-16) or --maxit sweeps (from --ref's fixed point if given). Pass: max |R_fact - R_old| <= eps_qd G (qd's old
floor, the issue's criterion), and the sharp reading <= 100 2^-prec G (the old series' own floor at this precision).
Control --mutate=sign must fail. At a stored state gamma is the mass mode's amplitude left by the convergence, 1e-28
or less, so chi_hat hardly enters the rows and the control cannot fire there; --move 1e-8 moves a_h by 1e-8 / G,
gamma about 1e-8. --recip exact puts an exact 1/k in both series (hp's rounded 1/k has no counterpart in B's rows).

curvjac, curv, curvx, curvxq: the curvature test of 5.2 (stage 1b). With an exact 1/k, the base is the stored state
with a_h re-solved so that a's sum at u0 equals the node's a (curvjac, which also caches the M x M Jacobian of a(u0) in
a_h). curv reads Q = [R(t) + R(-t) - 2 R(0)] / t^2 over the p, m and c rows along one move, Q_old at fixed a_h over
CURV_PAIRS_OLD and Q_fact at fixed a_K (method A: a_h re-solved per t) at CURV_PAIR, with the t^2 checks, the ratio and
its gate; with --plant (a curv record along D) it adds c t^2 to Delta and checks the planted term's recovery to 1%,
c from the record (the gate-1 level) or --plant_c.
Exit 1 when Q_fact or every pair of Q_old fails its t^2 check, or the plant is not recovered. Controls that must fail:
--mutate=r0 shifts R(0)'s p, m and c rows by 1e-40, so Q gains 2e-40 / t^2 and Q_fact's t^2 check fires;
--mutate=plant1 with --plant plants the term at +t only, so half of it is recovered.
curvx and curvxq repeat Q_fact along D with the stage-1 fixed point, the cross-check.

raisecheck: 5.3's Mac-side check of the factored raise's projection. At order --ks with hp's 1/k, the rung of the
chain's projection (the m and a rows for m_h and a_h, the node held) is solved from the stored data, then along Delta
moves of 1e-16, 1e-15 and 1e-14 from the linear predictor (the corrector's own start) and from the second-order one
(factored_raise.py's, its t^2 term from a second difference at tau 1e-20 and 1e-19). It passes when the second-order
start converges at 1e-14 and the t^2 term holds to 10%; --mutate=linear judges the linear start instead.

loop: the sweep map at a converged state (gamma by B's sum): B* = 1/a^2 from the old series, |g(B*) - B*|, gamma*,
the power iteration of the linearized map from a random perturbation of size --h, and the forward pass's response in
p to y moved by 1e-30 at single orders. It prints; it checks nothing.
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

EPS_QD = 2.0 ** -212           # quad-double's unit roundoff (plot_algorithm.py)
TWO_PI_DD = (6.283185307179586, 2.4492935982947064e-16)  # multidouble dd.rs DD::TWO_PI
MUT_NH = mp.mpf("1e-8")        # --mutate=nh: n/2 times 1 + MUT_NH in the p and m equations


# ------------------------------------------------------------------ the segment's constants
def dd_round(v):
    """v rounded to double-double as multidouble's dd_from_mp (hi the nearest f64, lo the nearest f64 to the rest),
    as the pair (hi, lo): the caller sums them at its own precision (at 40 digits hi + lo is rounded to 2^-133, which
    the mass mode carries to the node as 1e-30 at D 73 and 1e-7 at D 242)"""
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return hi, lo


def dd_val(hl):
    return mp.mpf(hl[0]) + mp.mpf(hl[1])


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def _dd_add(a, b):
    """multidouble dd.rs `impl Add for DD`, on (hi, lo) pairs of floats"""
    s, e = _two_sum(a[0], b[0])
    t, f = _two_sum(a[1], b[1])
    s, e = _quick_two_sum(s, e + t)
    return _quick_two_sum(s, e + f)


def _dd_mul_f(a, b):
    """multidouble dd.rs `DD::mul_f`"""
    p = a[0] * b
    e = math.fma(a[0], b, -p)
    return _quick_two_sum(p, math.fma(a[1], b, e))


_RECIP = {}


def recip_int(k):
    """model.rs recip_int: DD::ONE / DD::from_f64(k) by multidouble's DD division (dd.rs `impl Div for DD`), as the
    exact mpf of its two parts. hp's series multiply by it (`muls(ik1)`) in place of dividing by k, in every lane, so it
    is part of hp's discrete problem; its rounding, about 2^-106 relative, excites the mass mode like any error of the
    a recursion (with an exact 1/k the old rows at T81's D 73.0 state read 4e-30 and at #603's D 241.96 state 7e-7)."""
    if k not in _RECIP:
        b = (float(k), 0.0)
        q1 = 1.0 / b[0]
        r = _dd_add((1.0, 0.0), tuple(-v for v in _dd_mul_f(b, q1)))
        q2 = r[0] / b[0]
        r = _dd_add(r, tuple(-v for v in _dd_mul_f(b, q2)))
        q3 = r[0] / b[0]
        h, lo = _quick_two_sum(q1, q2)
        _RECIP[k] = _dd_add((h, lo), (q3, 0.0))
    return dd_val(_RECIP[k])


class Seg:
    """The horizon block of the two-node layout: n, the folded odd d/dtheta, u0, the series order K and the M points.
    D: an mpf or a string (D_str); x_k: the last node (f64)."""

    def __init__(self, D, M, ks, x_k, kfreeze=None):
        self.M, self.K, self.kfreeze = int(M), int(ks), kfreeze
        with mp.workdps(60):
            Dm = mp.mpf(D)
        with mp.workdps(40):                           # make_solver: n = D - 2 at 40 digits, then double-double
            n_dd = dd_round(mp.mpf(Dm) - 2)
        self.n = dd_val(n_dd)
        self.D_str = str(D)
        self.n1 = self.n - 1
        self.nh = self.n / 2
        self.x_k = float(x_k)
        self.u0 = 1 - mp.mpf(self.x_k)                 # exact: DD::ONE - DD::from_f64(x_K)
        self.dmat = dmat_odd(self.M)
        self.two_pi = mp.mpf(TWO_PI_DD[0]) + mp.mpf(TWO_PI_DD[1])
        self.G = mp.mpf(self.x_k) ** (-(self.n - 1))   # the mass mode's gain x_K^-(n-1)

    def omega(self, delta):
        return self.two_pi / delta

    def dtau(self, x, omega):
        """omega D x (model.rs dtau at rate zero)"""
        return [omega * mp.fdot(row, x) for row in self.dmat]


def dmat_odd(M):
    """hpkernel.driver.dmat_odd: the folded d/dtheta for odd fields on the half grid of 2M points, each entry at 40
    digits rounded to double-double"""
    out = []
    with mp.workdps(40):
        N = 2 * M
        for i in range(M):
            row = []
            for j in range(M):
                v = mp.mpf(0)
                for jj, sgn in ((j, 1), (j + M, -1)):
                    if jj != i:
                        v += sgn * mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
                row.append(dd_round(v))
            out.append(row)
    return [[dd_val(hl) for hl in row] for row in out]


def plow_odd(M, kf):
    """the projector onto the odd harmonics k <= kf of an odd field on the half grid (hpkernel.driver.plow_odd, here
    at the working precision, as multishoot forms it in octo-double)"""
    N = 2 * M
    th = [2 * mp.pi * j / N for j in range(M)]
    return [[sum(4 * mp.cos(k * (th[i] - th[j])) / N for k in range(1, kf + 1, 2)) for j in range(M)] for i in range(M)]


# ------------------------------------------------------------------ series arithmetic (per tau point)
def cs(x, y, k, lo=0, hi=None, w=None):
    """sum_{i=lo..hi} w(i) x_i y_(k-i) at each point (multidouble's cs, cs_w); x, y lists of orders of point lists"""
    hi = k if hi is None else hi
    idx = range(lo, hi + 1)
    M = len(y[0])
    if hi < lo:
        return [mp.mpf(0)] * M
    if w is None:
        return [mp.fdot([x[i][j] for i in idx], [y[k - i][j] for i in idx]) for j in range(M)]
    return [mp.fdot([w(i) * x[i][j] for i in idx], [y[k - i][j] for i in idx]) for j in range(M)]


def vadd(*xs):
    return [mp.fsum(t) for t in zip(*xs)]


def vsub(x, y):
    return [a - b for a, b in zip(x, y)]


def vscale(x, s):
    return [a * s for a in x]


def horner(coefs, u):
    """sum_k c_k u^k at each point"""
    M = len(coefs[0])
    out = [mp.mpf(0)] * M
    for c in reversed(coefs):
        out = [o * u + ck for o, ck in zip(out, c)]
    return out


def psolve(seg, kappa, rhs, omega):
    """(omega D + diag kappa) f = rhs to the working precision: an f64 inverse and refinement with the residual in
    mpmath (each step gains about 13 digits here; hp refines its double-double LU the same way, `psolve`)"""
    M = seg.M
    A = np.array([[float(omega * seg.dmat[i][j]) for j in range(M)] for i in range(M)])
    A[np.diag_indices(M)] += [float(k) for k in kappa]
    Ai = np.linalg.inv(A)
    f = [mp.mpf(0)] * M
    if max(abs(r) for r in rhs) == 0:
        return f
    tol = mp.mpf(2) ** (-mp.mp.prec + 2)
    for _ in range(100):
        Df = seg.dtau(f, omega)
        r = [rhs[i] - (Df[i] + kappa[i] * f[i]) for i in range(M)]
        rm = max(abs(v) for v in r)
        if rm == 0:
            return f
        d = Ai @ np.array([float(v / rm) for v in r])
        dx = [mp.mpf(float(d[i])) * rm for i in range(M)]
        f = [f[i] + dx[i] for i in range(M)]
        if max(abs(v) for v in dx) <= tol * max(abs(v) for v in f):   # the correction below the rounding of f
            return f
    raise RuntimeError("psolve: refinement did not converge")


# ------------------------------------------------------------------ the horizon series, old and factored
RECIP_DD = [True]             # --recip dd (hp's recip_int, the default) or exact (1/k at the working precision)


def series(seg, m0, omega, a0=None, B=None, mutate="", recip=None):
    """model.rs horizon_series to order K in u, at the data m0 and either a0 (the old series: a's recursion, c from
    c (a^2 - 1)) or B (the factored forward pass: C's recursion and c from (n - 1) C (1 - B), B given at orders 0..K
    as beta = B - 1, so that 1 - B = -beta does not cancel; the alpha form of HALPHA_ON does the same for a).
    Returns dict p, m, c (orders 0..K), a or C (0..K), q (0..K), each a list of point lists."""
    M, K = seg.M, seg.K
    n1 = seg.n1
    nh = seg.nh * (1 + MUT_NH) if mutate == "nh" else seg.nh
    fact = B is not None
    one = [mp.mpf(1)] * M
    zero = [mp.mpf(0)] * M
    c = [one]
    m = [list(m0)]
    p, dp, q, vv = [], [], [], []
    dm = [seg.dtau(m0, omega)]
    if fact:
        C = [[1 / (1 + b) for b in B[0]]]              # c_0 = 1: C_0 = 1 / b_0, B held as beta = B - 1
    else:
        a = [list(a0)]
        a2 = [[v * v for v in a0]]

    def wf(i):                                         # (1 - u) - c
        return [1 - v for v in c[0]] if i == 0 else ([-v - 1 for v in c[1]] if i == 1 else [-v for v in c[i]])

    def zf(i):                                         # c + (1 - u)
        return [v + 1 for v in c[0]] if i == 0 else ([v - 1 for v in c[1]] if i == 1 else list(c[i]))

    ww = [wf(0)]
    zc = [zf(0)]
    izc0 = [1 / v for v in zc[0]]
    for kk in range(K + 1):
        if fact:
            cB = cs(C, B, kk)
            g1 = [-v for v in cB]                      # (C (1 - B))_kk = -(C beta)_kk, without cancellation
        else:
            ca = cs(c, a2, kk)
            g1 = [ca[j] - c[kk][j] for j in range(M)]  # (c (a^2 - 1))_kk
        ik1 = recip_int(kk + 1) if (RECIP_DD[0] if recip is None else recip) else 1 / mp.mpf(kk + 1)
        c.append([(kk * c[kk][j] - n1 * g1[j]) * ik1 for j in range(M)])
        ww.append(vsub(wf(kk + 1), wf(kk)))
        vv.append([(kk + 1) * c[kk + 1][j] - kk * c[kk][j] for j in range(M)])
        zc.append(vsub(zf(kk + 1), zf(kk)))
        # p_kk, with p_kk = 0 in the sums formed before its solve (p + m there is m_kk)
        pz = p + [zero]
        pm = [vadd(p[i], m[i]) for i in range(kk)] + [m[kk]]
        wpu = cs(pz, ww, kk + 1, 0, kk, w=lambda i: i)
        tau = vscale(dp[kk - 1], -1) if kk >= 1 else zero
        r1, r2, r3 = cs(vv, pz, kk), cs(c, pz, kk), cs(c, pm, kk)
        r = [((tau[j] - wpu[j]) + r1[j] + r2[j]) - nh * r3[j] for j in range(M)]
        c1 = c[1]
        kappa = [(c1[j] + 1) * kk + c1[j] + 1 - nh for j in range(M)]
        pk = psolve(seg, kappa, [-v for v in r], omega)
        p.append(pk)
        dp.append(seg.dtau(pk, omega))
        qa, qb = cs(p, p, kk), cs(m, m, kk)
        q.append([qa[j] + qb[j] for j in range(M)])
        if kk == K:
            break
        if fact:
            qC = cs(q, C, kk)
            C.append([(kk * C[kk][j] - qC[j]) * ik1 for j in range(M)])
        else:
            aq, aa2 = cs(a, q, kk), cs(a, a2, kk)
            ag = [aq[j] + n1 * (a[kk][j] - aa2[j]) for j in range(M)]
            a.append([(kk * a[kk][j] - ag[j] / 2) * ik1 for j in range(M)])
            a2.append(cs(a, a, kk + 1))
        tau = vsub(dm[kk], dm[kk - 1]) if kk >= 1 else dm[kk]
        pm = [vadd(p[i], m[i]) for i in range(kk + 1)]
        zmu = cs(m, zc, kk + 1, 0, kk, w=lambda i: i)
        s1, s2, s3 = cs(vv, m, kk), cs(c, m, kk), cs(c, pm, kk)
        rest = [(((tau[j] - zmu[j]) - s1[j]) - s2[j]) + nh * s3[j] for j in range(M)]
        m.append([rest[j] * izc0[j] * ik1 for j in range(M)])
        dm.append(seg.dtau(m[kk + 1], omega))
    out = dict(p=p, m=m, c=c[:K + 1], q=q, vv=vv)
    if fact:
        out["C"] = C
    else:
        out["a"] = a
    return out


# ------------------------------------------------------------------ the factored update: B_sl, chi, gamma
def b_sl(seg, q, mutate="", closure=0):
    """the minimal solution of B's rows k = 0..K in beta = B - 1, (k+1) beta_(k+1) = (n-1+k) beta_k + (q beta)_k + q_k
    (B's rows (k+1) b_(k+1) = (n-1+k) b_k + (q b)_k - (n-1) delta_k0 less those of B = 1), with beta_(K+1) = closure
    (0, the design): the lower-Hessenberg system eliminated from the last row up, then substituted forward. mutate
    'bsl' drops q_0 from the diagonal (the control of S1)."""
    M, K, n1 = seg.M, seg.K, seg.n1
    out = [[None] * M for _ in range(K + 1)]
    for j in range(M):
        qj = [q[k][j] for k in range(K + 1)]
        q0 = 0 if mutate == "bsl" else qj[0]
        # row k over columns 0..k (H_k,l = -q_(k-l), l < k; H_k,k = -(n-1+k) - q_0), right-hand side q_k
        R = [None] * (K + 1)
        s = [mp.mpf(0)] * (K + 1)
        R[K] = [-qj[K - l] for l in range(K)] + [-(n1 + K) - q0]
        s[K] = qj[K] - (K + 1) * mp.mpf(closure)
        for k in range(K - 1, -1, -1):
            f = (k + 1) / R[k + 1][k + 1]
            Rk1 = R[k + 1]
            R[k] = [-qj[k - l] - f * Rk1[l] for l in range(k)] + [-(n1 + k) - q0 - f * Rk1[k]]
            s[k] = qj[k] - f * s[k + 1]
        b = [s[0] / R[0][0]]
        for k in range(1, K + 1):
            b.append((s[k] - mp.fdot(R[k][:k], b)) / R[k][k])
        for k in range(K + 1):
            out[k][j] = b[k]
    return out


def chi_hat(seg, q, mutate=""):
    """the mass mode: (k+1) chi_(k+1) = (n-1+k) chi_k + (q chi)_k from chi_0 = 1 (forward, the dominant solution), to
    order K, divided by its sum at u0. mutate 'sign' flips the transport, (q chi)_k -> -(q chi)_k."""
    M, K, n1 = seg.M, seg.K, seg.n1
    sg = -1 if mutate == "sign" else 1
    ch = [[mp.mpf(1)] * M]
    for k in range(K):
        qc = cs(q, ch, k)
        ch.append([((n1 + k) * ch[k][j] + sg * qc[j]) / (k + 1) for j in range(M)])
    S = horner(ch, seg.u0)
    return [[ch[k][j] / S[j] for j in range(M)] for k in range(K + 1)], [S[j] / ch[0][j] for j in range(M)]


def miller(B, chi=None, mutate=""):
    """a = B^(-1/2) from B's coefficients, B given as beta = B - 1 (J. C. P. Miller's recurrence, y = B^r with r = -1/2:
    k b_0 y_k = sum_{j=1..k} ((r+1) j - k) b_j y_(k-j)) and, given chi, its tangent along B -> B + t chi. Per point,
    the weighted sum as sum j b_j y_(k-j) / 2 - k sum b_j y_(k-j) (two dot products). mutate 'miller' uses the weights
    j - k (the control of S3)."""
    K1, M = len(B), len(B[0])
    half = 1 if mutate == "miller" else mp.mpf(1) / 2
    y = [[None] * M for _ in range(K1)]
    dy = [[None] * M for _ in range(K1)] if chi is not None else None
    for j in range(M):
        b = [B[k][j] for k in range(K1)]
        b0 = 1 + b[0]
        jb = [i * b[i] for i in range(K1)]
        yy = [1 / mp.sqrt(b0)]
        if chi is not None:
            c = [chi[k][j] for k in range(K1)]
            jc = [i * c[i] for i in range(K1)]
            dd = [-yy[0] / (2 * b0) * c[0]]
        for k in range(1, K1):
            rev = yy[k - 1::-1]
            T = half * mp.fdot(jb[1:k + 1], rev) - k * mp.fdot(b[1:k + 1], rev)
            yk = T / (k * b0)
            yy.append(yk)
            if chi is not None:
                drev = dd[k - 1::-1]
                T1 = half * mp.fdot(jc[1:k + 1], rev) - k * mp.fdot(c[1:k + 1], rev)
                T2 = half * mp.fdot(jb[1:k + 1], drev) - k * mp.fdot(b[1:k + 1], drev)
                dd.append((T1 + T2) / (k * b0) - yk * c[0] / b0)
        for k in range(K1):
            y[k][j] = yy[k]
            if chi is not None:
                dy[k][j] = dd[k]
    return y, dy


def update(seg, q, aK, gamma0=None, mode="a", mutate="", closure=0):
    """B = B_sl + gamma chi_hat from q, gamma per point by the datum at the node: mode 'a' (a's sum at u0 = a_K, Newton
    from gamma0 or from mode 'B''s value) or 'B' (B's sum at u0 = 1/a_K^2). Returns (B, gamma, info)."""
    M, K, u0 = seg.M, seg.K, seg.u0
    Bs = b_sl(seg, q, mutate, closure)
    ch, Gm = chi_hat(seg, q, mutate)
    Ssl = horner(Bs, u0)
    gam = [(1 / (aK[j] * aK[j]) - 1) - Ssl[j] for j in range(M)]   # mode 'B'
    if mode == "zero":                                              # B = B_sl: the slaved data (planted only)
        gam = [mp.mpf(0)] * M
    newton = 0
    if mode == "a":
        if gamma0 is not None:
            gam = list(gamma0)
        tol = mp.mpf(2) ** (-mp.mp.prec + 8)
        for newton in range(1, 12):
            B = [[Bs[k][j] + gam[j] * ch[k][j] for j in range(M)] for k in range(K + 1)]
            y, dy = miller(B, ch, mutate)
            Sa, dSa = horner(y, u0), horner(dy, u0)
            step = [(Sa[j] - aK[j]) / dSa[j] for j in range(M)]
            gam = [gam[j] - step[j] for j in range(M)]
            if max(abs(s) for s in step) <= tol * max(1, max(abs(g) for g in gam)):
                break
    B = [[Bs[k][j] + gam[j] * ch[k][j] for j in range(M)] for k in range(K + 1)]
    return B, gam, dict(newton=newton, G_trunc=Gm)


def initial_B(seg, aK, mode="a", mutate="", start="slaved"):
    """the cold starts. 'slaved': B = 1 (beta = 0) and gamma = 0, the slaved horizon value with q = 0 (at a converged
    state gamma is the mass mode's amplitude at the node, 3e-28 at D 73 and 5e-35 at D 242.36, so its datum is nearly
    the slaved value). 'naive': q = 0, so B_sl = 1, and gamma from the datum, B = 1 + gamma chi (gamma = 1/a_K^2 - 1,
    about 1e-3: a mass mode that the p and m series amplify, the start that diverges at D 73)."""
    zero = [[mp.mpf(0)] * seg.M for _ in range(seg.K + 1)]
    if start == "slaved":
        return zero, [mp.mpf(0)] * seg.M, {}
    return update(seg, zero, aK, None, mode, mutate)


def rows_fact(seg, fw, aK, node):
    """the factored rows: p, m, c at u0 minus the node, and a_K minus the node's a"""
    M = seg.M
    pu, mu, cu = horner(fw["p"], seg.u0), horner(fw["m"], seg.u0), horner(fw["c"], seg.u0)
    return (vsub(pu, node[0:M]) + vsub(mu, node[M:2 * M]) + vsub(aK, node[2 * M:3 * M]) + vsub(cu, node[3 * M:4 * M]))


def rows_old(seg, s, node):
    M = seg.M
    vals = [horner(s[f], seg.u0) for f in ("p", "m", "a", "c")]
    return sum((vsub(vals[i], node[i * M:(i + 1) * M]) for i in range(4)), []), vals


def yvec(seg, B):
    """the fixed point's variable: y_k = b_k u0^k (each coefficient's share of B's sum at the node), flattened"""
    u = mp.mpf(1)
    out = []
    for k in range(seg.K + 1):
        out.extend(v * u for v in B[k])
        u *= seg.u0
    return out


def from_y(seg, y):
    M = seg.M
    B, u = [], mp.mpf(1)
    for k in range(seg.K + 1):
        B.append([v / u for v in y[k * M:(k + 1) * M]])
        u *= seg.u0
    return B


def amax(v):
    return max(abs(x) for x in v)


# ------------------------------------------------------------------ the fixed point
def fixed_point(seg, m0, aK, omega, node, B0=None, gam0=None, accel="plain", mem=5, tol=None, maxit=60, ref=None,
                mode="a", mutate="", log=None, closure=0, start="slaved", keep=None):
    """sweeps B -> forward pass -> q -> B_sl, chi, gamma -> B'. accel 'plain' (y' = g(y)) or 'anderson' (memory mem).
    ref (rows, y) of the converged map, to record errors. Stops when |g(y) - y| <= tol or after maxit sweeps.
    Returns dict: B, gamma, fw (the last forward pass), rows (of the last pass), hist, sweeps."""
    tol = mp.mpf(2) ** (-mp.mp.prec + 16) if tol is None else tol
    if B0 is None:
        B0, gam0, _ = initial_B(seg, aK, mode, mutate, start)
    B, gam = B0, gam0
    y = yvec(seg, B)
    Ys, Fs, Gs = [], [], []
    hist = []
    t0 = time.time()
    for s in range(1, maxit + 1):
        fw = series(seg, m0, omega, B=B)
        rows = rows_fact(seg, fw, aK, node)
        Bn, gam, info = update(seg, fw["q"], aK, gam, mode, mutate, closure)
        g = yvec(seg, Bn)
        f = [a - b for a, b in zip(g, y)]
        fn = amax(f)
        rec = dict(sweep=s, fp_res=float(fn), newton=info["newton"], t=round(time.time() - t0, 1))
        if ref is not None:
            rec["err_rows"] = float(amax(vsub(rows, ref[0])))
            rec["err_y"] = float(amax(vsub(y, ref[1])))
        hist.append(rec)
        if keep is not None:
            keep.append((y, rows))
        if log:
            log(rec)
        if fn <= tol:
            break
        if accel == "anderson":
            Ys.append(y)
            Fs.append(f)
            Gs.append(g)
            if len(Fs) > mem + 1:
                Ys.pop(0)
                Fs.pop(0)
                Gs.pop(0)
            y = anderson(Fs, Gs)
        else:
            y = g
        B = from_y(seg, y)
    return dict(B=B, gamma=gam, fw=fw, rows=rows, hist=hist, sweeps=len(hist), y=y)


def anderson(Fs, Gs):
    """type-II Anderson mixing: theta minimizing |f_s - sum theta_i (f_s - f_(s-i))|_2, by modified Gram-Schmidt on
    the differences with a column dropped when it is dependent on the others to 2^-(prec/2); y' = g_s - sum theta_i
    (g_s - g_(s-i))"""
    fs, gs = Fs[-1], Gs[-1]
    cols, gcols, Q, R = [], [], [], []
    drop = mp.mpf(2) ** (-mp.mp.prec // 2)
    for i in range(1, len(Fs)):
        v = [a - b for a, b in zip(fs, Fs[-1 - i])]
        n0 = mp.sqrt(mp.fdot(v, v))
        if n0 == 0:
            continue
        r = []
        for qv in Q:
            h = mp.fdot(qv, v)
            r.append(h)
            v = [a - h * b for a, b in zip(v, qv)]
        nv = mp.sqrt(mp.fdot(v, v))
        if nv <= drop * n0:
            continue
        Q.append([a / nv for a in v])
        R.append(r + [nv])
        gcols.append([a - b for a, b in zip(gs, Gs[-1 - i])])
    if not Q:
        return list(gs)
    mm = len(Q)
    rhs = [mp.fdot(qv, fs) for qv in Q]
    th = [mp.mpf(0)] * mm
    for i in range(mm - 1, -1, -1):                    # R upper triangular: R[c][r] is entry (r, c)
        th[i] = (rhs[i] - mp.fsum(R[j][i] * th[j] for j in range(i + 1, mm))) / R[i][i]
    return [gs[k] - mp.fsum(th[i] * gcols[i][k] for i in range(mm)) for k in range(len(gs))]


# ------------------------------------------------------------------ states
def load_state(path):
    """(meta, z): z the exact sum of the stored parts (zh, zl, z2, ...) at the working precision"""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    parts = [d["zh"], d["zl"]] + [d[f"z{i}"] for i in range(2, 8) if f"z{i}" in d.files]
    L = len(parts[0])
    z = [mp.fsum(mp.mpf(float(p[i])) for p in parts) for i in range(L)]
    meta["parts"] = len(parts)
    return meta, z


def seg_of(meta, ks=None):
    D = meta.get("D_str", meta["D"])
    return Seg(D, meta["M"], ks or meta.get("ks", 130), meta["nodes"][-1], meta.get("kfreeze"))


def horizon_data(meta, z):
    """(m_h, a_h, Delta, the last node's p, m, a, c) from a state of the chain's layout"""
    M, K = int(meta["M"]), len(meta["nodes"])
    off_node = lambda i: 2 * M + 4 * M * i  # noqa: E731
    mh = 2 * M + 4 * M * K
    return z[mh:mh + M], z[mh + M:mh + 2 * M], z[-1], z[off_node(K - 1):off_node(K - 1) + 4 * M]


def fmt(x):
    return f"{float(x):.3e}"


# ------------------------------------------------------------------ commands
def cmd_check(path, bound, mutate, prec):
    mp.mp.prec = prec
    meta, z = load_state(path)
    seg = seg_of(meta)
    mh, ah, delta, node = horizon_data(meta, z)
    t0 = time.time()
    s = series(seg, mh, seg.omega(delta), a0=ah, mutate=mutate)
    R, _ = rows_old(seg, s, node)
    M = seg.M
    rp = R[:M]
    kf = meta.get("kfreeze")
    if kf:
        P = plow_odd(M, int(kf))
        rp_low = [mp.fdot(P[i], rp) for i in range(M)]
    else:
        rp_low = rp
    parts = dict(p_low=float(amax(rp_low)), p_all=float(amax(rp)), m=float(amax(R[M:2 * M])),
                 a=float(amax(R[2 * M:3 * M])), c=float(amax(R[3 * M:4 * M])))
    worst = max(parts["p_low"], parts["m"], parts["a"], parts["c"])
    ok = worst <= bound
    rec = dict(cmd="check", state=os.path.basename(path), D=seg.D_str, parts_stored=meta["parts"], prec=prec,
               mutate=mutate, kfreeze=kf, rows=parts, worst=worst, bound=bound, ok=ok, G=float(seg.G),
               seconds=round(time.time() - t0, 1))
    print(f"check {os.path.basename(path)} D {seg.D_str[:12]} ({meta['parts']} parts, {prec} bits"
          f"{', --mutate=' + mutate if mutate else ''}): rows p(k<={kf}) {fmt(parts['p_low'])}, m {fmt(parts['m'])}, "
          f"a {fmt(parts['a'])}, c {fmt(parts['c'])}; p all harmonics {fmt(parts['p_all'])}; "
          f"worst {fmt(worst)} vs bound {bound:.3e}: {'PASS' if ok else 'FAIL'} ({rec['seconds']} s)")
    return rec, ok


def setup(path, prec, ks=None, move=0):
    """the state's segment and its old series; move: a_h times 1 + move / G, so that the mass mode's amplitude at the
    node is about move (at the stored data it is 3e-28 at D 73 and 5e-35 at D 242.36, where chi_hat hardly matters)"""
    mp.mp.prec = prec
    meta, z = load_state(path)
    seg = seg_of(meta, ks)
    mh, ah, delta, node = horizon_data(meta, z)
    if move:
        ah = [v * (1 + mp.mpf(move) / seg.G) for v in ah]
    om = seg.omega(delta)
    t0 = time.time()
    s = series(seg, mh, om, a0=ah)
    t_old = time.time() - t0
    R_old, vals = rows_old(seg, s, node)
    aK = vals[2]                                       # the old series' a at u0
    return meta, seg, mh, ah, delta, node, om, R_old, aK, t_old


def ref_key(state, mode):
    """what a cached reference is the fixed point of: the state, the precision, gamma's datum and the 1/k"""
    return dict(state=os.path.basename(state), prec=mp.mp.prec, gamma=mode, recip="dd" if RECIP_DD[0] else "exact")


def save_ref(path, ref, key):
    with open(path, "w") as fh:
        json.dump(dict(key, rows=[str(v) for v in ref["rows"]], y=[str(v) for v in ref["y"]],
                       gamma_values=[str(v) for v in ref["gamma"]], sweeps=ref["sweeps"], hist=ref["hist"]), fh)


def load_ref(path, key):
    """the cached reference at path, refused unless it was made for key (ref_key). The two references of #1035 were
    written before the key was stored and carry only prec; they are accepted with a notice."""
    with open(path) as fh:
        d = json.load(fh)
    if "state" in d:
        bad = {k: (d.get(k), v) for k, v in key.items() if d.get(k) != v}
        if bad:
            raise SystemExit(f"{path} is the fixed point of another problem: {bad}")
    else:
        if d["prec"] != key["prec"]:
            raise SystemExit(f"{path} was made at {d['prec']} bits")
        print(f"notice: {path} does not record its state, gamma and 1/k (accepted as given)", flush=True)
    return dict(rows=[mp.mpf(v) for v in d["rows"]], y=[mp.mpf(v) for v in d["y"]],
                gamma=[mp.mpf(v) for v in d.get("gamma_values", d.get("gamma"))], sweeps=d["sweeps"], hist=d["hist"],
                prec=d["prec"])


def cmd_gain(path, prec, mem, warm, maxit, mode, start, accel, refpath, ref_start="slaved"):
    """one run of sweeps (accel plain or anderson) from one start (slaved, naive or warm) against the reference
    fixed point, read from refpath or computed (Anderson from the slaved start to 2^-(prec-16)) and written there"""
    meta, seg, mh, ah, delta, node, om, R_old, aK, t_old = setup(path, prec)
    print(f"gain {os.path.basename(path)} D {seg.D_str[:12]} n {fmt(seg.n)} K {seg.K} M {seg.M} prec {prec}: "
          f"old series {t_old:.1f} s; G = x_K^-(n-1) {fmt(seg.G)}", flush=True)
    pr = lambda tag: (lambda r: print(f"  {tag} sweep {r['sweep']:2d}: |g(y)-y| {r['fp_res']:.2e}"  # noqa: E731
                                      + (f", rows err {r['err_rows']:.2e}, y err {r['err_y']:.2e}" if 'err_rows' in r else "")
                                      + f", newton {r['newton']}, {r['t']} s", flush=True))
    key = ref_key(path, mode)
    if refpath and os.path.exists(refpath):
        ref = load_ref(refpath, key)
    else:
        B0r = None
        if ref_start == "old":                         # from B* = 1/a^2 of the old series (D 73: the slaved start diverges)
            B0r = b_of_old(seg, series(seg, mh, om, a0=ah))
        ref = fixed_point(seg, mh, aK, om, node, B0=B0r, gam0=[mp.mpf(0)] * seg.M if B0r else None, accel="anderson",
                          mem=mem, maxit=maxit, mode=mode, log=pr("ref"), tol=mp.mpf(EPS_QD) * mp.mpf("1e-8"))
        if ref["hist"][-1]["fp_res"] > float(EPS_QD) * 1e-2 or not math.isfinite(ref["hist"][-1]["fp_res"]):
            print(f"gain: the reference did not converge in {maxit} sweeps (|g(y)-y| {ref['hist'][-1]['fp_res']:.2e})")
            return dict(cmd="gain", state=os.path.basename(path), D=seg.D_str, prec=prec, ref_hist=ref["hist"],
                        ok=False), False
        if ref["hist"][-1]["fp_res"] > float(EPS_QD) * 1e-8:
            print(f"gain: the reference stops at |g(y)-y| {ref['hist'][-1]['fp_res']:.2e}, above 1e-8 eps_qd")
        if refpath:
            save_ref(refpath, ref, key)
    if start == "none":                                # --start none: the reference alone
        return dict(cmd="gain", state=os.path.basename(path), D=seg.D_str, prec=prec, ref_hist=ref["hist"], ok=True), True
    refp = (ref["rows"], ref["y"])
    B0 = g0 = None
    tag = f"{start}/{accel}"
    if start == "warm":
        # a Newton iterate's neighbour: the data moved by warm (m_h, a_K and Delta times 1 + warm), started from the
        # reference's fixed point; the run goes on to 1e-8 eps_qd and its last iterate is its own reference
        B0, g0 = from_y(seg, ref["y"]), ref["gamma"]
        f = 1 + mp.mpf(warm)
        ys = []
        r = fixed_point(seg, vscale(mh, f), vscale(aK, f), seg.omega(delta * f), node, B0=B0, gam0=g0, accel=accel,
                        mem=mem, maxit=maxit, mode=mode, tol=mp.mpf(EPS_QD) * mp.mpf("1e-8"), log=pr(tag),
                        keep=ys)
        if r["hist"][-1]["fp_res"] > float(EPS_QD) * 1e-8:
            # the last iterate is the run's own reference, so it must have converged
            print(f"gain {tag}: the moved problem did not converge in {maxit} sweeps (|g(y)-y| "
                  f"{r['hist'][-1]['fp_res']:.2e}); no errors are measured", flush=True)
            return dict(cmd="gain", state=os.path.basename(path), D=seg.D_str, prec=prec, start=start, accel=accel,
                        hist=r["hist"], sweeps_to_qd=None, ok=False), False
        for rec, (yy, rr) in zip(r["hist"], ys):
            rec["err_rows"] = float(amax(vsub(rr, r["rows"])))
            rec["err_y"] = float(amax(vsub(yy, r["y"])))
            print(f"  {tag} sweep {rec['sweep']:2d}: rows err {rec['err_rows']:.2e}, y err {rec['err_y']:.2e}")
    else:
        r = fixed_point(seg, mh, aK, om, node, B0=B0, gam0=g0, accel=accel, mem=mem, maxit=maxit, ref=refp,
                        mode=mode, tol=mp.mpf(EPS_QD) * mp.mpf("1e-4"), log=pr(tag), start=start)
    h = r["hist"]
    to_qd = next((x["sweep"] for x in h if max(x["err_rows"], x["err_y"]) <= EPS_QD), None)
    fl = 1e4 * 2.0 ** -prec
    ratios = [h[i + 1]["err_y"] / h[i]["err_y"] for i in range(len(h) - 1) if h[i + 1]["err_y"] > fl]
    gain = math.exp(sum(math.log(x) for x in ratios[-6:]) / len(ratios[-6:])) if ratios else None
    t_sweep = round(h[-1]["t"] / len(h), 1)
    print(f"gain {tag}: sweeps to eps_qd {to_qd} (of {len(h)}); error ratio per sweep, geometric mean of the last "
          f"{min(6, len(ratios))}: {gain if gain is None else f'{gain:.3e}'}; one sweep {t_sweep} s, the old series "
          f"{t_old:.1f} s in this code", flush=True)
    ok = to_qd is not None and to_qd <= 5
    return dict(cmd="gain", state=os.path.basename(path), D=seg.D_str, prec=prec, mem=mem, warm=warm, mode=mode,
                start=start, accel=accel, G=float(seg.G), t_old=round(t_old, 1), t_sweep=t_sweep, ref_sweeps=ref["sweeps"],
                sweeps_to_qd=to_qd, gain=gain, hist=h, ok=ok), ok


def cmd_p1(path, prec, mutate, mode, mem, maxit, refpath=None, move=0):
    """P1; --ref starts the sweeps from a cached fixed point (gain's reference) instead of the slaved start; --move
    moves a_h off the stored value by move / G (setup)"""
    meta, seg, mh, ah, delta, node, om, R_old, aK, t_old = setup(path, prec, move=move)
    pr = lambda r: print(f"  sweep {r['sweep']:2d}: |g(y)-y| {r['fp_res']:.2e}, newton {r['newton']}, {r['t']} s",  # noqa: E731
                         flush=True)
    B0 = g0 = None
    if refpath:
        # the start only: a reference of another 1/k or of unmoved data is a fine start, so the key is not enforced
        with open(refpath) as fh:
            d = json.load(fh)
        B0 = from_y(seg, [mp.mpf(v) for v in d["y"]])
        g0 = [mp.mpf(v) for v in d.get("gamma_values", d.get("gamma"))]
    r = fixed_point(seg, mh, aK, om, node, B0=B0, gam0=g0, accel="anderson", mem=mem, maxit=maxit, mode=mode,
                    mutate=mutate, log=pr)
    d = vsub(r["rows"], R_old)
    M = seg.M
    per = {f: float(amax(d[i * M:(i + 1) * M])) for i, f in enumerate("pmac")}
    diff = float(amax(d))
    b_qd = EPS_QD * float(seg.G)
    b_sharp = 100 * 2.0 ** -prec * float(seg.G)
    ah_der = [1 / mp.sqrt(1 + v) for v in r["B"][0]]
    dah = float(amax(vsub(ah_der, ah)))
    ok_qd, ok_sharp = diff <= b_qd, diff <= b_sharp
    print(f"p1 {os.path.basename(path)} D {seg.D_str[:12]} prec {prec} gamma by {mode}, 1/k {'dd' if RECIP_DD[0] else 'exact'}"
          f"{f', a_h moved by {move:g}/G' if move else ''}{', --mutate=' + mutate if mutate else ''}: max|R_fact - R_old| {diff:.3e} (p {per['p']:.2e}, m {per['m']:.2e}, "
          f"a {per['a']:.2e}, c {per['c']:.2e}); old rows max {fmt(amax(R_old))}; derived a_h - a_h {dah:.2e}; "
          f"{r['sweeps']} sweeps")
    print(f"  bound eps_qd G {b_qd:.3e}: {'PASS' if ok_qd else 'FAIL'}; sharp 100 2^-prec G {b_sharp:.3e}: "
          f"{'PASS' if ok_sharp else 'FAIL'}")
    rec = dict(cmd="p1", state=os.path.basename(path), D=seg.D_str, prec=prec, mode=mode, mutate=mutate, move=move,
               recip="dd" if RECIP_DD[0] else "exact", diff=diff,
               per_field=per, old_rows_max=float(amax(R_old)), dah=dah, b_qd=b_qd, b_sharp=b_sharp, ok_qd=ok_qd,
               ok_sharp=ok_sharp, sweeps=r["sweeps"], G=float(seg.G))
    return rec, ok_qd and ok_sharp


def b_of_old(seg, s):
    """beta = 1/a^2 - 1 as a series from the old series' a (series inverse of a a): the factored map's fixed point up to
    hp's rounding of 1/k and the truncation"""
    a, K, M = s["a"], seg.K, seg.M
    aa = [cs(a, a, k) for k in range(K + 1)]
    Bf = [[1 / v for v in aa[0]]]
    for k in range(1, K + 1):
        sm = cs(aa, Bf, k, 1, k)
        Bf.append([-sm[j] / aa[0][j] for j in range(M)])
    return [[v - 1 for v in Bf[0]]] + Bf[1:]


def cmd_loop(path, prec, its, h, orders, seed=2):
    """the loop at a converged state, gamma by B's sum: B* from the old series, g(B*) - B* and gamma* = B*(u0) -
    B_sl(u0); the power iteration of the linearized sweep map from a random perturbation of y of size h (its steps);
    and the forward pass's response in p, per order, to a perturbation of y of 1e-30 at one order"""
    import random
    meta, seg, mh, ah, delta, node, om, R_old, aK, t_old = setup(path, prec)
    s = series(seg, mh, om, a0=ah)
    beta = b_of_old(seg, s)
    K, M, u0 = seg.K, seg.M, seg.u0

    def g(B, fw=None):
        fw = fw or series(seg, mh, om, B=B)
        Bn, gam, _ = update(seg, fw["q"], aK, None, "B")
        return yvec(seg, Bn), fw

    y0 = yvec(seg, beta)
    g0, fw0 = g(beta)
    Bs = b_sl(seg, fw0["q"])
    gstar = vsub(horner(beta, u0), horner(Bs, u0))
    c1 = s["c"][1]
    kres = [float(seg.nh / (1 + v) - 1) for v in c1]                # kappa_k = 0: Re s = n / (2 (1 + c1)) - 1
    print(f"loop {os.path.basename(path)} D {seg.D_str[:12]} prec {prec}: |g(y*) - y*| {fmt(amax(vsub(g0, y0)))} "
          f"(|y*| {fmt(amax(y0))}); gamma* = B*(u0) - B_sl(u0) up to {fmt(amax(gstar))}; Re s from {min(kres):.1f} "
          f"to {max(kres):.1f}; the mass mode's peak order (n-1) u0/(1-u0) {float((seg.n - 1) * u0 / (1 - u0)):.1f}",
          flush=True)
    random.seed(seed)
    hh = mp.mpf(h)
    v = [hh * (2 * random.random() - 1) for _ in y0]
    ratios = []
    for it in range(its):
        w = vsub(g(from_y(seg, [p + q for p, q in zip(y0, v)]))[0], g0)
        r = amax(w) / amax(v)
        ratios.append(float(r))
        per = [amax(w[k * M:(k + 1) * M]) for k in range(K + 1)]
        kmax = max(range(K + 1), key=lambda k: per[k])
        print(f"  power step {it}: |G v| / |v| {fmt(r)}, largest at order {kmax}", flush=True)
        v = [x * hh / amax(w) for x in w]
    resp = {}
    for jo in orders:
        bp = [list(r) for r in beta]
        bp[jo] = [x + mp.mpf("1e-30") / u0 ** jo for x in bp[jo]]
        f2 = series(seg, mh, om, B=bp)
        dp = {k: float(amax(vsub(f2["p"][k], fw0["p"][k])) * u0 ** k) for k in range(jo, K + 1)}
        resp[jo] = dp
        print(f"  y moved by 1e-30 at order {jo}: dp at u0^k scale, order {jo} {dp[jo]:.2e}, order {K} {dp[K]:.2e}, "
              f"largest {max(dp.values()):.2e}", flush=True)
    return dict(cmd="loop", state=os.path.basename(path), D=seg.D_str, prec=prec, fp_star=float(amax(vsub(g0, y0))),
                gamma_star=float(amax(gstar)), re_s=[min(kres), max(kres)], power=ratios, response={str(k): v for k, v in resp.items()}), True


# ------------------------------------------------------------------ stage 1b: the curvature test (5.2)
CURV_PAIRS_OLD = ((1e-20, 1e-19), (1e-30, 1e-29), (1e-40, 1e-39), (1e-50, 1e-49))
CURV_PAIR = (1e-20, 1e-19)
CURV_T1 = 1e-25                                    # the level that fixes the predictor's quadratic term
MUT_R0 = "1e-40"                                   # --mutate=r0: the shift of R(0)'s p, m and c rows
CURVX_BAR = 1e-70                                  # curvx's convergence bar on |g(y) - y|
PMC = None                                         # row indices of p, m and c (set per segment)


def pmc_rows(seg):
    M = seg.M
    return list(range(0, 2 * M)) + list(range(3 * M, 4 * M))


def move(seg, name):
    """(delta m_h, delta Delta) of a named move: 'D' (delta Delta = 1) or 'm<h>' (delta m_h = cos(h theta_j) on the half
    grid, an odd field for odd h)"""
    M = seg.M
    if name == "D":
        return [mp.mpf(0)] * M, mp.mpf(1)
    h = int(name[1:])
    return [mp.cos(h * 2 * mp.pi * j / (2 * M)) for j in range(M)], mp.mpf(0)


def old_at(seg, mh, ah, delta, node, dm, dD, t, c2=0):
    """the old rows and a's sum at u0 for m_h + t dm, a_h, Delta + t dD + c2 t^2"""
    t = mp.mpf(t)
    s = series(seg, [m + t * d for m, d in zip(mh, dm)], seg.omega(delta + t * dD + mp.mpf(c2) * t * t), a0=ah)
    R, vals = rows_old(seg, s, node)
    return R, vals[2]


def solve_ah(seg, mh, guess, delta, node, dm, dD, t, aK, J, tol, c2=0, maxit=8):
    """method A: the a_h with a's sum at u0 equal to a_K at the moved data, by Newton with the fixed Jacobian J of a(u0)
    in a_h, from the guess. Returns (rows with the a row a_K - a(node), a_h, |F| history)."""
    ah, hist = list(guess), []
    for _ in range(maxit):
        R, au = old_at(seg, mh, ah, delta, node, dm, dD, t, c2)
        F = vsub(au, aK)
        hist.append(float(amax(F)))
        if hist[-1] <= tol or (len(hist) > 1 and hist[-1] > 0.1 * hist[-2]) or len(hist) == maxit:
            break                                  # R and a_h stay those of the last evaluation
        st = mp.lu_solve(J, mp.matrix(F))
        ah = [a - st[i] for i, a in enumerate(ah)]
    M = seg.M
    R = R[:2 * M] + vsub(aK, node[2 * M:3 * M]) + R[3 * M:]
    return R, ah, hist


def second_diff(Rp, Rm, R0, t, rows):
    t = mp.mpf(t)
    return [(Rp[i] + Rm[i] - 2 * R0[i]) / (t * t) for i in rows]


def qmax(Q):
    return float(amax(Q)) if Q else 0.0


def scales(Q1, Q2):
    """the relative change between two second-difference quotients (the t^2 check): max |Q2 - Q1| / max(|Q1|, |Q2|)"""
    big = max(amax(Q1), amax(Q2))
    return float(amax(vsub(Q2, Q1)) / big) if big else float("inf")


def curv_cache(path):
    with open(path) as fh:
        d = json.load(fh)
    return d


def cmd_curvjac(path, prec, out_cache):
    """the curvature test's base (5.2): an exact 1/k, a_h re-solved so that a's sum at u0 equals the node's a, and the
    M x M Jacobian of a(u0) in a_h by forward differences (step a_h 1e-50 per point), cached for cmd_curv"""
    RECIP_DD[0] = False
    meta, seg, mh, ah, delta, node, om, R_old, aK, t_old = setup(path, prec)
    M = seg.M
    t0 = time.time()
    eta = mp.mpf("1e-50")
    J = mp.matrix(M, M)
    for j in range(M):
        a2 = list(ah)
        a2[j] = ah[j] * (1 + eta)
        _, au = old_at(seg, mh, a2, delta, node, [mp.mpf(0)] * M, 0, 0)
        for i in range(M):
            J[i, j] = (au[i] - aK[i]) / (ah[j] * eta)
        print(f"  column {j + 1}/{M}, {time.time() - t0:.0f} s", flush=True)
    a_node = node[2 * M:3 * M]
    tol = mp.mpf(2) ** (-prec + 10) * seg.G
    R0, ahs, hist = solve_ah(seg, mh, ah, delta, node, [mp.mpf(0)] * M, 0, 0, a_node, J, tol)
    diag = [abs(J[i, i]) for i in range(M)]
    off = max(abs(J[i, j]) for i in range(M) for j in range(M) if i != j)
    print(f"curvjac {os.path.basename(path)} D {seg.D_str[:12]} prec {prec}, exact 1/k: a(u0) - a(node) with the stored "
          f"a_h {fmt(amax(vsub(aK, a_node)))}; re-based in {len(hist)} passes to {hist[-1]:.2e}; a_h moved by "
          f"{fmt(amax([(x - y) / y for x, y in zip(ahs, ah)]))} relative; J diagonal {fmt(min(diag))}..{fmt(max(diag))}, "
          f"largest off-diagonal {fmt(off)} (G {fmt(seg.G)}); {time.time() - t0:.0f} s", flush=True)
    with open(out_cache, "w") as fh:
        json.dump(dict(state=os.path.basename(path), prec=prec, ah=[str(v) for v in ahs],
                       J=[[str(J[i, j]) for j in range(M)] for i in range(M)], rebase_hist=hist), fh)
    return dict(cmd="curvjac", state=os.path.basename(path), D=seg.D_str, prec=prec, rebase_hist=hist,
                diag=[float(min(diag)), float(max(diag))], off=float(off)), True


def curv_base(path, prec, cache):
    RECIP_DD[0] = False
    mp.mp.prec = prec
    meta, z = load_state(path)
    seg = seg_of(meta)
    mh, ah_stored, delta, node = horizon_data(meta, z)
    c = curv_cache(cache)
    if c["state"] != os.path.basename(path) or c["prec"] != prec:
        raise SystemExit(f"{cache} is for {c['state']} at {c['prec']} bits")
    M = seg.M
    ah = [mp.mpf(v) for v in c["ah"]]
    J = mp.matrix([[mp.mpf(v) for v in row] for row in c["J"]])
    aK = node[2 * M:3 * M]
    return seg, mh, ah, delta, node, J, aK


def cmd_curv(path, prec, cache, name, plant_from=None, mutate="", plant_c=None):
    """the curvature test along one move (5.2): Q_old at fixed a_h over the pairs of CURV_PAIRS_OLD, Q_fact at fixed a_K
    by method A at CURV_PAIR, the t^2 checks and the ratio; with plant_from (a curv record along D), the planted
    quadratic term instead"""
    seg, mh, ah, delta, node, J, aK = curv_base(path, prec, cache)
    M = seg.M
    rows = pmc_rows(seg)
    dm, dD = move(seg, name)
    tol = mp.mpf(2) ** (-prec + 10) * seg.G
    t0 = time.time()
    R0, a0 = old_at(seg, mh, ah, delta, node, dm, dD, 0)
    if mutate == "r0":
        R0 = [v + mp.mpf(MUT_R0) if i in rows else v for i, v in enumerate(R0)]
    rec = dict(cmd="curv", state=os.path.basename(path), D=seg.D_str, prec=prec, move=name, recip="exact",
               mutate=mutate,
               base_F=float(amax(vsub(a0, aK))))
    # the predictor: a_h(t) = a_h + t s + t^2 w, s from the derivative of a(u0) along the move at fixed a_h
    eta = mp.mpf("1e-45")
    _, ae = old_at(seg, mh, ah, delta, node, dm, dD, eta)
    da = [(x - y) / eta for x, y in zip(ae, a0)]
    sv = mp.lu_solve(J, mp.matrix(da))
    s_ = [-sv[i] for i in range(M)]
    w = [mp.mpf(0)] * M
    c2 = 0
    if plant_from:
        pl = json.load(open(plant_from))
        c2 = mp.mpf(plant_c) if plant_c else mp.mpf(pl["plant_c2"])
    def fact(t):
        t = mp.mpf(t)
        g = [a + t * x + t * t * y for a, x, y in zip(ah, s_, w)]
        cc = 0 if mutate == "plant1" and t < 0 else c2      # plant1: the planted term at +t only
        R, a_t, hist = solve_ah(seg, mh, g, delta, node, dm, dD, t, aK, J, tol, cc)
        return R, a_t, hist
    out = {}
    # the predictor's quadratic term from the level CURV_T1
    Rp, ap, hp_ = fact(CURV_T1)
    Rm, am, hm_ = fact(-CURV_T1)
    t1 = mp.mpf(CURV_T1)
    w = [(x + y - 2 * a) / (2 * t1 * t1) for x, y, a in zip(ap, am, ah)]
    out["t1_newton"] = [hp_, hm_]
    Qf = {}
    for t in CURV_PAIR:
        Rp, _, hp_ = fact(t)
        Rm, _, hm_ = fact(-t)
        Qf[t] = second_diff(Rp, Rm, R0, t, rows)
        out[f"fact_newton_{t:g}"] = [hp_, hm_]
        if t == CURV_PAIR[0]:
            d1 = [(Rp[i] - Rm[i]) / (2 * mp.mpf(t)) for i in rows]
        print(f"  {name} fact t {t:g}: |Q| {qmax(Qf[t]):.3e}, Newton |F| {hp_[-1]:.1e}/{hm_[-1]:.1e} in "
              f"{len(hp_)}/{len(hm_)} passes, {time.time() - t0:.0f} s", flush=True)
    sc_f = scales(Qf[CURV_PAIR[0]], Qf[CURV_PAIR[1]])
    rec.update(Q_fact=qmax(Qf[CURV_PAIR[0]]), Q_fact_19=qmax(Qf[CURV_PAIR[1]]), scale_fact=sc_f,
               d1_fact=qmax(d1), newton=out)
    if plant_from:
        Qb = [mp.mpf(v) for v in pl["Q_fact_vec"]]
        pred = [2 * c2 * mp.mpf(v) for v in pl["d1_fact_vec"]]
        got = vsub(Qf[CURV_PAIR[0]], Qb)
        err = float(amax(vsub(got, pred)) / amax(pred))
        ok = err <= 0.01
        rec.update(cmd="curvplant", plant_c2=str(c2), plant_c2_gate=pl["plant_c2"], planted=float(amax(pred)),
                   recovered_err=err, ok=ok)
        print(f"curvplant {os.path.basename(path)} D {seg.D_str[:12]} along {name}: planted |2 c dR/dD| "
              f"{float(amax(pred)):.3e}, recovered to {err:.2e} relative: {'PASS' if ok else 'FAIL'} (bar 1e-2), "
              f"{time.time() - t0:.0f} s", flush=True)
        return rec, ok
    # Q_old at fixed a_h over the pairs
    olds = []
    for ta, tb in CURV_PAIRS_OLD:
        Qs = []
        for t in (ta, tb):
            Rp, _ = old_at(seg, mh, ah, delta, node, dm, dD, t)
            Rm, _ = old_at(seg, mh, ah, delta, node, dm, dD, -t)
            Qs.append(second_diff(Rp, Rm, R0, t, rows))
        sc = scales(Qs[0], Qs[1])
        olds.append(dict(pair=[ta, tb], Q=qmax(Qs[0]), Q_b=qmax(Qs[1]), scale=sc, ok=sc <= 0.1))
        print(f"  {name} old pair {ta:g}/{tb:g}: |Q| {qmax(Qs[0]):.3e} / {qmax(Qs[1]):.3e}, t^2 check {sc:.2e} "
              f"({'in range' if sc <= 0.1 else 'out of range'}), {time.time() - t0:.0f} s", flush=True)
    use = next((o for o in olds if o["ok"]), None)
    rec.update(old_pairs=olds)
    if use is None or sc_f > 0.1:
        verdict = "not resolved"
        ratio = None
    else:
        ratio = rec["Q_fact"] / use["Q"]
        verdict = ("gate 1 (<= 1e-10)" if ratio <= 1e-10 else
                   ("gate 2 (>= 1e-2)" if ratio >= 1e-2 else "gate 3 (between: not resolved)"))
    rec.update(Q_old=use["Q"] if use else None, Q_old_pair=use["pair"] if use else None, ratio=ratio, verdict=verdict,
               Q_fact_vec=[str(v) for v in Qf[CURV_PAIR[0]]], d1_fact_vec=[str(v) for v in d1],
               plant_c2=str(mp.mpf("1e-10") * mp.mpf(use["Q"]) / (2 * amax(d1))) if use else None)
    print(f"curv {os.path.basename(path)} D {seg.D_str[:12]} along {name}: Q_fact {rec['Q_fact']:.3e} (t^2 check "
          f"{sc_f:.2e}), Q_old {use['Q'] if use else float('nan'):.3e} at the pair {use['pair'] if use else None}; ratio "
          f"{ratio if ratio is None else f'{ratio:.3e}'}: {verdict}; {time.time() - t0:.0f} s", flush=True)
    return rec, verdict.startswith(("gate 1", "gate 2"))     # gate 3 is reported as not resolved (5.2)


def cmd_curvx(path, prec, refpath, t, cache448):
    """the cross-check of 5.2: the stage-1 fixed point (exact 1/k, gamma by a's sum, Anderson from the reference) at
    the base with Delta moved by t (t = 0 for the base), its rows written for curvxq"""
    RECIP_DD[0] = False
    mp.mp.prec = prec
    meta, z = load_state(path)
    seg = seg_of(meta)
    mh, _, delta, node = horizon_data(meta, z)
    M = seg.M
    aK = node[2 * M:3 * M]
    with open(refpath) as fh:
        d = json.load(fh)
    B0 = from_y(seg, [mp.mpf(v) for v in d["y"]])
    g0 = [mp.mpf(v) for v in d.get("gamma_values", d.get("gamma"))]
    t = mp.mpf(t)
    pr = lambda r: print(f"  sweep {r['sweep']:2d}: |g(y)-y| {r['fp_res']:.2e}, {r['t']} s", flush=True)  # noqa: E731
    r = fixed_point(seg, mh, aK, seg.omega(delta + t), node, B0=B0, gam0=g0, accel="anderson", mem=5, maxit=40,
                    tol=mp.mpf("1e-72"), log=pr)
    rec = dict(cmd="curvx", state=os.path.basename(path), D=seg.D_str, prec=prec, t=str(t), recip="exact",
               fp_res=r["hist"][-1]["fp_res"], sweeps=r["sweeps"], rows=[str(v) for v in r["rows"]])
    print(f"curvx {os.path.basename(path)} t {float(t):g}: {r['sweeps']} sweeps, |g(y)-y| {r['hist'][-1]['fp_res']:.2e}",
          flush=True)
    return rec, r["hist"][-1]["fp_res"] <= CURVX_BAR


def cmd_curvxq(paths, t, curv_D):
    """Q_fact from the three fixed points of curvx (t, -t, 0) against method A's Q_fact along D (5.2's cross-check)"""
    mp.mp.prec = max(json.load(open(q))["prec"] for q in paths) + 32
    recs = [json.load(open(q)) for q in paths]
    by = {mp.mpf(r["t"]): [mp.mpf(v) for v in r["rows"]] for r in recs}
    t = mp.mpf(t)
    M = len(by[t]) // 4
    rows = list(range(0, 2 * M)) + list(range(3 * M, 4 * M))
    Q = second_diff(by[t], by[-t], by[mp.mpf(0)], t, rows)
    cv = json.load(open(curv_D))
    QA = [mp.mpf(v) for v in cv["Q_fact_vec"]]
    conv = all(float(r["fp_res"]) <= CURVX_BAR for r in recs)      # a curvx that missed its bar fails here too
    noise = max(float(r["fp_res"]) for r in recs) * 1e3 / float(t) ** 2
    noise_A = max(h[-1] for h in cv["newton"][f"fact_newton_{float(t):g}"]) * 1e3 / float(t) ** 2   # A's last |F|
    diff = float(amax(vsub(Q, QA)) / max(amax(Q), amax(QA)))
    ok = conv and (diff <= 0.01 or (qmax(Q) <= noise and qmax(QA) <= noise_A))
    print(f"curvxq: fixed point |Q_fact| {qmax(Q):.3e}, method A {qmax(QA):.3e}, relative difference {diff:.2e} "
          f"(noise about {noise:.1e} and {noise_A:.1e}; fixed points {'converged' if conv else 'NOT converged'}): "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return dict(cmd="curvxq", Q_fp=qmax(Q), Q_A=qmax(QA), diff=diff, noise=noise, noise_A=noise_A, converged=conv,
                ok=ok), ok


# ------------------------------------------------------------------ stage 2: the rung along a raise-sized move (5.3)
RAISE_MOVES = (1e-16, 1e-15, 1e-14)                # Delta moves; the raise's Newton step has dDelta -1.0e-14 (#1064)
RAISE_TAUS = (1e-20, 1e-19)                        # the second difference that gives the predictor's t^2 term
RAISE_ETA = "1e-60"                                # the finite-difference step of the rung's Jacobian, relative


def rung_F(seg, mh, ah, delta, node):
    """the rung of the chain's projection (reseed.py, rows 'ma'): the m and a rows of the horizon block, the old series
    at u0 minus the node's m and a, for the horizon data (m_h, a_h) at Delta"""
    s = series(seg, mh, seg.omega(delta), a0=ah)
    R, _ = rows_old(seg, s, node)
    return R[seg.M:3 * seg.M], R


def rung_newton(seg, h, delta, node, J, tol, maxit=10):
    """chord Newton on the 2M horizon data h = (m_h, a_h) for the rung at Delta with the fixed Jacobian J; ends at tol,
    when a pass gains less than tenfold, or after maxit passes. Returns (h, |F| history, full rows at the end)"""
    M = seg.M
    hist = []
    for _ in range(maxit):
        F, R = rung_F(seg, h[:M], h[M:], delta, node)
        hist.append(float(amax(F)))
        if hist[-1] <= tol or (len(hist) > 1 and hist[-1] > 0.1 * hist[-2]) or len(hist) == maxit:
            break
        st = mp.lu_solve(J, mp.matrix(F))
        h = [x - st[i] for i, x in enumerate(h)]
    return h, hist, R


def cmd_raisecheck(path, prec, ks, cache, mutate=""):
    """5.3's Mac-side check: at the state's own Delta and node, the rung (the m and a rows for m_h and a_h, the node
    held) at order ks with hp's 1/k, solved from the stored data (the start projection), then along Delta moves of
    RAISE_MOVES from the linear predictor (the corrector's own start, ElimO's) and from the second-order one
    (factored_raise.py's). Passes when the second-order start converges at the largest move; mutate 'linear' judges
    the linear start instead (the control, expected to fail there)."""
    RECIP_DD[0] = True
    mp.mp.prec = prec
    meta, z = load_state(path)
    seg = seg_of(meta, ks=ks)
    mh, ah, delta, node = horizon_data(meta, z)
    M = seg.M
    tol = mp.mpf(2) ** (-prec + 40) * seg.G
    t0 = time.time()
    h0 = list(mh) + list(ah)
    F0, R0 = rung_F(seg, mh, ah, delta, node)
    rec = dict(cmd="raisecheck", state=os.path.basename(path), D=seg.D_str, prec=prec, ks=ks, recip="dd",
               G=float(seg.G),
               stored=dict(p=float(amax(R0[:M])), m=float(amax(R0[M:2 * M])), a=float(amax(R0[2 * M:3 * M])),
                           c=float(amax(R0[3 * M:]))), mutate=mutate)
    print(f"raisecheck {os.path.basename(path)} D {seg.D_str[:12]} ks {ks} (state's {meta.get('ks')}), {prec} bits, "
          f"hp's 1/k: stored data's rows m {rec['stored']['m']:.3e}, a {rec['stored']['a']:.3e}, "
          f"c {rec['stored']['c']:.3e}"
          f" (p all harmonics {rec['stored']['p']:.3e}); G {float(seg.G):.3e}", flush=True)
    # the rung's Jacobian in (m_h, a_h), by forward differences, cached
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            c = json.load(fh)
        J = mp.matrix([[mp.mpf(v) for v in row] for row in c["J"]])
        print(f"  Jacobian from {cache}", flush=True)
    else:
        eta = mp.mpf(RAISE_ETA)
        J = mp.matrix(2 * M, 2 * M)
        for j in range(2 * M):
            hj = list(h0)
            e = eta * max(abs(hj[j]), mp.mpf(1))
            hj[j] += e
            Fj, _ = rung_F(seg, hj[:M], hj[M:], delta, node)
            for i in range(2 * M):
                J[i, j] = (Fj[i] - F0[i]) / e
            if (j + 1) % 8 == 0:
                print(f"  column {j + 1}/{2 * M}, {time.time() - t0:.0f} s", flush=True)
        if cache:
            with open(cache, "w") as fh:
                json.dump(dict(J=[[str(J[i, j]) for j in range(2 * M)] for i in range(2 * M)], prec=prec, ks=ks), fh)
    dg = [abs(J[i, i]) for i in range(2 * M)]
    rec["J_diag"] = dict(m=[float(min(dg[:M])), float(max(dg[:M]))], a=[float(min(dg[M:])), float(max(dg[M:]))])
    # the start projection: the stored data re-solved at order ks
    hb, hist_b, Rb = rung_newton(seg, h0, delta, node, J, tol)
    rec["start"] = dict(hist=hist_b, c=float(amax(Rb[3 * M:])), p=float(amax(Rb[:M])),
                        moved=float(amax(vsub(hb, h0))))
    print(f"  start projection at ks {ks}: |F| {' -> '.join(f'{x:.2e}' for x in hist_b)}; then c rows "
          f"{rec['start']['c']:.3e}, data moved by {rec['start']['moved']:.3e}; {time.time() - t0:.0f} s", flush=True)
    if hist_b[-1] > tol:
        print("  FAIL: the start projection did not converge", flush=True)
        return rec, False
    # the predictor: s = -J^-1 dF/dDelta (forward difference at the base), w from the second difference along (s, 1)
    eta = mp.mpf("1e-45")
    Fe, _ = rung_F(seg, hb[:M], hb[M:], delta + eta, node)
    Fb, _ = rung_F(seg, hb[:M], hb[M:], delta, node)
    sv = mp.lu_solve(J, mp.matrix([(x - y) / eta for x, y in zip(Fe, Fb)]))
    s_ = [-sv[i] for i in range(2 * M)]
    ws = {}
    for tau in RAISE_TAUS:
        tau = mp.mpf(tau)
        Fp, _ = rung_F(seg, *(lambda v: (v[:M], v[M:]))([x + tau * y for x, y in zip(hb, s_)]), delta + tau, node)
        Fm, _ = rung_F(seg, *(lambda v: (v[:M], v[M:]))([x - tau * y for x, y in zip(hb, s_)]), delta - tau, node)
        q2 = [(a + b - 2 * c) / (2 * tau * tau) for a, b, c in zip(Fp, Fm, Fb)]
        wv = mp.lu_solve(J, mp.matrix(q2))
        ws[float(tau)] = [-wv[i] for i in range(2 * M)]
    w = ws[RAISE_TAUS[0]]
    w2 = ws[RAISE_TAUS[1]]
    sc = float(amax(vsub(w, w2)) / amax(w))
    rec["predictor"] = dict(s=float(amax(s_)), w=float(amax(w)), t2_check=sc)
    print(f"  predictor: |s| {float(amax(s_)):.3e}, |w| {float(amax(w)):.3e} (t^2 check {sc:.2e} between tau "
          f"{RAISE_TAUS[0]:g} and {RAISE_TAUS[1]:g}); {time.time() - t0:.0f} s", flush=True)
    rec["moves"] = []
    for t in RAISE_MOVES:
        tm = mp.mpf(t)
        out = dict(t=t)
        for name, g in (("linear", [x + tm * y for x, y in zip(hb, s_)]),
                        ("second", [x + tm * y + tm * tm * v for x, y, v in zip(hb, s_, w)])):
            h, hist, _ = rung_newton(seg, g, delta + tm, node, J, tol)
            ok = bool(hist[-1] <= tol)
            out[name] = dict(hist=hist, converged=ok, passes=len(hist))
            if name == "second" and ok:
                d3 = vsub(h, g)
                out["third"] = dict(miss=float(amax(d3)), per_t3=float(amax(d3) / tm ** 3))
            print(f"  Delta + {t:g}, {name} start: |F| {' -> '.join(f'{x:.2e}' for x in hist)} "
                  f"({'converged' if ok else 'NOT converged'}); {time.time() - t0:.0f} s", flush=True)
        if "third" in out:
            print(f"    the second-order start missed the solution by {out['third']['miss']:.3e} "
                  f"({out['third']['per_t3']:.3e} t^3)", flush=True)
        rec["moves"].append(out)
    judged = "linear" if mutate == "linear" else "second"
    ok = bool(rec["moves"][-1][judged]["converged"]) and sc <= 0.1
    rec.update(judged=judged, ok=ok, seconds=round(time.time() - t0))
    print(f"raisecheck: the {judged} start at Delta + {RAISE_MOVES[-1]:g} "
          f"{'converged' if rec['moves'][-1][judged]['converged'] else 'did not converge'}"
          f"{'' if sc <= 0.1 else ', and the predictor failed its t^2 check'}: {'PASS' if ok else 'FAIL'}; "
          f"{rec['seconds']} s", flush=True)
    return rec, ok


# ------------------------------------------------------------------ planted truth
def planted(mutate=""):
    """the small planted problem: n = 100, M = 8, K = 40 (below Re s, about n/2, so the generic data hold no transient),
    x_K = 0.7, Delta = 3.3 and a smooth m_h. a_h is the slaved value (the factored fixed point with gamma = 0,
    B = B_sl) times 1 + (1e-3 / G) (1 + cos 2 theta / 2), so the series holds a mass mode of about 1e-3 at the node
    and converges at u0 as at the chain's states; the node is the old series' own sums there (the old rows are
    zero)"""
    M = 8
    seg = Seg("102", M, 40, 0.7)
    th = [2 * mp.pi * j / (2 * M) for j in range(M)]
    mh = [mp.mpf("0.02") * (mp.sin(t) + mp.mpf("0.2") * mp.sin(3 * t)) for t in th]
    delta = mp.mpf("3.3")
    om = seg.omega(delta)
    one = [mp.mpf(1)] * M
    sl = fixed_point(seg, mh, one, om, [mp.mpf(0)] * (4 * M), accel="anderson", maxit=60, mode="zero")
    dl = mp.mpf("1e-3") / seg.G                        # the mass mode's amplitude at the node about 1e-3
    ah = [(1 + dl * (1 + mp.cos(2 * t) / 2)) / mp.sqrt(1 + sl["B"][0][j]) for j, t in enumerate(th)]
    s = series(seg, mh, om, a0=ah, mutate=mutate)
    node = sum((horner(s[f], seg.u0) for f in ("p", "m", "a", "c")), [])
    return seg, mh, ah, om, s, node


def ode_residual(seg, s, om, u):
    """model.rs's header equations at u (x = 1 - u), from the series and their u-derivatives: the largest residual
    over the four equations and the points"""
    M = seg.M
    val = {f: horner(s[f], u) for f in ("p", "m", "a", "c")}
    # d/du sum_k c_k u^k = sum_k (k+1) c_(k+1) u^k
    der = {f: horner([[k * v for v in s[f][k]] for k in range(1, len(s[f]))], u) for f in ("p", "m", "a", "c")}
    x = 1 - u
    n1, nh = seg.n1, seg.nh
    p, m, a, c = val["p"], val["m"], val["a"], val["c"]
    px, mx, ax, cx = ([-v for v in der[f]] for f in ("p", "m", "a", "c"))   # d/dx = -d/du
    dp, dm = seg.dtau(p, om), seg.dtau(m, om)
    res = []
    for j in range(M):
        G = n1 * (c[j] * a[j] ** 2 - c[j])
        q = p[j] ** 2 + m[j] ** 2
        res.append(x * cx[j] - G)
        res.append(2 * x * ax[j] - (a[j] * q + n1 * (a[j] - a[j] ** 3)))
        res.append(x * (x - c[j]) * px[j] - (-x * dp[j] + G * p[j] - c[j] * p[j] + nh * c[j] * (p[j] + m[j])))
        res.append(x * (x + c[j]) * mx[j] - (-x * dm[j] - G * m[j] + c[j] * m[j] - nh * c[j] * (p[j] + m[j])))
    return amax(res)


def chi_closed(seg, q):
    """(1 - u)^-(n-1) exp(int_0^u q/(1 - s) ds) as a series to order K, by series log and exp: independent of
    chi_hat's recurrence"""
    M, K, n1 = seg.M, seg.K, seg.n1
    out = []
    for j in range(M):
        # L(u) = -(n-1) ln(1 - u) + int_0^u q/(1 - s) ds; ln(1 - u) = -sum u^k / k; q/(1 - s) = sum_k (sum_{i<=k} q_i) s^k
        Lc = [mp.mpf(0)] * (K + 1)
        csum = mp.mpf(0)
        for k in range(1, K + 1):
            csum += q[k - 1][j]
            Lc[k] = n1 / k + csum / k
        # E = exp(L): k E_k = sum_{i=1..k} i L_i E_(k-i)
        E = [mp.mpf(1)]
        for k in range(1, K + 1):
            E.append(mp.fdot([i * Lc[i] for i in range(1, k + 1)], [E[k - i] for i in range(1, k + 1)]) / k)
        out.append(E)
    return [[out[j][k] for j in range(M)] for k in range(K + 1)]


def selftest(mutate):
    mp.mp.prec = 256
    RECIP_DD[0] = False      # the algebra in exact 1/k: hp's double-double 1/k has no counterpart in B's rows (p1)
    eps = mp.mpf(2) ** -256
    tolr = mp.mpf(2) ** (-256 + 24)
    res = []

    def report(name, ok, detail):
        res.append((name, ok))
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)

    seg, mh, ah, om, s, node = planted(mutate)
    # S0: the old series against the header equations (its own mutation 'nh' is inside series)
    r0 = ode_residual(seg, s, om, seg.u0 / 8)
    report("S0 the old series solves model.rs's header equations at u0/8", r0 <= 1e-20, f"residual {fmt(r0)} (bound 1e-20)")
    if mutate == "nh":                                 # the node was made by the mutated series: rebuild it
        seg, mh, ah, om, s, node = planted("")
    q = s["q"]
    # S1: B_sl solves its rows
    Bs = b_sl(seg, q, mutate)
    K, M, n1 = seg.K, seg.M, seg.n1
    worst, big = mp.mpf(0), mp.mpf(0)
    for k in range(K + 1):
        qb = cs(q, Bs, k)
        for j in range(M):
            nxt = Bs[k + 1][j] if k < K else 0
            terms = [(k + 1) * nxt, (n1 + k) * Bs[k][j], qb[j], q[k][j]]
            worst = max(worst, abs(terms[0] - terms[1] - terms[2] - terms[3]))
            big = max(big, max(abs(t) for t in terms))
    report("S1 B_sl solves B's rows 0..K with b_(K+1) = 0", worst <= tolr * big, f"residual {fmt(worst / big)} relative")
    # S2: chi against its closed form
    ch, _ = chi_hat(seg, q, mutate)
    chu, _ = chi_hat(seg, q, "")
    cl = chi_closed(seg, q)
    # compare shapes: chi_hat / chi_hat_0 against the closed form's coefficients (chi_0 = 1)
    e2 = max(abs(ch[k][j] / ch[0][j] - cl[k][j]) / abs(cl[k][j]) for k in range(K + 1) for j in range(M))
    report("S2 chi equals (1-u)^-(n-1) exp(int q/(1-s)) coefficient by coefficient", e2 <= tolr, f"relative {fmt(e2)}")
    # S3: Miller's a = B^(-1/2), the residual of a a B = 1 relative to the sum of its terms' sizes
    B = [[Bs[k][j] + mp.mpf("1e-3") * chu[k][j] for j in range(M)] for k in range(K + 1)]
    y, _ = miller(B, None, mutate)
    yy = [cs(y, y, k) for k in range(K + 1)]
    ya = [[abs(v) for v in r] for r in y]
    yya = [cs(ya, ya, k) for k in range(K + 1)]
    B = [[1 + v for v in B[0]]] + B[1:]               # B itself from beta
    Ba = [[abs(v) for v in r] for r in B]
    e3 = max(abs(cs(yy, B, k)[j] - (1 if k == 0 else 0)) / cs(yya, Ba, k)[j] for k in range(K + 1) for j in range(M))
    report("S3 Miller's recurrence gives a with a a B = 1", e3 <= tolr, f"residual {fmt(e3)} relative")
    # S4, S5, S6: the fixed point and P1 on the planted problem
    R_old, vals = rows_old(seg, s, node)
    aK = vals[2]
    maxit = 1 if mutate == "nofp" else 60
    an = fixed_point(seg, mh, aK, om, node, accel="anderson", mem=5, maxit=maxit, mutate=mutate)
    pl = fixed_point(seg, mh, aK, om, node, accel="plain", maxit=maxit, mutate=mutate)
    G = seg.G
    d4 = amax(vsub(an["rows"], R_old))
    report("S4 P1 planted: factored rows equal the old rows", d4 <= 100 * eps * G,
           f"max diff {fmt(d4)} (bound {fmt(100 * eps * G)}, G {fmt(G)})")
    f5 = max(an["hist"][-1]["fp_res"], pl["hist"][-1]["fp_res"])
    report("S5 plain and Anderson sweeps reach the fixed point", f5 <= float(tolr),
           f"|g(y)-y| plain {pl['hist'][-1]['fp_res']:.2e} in {pl['sweeps']}, Anderson {an['hist'][-1]['fp_res']:.2e} "
           f"in {an['sweeps']} sweeps")
    cl1 = fixed_point(seg, mh, aK, om, node, accel="anderson", mem=5, maxit=maxit, mutate=mutate, closure=1)
    d6 = amax(vsub(cl1["rows"], an["rows"]))
    report("S6 the closure b_(K+1) = 1 gives the same rows", d6 <= 100 * eps * G, f"max diff {fmt(d6)}")
    fails = [n for n, ok in res if not ok]
    print(f"selftest{' --mutate=' + mutate if mutate else ''}: {len(res) - len(fails)} of {len(res)} pass"
          + (f"; FAIL: {', '.join(f.split()[0] for f in fails)}" if fails else ""))
    return dict(cmd="selftest", mutate=mutate, results=[(n, bool(o)) for n, o in res]), not fails


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opt = {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opt[k] = v
    # also accept "--key value"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" not in a and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            opt[a[2:]] = argv[i + 1]
            args.remove(argv[i + 1])
            i += 1
        i += 1
    if not args:
        print(__doc__)
        return 2
    cmd = args[0]
    mutate = opt.get("mutate", "")
    prec = int(opt.get("prec", 512))
    mem = int(opt.get("mem", 5))
    maxit = int(opt.get("maxit", 40))
    mode = opt.get("gamma", "a")
    RECIP_DD[0] = opt.get("recip", "dd") != "exact"
    if cmd == "selftest":
        rec, ok = selftest(mutate)
    elif cmd == "check":
        rec, ok = cmd_check(args[1], float(opt["bound"]), mutate, prec)
    elif cmd == "gain":
        rec, ok = cmd_gain(args[1], prec, mem, float(opt.get("warm", "1e-20")), maxit, mode, opt.get("start", "slaved"),
                           opt.get("accel", "plain"), opt.get("ref"), opt.get("ref_start", "slaved"))
    elif cmd == "p1":
        rec, ok = cmd_p1(args[1], prec, mutate, mode, mem, maxit, opt.get("ref"), float(opt.get("move", 0)))
    elif cmd == "curvjac":
        rec, ok = cmd_curvjac(args[1], prec, opt["cache"])
    elif cmd == "curv":
        rec, ok = cmd_curv(args[1], prec, opt["cache"], opt.get("move", "D"), opt.get("plant"), mutate,
                           opt.get("plant_c"))
    elif cmd == "curvx":
        rec, ok = cmd_curvx(args[1], prec, opt["ref"], opt.get("t", "0"), opt.get("cache"))
    elif cmd == "curvxq":
        rec, ok = cmd_curvxq(args[1:4], opt["t"], opt["curv"])
    elif cmd == "raisecheck":
        rec, ok = cmd_raisecheck(args[1], prec, int(opt["ks"]), opt.get("cache"), mutate)
    elif cmd == "loop":
        rec, ok = cmd_loop(args[1], prec, int(opt.get("its", 5)), opt.get("h", "1e-100"),
                           [int(x) for x in opt.get("orders", "60,100,125").split(",")])
    else:
        print(__doc__)
        return 2
    rec["mpmath"] = mp.__version__
    if opt.get("out"):
        with open(opt["out"], "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
