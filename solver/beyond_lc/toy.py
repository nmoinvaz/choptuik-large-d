"""The decisive toy test of notes/BEYOND_LIGHT_CONE.md section 0 (criteria written before this script was run).

Toy: Gauss's equation z(1-z) w'' + [kappa - (alpha+beta+1) z] w' - alpha beta w = 0 in z = x^2 (y = x w), with
kappa = (n+1)/2, alpha = (s - s*)/2 - 1, beta = 1 + s/2. It is the linear, gravity-free, one-harmonic problem of
Proposition 2 (alpha = (1+s)/2 there) with alpha shifted by a constant, so that the light-cone (Stokes) coefficient
C ~ 1/(Gamma(alpha) Gamma(beta)) vanishes at s = s*, exactly and for every n.

State vector Y = (w, (1-z) z w'/kappa): the scaling of (p - m, p + m) in the full problem.

Methods (each a residual R that vanishes at s*):
  S:xm   standard hp/ layout: centre series at x = 0.3, Taylor steps along the real axis to x_m, horizon series at the
         light cone evaluated at x_m; R = Y_c(x_m) - A_h Y_h(x_m), unknowns (s, A_h).
  C:xi0  candidate (c): from z = 1/2 along the upper and the lower semicircle in Re z >= 1/2 (the fast mode is never
         larger than the slow one there) to z0 = xi0^2 n on the real axis beyond the light cone; R = Y+(z0) - Y-(z0).
  A:zs   candidate (a) on the real axis: the light-cone-analytic solution from its horizon series at z = zs > 1,
         Taylor-stepped outward along the real axis to z0 = n, matched to Y+(z0) of C; unknowns (s, A_h).
Controls (must fail): Cuu (both paths above the light cone: no monodromy, nothing selected); Csing (C started from
the centre-singular solution, the x^{2-n} branch, instead of the regular one).

Measures: kappa_J = |row_s(J^-1)| |Y| (error in s per unit relative perturbation of the matched states), and the
empirical error delta_s(P) = row_s(J^-1) R_P(s*) of a residual evaluated at P decimal digits (R(s*) = 0 exactly, so
R_P is pure rounding); kappa_emp = median of |delta_s(P)| 10^P, and P_min, the smallest P with |delta_s| < 1e-8.

Usage: python3 toy.py [--n 20,60,100,200] [--methods S:0.7,S:0.9,C:1,A:1.5,...] [--out toy_results.jsonl]
       python3 toy.py --plant           (second planted value s* = 0.3 + 2.5i, Newton solves from a perturbed start)
Results are appended to the JSONL file one line per (method, n); lines already present are skipped (resumable).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OMEGA = 2 * math.pi / 1.6438937
PLAN_DPS = 110
NOISE = {'eps': None, 'rng': None}


def perturb(z, w, wp, ka):
    """rounding model: add a random complex error of size eps |Y| to each component of Y = (w, (1-z) z w'/kappa)"""
    if NOISE['eps'] is None:
        return w, wp
    rng = NOISE['rng']
    f = (1 - z) * z / ka
    y0, y1 = w, f * wp
    e = NOISE['eps'] * mp.sqrt(abs(y0) ** 2 + abs(y1) ** 2)
    y0 += e * mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1)) / math.sqrt(2)
    y1 += e * mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1)) / math.sqrt(2)
    return y0, y1 / f


def phi(x):
    return math.log(x) - 0.5 * math.log(1 - x * x)


class Toy:
    def __init__(self, n, sstar):
        self.n = n
        self.sstar = mp.mpc(sstar)

    def par(self, s):
        al = (s - self.sstar) / 2 - 1
        be = 1 + s / 2
        ka = mp.mpf(self.n + 1) / 2
        return al, be, ka

    def state(self, z, w, wp, ka):
        return [w, (1 - z) * z * wp / ka]

    # ---------------- series ----------------
    def centre(self, s, z, tol):
        al, be, ka = self.par(s)
        c = mp.mpc(1)
        w = mp.mpc(0)
        wp = mp.mpc(0)
        zj = mp.mpc(1)
        big = mp.mpf(0)
        absum = mp.mpf(0)
        j = 0
        run = 0
        while True:
            term = c * zj
            w += term
            absum += abs(term)
            if j > 0:
                wp += j * c * zj / z
            big = max(big, abs(term))
            run = run + 1 if abs(term) < tol * big else 0
            if run >= 4 and j > 10:
                break
            c = c * (al + j) * (be + j) / ((ka + j) * (j + 1))
            zj *= z
            j += 1
            if j > 20000:
                raise RuntimeError('centre series did not converge')
        return w, wp, absum / max(abs(w), mp.mpf(10) ** (-mp.mp.dps))

    def centre_singular(self, s, z, tol):
        """the second solution at z = 0: z^{1-kappa} 2F1(alpha-kappa+1, beta-kappa+1; 2-kappa; z) (principal branch)"""
        al, be, ka = self.par(s)
        a2, b2, c2 = al - ka + 1, be - ka + 1, 2 - ka
        c = mp.mpc(1)
        f = mp.mpc(0)
        fp = mp.mpc(0)
        zj = mp.mpc(1)
        big = mp.mpf(0)
        j = 0
        run = 0
        while True:
            term = c * zj
            f += term
            if j > 0:
                fp += j * c * zj / z
            big = max(big, abs(term))
            run = run + 1 if abs(term) < tol * big else 0
            if run >= 4 and j > 10:
                break
            c = c * (a2 + j) * (b2 + j) / ((c2 + j) * (j + 1))
            zj *= z
            j += 1
        pw = z ** (1 - ka)
        return pw * f, pw * fp + (1 - ka) * pw * f / z

    def horizon(self, s, z, tol):
        al, be, ka = self.par(s)
        cp = al + be - ka + 1
        t = 1 - z
        h = mp.mpc(1)
        w = mp.mpc(0)
        wp = mp.mpc(0)
        tj = mp.mpc(1)
        big = mp.mpf(0)
        absum = mp.mpf(0)
        jmin = int(max(0, -float(mp.re(cp)))) + 20
        j = 0
        run = 0
        while True:
            term = h * tj
            w += term
            absum += abs(term)
            if j > 0:
                wp -= j * h * tj / t
            big = max(big, abs(term))
            run = run + 1 if abs(term) < tol * big else 0
            if (run >= 6 and j > jmin) or h == 0:
                break
            h = h * (al + j) * (be + j) / ((cp + j) * (j + 1))
            tj *= t
            j += 1
            if j > 200000:
                raise RuntimeError('horizon series did not converge')
        return w, wp, absum / max(abs(w), mp.mpf(10) ** (-mp.mp.dps))

    # ---------------- Taylor steps ----------------
    def coeffs(self, s, zc, N, tol=None, h=None):
        """Taylor coefficients about zc of the two basis solutions (w, w') = (1, 0) and (0, 1)."""
        al, be, ka = self.par(s)
        q0 = zc * (1 - zc)
        q1 = 1 - 2 * zc
        r0 = ka - (al + be + 1) * zc
        r1 = -(al + be + 1)
        ab = al * be
        out = []
        for a0, a1 in ((1, 0), (0, 1)):
            a = [mp.mpc(a0), mp.mpc(a1)]
            run = 0
            for k in range(N - 1):
                num = (q1 * k + r0) * (k + 1) * a[k + 1] + (-k * (k - 1) + r1 * k - ab) * a[k]
                a.append(-num / (q0 * (k + 2) * (k + 1)))
                if tol is not None and k > 8:
                    run = run + 1 if abs(a[-1]) * abs(h) ** (k + 2) < tol else 0
                    if run >= 3:
                        break
            out.append(a)
        return out

    @staticmethod
    def stepmat(seqs, h):
        M = []
        for a in seqs:
            w = mp.mpc(0)
            wp = mp.mpc(0)
            for k in range(len(a) - 1, -1, -1):
                w = w * h + a[k]
                if k >= 1:
                    wp = wp * h + k * a[k]
            M.append((w, wp))
        # columns are the basis solutions
        return [[M[0][0], M[1][0]], [M[0][1], M[1][1]]]

    def plan(self, s, pts_fun, N, dps):
        """adaptive step points along a path z(t), t in [0, 1] (pts_fun maps t -> z); returns the list of z."""
        with mp.workdps(dps):
            tol = mp.mpf(10) ** (-dps)
            zs = [pts_fun(0.0)]
            t = 0.0
            while t < 1.0:
                zc = zs[-1]
                seqs = self.coeffs(s, zc, N)
                hmax = None
                for a in seqs:
                    for k in (N - 1, N - 2):
                        if abs(a[k]) > 0:
                            hk = float((tol / abs(a[k])) ** (mp.mpf(1) / k))
                            hmax = hk if hmax is None else min(hmax, hk)
                dist = min(abs(complex(zc)), abs(complex(1 - zc)))
                hmax = min(0.8 * hmax, 0.5 * dist)
                # advance t so that |z(t') - zc| <= hmax (bisection on t')
                lo, hi = t, 1.0
                if abs(complex(pts_fun(1.0)) - complex(zc)) <= hmax:
                    zs.append(pts_fun(1.0))
                    break
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if abs(complex(pts_fun(mid)) - complex(zc)) <= hmax:
                        lo = mid
                    else:
                        hi = mid
                t = lo
                zs.append(pts_fun(t))
                if len(zs) > 200000:
                    raise RuntimeError('path planning runaway')
            return zs

    def transport(self, s, zs, w, wp, N, tol):
        ka = self.par(s)[2]
        for i in range(len(zs) - 1):
            zc = mp.mpc(zs[i])
            h = mp.mpc(zs[i + 1]) - zc
            M = self.stepmat(self.coeffs(s, zc, N, tol=tol, h=h), h)
            w, wp = M[0][0] * w + M[0][1] * wp, M[1][0] * w + M[1][1] * wp
            w, wp = perturb(zc + h, w, wp, ka)
        return w, wp


# ------------------------------------------------------------------------------------------------------------
# methods: each returns (residual list, scale, unknowns-extra) at given s (and A_h for two-unknown methods)
# ------------------------------------------------------------------------------------------------------------
class Method:
    N = 120

    def __init__(self, toy, spec):
        self.toy = toy
        self.spec = spec
        kind, _, arg = spec.partition(':')
        self.kind = kind
        self.arg = float(arg) if arg else None
        n = toy.n
        st = toy.sstar
        if kind == 'S':
            z1, zm = mp.mpf('0.09'), mp.mpf(self.arg) ** 2
            self.zm = zm
            self.path = toy.plan(st, lambda t: z1 + (zm - z1) * t, self.N, PLAN_DPS)
            self.path[-1] = zm
            self.two = True
        elif kind in ('C', 'Cuu', 'Csing', 'A'):
            xi0 = self.arg if kind != 'A' else 1.0
            z0 = mp.mpf(xi0 ** 2 * n) if kind != 'A' else mp.mpf(n)
            self.z0 = z0
            za = mp.mpf('0.5')
            cen = (za + z0) / 2
            rad = (z0 - za) / 2
            self.pre = toy.plan(st, lambda t: mp.mpf('0.09') + (za - mp.mpf('0.09')) * t, self.N, PLAN_DPS)
            up = lambda t: cen + rad * mp.expj(mp.pi * (1 - t))
            dn = lambda t: cen + rad * mp.expj(-mp.pi * (1 - t))
            self.up = toy.plan(st, up, self.N, PLAN_DPS)
            if kind == 'Cuu':
                # a second path above the light cone: up to Im z = rad/2, across, and down to z0
                h2 = rad / 2

                def up2(t):
                    if t < 1 / 3:
                        return za + 1j * h2 * (3 * t)
                    if t < 2 / 3:
                        return za + (z0 - za) * (3 * t - 1) + 1j * h2
                    return z0 + 1j * h2 * (1 - (3 * t - 2))
                self.dn = toy.plan(st, up2, self.N, PLAN_DPS)
            else:
                self.dn = toy.plan(st, dn, self.N, PLAN_DPS)
            self.two = kind == 'A'
            if kind == 'A':
                zs = mp.mpf(self.arg)
                self.zs = zs
                self.out = toy.plan(st, lambda t: zs + (z0 - zs) * t, self.N, PLAN_DPS)
        else:
            raise ValueError(spec)

    def nsteps(self):
        if self.kind == 'S':
            return len(self.path) - 1
        k = len(self.pre) + len(self.up) + len(self.dn) - 3
        if self.kind == 'A':
            k += len(self.out) - 1
        return k

    def residual(self, s, Ah=None):
        toy = self.toy
        tol = mp.mpf(10) ** (-mp.mp.dps - 8)
        al, be, ka = toy.par(s)
        if self.kind == 'S':
            w, wp, amp_c = toy.centre(s, mp.mpf('0.09'), tol)
            w, wp = perturb(mp.mpf('0.09'), w, wp, ka)
            w, wp = toy.transport(s, self.path, w, wp, self.N, tol)
            zm = mp.mpf(self.zm)
            Yc = toy.state(zm, w, wp, ka)
            wh, whp, amp_h = toy.horizon(s, zm, tol)
            wh, whp = perturb(zm, wh, whp, ka)
            Yh = toy.state(zm, wh, whp, ka)
            return Yc, Yh, max(amp_c, amp_h)
        if self.kind == 'Csing':
            w0, wp0 = toy.centre_singular(s, mp.mpf('0.09'), tol)
            amp = 1
        else:
            w0, wp0, amp = toy.centre(s, mp.mpf('0.09'), tol)
        w0, wp0 = perturb(mp.mpf('0.09'), w0, wp0, ka)
        w, wp = toy.transport(s, self.pre, w0, wp0, self.N, tol)
        wu, wpu = toy.transport(s, self.up, w, wp, self.N, tol)
        wd, wpd = toy.transport(s, self.dn, w, wp, self.N, tol)
        z0 = self.z0
        Yu = toy.state(z0, wu, wpu, ka)
        Yd = toy.state(z0, wd, wpd, ka)
        if self.kind == 'A':
            wh, whp, amp_h = toy.horizon(s, self.zs, tol)
            wh, whp = perturb(self.zs, wh, whp, ka)
            wh, whp = toy.transport(s, self.out, wh, whp, self.N, tol)
            Yh = toy.state(z0, wh, whp, ka)
            return Yu, Yh, max(amp, amp_h)
        return Yu, Yd, amp


def vec_norm(v):
    return mp.sqrt(sum(abs(x) ** 2 for x in v))


def solve2(J, r):
    """least-squares / exact solve of J x = r for 2 x m complex J (m = 1 or 2)"""
    m = len(J[0])
    if m == 1:
        a = J[0][0]
        b = J[1][0]
        den = abs(a) ** 2 + abs(b) ** 2
        return [(mp.conj(a) * r[0] + mp.conj(b) * r[1]) / den]
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    return [(J[1][1] * r[0] - J[0][1] * r[1]) / det, (-J[1][0] * r[0] + J[0][0] * r[1]) / det]


def row_s(J):
    """row of the (pseudo-)inverse that gives s: as a 2-vector g with delta s = g . delta r"""
    m = len(J[0])
    if m == 1:
        a = J[0][0]
        b = J[1][0]
        den = abs(a) ** 2 + abs(b) ** 2
        return [mp.conj(a) / den, mp.conj(b) / den]
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    return [J[1][1] / det, -J[0][1] / det]


def resid_vec(meth, s, Ah):
    A, B, amp = meth.residual(s)
    if meth.two:
        return [A[0] - Ah * B[0], A[1] - Ah * B[1]], A, B, amp
    return [A[0] - B[0], A[1] - B[1]], A, B, amp


def jacobian(meth, s, Ah, dps):
    with mp.workdps(dps):
        # five-point stencil: truncation O(h^4) = 1e-80 at h = 1e-20 (a central difference's O(h^2) = 1e-40 would
        # mask determinants below about 1e-40, i.e. kappa above 1e40)
        h = mp.mpf(10) ** (-20 if dps >= 60 else -(dps // 4))
        rp, A, B, _ = resid_vec(meth, s + h, Ah)
        rm, _, _, _ = resid_vec(meth, s - h, Ah)
        rp2, _, _, _ = resid_vec(meth, s + 2 * h, Ah)
        rm2, _, _, _ = resid_vec(meth, s - 2 * h, Ah)
        r0, A, B, amp = resid_vec(meth, s, Ah)
        ds = [(8 * (rp[i] - rm[i]) - (rp2[i] - rm2[i])) / (12 * h) for i in range(2)]
        if meth.two:
            J = [[ds[0], -B[0]], [ds[1], -B[1]]]
        else:
            J = [[ds[0]], [ds[1]]]
        return J, A, B, r0, amp


def measure(spec, n, sstar):
    toy = Toy(n, sstar)
    t0 = time.time()
    with mp.workdps(PLAN_DPS):
        meth = Method(toy, spec)
        st = toy.sstar
        A0, B0, _ = meth.residual(st)
        Ah = None
        if meth.two:
            Ah = (A0[0] * mp.conj(B0[0]) + A0[1] * mp.conj(B0[1])) / (abs(B0[0]) ** 2 + abs(B0[1]) ** 2)
        J, A, B, r0, amp = jacobian(meth, st, Ah, PLAN_DPS)
        g = row_s(J) if not all(abs(J[i][0]) == 0 for i in range(2)) else None
        scale = vec_norm(A)
        jn = vec_norm([J[0][0], J[1][0]]) / scale
        kJ = float(vec_norm(g) * scale) if g is not None else float('inf')
        res0 = float(vec_norm(r0) / scale)
    rec = {'method': spec, 'n': n, 'sstar': [float(mp.re(sstar)), float(mp.im(sstar))], 'steps': meth.nsteps(),
           'kappa_J': kJ, 'J_s_rel': float(jn), 'res_at_sstar_rel_plan': res0, 'series_amp': float(amp)}
    if meth.kind == 'S':
        rec['pred_exp_nPhi'] = n * phi(meth.arg)
    # rounding model: relative noise 10^-P injected after every series evaluation and Taylor step, arithmetic at
    # P + 15 digits; one Newton step from the exact s* gives the error in s that precision P leaves.
    if g is not None:
        P = int(max(30, math.ceil(math.log10(max(kJ, 1.0))) + 15))
        draws = []
        for seed in range(3):
            NOISE['eps'] = mp.mpf(10) ** (-P)
            NOISE['rng'] = random.Random(1000 * n + seed)
            with mp.workdps(P + 15):
                r, A, B, _ = resid_vec(meth, mp.mpc(st), mp.mpc(Ah) if Ah is not None else None)
                draws.append(float(abs(sum(mp.mpc(g[i]) * r[i] for i in range(2)))) * 10.0 ** P)
            NOISE['eps'] = None
        rec['noise_P'] = P
        rec['kappa_emp_draws'] = draws
        rec['kappa_emp'] = max(draws)
        rec['P_min'] = int(math.ceil(math.log10(max(draws)) + 8)) if max(draws) > 0 else None
    else:
        rec['kappa_emp'] = None
        rec['P_min'] = None
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


def newton_check(spec, n, sstar, start_shift, dps=60, iters=30):
    """Newton (Gauss-Newton) from s* + start_shift; returns the final s and the step history."""
    toy = Toy(n, sstar)
    with mp.workdps(PLAN_DPS):
        meth = Method(toy, spec)
    with mp.workdps(dps):
        s = mp.mpc(sstar) + start_shift
        Ah = None
        if meth.two:
            A0, B0, _ = meth.residual(s)
            Ah = A0[0] / B0[0]
        hist = []
        for it in range(iters):
            J, A, B, r, _ = jacobian(meth, s, Ah, dps)
            x = solve2(J, r)
            s = s - x[0]
            if meth.two:
                Ah = Ah - x[1]
            hist.append(float(abs(x[0])))
            if abs(x[0]) < mp.mpf(10) ** (-dps + 10):
                break
        return s, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='20,60,100,200')
    ap.add_argument('--methods', default='S:0.7,C:1,S:0.9,A:1.5,S:0.5,S:0.6,S:0.8,S:0.7071067811865476')
    ap.add_argument('--out', default=os.path.join(HERE, 'toy_results.jsonl'))
    ap.add_argument('--sstar', default=None)
    ap.add_argument('--plant', action='store_true')
    a = ap.parse_args()
    sstar = mp.mpc(0, OMEGA) if a.sstar is None else mp.mpc(complex(a.sstar))
    if a.plant:
        for spec in ['S:0.7', 'C:1']:
            for sst, n in ((mp.mpc(0.3, 2.5), 60), (mp.mpc(0, OMEGA), 60)):
                s, hist = newton_check(spec, n, sst, mp.mpc(0.05, -0.04))
                print(f'plant {spec} n={n} s*={complex(sst)}: Newton -> {complex(s)}  |s-s*| = {float(abs(s - sst)):.1e}'
                      f'  steps {["%.0e" % h for h in hist]}', flush=True)
        for spec in ['Cuu:1', 'Csing:1']:
            try:
                s, hist = newton_check(spec, 60, mp.mpc(0, OMEGA), mp.mpc(0.05, -0.04), iters=12)
                print(f'control {spec} n=60: Newton -> {complex(s)}  |s-s*| = {float(abs(s - mp.mpc(0, OMEGA))):.1e}'
                      f'  steps {["%.0e" % h for h in hist]}', flush=True)
            except Exception as e:  # a singular Jacobian is the expected failure of Cuu
                print(f'control {spec} n=60: Newton failed ({type(e).__name__}: {e})', flush=True)
        return 0
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r['method'], r['n'], tuple(r['sstar'])))
    key_s = (float(mp.re(sstar)), float(mp.im(sstar)))
    for spec in a.methods.split(','):
        for n in [int(v) for v in a.n.split(',')]:
            if (spec, n, key_s) in done:
                continue
            rec = measure(spec, n, sstar)
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{spec:22s} n={n:4d} steps={rec['steps']:5d} kappa_J={rec['kappa_J']:.2e} kappa_emp="
                  f"{rec['kappa_emp'] if rec['kappa_emp'] is None else '%.2e' % rec['kappa_emp']} P_min={rec['P_min']}"
                  f" res(s*)={rec['res_at_sstar_rel_plan']:.1e} amp={rec['series_amp']:.1e} ({rec['seconds']} s)",
                  flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
