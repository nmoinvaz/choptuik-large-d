"""#1098: the shared numerics of hp/sing_types.py: data selection, planted truths, weighted rational least squares
in arb (Sanathanan-Koerner start, then Levenberg-Marquardt on the true residuals), and the poles, zeros and residues
of a fit.

Variable: e = 1/(D - 2), scaled t = e / s with s the largest e fitted, so the fitted t lie in [0, 1]. A fit [L/M] is
P(t) / Q(t), P of degree L and Q of degree M with Q(0) = 1, in the monomial basis of t (arb at PREC bits, so the
normal equations' condition, up to about 1e40 here, costs no digits that matter).
Sanathanan and Koerner, IEEE Trans. Automat. Control 8 (1963) 56 (the reweighted linear start); Levenberg, Quart.
Appl. Math. 2 (1944) 164, and Marquardt, J. SIAM 11 (1963) 431 (the damped Gauss-Newton steps).
"""
import json
import math
import random

import mpmath as mp
from flint import arb, arb_mat, ctx

PREC = 384
ctx.prec = PREC
mp.mp.dps = 80


def a(v):
    return arb(mp.nstr(mp.mpf(v), 100)) if not isinstance(v, arb) else v


LOCAL_UP = ("local-up15", "local-up16", "local-up17", "up18", "up20", "up21", "local-up22", "local-up23")


def sigma_eff(p):
    """the sigma used in the fits: the ladder's err_sum; for the low-D rows about twice the point-to-point scatter
    that sing_noise.py measures (leave-one-out local polynomials, K 8, H 7; its planted check passes for D >= 16),
    which differs from the table's err in both directions (raw I1098-mac-sing/noise_real_K8H7.log):
      dd, runs up7 on (D >= 35.35, err 2e-15), and qd (err 2.4e-23): rms 4e-17 to 6.5e-17, so 1e-16;
      dd, runs up2-up5 (D 16.76-26.84, err 1e-16): rms 9e-17 to 1.3e-16, so 2e-16;
      dd, run up6 (D 26.9-35.2, err 1e-16 and 2e-15): deviations to 1e-14 with alternating signs, so 5e-15;
      dd, runs local-up15 to up23 (D 60.85-61.03, err 2e-15): deviations to 3.4e-15, so 4e-15;
      dd, run up (D 8.25-16.75, err 1e-16): not measurable this way (the local polynomial's truncation dominates
      below D 16.5), so 1e-15 as a guard;
      chain_fit keeps its err: it scatters by 5e-17 to 7e-17 below D 200, but departs smoothly from the fits to the
      other data by up to 1.6e-12 (a systematic error, inside its err)."""
    D, k = float(p["D"]), p["kind"]
    if k == "ladder" or k == "chain_fit":
        return p["sigma"]
    if k == "qd":
        return 1e-16
    if k == "dd":
        run = p["src"].split(":")[-1]
        if run in LOCAL_UP:
            return 4e-15
        if run == "up":
            return 1e-15
        if run == "up6":
            return 5e-15
        if run in ("up2", "up3", "up4", "up5"):
            return 2e-16
        return 1e-16
    raise ValueError(k)


def thin_kind(pts, kind, n):
    """about n of the points of this kind, the nearest to n targets evenly spaced in e"""
    sel = sorted((p for p in pts if p["kind"] == kind), key=lambda p: 1 / (float(p["D"]) - 2))
    if not n or len(sel) <= n:
        return pts
    es = [1 / (float(p["D"]) - 2) for p in sel]
    keep = set()
    for i in range(n):
        tgt = es[0] + (es[-1] - es[0]) * i / (n - 1)
        keep.add(min(range(len(sel)), key=lambda j: abs(es[j] - tgt)))
    ids = {id(sel[j]) for j in keep}
    return [p for p in pts if p["kind"] != kind or id(p) in ids]


def load(path, kinds=("ladder", "qd", "dd", "chain_fit"), Dmin=0.0, Dmax=1e9, thin=None, sig="eff", drop_runs=()):
    """points of the data file (sing_data.py) as dicts with D, e, y (mpf), sigma, kind; thin {kind: n} keeps about n
    points of that kind, evenly spaced in e (an int: the dd rows only); sig 'eff' (sigma_eff) or 'table' (the err);
    drop_runs: dd runs left out (their source's run name)"""
    pts = [p for p in json.load(open(path))["points"] if p["kind"] in kinds and Dmin <= float(p["D"]) <= Dmax and
           not (p["kind"] == "dd" and p["src"].split(":")[-1] in drop_runs)]
    if sig == "eff":
        for p in pts:
            p["sigma"] = sigma_eff(p)
    if isinstance(thin, int):
        thin = {"dd": thin}
    for k, n in (thin or {}).items():
        pts = thin_kind(pts, k, n)
    out = []
    for p in pts:
        D = mp.mpf(p["D"])
        out.append(dict(D=D, e=1 / (D - 2), y=mp.mpf(p["Delta"]), sigma=p["sigma"], kind=p["kind"]))
    out.sort(key=lambda p: p["e"])
    return out


# ---------------------------------------------------------------- planted truths: T1 background plus one singularity
RA = ["1.6438975180", "4.0378045077e1", "2.5233285989e2", "2.0673502321e2", "-3.5709850575e2", "3.1347320996e2",
      "-4.1891374574e2", "3.2695252782e2", "-1.6050467699e2"]           # track_c.RA, the README's rational formula
RB = ["1.8251453626e1", "6.8182050167e1"]
E0 = mp.mpf("-0.061")                                                     # the planted singularity, n = -16.39


def t1(e):
    return mp.fsum(mp.mpf(c) * e ** k for k, c in enumerate(RA)) / (1 + mp.mpf(RB[0]) * e + mp.mpf(RB[1]) * e ** 2)


def t1_taylor(K):
    return mp.taylor(t1, 0, K)


def sing_part(kind, e, r=None):
    """the singular function g(u), u = 1 + e / |e0|, of each planted type, and its exponent where one is defined"""
    r = -E0 if r is None else r
    u = 1 + e / r
    if kind == "pole":
        return 1 / u
    if kind == "pole2":
        return 1 / u ** 2
    if kind == "sqrt":
        return mp.sqrt(u)
    if kind == "isqrt":
        return 1 / mp.sqrt(u)
    if kind == "s32":
        return u * mp.sqrt(u)
    if kind == "cbrt":
        return mp.cbrt(u)
    if kind == "log":
        return mp.log(u)
    if kind == "ulog":
        return u * mp.log(u)
    if kind == "ess":
        return mp.exp(-mp.mpf("0.25") / u)
    raise ValueError(kind)


SING_KINDS = ("pole", "pole2", "sqrt", "isqrt", "s32", "cbrt", "log", "ulog", "ess")
EXPONENT = dict(pole=-1, pole2=-2, sqrt=0.5, isqrt=-0.5, s32=1.5, cbrt=1 / 3, log=0, ulog=1)


class Truth:
    """T1 + A [g(u) - its Taylor polynomial to order 3 in e], A so that the added e^6 coefficient equals T1's e^6
    coefficient, sign included (frac = 1; the total's a_6 is then twice T1's), so that the low orders are T1's and
    the added part reinforces T1's alternating series; kind 'T1' is T1 alone. For pole2's and the pole's even
    coefficients the subtracted cubic leaves e^6 as the first nonzero... (all kinds have a nonzero e^6 coefficient)"""

    def __init__(self, kind, e0=E0, K6=6, frac=1):
        self.kind = kind
        self.e0 = mp.mpf(e0)
        if kind == "T1":
            self.A = mp.mpf(0)
            self.c = []
            return
        r = -self.e0
        g = lambda e: sing_part(kind, e, r)                               # noqa: E731
        c = mp.taylor(g, 0, K6)
        self.c = c[:4]
        self.A = frac * t1_taylor(K6)[K6] / c[K6]
        self.r = r

    def __call__(self, e):
        v = t1(e)
        if self.kind == "T1":
            return v
        g = sing_part(self.kind, e, self.r)
        return v + self.A * (g - mp.fsum(ck * e ** k for k, ck in enumerate(self.c)))


class Sum:
    """T1 plus the added parts of several Truths (each Truth minus T1)"""

    def __init__(self, parts):
        self.parts = parts

    def __call__(self, e):
        return t1(e) + mp.fsum(p(e) - t1(e) for p in self.parts)


def truth_spec(spec):
    """a planted truth from its name: KIND[@E0][*FRAC], or several joined by '+' (T1 plus each added part), e.g.
    'pole@-0.061+sqrt@-0.07*0.5' (a pole with a square-root branch point behind it)"""
    parts = []
    for one in spec.split("+"):
        one, _, frac = one.partition("*")
        kind, _, e0 = one.partition("@")
        parts.append(Truth(kind, e0=mp.mpf(e0) if e0 else E0, frac=mp.mpf(frac) if frac else 1))
    return parts[0] if len(parts) == 1 else Sum(parts)


def planted(pts, truth, seed, pattern="gauss"):
    """the points with y replaced by truth(e) + noise: Gaussian of each sigma (seeded), or a systematic pattern of size
    sigma per kind ('ramp': linear in e from -sigma to +sigma within each kind)"""
    rng = random.Random(seed)
    out = []
    lims = {}
    for p in pts:
        lo, hi = lims.get(p["kind"], (p["e"], p["e"]))
        lims[p["kind"]] = (min(lo, p["e"]), max(hi, p["e"]))
    for p in pts:
        q = dict(p)
        z = rng.gauss(0, 1)
        if pattern == "ramp":
            lo, hi = lims[p["kind"]]
            z = -1 + 2 * (p["e"] - lo) / (hi - lo) if hi > lo else 0
        q["y"] = truth(p["e"]) + mp.mpf(z) * mp.mpf(p["sigma"])
        out.append(q)
    return out


# ---------------------------------------------------------------- weighted least squares in arb
def lstsq(rows, rhs):
    """least squares for the arb rows (lists) and rhs (list); normal equations solved at PREC bits; midpoints"""
    A = arb_mat(rows)
    b = arb_mat([[v] for v in rhs])
    At = A.transpose()
    try:
        x = (At * A).solve(At * b, nonstop=True)
    except Exception:
        return None
    out = []
    for i in range(x.nrows()):
        v = x[i, 0]
        if not v.is_finite():
            return None
        out.append(v.mid())
    return out


def polyval(c, t):
    v = arb(0)
    for ck in reversed(c):
        v = v * t + ck
    return v


class Fit:
    """(P + S R) / Q of degrees Lp, Lr, M fitted to pts (dicts with e, y, sigma) by weighted least squares, S a given
    function of e (the singular factor of a typed form; none for a rational fit [Lp/M]); ref is subtracted from y
    first (the fit is of y - ref) so that the arithmetic keeps the ladder's digits"""

    def __init__(self, pts, L, M, ref=None, iters=40, Lr=-1, S=None):
        self.L, self.M, self.Lr, self.S = L, M, Lr, S
        self.s = max(p["e"] for p in pts)
        self.ref = mp.mpf(ref) if ref is not None else mp.mpf(0)
        ts = [a(p["e"] / self.s) for p in pts]
        ys = [a(p["y"] - self.ref) for p in pts]
        ws = [1 / a(p["sigma"]) for p in pts]
        Ss = [a(S(p["e"])) for p in pts] if Lr >= 0 else [arb(0)] * len(pts)
        self.N = len(pts)
        self.ok = False
        K = max(L, M, Lr) + 1
        tp = [[t ** k for k in range(K)] for t in ts]
        nP, nR = L + 1, Lr + 1
        npar = nP + nR + M
        self.npar = npar

        def basis(i, Qv):
            return ([tp[i][k] / Qv for k in range(nP)] + [Ss[i] * tp[i][k] / Qv for k in range(nR)])

        # Sanathanan-Koerner: min sum w^2 / Qold^2 (P + S R - y Q)^2, Q = 1 + sum q_j t^j
        qold = [arb(1)] * self.N
        par = None
        for _ in range(6):
            rows, rhs = [], []
            for i in range(self.N):
                wi = ws[i] / qold[i]
                rows.append([wi * v for v in basis(i, arb(1))] + [-wi * ys[i] * tp[i][j] for j in range(1, M + 1)])
                rhs.append(wi * ys[i])
            par = lstsq(rows, rhs)
            if par is None:
                return
            q = [arb(1)] + par[nP + nR:]
            qold = [polyval(q, t) for t in ts]
            if any(abs(v) < arb("1e-30") for v in qold):
                return

        def resid(par):
            p, rr, q = par[:nP], par[nP:nP + nR], [arb(1)] + par[nP + nR:]
            r, J = [], []
            for i in range(self.N):
                Nv = polyval(p, ts[i]) + (Ss[i] * polyval(rr, ts[i]) if nR else 0)
                Qv = polyval(q, ts[i])
                r.append(ws[i] * (Nv / Qv - ys[i]))
                J.append([ws[i] * v for v in basis(i, Qv)] + [-ws[i] * Nv * tp[i][j] / Qv ** 2 for j in range(1, M + 1)])
            return r, J
        # Levenberg-Marquardt on r_i = w_i (N/Q - y_i)
        lam = arb("1e-8")
        r, J = resid(par)
        c2 = sum((v * v for v in r), arb(0))
        for _ in range(iters):
            Jm = arb_mat(J)
            JT = Jm.transpose()
            JTJ = JT * Jm
            g = JT * arb_mat([[v] for v in r])
            Ad = arb_mat([[JTJ[i, j] + (lam * JTJ[i, i] if i == j else 0) for j in range(npar)] for i in range(npar)])
            try:
                dp = Ad.solve(-g, nonstop=True)
            except Exception:
                break
            new = [(par[k] + dp[k, 0]).mid() for k in range(npar)]
            if not all(v.is_finite() for v in new):
                lam *= 100
                if lam > arb("1e12"):
                    break
                continue
            rn, Jn = resid(new)
            cn = sum((v * v for v in rn), arb(0))
            if cn < c2:
                rel = float((c2 - cn) / c2) if c2 > 0 else 0.0
                par, r, J, c2 = new, rn, Jn, cn
                lam = max(lam / 10, arb("1e-30"))
                if rel < 1e-12:
                    break
            else:
                lam *= 10
                if lam > arb("1e12"):
                    break
        self.par = par
        self.p = par[:nP]
        self.rr = par[nP:nP + nR]
        self.q = [arb(1)] + par[nP + nR:]
        self.res = [float(v) for v in r]
        self.chi2 = float(c2)
        self.dof = self.N - npar
        self.maxres = max(abs(v) for v in self.res)
        self.ok = all(math.isfinite(v) for v in self.res)

    def _m(self, c):
        return [mp.mpf(x.str(70, radius=False)) for x in c]

    def __call__(self, e):
        e = mp.mpf(e)
        t = e / self.s
        P, Q = mp.polyval(list(reversed(self._m(self.p))), t), mp.polyval(list(reversed(self._m(self.q))), t)
        if self.Lr >= 0:
            P += self.S(e) * mp.polyval(list(reversed(self._m(self.rr))), t)
        return P / Q + self.ref

    def ev_c(self, e):
        """complex evaluation of a rational fit (mpmath)"""
        t = mp.mpc(e) / self.s
        P = mp.polyval(list(reversed(self._m(self.p))), t)
        Q = mp.polyval(list(reversed(self._m(self.q))), t)
        return P / Q + self.ref

    def amp(self, e0):
        """R(e0) / Q(e0): the singular term's coefficient at e0 (typed fits)"""
        t = mp.mpf(e0) / self.s
        return (mp.polyval(list(reversed(self._m(self.rr))), t) / mp.polyval(list(reversed(self._m(self.q))), t)
                if self.Lr >= 0 else mp.mpf(0))

    def taylor(self, K):
        """the Taylor coefficients a_0..a_K in e of a rational fit, exactly (power-series division of P by Q)"""
        pm, qm = self._m(self.p), self._m(self.q)
        c = []
        for k in range(K + 1):
            v = (pm[k] if k < len(pm) else 0) - mp.fsum(qm[j] * c[k - j] for j in range(1, min(k, len(qm) - 1) + 1))
            c.append(v)
        c[0] += self.ref
        return [c[k] / self.s ** k for k in range(K + 1)]

    def poles(self):
        qm = self._m(self.q)
        if len(qm) < 2:
            return []
        try:
            return [complex(t * self.s) for t in mp.polyroots(list(reversed(qm)), maxsteps=400, extraprec=400)]
        except mp.libmp.libhyper.NoConvergence:
            return None

    def roots(self):
        """poles (roots of Q) and zeros (roots of P + ref Q, the zeros of the fitted function itself), in e, each pole
        with its residue in e (rational fits)"""
        pm, qm = self._m(self.p), self._m(self.q)
        full = [pm[k] if k < len(pm) else mp.mpf(0) for k in range(max(len(pm), len(qm)))]
        for k in range(len(qm)):
            full[k] += self.ref * qm[k]
        while len(full) > 1 and full[-1] == 0:
            full.pop()

        def rts(c):
            if len(c) < 2:
                return []
            try:
                return mp.polyroots(list(reversed(c)), maxsteps=400, extraprec=400)
            except mp.libmp.libhyper.NoConvergence:
                return None
        qr, pr = rts(qm), rts(full)
        if qr is None or pr is None:
            return None
        dq = [k * qm[k] for k in range(1, len(qm))]
        poles = []
        for t in qr:
            Pv = mp.polyval(list(reversed(full)), t)
            Qd = mp.polyval(list(reversed(dq)), t)
            poles.append((complex(t * self.s), complex(self.s * Pv / Qd)))
        zeros = [complex(t * self.s) for t in pr]
        return poles, zeros


def RatFit(pts, L, M, ref=None, iters=40):
    return Fit(pts, L, M, ref=ref, iters=iters)


class Quad:
    """quadratic Hermite-Pade form A g^2 + B g + C = 0 fitted to g = y - ref: polynomials A, B, C in t = e / s of
    degrees na, nb, nc, B(0) = 1, by weighted least squares on the algebraic residual, reweighted (as
    Sanathanan-Koerner) by 1 / (sigma |2 A g + B|) of the previous pass so that it approximates the residual in g.
    The physical branch is the root nearer the data; its residuals in sigma, the zeros of the discriminant
    B^2 - 4 A C (the branch points) and of A (where a branch is infinite) are reported. Hermite, J. reine angew. Math.
    76 (1873) 342; Pade, Ann. Sci. Ec. Norm. Sup. 9 (1892) suppl. 3 (the approximants); Shafer, SIAM J. Numer. Anal.
    11 (1974) 447 (the quadratic ones)."""

    def __init__(self, pts, na, nb, nc, ref, passes=8):
        self.na, self.nb, self.nc = na, nb, nc
        self.s = max(p["e"] for p in pts)
        self.ref = mp.mpf(ref)
        ts = [a(p["e"] / self.s) for p in pts]
        gs = [a(p["y"] - self.ref) for p in pts]
        ws = [1 / a(p["sigma"]) for p in pts]
        N = len(pts)
        K = max(na, nb, nc) + 1
        tp = [[t ** k for k in range(K)] for t in ts]
        dB = [arb(1)] * N
        self.ok = False
        for _ in range(passes):
            rows, rhs = [], []
            for i in range(N):
                wi = ws[i] / dB[i]
                g = gs[i]
                rows.append([wi * g * g * tp[i][k] for k in range(na + 1)] +
                            [wi * g * tp[i][k] for k in range(1, nb + 1)] + [wi * tp[i][k] for k in range(nc + 1)])
                rhs.append(-wi * g)
            par = lstsq(rows, rhs)
            if par is None:
                return
            A = par[:na + 1]
            B = [arb(1)] + par[na + 1:na + 1 + nb]
            C = par[na + 1 + nb:]
            dB = [abs(2 * polyval(A, ts[i]) * gs[i] + polyval(B, ts[i])) for i in range(N)]
            if any(v < arb("1e-40") for v in dB):
                return
        self.A, self.B, self.C = [self._m(c) for c in (A, B, C)]
        res = []
        for i in range(N):
            gi = self.g(pts[i]["e"], near=pts[i]["y"] - self.ref)
            res.append(float((gi - (pts[i]["y"] - self.ref)) / mp.mpf(pts[i]["sigma"])))
        self.res = res
        self.chi2 = sum(v * v for v in res)
        self.dof = N - (na + 1 + nb + nc + 1)
        self.maxres = max(abs(v) for v in res)
        self.ok = all(math.isfinite(v) for v in res)

    @staticmethod
    def _m(c):
        return [mp.mpf(x.str(70, radius=False)) for x in c]

    def _ev(self, c, t):
        return mp.polyval(list(reversed(c)), t)

    def g(self, e, near=0):
        """the root of A g^2 + B g + C nearer `near` (both roots formed stably; complex where the discriminant is)"""
        t = mp.mpmathify(e) / self.s
        A, B, C = self._ev(self.A, t), self._ev(self.B, t), self._ev(self.C, t)
        d = mp.sqrt(B * B - 4 * A * C)
        if mp.re(mp.conj(B) * d) < 0:
            d = -d
        q = -(B + d) / 2
        roots = [C / q] + ([q / A] if A != 0 else [])
        r = min(roots, key=lambda v: abs(v - near))
        return mp.re(r) if isinstance(e, mp.mpf) and abs(mp.im(r)) == 0 else r

    def __call__(self, e, near=0):
        return self.g(e, near) + self.ref

    def disc_roots(self):
        A, B, C = self.A, self.B, self.C
        n = max(2 * len(B) - 1, len(A) + len(C) - 1)
        Dc = [mp.mpf(0)] * n
        for i, x in enumerate(B):
            for j, y in enumerate(B):
                Dc[i + j] += x * y
        for i, x in enumerate(A):
            for j, y in enumerate(C):
                Dc[i + j] -= 4 * x * y
        while len(Dc) > 1 and abs(Dc[-1]) < mp.mpf(10) ** -60 * max(abs(v) for v in Dc):
            Dc.pop()
        return self._rts(Dc)

    def a_roots(self):
        return self._rts(list(self.A))

    def _rts(self, c):
        if len(c) < 2:
            return []
        try:
            return [complex(t * self.s) for t in mp.polyroots(list(reversed(c)), maxsteps=500, extraprec=500)]
        except mp.libmp.libhyper.NoConvergence:
            return None

