"""Kantorovich quantities of the octo-double corrector's attempts, read from anm_run.py's log (#655,
notes/CORRECTOR_BASIN.md).

The corrector (octo_corrector.newton_od, printed by anm_run.py) logs per Newton step k the max norms |F_k| and
|dz_k| (leading part) at the iterate, the correction |dz_{k+1}| at the accepted point with that point's own Jacobian,
and the damping lam. From these the a-posteriori estimates of the affine covariant local Newton method (Deuflhard,
Newton Methods for Nonlinear Problems, Springer 2004, sec. 2.1.1) are

    theta_k = |dz_{k+1}| / |dz_k|,   [h_k] = 2 theta_k,   [omega]_k = [h_k] / |dz_k|,

from |dz_{k+1}| <= (omega/2) |dz_k|^2 (Newton-Mysovskikh). With lam < 1 the logged values are used as they are,
and the record says so (lam_k). They are lower bounds of the true h_k and omega, not enclosures.

    python3 kantorovich_log.py selftest [--mutate F] [--log OUT]
    python3 kantorovich_log.py parse LOG --sample NAME [--jsonl RECORDS] [--csv OUT] [--append 1] [--from_step N]
    python3 kantorovich_log.py analyse CSV --train NAMES [--held NAMES] [--h_star X] [--h_prior 1.0] [--mutate F]

selftest is the planted check: Newton histories of test functions with a known affine covariant Lipschitz constant
omega (a weak direction with a quadratic fold) and with a known order (a cubic weak direction), printed in
newton_od's and anm_run.py's log formats, parsed by the same parser and judged by the same estimator. It passes when
(P1) [omega]_0 from the first step is within a factor 2 of the planted omega for every start with h_0 <= 0.5,
(P2) the fitted exponent beta of theta_0 ~ |dz_0|^beta is within 0.2 of 1 (quadratic fold) and of 2 (cubic), and
(P3) the fit of |dz_0| against a (|dz_0|^2 = (k |z_N| a^p)^2 + f^2) recovers p within 1.5 and the floor f within a
factor 2. --mutate F takes theta from |F| instead of |dz| (in selftest and analyse): (P1) and (P2) must then fail,
and the selftest exits 1. --mutate nofloor fits |dz_0| without its floor: (P3) must then fail.

parse writes one CSV row per corrector attempt and, with --jsonl, checks the parsed a, accepted and step count
against anm_run.py's records (exit 1 on a mismatch). analyse runs the separation test of h_0 (threshold h_star from
the training samples by the rule of notes/CORRECTOR_BASIN.md 3 unless given), the fits, and the a-rule."""
import argparse
import csv
import json
import math
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# the parser

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf"
RE_STEPHDR = re.compile(rf"^step (\d+): D ({_NUM}), pivots ({_NUM})\.\.({_NUM}), \|z_1\| ({_NUM}), \|z_N\| ({_NUM}),"
                        rf".*a_max ({_NUM}), ")
RE_RESEED = re.compile(rf"^\s+a ({_NUM}): re-seeded \((ok|FAILED)")
RE_SCREEN = re.compile(rf"^\s+a ({_NUM}): predicted \|F\| ({_NUM}) above")
RE_ON = re.compile(rf"^\s+on (\d+): \|F\| ({_NUM}) -> ({_NUM}), \|dz\| ({_NUM}) -> ({_NUM}), lam ({_NUM}), "
                   rf"dDelta ({_NUM}), piv ({_NUM})\.\.({_NUM}), ({_NUM}) s")
RE_NOTRIAL = re.compile(rf"^\s+on (\d+): \|F\| ({_NUM}), \|dz\| ({_NUM}): no trial step accepted")
RE_CONV = re.compile(rf"^\s+on (\d+): \|F\| ({_NUM}) converged")
RE_STALL = re.compile(r"^\s+on (\d+): lam at or below .* \(stalled\)")
RE_JUNK = re.compile(rf"^\s+on (\d+): projected correction below .* converged at \|F\| ({_NUM})")
RE_TRIAL = re.compile(rf"^\s+on (\d+) trial lam ({_NUM}): max\|F\| ({_NUM}); simplified correction / \|dz\| = "
                      rf"({_NUM}) \(bound ({_NUM})\); (accepted|refused)")
RE_ODLINE = re.compile(r"^\s+octo-double Newton: (\d+) steps, .*?, (converged|not converged);")
RE_ATT = re.compile(rf"^\s+a ({_NUM}) -> D ({_NUM}): predicted \|F\| ({_NUM}), (\w+) (\d+) steps -> ({_NUM}) "
                    rf"\(tol ({_NUM})(, stall accepted)?.*?moved ({_NUM}), ({_NUM}) s")


def parse_log(text):
    """anm_run.py's log -> list of attempts, each a dict with the step header's values and its Newton records"""
    attempts, hdr, cur, idx = [], None, None, {}

    def new_attempt(a, reseed):
        k = idx.get(hdr["step"], 0)
        idx[hdr["step"]] = k + 1
        return dict(step=hdr["step"], attempt=k, D_base=hdr["D"], a=a, reseed=reseed, a_max=hdr["a_max"],
                    z1=hdr["z1"], zN=hdr["zN"], piv_lo=hdr["piv_lo"], piv_hi=hdr["piv_hi"], recs=[], trials=[],
                    stop=None, od_steps=None, od_conv=None)

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        m = RE_STEPHDR.match(line)
        if m:
            hdr = dict(step=int(m[1]), D=float(m[2]), piv_lo=float(m[3]), piv_hi=float(m[4]), z1=float(m[5]),
                       zN=float(m[6]), a_max=float(m[7]))
            cur = None
            continue
        if hdr is None:         # the base polish before the first step header is not an attempt
            continue
        m = RE_RESEED.match(line)
        if m:
            cur = new_attempt(float(m[1]), m[2])
            continue
        if RE_SCREEN.match(line):
            continue
        m = RE_ON.match(line)
        if m:
            if cur is None:
                cur = new_attempt(float("nan"), "none")
            cur["recs"].append(dict(it=int(m[1]), F=float(m[2]), F_new=float(m[3]), dz=float(m[4]),
                                    dz_new=float(m[5]), lam=float(m[6]), dDelta=float(m[7]), piv_lo=float(m[8]),
                                    piv_hi=float(m[9]), seconds=float(m[10])))
            continue
        m = RE_NOTRIAL.match(line)
        if m and cur is not None:
            cur["recs"].append(dict(it=int(m[1]), F=float(m[2]), dz=float(m[3]), lam=0.0))
            cur["stop"] = "no_trial"
            continue
        m = RE_CONV.match(line)
        if m and cur is not None:
            cur["F_conv"] = float(m[2])
            cur["stop"] = "converged"
            continue
        m = RE_JUNK.match(line)
        if m and cur is not None:
            cur["stop"] = "junk_floor"
            continue
        if RE_STALL.match(line) and cur is not None:
            cur["stop"] = "stalled"
            continue
        m = RE_TRIAL.match(line)
        if m and cur is not None:
            cur["trials"].append(dict(it=int(m[1]), lam=float(m[2]), F=float(m[3]), theta=float(m[4]),
                                      bound=float(m[5]), accepted=m[6] == "accepted"))
            continue
        m = RE_ODLINE.match(line)
        if m and cur is not None:
            cur["od_steps"], cur["od_conv"] = int(m[1]), m[2] == "converged"
            continue
        m = RE_ATT.match(line)
        if m and cur is not None:
            Fc, tol = float(m[6]), float(m[7])
            cur.update(a3=float(m[1]), D=float(m[2]), F_pred=float(m[3]), corrector=m[4], steps=int(m[5]),
                       F_final=Fc, tol=tol, accepted=bool(m[8]) or Fc <= tol, t_corr=float(m[10]))
            if cur["stop"] is None:
                cur["stop"] = "it_max"
            attempts.append(cur)
            cur = None
    return attempts


def check_records(atts, jsonl_text):
    """the parsed attempts against anm_run.py's records: a (to the log's 3 decimals), accepted and steps. Returns
    the list of mismatches (empty when they agree) and fills each attempt's exact a and D from the records"""
    rec = {}
    for ln in jsonl_text.splitlines():
        if ln.strip():
            r = json.loads(ln)
            for at in r.get("attempts", []):
                rec[(r["step"], at["attempt"])] = at
    bad = []
    for at in atts:
        r = rec.get((at["step"], at["attempt"]))
        if r is None:
            bad.append(f"step {at['step']} attempt {at['attempt']}: not in the records")
            continue
        if abs(r["a"] - at["a3"]) > 6e-4 or bool(r["accepted"]) != at["accepted"] or int(r["steps"]) != at["steps"]:
            bad.append(f"step {at['step']} attempt {at['attempt']}: log a {at['a3']} acc {at['accepted']} steps "
                       f"{at['steps']}, records a {r['a']} acc {r['accepted']} steps {r['steps']}")
        at["a"], at["D"] = float(r["a"]), float(r["D"])
    n_rec = len(rec)
    if n_rec != len(atts):
        bad.append(f"{len(atts)} attempts parsed, {n_rec} in the records")
    return bad


# ---------------------------------------------------------------------------------------------------------------
# the estimator

def estimates(at, mutate=""):
    """per step theta_k, [h_k], [omega]_k from the logged |dz| (|F| with mutate 'F'), lam as logged"""
    out = []
    for r in at["recs"]:
        if "dz_new" not in r:
            break
        if mutate == "F":
            th = r["F_new"] / r["F"] if r["F"] > 0 else float("inf")
        else:
            th = r["dz_new"] / r["dz"] if r["dz"] > 0 else float("inf")
        out.append(dict(it=r["it"], lam=r["lam"], theta=th, h=2 * th, omega=2 * th / r["dz"] if r["dz"] > 0 else float("nan")))
    return out


CSV_COLS = ["sample", "step", "attempt", "D_base", "a", "D", "a_max", "zN", "z1", "piv_lo", "piv_hi", "reseed",
            "F_pred", "steps", "stop", "accepted", "t_corr", "F0", "dz0", "lam0", "F1", "dz1", "theta0", "h0",
            "omega0", "theta1", "lam1", "theta_max_lam1", "fail_it", "thetas", "lams", "dzs", "Fs"]


def rows_of(atts, sample, mutate=""):
    rows = []
    for at in atts:
        e = estimates(at, mutate)
        rs = at["recs"]
        e0 = e[0] if e else None
        e1 = e[1] if len(e) > 1 else None
        full = [x["theta"] for x in e if x["lam"] == 1.0]
        rows.append(dict(
            sample=sample, step=at["step"], attempt=at["attempt"], D_base=at["D_base"], a=at["a"], D=at["D"],
            a_max=at["a_max"], zN=at["zN"], z1=at["z1"], piv_lo=at["piv_lo"], piv_hi=at["piv_hi"], reseed=at["reseed"],
            F_pred=at["F_pred"], steps=at["steps"], stop=at["stop"], accepted=int(at["accepted"]), t_corr=at["t_corr"],
            F0=rs[0]["F"] if rs else "", dz0=rs[0]["dz"] if rs else "", lam0=rs[0]["lam"] if rs else "",
            F1=rs[0].get("F_new", "") if rs else "", dz1=rs[0].get("dz_new", "") if rs else "",
            theta0=e0["theta"] if e0 else "", h0=e0["h"] if e0 else "", omega0=e0["omega"] if e0 else "",
            theta1=e1["theta"] if e1 else "", lam1=e1["lam"] if e1 else "",
            theta_max_lam1=max(full) if full else "",
            fail_it=(rs[-1]["it"] if rs and at["stop"] in ("no_trial", "stalled", "it_max") else ""),
            thetas=";".join(f"{x['theta']:.3g}" for x in e), lams=";".join(f"{r['lam']:g}" for r in rs),
            dzs=";".join(f"{r['dz']:.1e}" for r in rs), Fs=";".join(f"{r['F']:.2e}" for r in rs)))
    return rows


def write_csv(rows, path, append=False):
    import os
    new = not (append and os.path.exists(path))
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("D_base", "a", "D", "a_max", "zN", "z1", "piv_lo", "piv_hi", "F_pred", "t_corr", "F0", "dz0",
                      "lam0", "F1", "dz1", "theta0", "h0", "omega0", "theta1", "lam1", "theta_max_lam1"):
                r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
            for k in ("step", "attempt", "steps", "accepted"):
                r[k] = int(r[k])
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------
# fits

def fit_beta(dz0, theta0, D=None):
    """log theta_0 = log K + beta log |dz_0| (+ gamma (D - mean D)): least squares; (beta, its standard error from
    the residuals, K, gamma, rms residual in log10)"""
    x = np.log(np.asarray(dz0))
    y = np.log(np.asarray(theta0))
    cols = [np.ones_like(x), x]
    if D is not None:
        Dm = np.asarray(D) - np.mean(D)
        cols.append(Dm)
    A = np.stack(cols, 1)
    c, res, rk, sv = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    dof = max(len(y) - A.shape[1], 1)
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return dict(beta=float(c[1]), beta_se=float(math.sqrt(cov[1, 1])), logK=float(c[0]),
                gamma=float(c[2]) if D is not None else 0.0, gamma_se=float(math.sqrt(cov[2, 2])) if D is not None else 0.0,
                rms10=float(math.sqrt(s2)) / math.log(10), n=len(y), Dmean=float(np.mean(D)) if D is not None else 0.0)


def fit_dz0(a, zN, dz0, N=24, p_free=True, floor=True):
    """|dz_0|^2 = (k |z_N| a^p)^2 + f^2 in log space, least squares over (log k, log f[, p]) by a grid in p and
    Gauss-Newton in (log k, log f); p fixed at N+1 when p_free is False; floor False (--mutate nofloor, the
    control of (P3)): f = 0, a pure power law. Returns dict(p, k, f, rms10)"""
    a, zN, y = np.log(np.asarray(a)), np.log(np.asarray(zN)), np.log(np.asarray(dz0))
    if not floor:
        A = np.stack([np.ones_like(a), a], 1)
        c, *_ = np.linalg.lstsq(A, y - zN, rcond=None)
        r = y - zN - A @ c
        return dict(p=float(c[1]), k=math.exp(c[0]), f=0.0, rms10=math.sqrt(float(r @ r) / len(y)) / math.log(10), n=len(y))

    def solve(p):
        lk, lf = float(np.median(y - zN - p * a)), float(np.min(y))
        for _ in range(200):
            u = lk + zN + p * a
            m = np.logaddexp(2 * u, 2 * lf) / 2
            r = y - m
            wu = np.exp(2 * u - 2 * m)
            J = np.stack([wu, 1 - wu], 1)
            step, *_ = np.linalg.lstsq(J, r, rcond=None)
            lk, lf = lk + step[0], lf + step[1]
            if np.abs(step).max() < 1e-12:
                break
        u = lk + zN + p * a
        r = y - np.logaddexp(2 * u, 2 * lf) / 2
        return float(r @ r), lk, lf
    if p_free:
        ps = np.arange(1.0, 40.01, 0.05)
        best = min(((solve(p), p) for p in ps), key=lambda t: t[0][0])
        (ss, lk, lf), p = best
    else:
        p = N + 1.0
        ss, lk, lf = solve(p)
    return dict(p=float(p), k=math.exp(lk), f=math.exp(lf), rms10=math.sqrt(ss / max(len(y), 1)) / math.log(10), n=len(y))


def dz0_model(m, a, zN):
    return math.hypot(m["k"] * zN * a ** m["p"], m["f"])


# ---------------------------------------------------------------------------------------------------------------
# the planted check

def _gauss(Am, b):
    """mpmath Gaussian elimination with partial pivoting: (x, min |pivot|, max |pivot|)"""
    import mpmath as mp
    n = Am.rows
    M = [[Am[i, j] for j in range(n)] + [b[i]] for i in range(n)]
    piv = []
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[p] = M[p], M[k]
        piv.append(abs(M[k][k]))
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [mp.mpf(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x, min(piv), max(piv)


class Planted:
    """F(x) = A F0(x - x*), F0's weak row sigma y0 + (c/2) y0^2 (kind 'quad') or sigma y0 + (T/6) y0^3 ('cubic'),
    strong rows S_i (y_i + (kappa/2) y0^2). Newton corrections do not depend on A (affine covariance), |F| does.
    For 'quad' the affine covariant Lipschitz constant in the max norm is omega(z) = max(c, kappa sigma)/|sigma +
    c z0|, so omega = c/sigma on z0 >= 0, where the iterates stay (derivation in notes/CORRECTOR_BASIN.md 2)"""

    def __init__(self, kind="quad", sigma=1e-28, c=1.0, T=6.0, kappa=0.5, S=(1e30, 1e31, 1e32, 1e33, 1e34), seed=7):
        import mpmath as mp
        self.mp = mp
        self.kind, self.sigma, self.c, self.T, self.kappa = kind, mp.mpf(sigma), mp.mpf(c), mp.mpf(T), mp.mpf(kappa)
        self.S = [mp.mpf(s) for s in S]
        self.n = 1 + len(S)
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((self.n, self.n)))
        self.A = mp.matrix([[mp.mpf(float(Q[i, j])) * mp.mpf(10) ** int(rng.integers(-3, 4)) for j in range(self.n)]
                            for i in range(self.n)])
        self.xs = [mp.mpf(float(v)) for v in rng.uniform(0.5, 1.5, self.n)]
        self.omega = float(max(self.c, self.kappa * self.sigma) / self.sigma) if kind == "quad" else float("nan")

    def F0(self, y):
        w = self.sigma * y[0] + (self.c / 2 * y[0] ** 2 if self.kind == "quad" else self.T / 6 * y[0] ** 3)
        return [w] + [s * (y[i + 1] + self.kappa / 2 * y[0] ** 2) for i, s in enumerate(self.S)]

    def J0(self, y):
        mp = self.mp
        J = mp.matrix(self.n, self.n)
        J[0, 0] = self.sigma + (self.c * y[0] if self.kind == "quad" else self.T / 2 * y[0] ** 2)
        for i, s in enumerate(self.S):
            J[i + 1, 0] = s * self.kappa * y[0]
            J[i + 1, i + 1] = s
        return J

    def newton(self, x):
        """(dz, F, min pivot, max pivot) at x: dz = -J^-1 F of F = A F0, J = A J0"""
        mp = self.mp
        y = [xi - si for xi, si in zip(x, self.xs)]
        F = self.A * mp.matrix(self.F0(y))
        J = self.A * self.J0(y)
        d, plo, phi = _gauss(J, [-F[i] for i in range(self.n)])
        return d, [F[i] for i in range(self.n)], plo, phi

    def start(self, y0, rng):
        mp = self.mp
        y = [mp.mpf(y0)] + [-self.kappa / 2 * mp.mpf(y0) ** 2 * (1 + mp.mpf(float(rng.uniform(-1e-3, 1e-3))))
                            for _ in self.S]
        return [si + yi for si, yi in zip(self.xs, y)]


def planted_newton(P, x, tol=1e-60, it_max=12, max_halvings=8, lam_stall=0.04, out=print):
    """newton_od's iteration with its default ('own') test, printing its lines; returns (x, records, converged)"""
    mp = P.mp
    st = P.newton(x)
    hist, small = [], 0

    def nmax(v):
        return float(max(abs(t) for t in v))
    for it in range(it_max + 1):
        nf = nmax(st[1])
        if nf <= tol:
            out(f"  on {it}: |F| {nf:.2e} converged, Delta {float(x[0]):.17f} {0.0:+.4e}")
            hist.append(dict(it=it, F=nf, dz=nmax(st[0])))
            return x, hist, True
        if it == it_max:
            break
        d = st[0]
        nd = nmax(d)
        rec = dict(it=it, F=nf, dz=nd, dDelta=float(d[0]), piv=[float(st[2]), float(st[3])])
        lam = mp.mpf(1)
        for _ in range(max_halvings + 1):
            xt = [xi + lam * di for xi, di in zip(x, d)]
            stt = P.newton(xt)
            if nmax(stt[0]) < nd or nmax(stt[1]) <= tol:
                break
            lam /= 2
        else:
            hist.append(dict(rec, lam=0.0))
            out(f"  on {it}: |F| {nf:.2e}, |dz| {nd:.1e}: no trial step accepted, stopping")
            return x, hist, False
        x, st = xt, stt
        rec.update(lam=float(lam), F_new=nmax(st[1]), dz_new=nmax(st[0]), seconds=0.01)
        hist.append(rec)
        out(f"  on {it}: |F| {nf:.2e} -> {rec['F_new']:.2e}, |dz| {rec['dz']:.1e} -> {rec['dz_new']:.1e}, "
            f"lam {float(lam):g}, dDelta {rec['dDelta']:+.2e}, piv {float(st[2]):.1e}..{float(st[3]):.1e}, "
            f"{rec['seconds']} s")
        small = small + 1 if (lam_stall and lam <= lam_stall) else 0
        if small >= 2:
            out(f"  on {it}: lam at or below {lam_stall:g} twice in a row, stopping (stalled)")
            return x, hist, False
    return x, hist, False


def planted_log(P, starts, out):
    """one chain step per start (y0, a, zN, D): anm_run.py's step header, re-seed, newton_od and attempt lines"""
    import mpmath as mp
    rng = np.random.default_rng(11)
    truth = []
    for step, (y0, a, zN, D) in enumerate(starts):
        out(f"step {step}: D {mp.nstr(mp.mpf(D), 12)}, pivots 1.0e-28..1.0e+34, |z_1| 3.50e-04, |z_N| {zN:.2e}, "
            f"ratios 2-5 300.0 280.0 260.0 240.0, a_max 2.000, 1 s")
        out(f"    a {a:.4f}: re-seeded (ok), 0.1 s")
        x0 = P.start(y0, rng)
        d0 = P.newton(x0)[0]
        x, hist, conv = planted_newton(P, x0, out=out)
        out(f"      octo-double Newton: {len(hist)} steps, max|F| {hist[-1]['F']:.1e}, "
            f"{'converged' if conv else 'not converged'}; double-double |F| of its first two parts 1.0e+00")
        out(f"    a {a:.3f} -> D {mp.nstr(mp.mpf(D) + a, 12)}: predicted |F| 1.00e+00, od {len(hist)} steps -> "
            f"1.00e+00 (tol 1.0e-17{', stall accepted, Delta correction +0.0e+00' if conv else ''}), "
            f"Delta 1.000000000000 (formula diff +0.0e+00), moved 1.0e-30, 1 s")
        truth.append(dict(y0=y0, a=a, zN=zN, D=D, dz0=float(max(abs(t) for t in d0)), conv=conv))
    return truth


def selftest(mutate="", log=None):
    import io
    import mpmath as mp
    mp.mp.dps = 110
    buf = io.StringIO()

    def out(s):
        buf.write(s + "\n")
    ok_all = True
    lines = []

    def say(s):
        lines.append(s)
        print(s, flush=True)
    say(f"kantorovich_log selftest{' --mutate ' + mutate if mutate else ''}: planted Newton histories in mpmath "
        f"(dps {mp.mp.dps}), parsed from their printed log")
    # (P1) omega from the first step, quadratic fold, known omega = c/sigma
    P = Planted("quad")
    hs = [1e-6, 1e-4, 1e-2, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 100.0]
    D0 = 300.0
    jit = np.random.default_rng(17).uniform(-0.3, 0.3, len(hs))        # starts off round numbers (2-digit prints)
    starts = [(h * 10 ** j / P.omega, 1.0, 1e-25, D0 + i) for i, (h, j) in enumerate(zip(hs, jit))]
    truth = planted_log(P, starts, out)
    atts = parse_log(buf.getvalue())
    if len(atts) != len(starts):
        say(f"  FAIL: {len(atts)} attempts parsed from {len(starts)} planted")
        return False, lines, buf.getvalue()
    say(f"  (P1) quadratic fold, sigma {float(P.sigma):.0e}, c {float(P.c):g}: planted omega {P.omega:.3e} (max norm)")
    p1 = True
    for h, at, tr in zip(hs, atts, truth):
        e = estimates(at, mutate)[0]
        h_true = P.omega * tr["dz0"]
        r = e["omega"] / P.omega
        judged = h_true <= 0.5
        good = 0.5 <= r <= 2.0
        if judged:
            p1 &= good
        say(f"    h_0 = omega |dz_0| {h_true:.2e}: theta_0 {e['theta']:.3e}, [h_0] {e['h']:.3e}, [omega]_0 "
            f"{e['omega']:.3e}, ratio {r:.3g}{'' if judged else ' (not judged, h_0 > 0.5)'}"
            f"{'' if good or not judged else '  <- outside the factor 2'}")
    say(f"  (P1) {'PASS' if p1 else 'FAIL'}: [omega]_0 within a factor 2 of the planted omega for every h_0 <= 0.5")
    ok_all &= p1
    # (P2) the order of the first step's contraction: theta_0 ~ |dz_0|^beta, beta 1 (quadratic fold), 2 (cubic)
    p2 = True
    for kind, ys, beta_true in (("quad", [1e-34, 1e-33, 1e-32, 1e-31, 1e-30], 1.0),
                                ("cubic", [1e-18, 3e-18, 1e-17, 3e-17, 1e-16], 2.0)):
        P2 = Planted(kind, seed=3)
        buf2 = io.StringIO()
        rng = np.random.default_rng(5)
        Ds = list(rng.permutation(len(ys)) + 240.0)
        jit = rng.uniform(-0.3, 0.3, len(ys))
        planted_log(P2, [(y * 10 ** j, 1.0, 1e-25, float(D)) for y, j, D in zip(ys, jit, Ds)],
                    lambda s: buf2.write(s + "\n"))
        at2 = parse_log(buf2.getvalue())
        ee = [estimates(a_, mutate)[0] for a_ in at2]
        fb = fit_beta([a_["recs"][0]["dz"] for a_ in at2], [x["theta"] for x in ee], [a_["D_base"] for a_ in at2])
        good = abs(fb["beta"] - beta_true) <= 0.2
        p2 &= good
        say(f"  (P2) {kind}: planted order beta {beta_true:g}, fitted {fb['beta']:.3f} +- {fb['beta_se']:.3f} "
            f"(standard error), gamma {fb['gamma']:+.2e}: {'ok' if good else 'outside 0.2'}")
    say(f"  (P2) {'PASS' if p2 else 'FAIL'}: the fitted beta within 0.2 of the planted order for both")
    ok_all &= p2
    # (P3) |dz_0| against a: |dz_0|^2 = (k zN a^p)^2 + f^2 planted in the start, recovered from the parsed log
    P3 = Planted("quad", sigma=1e-20, seed=9)          # omega 1e20: every start in the quadratic regime
    k_t, p_t, f_t = 0.05, 25.0, 1e-35
    rng = np.random.default_rng(13)
    st3 = []
    for i in range(14):
        zN = float(10 ** rng.uniform(-29, -24))
        a = float(rng.choice([1.0, 0.977, 0.781, 0.625, 0.5, 0.39, 0.3]))
        y0 = math.hypot(k_t * zN * a ** p_t, f_t * float(rng.uniform(0.7, 1.4)))
        st3.append((y0, a, zN, 230.0 + i))
    buf3 = io.StringIO()
    planted_log(P3, st3, lambda s: buf3.write(s + "\n"))
    at3 = parse_log(buf3.getvalue())
    m3 = fit_dz0([x["a"] for x in at3], [x["zN"] for x in at3], [x["recs"][0]["dz"] for x in at3],
                 floor=mutate != "nofloor")
    p3 = abs(m3["p"] - p_t) <= 1.5 and 0.5 <= m3["f"] / f_t <= 2.0
    say(f"  (P3) planted p {p_t:g}, k {k_t:g}, floor {f_t:.0e} (x 0.7-1.4): fitted p {m3['p']:.2f}, k {m3['k']:.3g}, "
        f"floor {m3['f']:.2e}, rms {m3['rms10']:.2f} dex")
    say(f"  (P3) {'PASS' if p3 else 'FAIL'}: p within 1.5 and the floor within a factor 2")
    ok_all &= p3
    say(f"selftest {'PASS' if ok_all else 'FAIL'}")
    if log:
        with open(log, "w") as f:
            f.write("\n".join(lines) + "\n\n# the planted log of (P1)\n" + buf.getvalue())
    return ok_all, lines, buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------
# the analysis

def h_star_rule(train):
    """the training rule of notes/CORRECTOR_BASIN.md 3: the geometric mean of the largest accepted [h_0] and the
    smallest refused [h_0] above it (attempts whose first step has lam 1 and a finite theta_0)"""
    acc = [r["h0"] for r in train if r["accepted"] and math.isfinite(r["h0"])]
    ref = [r["h0"] for r in train if not r["accepted"] and math.isfinite(r["h0"])]
    hmax = max(acc)
    above = [h for h in ref if h > hmax]
    if not above:
        return float("nan"), hmax, float("nan")
    return math.sqrt(hmax * min(above)), hmax, min(above)


def separation(rows, h):
    tp = [r for r in rows if not r["accepted"] and r["h0"] >= h]
    fn = [r for r in rows if not r["accepted"] and not r["h0"] >= h]
    fp = [r for r in rows if r["accepted"] and r["h0"] >= h]
    tn = [r for r in rows if r["accepted"] and not r["h0"] >= h]
    return tp, fn, fp, tn


def a_star(h_star, rows_fit_dz, beta_fit, D, zN, N=24, h_max=1.0):
    """the largest a <= h_max whose predicted [h_0] = 2 K(D) |dz_0|(a)^beta is below h_star (bisection in log a)"""
    def h_pred(a):
        dz = dz0_model(rows_fit_dz, a, zN)
        try:
            return 2 * math.exp(beta_fit["logK"] + beta_fit["beta"] * math.log(dz) + beta_fit["gamma"] * (D - beta_fit["Dmean"]))
        except (OverflowError, ValueError):
            return float("inf")
    if h_pred(h_max) < h_star:
        return h_max, h_pred
    lo, hi = 1e-3, h_max
    if h_pred(lo) >= h_star:
        return lo, h_pred
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if h_pred(mid) < h_star else (lo, mid)
    return lo, h_pred


def analyse(rows, train, held, h_star=None, h_prior=1.0, mutate=""):
    if mutate == "F":
        for r in rows:
            r["theta0"] = r["F1"] / r["F0"]
            r["h0"] = 2 * r["theta0"]
            r["omega0"] = r["h0"] / r["dz0"]
    tr = [r for r in rows if r["sample"] in train]
    he = [r for r in rows if r["sample"] in held]
    print(f"training {sorted(set(train))}: {len(tr)} attempts, {sum(1 for r in tr if not r['accepted'])} refused")
    rule, hmax, hmin = h_star_rule(tr)
    print(f"  rule: largest accepted [h_0] {hmax:.3g}, smallest refused above it {hmin:.3g}, geometric mean {rule:.3g}")
    hs = h_star if h_star is not None else rule
    print(f"  threshold h_star {hs:.3g}{' (given)' if h_star is not None else ' (the rule)'}; a-priori threshold {h_prior:g}")

    def report(name, rs):
        for h, lab in ((hs, "h_star"), (h_prior, "a priori")):
            tp, fn, fp, tn = separation(rs, h)
            print(f"  {name}, {lab} {h:.3g}: refused caught {len(tp)}/{len(tp) + len(fn)}, false alarms "
                  f"{len(fp)}/{len(fp) + len(tn)}: {'separates' if not fn and not fp else 'does not separate'}")
            for r in tp + fn:
                print(f"      refused  D {r['D_base']:.3f} -> {r['D']:.3f}, a {r['a']:.4f}: [h_0] {r['h0']:.3g} "
                      f"(lam0 {r['lam0']:g}), stop {r['stop']} at step {r['fail_it']}, thetas {r['thetas']}, lams {r['lams']}")
            for r in fp:
                print(f"      false alarm  D {r['D_base']:.3f} -> {r['D']:.3f}, a {r['a']:.4f}: [h_0] {r['h0']:.3g}")
    report("training", tr)
    if he:
        print(f"held out {sorted(set(held))}: {len(he)} attempts, {sum(1 for r in he if not r['accepted'])} refused")
        report("held out", he)
    # descriptive numbers of both samples (not part of the held-out test)
    al = tr + he
    for smp in sorted(set(r["sample"] for r in al)):
        rs = [r for r in al if r["sample"] == smp]
        tref = sum(r["t_corr"] for r in rs if not r["accepted"])
        print(f"corrector seconds, {smp}: refused {tref:.0f} of {sum(r['t_corr'] for r in rs):.0f}; per refusal "
              + ", ".join(f"{r['t_corr']:.0f}" for r in rs if not r["accepted"]))
    thA, thA1 = [], []
    for r in al:
        if not r["accepted"] and r["h0"] >= hs:
            for k, (t, lm) in enumerate(zip(r["thetas"].split(";"), r["lams"].split(";"))):
                if lm == "1" and t:
                    thA.append(float(t))
                    if k > 0:
                        thA1.append(float(t))
    if thA:
        print(f"class A (refused, [h_0] >= {hs:.3g}): theta at lam 1, all steps: n {len(thA)}, {min(thA):.3f}-{max(thA):.3f}, "
              f"mean {np.mean(thA):.3f}; after the first step: n {len(thA1)}, mean "
              f"{np.mean(thA1) if thA1 else float('nan'):.3f} (1/2: first-order singular root, 2/3: order 2)")
    rise = [r for r in al if r["F1"] > r["F0"]]
    print(f"|F| rose in the first step of {len(rise)} of {len(al)} attempts (max ratio "
          f"{max(r['F1'] / r['F0'] for r in al):.2g}) while |dz| fell in {sum(1 for r in al if r['dz1'] < r['dz0'])}")
    damped = [r for r in al if r["lam1"] < 1]
    print(f"post hoc, a damped second step (lam_1 < 1): refused {sum(1 - r['accepted'] for r in damped)} of "
          f"{sum(1 - r['accepted'] for r in al)}, accepted {sum(r['accepted'] for r in damped)} of {sum(r['accepted'] for r in al)}")
    for lo in (225.0, 240.0):
        b = [r for r in tr if r["attempt"] == 0 and r["D_base"] >= lo]
        if len(b) > 2:
            sz = np.polyfit([r["D_base"] for r in b], np.log([r["zN"] for r in b]), 1)[0]
            print(f"ln |z_N| per unit D over the training bases from D {lo:g}: {sz:+.3f}")
    # fits on the training sample
    clean = [r for r in tr if r["lam0"] == 1.0 and 0 < r["theta0"] < 0.3]
    fb = fit_beta([r["dz0"] for r in clean], [r["theta0"] for r in clean], [r["D_base"] for r in clean])
    fb1 = fit_beta([r["dz0"] for r in clean], [r["theta0"] for r in clean])
    print(f"fit theta_0 = K |dz_0|^beta exp(gamma (D - {fb['Dmean']:.2f})) over {fb['n']} training attempts (lam0 1, "
          f"theta_0 < 0.3): beta {fb['beta']:.3f} +- {fb['beta_se']:.3f}, gamma {fb['gamma']:+.3f} +- "
          f"{fb['gamma_se']:.3f} per unit D, rms {fb['rms10']:.2f} dex (+-: standard errors of the regression)")
    print(f"  without the D term: beta {fb1['beta']:.3f} +- {fb1['beta_se']:.3f}, rms {fb1['rms10']:.2f} dex")
    # the base's smallest LU pivot in place of D: theta_0 ~ |dz_0|^beta piv_lo^delta (delta -1 if theta_0 ~ 1/sigma)
    y = np.log([r["theta0"] for r in clean])
    A = np.stack([np.ones(len(clean)), np.log([r["dz0"] for r in clean]), np.log([r["piv_lo"] for r in clean])], 1)
    cp, *_ = np.linalg.lstsq(A, y, rcond=None)
    rp = y - A @ cp
    covp = float(rp @ rp) / max(len(y) - 3, 1) * np.linalg.inv(A.T @ A)
    print(f"  with the base's smallest LU pivot instead of D: beta {cp[1]:.3f} +- {math.sqrt(covp[1, 1]):.3f}, "
          f"delta {cp[2]:+.3f} +- {math.sqrt(covp[2, 2]):.3f} (theta_0 ~ piv_lo^delta), rms "
          f"{math.sqrt(float(rp @ rp) / len(y)) / math.log(10):.2f} dex")
    Dh = np.array([r["D_base"] for r in tr if r["attempt"] == 0])
    ph = np.log([r["piv_lo"] for r in tr if r["attempt"] == 0])
    sl = np.polyfit(Dh, ph, 1)[0]
    print(f"  the base's smallest pivot falls as exp({sl:+.3f} D) over the training bases; gamma above is "
          f"{fb['gamma']:+.3f}")
    om = np.array([r["omega0"] for r in clean])
    Dv = np.array([r["D_base"] for r in clean])
    co = np.polyfit(Dv, np.log10(om), 1)
    res = np.log10(om) - np.polyval(co, Dv)
    cc = float(np.corrcoef(res, np.log10([r["dz0"] for r in clean]))[0, 1])
    print(f"fit log10 [omega]_0 = {co[1]:.2f} + {co[0]:.4f} D over the same attempts: rms {res.std():.2f} dex; the "
          f"residual's correlation with log |dz_0| {cc:+.2f} (0 if omega were one constant per D)")
    dzr = [r for r in tr if r["dz0"] > 0]
    m = fit_dz0([r["a"] for r in dzr], [r["zN"] for r in dzr], [r["dz0"] for r in dzr])
    mfix = fit_dz0([r["a"] for r in dzr], [r["zN"] for r in dzr], [r["dz0"] for r in dzr], p_free=False)
    print(f"fit |dz_0|^2 = (k |z_N| a^p)^2 + f^2 over {m['n']} training attempts: p {m['p']:.2f}, k {m['k']:.3g}, "
          f"floor {m['f']:.2e}, rms {m['rms10']:.2f} dex; with p = N + 1 = 25: k {mfix['k']:.3g}, floor "
          f"{mfix['f']:.2e}, rms {mfix['rms10']:.2f} dex")
    # the basin: the |dz_0| at which the predicted [h_0] reaches h_star, and where the floor alone reaches it
    def radius(f, D):
        try:
            return math.exp((math.log(hs / 2) - f["logK"] - f["gamma"] * (D - f["Dmean"])) / f["beta"])
        except (OverflowError, ZeroDivisionError):
            return float("nan")

    def d_floor(f):
        if not m["f"] > 0 or f["gamma"] == 0:
            return float("nan")
        return f["Dmean"] + (math.log(hs / 2) - f["logK"] - f["beta"] * math.log(m["f"])) / f["gamma"]

    def shrink(f):
        try:
            return math.exp(-f["gamma"] / f["beta"])
        except (OverflowError, ZeroDivisionError):
            return float("nan")
    print("basin radius, the |dz_0| whose predicted [h_0] is h_star: " + ", ".join(
        f"D {D}: {radius(fb, D):.2e}" for D in (230, 235, 240, 245, 250)))
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        c = [clean[i] for i in rng.integers(0, len(clean), len(clean))]
        f = fit_beta([r["dz0"] for r in c], [r["theta0"] for r in c], [r["D_base"] for r in c])
        bs.append((f["beta"], shrink(f), d_floor(f)))
    q = np.nanpercentile(np.array(bs), [2.5, 16, 50, 84, 97.5], axis=0)
    print(f"  per unit D the radius changes by {shrink(fb):.3f}; the floor {m['f']:.2e} reaches "
          f"h_star at D {d_floor(fb):.1f} (an extrapolation past the training D)")
    for j, name in enumerate(("beta", "radius change per unit D", "D where the floor reaches h_star")):
        print(f"  bootstrap over the {len(clean)} attempts (2000 resamples, seed 1), {name}: median {q[2, j]:.3f}, "
              f"16-84% {q[1, j]:.3f}-{q[3, j]:.3f}, 2.5-97.5% {q[0, j]:.3f}-{q[4, j]:.3f} (a spread, not a bound)")
    by_base = {}
    for r in tr + he:
        by_base.setdefault((r["sample"], r["step"]), []).append(r)
    pairs = []
    for (smp, st), rs in sorted(by_base.items()):
        if len(rs) >= 2:
            r1, r2 = rs[0], rs[1]
            p2 = math.log(r1["dz0"] / r2["dz0"]) / math.log(r1["a"] / r2["a"])
            b2 = (math.log(r1["theta0"] / r2["theta0"]) / math.log(r1["dz0"] / r2["dz0"])
                  if r1["theta0"] > 0 and r2["theta0"] > 0 else float("nan"))
            pairs.append((smp, r1["D_base"], r1["a"], r2["a"], p2, b2))
    print("two-point exponents per base (attempts a and a/2): " + "; ".join(
        f"{s} D {D:.2f} a {a1:.3f}/{a2:.3f}: p {p:.1f}, beta {b:.2f}" for s, D, a1, a2, p, b in pairs))
    # the a-rule against every attempt
    print(f"a-rule: the largest a <= 1 with predicted [h_0] < h_star {hs:.3g} (|dz_0| from the a-fit, theta_0 from "
          f"the beta fit with the D term)")
    out = []
    fo = dict(logK=co[1] * math.log(10) - math.log(2), beta=1.0, gamma=co[0] * math.log(10), Dmean=0.0)  # [omega](D)|dz|/2
    for r in tr + he:
        a_s, hp = a_star(hs, m, fb, r["D_base"], r["zN"])
        a_o, _ = a_star(hs, m, fo, r["D_base"], r["zN"])
        out.append((r, a_s, hp(r["a"]), a_o))
    print(f"  (a*_omega: the issue's form, one [omega](D) from the fit above, theta_0 = [omega](D) |dz_0| / 2)")
    for r, a_s, hpr, a_o in out:
        if not r["accepted"]:
            print(f"    refused {r['sample']} D {r['D_base']:.3f} -> {r['D']:.3f}: a taken {r['a']:.4f}, a* {a_s:.4f}, "
                  f"predicted [h_0] {hpr:.3g}, measured {r['h0']:.3g}: rule {'refuses it' if r['a'] > a_s else 'allows it'}"
                  f"; a*_omega {a_o:.4f} ({'refuses' if r['a'] > a_o else 'allows'})")
    for smp in sorted(set(r["sample"] for r, _, _, _ in out)):
        fa_o = [r for r, _, _, a_o in out if r["accepted"] and r["sample"] == smp and r["a"] > a_o]
        print(f"    accepted {smp}: a*_omega would have refused {len(fa_o)}")
    print("  a* per base (first attempt), D_base: a taken, a*, a*_omega, |z_N|, verdict")
    for r, a_s, _, a_o in out:
        if r["attempt"] == 0:
            print(f"    {r['sample']} {r['D_base']:.3f}: {r['a']:.4f} {a_s:.4f} {a_o:.4f} {r['zN']:.2e} "
                  f"{'accepted' if r['accepted'] else 'refused'}")
    out = [(r, a_s, hpr) for r, a_s, hpr, _ in out]
    for smp in sorted(set(r["sample"] for r, _, _ in out)):
        acc = [(r, a_s) for r, a_s, _ in out if r["accepted"] and r["sample"] == smp]
        fa = [(r, a_s) for r, a_s in acc if r["a"] > a_s]
        print(f"    accepted {smp}: {len(fa)} of {len(acc)} would have been refused by the rule"
              + ("".join(f"; D {r['D_base']:.2f} a {r['a']:.3f} > a* {a_s:.3f}" for r, a_s in fa[:8])))
    return dict(h_star=hs, rule=rule, beta=fb, dzfit=m)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--mutate", default="")
    s.add_argument("--log", default=None)
    p = sub.add_parser("parse")
    p.add_argument("log")
    p.add_argument("--sample", required=True)
    p.add_argument("--jsonl", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--append", type=int, default=0)
    p.add_argument("--from_step", type=int, default=0)
    q = sub.add_parser("analyse")
    q.add_argument("csv")
    q.add_argument("--train", required=True)
    q.add_argument("--held", default="")
    q.add_argument("--h_star", type=float, default=None)
    q.add_argument("--h_prior", type=float, default=1.0)
    q.add_argument("--mutate", default="")
    o = ap.parse_args(argv)
    if o.cmd == "selftest":
        ok, _, _ = selftest(o.mutate, o.log)
        return 0 if ok else 1
    if o.cmd == "parse":
        with open(o.log, encoding="utf-8", errors="replace") as f:
            atts = parse_log(f.read())
        bad = []
        if o.jsonl:
            with open(o.jsonl, encoding="utf-8", errors="replace") as f:
                bad = check_records(atts, f.read())
        else:
            for at in atts:
                at.setdefault("D", at["D_base"] + at["a3"])
                if not math.isfinite(at["a"]):
                    at["a"] = at["a3"]
        n_ref = sum(1 for a in atts if not a["accepted"])
        print(f"{o.log}: {len(atts)} attempts, {n_ref} refused; records check: "
              + ("skipped" if not o.jsonl else ("ok" if not bad else f"{len(bad)} MISMATCHES")))
        for b in bad:
            print("  " + b)
        for at in atts:
            n_rec = len(at["recs"]) + (1 if at["stop"] == "converged" else 0)
            if at["od_steps"] is not None and n_rec != at["od_steps"]:
                print(f"  step {at['step']} attempt {at['attempt']}: {n_rec} records parsed, the log says {at['od_steps']}")
                bad.append("steps")
        if o.csv:
            keep = [at for at in atts if at["step"] >= o.from_step]
            print(f"  {len(keep)} attempts from step {o.from_step} on written to {o.csv}")
            write_csv(rows_of(keep, o.sample), o.csv, append=bool(o.append))
        return 1 if bad else 0
    if o.cmd == "analyse":
        rows = read_csv(o.csv)
        analyse(rows, set(o.train.split(",")), set(x for x in o.held.split(",") if x), o.h_star, o.h_prior, o.mutate)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
