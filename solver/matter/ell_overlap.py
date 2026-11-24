#!/usr/bin/env python3
"""Branch-point test for the l = 1 DSS family (#1085): is the falling regular singular mode of the solver's Jacobian
J the lambda_1 eigenmode of the growth-mode matrix M(lambda)?

Relation (dss_ell.py). With n spectral unknowns, J = jacobian(z) is (n+1) x (n+1) and M(lambda) = mode_matrix(z, lambda)
is n x n, and
  (R1) J[:n, :n] = M(0)                                  (the same blocks, the same complex steps);
  (R2) J[n] = e_p, J[n, n] = 0                           (the phase row: Im of the centre datum's first harmonic);
  (R3) J[:n, n] = dF/dDelta = -(2 pi / Delta^2) M'(0) t  (omega enters only as omega d/dtheta, which lambda shifts);
  (R4) M(0) t = O(|F|)                                   (t = d/dtheta z, the translation mode).
For M(0) t = 0, the bordered determinant and Jacobi's formula give det J = (2 pi / Delta^2) (e_p . t) d/dlam det M(lam)
at lam = 0, which carries the factor (0 - lambda_1): J turns singular when a root of det M reaches the translation root.

Test, per state (the hypothesis H: the falling mode is the lambda_1 eigenmode):
  v1     the null vector of M(lambda_1): the eigenvector of the linearised pencil M(lam_r) v = -mu M'(lam_r) v with mu
         nearest 0 (shift-invert Arnoldi on -M^-1 M', as ell_eigs.nearest_mu; Ruhe 1973), at the scanned root lam_r;
         lambda_1 = lam_r + mu. Checks: (E1) lambda_1 inside the scan's bracket; (E2) |M(lam) v1| is V-shaped, its value
         at lambda_1 at most 0.05 of its values at lambda_1 -+ 0.005.
  P_e v1 = v1 - (e_p . v1 / e_p . t) t, which satisfies J's phase row.
  O      |c_m| / sqrt(1 - w_null), c = V^T (P_e v1, 0) / |.| over J's right singular vectors, c_m on the falling mode,
         w_null the weight on J's numerical null space (sigma < 1e-12 sigma_max): the cosine between the falling mode's
         vector and P_e v1 with the near-null directions removed.
  W_far  the junk-free weight of (P_e v1, 0) on J's right singular vectors with sigma >= 100 sigma_m (the lambda_1 mode
         far from J's near-null space).
  rho    |J (P_e v1, 0)| / |P_e v1|; C = |M'(lambda_1) v1| from the V (|v1| = 1). Information only: v1's junk content is
         arbitrary, and its rounding along J's large directions (1e-10 in the planted P1) floors rho near 1e-9.
  controls: the same O for J's other three small regular modes, and O with v1 replaced by y_w, the smallest regular
         right singular vector of M(lambda_1 -+ 0.05) (the wrong-lambda null direction).
  reading (written before the run, #1085; W_far replaced rho / sigma_m after the planted P1 failed on rho's floor, before
         any real overlap): consistent with H if O >= 0.9, W_far <= 0.01 and every control <= 0.3; refuted if O <= 0.3
         or W_far >= 0.5; otherwise no verdict.
A. Ruhe, Algorithms for the nonlinear eigenvalue problem, SIAM J. Numer. Anal. 10 (1973) 674.

usage: python ell_overlap.py run REC.jsonl STATE.npy D LAM_R LO HI --mode I [--wrong=-0.05,0.05] [--out F.jsonl]
       python ell_overlap.py planted [--out F.jsonl]
       python ell_overlap.py refit --records A.jsonl,B.jsonl --reads D1=read1.txt,... --start D:I --ranges lo:hi,...
  refit: the falling mode's sigma, followed from sv_reg_small[I] at D by mode_tracker.py's identity rule (the change of
  n_null), against lambda_1 from the readings' last "robust change" line; ln sigma = ln A + p ln lambda_1 per range.
  --mode I: the falling mode's index in the record's sv_reg_small (0 = the smallest regular value).
  run exits 0 when R1-R4, E1 and E2 pass and J's four smallest regular values match the record's (relative 1e-3), 1 if
  not; the reading is printed and recorded, not exit-coded. planted runs a synthetic pencil M(lam) = A - lam B (n 120:
  a translation null vector, lambda_1 0.8, three weak lambda-independent modes, ten junk directions near 1e-15, all in a
  random generalised Schur form, bordered as in R1-R3) through the same code: P1 (lambda_1 slope 1e-10) must read
  "consistent with H", P2 (slope 1) "refuted"; exit 0 only then. P1b (a diagnostic, no wanted reading) spaces J's weak
  values as at k_c 87, D 21.125 (0.43, 1.4, 1.89, 1.894 e-10), to show how near-equal singular vectors mix.
  --mutate=border scales J's Delta column by 1.001 (R3 must fail); --mutate=trans builds t with k^2 for k (R3, R4 must
  fail); --mutate=vec rolls v1 by one entry (E2 must fail). Each makes run and planted exit 1.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import scipy.linalg as sl
from scipy.sparse.linalg import LinearOperator, eigs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
CUT = 1e-12                                   # null cutoff relative to sigma_max, as ell_leap.py
TOL = dict(R1=1e-12, R3=1e-6, R4=1e-8, E2=0.05, sv=1e-3)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def unit(v):
    return v / np.linalg.norm(v)


# ---------------- the l = 1 system ----------------
def translation(S, z, power=1):
    """d/dtheta of the spectral unknowns (re, im of harmonic k -> -k im, k re); Delta not included"""
    parts, _ = S.split(z[None, :])
    out = []
    for c, par in zip(parts, S.parities()):
        c = c[0]
        if par == "odd":
            k = S.kodd.astype(float) ** power; m = len(k)
            out.append(np.concatenate([-k * c[m:], k * c[:m]]))
        else:
            k = S.keven[1:].astype(float) ** power; m = len(k)
            out.append(np.concatenate([[0.0], -k * c[1 + m:], k * c[1:1 + m]]))
    return np.concatenate(out)


def mode_apply(S, z, lam, v, h=1e-20):
    """M(lam) v by one complex step along v per block (mode_matrix's columns, contracted with v)"""
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    node0 = offs[2]
    out = np.zeros(len(z) - 1)
    S.lam = lam
    try:
        for b, (c0, nf, fun, tgt) in enumerate(blocks):
            X = np.stack([z[c0:c0 + nf], z[c0:c0 + nf] + 1j * h * v[c0:c0 + nf]]).astype(complex)
            om = np.full(2, 2 * np.pi / z[-1], dtype=complex)
            R = S.tospec(fun(X, om))[1].imag / h
            t0 = node0 + blk * tgt
            out[b * blk:(b + 1) * blk] = R - v[t0:t0 + blk]
    finally:
        S.lam = 0.0
    return out


class EllProblem:
    """the callables the test needs, on a DSSEll state"""

    def __init__(self, S, z):
        self.S, self.z = S, z
        self.n = len(z) - 1
        self.Delta = float(z[-1])
        self.ep = len(S.kodd)

    def Mfull(self, lam):
        with np.errstate(all="ignore"):
            return self.S.mode_matrix(self.z, lam)

    def Mapply(self, lam, v):
        with np.errstate(all="ignore"):
            return mode_apply(self.S, self.z, lam, v)

    def J(self):
        with np.errstate(all="ignore"):
            return self.S.jacobian(self.z)[1]

    def t(self):
        return translation(self.S, self.z, power=2 if "trans" in MUT else 1)


class PlantedProblem:
    """M(lam) = A - lam B = W (TA - lam TB) V^T, TA, TB upper triangular (generalised Schur form): column 0 the
    translation (TA[0, 0] = 0), 1 the lambda_1 mode (lam1 slope, slope), 2-4 weak modes with eigenvalues 3-5 (sizes
    0.4, 2, 3.5 e-10 unless weak= gives them), 5-14 junk near 1e-15 (eigenvalues 2-4), the rest O(1) with eigenvalues
    above 1.5; off-diagonal entries 0.3 randn min(d_i, d_j), d = |a| + |b| (a weak or junk column or row stays weak).
    Delta = sqrt(2 pi), so J[:n, n] = -M'(0) t = B t."""

    def __init__(self, slope, lam1=0.8, n=120, seed=1, weak=(0.4e-10, 2.0e-10, 3.5e-10)):
        rng = np.random.default_rng(seed)
        V = np.linalg.qr(rng.standard_normal((n, n)))[0]
        W = np.linalg.qr(rng.standard_normal((n, n)))[0]
        a, b = np.zeros(n), np.zeros(n)
        b[0] = 1.0
        a[1], b[1] = lam1 * slope, slope
        for j, (sz, ev) in enumerate(zip(weak, (3.0, 4.0, 5.0))):
            a[2 + j], b[2 + j] = sz, sz / ev
        u = 1e-15 * rng.uniform(0.5, 1.0, 10)
        a[5:15], b[5:15] = u, u / rng.uniform(2.0, 4.0, 10)
        a[15:] = rng.uniform(1.0, 10.0, n - 15)
        b[15:] = a[15:] / rng.uniform(1.5, 6.0, n - 15)
        d = np.abs(a) + np.abs(b)
        cpl = lambda: np.triu(0.3 * rng.standard_normal((n, n)) * np.minimum.outer(d, d), 1)
        TA, TB = np.diag(a) + cpl(), np.diag(b) + cpl()
        TA[:, 0] = 0.0                              # column 0: the translation, A t = 0 exactly
        TA[0, 0] = 0.0
        self.A, self.B = W @ TA @ V.T, W @ TB @ V.T
        self.n, self.Delta, self.ep = n, np.sqrt(2 * np.pi), int(np.argmax(np.abs(V[:, 0])))
        self._t = V[:, 0] * np.sign(V[self.ep, 0])
        self.lam1 = lam1

    def Mfull(self, lam):
        return self.A - lam * self.B

    def Mapply(self, lam, v):
        return (self.A - lam * self.B) @ v

    def J(self):
        n = self.n
        J = np.zeros((n + 1, n + 1))
        J[:n, :n] = self.A
        J[:n, n] = self.B @ self._t                  # -(2 pi / Delta^2) M'(0) t with M' = -B
        J[n, self.ep] = 1.0
        return J

    def t(self):
        if "trans" in MUT:
            return self._t + 0.01 * np.roll(self._t, 1)
        return self._t.copy()


# ---------------- the test ----------------
def dM(P, lam, v, h=1e-3):
    """M'(lam) v, 4-point central differences"""
    return (8 * (P.Mapply(lam + h, v) - P.Mapply(lam - h, v)) - (P.Mapply(lam + 2 * h, v) - P.Mapply(lam - 2 * h, v))) / (12 * h)


def relation(P, J, t, rng):
    n = P.n
    r = {}
    Jn = np.linalg.norm(J, 2) if n < 2000 else float(np.max(np.linalg.norm(J, axis=0)))   # |J|, or a column bound
    x = rng.standard_normal((3, n))
    r["R1"] = max(float(np.linalg.norm(P.Mapply(0.0, xi) - J[:n, :n] @ xi) / np.linalg.norm(J[:n, :n] @ xi)) for xi in x)
    row = np.zeros(n + 1); row[P.ep] = 1.0
    r["R2"] = bool(np.array_equal(J[n], row))
    b = J[:n, n] * (1.001 if "border" in MUT else 1.0)
    pred = -(2 * np.pi / P.Delta ** 2) * dM(P, 0.0, t)
    r["R3"] = float(np.linalg.norm(pred - b) / np.linalg.norm(b))
    r["R4"] = float(np.linalg.norm(P.Mapply(0.0, t)) / (Jn * np.linalg.norm(t)))
    r["ep_t"] = float(t[P.ep] / np.linalg.norm(t))
    r["ok"] = r["R1"] <= TOL["R1"] and r["R2"] and r["R3"] <= TOL["R3"] and r["R4"] <= TOL["R4"]
    return r


def pencil_vec(P, lam_r, k=6, dl=1e-3):
    """eigenpair of M(lam_r) v = -mu M'(lam_r) v with mu nearest 0 (shift-invert Arnoldi on -M^-1 M')"""
    lu = sl.lu_factor(P.Mfull(lam_r), overwrite_a=True)
    Mp = lambda x: (P.Mapply(lam_r + dl, x) - P.Mapply(lam_r - dl, x)) / (2 * dl)
    op = LinearOperator((P.n, P.n), matvec=lambda x: -sl.lu_solve(lu, Mp(np.asarray(x).ravel())), dtype=float)
    nu, X = eigs(op, k=k, which="LM", tol=1e-12, ncv=min(P.n - 1, 40), v0=np.ones(P.n) / np.sqrt(P.n))
    del lu
    mu = 1 / nu
    i = int(np.argmin(np.abs(mu)))
    v = X[:, i]
    v = v / (v[np.argmax(np.abs(v))] / np.abs(v[np.argmax(np.abs(v))]))      # a real eigenvector up to phase
    imag = float(np.linalg.norm(v.imag) / np.linalg.norm(v))
    return float(mu[i].real), float(abs(mu[i].imag)), unit(v.real), imag, sorted(float(abs(m)) for m in mu)


def decompose(Vt, w, null):
    c = Vt @ w
    wn = float(np.sum(c[null] ** 2))
    return c, wn


def analyze(P, lam_r, lo, hi, mode, wrong=(-0.05, 0.05), rec_sv=None, verbose=True):
    rng = np.random.default_rng(7)
    out = {}
    say = log if verbose else (lambda m: None)
    n = P.n
    t = P.t()
    J = P.J()
    say(f"J built ({n + 1} x {n + 1}); peak RSS {rss_gb():.2f} GB")
    rel = relation(P, J, t, rng)
    out["relation"] = rel
    say("relation: R1 |M(0)x - J[:n,:n]x|/|J x| %.2e (tol %.0e), R2 phase row %s, R3 |pred b - b|/|b| %.2e (tol %.0e), "
        "R4 |M(0)t|/(|J||t|) %.2e (tol %.0e), e_p.t/|t| %.3e: %s" % (rel["R1"], TOL["R1"], rel["R2"], rel["R3"], TOL["R3"],
                                                                     rel["R4"], TOL["R4"], rel["ep_t"], "OK" if rel["ok"] else "BAD"))
    # J's SVD
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    del U
    say(f"J SVD done; peak RSS {rss_gb():.2f} GB")
    reg = np.nonzero(s > CUT * s[0])[0]
    null = np.nonzero(s <= CUT * s[0])[0]
    small = reg[-4:][::-1]
    out.update(sv_max=float(s[0]), n_null=int(len(null)), sv_reg_small=s[small].tolist())
    g = int(small[mode])
    sig_m = float(s[g])
    sv_ok = True
    if rec_sv is not None:
        dev = [abs(a / b - 1) for a, b in zip(s[small], rec_sv)]
        sv_ok = max(dev) <= TOL["sv"]
        out["sv_rec_dev"] = dev
        say("J's four smallest regular values %s; the record's %s; max relative deviation %.1e (tol %.0e): %s"
            % (", ".join(f"{x:.4e}" for x in s[small]), ", ".join(f"{x:.4e}" for x in rec_sv), max(dev), TOL["sv"],
               "OK" if sv_ok else "BAD"))
    out.update(mode=mode, sigma_m=sig_m, delta_comp_m=float(abs(Vt[g, n])))
    say(f"falling mode: sv_reg_small[{mode}] = {sig_m:.4e} (n_null {len(null)}, sigma_max {s[0]:.4f}); "
        f"its Delta component {abs(Vt[g, n]):.2e}")
    # lambda_1 eigenvector
    mu, mu_im, v1, imag, mus = pencil_vec(P, lam_r)
    lam1 = lam_r + mu
    if "vec" in MUT:
        v1 = np.roll(v1, 1)
    e1 = lo <= lam1 <= hi
    rV = {d: float(np.linalg.norm(P.Mapply(lam1 + d, v1))) for d in (-0.005, 0.0, 0.005)}
    e2 = rV[0.0] <= TOL["E2"] * min(rV[-0.005], rV[0.005])
    C = 0.5 * (rV[-0.005] + rV[0.005]) / 0.005
    out.update(lam_r=lam_r, mu=mu, mu_imag=mu_im, lam1=lam1, v1_imag=imag, mus_abs=mus[:6], E1=e1, V=list(rV.values()),
               E2=e2, C=C)
    say(f"pencil at lam_r {lam_r}: mu {mu:+.3e} (|Im| {mu_im:.1e}; next |mu| {mus[1]:.3e}), lambda_1 {lam1:.6f} "
        f"(bracket [{lo}, {hi}]: {'OK' if e1 else 'BAD'}); eigenvector Im/Re {imag:.1e}")
    say(f"V: |M(lam) v1| at lambda_1 -0.005, 0, +0.005: {rV[-0.005]:.3e}, {rV[0.0]:.3e}, {rV[0.005]:.3e} "
        f"(E2: middle <= {TOL['E2']} x min: {'OK' if e2 else 'BAD'}); slope C {C:.3e}")
    # overlaps
    ept = t[P.ep]

    def stat(v, Jm=None):
        pe = v - (v[P.ep] / ept) * t
        w = unit(np.concatenate([pe, [0.0]]))
        c, wn = decompose(Vt, w, null)
        rest = max(1e-300, 1 - wn)
        O = [float(abs(c[i]) / np.sqrt(rest)) for i in small]
        far = float(np.sum(c[s >= 100 * sig_m] ** 2) / rest)        # junk-free weight on sigma >= 100 sigma_m
        rho = float(np.linalg.norm(Jm @ w)) if Jm is not None else None
        cos_raw = float(abs(Vt[g, :n] @ unit(pe)) / np.linalg.norm(Vt[g, :n]))
        return dict(O=O, w_null=wn, W_far=far, rho=rho, cos_raw=cos_raw)
    st = stat(v1, J)
    O_m = st["O"][mode]
    ctrl = [st["O"][i] for i in range(4) if i != mode]
    out.update(O=st["O"], O_m=O_m, w_null=st["w_null"], W_far=st["W_far"], rho=st["rho"], rho_ratio=st["rho"] / sig_m,
               lamC_ratio=lam1 * C / sig_m, cos_raw=st["cos_raw"])
    say("O on J's four smallest regular modes (m%d is the falling one): %s; weight on J's null space %.3f; "
        "raw cos(x_m, P_e v1) %.3f" % (mode, ", ".join(f"m{i} {o:.3f}" for i, o in enumerate(st["O"])), st["w_null"], st["cos_raw"]))
    say(f"W_far (junk-free weight on sigma >= 100 sigma_m) {st['W_far']:.3e}; rho |J (P_e v1, 0)| {st['rho']:.3e}, "
        f"rho / sigma_m {st['rho'] / sig_m:.3g}; lambda_1 C / sigma_m {lam1 * C / sig_m:.3g} (rho and C carry the eigenvector's "
        f"rounding in J's large directions; information only)")
    del J
    # wrong-lambda controls
    wr = []
    for dw in wrong:
        lw = lam1 + dw
        Mw = P.Mfull(lw)
        _, sw, Vw = sl.svd(Mw, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver="gesdd")
        del Mw
        regw = np.nonzero(sw > CUT * sw[0])[0]
        y = Vw[regw[-1]].copy()
        del Vw
        sy = stat(y)
        wr.append(dict(lam=lw, sigma=float(sw[regw[-1]]), n_null=int(len(sw) - len(regw)), O_m=sy["O"][mode], O=sy["O"],
                       cos_v1=float(abs(y @ v1))))
        say(f"wrong lambda {lw:.6f}: smallest regular sigma of M {sw[regw[-1]]:.3e} (n_null {len(sw) - len(regw)}); "
            f"O on the falling mode {sy['O'][mode]:.3f} (all four: {', '.join(f'{o:.3f}' for o in sy['O'])}); "
            f"|cos(y_w, v1)| {abs(y @ v1):.3f}; peak RSS {rss_gb():.2f} GB")
    out["wrong"] = wr
    ctrl_all = ctrl + [w["O_m"] for w in wr]
    if O_m >= 0.9 and out["W_far"] <= 0.01 and max(ctrl_all) <= 0.3:
        reading = "consistent with H"
    elif O_m <= 0.3 or out["W_far"] >= 0.5:
        reading = "refuted"
    else:
        reading = "no verdict"
    out.update(controls_max=float(max(ctrl_all)), reading=reading, checks_ok=bool(rel["ok"] and e1 and e2 and sv_ok),
               peak_rss_gb=rss_gb())
    say(f"reading: {reading} (O {O_m:.3f}, largest control {max(ctrl_all):.3f}, W_far {out['W_far']:.3g}); "
        f"checks {'OK' if out['checks_ok'] else 'BAD'}" + (f"  [MUTATION {MUT}]" if MUT else ""))
    return out


def link(nold, nnew):
    """mode_tracker.py's identity rule: old index of each of the 4 new sv_reg_small entries (None = new)"""
    dn = nnew - nold
    return {0: [0, 1, 2, 3], -1: [None, 0, 1, 2], 1: [1, 2, 3, None]}.get(dn, [None] * 4)


def refit(records, reads, start_D, start_index, ranges):
    """sigma of the tracked falling mode against lambda_1: ln sigma = ln A + p ln lambda_1 by least squares"""
    recs = []
    for f in records:
        recs += [json.loads(l) for l in open(f) if l.strip()]
    recs = sorted({r["D"]: r for r in recs}.values(), key=lambda r: r["D"])
    lam = {}
    for D, path in reads:
        line = [l for l in open(path) if "robust change (" in l][-1]
        lam[D] = float(line.split("det-linear")[1].split(",")[0])
    track, i, prev = {}, None, None
    for r in recs:
        if r["D"] < start_D - 1e-12:
            continue
        if prev is None:
            i = start_index
        else:
            m = link(prev["n_null"], r["n_null"])
            i = m.index(i) if i in m else None
            if i is None:
                break
        track[r["D"]] = (r["sv_reg_small"][i], i, r["n_null"])
        prev = r
    rows = [(D, track[D][0], lam[D], track[D][1]) for D in sorted(track) if D in lam]
    for D, s, l, i in rows:
        print(f"  D {D:<9g} sigma {s:.4e} (sv_reg_small[{i}])  lambda_1 {l:.6f}")
    out = []
    for lo, hi in ranges:
        sel = [(s, l) for D, s, l, _ in rows if lo - 1e-12 <= D <= hi + 1e-12]
        if len(sel) < 3:
            continue
        y, x = np.log([s for s, _ in sel]), np.log([l for _, l in sel])
        p, lnA = np.polyfit(x, y, 1)
        res = y - (lnA + p * x)
        rms = float(np.sqrt(np.mean(res ** 2)))
        out.append(dict(lo=lo, hi=hi, n=len(sel), p=float(p), A=float(np.exp(lnA)), rms=rms))
        print(f"  D {lo:g}-{hi:g}, {len(sel)} points: sigma = {np.exp(lnA):.4e} lambda_1^{p:.3f}, rms {100 * rms:.2f}% "
              f"(max |residual| {100 * np.max(np.abs(res)):.2f}%)")
    loc = [(rows[k][0], rows[k + 1][0], float(np.log(rows[k + 1][1] / rows[k][1]) / np.log(rows[k + 1][2] / rows[k][2])))
           for k in range(len(rows) - 1)]
    print("  local exponents d ln sigma / d ln lambda_1: " + ", ".join(f"{a:g}-{b:g} {e:.2f}" for a, b, e in loc))
    # extrapolations, not results: lambda_1 = 0 from the last two readings (line) and the last three (parabola)
    Ds, ls = np.array(sorted(lam)), np.array([lam[D] for D in sorted(lam)])
    zl = float(Ds[-1] - ls[-1] * (Ds[-1] - Ds[-2]) / (ls[-1] - ls[-2]))
    q = np.polyfit(Ds[-3:], ls[-3:], 2)
    zq = [float(r.real) for r in np.roots(q) if abs(r.imag) < 1e-12 and r.real > Ds[-1]]
    zq = min(zq) if zq else None
    print(f"  lambda_1 = 0 extrapolated (not a result): line through D {Ds[-2]:g}, {Ds[-1]:g}: D {zl:.3f}; parabola through "
          f"the last three: D {zq:.3f}" if zq else f"  lambda_1 = 0 extrapolated: line D {zl:.3f}; parabola: no zero past the data")
    return dict(rows=rows, fits=out, local=loc, zero_line=zl, zero_parabola=zq)


def write(path, rec):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "planted", "refit"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--wrong", default="-0.05,0.05")
    ap.add_argument("--out", default="")
    ap.add_argument("--mutate", default="")
    ap.add_argument("--records", default=""); ap.add_argument("--reads", default="")
    ap.add_argument("--start", default=""); ap.add_argument("--ranges", default="")
    a = ap.parse_args()
    wrong = [float(w) for w in a.wrong.split(",") if w]
    if a.cmd == "refit":
        reads = [(float(x.split("=", 1)[0]), x.split("=", 1)[1]) for x in a.reads.split(",") if x]
        sD, si = a.start.split(":")
        ranges = [tuple(float(v) for v in r.split(":")) for r in a.ranges.split(",") if r]
        r = refit(a.records.split(","), reads, float(sD), int(si), ranges)
        write(a.out, dict(cmd="refit", records=[os.path.basename(f) for f in a.records.split(",")], **r))
        sys.exit(0)
    if a.cmd == "planted":
        ok = True
        cases = (("P1", 1e-10, None, "consistent with H"), ("P2", 1.0, None, "refuted"),
                 ("P1b", 1.75e-10, (0.43e-10, 1.89e-10, 1.894e-10), None))
        for name, slope, weak, want in cases:
            P = PlantedProblem(slope, weak=weak) if weak else PlantedProblem(slope)
            r = analyze(P, 0.8 + 4e-4, 0.795, 0.805, mode=1, wrong=wrong, verbose=False)
            # lambda_1 of a weak mode is conditioned like eps |M| / slope (1e-5 at slope 1e-10): bar 1e-4
            good = r["checks_ok"] and abs(r["lam1"] - P.lam1) <= 1e-4 and (want is None or r["reading"] == want)
            ok &= good
            log(f"planted {name} (slope {slope:g}): lambda_1 {r['lam1']:.9f} (planted 0.8), O {r['O_m']:.3f}, controls max "
                f"{r['controls_max']:.3f}, W_far {r['W_far']:.3g}, rho / sigma_m {r['rho_ratio']:.3g}, checks "
                f"{'OK' if r['checks_ok'] else 'BAD'}: "
                f"reads '{r['reading']}', wanted '{want or 'any (diagnostic)'}': {'PASS' if good else 'FAIL'}; "
                f"O on m0-m3 {', '.join(f'{o:.3f}' for o in r['O'])}" + (f"  [MUTATION {MUT}]" if MUT else ""))
            write(a.out, dict(case=name, slope=slope, want=want, mutate=MUT, **{k: v for k, v in r.items()}))
        log("planted: " + ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
    recf, st, D, lam_r, lo, hi = a.args[0], a.args[1], float(a.args[2]), float(a.args[3]), float(a.args[4]), float(a.args[5])
    from dss_ell import DSSEll, NODES_STD
    rec = [json.loads(l) for l in open(recf) if json.loads(l).get("D") == D][-1]
    z = np.load(st)
    S = DSSEll(kc=rec["kc"], nodes=NODES_STD, n=D - 2, ell=rec["l"], Delta_ref=z[-1], Ks=rec["Ks"], u0=rec["u0"],
               rk=rec.get("rk", 1))
    steps = S.freeze_from(z, rec["kappa"])
    with np.errstate(all="ignore"):
        F = float(np.linalg.norm(S.residual(z[None, :])[0]))
    same = abs(F - rec["F"]) <= max(1e-2 * rec["F"], 1e-13) and sum(steps.values()) == rec["steps"]
    log(f"D {D} k_c {rec['kc']}: |F| {F:.10e} (record {rec['F']:.10e}), steps {sum(steps.values())} (record "
        f"{rec['steps']}): {'the record system' if same else 'NOT the record system'}; state {os.path.basename(st)}")
    if not same:
        sys.exit(3)
    r = analyze(EllProblem(S, z), lam_r, lo, hi, a.mode, wrong=wrong, rec_sv=rec["sv_reg_small"])
    r.update(D=D, kc=rec["kc"], state=os.path.basename(st), record=os.path.basename(recf), F=F, mutate=MUT,
             sv_reg_small_rec=rec["sv_reg_small"], seconds=round(time.time() - T0))
    write(a.out, r)
    sys.exit(0 if r["checks_ok"] else 1)


if __name__ == "__main__":
    main()
