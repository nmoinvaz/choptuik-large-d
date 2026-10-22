"""First-order coefficients omega_1 and lambda_1 of the M-harmonic inner model, with deflated (two-pass) jets, for
M >= 5 where the one-shot linear algebra of first_order_run.py is beyond double precision (notes/FIRST_ORDER.md).

Why. The de-mixed residual r = (r_1, r_3, ..., r_M) at xi1 is block triangular with enormous couplings: harmonic
nu's multiplier leaks into the lower rows with weight ~e^{pi (nu - 1) w / 2} times the finite-xi de-mixing error
(M = 5: dr_1/dB5 ~ 1e17 against dr_5/dB5 ~ 4e8). A Jacobian column whose own r_H (harmonics >= 3) is large carries
an absolute error rtol |r_H| there, which the Schur correction Lambda = K_lH K_HH^-1 (~2.5e8) multiplies. The one-shot
Jacobian at M = 5 has condition ~4e19 and Newton diverges.

Deflation. Every "outer" direction d (A, omega, the 1/n direction n, the harmonic-1 perturbation profiles, the mixed
component e|n) gets inner directions added (higher-harmonic profile or parameter changes, coefficients
c = -K_HH^-1 r_H(d) from a first integration, block back-substitution from the top harmonic down), so that in a
second integration its own r_H is ~0. Its harmonic-1 jump then has no large leakage, and the remaining Schur
correction Lambda r_H(d') is small. Results are the M-model's exact first-order coefficients, as in
first_order_run.py, only computed in a better-conditioned way (for M = 1, 3 the two drivers must agree).

Steps: zero (deflated Newton from hier.py's M = 5 zero), omega1, lam* (Newton/secant on det Sch(lam)/(lam(lam-1)),
Sch the 2 x 2 Schur complement of the perturbation map), lam1 = -(ell_l . J_en^corr)/(ell_l Sch'(lam*) e*) with
Sch' by centred differences.

    python3 -u first_order_hier.py M [--xi1=0.2] [--rtol=1e-12] [--x=...] [--lam=2.0] [--redo=zero,omega1,lam,lam1]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import first_order_run as R  # noqa: E402
from first_order_jet import JetModel, Dir, Fh_from, dFh, W0, F2h  # noqa: E402

T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def hsolve(K, b):
    """solve K x = b (K: 2m x 2m over harmonics 3, 5, ..., in pairs) by eliminating the highest harmonic first"""
    n = K.shape[0]
    if n == 0:
        return np.zeros_like(b)
    if n == 2:
        return np.linalg.solve(K, b)
    a, z = slice(0, n - 2), slice(n - 2, n)
    X = np.linalg.solve(K[z, z], K[z, a])
    y = np.linalg.solve(K[z, z], b[z])
    xa = hsolve(K[a, a] - K[a, z] @ X, b[a] - K[a, z] @ y)
    return np.concatenate([xa, y - X @ xa])


def hsolve_mat(K, B):
    return np.column_stack([hsolve(K, B[:, i]) for i in range(B.shape[1])]) if B.ndim == 2 else hsolve(K, B)


class Lin:
    """pass-1 block structure of a square Jacobian K (rows r_1, r_H; columns outer (2), inner (2N-2))"""

    def __init__(self, K):
        self.K = K
        self.Kll, self.KlH, self.KHl, self.KHH = K[:2, :2], K[:2, 2:], K[2:, :2], K[2:, 2:]

    def deflate(self, rH):
        """inner coefficients c with K_HH c = -rH"""
        return -hsolve(self.KHH, rH)

    def corr(self, r):
        """Schur-corrected harmonic-1 rows: r_l - K_lH K_HH^-1 r_H"""
        if r.shape[0] == 2:
            return r.copy()
        return r[:2] - self.KlH @ hsolve(self.KHH, r[2:])


def run_model(x, M, xi1, dirs, masks, data):
    jm = JetModel(M, W0 * x[1], Fh_from(x, M), dirs, masks, data)
    Rr, info = R.residuals(jm, x, M, xi1)
    return Rr, jm


def param_dirs(M, coefs):
    """one jet direction per given coefficient vector over the parameters x (profile part -> d0, omega part)"""
    dirs, masks, data = [], [0], {}
    for i, c in enumerate(coefs):
        prof = {}
        for p in range(R.npar(M)):
            if p == 1 or c[p] == 0:
                continue
            for k, v in dFh(p, M).items():
                prof[k] = prof.get(k, 0) + c[p] * v
        dirs.append(Dir(f"d{i}", omega=W0 * c[1]))
        masks.append(1 << i)
        data[1 << i] = (prof, {})
    return dirs, masks, data


def jacobian(x, M, xi1, extra=None):
    """pass 1: value + every parameter direction (+ extra Dir with data, appended last); returns r, K, extra's r"""
    n = R.npar(M)
    dirs, masks, data = param_dirs(M, list(np.eye(n)))
    if extra is not None:
        d, dat = extra
        dirs.append(d)
        masks.append(1 << n)
        data[1 << n] = dat
    Rr, _ = run_model(x, M, xi1, dirs, masks, data)
    return Rr[0], Rr[1:1 + n].T, (Rr[1 + n] if extra is not None else None)


CKPT = None


def find_zero(x, M, xi1, maxit=8, tol=3e-15, floor=1e-9):
    """deflated Newton; checkpoints x after every iteration (CKPT, resumed on restart) and stops after two outer
    steps below `floor` (the noise floor of the zero at M = 5 is ~1e-10)"""
    n = R.npar(M)
    if CKPT and os.path.exists(CKPT):
        x = np.array(json.load(open(CKPT))["x"])
        log(f"resuming the zero from the checkpoint {CKPT}")
    small = 0
    for it in range(maxit):
        r, K, _ = jacobian(x, M, xi1)
        L = Lin(K)
        # inner fix at fixed (A, omega), then deflated outer directions
        x = x.copy()
        x[2:] += L.deflate(r[2:])
        cA = np.zeros(n); cA[0] = 1; cA[2:] = L.deflate(K[2:, 0])
        cw = np.zeros(n); cw[1] = 1; cw[2:] = L.deflate(K[2:, 1])
        dirs, masks, data = param_dirs(M, [cA, cw])
        Rr, _ = run_model(x, M, xi1, dirs, masks, data)
        r2 = Rr[0]
        Sch = np.column_stack([L.corr(Rr[1]), L.corr(Rr[2])])
        r1c = L.corr(r2)
        st = -np.linalg.solve(Sch, r1c)
        dx = st[0] * cA + st[1] * cw
        log(f"zero it {it}: pass 1 |r_1| {np.linalg.norm(r[:2]):.2e} |r_H| {np.linalg.norm(r[2:]):.2e}; after the inner fix "
            f"|r_1| {np.linalg.norm(r2[:2]):.2e} (corrected {np.linalg.norm(r1c):.2e}) |r_H| {np.linalg.norm(r2[2:]):.2e}; "
            f"outer step {np.abs(st).max():.2e}; Sch sv {np.linalg.svd(Sch, compute_uv=False)}")
        x = x + dx
        if CKPT:
            json.dump(dict(x=x.tolist(), it=it), open(CKPT, "w"))
        log(f"   x {x.tolist()}")
        small = small + 1 if np.abs(st).max() < floor else 0
        if (np.abs(st).max() < tol * 10 and it > 0) or small >= 2:
            break
    r, K, _ = jacobian(x, M, xi1)
    L = Lin(K)
    return x, r, K, L




def refine(run, C, L, nit=5, label=""):
    """Iterative deflation. run(C) -> list of residual vectors (one per deflated direction) for inner coefficients
    C (list of arrays over the inner directions); C <- C - K_HH^-1 r_H until |r_H| stops falling by 10x.
    Returns (rows, C, history of max |r_H|) of the best iterate."""
    hist, best = [], None
    C = [c.copy() for c in C]
    for it in range(nit):
        rows = run(C)
        rH = max(np.linalg.norm(r[2:]) for r in rows) if len(rows[0]) > 2 else 0.0
        hist.append(rH)
        if best is None or rH < best[0]:
            best = (rH, rows, [c.copy() for c in C])
        if len(rows[0]) <= 2 or (it > 0 and rH > 0.1 * hist[-2]) or it == nit - 1:
            break
        C = [c + L.deflate(r[2:]) for c, r in zip(C, rows)]
    if label:
        log(f"   {label}: max |r_H| of the deflated directions by iteration {['%.1e' % v for v in hist]}")
    return best[1], best[2], hist


def omega1(x, M, xi1, K, L, gauge_scale=1.0):
    """gauge_scale = 1: c1(0) = -g1 (the light-cone gauge c(x = 1) = 1); 0: c1(0) = 0. In the full problem a change of
    c(xi = 0) by a function of theta is a time reparametrization, which maps jump-free solutions to jump-free ones, so
    omega1 does not depend on it; in the M-model the dependence measures the truncation's breaking of that symmetry."""
    n = R.npar(M)
    g1 = F2h(Fh_from(x, M), M)
    gauge = {k: -gauge_scale * v for k, v in g1.items()}
    # pass 1: the 1/n direction's r_H
    _, _, Jn = jacobian(x, M, xi1, extra=(Dir("n", q1=True), ({}, gauge)))
    nin = n - 2

    def run(C):
        cA = np.zeros(n); cA[0] = 1; cA[2:] = C[0]
        cw = np.zeros(n); cw[1] = 1; cw[2:] = C[1]
        dirs, masks, data = param_dirs(M, [cA, cw])
        prof = {}
        for p in range(2, n):
            for k, v in dFh(p, M).items():
                prof[k] = prof.get(k, 0) + C[2][p - 2] * v
        dirs.append(Dir("n", q1=True))
        masks.append(1 << 2)
        data[1 << 2] = (prof, gauge)
        Rr, _ = run_model(x, M, xi1, dirs, masks, data)
        return [Rr[1], Rr[2], Rr[3]]
    C0 = [L.deflate(K[2:, 0]), L.deflate(K[2:, 1]), L.deflate(Jn[2:])] if nin else [np.zeros(0)] * 3
    rows, C, hist = refine(run, C0, L, label="omega1 deflation")
    Sch = np.column_stack([L.corr(rows[0]), L.corr(rows[1])])
    Jc = L.corr(rows[2])
    st = -np.linalg.solve(Sch, Jc)
    cA = np.zeros(n); cA[0] = 1; cA[2:] = C[0]
    cw = np.zeros(n); cw[1] = 1; cw[2:] = C[1]
    cn = np.zeros(n); cn[2:] = C[2]
    dx = cn + st[0] * cA + st[1] * cw
    info = dict(Jn_pass1=Jn.tolist(), Jn_final=rows[2].tolist(), JnH_final=float(np.linalg.norm(rows[2][2:])),
                JnH_pass1=float(np.linalg.norm(Jn[2:])), deflation_hist=hist, Sch_omega=Sch.tolist(),
                Jn_corr=Jc.tolist())
    return dx, info


# ---------------- perturbations
def sch_lam(x, M, xi1, lam, S=None):
    """Sch(lam) (2 x 2) from deflated harmonic-1 profiles; returns Sch, C (2N x 2: full profile vectors), L, |r_H|"""
    if S is None:
        S, _, _ = R.pert_matrix(x, M, xi1, lam)
    L = Lin(S)
    nb = S.shape[0]
    C = np.zeros((nb, 2))
    C[0, 0] = C[1, 1] = 1
    if nb == 2:
        return S.copy(), C, L, 0.0
    basis = R.pert_basis(M)

    def run(Cin):
        dirs = [Dir(f"pert{i}", lam=lam) for i in range(2)]
        data = {}
        for i in range(2):
            v = np.concatenate([C[:2, i], Cin[i]])
            prof = {}
            for b, c in zip(basis, v):
                for k, w in b.items():
                    prof[k] = prof.get(k, 0) + c * w
            data[1 << i] = (prof, {})
        Rr, _ = run_model(x, M, xi1, dirs, [0, 1, 2], data)
        return [Rr[1], Rr[2]]
    rows, Cin, hist = refine(run, [L.deflate(S[2:, 0]), L.deflate(S[2:, 1])], L)
    C[2:, 0], C[2:, 1] = Cin[0], Cin[1]
    Sch = np.column_stack([L.corr(rows[0]), L.corr(rows[1])])
    return Sch, C, L, hist[-1] if hist else 0.0


def find_lam(x, M, xi1, lam, maxit=8, h=1e-4):
    g = lambda Sch, l: np.linalg.det(Sch) / (l * (l - 1))
    best = None
    for it in range(maxit):
        Sch, C, L, resH = sch_lam(x, M, xi1, lam)
        Schp, _, _, _ = sch_lam(x, M, xi1, lam + h)
        g0, g1 = g(Sch, lam), g(Schp, lam + h)
        step = -g0 * h / (g1 - g0)
        step = float(np.clip(step, -0.15, 0.15))
        log(f"lam it {it}: lam {lam:.12f}  g {g0:+.6e}  Sch sv {np.linalg.svd(Sch, compute_uv=False)}  "
            f"r_H of the deflated directions {resH:.1e}  step {step:+.2e}")
        if best is not None and abs(step) > 0.5 * abs(best):
            if abs(step) < 1e-5:
                break                              # noise floor
        best = step if best is None else min(abs(best), abs(step))
        lam = lam + step
        if abs(step) < 1e-11:
            break
    return lam


def lam1_step_w(x, M, xi1, lam, f, dx, wprof):
    """the hyper-dual run (perturbation e at lam with profile f, 1/n direction n with the selected dx, mixed e|n with
    inner profile wprof and gauge -h1); returns J_en, J_e, J_n rows"""
    Fh = Fh_from(x, M)
    g1 = F2h(Fh, M)
    h1 = {}
    for a, fa in Fh.items():
        for b, fb in f.items():
            if abs(a + b) <= M + 1:
                h1[a + b] = h1.get(a + b, 0) + 2 * fa * fb
    K1 = {}
    for p in range(R.npar(M)):
        if p == 1:
            continue
        for k, v in dFh(p, M).items():
            K1[k] = K1.get(k, 0) + dx[p] * v
    e, n = 1, 2
    dirs = [Dir("pert", lam=lam), Dir("n", q1=True, omega=W0 * dx[1])]
    masks = [0, e, n, e | n]
    data = {e: (f, {}), n: (K1, {k: -v for k, v in g1.items()}), e | n: (wprof, {k: -v for k, v in h1.items()})}
    jm = JetModel(M, W0 * x[1], Fh, dirs, masks, data)
    Rr, info = R.residuals(jm, x, M, xi1)
    return Rr[jm.idx[e | n]], Rr[jm.idx[e]], Rr[jm.idx[n]]


def lam1(x, M, xi1, lam, dx, h=1e-3):
    Sch, C, Lp, resH = sch_lam(x, M, xi1, lam)
    U, sv, Vt = np.linalg.svd(Sch)
    e, ll = Vt[-1], U[:, -1]
    Schp, _, _, _ = sch_lam(x, M, xi1, lam + h)
    Schm, _, _, _ = sch_lam(x, M, xi1, lam - h)
    fvec = C @ e
    nrm = fvec[0]
    fvec, e = fvec / nrm, e / nrm
    den = ll @ ((Schp - Schm) / (2 * h)) @ e
    f = R.profile_of(fvec, M)
    log(f"LAMBDA*: {lam:.12f}  Sch sv {sv}  eigenprofile {fvec.tolist()}  ell_l.Sch'.e* {den:.6e} (h = {h:g})")
    basis = R.pert_basis(M)
    nb = len(basis)

    def run(Cin):
        wprof = {}
        for b, c in zip(basis[2:], Cin[0]):
            for k, v in b.items():
                wprof[k] = wprof.get(k, 0) + c * v
        Jen, Je, Jn = lam1_step_w(x, M, xi1, lam, f, dx, wprof)
        run.last = (Je, Jn)
        return [Jen]
    rows, Cin, hist = refine(run, [np.zeros(nb - 2)], Lp, nit=6, label="e|n deflation")
    Je, Jn = run.last
    num = ll @ Lp.corr(rows[0])
    return dict(lam1=-num / den, lam_den=den, lam_num=num, fvec=fvec.tolist(), ell_l=ll.tolist(), Sch_lam=Sch.tolist(),
                Jen_final=rows[0].tolist(), en_deflation_hist=hist, Je=Je.tolist(), Jn_sel=Jn.tolist(),
                ellJe=float(ll @ Lp.corr(Je)), h=h)


def main():
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    R.RTOL = float(opts.get("--rtol", 1e-12))
    if "--rows" in opts:              # per-harmonic radii, e.g. --rows=5:0.2,7:0.15
        R.ROW_RADII.update({int(a.split(":")[0]): float(a.split(":")[1]) for a in opts["--rows"].split(",")})
    # M = 7 start: the M = 5 zero, with B5, C5, B7, C7 from hier.py's M = 7 inner solve at (2.72, w_inf)
    # (solver/inner_stokes/logs/hier_M7_fixA_run2.log)
    x0 = {1: [2.5421480476347464, 0.7769940135984376],
          3: [2.7264758856227167, 1.015603094848114, 0.14441947847126993, 0.011087425888277868],
          5: [2.718579189, 0.999228404, 0.149203575, 0.014719052, 0.008405347, 0.001470345],
          7: [2.718579189229888, 0.9992284031945021, 0.1492035749173264, 0.01471905249087869, 0.00844556, 0.001508163,
              0.000475916, 0.000121569]}[M] \
        if "--x" not in opts else [float(t) for t in opts["--x"].split(",")]
    tag = f"M{M}_xi{xi1:g}" + (f"_rtol{R.RTOL:g}" if R.RTOL != 1e-12 else "") \
        + ("_rows" + "_".join(f"{k}-{v:g}" for k, v in sorted(R.ROW_RADII.items())) if R.ROW_RADII else "")
    fn = opts.get("--out", os.path.join(HERE, "data", f"first_order_hier_{tag}.json"))
    global CKPT
    CKPT = fn.replace(".json", "_zero_ckpt.json")
    out = json.load(open(fn)) if os.path.exists(fn) else dict(M=M, xi1=xi1, rtol=R.RTOL)

    def save():
        json.dump(out, open(fn, "w"), indent=1)
    if "--redo" in opts:
        keys = {"zero": ["x"], "omega1": ["dx"], "lam": ["lam"], "lam1": ["lam1"], "gauge": ["omega1_gauge0"]}
        for key in opts["--redo"].split(","):
            for k2 in keys[key]:
                out.pop(k2, None)
    log(f"M {M} xi1 {xi1} rtol {R.RTOL}; have {sorted(k for k in out if k in ('x', 'dx', 'lam', 'lam1', 'omega1_gauge0'))}")
    if "x" not in out:
        x, r, K, L = find_zero(np.array(x0, float), M, xi1)
        out.update(x=x.tolist(), r=r.tolist(), r1_corr=L.corr(r).tolist(), K=K.tolist(), wf=x[1], omega=W0 * x[1])
        save()
    x = np.array(out["x"]); K = np.array(out["K"]); L = Lin(K)
    log(f"ZERO: x {x.tolist()}  r_1 {out['r'][:2]} (Schur-corrected {out['r1_corr']})  |r_H| {np.linalg.norm(out['r'][2:]):.1e}")
    if "--zero-only" in opts:
        return
    if "dx" not in out:
        dx, info = omega1(x, M, xi1, K, L)
        out.update(dx=dx.tolist(), omega1=W0 * dx[1], omega1_rel=dx[1] / x[1], **info)
        save()
    dx = np.array(out["dx"])
    log(f"OMEGA1: dx {dx.tolist()}  (r_H of the 1/n direction: pass 1 {out['JnH_pass1']:.1e}, deflated "
        f"{out.get('JnH_final', out.get('JnH_pass2', 0)):.1e})")
    log(f"   omega = {W0 * x[1]:.9f} + ({W0 * dx[1]:.6f})/n,  omega1/omega* = {dx[1] / x[1]:.6f};  "
        f"Delta1 = {-2 * np.pi * W0 * dx[1] / (W0 * x[1]) ** 2:.6f}")
    if "omega1_gauge0" not in out and "--no-gauge" not in opts:
        dx0, info0 = omega1(x, M, xi1, K, L, gauge_scale=0.0)
        out.update(omega1_gauge0=W0 * dx0[1], dx_gauge0=dx0.tolist())
        save()
    if "omega1_gauge0" in out:
        log(f"GAUGE: omega1 with c1(0) = -g1: {out['omega1']:.6f};  with c1(0) = 0: {out['omega1_gauge0']:.6f}  "
            f"(difference {out['omega1'] - out['omega1_gauge0']:+.2e}; zero in the full problem)")
    if "--omega-only" in opts:
        return
    if "--checks" in opts and "checks" not in out:
        # the exact modes: lam = 0 (theta shift, f = F') and lam = 1 (time translation, f = F + w F'): Sch(lam) is
        # singular there, with null vector the harmonic-1 part of f; lam = 0.5 for comparison
        Fh = Fh_from(x, M)
        w = W0 * x[1]
        F1 = Fh[1]
        chk = {}
        for lam_c, f1 in ((0.0, 1j * F1), (1.0, (1 + 1j * w) * F1), (0.5, None)):
            Sch, C, _, resH = sch_lam(x, M, xi1, lam_c)
            sv = np.linalg.svd(Sch, compute_uv=False)
            e = np.linalg.svd(Sch)[2][-1]
            ent = dict(sv=sv.tolist(), sv_ratio=float(sv[-1] / sv[0]), null=(e / np.linalg.norm(e)).tolist())
            if f1 is not None:
                ex = np.array([-2 * f1.imag, 2 * f1.real])      # (sin, cos) coefficients of the harmonic-1 part
                ex = ex / np.linalg.norm(ex)
                ent["exact"] = ex.tolist()
                ent["sin_angle"] = float(abs(e[0] * ex[1] - e[1] * ex[0]) / np.linalg.norm(e))
            chk[f"{lam_c:g}"] = ent
            log(f"CHECK lam = {lam_c:g}: Sch singular values {sv} (ratio {ent['sv_ratio']:.1e}); null vector {ent['null']}"
                + (f"; exact {ent['exact']}; sin(angle) {ent['sin_angle']:.1e}" if f1 is not None else ""))
        out.update(checks=chk)
        save()
    if "lam" not in out:
        lam = find_lam(x, M, xi1, float(opts.get("--lam", 2.0)))
        out.update(lam=lam)
        save()
    lam = out["lam"]
    if "lam1" not in out:
        out.update(lam1(x, M, xi1, lam, dx, h=float(opts.get("--h", 1e-3))))
        save()
    log(f"   e|n deflation |r_H| by iteration {out.get('en_deflation_hist')}; ell.J_e {out.get('ellJe', 0):.2e} "
        f"(eigenmode, against ell.Sch'.e* {out['lam_den']:.3e})")
    log(f"LAMBDA1: lam = {lam:.9f} + ({out['lam1']:.6f})/n")
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
