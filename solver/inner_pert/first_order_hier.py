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


def find_zero(x, M, xi1, maxit=8, tol=3e-15):
    n = R.npar(M)
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
        if np.abs(st).max() < tol * 10 and it > 0:
            break
    r, K, _ = jacobian(x, M, xi1)
    L = Lin(K)
    return x, r, K, L


def omega1(x, M, xi1, K, L):
    n = R.npar(M)
    g1 = F2h(Fh_from(x, M), M)
    gauge = {k: -v for k, v in g1.items()}
    # pass 1: the 1/n direction's r_H
    _, _, Jn = jacobian(x, M, xi1, extra=(Dir("n", q1=True), ({}, gauge)))
    cn = np.zeros(n); cn[2:] = L.deflate(Jn[2:])
    cA = np.zeros(n); cA[0] = 1; cA[2:] = L.deflate(K[2:, 0])
    cw = np.zeros(n); cw[1] = 1; cw[2:] = L.deflate(K[2:, 1])
    # pass 2: deflated A, omega and n' = n + inner directions cn
    dirs, masks, data = param_dirs(M, [cA, cw])
    prof = {}
    for p in range(2, n):
        for k, v in dFh(p, M).items():
            prof[k] = prof.get(k, 0) + cn[p] * v
    dirs.append(Dir("n", q1=True))
    masks.append(1 << 2)
    data[1 << 2] = (prof, gauge)
    Rr, _ = run_model(x, M, xi1, dirs, masks, data)
    Sch = np.column_stack([L.corr(Rr[1]), L.corr(Rr[2])])
    Jc = L.corr(Rr[3])
    st = -np.linalg.solve(Sch, Jc)
    dx = cn + st[0] * cA + st[1] * cw
    info = dict(Jn_pass1=Jn.tolist(), Jn_pass2=Rr[3].tolist(), JnH_pass2=float(np.linalg.norm(Rr[3][2:])),
                JnH_pass1=float(np.linalg.norm(Jn[2:])), Sch_omega=Sch.tolist())
    return dx, info


# ---------------- perturbations
def pert_pass1(x, M, xi1, lam):
    S, dS, r0 = R.pert_matrix(x, M, xi1, lam)
    return S


def sch_lam(x, M, xi1, lam):
    """Sch(lam) (2 x 2) from deflated harmonic-1 profiles; returns Sch, C (2N x 2: full profile vectors), L"""
    S = pert_pass1(x, M, xi1, lam)
    L = Lin(S)
    nb = S.shape[0]
    C = np.zeros((nb, 2))
    C[0, 0] = C[1, 1] = 1
    if nb > 2:
        C[2:, 0] = L.deflate(S[2:, 0])
        C[2:, 1] = L.deflate(S[2:, 1])
        basis = R.pert_basis(M)
        dirs = [Dir(f"pert{i}", lam=lam) for i in range(2)]
        masks = [0, 1, 2]
        data = {}
        for i in range(2):
            prof = {}
            for b, c in zip(basis, C[:, i]):
                for k, v in b.items():
                    prof[k] = prof.get(k, 0) + c * v
            data[1 << i] = (prof, {})
        Rr, _ = run_model(x, M, xi1, dirs, masks, data)
        Sch = np.column_stack([L.corr(Rr[1]), L.corr(Rr[2])])
        resH = max(np.linalg.norm(Rr[1][2:]), np.linalg.norm(Rr[2][2:]))
    else:
        Sch = S.copy()
        resH = 0.0
    return Sch, C, L, resH


def find_lam(x, M, xi1, lam, maxit=10, h=1e-4):
    g = lambda Sch, l: np.linalg.det(Sch) / (l * (l - 1))
    hist = []
    for it in range(maxit):
        Sch, C, L, resH = sch_lam(x, M, xi1, lam)
        Schp, _, _, _ = sch_lam(x, M, xi1, lam + h)
        g0, g1 = g(Sch, lam), g(Schp, lam + h)
        step = -g0 * h / (g1 - g0)
        step = float(np.clip(step, -0.15, 0.15))
        log(f"lam it {it}: lam {lam:.12f}  g {g0:+.6e}  Sch sv {np.linalg.svd(Sch, compute_uv=False)}  "
            f"r_H of the deflated directions {resH:.1e}  step {step:+.2e}")
        hist.append((lam, g0))
        lam = lam + step
        if abs(step) < 1e-11:
            break
    return lam


def main():
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    R.RTOL = float(opts.get("--rtol", 1e-12))
    x0 = {1: [2.5421480476347464, 0.7769940135984376],
          3: [2.7264758856227167, 1.015603094848114, 0.14441947847126993, 0.011087425888277868],
          5: [2.718579189, 0.999228404, 0.149203575, 0.014719052, 0.008405347, 0.001470345]}[M] \
        if "--x" not in opts else [float(t) for t in opts["--x"].split(",")]
    tag = f"M{M}_xi{xi1:g}" + (f"_rtol{R.RTOL:g}" if R.RTOL != 1e-12 else "")
    fn = opts.get("--out", os.path.join(HERE, "data", f"first_order_hier_{tag}.json"))
    out = json.load(open(fn)) if os.path.exists(fn) else dict(M=M, xi1=xi1, rtol=R.RTOL)
    save = lambda: json.dump(out, open(fn, "w"), indent=1)
    if "--redo" in opts:
        keys = {"zero": ["x"], "omega1": ["dx"], "lam": ["lam"], "lam1": ["lam1"]}
        for key in opts["--redo"].split(","):
            for k2 in keys[key]:
                out.pop(k2, None)
    log(f"M {M} xi1 {xi1} rtol {R.RTOL}; have {sorted(k for k in out if k in ('x', 'dx', 'lam', 'lam1'))}")
    if "x" not in out:
        x, r, K, L = find_zero(np.array(x0, float), M, xi1)
        out.update(x=x.tolist(), r=r.tolist(), r1_corr=L.corr(r).tolist(), K=K.tolist(), wf=x[1], omega=W0 * x[1])
        save()
    x = np.array(out["x"]); K = np.array(out["K"]); L = Lin(K)
    log(f"ZERO: x {x.tolist()}  r_1 {out['r'][:2]} (Schur-corrected {out['r1_corr']})  |r_H| {np.linalg.norm(out['r'][2:]):.1e}")
    if "dx" not in out:
        dx, info = omega1(x, M, xi1, K, L)
        out.update(dx=dx.tolist(), omega1=W0 * dx[1], omega1_rel=dx[1] / x[1], **info)
        save()
    dx = np.array(out["dx"])
    log(f"OMEGA1: dx {dx.tolist()}  (r_H of the 1/n direction: pass 1 {out['JnH_pass1']:.1e}, deflated {out['JnH_pass2']:.1e})")
    log(f"   omega = {W0 * x[1]:.9f} + ({W0 * dx[1]:.6f})/n,  omega1/omega* = {dx[1] / x[1]:.6f};  "
        f"Delta1 = {-2 * np.pi * W0 * dx[1] / (W0 * x[1]) ** 2:.6f}")
    if "--omega-only" in opts:
        return
    if "lam" not in out:
        lam = find_lam(x, M, xi1, float(opts.get("--lam", 2.0)))
        out.update(lam=lam)
        save()
    lam = out["lam"]
    if "lam1" not in out:
        h = 1e-4
        Sch, C, Lp, _ = sch_lam(x, M, xi1, lam)
        U, sv, Vt = np.linalg.svd(Sch)
        e, ll = Vt[-1], U[:, -1]
        Schp, _, _, _ = sch_lam(x, M, xi1, lam + h)
        Schm, _, _, _ = sch_lam(x, M, xi1, lam - h)
        den = ll @ ((Schp - Schm) / (2 * h)) @ e
        fvec = C @ e
        nrm = fvec[0]
        fvec, e = fvec / nrm, e / nrm
        den = den / nrm
        f = R.profile_of(fvec, M)
        log(f"LAMBDA*: {lam:.12f}  Sch sv {sv}  eigenprofile {fvec.tolist()}  ell_l.Sch'.e* {den:.6e}")
        # mixed run, pass 1 (with the inner perturbation profiles to deflate e|n)
        Jen, Jel, Je, Jn2, r0 = R.lam1_step(x, M, xi1, lam, f, dx)
        cen = Lp.deflate(Jen[2:]) if len(Jen) > 2 else np.zeros(0)
        if len(Jen) > 2:
            # pass 2: e|n with the inner perturbation profiles added to its data
            basis = R.pert_basis(M)
            wprof = {}
            for b, c in zip(basis[2:], cen):
                for k, v in b.items():
                    wprof[k] = wprof.get(k, 0) + c * v
            Jen2 = lam1_step_w(x, M, xi1, lam, f, dx, wprof)
        else:
            Jen2 = Jen
        num = ll @ Lp.corr(Jen2)
        lam1 = -num / den
        out.update(lam1=lam1, lam_den=den, lam_num=num, fvec=fvec.tolist(), Jen_pass1=Jen.tolist(),
                   Jen_pass2=Jen2.tolist(), Je=Je.tolist(), Jn_sel=Jn2.tolist(), ell_l=ll.tolist(), Sch_lam=Sch.tolist(),
                   lam1_oneshot_like=float(-(ll @ Lp.corr(Jen)) / den))
        save()
    log(f"   e|n: |r_H| pass 1 {np.linalg.norm(out['Jen_pass1'][2:]):.2e}, deflated {np.linalg.norm(out['Jen_pass2'][2:]):.2e}; "
        f"|J_n| (r_1) after the selection {np.linalg.norm(out['Jn_sel'][:2]):.2e}")
    log(f"LAMBDA1: lam = {lam:.9f} + ({out['lam1']:.6f})/n   (pass-1 value {out['lam1_oneshot_like']:.6f})")
    log(f"wrote {fn}")


def lam1_step_w(x, M, xi1, lam, f, dx, wprof):
    """R.lam1_step with an extra profile wprof in the e|n component; returns J_en"""
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
    return Rr[jm.idx[e | n]]


if __name__ == "__main__":
    main()
