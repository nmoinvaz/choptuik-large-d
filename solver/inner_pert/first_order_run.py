"""First-order (1/n) coefficients of omega and of the growth rate lambda = 1/gamma from the M-harmonic inner model
(notes/FIRST_ORDER.md). Everything is a jet of the lateral Stokes-jump integration (first_order_jet.py), so every
Jacobian is exact (no finite differences in parameters).

Steps, for the M-harmonic model (x = (A, w/w_inf, B3, C3, ...), F = A sin t + sum (B_nu sin nu t + C_nu cos nu t)):
  1. zero    Newton on the de-mixed residual r(x) (harmonics 1..M at xi1) with the exact Jacobian dr/dx.
  2. omega1  O(1/n) selection: the 1/n direction (the O(1/n) terms of the inner equations, profile correction K1 = 0,
             gauge c1(0) = -g1 = -[F^2]) has a jump J_n; the corrections dx = (A1, w1/w_inf, B3_1, ...) solve
             (dr/dx) dx = -J_n, so omega = omega* + omega1/n + ...  with omega1 = w_inf * dx[1].
  3. lam*    the M-model's growth rate near 2: real lam with det S(lam) = 0, S the 2N x 2N real map from the
             perturbation profile (sin nu t, cos nu t) to its de-mixed jump; Newton with dS/dlam from a jet direction.
             Checks: det S(0) = 0 (theta shift, f = F') and det S(1) = 0 (time translation, f = F + w F').
  4. lam1    hyper-dual run: perturbation e (lam*, eigenprofile f*), 1/n direction n (with dx from step 2:
             K1 = profile part, omega1, gauge -g1), d/dlam direction l. The e|n component (gauge c(0) = -h1,
             h1 = 2[F f*], the xi^2 coefficient of the perturbation's c) has the jump J_en; solvability against the
             left null vector ell of S(lam*): lam1 = -(ell . J_en) / (ell . J_el).

    python3 -u first_order_run.py M [--x=A,wf,B3,C3,...] [--xi1=0.2] [--lam=1.8] [--out=FILE]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from first_order_jet import JetModel, Dir, Fh_from, dFh, W0, F2h, demix_matrix, ri  # noqa: E402

T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def npar(M):
    return M + 1


def model_params(x, M, extra_dirs=(), extra_data=None, with_params=True):
    """background at x with one jet direction per parameter (bits 0..npar-1), plus extra directions"""
    n = npar(M)
    dirs, masks, data = [], [0], {}
    if with_params:
        for p in range(n):
            dirs.append(Dir(f"x{p}", omega=W0 if p == 1 else 0.0))
            masks.append(1 << p)
            data[1 << p] = (dFh(p, M), {})
    base = len(dirs)
    for i, d in enumerate(extra_dirs):
        dirs.append(d)
    for m, v in (extra_data or {}).items():
        data[m << base if isinstance(m, int) and m > 0 else m] = v
    return dirs, masks, data, base


def residuals(jm, x, M, xi1):
    J, info = jm.jumps((xi1,), rtol=RTOL)
    dm = demix_matrix(x, M, xi1)
    return np.array([ri(dm(J[xi1][c])) for c in range(jm.nc)]), info


RTOL = 1e-12


def find_zero(x, M, xi1, maxit=14, tol=1e-13):
    """Newton with the exact Jacobian (jet directions), Schur solve; stops at the noise floor (|r| not decreasing
    twice) and returns the best point"""
    best, worse = None, 0
    for it in range(maxit):
        dirs, masks, data, _ = model_params(x, M)
        jm = JetModel(M, W0 * x[1], Fh_from(x, M), dirs, masks, data)
        R, info = residuals(jm, x, M, xi1)
        r, K = R[0], R[1:].T
        nr = np.linalg.norm(r)
        dx = -schur_solve(K, r)
        log(f"zero it {it}: |r| {nr:.3e}  r_1 {r[0]:+.3e}{r[1]:+.3e}i  step {np.abs(dx).max():.2e}  "
            f"x {np.array2string(x, precision=12, max_line_width=300)}")
        if best is None or nr < best[0]:
            best, worse = (nr, x.copy(), r, K), 0
        else:
            worse += 1
            if worse >= 2:
                break
        x = x + dx
        if np.abs(dx).max() < tol * max(1, np.abs(x).max()):
            break
    nr, x, r, K = best
    return x, r, K, None


def omega1_step(x, M, xi1, K):
    """step 2: jump of the 1/n direction at the zero, and the selection correction dx"""
    n = npar(M)
    g1 = F2h(Fh_from(x, M), M)
    dirs, masks, data, base = model_params(x, M)
    dirs.append(Dir("n", q1=True))
    masks.append(1 << base)
    data[1 << base] = ({}, {k: -v for k, v in g1.items()})
    jm = JetModel(M, W0 * x[1], Fh_from(x, M), dirs, masks, data)
    R, info = residuals(jm, x, M, xi1)
    r, Kp, Jn = R[0], R[1:1 + n].T, R[1 + n]
    rs = 1 / np.linalg.norm(Kp, axis=1)
    Ks = Kp * rs[:, None]
    cs = 1 / np.linalg.norm(Ks, axis=0)
    dx_lu = -np.linalg.solve(Ks * cs[None, :], Jn * rs) * cs
    dx = -schur_solve(Kp, Jn)
    log(f"   omega1 step: Schur solve vs equilibrated LU, max difference {np.abs(dx - dx_lu).max():.2e} "
        f"(omega part {dx[1] - dx_lu[1]:+.2e})")
    return dx, Jn, r, Kp, info


def pert_basis(M):
    """2N real profile directions: sin nu t, cos nu t for odd nu <= M"""
    out = []
    for nu in range(1, M + 1, 2):
        out.append({nu: 1 / 2j, -nu: -1 / 2j})
        out.append({nu: 0.5, -nu: 0.5})
    return out


def pert_matrix(x, M, xi1, lam):
    """S(lam) (2N x 2N, columns = jumps of the basis profiles) and dS/dlam"""
    basis = pert_basis(M)
    nb = len(basis)
    dirs = [Dir(f"pert{i}", lam=lam) for i in range(nb)] + [Dir("l", dlam=True)]
    lb = 1 << nb
    masks = [0] + [1 << i for i in range(nb)] + [(1 << i) | lb for i in range(nb)]
    data = {1 << i: (basis[i], {}) for i in range(nb)}
    jm = JetModel(M, W0 * x[1], Fh_from(x, M), dirs, masks, data)
    R, info = residuals(jm, x, M, xi1)
    S = R[1:1 + nb].T
    dS = R[1 + nb:1 + 2 * nb].T
    return S, dS, R[0]


def scaled(S):
    rs = 1 / np.linalg.norm(S, axis=1)
    return S * rs[:, None], rs


def null_vectors(S):
    """right and left (near-)null vectors of S by block elimination of the harmonics >= 3 (rows and columns 2..):
    the jumps of the higher harmonics leak into the harmonic-1 rows with weights ~1e4 (M = 3), so a plain SVD, even
    row-scaled, does not resolve the harmonic-1 block. Schur complement Sch = S_ll - S_lh S_hh^-1 S_hl (2 x 2);
    returns f, ell (ell S ~ 0, S f ~ 0) and the singular values of Sch."""
    if S.shape[0] == 2:
        U, sv, Vt = np.linalg.svd(S)
        return Vt[-1], U[:, -1], sv
    Sll, Slh, Shl, Shh = S[:2, :2], S[:2, 2:], S[2:, :2], S[2:, 2:]
    X = np.linalg.solve(Shh, Shl)
    Sch = Sll - Slh @ X
    U, sv, Vt = np.linalg.svd(Sch)
    e = Vt[-1]
    ll = U[:, -1]
    f = np.concatenate([e, -X @ e])
    lh = -np.linalg.solve(Shh.T, Slh.T @ ll)
    return f, np.concatenate([ll, lh]), sv


def schur_solve(K, b):
    """solve K x = b by eliminating the harmonic >= 3 rows/columns first (as null_vectors)"""
    if K.shape[0] == 2:
        return np.linalg.solve(K, b)
    Kll, Klh, Khl, Khh = K[:2, :2], K[:2, 2:], K[2:, :2], K[2:, 2:]
    X = np.linalg.solve(Khh, Khl)
    y = np.linalg.solve(Khh, b[2:])
    xl = np.linalg.solve(Kll - Klh @ X, b[:2] - Klh @ y)
    return np.concatenate([xl, y - X @ xl])


def find_lam(x, M, xi1, lam, maxit=12):
    """Newton on g(lam) = det S(lam) / (lam (lam - 1)) (the exact roots 0 and 1 deflated):
    d ln g / d lam = tr(S^-1 dS) - 1/lam - 1/(lam - 1)"""
    prev = None
    for it in range(maxit):
        S, dS, r0 = pert_matrix(x, M, xi1, lam)
        dl = np.trace(np.linalg.solve(S, dS)) - 1 / lam - 1 / (lam - 1)
        step = -1 / dl
        if abs(step) > 0.15:
            step = 0.15 * np.sign(step)
        g = np.linalg.det(S) / (lam * (lam - 1))
        log(f"lam it {it}: lam {lam:.12f}  deflated det {g:+.6e}  step {step:+.2e}")
        if prev is not None and abs(step) >= abs(prev) and abs(step) < 1e-9:
            break                                  # noise floor
        lam = lam + step
        prev = step
        if abs(step) < 1e-13:
            break
    S, dS, r0 = pert_matrix(x, M, xi1, lam)
    return lam, S, dS


def profile_of(vec, M):
    out = {}
    for b, c in zip(pert_basis(M), vec):
        for k, v in b.items():
            out[k] = out.get(k, 0) + c * v
    return out


def lam1_step(x, M, xi1, lam, f, dx):
    """step 4: hyper-dual run (e, n, l); returns J_en, J_el, J_e, J_n (de-mixed, real-stacked)"""
    Fh = Fh_from(x, M)
    g1 = F2h(Fh, M)
    # h1 = 2 [F f] (even, truncated)
    h1 = {}
    for a, fa in Fh.items():
        for b, fb in f.items():
            if abs(a + b) <= M + 1:
                h1[a + b] = h1.get(a + b, 0) + 2 * fa * fb
    # profile correction K1 = sum_p dx_p dF/dx_p (p != 1)
    K1 = {}
    for p in range(npar(M)):
        if p == 1:
            continue
        for k, v in dFh(p, M).items():
            K1[k] = K1.get(k, 0) + dx[p] * v
    e, n, l = 1, 2, 4
    dirs = [Dir("pert", lam=lam), Dir("n", q1=True, omega=W0 * dx[1]), Dir("l", dlam=True)]
    masks = [0, e, n, e | n, e | l]
    data = {e: (f, {}), n: (K1, {k: -v for k, v in g1.items()}), e | n: ({}, {k: -v for k, v in h1.items()})}
    jm = JetModel(M, W0 * x[1], Fh, dirs, masks, data)
    R, info = residuals(jm, x, M, xi1)
    idx = jm.idx
    return R[idx[e | n]], R[idx[e | l]], R[idx[e]], R[idx[n]], R[0]


def sym_step(x, M, xi1):
    """step 5 (symmetry route): the lambda = 2 mode e = D_P Z_0 (Proposition 5's profile f2, truncated as the model)
    and its O(1/n) part W forced by the s-translation breaking (SymForce); lam1 = -(ell2 . J_W)/(ell2 . J_el) with
    ell2 the left singular vector of S(2) of smallest singular value. In the full problem S(2) f2 = 0 exactly; in the
    model the mismatch -(ell2 . J_e)/(ell2 . J_el) estimates lam* - 2."""
    from first_order_jet import SymForce
    w = W0 * x[1]
    Fh = Fh_from(x, M)
    bg = JetModel(M, w, Fh, [], [0], {})
    bg.series_at(1.0, 3, stop=False)
    L = bg.L
    k = bg.k
    d1 = bg.ser["D"][1][0]
    g1 = bg.ser["G"][1][0]
    P = -2 * g1 / (1 + 1j * w * k)
    Xd0 = (1 + 1j * k * w) * bg.ser["D"][0][0]
    a0 = 2 * d1 + np.convolve(P, Xd0)[L:3 * L + 1] * bg.odd
    f2 = {int(kk): a0[kk + L] / 2 for kk in range(-M, M + 1) if kk % 2}
    S2, dS2, _ = pert_matrix(x, M, xi1, 2.0)
    fv, ell2, sv = null_vectors(S2)
    e, l, Wb = 1, 2, 4
    dirs = [Dir("pert", lam=2.0), Dir("l", dlam=True), Dir("W", lam=2.0, force=SymForce(Wb))]
    jm = JetModel(M, w, Fh, dirs, [0, e, e | l, Wb], {e: (f2, {})})
    R, info = residuals(jm, x, M, xi1)
    Je, Jel, JW = R[jm.idx[e]], R[jm.idx[e | l]], R[jm.idx[Wb]]
    lam1 = -ell2 @ JW / (ell2 @ Jel)
    mism = -ell2 @ Je / (ell2 @ Jel)
    # the profile f2 in the (sin, cos) basis, against the model's own null vector of S(2)
    f2v = []
    for nu in range(1, M + 1, 2):
        c = f2.get(nu, 0)
        f2v += [-2 * c.imag, 2 * c.real]
    f2v = np.array(f2v) / (-2 * f2[1].imag)
    return dict(lam1_sym=lam1, lam_mismatch=mism, sv_S2=sv.tolist(), f2=f2v.tolist(), Je=Je.tolist(), Jel=Jel.tolist(),
                JW=JW.tolist(), ell2=ell2.tolist(), nullvec_S2=(fv / fv[0]).tolist())


def main():
    global RTOL
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    RTOL = float(opts.get("--rtol", 1e-12))
    x0 = {1: [2.54214805, 0.77699401], 3: [2.7265, 1.0156, 0, 0],
          5: [2.718579189, 0.999228404, 0.149203575, 0.014719052, 0.008405347, 0.001470345]}[M] \
        if "--x" not in opts else [float(t) for t in opts["--x"].split(",")]
    x = np.array(x0, float)
    if "--x" not in opts and M == 3:
        Fn = (x[0] / 2j) * (0.0530 + 0.0041j)
        x[2], x[3] = -2 * Fn.imag, 2 * Fn.real
    tag = f"M{M}_xi{xi1:g}" + (f"_rtol{RTOL:g}" if RTOL != 1e-12 else "")
    fn = opts.get("--out", os.path.join(HERE, "data", f"first_order_{tag}.json"))
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    out = json.load(open(fn)) if os.path.exists(fn) and "--fresh" not in opts else dict(M=M, xi1=xi1, rtol=RTOL)

    def save():
        json.dump(out, open(fn, "w"), indent=1)
    if "--redo" in opts:
        keys = {"zero": ["x"], "omega1": ["dx"], "lam": ["lam", "lam1"], "lam1": ["lam1"], "sym": ["lam1_sym"],
                "checks": ["sch_sv_lam0", "sch_sv_lam1"]}
        for key in opts["--redo"].split(","):
            for k2 in keys[key]:
                out.pop(k2, None)
    if "x" in out:
        log(f"M {M} xi1 {xi1} rtol {RTOL}: resuming from {fn}")
    else:
        log(f"M {M} xi1 {xi1} rtol {RTOL}: start x {x.tolist()}")
    # 1. zero
    if "x" not in out:
        x, r, K, _ = find_zero(x, M, xi1)
        out.update(x=x.tolist(), r=r.tolist(), wf=x[1], omega=W0 * x[1],
                   jac_sv=np.linalg.svd(K, compute_uv=False).tolist())
        save()
    x = np.array(out["x"])
    log(f"ZERO: x {x.tolist()}  |r| {np.linalg.norm(out['r']):.2e}  Jacobian singular values {out['jac_sv']}")
    # 2. omega1
    if "dx" not in out:
        dx, Jn, r, Kp, info = omega1_step(x, M, xi1, None)
        w1 = W0 * dx[1]
        out.update(dx=dx.tolist(), Jn=Jn.tolist(), omega1=w1, A1=dx[0], omega1_rel=w1 / (W0 * x[1]))
        save()
    dx = np.array(out["dx"])
    w1 = W0 * dx[1]
    log(f"OMEGA1: dx {dx.tolist()}")
    log(f"   omega = {W0 * x[1]:.9f} + ({w1:.6f})/n,  omega1/omega* = {w1 / (W0 * x[1]):.6f};  "
        f"Delta1 = -2 pi omega1/omega*^2 = {-2 * np.pi * w1 / (W0 * x[1]) ** 2:.6f}")
    if "--omega-only" in opts:
        return
    # growth-rate checks at lam = 0 and 1 (singular values of the harmonic-1 Schur complement; the null vector
    # should be F' and F + w F')
    for lam_c, fc in ((0.0, "F'"), (1.0, "F + w F'")):
        key = f"sch_sv_lam{lam_c:g}"
        if key not in out:
            S, dS, _ = pert_matrix(x, M, xi1, lam_c)
            fv, ell, sv = null_vectors(S)
            out[key] = sv.tolist()
            out[f"nullvec_lam{lam_c:g}"] = (fv / fv[0]).tolist()
            save()
        log(f"check lam = {lam_c:g}: Schur-complement singular values {out[key]}; null vector {out[f'nullvec_lam{lam_c:g}']} "
            f"(exact mode f = {fc})")
    # 3. lam*
    if "lam" not in out:
        lam = float(opts.get("--lam", 2.0))
        lam, S, dS = find_lam(x, M, xi1, lam)
        fvec, ell, sv = null_vectors(S)
        fvec = fvec / fvec[0]
        out.update(lam=lam, fvec=fvec.tolist(), ell=ell.tolist(), sch_sv_lam=sv.tolist())
        save()
    lam = out["lam"]
    fvec, ell = np.array(out["fvec"]), np.array(out["ell"])
    log(f"LAMBDA*: {lam:.12f}  profile (sin t, cos t, sin 3t, ...) {fvec.tolist()}  Schur sv {out['sch_sv_lam']}")
    # 4. lam1, direct route
    if "lam1" not in out:
        Jen, Jel, Je, Jn2, r0 = lam1_step(x, M, xi1, lam, profile_of(fvec, M), dx)
        lam1 = -ell @ Jen / (ell @ Jel)
        out.update(lam1=lam1, Jen=Jen.tolist(), Jel=Jel.tolist(), Je=Je.tolist(), Jn_sel=Jn2.tolist(),
                   ellJe=ell @ Je, ellJel=ell @ Jel, r_hyper=r0.tolist())
        save()
    log(f"   check: ell.J_e {out['ellJe']:.2e} against ell.J_el {out['ellJel']:.3e} (eigenmode); |J_n| after the "
        f"selection {np.linalg.norm(out['Jn_sel']):.2e} (before: {np.linalg.norm(out['Jn']):.2e})")
    log(f"LAMBDA1 (direct): lam = {lam:.9f} + ({out['lam1']:.6f})/n")
    # 5. lam1, symmetry route
    if "lam1_sym" not in out:
        out.update(sym_step(x, M, xi1))
        save()
    log(f"SYMMETRY ROUTE: lam1 = {out['lam1_sym']:.6f}  (model's lam* - 2 estimated by the mismatch: "
        f"{out['lam_mismatch']:+.4e}; Schur singular values of S(2) {out['sv_S2']})")
    log(f"   f2 (sin t, cos t, ...) {out['f2']}  null vector of S(2) {out['nullvec_S2']}")
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
