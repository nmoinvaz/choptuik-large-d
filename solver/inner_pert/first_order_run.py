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
    J, info = jm.jumps((xi1,))
    dm = demix_matrix(x, M, xi1)
    return np.array([ri(dm(J[xi1][c])) for c in range(jm.nc)]), info


def find_zero(x, M, xi1, maxit=12, tol=1e-13):
    for it in range(maxit):
        dirs, masks, data, _ = model_params(x, M)
        jm = JetModel(M, W0 * x[1], Fh_from(x, M), dirs, masks, data)
        R, info = residuals(jm, x, M, xi1)
        r, K = R[0], R[1:].T
        # equilibrate: rows by their norm, columns by their norm
        rs = 1 / np.linalg.norm(K, axis=1)
        Ks = K * rs[:, None]
        cs = 1 / np.linalg.norm(Ks, axis=0)
        dx = -np.linalg.solve(Ks * cs[None, :], r * rs) * cs
        log(f"zero it {it}: |r| {np.linalg.norm(r):.3e}  r_1 {r[0]:+.3e}{r[1]:+.3e}i  step {np.abs(dx).max():.2e}  "
            f"x {np.array2string(x, precision=12, max_line_width=300)}")
        x = x + dx
        if np.abs(dx).max() < tol * max(1, np.abs(x).max()):
            break
    return x, r, K, jm


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
    dx = -np.linalg.solve(Ks * cs[None, :], Jn * rs) * cs
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
    """right and left null vectors of S via the row-scaled matrix's SVD"""
    Ss, rs = scaled(S)
    U, sv, Vt = np.linalg.svd(Ss)
    f = Vt[-1]
    ell = U[:, -1] * rs
    return f, ell, sv


def find_lam(x, M, xi1, lam, maxit=10):
    for it in range(maxit):
        S, dS, r0 = pert_matrix(x, M, xi1, lam)
        Ss, rs = scaled(S)
        step = -1 / np.trace(np.linalg.solve(S, dS))
        sv = np.linalg.svd(Ss, compute_uv=False)
        log(f"lam it {it}: lam {lam:.12f}  smallest/largest singular value (row-scaled) {sv[-1] / sv[0]:.2e}  step {step:+.2e}")
        lam = lam + step
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


def main():
    argv = sys.argv[1:]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    M = int(args[0])
    xi1 = float(opts.get("--xi1", 0.2))
    x0 = {1: [2.54214805, 0.77699401], 3: [2.7265, 1.0156, 0, 0], 5: [2.718579, 0.999228, 0, 0, 0, 0]}[M] \
        if "--x" not in opts else [float(t) for t in opts["--x"].split(",")]
    x = np.array(x0, float)
    if "--x" not in opts and M >= 3:
        F1 = x[0] / 2j
        rat = {3: 0.054883 + 0.005414j, 5: 0.003092 + 0.000541j} if M == 5 else {3: 0.0530 + 0.0041j}
        for i, nu in enumerate(range(3, M + 1, 2)):
            Fn = F1 * rat[nu]
            x[2 + 2 * i], x[3 + 2 * i] = -2 * Fn.imag, 2 * Fn.real
    out = dict(M=M, xi1=xi1)
    log(f"M {M} xi1 {xi1} start x {x.tolist()}")
    x, r, K, jm = find_zero(x, M, xi1)
    out.update(x=x.tolist(), r=r.tolist(), wf=x[1], omega=W0 * x[1])
    log(f"ZERO: x {x.tolist()}  |r| {np.linalg.norm(r):.2e}  Jacobian singular values "
        f"{np.linalg.svd(K, compute_uv=False)}")
    dx, Jn, r, Kp, info = omega1_step(x, M, xi1, K)
    w1 = W0 * dx[1]
    out.update(dx=dx.tolist(), Jn=Jn.tolist(), omega1=w1, A1=dx[0])
    log(f"OMEGA1: dx {dx.tolist()}")
    log(f"   omega = {W0 * x[1]:.9f} + ({w1:.6f})/n,  omega1/omega* = {w1 / (W0 * x[1]):.6f};  "
        f"Delta1 = -2 pi omega1/omega*^2 = {-2 * np.pi * w1 / (W0 * x[1]) ** 2:.6f}")
    # growth-rate checks at lam = 0 and 1
    for lam_c, fc in ((0.0, "F'"), (1.0, "F + w F'")):
        S, dS, _ = pert_matrix(x, M, xi1, lam_c)
        Ss, _ = scaled(S)
        sv = np.linalg.svd(Ss, compute_uv=False)
        fv, ell, _ = null_vectors(S)
        log(f"check lam = {lam_c}: row-scaled singular values {sv}  (exact mode f = {fc})")
        out[f"sv_lam{lam_c:g}"] = sv.tolist()
    lam = float(opts.get("--lam", 1.8))
    lam, S, dS = find_lam(x, M, xi1, lam)
    fvec, ell, sv = null_vectors(S)
    fvec = fvec / fvec[0]
    f = profile_of(fvec, M)
    out.update(lam=lam, fvec=fvec.tolist(), ell=ell.tolist(), sv_lam=sv.tolist())
    log(f"LAMBDA*: {lam:.12f}  profile (sin t, cos t, sin 3t, ...) {fvec.tolist()}")
    Jen, Jel, Je, Jn2, r0 = lam1_step(x, M, xi1, lam, f, dx)
    lam1 = -ell @ Jen / (ell @ Jel)
    out.update(lam1=lam1, Jen=Jen.tolist(), Jel=Jel.tolist(), Je=Je.tolist(), Jn_sel=Jn2.tolist())
    log(f"   check: |J_e| {np.linalg.norm(Je):.2e} (eigenmode), |J_n| after selection {np.linalg.norm(Jn2):.2e}, |r| {np.linalg.norm(r0):.2e}")
    log(f"LAMBDA1: lam = {lam:.9f} + ({lam1:.6f})/n   (ell.J_el {ell @ Jel:.4e})")
    fn = opts.get("--out", os.path.join(HERE, "data", f"first_order_M{M}_xi{xi1:g}.json"))
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(out, open(fn, "w"), indent=1)
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
