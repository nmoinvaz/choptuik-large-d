#!/usr/bin/env python3
"""DSSEll.mode_matrix by the variational (tangent-linear) RK4 map: the same float64 M(lambda), faster (#1025).

DSSEll.mode_matrix differentiates each shooting block by a complex step: one batch member per input column, so every RK4
stage evaluates the nonlinear right side in complex arithmetic on nf + 1 members, with complex products against the real
tau operators. Here each block's base trajectory is integrated once in real arithmetic, and its nf columns are carried
together as real tangents through the linearised right side (forward mode), whose coefficients are (1, N) arrays of the
base stage state. Forward mode of the RK4 step is the RK4 step of the variational equation at the base stage states, so
the result is the derivative of the same discrete map, equal in exact arithmetic to the complex step's. Unchanged: the
frozen RK4 step counts, the x grid, the tau operators (DSSEll._Dm, _Tm, phys, spec), and lambda entering every tau
derivative of a perturbation as D f + lambda f (dss.py).
  - blocks 1-11 (node to node): tangents start as the coefficient -> grid rows of DSSEll.phys;
  - block 0 (centre) and block 12 (horizon): the Taylor series and its evaluation keep the complex step, then the
    integration is variational; the horizon series' periodic solves use one real LU of the background operator
    omega D + kappa_0 + lambda for all members (the first-order part of each member's solve, which is all the complex
    step's imaginary part keeps; DSSEll.solve_periodic factors one complex matrix per member).

    python ell_fastm.py check STUB.jsonl STATE.npy D LAM [--blocks=0,1,12] [--basis=coef|grid] [--mutate=nolam]
                              [--out=F.json]
        M_fast against DSSEll.mode_matrix block by block (max |difference| / max |block| <= 1e-10) and slogdet (the same
        sign, ln|det| within 1e-8), with both timings; exit 0 only if every compared block and the determinant pass.
        --blocks limits both builds to those blocks (the determinant is then not compared). --mutate=nolam drops lambda
        from the variational map (the series keep it): the check must fail.
    python ell_fastm.py selftest [--mutate=nolam|coef]
        a k_c 15 state made here: the linearised right side against a central difference of DSSEll.rhs (1e-7), and
        M_fast against DSSEll.mode_matrix at two lambda (1e-10); --mutate=coef perturbs one tangent coefficient.
        Exit 0 only if every comparison passes.
References: forward-mode differentiation of a Runge-Kutta step (the variational equation integrated with the same
scheme), Hairer, Norsett and Wanner, Solving Ordinary Differential Equations I (Springer 1993), section II.14
(sensitivity of the numerical solution); complex-step derivative, Squire and Trapp, SIAM Rev. 40, 110 (1998)."""
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, PAR       # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


# ---------------------------------------------------------------- the linearised right side
def _base(S, x, s, om, lam):
    """DSSEll.rhs at the base state s (5, 1, N), before its truncation, and the (1, N) coefficients of its derivative:
    d(field_x) = sum over the inputs of coef * d(input), plus coef_D * D_unit d(p) (or d(m)); lam enters the tangents'
    tau derivatives (D f + lam f)"""
    p, m, a, c, q = s
    n, L = S.n, S.L
    a2 = a * a
    qq = q * q
    g = a2 * qq
    W = ((n - 1) * (a2 - 1) - 2 * L * g) / x                     # c_x = c W
    cx = c * W
    U = ((p * p + m * m) + 2 * L * g + (n - 1) * (1 - a2)) / (2 * x)   # a_x = a U
    ax = a * U
    qx = (p + m) / (2 * x)
    pc = n * c * (p + m) / (2 * x)
    lt = L * c * a2 * q / x
    Dp = (p @ S._Dm) * om
    Dmm = (m @ S._Dm) * om
    ip = 1.0 / (x - c)
    im = 1.0 / (x + c)
    px = (-Dp + cx * p - c * p / x + pc - lt) * ip
    mx = (-Dmm - cx * m + c * m / x - pc + lt) * im
    base = np.stack([px, mx, ax, cx, qx])
    wA = (2 * (n - 1) * a - 4 * L * a * qq) / x                   # dW = wA dA + wQ dQ
    wQ = -4 * L * a2 * q / x
    cxA, cxQ = c * wA, c * wQ                                     # dc_x = W dC + cxA dA + cxQ dQ
    axA = U + a * (2 * L * a * qq - (n - 1) * a) / x              # da_x = axA dA + axP dP + axM dM + axQ dQ
    axP, axM, axQ = a * p / x, a * m / x, 2 * L * a * a2 * q / x
    hx = 0.5 / x                                                  # dpsi_x = (dP + dM) / (2x)
    pcC, pcPM = n * (p + m) / (2 * x), n * c / (2 * x)
    ltC, ltA, ltQ = L * a2 * q / x, 2 * L * c * a * q / x, L * c * a2 / x
    lt_ = 0.0 if MUTATE == "nolam" else lam
    if MUTATE == "coef":
        axP = axP * (1 + 1e-6)
    cf = dict(pP=(-lt_ + cx - c / x + pcPM) * ip, pM=pcPM * ip, pA=(cxA * p - ltA) * ip,
              pC=(W * p - p / x + pcC - ltC + px) * ip, pQ=(cxQ * p - ltQ) * ip, pD=-om * ip,
              mM=(-lt_ - cx + c / x - pcPM) * im, mP=-pcPM * im, mA=(-cxA * m + ltA) * im,
              mC=(-W * m + m / x - pcC + ltC - mx) * im, mQ=(-cxQ * m + ltQ) * im, mD=-om * im,
              aA=axA, aP=axP, aM=axM, aQ=axQ, cC=W, cA=cxA, cQ=cxQ, hx=hx)
    return base, cf


def _tan(cf, P, M, A, C, Q, DP, DM):
    """the five tangent right sides on the grid (before truncation)"""
    return (cf["pP"] * P + cf["pM"] * M + cf["pA"] * A + cf["pC"] * C + cf["pQ"] * Q + cf["pD"] * DP,
            cf["mP"] * P + cf["mM"] * M + cf["mA"] * A + cf["mC"] * C + cf["mQ"] * Q + cf["mD"] * DM,
            cf["aA"] * A + cf["aP"] * P + cf["aM"] * M + cf["aQ"] * Q,
            cf["cC"] * C + cf["cA"] * A + cf["cQ"] * Q,
            cf["hx"] * (P + M))


def rhs_lin(S, x, s, ds, om, lam, scale=None):
    """DSSEll.rhs at the base state s (5, 1, N) and its derivative applied to grid tangents ds (5, nf, N), both truncated
    (DSSEll.trunc) and times scale (the integrate_u / integrate_log factor) when scale is not None"""
    base, cf = _base(S, x, s, om, lam)
    N = s.shape[-1]
    nf = ds.shape[1]
    DPM = (ds[:2].reshape(2 * nf, N) @ S._Dm).reshape(2, nf, N)
    out = np.empty_like(ds)
    for f, v in enumerate(_tan(cf, *ds, DPM[0], DPM[1])):
        out[f] = v
    base = S.trunc(base)
    dk = (out.reshape(5 * nf, N) @ S._Tm).reshape(5, nf, N)
    if scale is not None:
        base = scale * base
        dk = scale * dk
    return base, dk


class CoefOps:
    """the tau maps for tangents carried as parity-restricted Fourier coefficients (nf, blk), the block layout"""

    def __init__(self, S):
        no, ne = S.n_odd, S.n_even
        self.cut = np.cumsum([0, no, no, ne, ne, no])
        self.Bo, self.Be = S._B["odd"], S._B["even"]
        self.BBo = np.ascontiguousarray(np.concatenate([self.Bo, self.Bo @ S._Dm], axis=1))   # values, d/dtau
        self.So, self.Se = S._S["odd"], S._S["even"]


def rhs_lin_coef(S, x, s, dc, om, lam, scale, ops):
    """rhs_lin with the tangents as coefficients dc (nf, blk): grid values by DSSEll.phys's matrices, the right sides back
    by DSSEll.spec's (the parity-restricted projection, which equals DSSEll.trunc on fields of the right parity)"""
    base, cf = _base(S, x, s, om, lam)
    N = S.N
    k = ops.cut
    PP = dc[:, k[0]:k[1]] @ ops.BBo
    MM = dc[:, k[1]:k[2]] @ ops.BBo
    A = dc[:, k[2]:k[3]] @ ops.Be
    C = dc[:, k[3]:k[4]] @ ops.Be
    Q = dc[:, k[4]:k[5]] @ ops.Bo
    tp, tm, ta, tc, tq = _tan(cf, PP[:, :N], MM[:, :N], A, C, Q, PP[:, N:], MM[:, N:])
    out = np.empty_like(dc)
    out[:, k[0]:k[1]] = tp @ ops.So
    out[:, k[1]:k[2]] = tm @ ops.So
    out[:, k[2]:k[3]] = ta @ ops.Se
    out[:, k[3]:k[4]] = tc @ ops.Se
    out[:, k[4]:k[5]] = tq @ ops.So
    base = S.trunc(base)
    if scale is not None:
        base = scale * base
        out = scale * out
    return base, out


def _rk4(S, s, ds, y0, h, steps, xmap, om, lam, ops=None):
    """RK4 in y (as dss.DSS.integrate, DSSEll.integrate_u, DSSEll.integrate_log) on the base and its tangents;
    xmap(y) = (x, scale); ops: CoefOps for coefficient tangents, None for grid tangents"""
    y = y0

    def f(y, s, ds):
        x, sc = xmap(y)
        if ops is None:
            return rhs_lin(S, x, s, ds, om, lam, sc)
        return rhs_lin_coef(S, x, s, ds, om, lam, sc, ops)
    for _ in range(steps):
        k1, d1 = f(y, s, ds)
        k2, d2 = f(y + h / 2, s + h / 2 * k1, ds + h / 2 * d1)
        k3, d3 = f(y + h / 2, s + h / 2 * k2, ds + h / 2 * d2)
        k4, d4 = f(y + h, s + h * k3, ds + h * d3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ds = ds + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        y += h
    return s, ds


def seg_lin(S, s, ds, xa, xb, om, lam, ops=None):
    """DSSEll.seg on base and tangents (frozen step counts required)"""
    k = S._steps(xa, xb)
    if k is None:
        raise ValueError("ell_fastm needs frozen RK step counts (S.freeze_from)")
    if min(xa, xb) >= 0.4:
        ya, yb = np.log(1 - xa), np.log(1 - xb)
        return _rk4(S, s, ds, ya, (yb - ya) / k, k, lambda y: (1 - np.exp(y), -np.exp(y)), om, lam, ops)
    return _rk4(S, s, ds, xa, (xb - xa) / k, k, lambda y: (y, None), om, lam, ops)


def log_lin(S, s, ds, xa, xb, om, lam, ops=None):
    """DSSEll.integrate_log on base and tangents"""
    k = S.steps_map["log"]
    return _rk4(S, s, ds, np.log(xa), (np.log(xb) - np.log(xa)) / k, k, lambda y: (np.exp(y), np.exp(y)), om, lam,
                ops)


# ---------------------------------------------------------------- series by complex step
def _solve_periodic_cs(S, kappa, rhs, omega, h):
    """DSSEll.solve_periodic for a complex-step batch (member 0 real, members b > 0 at first order in i h): member 0 by
    one real LU of omega D + kappa_0, members b > 0 as f_0 + i h df_b with (omega D + kappa_0 + lam) df_b = Im(rhs_b)/h
    - Im(kappa_b)/h f_0, one real LU for all of them"""
    B, N = rhs.shape
    om = float(np.real(omega[0]))
    A = om * S._Dm.T + np.diag(np.real(kappa[0]))
    f0 = sla.lu_solve(sla.lu_factor(A), np.real(rhs[0]))
    out = np.empty(rhs.shape, complex)
    out[0] = f0
    if B > 1:
        lu = sla.lu_factor(A + S.lam * np.eye(N)) if S.lam != 0.0 else sla.lu_factor(A)
        r = np.imag(rhs[1:]) / h - (np.imag(kappa[1:]) / h) * f0
        out[1:] = f0 + 1j * h * sla.lu_solve(lu, r.T).T
    return out


def series_cs(S, b, X, om, lam, h=1e-20):
    """block 0 or block 12's starting state at x0 or 1 - u0 (truncated, as _centre_to_node0 / _ssh_to_last before their
    integration) and its tangents, by a complex step through the series"""
    no, ne = S.n_odd, S.n_even
    nf = len(X)
    Xc = np.repeat(X[None, :], nf + 1, axis=0).astype(complex)
    Xc[1:] += 1j * h * np.eye(nf)
    omb = np.full(nf + 1, om, dtype=complex)
    S.lam = lam
    try:
        if b == 0:
            Sx = S.centre_series(S.phys(Xc[:, :no], "odd"), S.phys(Xc[:, no:], "even"), omb)
            xe = S.x0
        else:
            S.solve_periodic = lambda kappa, rhs, omega, bg=None: _solve_periodic_cs(S, kappa, rhs, omega, h)
            try:
                Sx = S.ssh_series(S.phys(Xc[:, :no], "odd"), S.phys(Xc[:, no:no + ne], "even"),
                                  S.phys(Xc[:, no + ne:], "odd"), omb)
            finally:
                del S.solve_periodic
            xe = S.u0
        st = S.trunc(np.stack([S.eval_series(S_, xe) for S_ in Sx]))
    finally:
        S.lam = 0.0
    return np.ascontiguousarray(st[:, :1].real), np.ascontiguousarray(st[:, 1:].imag / h)


def node_start(S, X):
    """a node block's base state and tangents: the coefficient -> grid rows of DSSEll.phys per field"""
    no, ne = S.n_odd, S.n_even
    cut = np.cumsum([0, no, no, ne, ne, no])
    nf = cut[-1]
    s = np.stack([S.phys(X[None, cut[f]:cut[f + 1]], PAR[f]) for f in range(5)])
    ds = np.zeros((5, nf, S.N))
    for f in range(5):
        ds[f, cut[f]:cut[f + 1]] = S._B[PAR[f]]
    return s, ds


def block_lin(S, b, X, om, lam, basis="coef"):
    """block b's end state and its tangents: (nf, blk) coefficients (basis coef) or (5, nf, N) grid values (basis grid)"""
    K = len(S.nodes)
    ops = CoefOps(S) if basis == "coef" else None
    if b == 0 or b == K:
        s, ds = series_cs(S, b, X, om, lam)
        if ops is not None:
            ds = S.tospec(ds)
        if b == 0:
            return log_lin(S, s, ds, S.x0, S.nodes[0], om, lam, ops)
        if S.long:
            return s, ds
        return seg_lin(S, s, ds, 1 - S.u0, S.nodes[-1], om, lam, ops)
    s, ds = node_start(S, X)
    if ops is not None:
        ds = np.eye(len(X))
    return seg_lin(S, s, ds, S.nodes[b - 1], S.nodes[b], om, lam, ops)


def mode_matrix_fast(S, z, lam, only=None, log=None, basis="coef"):
    """DSSEll.mode_matrix(z, lam) (float64, Delta fixed) by the variational map; only: block indices to build (the others
    are left zero, as ell_ddm.DDEll.mode_matrix's only=); basis: the tangents as coefficients (coef) or grid values
    (grid)"""
    sizes, K, L = S.layout()
    nv = L - 1
    Mm = np.zeros((nv, nv))
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    node0 = offs[2]
    om = 2 * np.pi / z[-1]
    lam = float(lam)
    for b, (c0, nf, fun, tgt) in enumerate(blocks):
        if only is not None and b not in only:
            continue
        t0 = time.time()
        _, ds = block_lin(S, b, z[c0:c0 + nf], om, lam, basis)
        R = ds if basis == "coef" else S.tospec(ds)
        r0 = b * blk
        Mm[r0:r0 + blk, c0:c0 + nf] = R.T
        t0_ = node0 + blk * tgt
        Mm[r0:r0 + blk, t0_:t0_ + blk] -= np.eye(blk)
        if log:
            log(f"  block {b}: {time.time() - t0:.1f} s")
    return Mm


def mode_matrix_cs(S, z, lam, only=None, h=1e-20, log=None):
    """DSSEll.mode_matrix restricted to the blocks in only (the reference, unchanged code per block)"""
    sizes, K, L = S.layout()
    nv = L - 1
    Mm = np.zeros((nv, nv))
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    node0 = offs[2]
    D = z[-1]
    S.lam = lam
    try:
        for b, (c0, nf, fun, tgt) in enumerate(blocks):
            if only is not None and b not in only:
                continue
            t0 = time.time()
            X = np.repeat(z[None, c0:c0 + nf], nf + 1, axis=0).astype(complex)
            X[1:] += 1j * h * np.eye(nf)
            omc = np.full(nf + 1, 2 * np.pi / D, dtype=complex)
            R = S.tospec(fun(X, omc))[1:].imag / h
            r0 = b * blk
            Mm[r0:r0 + blk, c0:c0 + nf] = R.T
            t0_ = node0 + blk * tgt
            Mm[r0:r0 + blk, t0_:t0_ + blk] -= np.eye(blk)
            if log:
                log(f"  block {b}: {time.time() - t0:.1f} s")
    finally:
        S.lam = 0.0
    return Mm


# ---------------------------------------------------------------- checks
def compare(S, Mf, Mc, only=None, tol_blk=1e-10, tol_ln=1e-8, out=print):
    """per-block max |M_fast - M_cs| / max |M_cs block| and the slogdet comparison; returns (ok, rows)"""
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    ok = True
    rows = []
    for b, (c0, nf, fun, tgt) in enumerate(blocks):
        if only is not None and b not in only:
            continue
        sl = (slice(b * blk, (b + 1) * blk), slice(c0, c0 + nf))
        d = float(np.max(np.abs(Mf[sl] - Mc[sl])))
        sc = float(np.max(np.abs(Mc[sl])))
        rel = d / sc
        good = rel <= tol_blk
        ok &= good
        rows.append(dict(block=b, rel=rel, maxabs=sc))
        out(f"  block {b:2d}: max |M_fast - M_cs| / max |M_cs| = {rel:.3e} (max |M_cs| {sc:.3e}) {'ok' if good else 'FAIL'}")
    det = None
    if only is None:
        sf, lf = np.linalg.slogdet(Mf)
        sc_, lc = np.linalg.slogdet(Mc)
        good = sf == sc_ and abs(lf - lc) <= tol_ln
        ok &= good
        det = dict(sign_fast=float(sf), ln_fast=float(lf), sign_cs=float(sc_), ln_cs=float(lc), dln=float(abs(lf - lc)))
        out(f"  slogdet: fast {sf:+.0f} {lf:.12f}, complex step {sc_:+.0f} {lc:.12f}, |d ln| {abs(lf - lc):.2e} "
            f"{'ok' if good else 'FAIL'}")
    return ok, dict(blocks=rows, det=det)


def load_system(recf, st, D):
    """DSSEll and state as d18_hybsign.py builds them, with its 'the record system' check (|F| within 1e-2 relative and
    the frozen step total equal to the record's)"""
    rec = [json.loads(l) for l in open(recf) if l.strip() and json.loads(l).get("D") == D][-1]
    z = np.load(st)
    S = DSSEll(kc=rec["kc"], nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=rec["Ks"], u0=rec["u0"],
               rk=rec.get("rk", 1))
    steps = S.freeze_from(z, rec["kappa"])
    with np.errstate(all="ignore"):
        F = float(np.linalg.norm(S.residual(z[None, :])[0]))
    same = abs(F - rec["F"]) <= max(1e-2 * rec["F"], 1e-13) and sum(steps.values()) == rec["steps"]
    msg = (f"D {D} k_c {rec['kc']}: |F| {F:.10e} (record {rec['F']:.10e}), steps {sum(steps.values())} (record "
           f"{rec['steps']}): {'the record system' if same else 'NOT the record system'}")
    return S, z, rec, same, msg


def checkcmd(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    recf, st, D, lam = args[0], args[1], float(args[2]), float(args[3])
    only = [int(v) for v in opts["blocks"].split(",")] if "blocks" in opts else None
    basis = opts.get("basis", "coef")
    S, z, rec, same, msg = load_system(recf, st, D)
    print(msg, flush=True)
    if not same:
        sys.exit(3)
    print(f"N {S.N}, n_odd {S.n_odd}, n_even {S.n_even}, blocks {only or 'all'}, lambda {lam!r}, tangents {basis}, "
          f"mutate {MUTATE or 'none'}",
          flush=True)
    with np.errstate(all="ignore"):
        t0 = time.time()
        Mf = mode_matrix_fast(S, z, lam, only=only, log=lambda s: print("fast" + s, flush=True), basis=basis)
        tf = time.time() - t0
        t0 = time.time()
        Mc = mode_matrix_cs(S, z, lam, only=only, log=lambda s: print("cs  " + s, flush=True))
        tc = time.time() - t0
    print(f"times: fast {tf:.1f} s, complex step {tc:.1f} s, ratio {tc / tf:.2f}", flush=True)
    ok, res = compare(S, Mf, Mc, only)
    print(f"check: {'PASS' if ok else 'FAIL'}", flush=True)
    if "out" in opts:
        open(opts["out"], "w").write(json.dumps(dict(D=D, kc=rec["kc"], lam=lam, only=only, mutate=MUTATE, basis=basis, t_fast=tf,
                                                     t_cs=tc, ok=bool(ok), **res)) + "\n")
    sys.exit(0 if ok else 1)


def _toy_state(kc=15, n=4.0):
    """a smooth parity-respecting state (not a solution) on the standard nodes: enough to compare two derivatives of the
    same discrete map"""
    S = DSSEll(kc=kc, nodes=NODES_STD, n=n, ell=1, Delta_ref=1.0, Ks=6, u0=0.04)
    sizes, K, L = S.layout()
    rng = np.random.default_rng(7)
    parts = []
    for sz, par in zip(sizes, S.parities()):
        ks = S.kodd if par == "odd" else S.keven
        v = rng.standard_normal(sz) * 0.02
        if par == "even":
            v[0] = 1.0 + 0.05 * rng.standard_normal()       # a, c near 1 (c(0) away from the nodes' x)
            kk = np.concatenate([ks[:1], ks[1:], ks[1:]])
        else:
            kk = np.concatenate([ks, ks])
        parts.append(v / (1.0 + kk) ** 2)
    z = np.concatenate(parts + [[1.0]])
    # c at the centre and at the nodes: mean values that keep x - c away from zero
    offs = np.concatenate([[0], np.cumsum(sizes)])
    pars = S.parities()
    for i, (o, par) in enumerate(zip(offs[:-1], pars)):
        if i == 1:
            z[o] = 0.3                                       # c0 mean
    for k in range(K):
        z[offs[2 + 5 * k + 3]] = 0.5 * NODES_STD[k]         # c at node k, mean: x - c stays away from 0 in each segment
    S.freeze_from(z, 0.7)
    return S, z


def selftest():
    S, z = _toy_state()
    ok = True
    om = 2 * np.pi / z[-1]
    lam = 0.7
    # 1. the linearised right side against a central difference of DSSEll.rhs (lam on the perturbation's D)
    rng = np.random.default_rng(3)
    x = 0.37
    s = np.stack([S.phys(rng.standard_normal((1, S.n_odd if PAR[f] == "odd" else S.n_even)) * 0.05, PAR[f])
                  for f in range(5)])
    s[2] += 1.0
    s[3] += 0.2
    ds = S.trunc(rng.standard_normal((5, 3, S.N)))
    base, dk = rhs_lin(S, x, s, ds, om, lam)
    eps = 1e-6
    S.lam = lam
    try:
        Bp = np.concatenate([s, s + eps * ds, s - eps * ds], axis=1)   # member 0 the background (lambda (f - f0))
        R = S.rhs(x, Bp, np.full(7, om))
    finally:
        S.lam = 0.0
    fd = (R[:, 1:4] - R[:, 4:7]) / (2 * eps)
    e1 = float(np.max(np.abs(fd - dk)) / np.max(np.abs(dk)))
    e0 = float(np.max(np.abs(R[:, :1] - base)) / np.max(np.abs(base)))
    g = e1 <= 1e-7 and e0 <= 1e-13
    ok &= g
    print(f"rhs_lin: base against DSSEll.rhs {e0:.2e} (<= 1e-13), tangents against a central difference {e1:.2e} (<= 1e-7) "
          f"{'ok' if g else 'FAIL'}", flush=True)
    # 1b. coefficient tangents against the grid ones on parity-respecting tangents
    blk = 3 * S.n_odd + 2 * S.n_even
    dc = rng.standard_normal((3, blk)) * 0.1
    ops = CoefOps(S)
    k = ops.cut
    dsg = np.stack([S.phys(dc[:, k[f]:k[f + 1]], PAR[f]) for f in range(5)])
    _, dkg = rhs_lin(S, x, s, dsg, om, lam)
    _, dkc = rhs_lin_coef(S, x, s, dc, om, lam, None, ops)
    e2 = float(np.max(np.abs(S.tospec(dkg) - dkc)) / np.max(np.abs(dkc)))
    g = e2 <= 1e-13
    ok &= g
    print(f"rhs_lin_coef against rhs_lin on parity-respecting tangents {e2:.2e} (<= 1e-13) {'ok' if g else 'FAIL'}",
          flush=True)
    # 2. the whole matrix at two lambda, both tangent bases
    for lam in (0.0, 0.9):
        with np.errstate(all="ignore"):
            Mc = mode_matrix_cs(S, z, lam)
            for basis in ("coef", "grid"):
                Mf = mode_matrix_fast(S, z, lam, basis=basis)
                print(f"M at lambda {lam}, {basis} tangents:", flush=True)
                g, _ = compare(S, Mf, Mc)
                ok &= g
    print(f"selftest (mutate {MUTATE or 'none'}): {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        checkcmd(sys.argv[2:])
    elif cmd == "selftest":
        selftest()
    else:
        print(__doc__)
        sys.exit(2)
