#!/usr/bin/env python3
"""The mode count of notes/MODE_COUNT.md with the f64 solver (solver/: DSSSpec, spectral in tau, RK4 in x), at D = 4,
where no double-double state exists (hp/data/states start at D = 8.25).

    python3 modecount_f64.py [STATE.npy] [--D=4] [--kc=39] [--Ks=60] [--eps=1e-6] [--check] [count options as modecount_wind.py]

Default state: solver/data/neB_kc39_D4.0000.npy (kc 39, nodes 0.02 ... 0.6, horizon series Ks = 60 at u = 0.4, the
layout of solver/rust_solver.make). det M(lambda) at complex lambda: the Python reference's mode matrix
(DSSSpec.mode_matrix: central differences, eps = 1e-6, Delta fixed, no phase row; every tau-derivative of a perturbation
gains + lambda) with complex arithmetic throughout (solver/fast.py's matrix forms of d/dtau, the truncation and the
periodic solves; phys / spec applied to real and imaginary parts), then numpy's complex slogdet.

--check: at real lambda, against the Rust f64 kernel's exact mode matrix (rustkernel.RustResidual.mode_matrix), the
Cauchy-Riemann test, and the real zeros near lam_g and lam_phys; no count.

The periodic solves of the Python reference work on the full tau grid (both parities), so besides the folded
layout's resonances (Im = odd multiples of omega) the matrices omega D + diag kappa_kk + lambda are also singular at
Im = even multiples of omega; there the odd right-hand side has no component (in exact arithmetic) and det M has no
pole. Both families are listed as possible poles (an upper bound on the zero count only gets looser).
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, "..", "solver"))
sys.path.insert(0, HERE)
sys.path.insert(0, SOLVER)
import fast  # noqa: E402

fast.install()
from spec_long import make_long  # noqa: E402
from spec_c import freeze_steps, _cplx  # noqa: E402
from modecount_wind import count, parse, DATA  # noqa: E402

NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def make(D, z, kc, Ks):
    S = freeze_steps(make_long(kc, D - 2, nodes=NODES_L, Ks=Ks), z[-1])
    phys0, spec0 = S.phys, S.spec
    S.phys = _cplx(lambda c, par: phys0(c, par))
    S.spec = _cplx(lambda f, par: spec0(f, par))
    return S


def mode_matrix_c(S, z, lam, eps=1e-6):
    """DSSSpec.mode_matrix with complex lam (central differences in each block's inputs, Delta fixed)."""
    sizes, K, L = S.layout()
    offs = np.concatenate([[0], np.cumsum(sizes)])
    parts, Dl = S.split(z[None, :])
    parts = [p[0] for p in parts]
    D = float(Dl[0])
    pars = S.parities()
    nv = L - 1
    Mx = np.zeros((nv, nv), complex)
    blk = sum(sizes[2:6])
    node0 = offs[2]

    def batch(vecs):
        base = np.concatenate(vecs)
        nf = len(base)
        X = np.repeat(base[None, :], 1 + 2 * nf, axis=0).astype(complex)
        X[1:1 + nf] += eps * np.eye(nf)
        X[1 + nf:] -= eps * np.eye(nf)
        return X, np.full(len(X), 2 * np.pi / D), nf

    def tospec(s):
        return np.concatenate([S.spec(s[0], "odd"), S.spec(s[1], "odd"), S.spec(s[2], "even"), S.spec(s[3], "even")], axis=1)

    def put(b, c0, R, nf):
        dR = ((R[1:1 + nf] - R[1 + nf:]) / (2 * eps)).T
        r0 = b * blk
        Mx[r0:r0 + blk, c0:c0 + nf] = dR
        tgt = node0 + blk * (b if b < K else K - 1)
        Mx[r0:r0 + blk, tgt:tgt + blk] -= np.eye(blk)

    S.lam = lam
    try:
        X, om, nf = batch([parts[0], parts[1]])
        no = sizes[0]
        s = S._centre_to_node0(S.phys(X[:, :no], "odd"), S.phys(X[:, no:], "even"), om)
        put(0, 0, tospec(s), nf)
        for i in range(K - 1):
            j = 2 + 4 * i
            X, om, nf = batch(parts[j:j + 4])
            o = np.concatenate([[0], np.cumsum(sizes[j:j + 4])])
            st = np.stack([S.phys(X[:, o[q]:o[q + 1]], pars[j + q]) for q in range(4)])
            s = S.seg(st, S.nodes[i], S.nodes[i + 1], om)
            put(i + 1, offs[j], tospec(s), nf)
        X, om, nf = batch([parts[-2], parts[-1]])
        s = S._ssh_to_last(S.phys(X[:, :sizes[-2]], "odd"), S.phys(X[:, sizes[-2]:], "even"), om)
        put(K, offs[len(sizes) - 2], tospec(s), nf)
    finally:
        S.lam = 0.0
    return Mx


def horizon_data(S, z):
    sizes, K, L = S.layout()
    parts, Dl = S.split(z[None, :])
    aS = S.phys(parts[-1][None, :] if parts[-1].ndim == 1 else parts[-1], "even")[0].real
    return aS, float(Dl[0])


def main():
    args, opts = parse(sys.argv[1:])
    path = args[0] if args else os.path.join(SOLVER, "data", "neB_kc39_D4.0000.npy")
    D = float(opts.get("D", 4.0))
    kc = int(opts.get("kc", 39))
    Ks = int(opts.get("Ks", 60))
    eps = float(opts.get("eps", 1e-6))
    z = np.load(path)
    S = make(D, z, kc, Ks)
    n = D - 2
    F = np.abs(S.residual(z[None, :])[0]).max()
    aS, Delta = horizon_data(S, z)
    omega = 2 * np.pi / Delta
    c1 = (n - 1) * (1 - aS ** 2)
    lam_g = 1 + float(np.mean(c1))
    tag = opts.get("tag", f"f64_D{D:.4f}_kc{kc}")
    logf = open(os.path.join(DATA, f"modecount_{tag}.log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    def det_fn(re, im):
        Mx = mode_matrix_c(S, z, complex(re, im), eps)
        s, la = np.linalg.slogdet(Mx)
        return float(la), float(np.angle(s)), float("nan"), float("nan")

    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {os.path.basename(path)}: D {D}, kc {kc}, Ks {Ks}, N {S.N}, "
        f"unknowns {S.layout()[2]}; |F| {F:.1e}; Delta {Delta:.10f}, omega {omega:.6f}; lam_g = 1 + <c1> = {lam_g:.10f}")
    # possible poles: eigenvalues of -(omega D + diag kappa_kk) on the full grid (both parities)
    Dm = fast._orig["_D_unit"](S, np.eye(S.N)).T if fast._orig else S.Dmat1.T
    Dm = S.Dmat1.T
    poles = []
    for kk in range(Ks + 1):
        kap = (c1 + 1) * (kk + 1) - n / 2
        ev = np.linalg.eigvals(omega * Dm + np.diag(kap))
        poles += [-complex(e) for e in ev]
    if "check" in opts:
        check(S, z, D, lam_g, det_fn, log, tag)
        return
    from gamma_dd import gamma_formula
    lam_f = 1.0 / gamma_formula(D)
    info = dict(D=D, M=None, kc=kc, Ks=Ks, F=float(F), state=os.path.basename(path), solver="f64 (solver/ DSSSpec, complex lambda)",
                Delta=Delta, omega=omega, lam_g=lam_g, lam_f=lam_f)
    count(det_fn, info, poles, opts, log, tag, opts.get("res", os.path.join(DATA, "modecount_results.jsonl")))


def check(S, z, D, lam_g, det_fn, log, tag):
    from rustkernel import RustResidual
    from scipy.optimize import brentq
    from gamma_dd import gamma_formula
    R = RustResidual(S, threads=1)
    rec = dict(D=D, tag=tag, lam_g=lam_g)
    rows = []
    for lam in (0.3, 1.7, 4.0):
        t = time.time()
        Mr = R.mode_matrix(z, lam)
        sr, lr = np.linalg.slogdet(Mr)
        t1 = time.time() - t
        t = time.time()
        Mc = mode_matrix_c(S, z, complex(lam, 0.0))
        t2 = time.time() - t
        sc, lc = np.linalg.slogdet(Mc)
        dm = np.abs(Mc - Mr).max() / np.abs(Mr).max()
        rows.append(dict(lam=lam, lndet_rust=lr, sign_rust=sr, lndet_py=lc, arg_py=float(np.angle(sc)), max_rel_diff=dm))
        log(f"  lam {lam}: Rust (exact duals) sign {sr:+.0f} ln|det| {lr:.10f} ({t1:.1f} s) | Python complex FD ln|det| {lc:.10f} "
            f"arg {np.angle(sc):+.2e} ({t2:.1f} s); max |dM| / max |M| = {dm:.1e}")
    rec["real_lambda"] = rows
    cr = []
    for lam in (1.3, 3.0):
        h = 1e-4
        l0, a0, _, _ = det_fn(lam, 0.0)
        lx, ax, _, _ = det_fn(lam + h, 0.0)
        ly, ay, _, _ = det_fn(lam, h)
        dx = complex(lx - l0, math.remainder(ax - a0, 2 * math.pi)) / h
        dy = complex(ly - l0, math.remainder(ay - a0, 2 * math.pi)) / (1j * h)
        cr.append(dict(lam=lam, h=h, along_h=[dx.real, dx.imag], along_ih=[dy.real, dy.imag], rel_diff=abs(dx - dy) / abs(dx)))
        log(f"  d ln det / d lam at {lam}: along h {dx:.6f}, along i h {dy:.6f}, rel. diff {abs(dx - dy) / abs(dx):.1e}")
    rec["cauchy_riemann"] = cr

    def f(lam):
        s, la = np.linalg.slogdet(R.mode_matrix(z, lam))
        return s * math.exp(la - ref)

    zeros = {}
    for name, c, w in (("gauge", lam_g, 0.03), ("phys", 1 / gamma_formula(D), 0.01)):
        ref = np.linalg.slogdet(R.mode_matrix(z, c))[1]
        a, b = c - w, c + w
        if f(a) * f(b) > 0:
            zeros[name] = None
            log(f"  no sign change of the Rust det in [{a:.4f}, {b:.4f}] around {name} {c:.6f}")
            continue
        zeros[name] = brentq(f, a, b, xtol=1e-12, rtol=1e-14)
        log(f"  real zero near {name} value {c:.10f}: {zeros[name]:.10f} (diff {zeros[name] - c:+.1e})")
    rec["real_zeros"] = zeros
    with open(os.path.join(DATA, "modecount_check.jsonl"), "a") as fo:
        fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
