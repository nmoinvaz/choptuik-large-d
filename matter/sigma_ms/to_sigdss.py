#!/usr/bin/env python3
"""A sigma_ms kernel state as a sigdss_solver.py state, to seed sigdss from the kernel's chain (#846).

    python3 matter/sigma_ms/to_sigdss.py STATE.npz OUT.npy [--kc 15] [--D D --eps EPS]
    python3 matter/sigma_ms/to_sigdss.py --selftest [--mutate=shift|parity]

The kernel's state (smdriver.save's npz: zh + zl, the M-point half grid theta_j = pi j / M, smdriver.NODES) and sigdss's
(Fourier to harmonic kc, solver/matter/sigdss_solver.NODES_STD) hold the same fields: the centre data (q, c0), at each
node (p, m, a, c, psi), the horizon data (m, a, psi), then Delta. sigdss's 12 nodes are all among the kernel's 21, so
the map picks those node blocks and takes each field's harmonics k <= kc from the half grid, extended by parity
(f(theta + pi) = -f(theta) for p, m, psi, q and the horizon m and psi; +f for the rest). No integration. Harmonics
above kc are dropped, which is sigdss's truncation. With --D and --eps it prints sigdss's |F| (2-norm, as
sigdss_run.py) at the converted state.

--selftest: the round trip of sigdss's own D 9.0 solution at eta 1 (solver/matter/data/sigdss_m1up.npz, the first
entry): its fields are put on a kernel half grid of M = 32 by smdriver.resample's interpolation, with NaN at the
kernel's nine nodes that sigdss lacks, and mapped back. It passes when the result equals the solution to 1e-12
relative (max |dz| / max |z|). Controls (each must fail): --mutate=shift reads the half grid a quarter step late in
tau, --mutate=parity extends every field as even.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MATTER = os.path.normpath(os.path.join(HERE, "..", "..", "solver", "matter"))
sys.path.insert(0, MATTER)
from sigdss_solver import NODES_STD, SigDSS, make  # noqa: E402

# smdriver.NODES (smdriver imports the Rust module sigmams, so it is not imported here)
KNODES = [0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
          0.9, 0.95, 0.98]
FPAR = ("odd", "odd", "even", "even", "odd")           # p, m, a, c, psi
CPAR, HPAR = ("odd", "even"), ("odd", "even", "odd")   # (q, c0), horizon (m, a, psi)
BAR = 1e-12


def harmonics(h, parity, kc, mutate=""):
    """half-grid samples h (theta_j = pi j / M) of a field of the given parity -> its complex Fourier coefficients
    c_k, k = 0..kc, in sigdss's normalization f = c_0 + 2 Re sum_k c_k e^{i k theta}"""
    M = len(h)
    odd = parity == "odd" and mutate != "parity"
    c = np.fft.rfft(np.concatenate([h, -h] if odd else [h, h])) / (2 * M)
    if kc >= M:
        raise SystemExit(f"to_sigdss.py: kc {kc} needs M > kc (the state has M = {M})")
    k = np.arange(kc + 1)
    if mutate == "shift":
        c = c * np.exp(1j * np.arange(len(c)) * np.pi / (4 * M))     # the samples read a quarter step late
    return c[k]


def spec_block(c, parity, kc):
    """sigdss's spectral block of one field from its coefficients c_0..c_kc (sigdss_solver/spec.py layout)"""
    if parity == "odd":
        ks = np.arange(1, kc + 1, 2)
        return np.concatenate([c[ks].real, c[ks].imag])
    ks = np.arange(2, kc + 1, 2)
    return np.concatenate([[c[0].real], c[ks].real, c[ks].imag])


def kernel_blocks(z, K=len(KNODES)):
    """the kernel's z (f64) -> (M, (q, c0), node states (K, 5, M), horizon (3, M), Delta)"""
    L = len(z)
    M = (L - 1) // (5 * (K + 1))
    if 5 * M * (K + 1) + 1 != L:
        raise SystemExit(f"to_sigdss.py: a state of length {L} is not 5 M (K + 1) + 1 with K = {K}")
    q, c0 = z[:M], z[M:2 * M]
    st = z[2 * M:2 * M + 5 * M * K].reshape(K, 5, M)
    o = 2 * M + 5 * M * K
    return M, (q, c0), st, z[o:o + 3 * M].reshape(3, M), z[-1]


def convert(z, kc, mutate=""):
    """the kernel's z (f64, zh + zl) -> sigdss's z at harmonic kc on NODES_STD"""
    M, cen, st, hor, Delta = kernel_blocks(z)
    out = [spec_block(harmonics(f, p, kc, mutate), p, kc) for f, p in zip(cen, CPAR)]
    for x in NODES_STD:
        i = KNODES.index(x)
        out += [spec_block(harmonics(st[i, f], FPAR[f], kc, mutate), FPAR[f], kc) for f in range(5)]
    out += [spec_block(harmonics(f, p, kc, mutate), p, kc) for f, p in zip(hor, HPAR)]
    return np.concatenate(out + [[Delta]])


def load_kernel(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return np.asarray(d["zh"], float) + np.asarray(d["zl"], float), meta


def resample(F, M):
    """smdriver.resample: full-grid samples F (any N0) -> half-grid samples of the N = 2M grid (trigonometric
    interpolation)"""
    N0 = len(F)
    c = np.fft.rfft(F) / N0
    th = 2 * np.pi * np.arange(M) / (2 * M)
    k = np.arange(len(c))
    w = np.where((k == 0) | ((N0 % 2 == 0) & (k == N0 // 2)), 1.0, 2.0)
    return (w * c[None, :] * np.exp(1j * k[None, :] * th[:, None])).real.sum(1)


def selftest(mutate):
    d = np.load(os.path.join(MATTER, "data", "sigdss_m1up.npz"))
    D, zs = float(d["v"][0]), np.asarray(d["Z"][0], float)
    S = SigDSS(kc=15, nodes=NODES_STD, n=D - 2, eps=1.0, Delta_ref=zs[-1])
    f = S.fields(zs)
    M, K = 32, len(KNODES)
    st = np.full((K, 5, M), np.nan)                     # the kernel's nodes that sigdss lacks stay NaN
    for i, x in enumerate(NODES_STD):
        st[KNODES.index(x)] = [resample(f["states"][i][j], M) for j in range(5)]
    zk = np.concatenate([resample(f["d"], M), resample(f["c0"], M), st.ravel()]
                        + [resample(f[k], M) for k in ("mS", "aS", "qS")] + [[f["Delta"]]])
    zr = convert(zk, 15, mutate)
    rel = float(np.max(np.abs(zr - zs)) / np.max(np.abs(zs))) if np.all(np.isfinite(zr)) else float("inf")
    ok = rel <= BAR
    print(f"to_sigdss selftest{' --mutate=' + mutate if mutate else ''}: sigdss's D {D} solution (kc 15, {len(zs)} "
          f"unknowns) through a kernel half grid of M {M} and back: max |dz| / max |z| = {rel:.3e} (bar {BAR:g}): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv):
    if argv and argv[0] == "--selftest":
        return selftest(next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--mutate=")), ""))
    if len(argv) < 2:
        raise SystemExit(__doc__)
    src, out = argv[0], argv[1]
    opt = dict(kc="15", D="", eps="")
    for k in opt:
        if "--" + k in argv:
            opt[k] = argv[argv.index("--" + k) + 1]
    z, meta = load_kernel(src)
    kc = int(opt["kc"])
    zs = convert(z, kc)
    np.save(out, zs)
    M = kernel_blocks(z)[0]
    msg = f"to_sigdss {os.path.basename(src)}: M {M}, Delta {float(z[-1])!r} -> {out}: kc {kc}, {len(zs)} unknowns"
    if opt["D"] and opt["eps"]:
        S = make(float(opt["D"]), float(opt["eps"]), zs, kc=kc)
        msg += f"; sigdss |F| at D {opt['D']}, eps {opt['eps']}: {float(np.linalg.norm(S.residual(zs[None, :])[0])):.4e}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
