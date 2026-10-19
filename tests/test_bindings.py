"""Bindings test: a synthetic production-sized configuration (36 folded tau points, 941 unknowns)
must give bitwise the same residual on every backend this CPU runs, and the exact Jacobian must match
complex-step columns. The matrices are random, so the values are only good for these comparisons.

    pip install . && python tests/test_bindings.py [--time]
"""
import sys
import time

import numpy as np
import rustkernel as rk


def config(seed=12345):
    rng = np.random.default_rng(seed)
    nh, no, ne = 36, 23, 24
    nodes = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]
    kn = len(nodes)
    sizes = [no, ne] + [no, no, ne, ne] * kn + [no, ne]

    def mat(r, c, s, d=0.0):
        a = s * rng.uniform(-1, 1, (r, c))
        a[: min(r, c), : min(r, c)] += d * np.eye(min(r, c))
        return np.ascontiguousarray(a)

    phys_even = mat(ne, nh, 0.5)
    phys_even[0, :] = 1.0  # constant mode, so coefficient 0 sets the level of a and c
    seg_u = [min(a, b) >= 0.4 for a, b in zip(nodes[:-1], nodes[1:])]
    args = (3.0, nh, mat(nh, nh, 0.05), mat(nh, nh, 0.02, 1.0), mat(nh, nh, 0.02, 1.0), mat(no, nh, 0.5),
            phys_even, mat(nh, no, 0.1), mat(nh, ne, 0.1), 0.01, nodes, 40, [20] * (kn - 1), seg_u, 0.2, 23, 60,
            sizes, 3)
    offs = np.cumsum([0] + sizes)
    z = 0.05 * rng.uniform(-1, 1, offs[-1] + 1)
    z[offs[1]] = 2.0
    for i in range(kn):
        z[offs[2 + 4 * i + 2]] = 1.0
        z[offs[2 + 4 * i + 3]] = 2.0
    z[offs[len(sizes) - 1]] = 1.0
    z[-1] = 3.44
    zb = np.repeat(z[None, :], 4, axis=0)
    zb[1, offs[6]] += 1e-3
    zb[3, -1] += 1e-3
    return args, zb


def main():
    print(rk.build_info())
    backends = rk.available_backends()
    print("backends:", backends)
    args, zb = config()
    k = rk.Kernel(*args)
    assert zb.shape[1] == 941
    want = k.residual(zb, 1, "baseline")
    assert np.isfinite(want).all()
    for be in backends:
        for threads, dedup in [(1, False), (2, True)]:
            got = k.residual(zb, threads, be, dedup)
            assert np.array_equal(want.view(np.uint64), got.view(np.uint64)), f"{be} threads={threads} dedup={dedup}"
    assert rk.simd_available() == any(":" not in b and b != "baseline" for b in backends)
    # complex division and multiplication round differently from f64, so only to rounding level
    fc = k.residual_complex(zb.astype(complex) + 1e-20j)
    assert np.allclose(fc.real, want, rtol=1e-10, atol=0)
    print("ok: all backends bitwise equal")
    # exact Jacobian (forward mode) against complex-step columns, and the growth-mode matrix
    z = np.ascontiguousarray(zb[0])
    J = k.jacobian(z, 2)
    L = z.size
    assert J.shape == (L, L)
    cols = list(range(0, L, 37)) + [L - 1]
    zc = np.repeat(z[None, :].astype(complex), len(cols), axis=0)
    zc[np.arange(len(cols)), cols] += 1e-30j
    cs = (k.residual_complex(zc).imag / 1e-30).T
    assert np.allclose(J[:, cols], cs, rtol=0, atol=1e-9 * np.abs(J).max()), "jacobian vs complex step"
    M0 = k.mode_matrix(z, 0.0, 2)
    assert np.array_equal(M0, J[:-1, :-1]), "mode matrix at lam = 0 is the Jacobian block"
    M1 = k.mode_matrix(z, 0.5, 2)
    assert M1.shape == (L - 1, L - 1) and not np.array_equal(M1, M0)
    print("ok: jacobian matches complex step; mode matrix consistent")
    if "--time" in sys.argv:
        for be in backends:
            t = []
            for _ in range(20):
                t0 = time.perf_counter()
                k.residual(zb[:1], 1, be)
                t.append(time.perf_counter() - t0)
            print(f"{be:>12}: {np.median(t) * 1e3:7.3f} ms per residual")


if __name__ == "__main__":
    main()
