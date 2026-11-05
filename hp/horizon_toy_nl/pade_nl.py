#!/usr/bin/env python3
"""F4: the series formulation with the horizon series continued past its pole by a robust Pade approximant.

Robust Pade by SVD: Gonnet, Guettel, Trefethen, SIAM Rev. 55 (2013) 101-117 (the degree is lowered until the
Toeplitz block has full numerical rank at tolerance tol, which removes Froissart doublets). Applied to the scaled
coefficients c_k u0^k and evaluated at 1. The b_h column of the Jacobian by central differences.

    python3 pade_nl.py DPS S [L] [deltas...]      (planted test, the gamma control, then test-B starts)
"""
import random
import sys
import time

sys.path.insert(0, ".")
import mpmath as mp  # noqa: E402
from mpmath import mpf  # noqa: E402

import toy_nl as tn  # noqa: E402


def robust_pade(c, m, n, tol):
    """GGT 2013: numerator a (degree <= m), denominator b (degree <= n), b normalised to b_0 = 1"""
    c = list(c[: m + n + 1]) + [mpf(0)] * max(0, m + n + 1 - len(c))
    nrm = mp.sqrt(mp.fsum(x * x for x in c))
    if all(abs(x) <= tol * nrm for x in c[: m + 1]):
        return [mpf(0)], [mpf(1)]
    while True:
        if n == 0:
            return c[: m + 1], [mpf(1)]
        Z = mp.matrix(m + n + 1, n + 1)
        for i in range(m + n + 1):
            for j in range(n + 1):
                if i - j >= 0:
                    Z[i, j] = c[i - j]
        Ct = mp.matrix(n, n + 1)
        for i in range(n):
            for j in range(n + 1):
                Ct[i, j] = Z[m + 1 + i, j]
        U, S, V = mp.svd_r(Ct, full_matrices=True)
        rho = sum(1 for k in range(len(S)) if S[k] > tol * nrm)
        if rho == n:
            b = [V[n, j] for j in range(n + 1)]       # right singular vector of the smallest singular value
            break
        m, n = m - (n - rho), rho
    a = [mp.fsum(Z[i, j] * b[j] for j in range(n + 1)) for i in range(m + 1)]
    # normalise; b_0 may vanish only in degenerate cases
    if abs(b[0]) < tol:
        raise ZeroDivisionError("degenerate denominator")
    a = [x / b[0] for x in a]
    b = [x / b[0] for x in b]
    return a, b


def pade_value(coef, u, L, tol):
    sc = [coef[k] * u ** k for k in range(2 * L + 1)]
    a, b = robust_pade(sc, L, L, tol)
    return mp.fsum(a) / mp.fsum(b), len(b) - 1


def res_pade(T, L):
    tol = mpf(10) ** (-T.dps + 6)

    def ev(bh):
        B, Yv, dB, dY, nt, conv, be, et = T.hseries(bh, T.u0, tang=False, kfix=2 * L + 1, coeffs=True)
        pb, nb = pade_value(be, T.u0, L, tol)
        py, ny = pade_value(et, T.u0, L, tol)
        return pb, py, nb, ny

    def F(z):
        bh, b0, y0 = z
        ph, pb_, py_ = T.phi(b0, y0)
        try:
            B, Yv, nb, ny = ev(bh)
            h = abs(bh) * mpf(10) ** (-T.dps // 3)
            Bp, Yp, _, _ = ev(bh + h)
            Bm, Ym, _, _ = ev(bh - h)
        except (ZeroDivisionError, ValueError):
            big = mpf(10) ** 50
            return [big, big, ph], [[1, 0, 0], [0, 1, 0], [0, pb_, py_]], dict(ok=False)
        dB, dY = (Bp - Bm) / (2 * h), (Yp - Ym) / (2 * h)
        return ([B - b0, Yv - y0, ph], [[dB, mpf(-1), mpf(0)], [dY, mpf(0), mpf(-1)], [mpf(0), pb_, py_]],
                dict(nb=nb, ny=ny))
    return F


def main():
    dps, s = int(sys.argv[1]), sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    dels = sys.argv[4:]
    mp.mp.dps = dps
    for mut in (None, "gamma"):
        T = tn.Toy(s, dps, mutate=mut)
        F = res_pade(T, L)
        zs = [T.ref["bh"], T.ref["b0"], T.ref["y0"]]
        z, nF, h = tn.newton(F, [+x for x in zs], dps)
        err = max(abs(z[1] - T.ref["b0"]), abs(z[2] - T.ref["y0"]))
        print(f"pade L {L} dps {dps} s {s} {'planted' if mut is None else 'control ' + mut}: err {mp.nstr(err, 3)} "
              f"hist {[round(x, 1) for x in h]}", flush=True)
    T = tn.Toy(s, dps)
    F = res_pade(T, L)
    Fs = tn.res_series(T)
    zs = [T.ref["bh"], T.ref["b0"], T.ref["y0"]]
    for dl in dels:
        d = mpf(10) ** (-mpf(dl))
        for seed in (1, 2, 3):
            rnd = random.Random(seed)
            r = [rnd.choice((-1, 1)) * rnd.uniform(0.5, 1.0) for _ in zs]
            z0 = [zs[i] * (1 + d * mpf(r[i])) for i in range(3)]
            for name, FF in (("pade", F), ("series", Fs)):
                t = time.time()
                z, nF, h = tn.newton(FF, z0, dps, maxit=15)
                err = max(abs(z[1] - T.ref["b0"]), abs(z[2] - T.ref["y0"]))
                print(f"  {name}: test B delta 1e-{dl} seed {seed} (b_h {'up: real pole' if r[0] > 0 else 'down: complex pair'}): "
                      f"err {mp.nstr(err, 3)} hist {[round(x, 1) for x in h]} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
