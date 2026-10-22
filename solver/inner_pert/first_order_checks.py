"""Checks of the first-order machinery (notes/FIRST_ORDER.md, section 2). Each prints a deviation that should be at
roundoff (or at the finite-difference / truncation level stated).

  series     first_order_jet's background series = solver/inner_borel/inner_series.py (flint), M = 3;
             perturbation series = solver/inner_pert/pert.py (flint), M = 5, lambda = 1.7
  outer      the O(1/n) inner series (1/n direction: O(1/n) terms, K1 = 0, gauge c1(0) = -F^2) = the next-to-top
             degree coefficients of the outer 1/n expansion (solver/largeD_late_orders.py), F = 2.72 sin, w_inf
  jumps      the value jump = inner_np's (M = 3); the A and omega jet directions = centred finite differences
  symmetry   the O(1/n) part of the lambda = 2 mode two ways: the direct hyper-dual (perturbation e at lambda = 2
             with Proposition 5's profile, times the 1/n direction n) against D_P Z_1 + W, W forced by the
             s-translation breaking (SymForce); untruncated harmonics (M = 25), F = 2.72 sin + 0.15 sin 3t
  outer_pert the O(1/n) inner series of a perturbation (hyper-dual e|n) = the next-to-top degree coefficients of
             the outer 1/n expansion of the perturbation equations at lambda (run here to order 4, sympy-free)

    python3 first_order_checks.py [series] [outer] [jumps] [symmetry]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
sys.path.insert(0, os.path.join(HERE, "..", "inner_borel"))
sys.path.insert(0, os.path.join(HERE, ".."))
from first_order_jet import JetModel, Dir, Fh_from, dFh, W0, F2h, SymForce  # noqa: E402


def conv(a, b, L, sp=None):
    c = np.convolve(a, b)[L:3 * L + 1]
    return c * sp if sp is not None else c


def check_series():
    from inner_series import series
    from pert import series_pert
    from inner_stokes import to_mp
    from flint import acb
    M = 3
    x = np.array([2.7265, 1.0156, 0.0, 0.0])
    F1 = x[0] / 2j; Fn = F1 * (0.0530 + 0.0041j); x[2], x[3] = -2 * Fn.imag, 2 * Fn.real
    Fh = Fh_from(x, M)
    w = W0 * x[1]
    jm = JetModel(M, w, Fh, [], [0], {})
    jm.series_at(1.0, 31, stop=False)
    d, s, g = series(Fh, w, 30, M + 1, prec=300)
    L = jm.L
    err = 0
    for j in range(1, 31):
        a = jm.ser["D"][j][0][L - M:L + M + 1]
        b = np.array([complex(to_mp(d[j][mu + M + 1])) for mu in range(-M, M + 1)])
        err = max(err, np.abs(a - b).max() / np.abs(b).max())
    print(f"series: background d_j, j <= 30, M = 3: max relative deviation from inner_series (flint) {err:.1e}")
    M = 5
    A, B = 2.72, 0.15
    Fh = {1: -0.5j * A, -1: 0.5j * A, 3: -0.5j * B, -3: 0.5j * B}
    f = {1: 0.3 + 0.2j, -1: 0.3 - 0.2j, 3: 0.05j, -3: -0.05j}
    lam = 1.7
    jm = JetModel(M, W0, Fh, [Dir("pert", lam=lam)], [0, 1], {1: (f, {})})
    jm.series_at(1.0, 21, stop=False)
    Fa = {k: acb(v.real, v.imag) for k, v in Fh.items()}
    fa = {k: acb(v.real, v.imag) for k, v in f.items()}
    d, a, g, h = series_pert(Fa, fa, str(W0), str(lam), 20, M + 1, prec=300)
    L = jm.L
    err = 0
    for j in range(1, 21):
        for X, Y in ((jm.ser["D"][j][1], a[j]), (jm.ser["G"][j][1], h[j])):
            ref = np.array([complex(to_mp(Y[mu + M + 1])) for mu in range(-M, M + 1)])
            err = max(err, np.abs(X[L - M:L + M + 1] - ref).max() / np.abs(ref).max())
    print(f"series: perturbation a_j, h_j, j <= 20, M = 5, lambda = 1.7: max relative deviation from pert.py {err:.1e}")


def check_outer():
    import largeD_late_orders as lo
    K = 9
    th = 2 * np.pi * np.arange(lo.NT) / lo.NT
    S, D, G, _, _ = lo.run(2.72 * np.sin(th), W0, K, verbose=False)
    harm = lambda f, mu: np.fft.fft(f)[mu % lo.NT] / lo.NT
    M = 2 * K + 5
    Fh = {1: -0.5j * 2.72, -1: 0.5j * 2.72}
    g1 = F2h(Fh, M)
    jm = JetModel(M, W0, Fh, [Dir("n", q1=True)], [0, 1], {1: ({}, {k: -v for k, v in g1.items()})})
    jm.series_at(1.0, K + 1, stop=False)
    L = jm.L
    worst = 0
    for j in range(0, K - 1):
        ey = max(abs(jm.ser["D"][j][1][mu + L] - harm(D[j + 1][2 * j + 1], mu)) for mu in range(-2 * j - 3, 2 * j + 4, 2))
        es = max(abs(jm.ser["S"][j][1][mu + L] - harm(S[j + 2][2 * j + 2], mu)) for mu in range(-2 * j - 5, 2 * j + 6, 2))
        eg = max(abs(jm.ser["G"][j][1][mu + L] + harm(G[j + 1][2 * j], mu)) for mu in range(-2 * j - 4, 2 * j + 5, 2))
        sc = max(np.abs(jm.ser["D"][j][1]).max(), np.abs(jm.ser["S"][j][1]).max(), np.abs(jm.ser["G"][j][1]).max())
        worst = max(worst, max(ey, es, eg) / sc)
        print(f"outer: order {j}: |y1 - D_(j+1)[x^(2j+1)]| {ey:.1e}, |sigma1 - S_(j+2)[x^(2j+2)]| {es:.1e}, "
              f"|c1 + G_(j+1)[x^(2j)]| {eg:.1e}  (scale {sc:.1e})")
    print(f"outer: O(1/n) inner series = next-to-top outer coefficients, worst relative deviation {worst:.1e}")


def check_jumps():
    import inner_np as I
    M = 3
    x = np.array([2.7265, 1.0156, 0.0, 0.0])
    F1 = x[0] / 2j; Fn = F1 * (0.0530 + 0.0041j); x[2], x[3] = -2 * Fn.imag, 2 * Fn.real
    jm = JetModel(M, W0 * x[1], Fh_from(x, M), [Dir("A"), Dir("w", omega=W0)], [0, 1, 2], {1: (dFh(0, M), {})})
    J, _ = jm.jumps((0.2,))
    J0, _ = I.Inner(M, W0 * x[1], Fh_from(x, M), True).jumps((0.2,))
    print(f"jumps: value vs inner_np: {np.abs(J[0.2][0] - J0[0.2]).max() / np.abs(J0[0.2]).max():.1e} relative")
    for p, h in ((0, 1e-6), (1, 1e-7)):
        xp = x.copy(); xp[p] += h; xm = x.copy(); xm[p] -= h
        Jp, _ = I.Inner(M, W0 * xp[1], Fh_from(xp, M), True).jumps((0.2,))
        Jm, _ = I.Inner(M, W0 * xm[1], Fh_from(xm, M), True).jumps((0.2,))
        fd = (Jp[0.2] - Jm[0.2]) / (2 * h)
        print(f"jumps: d/d{'A' if p == 0 else 'wf'} jet vs centred difference (h = {h:g}): "
              f"{np.abs(J[0.2][p + 1] - fd).max() / np.abs(fd).max():.1e} relative")


def check_symmetry(M=25, J=9):
    A, B = 2.72, 0.15
    Fh = {1: -0.5j * A, -1: 0.5j * A, 3: -0.5j * B, -3: 0.5j * B}
    w = W0
    L = M + 2
    k = np.arange(-L, L + 1)
    odd = (np.abs(k) <= M) & (k % 2 == 1)
    even = (np.abs(k) <= M + 1) & (k % 2 == 0)
    imu = 1j * k
    g1 = F2h(Fh, M)
    # background and O(1/n) background series (K1 = 0, omega1 = 0, gauge -g1); unscaled (xi0 = 1)
    bn = JetModel(M, w, Fh, [Dir("n", q1=True)], [0, 1], {1: ({}, {kk: -v for kk, v in g1.items()})})
    bn.series_at(1.0, J + 3, stop=False)
    d = [v[0] for v in bn.ser["D"]]; g = [v[0] for v in bn.ser["G"]]
    dn = [v[1] for v in bn.ser["D"]]; gn = [v[1] for v in bn.ser["G"]]
    # gauge P: P + w P' = -2 g1
    P = -2 * g[1] / (1 + 1j * w * k)
    PwP = P * (1 + 1j * w * k)
    # Proposition 5 profile: a_0 = 2 d_1 + P X d_0
    a0 = 2 * d[1] + conv(P, (1 + imu * w) * d[0], L, odd)
    f2 = {int(kk): a0[kk + L] / 2 for kk in range(-M, M + 1) if abs(a0[kk + L]) > 0}
    # D_P Z_1: s-translation + gauge (P fixed) of the O(1/n) series
    DZ1y = lambda j: 2 * (j + 1) * dn[j + 1] + conv(P, (2 * j + 1 + imu * w) * dn[j], L, odd)
    DZ1c = lambda m: 2 * (m + 1) * gn[m + 1] + conv(PwP, gn[m], L, even) + conv(P, (2 * m + imu * w) * gn[m], L, even)
    wd0 = -DZ1y(0) / 2
    wg0 = -DZ1c(0)
    e, n, Wb = 1, 2, 4
    sf = SymForce(Wb)
    dirs = [Dir("pert", lam=2.0), Dir("n", q1=True), Dir("W", lam=2.0, force=sf)]
    data = {e: (f2, {}), n: ({}, {kk: -v for kk, v in g1.items()}),
            Wb: ({int(kk): wd0[kk + L] for kk in range(-M, M + 1)}, {int(kk): wg0[kk + L] for kk in range(-M - 1, M + 2)})}
    jm = JetModel(M, w, Fh, dirs, [0, e, n, e | n, Wb], data)
    jm.series_at(1.0, J + 1, stop=False)
    ie, ien, iW = jm.idx[e], jm.idx[e | n], jm.idx[Wb]
    worst = 0
    for j in range(0, J):
        lim = M - 2 * j - 7
        if lim < 1:
            break
        sel = np.abs(k) <= lim
        Dd = jm.ser["D"][j][ien]
        Ds = DZ1y(j) + jm.ser["D"][j][iW]
        Gd = jm.ser["G"][j][ien]
        Gs = DZ1c(j) + jm.ser["G"][j][iW]
        ry = np.abs(Dd - Ds)[sel].max() / np.abs(Dd)[sel].max()
        rc = np.abs(Gd - Gs)[sel].max() / max(np.abs(Gd)[sel].max(), 1e-300)
        worst = max(worst, ry, rc if j > 0 else 0)
        print(f"symmetry: order {j:2d} (|mu| <= {lim:2d}): y relative deviation {ry:.1e}, c {rc:.1e}")
    # also: the e component is the lambda = 2 mode D_P Z_0 (as pert_check3)
    ey = max(np.abs(jm.ser["D"][j][ie] - (2 * (j + 1) * d[j + 1] + conv(P, (2 * j + 1 + imu * w) * d[j], L, odd)))[np.abs(k) <= M - 2 * j - 7].max()
             / np.abs(jm.ser["D"][j][ie]).max() for j in range(1, 6))
    print(f"symmetry: the e component = D_P Z_0 (Proposition 5), orders 1-5: {ey:.1e}")
    print(f"symmetry: worst relative deviation D_P Z_1 + W vs direct e|n: {worst:.1e}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["series", "outer", "jumps", "symmetry"]
    for wch in which:
        t = time.time()
        {"series": check_series, "outer": check_outer, "jumps": check_jumps, "symmetry": check_symmetry}[wch]()
        print(f"   ({wch}: {time.time() - t:.0f} s)\n", flush=True)
