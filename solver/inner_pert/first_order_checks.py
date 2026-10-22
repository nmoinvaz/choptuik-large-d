"""Checks of the first-order machinery (notes/FIRST_ORDER.md, section 5). Each check prints its deviation and PASS or
FAIL against a stated threshold; the script exits with status 1 if any check fails. Every check can be made to fail
with a mutation (--mutate=...), which must then report FAIL.

  series     first_order_jet's background series = solver/inner_borel/inner_series.py (flint), M = 3, orders <= 30;
             perturbation series = solver/inner_pert/pert.py (flint), M = 5, lambda = 1.7, orders <= 20.
             Independent: shares only the equations (flint recursion vs numpy jet recursion).
             Mutation 'lam': the jet perturbation at lambda + 1e-3.
  outer      the O(1/n) inner series (1/n direction: O(1/n) terms, K1 = 0, gauge c1(0) = -F^2) = the next-to-top
             degree coefficients of the outer 1/n expansion (solver/largeD_late_orders.py), F = 2.72 sin, w_inf.
             Independent: the outer recursion shares nothing with the inner jets but the exact equations.
             Mutations 'q1y', 'q1c' (the O(1/n) terms), 'gauge' (c1(0) = 0).
  rhs        the jet right-hand side = the derivative of the summed jet series, every jet feature at once
             (O(1/n) terms, perturbation, d/dlambda, omega jet, breaking forcing), at xi = 0.04 e^{0.3 i}, M = 3.
             Shares the module, not the code path (series recursion vs right-hand side). Mutation 'rhs'.
  jumps      the value jump = solver/inner_stokes/inner_np.py's (M = 3); the A and omega jet directions = centred
             differences of inner_np's jumps. Mutation 'jet' (the omega direction scaled by 2).
  symmetry   the O(1/n) part of the lambda = 2 mode two ways: the direct hyper-dual (perturbation e at lambda = 2
             with Proposition 5's profile, times the 1/n direction n) against D_P Z_1 + W, W forced by the
             s-translation breaking (SymForce); untruncated harmonics (M = 25), F = 2.72 sin + 0.15 sin 3t.
             Mutations 'force', 'q1y'.

    python3 first_order_checks.py [series] [outer] [rhs] [jumps] [symmetry] [--mutate=NAME]
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
import first_order_jet as FJ  # noqa: E402
from first_order_jet import JetModel, Dir, Fh_from, dFh, W0, F2h, SymForce  # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
FJ.MUTATE.update({MUT} if MUT else set())
FAILS = []


def verdict(name, value, thr):
    ok = value <= thr
    if not ok:
        FAILS.append(name)
    print(f"   {name}: {value:.1e} (threshold {thr:.0e}) {'PASS' if ok else 'FAIL'}", flush=True)


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
    print("series: background d_j, j <= 30, M = 3, against inner_series (flint)")
    verdict("max relative deviation", err, 1e-11)
    M = 5
    A, B = 2.72, 0.15
    Fh = {1: -0.5j * A, -1: 0.5j * A, 3: -0.5j * B, -3: 0.5j * B}
    f = {1: 0.3 + 0.2j, -1: 0.3 - 0.2j, 3: 0.05j, -3: -0.05j}
    lam = 1.7
    jm = JetModel(M, W0, Fh, [Dir("pert", lam=lam + (1e-3 if MUT == "lam" else 0))], [0, 1], {1: (f, {})})
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
    print("series: perturbation a_j, h_j, j <= 20, M = 5, lambda = 1.7, against pert.py (flint)")
    verdict("max relative deviation", err, 1e-11)


def check_outer():
    import largeD_late_orders as lo
    K = 9
    th = 2 * np.pi * np.arange(lo.NT) / lo.NT
    S, D, G, _, _ = lo.run(2.72 * np.sin(th), W0, K, verbose=False)
    harm = lambda f, mu: np.fft.fft(f)[mu % lo.NT] / lo.NT
    M = 2 * K + 5
    Fh = {1: -0.5j * 2.72, -1: 0.5j * 2.72}
    g1 = F2h(Fh, M)
    gauge = {} if MUT == "gauge" else {k: -v for k, v in g1.items()}
    jm = JetModel(M, W0, Fh, [Dir("n", q1=True)], [0, 1], {1: ({}, gauge)})
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
    print("outer: O(1/n) inner series against the next-to-top outer coefficients")
    verdict("worst relative deviation", worst, 1e-10)


def check_rhs():
    M = 3
    A, B = 2.72, 0.15
    Fh = {1: -0.5j * A, -1: 0.5j * A, 3: -0.5j * B + 0.03, -3: 0.5j * B + 0.03}
    g1 = F2h(Fh, M)
    e, n, l, Wb, a = 1, 2, 4, 8, 16
    dirs = [Dir("pert", lam=1.7), Dir("n", q1=True, omega=0.3), Dir("l", dlam=True), Dir("W", lam=2.0, force=SymForce(Wb)),
            Dir("A", omega=1.0)]
    f = {1: 0.3 + 0.2j, -1: 0.3 - 0.2j, 3: 0.05j, -3: -0.05j}
    data = {e: (f, {}), n: ({1: 0.1j, -1: -0.1j}, {k: -v for k, v in g1.items()}), e | n: ({}, {0: 0.2, 2: 0.1, -2: 0.1}),
            Wb: ({1: 0.2, -1: 0.2}, {}), a: ({1: 0.5j, -1: -0.5j}, {})}
    jm = JetModel(M, W0, Fh, dirs, [0, e, n, e | n, e | l, Wb, a], data)
    xi = 0.04 * np.exp(0.3j)
    J = 30
    jm.series_at(1.0, J, stop=False)
    D, S, G = jm.ser["D"], jm.ser["S"], jm.ser["G"]
    y = sum(D[j] * xi ** (2 * j + 1) for j in range(J))
    yp = sum((2 * j + 1) * D[j] * xi ** (2 * j) for j in range(J))
    sg = sum(S[j] * xi ** (2 * j + 2) for j in range(J - 1))
    sgp = sum((2 * j + 2) * S[j] * xi ** (2 * j + 1) for j in range(J - 1))
    c = sum(G[j] * xi ** (2 * j) for j in range(J))
    cp = sum(2 * j * G[j] * xi ** (2 * j - 1) for j in range(1, J))
    dD, dS, dC = jm.unpack(jm.rhs(xi, jm.pack(y, sg, c)))
    worst = 0
    for m, i in jm.idx.items():
        devs = [np.abs(X[i] - Y[i]).max() / max(np.abs(Y[i]).max(), 1e-300) for X, Y in ((dD, yp), (dS, sgp), (dC, cp))]
        worst = max(worst, max(devs))
        print(f"rhs: component {m:2d}: y' {devs[0]:.1e}  sigma' {devs[1]:.1e}  c' {devs[2]:.1e}")
    print("rhs: right-hand side against the derivative of the summed series")
    verdict("worst relative deviation", worst, 1e-8)


def check_jumps():
    import inner_np as I
    M = 3
    x = np.array([2.7265, 1.0156, 0.0, 0.0])
    F1 = x[0] / 2j; Fn = F1 * (0.0530 + 0.0041j); x[2], x[3] = -2 * Fn.imag, 2 * Fn.real
    om = 2 * W0 if MUT == "jet" else W0
    jm = JetModel(M, W0 * x[1], Fh_from(x, M), [Dir("A"), Dir("w", omega=om)], [0, 1, 2], {1: (dFh(0, M), {})})
    J, _ = jm.jumps((0.2,))
    J0, _ = I.Inner(M, W0 * x[1], Fh_from(x, M), True).jumps((0.2,))
    print("jumps: value jump against inner_np")
    verdict("relative deviation", np.abs(J[0.2][0] - J0[0.2]).max() / np.abs(J0[0.2]).max(), 1e-10)
    for p, h in ((0, 1e-6), (1, 1e-7)):
        xp = x.copy(); xp[p] += h; xm = x.copy(); xm[p] -= h
        Jp, _ = I.Inner(M, W0 * xp[1], Fh_from(xp, M), True).jumps((0.2,))
        Jm, _ = I.Inner(M, W0 * xm[1], Fh_from(xm, M), True).jumps((0.2,))
        fd = (Jp[0.2] - Jm[0.2]) / (2 * h)
        print(f"jumps: d/d{'A' if p == 0 else 'wf'} jet against the centred difference (h = {h:g})")
        verdict("relative deviation", np.abs(J[0.2][p + 1] - fd).max() / np.abs(fd).max(), 1e-6)


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
    bn = JetModel(M, w, Fh, [Dir("n", q1=True)], [0, 1], {1: ({}, {kk: -v for kk, v in g1.items()})})
    bn.series_at(1.0, J + 3, stop=False)
    d = [v[0] for v in bn.ser["D"]]; g = [v[0] for v in bn.ser["G"]]
    dn = [v[1] for v in bn.ser["D"]]; gn = [v[1] for v in bn.ser["G"]]
    P = -2 * g[1] / (1 + 1j * w * k)
    PwP = P * (1 + 1j * w * k)
    a0 = 2 * d[1] + conv(P, (1 + imu * w) * d[0], L, odd)
    f2 = {int(kk): a0[kk + L] / 2 for kk in range(-M, M + 1) if abs(a0[kk + L]) > 0}
    DZ1y = lambda j: 2 * (j + 1) * dn[j + 1] + conv(P, (2 * j + 1 + imu * w) * dn[j], L, odd)
    DZ1c = lambda m: 2 * (m + 1) * gn[m + 1] + conv(PwP, gn[m], L, even) + conv(P, (2 * m + imu * w) * gn[m], L, even)
    wd0 = -DZ1y(0) / 2
    wg0 = -DZ1c(0)
    e, n, Wb = 1, 2, 4
    dirs = [Dir("pert", lam=2.0), Dir("n", q1=True), Dir("W", lam=2.0, force=SymForce(Wb))]
    data = {e: (f2, {}), n: ({}, {kk: -v for kk, v in g1.items()}),
            Wb: ({int(kk): wd0[kk + L] for kk in range(-M, M + 1)}, {int(kk): wg0[kk + L] for kk in range(-M - 1, M + 2)})}
    jm = JetModel(M, w, Fh, dirs, [0, e, n, e | n, Wb], data)
    jm.series_at(1.0, J + 1, stop=False)
    ie, ien, iW = jm.idx[e], jm.idx[e | n], jm.idx[Wb]
    worst = 0
    for j in range(1, J):
        lim = M - 2 * j - 7
        if lim < 1:
            break
        sel = np.abs(k) <= lim
        Dd, Ds = jm.ser["D"][j][ien], DZ1y(j) + jm.ser["D"][j][iW]
        Gd, Gs = jm.ser["G"][j][ien], DZ1c(j) + jm.ser["G"][j][iW]
        ry = np.abs(Dd - Ds)[sel].max() / np.abs(Dd)[sel].max()
        rc = np.abs(Gd - Gs)[sel].max() / np.abs(Gd)[sel].max()
        worst = max(worst, ry, rc)
        print(f"symmetry: order {j:2d} (|mu| <= {lim:2d}): y relative deviation {ry:.1e}, c {rc:.1e}")
    ey = max(np.abs(jm.ser["D"][j][ie] - (2 * (j + 1) * d[j + 1] + conv(P, (2 * j + 1 + imu * w) * d[j], L, odd)))[np.abs(k) <= M - 2 * j - 7].max()
             / np.abs(jm.ser["D"][j][ie]).max() for j in range(1, 6))
    print("symmetry: the e component = D_P Z_0 (Proposition 5), orders 1-5")
    verdict("relative deviation", ey, 1e-12)
    print("symmetry: D_P Z_1 + W against the direct e|n")
    verdict("worst relative deviation", worst, 1e-12)


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["series", "outer", "rhs", "jumps", "symmetry"]
    if MUT:
        print(f"MUTATION '{MUT}' active: some check below must FAIL\n")
    for wch in which:
        t = time.time()
        {"series": check_series, "outer": check_outer, "rhs": check_rhs, "jumps": check_jumps,
         "symmetry": check_symmetry}[wch]()
        print(f"   ({wch}: {time.time() - t:.0f} s)\n", flush=True)
    print("FAILED: " + ", ".join(FAILS) if FAILS else "ALL PASS")
    sys.exit(1 if FAILS else 0)
