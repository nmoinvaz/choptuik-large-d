"""Checks of the second-order jets (notes/SECOND_ORDER.md, section 3). Each check prints its deviation and PASS or
FAIL against a stated threshold; the script exits with status 1 if any check fails. Mutations (--mutate=NAME) must
make a check fail.

  outer   the O(1/n) and O(1/n^2) inner series (directions e_1, e_2 = 1/n, the mixed component e_1 e_2 = twice the
          1/n^2 part; K_1 = K_2 = 0, omega_1 = omega_2 = 0, light-cone gauge c_1(0) = -g_1, c_2(0) = -g_2 - h_1)
          equal the next-to-top and next-to-next-to-top coefficients of the outer 1/n expansion
          (solver/largeD_late_orders.py), F = 2.72 sin, omega_inf, untruncated harmonics.
          Independent: the outer recursion shares nothing with the inner jets but the exact equations.
          Mutations: 'eps2y' (the e^2 term of the y equation dropped), 'phi' ((1 - E) -> 1 in Phi), 'pol' (e^2 =
          e_1 e_2 instead of 2 e_1 e_2), 'gauge2' (c_2(0) = 0).
  rhs     the right-hand side = the derivative of the summed series, with every feature at once (two 1/n directions
          and their mixed component, an omega jet on the mixed component, a perturbation with a lambda jet on e_1,
          e_2 and e_1 e_2, d/dlambda, a parameter direction), M = 3, at xi = 0.04 e^{0.3 i}.
          Shares the module, not the code path (series recursion vs right-hand side). Mutation 'rhs2' (the e^2 term
          of the y equation dropped in the right-hand side only).

    python3 second_order_checks.py [outer] [rhs] [--mutate=NAME]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import second_order_jet as SJ  # noqa: E402
from second_order_jet import JetModel2, Dir, W0, F2h  # noqa: E402

MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
if MUT in ("eps2y", "phi", "pol"):
    SJ.MUTATE.add(MUT)
FAILS = []


def verdict(name, value, thr):
    ok = value <= thr
    if not ok:
        FAILS.append(name)
    print(f"   {name}: {value:.1e} (threshold {thr:.0e}) {'PASS' if ok else 'FAIL'}", flush=True)


def gauge2(M, w, Fh, K1h, w1, xi0=1.0):
    """c_1(0) = -g_1 and c_2(0) = -g_2 - h_1 (g_k: xi^(2k) coefficient of c_0; h_1: xi^2 coefficient of c_1), from a
    preliminary series of the value and one 1/n direction carrying K_1, omega_1 and c_1(0)"""
    g1 = F2h(Fh, M)
    gauge1 = {k: -v for k, v in g1.items()}
    jm = JetModel2(M, w, Fh, [Dir("e1", q1=True, omega=w1)], [0, 1], {1: (K1h, gauge1)})
    jm.series_at(xi0, 3, stop=False)
    L = jm.L
    g2 = jm.ser["G"][2][0]
    h1 = jm.ser["G"][1][1]
    k0 = -(g2 + h1)
    return gauge1, {int(k): k0[k + L] for k in range(-M - 1, M + 2) if k % 2 == 0 and abs(k0[k + L]) > 0}


def check_outer():
    import largeD_late_orders as lo
    K = 10
    th = 2 * np.pi * np.arange(lo.NT) / lo.NT
    S, D, G, _, _ = lo.run(2.72 * np.sin(th), W0, K, verbose=False)
    harm = lambda f, mu: np.fft.fft(f)[mu % lo.NT] / lo.NT
    M = 2 * K + 5
    Fh = {1: -0.5j * 2.72, -1: 0.5j * 2.72}
    gauge1, gauge2c = gauge2(M, W0, Fh, {}, 0.0)
    if MUT == "gauge2":
        gauge2c = {}
    e1, e2 = 1, 2
    data = {e1: ({}, gauge1), e2: ({}, gauge1), e1 | e2: ({}, {k: 2 * v for k, v in gauge2c.items()})}
    jm = JetModel2(M, W0, Fh, [Dir("e1", q1=True), Dir("e2", q1=True)], [0, e1, e2, e1 | e2], data)
    jm.series_at(1.0, K + 1, stop=False)
    L = jm.L
    i1, i12 = jm.idx[e1], jm.idx[e1 | e2]
    worst1 = worst2 = 0
    for j in range(0, K - 2):
        D1, S1, G1 = jm.ser["D"][j][i1], jm.ser["S"][j][i1], jm.ser["G"][j][i1]
        D2, S2, G2 = jm.ser["D"][j][i12] / 2, jm.ser["S"][j][i12] / 2, jm.ser["G"][j][i12] / 2
        mus_o = range(-2 * j - 7, 2 * j + 8, 2)
        mus_e = range(-2 * j - 8, 2 * j + 9, 2)
        e1y = max(abs(D1[mu + L] - harm(D[j + 1][2 * j + 1], mu)) for mu in mus_o)
        e1s = max(abs(S1[mu + L] - harm(S[j + 2][2 * j + 2], mu)) for mu in mus_o)
        e1c = max(abs(G1[mu + L] + harm(G[j + 1][2 * j], mu)) for mu in mus_e)
        e2y = max(abs(D2[mu + L] - harm(D[j + 2][2 * j + 1], mu)) for mu in mus_o)
        e2s = max(abs(S2[mu + L] - harm(S[j + 3][2 * j + 2], mu)) for mu in mus_o)
        e2c = max(abs(G2[mu + L] + harm(G[j + 2][2 * j], mu)) for mu in mus_e)
        sc1 = max(np.abs(D1).max(), np.abs(S1).max(), np.abs(G1).max())
        sc2 = max(np.abs(D2).max(), np.abs(S2).max(), np.abs(G2).max())
        worst1 = max(worst1, max(e1y, e1s, e1c) / sc1)
        worst2 = max(worst2, max(e2y, e2s, e2c) / sc2)
        print(f"outer: order {j}: 1/n: y {e1y:.1e} sigma {e1s:.1e} c {e1c:.1e} (scale {sc1:.1e});  "
              f"1/n^2: y {e2y:.1e} sigma {e2s:.1e} c {e2c:.1e} (scale {sc2:.1e})")
    print("outer: the O(1/n) inner series (e_1 component) against the next-to-top outer coefficients")
    verdict("worst relative deviation", worst1, 1e-10)
    print("outer: the O(1/n^2) inner series (e_1 e_2 component / 2) against the next-to-next-to-top coefficients")
    verdict("worst relative deviation", worst2, 1e-10)


def check_rhs():
    M = 3
    A, B = 2.72, 0.15
    Fh = {1: -0.5j * A, -1: 0.5j * A, 3: -0.5j * B + 0.03, -3: 0.5j * B + 0.03}
    g1 = F2h(Fh, M)
    p, e1, e2, l, a = 1, 2, 4, 8, 16
    dirs = [Dir("pert", lam=1.7), Dir("e1", q1=True, omega=0.3), Dir("e2", q1=True, omega=0.3), Dir("l", dlam=True),
            Dir("A", omega=1.0)]
    f = {1: 0.3 + 0.2j, -1: 0.3 - 0.2j, 3: 0.05j, -3: -0.05j}
    K1 = {1: 0.1j, -1: -0.1j, 3: 0.02, -3: 0.02}
    gg = {k: -v for k, v in g1.items()}
    data = {p: (f, {}), e1: (K1, gg), e2: (K1, gg), e1 | e2: ({1: 0.05, -1: 0.05}, {0: 0.3, 2: 0.1, -2: 0.1}),
            p | e1: ({1: 0.02j, -1: -0.02j}, {0: 0.2, 2: 0.1, -2: 0.1}), p | e2: ({1: 0.02j, -1: -0.02j}, {0: 0.2}),
            p | e1 | e2: ({3: 0.01, -3: 0.01}, {0: 0.05}), a: ({1: 0.5j, -1: -0.5j}, {})}
    masks = [0, p, e1, e2, e1 | e2, p | e1, p | e2, p | e1 | e2, p | l, a]
    jm = JetModel2(M, W0, Fh, dirs, masks, data, wextra={e1 | e2: 0.2}, lamj={e1: 0.11, e2: 0.11, e1 | e2: 0.05})
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
    if MUT == "rhs2":
        SJ.MUTATE.add("eps2y")
    dD, dS, dC = jm.unpack(jm.rhs(xi, jm.pack(y, sg, c)))
    SJ.MUTATE.discard("eps2y")
    worst = 0
    for m, i in jm.idx.items():
        devs = [np.abs(X[i] - Y[i]).max() / max(np.abs(Y[i]).max(), 1e-300) for X, Y in ((dD, yp), (dS, sgp), (dC, cp))]
        worst = max(worst, max(devs))
        print(f"rhs: component {m:2d}: y' {devs[0]:.1e}  sigma' {devs[1]:.1e}  c' {devs[2]:.1e}")
    print("rhs: right-hand side against the derivative of the summed series")
    verdict("worst relative deviation", worst, 1e-8)


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["outer", "rhs"]
    if MUT:
        print(f"MUTATION '{MUT}' active: some check below must FAIL\n")
    for wch in which:
        t = time.time()
        {"outer": check_outer, "rhs": check_rhs}[wch]()
        print(f"   ({wch}: {time.time() - t:.0f} s)\n", flush=True)
    print("FAILED: " + ", ".join(FAILS) if FAILS else "ALL PASS")
    sys.exit(1 if FAILS else 0)
