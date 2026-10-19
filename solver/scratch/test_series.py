import numpy as np
from dss import DSS
np.set_printoptions(precision=3)
for n in (2.0, 3.0):
    S = DSS(n=n, N=32, Kc=8, Ks=8)
    N = S.N; tau = np.arange(N) / N * 2 * np.pi        # period 2pi -> omega = 1
    omega = np.array([1.3])
    # centre test
    p1 = (0.6 * np.sin(tau) + 0.15 * np.cos(3 * tau))[None]
    c0 = (1.4 + 0.2 * np.cos(2 * tau) + 0.05 * np.sin(2 * tau))[None]
    P, M, A, C = S.centre_series(p1, c0, omega)
    for xa, xb in ((0.02, 0.04), (0.05, 0.1)):
        s0 = np.stack([S.eval_series(F, xa) for F in (P, M, A, C)])
        s1 = S.integrate(s0, xa, xb, 2000, omega)
        sb = np.stack([S.eval_series(F, xb) for F in (P, M, A, C)])
        print(f"n={n} centre series {xa}->{xb}: max |ODE - series| per field", np.abs(s1 - sb).max(axis=(1, 2)))
    # SSH test
    m0 = (0.5 * np.sin(tau) - 0.1 * np.sin(3 * tau) + 0.05 * np.cos(tau))[None]
    a0 = ((1.15 if n == 2 else 1.06) + 0.04 * np.cos(2 * tau))[None]
    P, M, A, C = S.ssh_series(m0, a0, omega)
    for ua, ub in ((0.02, 0.04), (0.05, 0.1)):
        s0 = np.stack([S.eval_series(F, ua) for F in (P, M, A, C)])
        s1 = S.integrate(s0, 1 - ua, 1 - ub, 2000, omega)
        sb = np.stack([S.eval_series(F, ub) for F in (P, M, A, C)])
        print(f"n={n} SSH series {ua}->{ub}:    max |ODE - series| per field", np.abs(s1 - sb).max(axis=(1, 2)))
    print("   SSH coefficient sizes |p_k|:", [f"{np.abs(x).max():.2e}" for x in P])
