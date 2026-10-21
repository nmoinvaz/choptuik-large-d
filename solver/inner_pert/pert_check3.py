"""Exact lambda = 2 mode of the D = infinity inner problem: s-translation plus a gauge (time-reparametrization) fix.

In physical variables the inner operator is L = xi^2 d_xi + omega xi d_theta = rho d_t (rho = r/sqrt(n), t the time with
xi = rho/(-t), tau = -ln(-t)). With s = rho^2/2, yhat = y/sqrt(s), sigmahat = sigma/(sqrt(2) s), the inner system
    rho y_t = c sigma,   rho sigma_t = c rho y_rho + c (y^2/2 - 1) y,   rho c_rho = c y^2/2
becomes autonomous in s and t:
    yhat_t = c sigmahat,   sigmahat_t = c (yhat_s + yhat^3/4),   c_s = c yhat^2/4,
a nonlinear heat equation with "time" s and "space" t (its fast mode e^{-1/(2 xi^2)} = e^{-t^2/(4 s)} is the heat kernel).
Its symmetries: t-translation (lambda = 1, the known X Z mode), scaling (the self-similarity), time reparametrization
t -> T(t), c -> T' c (gauge), and s-translation, whose generator d_s = (1/rho) d_rho at fixed t grows like e^{2 tau}:
    delta y = e^{2 tau} xi^-2 (xi y_xi - y),   delta c = e^{2 tau} xi^-2 xi c_xi        (a_k = 2(k+1) d_{k+1}, h_m = 2(m+1) g_{m+1}).
It changes c at xi = 0 (h_0 = 2 g_1), which the matching fixes (h_0 = 0); a reparametrization eta = e^tau P(theta) with
P + omega P' = -2 g_1 removes it:
    delta y += e^{2 tau} P X y,   delta c += e^{2 tau} [(P + omega P') c + P X c],   X = xi d_xi + omega d_theta.
This script checks that series_pert at lambda = 2 with that a_0 reproduces the predicted a_j, h_j for all j (harmonics
not reached by the Galerkin truncation), with gravity (F = A sin theta + B sin 3 theta).
    python3 pert_check3.py [M] [J]
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_borel"))
import mpmath as mp
from flint import acb, acb_poly, ctx
from pert import series_pert
from inner_stokes import W0, to_mp

M = int(sys.argv[1]) if len(sys.argv) > 1 else 25
J = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ctx.prec = 600
w = acb(str(W0))
I = acb(0, 1)
A, B = acb("2.72"), acb("0.15")
F = {1: acb(0, -0.5) * A, -1: acb(0, 0.5) * A, 3: acb(0, -0.5) * B, -3: acb(0, 0.5) * B}
N = 2 * M + 1


def conv(p, q):
    """product of two harmonic lists (index mu + M), truncated to |mu| <= M"""
    c = (acb_poly(p) * acb_poly(q)).coeffs()
    c = c + [acb(0)] * (4 * M + 1 - len(c))
    return c[M:3 * M + 1]


# background (lambda irrelevant for d, g): run once with a zero perturbation
d, _, g, _ = series_pert(F, {}, w, 0, J + 2, M, prec=600)
# gauge function P: P_mu (1 + i omega mu) = -2 g_{1,mu}
P = [-2 * g[1][k] / (1 + I * w * (k - M)) for k in range(N)]
Xd = lambda j: [(2 * j + 1 + I * w * (k - M)) * d[j][k] for k in range(N)]
Xg = lambda m: [(2 * m + I * w * (k - M)) * g[m][k] for k in range(N)]
PpwP = [P[k] * (1 + I * w * (k - M)) for k in range(N)]
a_pred = [[2 * (j + 1) * d[j + 1][k] + v for k, v in enumerate(conv(P, Xd(j)))] for j in range(J + 1)]
h_pred = [[2 * (m + 1) * g[m + 1][k] + u + v for k, (u, v) in enumerate(zip(conv(PpwP, g[m]), conv(P, Xg(m))))]
          for m in range(J + 1)]
print("h_pred[0] (must vanish):", max(float(abs(to_mp(v))) for v in h_pred[0]))
f = {k - M: a_pred[0][k] / 2 for k in range(N)}
_, a, _, h = series_pert(F, f, w, 2, J, M, prec=600)
for lam_test in (2,):
    worst = 0
    for j in range(1, J + 1):
        lim = M - 2 * j - 5                      # harmonics untouched by the truncation at order j
        if lim < 1:
            break
        devs = []
        for k in range(N):
            if abs(k - M) > lim:
                continue
            for X_, P_ in ((a[j][k], a_pred[j][k]), (h[j][k], h_pred[j][k])):
                sc = float(abs(to_mp(P_)))
                if sc > 1e-200:
                    devs.append(float(abs(to_mp(X_ - P_))) / sc)
        worst = max(worst, max(devs))
        print(f"j {j:2d}: |mu| <= {lim:2d}  max relative deviation {max(devs):.1e}")
    print(f"lambda = 2 mode (s-translation + gauge): worst {worst:.1e}")
# control: lambda = 1.9 and 2.1 with the same profile do NOT reproduce it
for lam in ("1.9", "2.1"):
    _, a2, _, _ = series_pert(F, f, w, lam, 4, M, prec=600)
    print(f"control lambda {lam}: relative deviation of a_3 harmonic 1:",
          mp.nstr(abs(to_mp(a2[3][M + 1] - a_pred[3][M + 1])) / abs(to_mp(a_pred[3][M + 1])), 3))
