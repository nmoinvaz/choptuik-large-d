#!/usr/bin/env python3
"""The first boundary layer of the supercritical branch at beta^2 x^2 - 1 = O(n^{-1/2}), integrated inward.

Derivation (notes/SUPERCRITICAL.md, "The layer"): near z = beta^2 x^2 - 1 -> 0 the outer series is a series in eps/z^2
(expand.py), so it breaks down at z ~ n^{-1/2}. There u = (n-1)(b-1) ~ n^{1/2} << n (1/b = 1 - O(n^{-1/2})),
c ~ n^{-1/4}, y = p - m ~ n^{1/4}, p + m ~ 1, and tau-derivatives at fixed layer coordinate are O(n^{-1/2}) smaller than
the radial ones, so the layer is quasi-static in tau. With y = n^{1/4} Y, c = n^{-1/4} C, p + m = S,
ln x - ln(1/beta) = n^{-1/2} eta and x0t = (beta + beta')/beta^2 (= 1/beta for CSS; = -dr_s/dt, the front's velocity):
    x0t Y_eta = C S,     x0t S_eta - C Y_eta = C Y^3/2,     C_eta = C Y^2/2.
These are Rozali and Way's D = infinity equations (README, Prop. 5(i)) for a front moving in s = r^2/(2n), without
the linear term. Scaling eta = (|x0t|/f) theta, Y = (f/|x0t|)^{1/2} Y1, C = sqrt(f |x0t|) C1, S = (f/x0t) S1 removes
x0t and f (the system depends on x0t only through x0t^2 once the first integral below is used):
    Y1' = C1 S1,   S1' - C1 Y1' = C1 Y1^3/2,   C1' = C1 Y1^2/2      (' = d/dtheta).
Since C1' Y1 = C1 Y1^3/2, the second equation is (S1 - C1 Y1)' = 0: S1 = C1 Y1 - sqrt2 (the constant from the outer
region, where S1 -> 0 and C1 Y1 -> sqrt2). Matching to the outer leading order plus its first correction:
    Y1 = theta^{-1/2} (1 - 1/(2 theta^2) + ...),   C1 = sqrt(2 theta) (1 + 1/(4 theta^2) + ...),   theta -> +infinity.
The mode the outer region forbids grows outward (like exp(theta^2/2)), so inward integration is stable.

Checks (exit 1 on failure): the full third-order system conserves S1 - C1 Y1 = -sqrt2 to 1e-8; Y1(-inf) from
theta0 = 40 and 80 agrees to 1e-7; Y1 stays bounded and C1 -> 0. Reported: Y1(-inf), the decay rate of C1 behind the
front (= Y1(-inf)^2/2), and the front position theta_f (where S1 = -sqrt2/2, i.e. C1 Y1 = sqrt2/2).

    python3 layer.py
    python3 layer.py --mutate=sign     # Y1'' with the cubic's sign flipped (S1' - C1 Y1' = -C1 Y1^3/2): must FAIL
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
sg = -1.0 if mut == "sign" else 1.0
R2 = np.sqrt(2.0)


def start(th):
    Y = th ** -0.5 * (1 - 1 / (2 * th ** 2))
    C = np.sqrt(2 * th) * (1 + 1 / (4 * th ** 2))
    return Y, C, C * Y - R2


def full(th, Z):           # third-order system in theta: (Y1, S1, C1)
    Y, S, C = Z
    Yt = C * S
    return [Yt, C * Yt + sg * C * Y ** 3 / 2, C * Y * Y / 2]


def run(th0, th_end=-25.0):
    Y0, C0, S0 = start(th0)
    sol = solve_ivp(full, [th0, th_end], [Y0, S0, C0], method="LSODA", rtol=1e-12, atol=1e-24, dense_output=True)
    return sol


ok = True
res = {}
for th0 in (40.0, 80.0):
    sol = run(th0)
    th = sol.t
    Y, S, C = sol.y
    inv = S - C * Y + R2
    res[th0] = (sol, np.max(np.abs(inv)), Y[-1], C[-1])
    print(f"theta0 = {th0:5.1f}: status '{sol.message}', max|S1 - C1 Y1 + sqrt2| = {np.max(np.abs(inv)):.2e}, "
          f"Y1(-25) = {Y[-1]:.10f}, C1(-25) = {C[-1]:.3e}, max Y1 = {np.max(np.abs(Y)):.6f}"
          + (f"   (mutation: {mut})" if mut else ""))
sol, inv, Yinf, Cend = res[80.0]
ok &= inv < 1e-8 and abs(res[40.0][2] - Yinf) < 1e-7 and Cend < 1e-10 and np.all(np.isfinite(sol.y))
# decay rate of C1 behind the front, front position
th = np.linspace(-12, -7, 6)
rate = np.polyfit(th, np.log(sol.sol(th)[2]), 1)[0]
try:
    thf = brentq(lambda t: sol.sol(t)[1] + R2 / 2, -10, 20)
except ValueError:
    thf = float("nan")
    ok = False
Yf, Sf, Cf = sol.sol(thf) if np.isfinite(thf) else (np.nan,) * 3
print(f"Y1(-inf) = {Yinf:.9f};  Y1(-inf)^2/2 = {Yinf ** 2 / 2:.9f};  d ln C1/d theta behind the front = {rate:.9f}")
print(f"front (S1 = -sqrt2/2, i.e. C1 Y1 = sqrt2/2): theta_f = {thf:.6f}, Y1 = {Yf:.6f}, C1 = {Cf:.6f}")
for tv in (10, 5, 2, 0, -1, -2, -4, -8):
    Yv, Sv, Cv = sol.sol(tv)
    print(f"   theta = {tv:+5.1f}: Y1 = {Yv:.6f}  S1 = {Sv:+.6f}  C1 = {Cv:.6e}   (outer LO: Y1 = "
          + (f"{tv ** -0.5:.6f}, C1 = {np.sqrt(2 * tv):.6f})" if tv > 0 else "singular)"))
ok &= abs(rate - Yinf ** 2 / 2) < 1e-4      # C1 decays at the asymptotic rate (Y1 has converged)
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
