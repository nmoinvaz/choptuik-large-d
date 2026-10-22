#!/usr/bin/env python3
"""The first boundary layer of the supercritical branch at beta^2 x^2 - 1 = O(n^{-1/2}), integrated inward.

Derivation (notes/SUPERCRITICAL.md, "The layer"): near z = beta^2 x^2 - 1 -> 0 the outer series is a series in
eps/z^2 (expand.py), so it breaks down at z ~ n^{-1/2}. There u = (n-1)(b-1) ~ n^{1/2} << n (h = 1/b = 1 - O(n^{-1/2})),
c ~ n^{-1/4}, y = p - m ~ n^{1/4}, p + m ~ 1, and the time derivative enters only through the motion of the front, so
at leading order the layer is quasi-static in tau and, with y = n^{1/4} Y, c = n^{-1/4} C, p + m = S,
ln x - ln x0 = n^{-1/2} eta and x0t = (beta + beta')/beta^2 (= 1/beta for CSS):
    x0t Y_eta = C S,     x0t S_eta - C Y_eta = C Y^3/2,     C_eta = C Y^2/2.
These are Rozali and Way's D = infinity equations (README, Prop. 5(i)) for a travelling wave Y(s - s_b(t)) in the
heat form, without the linear term (y >> 1). Scaling eta = (x0t/f) theta, C = sqrt(f x0t) C1, S = (f/x0t) S1 removes
x0t and f; with dT = C1 d theta the universal problem is
    Y'' = C Y' + Y^3/2,     C' = Y^2/2      (' = d/dT),     S = Y',
matched to the outer leading order as theta -> +infinity:  Y ~ theta^{-1/2},  C ~ sqrt(2 theta),  S ~ -sqrt2/(4 theta^2).
The only mode that the outer region forbids (exponential in theta^2) grows outward, so inward integration from large
theta is stable; errors in the two slow modes are the symmetries (T-shift, scaling) and only relabel the solution.

The script integrates inward from theta0 (default 40) and reports what happens: whether Y blows up at finite T,
whether C reaches 0, and the value of theta (i.e. of z = 2 n^{-1/2} theta x0t beta / f... see the notes) there.

    python3 layer.py [theta0]
    python3 layer.py --mutate=sign     # Y'' = C Y' - Y^3/2 (wrong sign of the cubic): the behaviour must change
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

args = [a for a in sys.argv[1:] if not a.startswith("--")]
mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
th0 = float(args[0]) if args else 40.0
sg = -1.0 if mut == "sign" else 1.0


def asym(th):
    """outer leading order plus its first correction in the layer variables (theta, Y, C, S) -> (Y, Y_T, C)"""
    Y = th ** -0.5
    C = np.sqrt(2 * th)
    S = -np.sqrt(2) / (4 * th ** 2)        # = Y_theta / C
    return Y, S, C


# integrate in T backwards, carrying theta as well (theta_T = 1/C)
def rhs(T, Z):
    Y, V, C, th = Z
    return [V, C * V + sg * Y ** 3 / 2, Y * Y / 2, 1.0 / C]


def blow(T, Z):
    return 1e8 - abs(Z[0])
blow.terminal = True


def czero(T, Z):
    return Z[2] - 1e-10
czero.terminal = True

Y0, V0, C0 = asym(th0)
sol = solve_ivp(rhs, [0.0, -1e4], [Y0, V0, C0, th0], method="LSODA", rtol=1e-12, atol=1e-14,
                events=[blow, czero], dense_output=True)
T = sol.t
Y, V, C, th = sol.y
print(f"start theta0 = {th0}: Y = {Y0:.6f}, Y' = {V0:.3e}, C = {C0:.6f}" + (f"   (mutation: {mut})" if mut else ""))
print(f"status: {sol.message}; events: blow-up {len(sol.t_events[0])}, C -> 0 {len(sol.t_events[1])}")
for frac in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0):
    i = min(int(frac * (len(T) - 1)), len(T) - 1)
    print(f"   T = {T[i]:+.6e}  theta = {th[i]:+.6e}  Y = {Y[i]:.6e}  Y' = {V[i]:+.6e}  C = {C[i]:.6e}"
          f"   Y theta^(1/2) = {Y[i] * np.sqrt(abs(th[i])):.4f}")
# local exponents near the end
i1, i2 = len(T) - 200 if len(T) > 400 else 0, len(T) - 1
if sol.t_events[0].size:
    Ts = sol.t_events[0][0]
    tail = slice(max(0, len(T) - 60), len(T) - 1)
    k = np.polyfit(np.log(np.abs(T[tail] - Ts)), np.log(np.abs(Y[tail])), 1)[0]
    kc = np.polyfit(np.log(np.abs(T[tail] - Ts)), np.log(np.abs(C[tail] - C[-1]) + 1e-300), 1)[0]
    print(f"blow-up at T* = {Ts:.8f}, theta* = {th[-1]:.8f}; Y ~ |T - T*|^{k:.4f}; C(T*) = {C[-1]:.6e}")
