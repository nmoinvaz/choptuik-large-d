"""Model 4 of matter/STRUCTURE.md: the phase-rotating (CSS) complex scalar at D = infinity.

For a complex massless scalar (each real component normalized as the solver's scalar), the D = infinity inner problem
(README, "The inner problem at large x") is, with y = y1 + i y2 and L = xi^2 d_xi + omega xi d_theta,
    L y = c sigma,   L sigma = c xi y_xi + c (eta - 1) y,   xi c_xi = c eta,   eta = |y|^2 / 2 .
The phase-rotating ansatz y = e^{i theta} Y(xi), sigma = e^{i theta} S(xi), c = c(xi) is exact (eta = |Y|^2/2 has no
theta-dependence): the problem is a system of ODEs and closes on ONE harmonic, with no truncation. It is the README's
one-harmonic model with F^3 replaced by |F|^2 F. Off the real axis |Y|^2 is continued as Y(xi) Ys(xi), where Ys is the
continuation of conj(Y(conj xi)) and obeys the equations with omega -> -omega.

Regular data: Y = 2 A xi + ..., c(0) = 1 (A > 0 by the phase symmetry). Selection rule (Conjecture of the README,
applied to this solution): the fast-mode Stokes multiplier vanishes, J(A, omega) = 0, one complex condition for two
real unknowns. J is computed from lateral solutions as in solver/inner_stokes/lateral_np.py: the optimally truncated
series at xi0 = r0 e^{3 i pi/8} (where the fast mode e^{-1/(2 xi^2)} is exponentially large, so the series data pick
the lateral sum uniquely), integrated along the ray to |xi| = r1 and the arc to the real axis; the lower lateral sum
is its mirror image, so J = Y+(r1) - conj(Ys+(r1)). Normalized as R = J / (2 pi i K'(s) f_fast(r1) 2A), R = 1 without
gravity (checked below against the repository's linear normalization).

    nice -n 10 python3 struct_css.py check                 linear check (R = 1) and series check against the README
    nice -n 10 python3 struct_css.py scan A0 A1 w0 w1 k     |R| on a k x k grid (A, omega)
    nice -n 10 python3 struct_css.py solve A w             root of the jump from (A, omega), at two r1
    nice -n 10 python3 struct_css.py verify [A w]          the same at five (r0, r1, rtol) setups (default start 1.997, 3.979)

Found (verify): A = 1.9970774009, omega = 3.9789695074, the same at every setup (matter/STRUCTURE.md, section 7).
Root finders started elsewhere also land on zeros at omega = 0 (A = 0.70710678, 1.14729637, 1.48880136, 1.77558388;
each a sign change of the real jump at r1 = 0.3 and 0.25): static real-scalar inner solutions, not rotating ones.
"""
import os
import sys
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import root

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_stokes"))
from lateral_np import fast_lin, Kp          # noqa: E402  (linear fast solution and K'(s) of the repository)


def series(A, w, J, dps=30, gravity=True):
    """Coefficients d_j, s_j, g_j (and the conjugate-partner ds_j, ss_j) of Y, S, c at xi = 0."""
    mp.mp.dps = dps
    A, w = mp.mpf(A), mp.mpf(w)
    I = mp.mpc(0, 1)
    d, ds = [2 * A], [2 * A]
    s, ss = [], []
    g = [mp.mpf(1)]
    e, h = [], []
    for j in range(J):
        # s_j from L Y = c S at xi^{2j+2}
        s.append((2 * j + 1 + I * w) * d[j] - sum(g[k] * s[j - k] for k in range(1, j + 1)))
        ss.append((2 * j + 1 - I * w) * ds[j] - sum(g[k] * ss[j - k] for k in range(1, j + 1)))
        if gravity:
            e.append(sum(d[k] * ds[j - k] for k in range(j + 1)) / 2)
            h.append(sum(g[k] * e[j - k] for k in range(j + 1)))
            g.append(h[j] / (2 * j + 2))
        else:
            e.append(mp.mpf(0)); h.append(mp.mpf(0)); g.append(mp.mpf(0))
        # d_{j+1} from L S = c xi Y' + c (eta - 1) Y at xi^{2j+3}
        def nxt(dd, sv, sign):
            acc = (2 * j + 2 + sign * I * w) * sv[j]
            acc -= sum(g[k] * 2 * (j + 1 - k) * dd[j + 1 - k] for k in range(1, j + 2))
            acc -= sum(h[k] * dd[j - k] for k in range(j + 1))
            return acc / (2 * j + 2)
        d.append(nxt(d, s, 1))
        ds.append(nxt(ds, ss, -1))
    return d, ds, s, ss, g


def series_value(coef, xi):
    """optimally truncated sums at complex xi"""
    d, ds, s, ss, g = coef
    xi = mp.mpc(xi)
    z = xi * xi
    terms = [abs(d[j] * z ** j) for j in range(len(s))]
    jt = min(range(5, len(s)), key=lambda j: terms[j])
    Y = sum(d[j] * xi ** (2 * j + 1) for j in range(jt))
    Ys = sum(ds[j] * xi ** (2 * j + 1) for j in range(jt))
    S = sum(s[j] * xi ** (2 * j + 2) for j in range(jt))
    Ss = sum(ss[j] * xi ** (2 * j + 2) for j in range(jt))
    C = sum(g[j] * xi ** (2 * j) for j in range(jt))
    return np.array([complex(v) for v in (Y, Ys, S, Ss, C)]), jt, float(terms[jt])


def rhs(xi, Z, w, gravity=True):
    Y, Ys, S, Ss, C = Z
    eta = Y * Ys / 2 if gravity else 0.0
    dY = (C * S - 1j * w * xi * Y) / (xi * xi)
    dYs = (C * Ss + 1j * w * xi * Ys) / (xi * xi)
    dS = (C * xi * dY + C * (eta - 1) * Y - 1j * w * xi * S) / (xi * xi)
    dSs = (C * xi * dYs + C * (eta - 1) * Ys + 1j * w * xi * Ss) / (xi * xi)
    dC = C * eta / xi if gravity else 0.0
    return np.array([dY, dYs, dS, dSs, dC])


def lateral(A, w, r0=0.1, r1=0.3, gravity=True, rtol=1e-12, atol=1e-15):
    phi0 = 3 * np.pi / 8
    J = int(1 / (2 * r0 ** 2)) + 25
    coef = series(A, w, J, gravity=gravity)
    Z0, jt, err = series_value(coef, r0 * np.exp(1j * phi0))
    u = np.exp(1j * phi0)
    sol = solve_ivp(lambda t, Z: rhs(t * u, Z, w, gravity) * u, (r0, r1), Z0, method="DOP853", rtol=rtol, atol=atol)
    Z = sol.y[:, -1]
    sol = solve_ivp(lambda t, Z: rhs(r1 * np.exp(1j * t), Z, w, gravity) * 1j * r1 * np.exp(1j * t), (phi0, 0.0), Z,
                    method="DOP853", rtol=rtol, atol=atol)
    Z = sol.y[:, -1]
    JY = Z[0] - np.conj(Z[1])
    ref = 2j * np.pi * Kp(1j * w) * fast_lin(1j * w, r1) * (2 * A)
    return JY / ref, Z, err


def R(A, w, r1=0.3):
    return lateral(A, w, r1=r1)[0]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        W0 = float(2 * mp.pi / mp.mpf("1.64389375"))
        for r1 in (0.3, 0.25):
            Rl, Z, err = lateral(1.0, W0, r1=r1, gravity=False)
            print(f"linear (no gravity), omega_inf, r1 = {r1}: R = {Rl:.12f}   (series truncation error {err:.1e})")
        # series against the README's first coefficients with F -> A e^{i theta}, F^3 -> |F|^2 F
        A, w = 1.3, 2.1
        d, ds, s, ss, g = series(A, w, 4)
        I = 1j
        print("g1 - A^2                       :", complex(g[1]) - A * A)
        print("s0 - 2(1 + i w) A              :", complex(s[0]) - 2 * (1 + I * w) * A)
        print("d1 - (A(2 + 3iw - w^2) - 2A^3) :", complex(d[1]) - (A * (2 + 3 * I * w - w * w) - 2 * A ** 3))
    elif cmd == "scan":
        A0, A1, w0, w1, k = map(float, sys.argv[2:7])
        k = int(k)
        for A in np.linspace(A0, A1, k):
            row = []
            for w in np.linspace(w0, w1, k):
                try:
                    row.append(R(A, w))
                except Exception:
                    row.append(np.nan)
            print(f"A {A:.4f}: " + " ".join(f"{abs(v):9.3e}/{np.angle(v) / np.pi:+.2f}" for v in row), flush=True)
    elif cmd in ("solve", "verify"):
        # zeros of z = R r1^{-2 A^2} (the factor removes the gravitational power xi^{2|F|^2} of the fast mode and
        # does not move zeros); a genuine zero does not depend on r0, r1 or the tolerance
        A, w = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (1.997, 3.979)
        # r1 must stay below about 0.335: beyond it the arc passes a movable singularity of the upper lateral
        # solution (at the zero, |z| ~ 5e-8 for r1 = 0.20-0.33, then ~1e5 at r1 = 0.34), i.e. a different continuation
        setups = [(0.1, 0.3, 1e-12), (0.1, 0.25, 1e-12)]
        if cmd == "verify":
            setups += [(0.09, 0.3, 1e-13), (0.11, 0.2, 1e-13), (0.1, 0.33, 1e-13)]
        for r0, r1, tol in setups:
            def f(v):
                z = lateral(v[0], v[1], r0=r0, r1=r1, rtol=tol)[0] * r1 ** (-2 * v[0] ** 2)
                return [z.real, z.imag]
            sol = root(f, [A, w], method="hybr", options={"xtol": 1e-13})
            print(f"r0 {r0} r1 {r1} rtol {tol:.0e}: A = {sol.x[0]:.10f}, omega = {sol.x[1]:.10f}, "
                  f"Delta = 2 pi/omega = {2 * np.pi / sol.x[1]:.10f}, |z| = {np.abs(sol.fun).max():.1e}, {sol.success}",
                  flush=True)
