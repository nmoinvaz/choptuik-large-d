"""
Gravitational collapse of the massless scalar l-multiplet in D = n + 2 (spherical symmetry, polar-areal gauge), to get
initial guesses for the periodic BVP (dss_ell.py). Adapted from the project's ell = 0 code (/home/claude/collapse/
gr_collapse.py: log-stretched grid r = L sinh(x/L), 2nd-order differences, Evans differencing at the origin, RK4,
Kreiss-Oliger dissipation, outgoing condition at r_max).

Variables: psi = r^l chi (chi even), X = chi_r (odd), Pt = Pi / r^l (even), Pi = (a/alpha) psi_t. Normalisation as in
the BVP: G_ab = 2n sum_m T_ab, L = l (l + n - 1). With N = n + 2l and c = alpha / a:
  chi_t = c Pt,   X_t = (c Pt)_r,
  Pt_t  = r^-N (r^N c X)_r - c chi [ l^2 (a^2 - 1)/r^2 + 2 l L r^(2l-2) a^2 chi^2 ]
(the last bracket uses c_r/c = (n-1)(a^2-1)/r - 2 L r^(2l-1) a^2 chi^2 from the constraints). Constraints, with
w = r^(n-1) (1 - 1/a^2) and K = r^(2l-1) [(l chi + r X)^2 + r^2 Pt^2]:
  w_r + 2 K w = 2 r^(n-1) K + 2 L r^(n+2l-2) chi^2,
  (ln alpha)_r = K - L r^(2l-1) a^2 chi^2 + (n-1)(a^2 - 1)/(2r),   alpha(t, 0) = 1 (t = central proper time).
Central slicing is used because the l >= 1 near-critical evolutions collapse the lapse strongly: in time normalised at
r_max the centre's proper time hardly advances. The RK4 step follows the CFL limit min(dr / c) every step.
A black hole is flagged when max 1 - 1/a^2 reaches bh_thresh (default 0.9999; the l = 1 critical solution itself
exceeds 0.95, so the 0.95 flag of the l = 0 code tunes to the wrong threshold).
"""
import numpy as np
from numba import njit


def make_grid(dx, Lg, rmax):
    xmax = Lg * np.arcsinh(rmax / Lg)
    nn = int(round(xmax / dx))
    x = np.arange(nn + 1) * dx
    return Lg * np.sinh(x / Lg)


@njit(cache=True)
def solve_constraints(r, chi, X, Pt, n, l, L, a, alpha, mu, pw):
    N = r.shape[0]
    w_prev = 0.0
    mu[0] = 0.0; a[0] = 1.0
    lnal = np.empty(N); lnal[0] = 0.0
    K0 = 0.0
    g_prev = 0.0
    maxmu = 0.0; imax = 0
    for i in range(N - 1):
        h = r[i + 1] - r[i]
        ri, rj = r[i], r[i + 1]
        Ki = pw[1, i] * ((l * chi[i] + ri * X[i]) ** 2 + ri * ri * Pt[i] ** 2)
        Kj = pw[1, i + 1] * ((l * chi[i + 1] + rj * X[i + 1]) ** 2 + rj * rj * Pt[i + 1] ** 2)
        si = 2 * pw[0, i] * Ki + 2 * L * pw[2, i] * chi[i] ** 2
        sj = 2 * pw[0, i + 1] * Kj + 2 * L * pw[2, i + 1] * chi[i + 1] ** 2
        w_new = (w_prev * (1 - h * Ki) + 0.5 * h * (si + sj)) / (1 + h * Kj)
        m = w_new / pw[0, i + 1]
        if m >= 0.999999:
            for j in range(i + 1, N):
                mu[j] = m; a[j] = 1e6; alpha[j] = 0.0
            return 1.0, i + 1
        mu[i + 1] = m
        if m > maxmu:
            maxmu = m; imax = i + 1
        a[i + 1] = 1.0 / np.sqrt(1.0 - m)
        a2 = a[i + 1] ** 2
        g_new = Kj - L * pw[1, i + 1] * a2 * chi[i + 1] ** 2 + (n - 1) * (a2 - 1) / (2 * rj)
        lnal[i + 1] = lnal[i] + 0.5 * h * (g_prev + g_new)
        g_prev = g_new
        w_prev = w_new
    # central slicing: alpha(t, 0) = 1, so t is central proper time
    for i in range(N):
        alpha[i] = np.exp(lnal[i])
    return maxmu, imax


@njit(cache=True)
def compute_rhs(r, chi, X, Pt, a, alpha, n, l, L, dchi, dX, dPt, eps, pw):
    N = r.shape[0]
    Ne = n + 2 * l
    f = np.empty(N); g = np.empty(N); extra = np.empty(N)
    for i in range(N):
        c = alpha[i] / a[i]
        f[i] = c * X[i]
        g[i] = c * Pt[i]
    for i in range(1, N):
        c = alpha[i] / a[i]; a2 = a[i] ** 2
        extra[i] = c * chi[i] * (l * l * (a2 - 1) / (r[i] * r[i]) + 2 * l * L * pw[3, i] * a2 * chi[i] ** 2)
    # r = 0 by even extrapolation in r^2
    r1, r2 = r[1], r[2]
    extra[0] = (r2 * r2 * extra[1] - r1 * r1 * extra[2]) / (r2 * r2 - r1 * r1)
    dchi[0] = g[0]
    dX[0] = 0.0
    c1 = (f[1] * r2 ** 3 - f[2] * r1 ** 3) / (r1 * r2 ** 3 - r2 * r1 ** 3)
    dPt[0] = (Ne + 1) * c1 - extra[0]
    for i in range(1, N - 1):
        dchi[i] = g[i]
        dX[i] = (g[i + 1] - g[i - 1]) / (r[i + 1] - r[i - 1])
        dPt[i] = (Ne + 1) * (pw[4, i + 1] * f[i + 1] - pw[4, i - 1] * f[i - 1]) / (pw[5, i + 1] - pw[5, i - 1]) - extra[i]
    k = N - 1
    c = alpha[k] / a[k]
    q0, q1, q2 = r[k - 2], r[k - 1], r[k]
    w0 = (q2 - q1) / ((q0 - q1) * (q0 - q2))
    w1 = (q2 - q0) / ((q1 - q0) * (q1 - q2))
    w2 = (2 * q2 - q0 - q1) / ((q2 - q0) * (q2 - q1))
    s = 0.5 * Ne
    dchi[k] = -c * (w0 * chi[k - 2] + w1 * chi[k - 1] + w2 * chi[k] + s * chi[k] / q2)
    dX[k] = -c * (w0 * X[k - 2] + w1 * X[k - 1] + w2 * X[k] + s * X[k] / q2)
    dPt[k] = -c * (w0 * Pt[k - 2] + w1 * Pt[k - 1] + w2 * Pt[k] + s * Pt[k] / q2)
    if eps > 0.0:
        for i in range(0, N - 2):
            hloc = r[i + 1] - r[i]
            sc = eps / (16.0 * hloc)
            for which in range(3):
                if which == 0:
                    u = chi; du = dchi; par = 1.0
                elif which == 1:
                    u = X; du = dX; par = -1.0
                else:
                    u = Pt; du = dPt; par = 1.0
                um2 = u[i - 2] if i >= 2 else par * u[2 - i]
                um1 = u[i - 1] if i >= 1 else par * u[1]
                d4 = u[i + 2] - 4.0 * u[i + 1] + 6.0 * u[i] - 4.0 * um1 + um2
                du[i] -= sc * d4
    dX[0] = 0.0


@njit(cache=True)
def cfl_dt(r, a, alpha, cfl):
    m = 1e300
    for i in range(r.shape[0] - 1):
        c = max(alpha[i] / a[i], alpha[i + 1] / a[i + 1])
        if c <= 0.0:                      # beyond a horizon (alpha set to 0 there): no constraint
            continue
        h = (r[i + 1] - r[i]) / c
        if h < m:
            m = h
    return cfl * m


@njit(cache=True)
def evolve(r, chi, X, Pt, n, l, L, t_end, cfl, eps, bh_thresh, central_dt, disp_tmin, disp_frac,
           snap_tmin, snap_dt, max_snaps, pw, rdisp, dt_floor, max_central, snap_Tstar):
    """RK4 in central proper time t (alpha(t, 0) = 1) with the CFL step recomputed every step.
    status: 1 = black hole (max 1 - 1/a^2 >= bh_thresh), 2 = step collapsed below dt_floor (lapse freezing outside a
    forming horizon), 0 = no black hole by t_end / dispersed.
    Returns status, final t, peak inner mu, central history (t, t, chi(0), alpha(0) = 1, max mu, dt),
    snapshot times and slices (chi, X, Pt, a, alpha)."""
    N = r.shape[0]
    mu = np.zeros(N); a = np.ones(N); alpha = np.ones(N)
    k1 = np.zeros((3, N)); k2 = np.zeros((3, N)); k3 = np.zeros((3, N)); k4 = np.zeros((3, N))
    t1 = np.zeros(N); t2 = np.zeros(N); t3 = np.zeros(N)
    central = np.zeros((max_central, 6)); nc = 0
    snaps_t = np.zeros((max_snaps, 2)); slices = np.zeros((max_snaps, 5, N)); ns = 0
    maxmu, imax = solve_constraints(r, chi, X, Pt, n, l, L, a, alpha, mu, pw)
    peak = 0.0; status = 0; t = 0.0
    next_c = 0.0; next_s = snap_tmin
    while True:
        dt = cfl_dt(r, a, alpha, cfl)
        if t >= next_c and nc < max_central:
            central[nc, 0] = t; central[nc, 1] = t; central[nc, 2] = chi[0]
            central[nc, 3] = alpha[0]; central[nc, 4] = maxmu; central[nc, 5] = dt; nc += 1
            next_c = t + central_dt
        if snap_dt > 0 and t >= next_s and ns < max_snaps:
            snaps_t[ns, 0] = t; snaps_t[ns, 1] = t
            for i in range(N):
                slices[ns, 0, i] = chi[i]; slices[ns, 1, i] = X[i]; slices[ns, 2, i] = Pt[i]
                slices[ns, 3, i] = a[i]; slices[ns, 4, i] = alpha[i]
            ns += 1
            next_s = t + (snap_dt * max(snap_Tstar - t, 1e-9) if snap_Tstar > 0 else snap_dt)
        inner = 0.0
        for i in range(N):
            if r[i] > rdisp:
                break
            if mu[i] > inner:
                inner = mu[i]
        if inner > peak:
            peak = inner
        if maxmu >= bh_thresh:
            status = 1; break
        if dt < dt_floor:
            status = 2; break
        if t > disp_tmin and inner < disp_frac * peak and inner < 0.3:
            break
        if t >= t_end:
            break
        compute_rhs(r, chi, X, Pt, a, alpha, n, l, L, k1[0], k1[1], k1[2], eps, pw)
        for i in range(N):
            t1[i] = chi[i] + 0.5 * dt * k1[0, i]; t2[i] = X[i] + 0.5 * dt * k1[1, i]; t3[i] = Pt[i] + 0.5 * dt * k1[2, i]
        c_, i_ = solve_constraints(r, t1, t2, t3, n, l, L, a, alpha, mu, pw)
        if c_ >= 1.0:
            status = 1; break
        compute_rhs(r, t1, t2, t3, a, alpha, n, l, L, k2[0], k2[1], k2[2], eps, pw)
        for i in range(N):
            t1[i] = chi[i] + 0.5 * dt * k2[0, i]; t2[i] = X[i] + 0.5 * dt * k2[1, i]; t3[i] = Pt[i] + 0.5 * dt * k2[2, i]
        c_, i_ = solve_constraints(r, t1, t2, t3, n, l, L, a, alpha, mu, pw)
        if c_ >= 1.0:
            status = 1; break
        compute_rhs(r, t1, t2, t3, a, alpha, n, l, L, k3[0], k3[1], k3[2], eps, pw)
        for i in range(N):
            t1[i] = chi[i] + dt * k3[0, i]; t2[i] = X[i] + dt * k3[1, i]; t3[i] = Pt[i] + dt * k3[2, i]
        c_, i_ = solve_constraints(r, t1, t2, t3, n, l, L, a, alpha, mu, pw)
        if c_ >= 1.0:
            status = 1; break
        compute_rhs(r, t1, t2, t3, a, alpha, n, l, L, k4[0], k4[1], k4[2], eps, pw)
        for i in range(N):
            chi[i] += dt / 6.0 * (k1[0, i] + 2 * k2[0, i] + 2 * k3[0, i] + k4[0, i])
            X[i] += dt / 6.0 * (k1[1, i] + 2 * k2[1, i] + 2 * k3[1, i] + k4[1, i])
            Pt[i] += dt / 6.0 * (k1[2, i] + 2 * k2[2, i] + 2 * k3[2, i] + k4[2, i])
        X[0] = 0.0
        maxmu, imax = solve_constraints(r, chi, X, Pt, n, l, L, a, alpha, mu, pw)
        t += dt
    return status, t, peak, central[:nc], snaps_t[:ns], slices[:ns]


def initial_data(r, A, r0=3.0, delta=1.0):
    """Time-symmetric Gaussian shell in chi = psi / r^l."""
    chi = A * np.exp(-((r - r0) / delta) ** 2)
    X = -2.0 * (r - r0) / delta ** 2 * chi
    X[0] = 0.0
    return chi, X, np.zeros_like(r)


def run(A, n=2, l=1, dx=4e-4, Lg=0.05, rmax=30.0, t_end=12.0, cfl=0.3, eps=0.5, bh_thresh=0.9999,
        central_dt=1e-3, disp_tmin=1e9, disp_frac=0.1, snap_tmin=1e9, snap_dt=0.0, max_snaps=1, r0=3.0, delta=1.0,
        rdisp=6.0, dt_floor_rel=1e-3, snap_Tstar=0.0):
    """t is central proper time. dt_floor = dt_floor_rel * dx * cfl. With snap_Tstar > 0 the snapshot spacing is
    snap_dt * (snap_Tstar - t) (uniform in ln(T* - T))."""
    r = make_grid(dx, Lg, rmax)
    chi, X, Pt = initial_data(r, A, r0, delta)
    L = l * (l + n - 1)
    Ne = n + 2 * l
    pw = np.stack([r ** (n - 1), r ** (2 * l - 1), r ** (n + 2 * l - 2), r ** (2 * l - 2), r ** Ne, r ** (Ne + 1)])
    max_central = int(t_end / central_dt) + 10
    out = evolve(r, chi, X, Pt, float(n), int(l), float(L), t_end, cfl, eps, bh_thresh, central_dt, disp_tmin,
                 disp_frac, snap_tmin, snap_dt, max(1, max_snaps), pw, rdisp, dt_floor_rel * dx * cfl, max_central,
                 snap_Tstar)
    status, t, peak, central, snaps_t, slices = out
    return dict(status=int(status), t=float(t), peak=float(peak), central=central, snaps_t=snaps_t, slices=slices,
                r=r)
