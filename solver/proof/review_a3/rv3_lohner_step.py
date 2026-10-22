#!/usr/bin/env python3
"""Review A3, check K2: one step of certify_M_lohner.LohnerM at M = 3 (the code under test, with its Taylor order
N and its lower Jacobian order NJ) against a reference solution computed independently: mpmath.odefun (Taylor method,
60 digits) on a right-hand side written here from the equations in solver/inner_stokes/inner_np.py's docstring,
    xi^2 D'  + i mu w xi D  = [C S]_mu
    xi^2 S'  + i mu w xi S  = [C xi D']_mu + [(C E) D]_mu - [C D]_mu,   E = [D D]/2
    xi C'                   = [C E]_nu
(D, S on odd |mu| <= M, C and E on even |nu| <= M + 1, every binary product projected), and on paths written here:
ray xi = (r0 + (r1 - r0) t) e^{i phi}, arc xi = r1 e^{i phi (1 - t)}, real xi = r0 + (r1 - r0) t.

For each leg (ray, arc, real) and each order pair (N, NJ) = (20, 12) (the certificate's) and (6, 4):
  * the reference y(t0 + h) from the exact start point must lie in the step's state set c' + Q' q';
  * reference derivative columns d y(t0 + h) / d y0 . e_j (central differences at 60 digits, delta 1e-20) for three
    directions (a D component, an S component, and w, which tests the d(omega) terms) must lie in the step's
    derivative set cV' + QV' qV'.
Negative controls (must fire): at (N, NJ) = (6, 4) with the Taylor remainder of the trajectory removed, the reference
state must fall outside; with the remainder of the step Jacobian removed, a reference column must fall outside.
Exit 0 iff every containment holds and both controls fire.

    python3 rv3_lohner_step.py            (about 2-4 minutes)
"""
import os
import sys

import mpmath
from mpmath import mp, mpc, mpf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from flint import acb, arb, acb_mat, ctx  # noqa: E402
import certify_M_sys as CS  # noqa: E402  (the code under test)
import certify_M_lohner as CL  # noqa: E402  (the code under test)

M = 3
ODD = [m for m in range(-M, M + 1) if m % 2]
EVEN = [v for v in range(-(M + 1), M + 2) if v % 2 == 0]
ND = 2 * len(ODD) + len(EVEN)
mp.dps = 60


# ------------------------------------------------------------------ reference (independent)

def proj(X, Y, out):
    return {k: sum((X[a] * Y[k - a] for a in X if (k - a) in Y), mpc(0)) for k in out}


def F_ref(Y, xi, w):
    """dY/dxi from the docstring equations; Y in the order [D (odd ascending), S, C (even ascending)]"""
    no = len(ODD)
    D = {m: Y[i] for i, m in enumerate(ODD)}
    S = {m: Y[no + i] for i, m in enumerate(ODD)}
    C = {v: Y[2 * no + i] for i, v in enumerate(EVEN)}
    I = mpc(0, 1)
    CS_ = proj(C, S, ODD)
    dD = {m: (CS_[m] - I * m * w * xi * D[m]) / (xi * xi) for m in ODD}
    E = {k: v / 2 for k, v in proj(D, D, EVEN).items()}
    CE = proj(C, E, EVEN)
    xdD = {m: xi * dD[m] for m in ODD}
    a, b, c = proj(C, xdD, ODD), proj(CE, D, ODD), proj(C, D, ODD)
    dS = {m: (a[m] + b[m] - c[m] - I * m * w * xi * S[m]) / (xi * xi) for m in ODD}
    dC = {v: CE[v] / xi for v in EVEN}
    return [dD[m] for m in ODD] + [dS[m] for m in ODD] + [dC[v] for v in EVEN]


def path_ref(kind, r0, r1, phi):
    I = mpc(0, 1)
    if kind == "ray":
        u = mpmath.exp(I * phi)
        return (lambda t: (r0 + (r1 - r0) * t) * u), (lambda t: (r1 - r0) * u)
    if kind == "real":
        return (lambda t: mpc(r0 + (r1 - r0) * t)), (lambda t: mpc(r1 - r0))
    return (lambda t: r1 * mpmath.exp(I * phi * (1 - t))), (lambda t: -I * phi * r1 * mpmath.exp(I * phi * (1 - t)))


_CACHE = {}


def flow_ref(kind, r0, r1, phi, t0, h, y0, w):
    key = (kind, str(t0), str(h), tuple(str(v) for v in y0), str(w))
    if key in _CACHE:
        return _CACHE[key]
    xi, dxi = path_ref(kind, r0, r1, phi)
    # odefun integrates real and imaginary parts as separate real unknowns (robust for complex systems)
    n = len(y0)

    def f(t, z):
        Y = [mpc(z[i], z[n + i]) for i in range(n)]
        x = xi(t)
        Fv = F_ref(Y, x, w)
        dx = dxi(t)
        out = [v * dx for v in Fv]
        return [v.real for v in out] + [v.imag for v in out]
    sol = mpmath.odefun(f, t0, [mpf(v.real) for v in y0] + [mpf(v.imag) for v in y0])
    z = sol(t0 + h)
    _CACHE[key] = [mpc(z[i], z[n + i]) for i in range(n)]
    return _CACHE[key]


# ------------------------------------------------------------------ helpers

def to_mp(z):
    return mpc(mpf(z.real.mid().str(70, radius=False)), mpf(z.imag.mid().str(70, radius=False)))


def contains(ball, v):
    """flint acb ball contains the mpmath complex v (converted exactly enough at 70 digits; margin 1e-65 added)"""
    re = arb(arb(mpmath.nstr(v.real, 62)), arb("1e-55"))
    im = arb(arb(mpmath.nstr(v.imag, 62)), arb("1e-55"))
    return ball.real.overlaps(re) and ball.imag.overlaps(im), max(float(abs(ball.real.mid() - re.mid())),
                                                                   float(abs(ball.imag.mid() - im.mid())))


# ------------------------------------------------------------------ the test

def one(kind, N, NJ, t0, h, y0f, wf, mutate=None):
    ctx.prec = 256
    s = CS.SysM(M)
    phi = 5 * arb.pi() / 16
    legs = {"ray": CS.LegM("ray", "0.0741", "0.2", phi), "arc": CS.LegM("arc", "0.0741", "0.2", phi),
            "real": CS.LegM("real", "0.2", "0.3", phi)}
    leg = legs[kind]
    w = CS.W0() * arb(wf)
    c = [acb(complex(v)) for v in y0f] + [acb(w.mid())]
    n = s.n
    q = [acb(0)] * n
    Q = acb_mat([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    dirs = [s.idx[("D", 1)], s.idx[("S", -3)], s.nd]          # D_1, S_-3, w
    cV = acb_mat([[1 if i == j else 0 for j in dirs] for i in range(n)])

    def make():
        lo_ = CL.LohnerM(s, c, Q, q, N, (cV, Q, acb_mat(n, len(dirs))))
        lo_.NJ = NJ
        return lo_
    saved = {}
    if mutate == "noremainder":
        saved["coef"] = CL.VI.coef
        CL.VI.coef = lambda ser, k: acb(0)
    if mutate == "noJremainder":
        saved["tv"] = CL.taylor_V
        calls = {"k": 0}

        def tv(sys_, ctxv, NN):
            calls["k"] += 1
            V = saved["tv"](sys_, ctxv, NN)
            if calls["k"] % 2 == 0:                       # the second call of a step is the remainder's
                V[-1] = V[-1] * 0
            return V
        CL.taylor_V = tv
    try:
        ok = False
        for _ in range(12):                               # halve a refused step, as LohnerM.run does
            lo = make()
            ok = lo._step(leg, t0, h)
            if ok:
                break
            h = h / 2
    finally:
        if "coef" in saved:
            CL.VI.coef = saved["coef"]
        if "tv" in saved:
            CL.taylor_V = saved["tv"]
    assert ok, f"step refused ({kind}, N = {N})"
    X = lo.box()
    Vb = lo.var_box()
    # reference
    r = {"ray": (mpf("0.0741"), mpf("0.2")), "arc": (mpf("0.0741"), mpf("0.2")), "real": (mpf("0.2"), mpf("0.3"))}[kind]
    phim = 5 * mp.pi / 16
    y0 = [to_mp(z) for z in c[:s.nd]]
    wm = to_mp(c[s.nd]).real
    t0m, hm = mpf(t0), mpf(h)
    yref = flow_ref(kind, r[0], r[1], phim, t0m, hm, y0, wm)
    st = [contains(X[i], yref[i]) for i in range(s.nd)]
    ok_state = all(a for a, _ in st)
    dev_state = max(d for _, d in st)
    rad_state = max(max(float(z.real.rad()), float(z.imag.rad())) for z in X[:s.nd])
    # derivative columns by central differences
    delta = mpf("1e-20")
    ok_cols, dev_cols, rad_cols = True, 0.0, 0.0
    for k, j in enumerate(dirs):
        if j < s.nd:
            yp = list(y0); ym = list(y0)
            yp[j] += delta; ym[j] -= delta
            fp = flow_ref(kind, r[0], r[1], phim, t0m, hm, yp, wm)
            fm = flow_ref(kind, r[0], r[1], phim, t0m, hm, ym, wm)
        else:
            fp = flow_ref(kind, r[0], r[1], phim, t0m, hm, y0, wm + delta)
            fm = flow_ref(kind, r[0], r[1], phim, t0m, hm, y0, wm - delta)
        col = [(fp[i] - fm[i]) / (2 * delta) for i in range(s.nd)]
        for i in range(s.nd):
            a, d = contains(Vb[i, k], col[i])
            ok_cols &= a
            dev_cols = max(dev_cols, d)
            rad_cols = max(rad_cols, float(Vb[i, k].real.rad()), float(Vb[i, k].imag.rad()))
    tag = f"[{mutate}]" if mutate else ""
    print(f"   {kind:4s} t0 = {t0}, h = {h:.3e}, N = {N}, NJ = {NJ} {tag}: state contains reference {ok_state} "
          f"(|mid - ref| {dev_state:.1e}, radius {rad_state:.1e}); columns contain FD reference {ok_cols} "
          f"(|mid - ref| {dev_cols:.1e}, radius {rad_cols:.1e})", flush=True)
    return ok_state, ok_cols


def main():
    import numpy as np
    rng = np.random.default_rng(3)
    # start states: the series values at the certificate's centre for the ray (a typical state there); O(1) random
    # states with C_0 near 1 elsewhere (the enclosure must hold for every state)
    ctx.prec = 256
    p = ["2.72647588563088", "1.01560309485773", "0.144419478470077", "0.0110874258881167"]
    _, vals, _, _ = CS.placeholder_start([arb(x) for x in p], M, "0.0741", 5 * arb.pi() / 16, dirs=[])
    y_ray = [complex(v.mid()) for v in vals]

    def rnd():
        y = list(0.5 * (rng.normal(size=ND) + 1j * rng.normal(size=ND)))
        y[2 * len(ODD) + EVEN.index(0)] = 1.0 + 0.1j
        return y
    cases = [("ray", 0.0, 2.0 ** -11, y_ray), ("arc", 0.5, 2.0 ** -9, rnd()), ("real", 0.25, 2.0 ** -7, rnd())]
    good = True
    print("K2: one LohnerM step at M = 3 against an independent 60-digit reference")
    for kind, t0, h, y in cases:
        for N, NJ in ((20, 12), (6, 4)):
            a, b = one(kind, N, NJ, t0, h, y, p[1])
            good &= a and b
    print("negative controls (must fire):")
    fired = True
    for kind, t0, h, y in cases[:1] + cases[2:]:
        a, _ = one(kind, 6, 4, t0, h, y, p[1], mutate="noremainder")
        fired &= not a
        _, b = one(kind, 6, 4, t0, h, y, p[1], mutate="noJremainder")
        fired &= not b
    print(f"K2: containments {'all hold' if good else 'FAIL'}; controls {'fire' if fired else 'DO NOT FIRE'}")
    return 0 if (good and fired) else 1


if __name__ == "__main__":
    sys.exit(main())
