#!/usr/bin/env python3
"""Review check R3 (agent RV): one Lohner step of validated_int.py against an independent high-precision solution.

For a step on the ray and a step on the arc, from a point state:
  * the enclosure of the state after the step (Lohner box) must contain the solution computed by mpmath's odefun
    (Taylor method, 50 digits) with an independently written right-hand side (inner_stokes/lateral.py's Model.rhs)
    and an independently written path xi(t) and chain-rule factor xi'(t);
  * the enclosure J of the step Jacobian (rebuilt from the same internals step() uses) must contain the
    central-difference Jacobian of the mpmath flow, column by column (a few columns, to keep it light);
  * the Taylor order is lowered to N = 6 so the remainder term dominates the radius: the test then checks that the
    Lagrange-type remainder really covers the truncation error (ratio actual error / radius reported);
  * negative controls: with the remainder removed, or with the wrong sign of the arc's chain-rule factor, the
    containment must FAIL (so the test can see those errors).

    python3 rv_taylor_step.py
"""
import os
import sys
import time

import mpmath as mp
from flint import acb, acb_mat, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "inner_stokes"))
import validated_int as VI  # noqa: E402
from lateral import Model  # noqa: E402

mp.mp.dps = 50
A_S, WF_S = "2.5421480475", "0.7769940136"
R0, R1 = "0.1", "0.3"


def to_mp(z):
    return mp.mpc(mp.mpf(z.real.mid().str(80, radius=False)), mp.mpf(z.imag.mid().str(80, radius=False)))


def mp_flow(kind, t0, h, Y0, w, sign=-1):
    """independent reference: dY/dt = F(xi(t), Y) xi'(t) along the leg, by mpmath odefun"""
    model = Model({1: -1j * mp.mpf(A_S) / 2, -1: 1j * mp.mpf(A_S) / 2}, mp.re(w), 1, gravity=True)
    phi0 = 3 * mp.pi / 8
    r0, r1 = mp.mpf(R0), mp.mpf(R1)
    if kind == "ray":
        u = mp.expj(phi0)

        def f(t, Z):
            xi = (r0 + (r1 - r0) * t) * u
            return [v * (r1 - r0) * u for v in model.rhs(xi, Z)]
    else:
        def f(t, Z):
            xi = r1 * mp.expj(phi0 * (1 - t))
            return [v * (sign * 1j * phi0) * xi for v in model.rhs(xi, Z)]
    sol = mp.odefun(f, mp.mpf(t0), list(Y0))
    return sol(mp.mpf(t0) + h)


def lohner_parts(lo, leg, t0, h):
    """X after one step (box) and the step Jacobian enclosure J, from the internals of Lohner.step"""
    s, N = lo.s, lo.N
    X = lo.box()
    Yt, T = VI.apriori(s, leg, t0, h, X)
    Vt = VI.apriori_V(s, leg, T, Yt, h)
    hh = acb(h)
    _, ctxX = VI.taylor_Y(s, leg, acb(t0), X, N)
    Vk = VI.taylor_V(s, ctxX, N)
    _, ctr = VI.taylor_Y(s, leg, T, Yt, N + 1)
    VR = VI.taylor_V(s, ctr, N + 1)
    J = Vk[N]
    for k in range(N - 1, -1, -1):
        J = J * hh + Vk[k]
    return J + hh ** (N + 1) * (VR[N + 1] * Vt)


def contains_mp(z, ref, slack=mp.mpf(10) ** -45):
    """acb z contains the mpmath number ref (up to the reference's own accuracy, slack)"""
    re, im = z.real, z.imag
    return (abs(ref.real - to_mp(acb(re.mid())).real) <= mp.mpf(re.rad().str(30, radius=False)) + slack and
            abs(ref.imag - to_mp(acb(im.mid())).real) <= mp.mpf(im.rad().str(30, radius=False)) + slack)


def one(kind, t0, h, Nord, drop_remainder=False, arc_sign=-1, cols=(1, 2, 5, 7)):
    ctx.prec = VI.PREC
    s = VI.Sys(gravity=True)
    A = arb(A_S)
    w = VI.W0 * arb(WF_S)
    n = VI.optimal_n(arb(A.mid()), arb(w.mid()), R0)
    Y0 = VI.series_value(acb(A.mid()), acb(w.mid()), R0, n)
    c = [acb(y.mid()) for y in Y0] + [acb(w.mid()), acb(A.mid())]          # an exact point state
    Q = acb_mat([[1 if i == j else 0 for j in range(s.n)] for i in range(s.n)])
    lo = VI.Lohner(s, c, Q, [acb(0)] * s.n, Nord)
    leg = VI.Leg(kind, R0, R1)
    old_coef = VI.coef
    if drop_remainder:
        VI.coef = lambda ser, k: acb(0) if k == Nord + 1 else old_coef(ser, k)
    old_dxi, old_series = VI.Leg.dxi, VI.Leg.series
    if arc_sign == +1:                       # deliberately wrong chain-rule factor on the arc (negative control)
        VI.Leg.dxi = lambda self, xi: (I_ * VI.PHI0 * xi if self.kind == "arc" else old_dxi(self, xi))

        def ser(self, t0_, L):
            xs, dxs = old_series(self, t0_, L)
            return (xs, -dxs) if self.kind == "arc" else (xs, dxs)
        VI.Leg.series = ser
    try:
        J = lohner_parts(lo, leg, t0, h) if not drop_remainder and arc_sign == -1 else None
        ok_step = lo.step(leg, t0, h)
    finally:
        VI.coef, VI.Leg.dxi, VI.Leg.series = old_coef, old_dxi, old_series
    assert ok_step
    Xn = lo.box()
    Ym = [to_mp(y) for y in c[:s.nd]]
    wm = to_mp(c[s.nd])
    t1 = time.time()
    ref = mp_flow(kind, t0, h, Ym, wm, sign=-1)
    inside = [contains_mp(Xn[i], ref[i]) for i in range(s.nd)]
    err = max(abs(ref[i] - to_mp(acb(Xn[i].mid()))) for i in range(s.nd))
    rad = max(max(float(x.real.rad()), float(x.imag.rad())) for x in Xn[:s.nd])
    out = dict(inside=all(inside), err=float(err), rad=rad, sec=time.time() - t1)
    if J is not None:
        # central differences of the mpmath flow, a few columns (state index j; j = 7 is w)
        dl = mp.mpf(10) ** -14
        jac_ok = []
        for j in cols:
            if j < s.nd:
                Yp = list(Ym); Yp[j] += dl
                Ym_ = list(Ym); Ym_[j] -= dl
                fp, fm = mp_flow(kind, t0, h, Yp, wm), mp_flow(kind, t0, h, Ym_, wm)
            else:
                fp, fm = mp_flow(kind, t0, h, Ym, wm + dl), mp_flow(kind, t0, h, Ym, wm - dl)
            col = [(fp[i] - fm[i]) / (2 * dl) for i in range(s.nd)]
            jac_ok.append(all(contains_mp(J[i, j], col[i], slack=mp.mpf(10) ** -20) for i in range(s.nd)))
        out["jac_cols_ok"] = dict(zip(cols, jac_ok))
        out["J_rad"] = max(max(float(J[i, j].real.rad()), float(J[i, j].imag.rad())) for i in range(s.nd) for j in cols)
    return out


I_ = acb(0, 1)

if __name__ == "__main__":
    allok = True
    for kind, t0, h in (("ray", 0.0, 2.0 ** -10), ("arc", 0.5, 2.0 ** -8)):
        for Nord in (6, 24):
            r = one(kind, t0, h, Nord)
            print(f"{kind} step t0 = {t0}, h = {h}, Taylor order {Nord}: reference inside the Lohner box: {r['inside']}; "
                  f"|error of the midpoint| = {r['err']:.2e}, box radius {r['rad']:.2e}; "
                  f"Jacobian columns contain the central differences: {r['jac_cols_ok']} (J radius {r['J_rad']:.1e}) "
                  f"[{r['sec']:.0f} s]", flush=True)
            allok &= r["inside"] and all(r["jac_cols_ok"].values())
    neg = one("ray", 0.0, 2.0 ** -10, 6, drop_remainder=True)
    print(f"negative control, remainder dropped (ray, order 6): reference inside: {neg['inside']} (must be False); "
          f"error {neg['err']:.2e} vs radius {neg['rad']:.2e}")
    neg2 = one("arc", 0.5, 2.0 ** -8, 24, arc_sign=+1)
    print(f"negative control, wrong sign of the arc chain-rule factor: reference inside: {neg2['inside']} (must be False)")
    allok &= (not neg["inside"]) and (not neg2["inside"])
    print("VERDICT:", "Lohner step enclosures, remainder and chain rule confirmed" if allok else "PROBLEM")
