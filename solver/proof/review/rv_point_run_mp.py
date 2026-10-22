#!/usr/bin/env python3
"""Review check R8 (agent RV): the certificate's point run against an independent high-precision integration.

validated_int.py's point run encloses D+_{-1}(xi1), D+_{+1}(xi1) at xi1 = 0.3 (the saved runs, radius ~3e-19). Here
y+ is recomputed with no code from solver/proof: the optimally truncated series of inner_stokes/lateral.py at
|xi| = 0.07 on the ray arg xi = 3 pi/8 (smallest term 4e-58, damped outward), then mpmath's odefun (Taylor method,
30 digits) along the ray to 0.3 and the arc |xi| = 0.3 down to the real axis (lateral.Model.integrate_path). The
result must lie in the saved enclosures (J_D is printed for comparison). A second reference at x^ + (1e-6, 0) and
(0, 1e-6) gives difference quotients that must lie in the saved derivative enclosures over X (they are evaluated at
points within 1e-6 of X; the enclosure radii are ~6e-4, so this checks orientation and column order, not tightness).

    python3 rv_point_run_mp.py [--no-derivatives]
"""
import json
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "inner_stokes"))
from lateral import Model  # noqa: E402

mp.mp.dps = 30


def yplus(A, wf, rs="0.07", r1="0.3"):
    w = 2 * mp.pi / mp.mpf("1.64389375") * mp.mpf(wf)
    Am = mp.mpf(A)
    m = Model({1: -1j * Am / 2, -1: 1j * Am / 2}, w, 1, gravity=True)
    rs, r1 = mp.mpf(rs), mp.mpf(r1)
    phi = 3 * mp.pi / 8
    d, s, g = m.series(int(1 / (2 * rs ** 2)) + 40)
    D, S, C, jt, err = m.series_value(d, s, g, rs * mp.expj(phi))
    Y = m.integrate_path(m.pack(D, S, C), [("ray", phi, rs, r1), ("arc", r1, phi, mp.mpf(0))])
    return Y[0], Y[1]                     # D_-1, D_+1


def ball(p):
    return (mp.mpc(mp.mpf(p[0].split("+/-")[0].strip("[ ")), mp.mpf(p[1].split("+/-")[0].strip("[ "))),
            mp.mpf(p[0].split("+/-")[1].strip(" ]")), mp.mpf(p[1].split("+/-")[1].strip(" ]")))


def inside(z, p, slack=mp.mpf(10) ** -24):
    c, rr, ri = ball(p)
    return abs(z.real - c.real) <= rr + slack and abs(z.imag - c.imag) <= ri + slack


if __name__ == "__main__":
    d = json.load(open(os.path.join(HERE, "..", "validated_int_krawczyk_1e-9.json")))
    t0 = time.time()
    Dm, Dp = yplus(d["A"], d["wf"])
    print(f"mpmath reference at x^ [{time.time() - t0:.0f} s]: D+_-1 = {mp.nstr(Dm, 20)}, D+_+1 = {mp.nstr(Dp, 20)}")
    print(f"   inside the saved point-run enclosures: D-1 {inside(Dm, d['point']['Dm1'])}, D+1 {inside(Dp, d['point']['Dp1'])}")
    cm, _, _ = ball(d["point"]["Dm1"])
    cp, _, _ = ball(d["point"]["Dp1"])
    print(f"   distance to the enclosure midpoints: {mp.nstr(abs(Dm - cm), 3)}, {mp.nstr(abs(Dp - cp), 3)} "
          f"(radii {d['point']['rad'][0]:.1e}, {d['point']['rad'][1]:.1e})")
    print(f"   J_D(x^) = {mp.nstr(Dp - mp.conj(Dm), 12)}")
    if "--no-derivatives" not in sys.argv:
        hd = mp.mpf("1e-6")
        for k, (dA, dw) in enumerate(((hd, 0), (0, hd))):
            t0 = time.time()
            Dm2, Dp2 = yplus(str(mp.mpf(d["A"]) + dA), str(mp.mpf(d["wf"]) + dw))
            qm, qp = (Dm2 - Dm) / hd, (Dp2 - Dp) / hd
            print(f"   d/d{'A' if k == 0 else 'wf'} (difference quotient, step 1e-6) [{time.time() - t0:.0f} s]: "
                  f"dD-1 {mp.nstr(qm, 8)} in saved enclosure: {inside(qm, d['box']['dDm1'][k], slack=mp.mpf('1e-5'))}; "
                  f"dD+1 {mp.nstr(qp, 8)} in saved enclosure: {inside(qp, d['box']['dDp1'][k], slack=mp.mpf('1e-5'))}")
