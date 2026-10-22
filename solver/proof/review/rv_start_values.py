#!/usr/bin/env python3
"""Review check R5 (agent RV): the step-(i) enclosure of y+(xi0) against an independent high-precision computation.

fixed_point.start_values encloses y+(xi0), xi0 = 0.1 e^{3 i pi/8}, to radius ~6e-34 at the zero. Independently: start
much closer to the origin on the same ray, at |xi| = rs (0.06 and 0.07), from the optimally truncated series of
inner_stokes/lateral.py (mpmath), whose smallest terms there are 1e-74 and 4e-58; integrate
outward to |xi| = 0.1 with mpmath's odefun at 70 digits. On this ray the fast modes decay outward (Re 1/(2 xi^2) < 0),
so the start error is damped, and the solution asymptotic to the series in this direction is y+ (the uniqueness
that step (i) proves). Both references must lie in the fixed-point enclosure and agree with each other far below
its radius.

    python3 rv_start_values.py
"""
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "inner_stokes"))
import fixed_point as FP  # noqa: E402
from lateral import Model  # noqa: E402

mp.mp.dps = 70


def reference(rs, A="2.5421480475", wf="0.7769940136", r0="0.1"):
    w = 2 * mp.pi / mp.mpf("1.64389375") * mp.mpf(wf)
    Am = mp.mpf(A)
    m = Model({1: -1j * Am / 2, -1: 1j * Am / 2}, w, 1, gravity=True)
    rs = mp.mpf(rs)
    phi = 3 * mp.pi / 8
    u = mp.expj(phi)
    J = int(1 / (2 * rs ** 2)) + 40
    d, s, g = m.series(J)
    D, S, C, jt, err = m.series_value(d, s, g, rs * u)
    Y0 = m.pack(D, S, C)
    f = lambda r, Z: [v * u for v in m.rhs(r * u, Z)]
    sol = mp.odefun(f, rs, Y0)
    return sol(mp.mpf(r0)), jt, err


if __name__ == "__main__":
    from flint import ctx
    ctx.prec = 320
    out = FP.run(FP.acb(FP.A_STAR), FP.acb(FP.WF_STAR), "0.1", verbose=False)
    xi0, Dv, Sv, Cv = FP.start_values(out, +1)
    enc = [Dv[-1], Dv[1], Sv[-1], Sv[1], Cv[-2], Cv[0], Cv[2]]          # lateral.py's pack order
    names = ["D-1", "D+1", "S-1", "S+1", "C-2", "C0", "C+2"]
    refs = []
    for rs in ("0.07", "0.06"):
        t0 = time.time()
        Y, jt, err = reference(rs)
        refs.append(Y)
        inside = []
        for z, y in zip(enc, Y):
            mr, mi = mp.mpf(z.real.mid().str(90, radius=False)), mp.mpf(z.imag.mid().str(90, radius=False))
            rr, ri = mp.mpf(z.real.rad().str(20, radius=False)), mp.mpf(z.imag.rad().str(20, radius=False))
            inside.append(abs(y.real - mr) <= rr and abs(y.imag - mi) <= ri)
        dist = max(abs(y - mp.mpc(mp.mpf(z.real.mid().str(90, radius=False)), mp.mpf(z.imag.mid().str(90, radius=False))))
                   for z, y in zip(enc, Y))
        print(f"start at |xi| = {rs} (series {jt} terms, smallest term {mp.nstr(err, 3)}), integrated to 0.1: inside the "
              f"step-(i) enclosure: {dict(zip(names, inside))}; max |ref - enclosure midpoint| = {mp.nstr(dist, 3)} "
              f"[{time.time() - t0:.0f} s]", flush=True)
    agree = max(abs(a - b) for a, b in zip(*refs))
    rad = max(max(float(z.real.rad()), float(z.imag.rad())) for z in enc)
    print(f"the two references agree to {mp.nstr(agree, 3)}; enclosure radius {rad:.2e}")
