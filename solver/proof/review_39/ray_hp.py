#!/usr/bin/env python3
"""A high-precision control of the lemma 6b1(5) start-ray boxes (#39 review; NOT rigorous: the Taylor truncation is
estimated, not bounded).

y+ at the centre is stepped inward along the start ray u = t e^{-3 i pi/4}, t from 405 to the row radii, by Taylor
series of order N at each point (the review's coefficient recursion, check_6b1.LZ, at 192 bits), from the midpoint of
the stored start box at s0. Each step is a third of the radius estimated from the coefficients, and the last kept
term must be below 1e-40. At every row radius the value is compared with the stored ray box (ray.json "starts").

    python3 ray_hp.py --ray=ray.json [--to=17] [--mutate=w]      (w x (1 + 1e-12): the boxes must miss)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import civ as C  # noqa: E402
import check_6b1 as K  # noqa: E402
from mpmath.libmp import from_str, round_floor, round_ceiling, to_float, from_float  # noqa: E402
import mpmath  # noqa: E402


def ball(s):
    """'[mid +/- rad]' -> (mid interval, rad float)"""
    s = s.strip("[]")
    if "+/-" in s:
        m, r = s.split("+/-")
    else:
        m, r = s, "0"
    p = C.PREC
    return (from_str(m.strip(), p, round_floor), from_str(m.strip(), p, round_ceiling)), float(r)


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    C.setprec(192)
    mpmath.mp.prec = 192
    with open(opts["ray"]) as fh:
        ray = json.load(fh)
    nd = 19
    st = ray["starts"]["0"]
    y = [(ball(p[0])[0], ball(p[1])[0]) for p in st[:nd]]
    w = (ball(st[nd][0])[0], ball(st[nd][1])[0])
    if opts.get("mutate") == "w":
        w = C.cmulr(w, C.ri_dec("1.000000000001"))           # MUTATION w x (1 + 1e-12)
    E3 = C.cexpi(C.rneg(C.rmul(C.rdiv(C.ri(3), C.ri(4)), C.rpi())))
    rhos = ray["rho"]
    jto = int(opts.get("to", str(len(rhos) - 1)))
    N = 40
    t = 405.0
    miss = 0
    for j in range(1, jto + 1):
        target = rhos[j]
        while t > target + 1e-12:
            c = C.cmulr(E3, C.ri_dec(repr(t)))
            zL = K.LZ("z", C.cinv(c))
            z2L = K.LZRing.pairs([(zL, zL)])
            sts = [K.LZ("state", y[i], [None]) for i in range(nd)]
            F = K.rhs_generic(5, sts, zL, z2L, w, K.LZRing)
            for i in range(nd):
                sts[i].b[0] = F[i]
            for k in range(N + 1):
                for i in range(nd):
                    sts[i].get(k)
            # radius estimate from the last coefficients, a third of it, landing on the row radius
            mags = [max(to_float(C.cabsup(sts[i].get(k))[1]) for i in range(nd)) for k in (N - 4, N)]
            R = (mags[0] / mags[1]) ** 0.25 if mags[1] > 0 else 1e9
            h = min(R / 3, t - target, 25.0)
            tail = mags[1] * h ** N
            if tail > 1e-40:
                h = h * (1e-40 / tail) ** (1.0 / N)
            tn = target if abs(t - h - target) < 1e-9 else t - h
            tau = C.cmulr(E3, C.rneg(C.rsub(C.ri_dec(repr(t)), C.ri_dec(repr(tn)))))
            ynew = []
            for i in range(nd):
                acc = sts[i].get(N)
                for k in range(N - 1, -1, -1):
                    acc = C.cadd(C.cmul(acc, tau), sts[i].get(k))
                ynew.append(acc)
            y, t = ynew, tn
        box = ray["starts"][str(j)]
        worst = 0.0
        for i in range(nd):
            for q in (0, 1):
                m, r = ball(box[i][q])
                dev = max(abs(to_float(C.rsub(y[i][q], m)[0])), abs(to_float(C.rsub(y[i][q], m)[1])))
                worst = max(worst, dev / r if r > 0 else float("inf"))
        brad = max(ball(box[i][q])[1] for i in range(nd) for q in (0, 1))
        miss += worst > 1
        print(f"row {j} (|u| = {target}): max |y_hp - mid(box)| / rad(box) = {worst:.3g}, box radius {brad:.2e}: "
              f"{'inside' if worst <= 1 else 'OUTSIDE'}", flush=True)
    mut = opts.get("mutate") == "w"
    print(f"rows outside: {miss} of {jto}" + ("   [MUTATION w: must be > 0]" if mut else "   (must be 0)"))
    return (0 if miss > 0 else 3) if mut else (0 if miss == 0 else 3)


if __name__ == "__main__":
    sys.exit(main())
