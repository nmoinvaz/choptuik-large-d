#!/usr/bin/env python3
"""Delta(D) from hp/data/formula_cut.json (hp/formula_cut_fit.py), two ways, and the checks that it reproduces the data.

Series: Delta(D) = sum_k c_k T_k(2 zeta - 1), zeta = u - sqrt(u^2 - 1), u = (2 (D - 2) - (1 + n_s)) / (1 - n_s): zeta
maps the D-plane cut along [D_s, 3] (D_s = n_s + 2) onto the unit disc, 0 at D = infinity, 1 at D = 3.
Integral (the same function, Cauchy's formula for a function analytic off the cut and finite at infinity):

    Delta(D) = Delta_inf + (1/pi) * integral_{D_s}^{3} rho(D') / (D' - D) dD',   rho(D') = Im Delta(D' + i0),

with rho the series' value on the cut's upper side, zeta = e^{-i theta}, D' = (D_s + 3)/2 + (3 - D_s)/2 cos(theta).
rho is that continuation onto the cut, not a measured quantity: it is large and oscillates (the series' continuation
to |zeta| = 1, where the data, at zeta <= 0.88, do not reach).

Checks (exit 1 if any fails): (1) every fitted point within --max_sigma (3.5) of its sigma; (2) the integral equals
the series to 1e-25 at D = 3.1, 4, 10, 30, 67.77, 100, 1000; (3) Delta_inf and Delta(3) from the coefficients match the
stored values. --mutate adds 1e-12 to c_10: check (1) must fail (the control).
    cd hp && python3 formula_cut.py [--table] [--mutate] [--max_sigma 3.5]
"""
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
mp.mp.dps = 60
J = json.load(open(os.path.join(HERE, "data", "formula_cut.json")))
C = [mp.mpf(v) for v in J["c"]]
if "--mutate" in sys.argv:
    C[10] += mp.mpf("1e-12")
NS = mp.mpf(J["n_s"])
DS = NS + 2
MAXS = float(sys.argv[sys.argv.index("--max_sigma") + 1]) if "--max_sigma" in sys.argv else 3.5


def zeta(D):
    u = (2 * (mp.mpf(D) - 2) - (1 + NS)) / (1 - NS)
    r = mp.sqrt(u * u - 1)
    z1, z2 = u - r, u + r
    return z1 if abs(z1) <= abs(z2) else z2              # the root inside the disc


def series_z(z):
    x = 2 * z - 1
    t0, t1, s = mp.mpf(1), x, C[0] + C[1] * x
    for c in C[2:]:
        t0, t1 = t1, 2 * x * t1 - t0
        s += c * t1
    return s


def delta(D):
    return series_z(zeta(D))


def rho(theta):
    """Im Delta on the cut's upper side: zeta = e^{-i theta}"""
    return mp.im(series_z(mp.expjpi(-theta / mp.pi)))


def delta_integral(D):
    a, b = (DS + 3) / 2, (3 - DS) / 2
    f = lambda th: rho(th) / (a + b * mp.cos(th) - mp.mpf(D)) * b * mp.sin(th)  # noqa: E731
    return series_z(mp.mpf(0)) + mp.quad(f, mp.linspace(0, mp.pi, 9)) / mp.pi


def main():
    from formula_cut_fit import points
    pts = points()
    drop = set(J["dropped"])
    worst, n = (0.0, None), 0
    for d, (hi, lo, s, run) in sorted(pts.items()):
        if d in drop:
            continue
        n += 1
        r = float((delta(d) - mp.mpf(hi) - mp.mpf(lo)) / s)
        if abs(r) > abs(worst[0]):
            worst = (r, d)
    ok1 = abs(worst[0]) <= MAXS
    for a, b in ((3.0, 3.1), (3.1, 4.01), (4.01, 8.3), (8.3, 30), (30, 68)):
        sel = [(d, v) for d, v in sorted(pts.items()) if a <= d < b and d not in drop]
        rr = [abs(float(delta(d) - mp.mpf(v[0]) - mp.mpf(v[1]))) for d, v in sel]
        print(f"    D {a:g}-{b:g}: {len(sel)} points, largest |model - point| {max(rr):.1e}")
    import csv
    m48 = [(float(r["D"]), float(r["Delta_hi"]) + float(r["Delta_lo"] or 0))
           for r in csv.DictReader(l for l in open(os.path.join(HERE, "data", "delta_dd_lowD.csv")) if not l.startswith("#"))
           if r["M"] == "48" and r["resolved"] == "0" and 4.0 < float(r["D"]) < 8.25 and float(r["D"]) not in (5.0, 6.0, 8.0)]
    gap = max(m48, key=lambda t: abs(float(delta(t[0])) - t[1]))
    print(f"    the unresolved M 48 rows 4.25-8.2 (not fitted, accuracy unmeasured): largest |model - row| "
          f"{abs(float(delta(gap[0])) - gap[1]):.1e} at D {gap[0]}")
    print(f"(1) {n} points: largest |model - point| / sigma {abs(worst[0]):.2f} at D {worst[1]}, against {MAXS}: {'PASS' if ok1 else 'FAIL'}")
    ok2 = True
    for D in (3.1, 4, 10, 30, 67.77, 100, 1000):
        s, i = delta(D), delta_integral(D)
        e = abs(s - i)
        ok2 &= e <= mp.mpf("1e-25")
        print(f"(2) D {D:>7}: series {mp.nstr(s, 16)}, integral {mp.nstr(i, 16)}, |difference| {mp.nstr(e, 3)}")
    print(f"(2) integral = series to 1e-25: {'PASS' if ok2 else 'FAIL'}")
    dinf, d3 = series_z(mp.mpf(0)), series_z(mp.mpf(1))
    ok3 = abs(dinf - mp.mpf(J["Delta_inf"])) < mp.mpf("1e-18") and abs(d3 - mp.mpf(J["Delta_3"])) < mp.mpf("1e-10")
    print(f"(3) Delta_inf {mp.nstr(dinf, 16)}, Delta(3) {mp.nstr(d3, 8)} (stored {J['Delta_inf'][:18]}, {J['Delta_3'][:10]}): {'PASS' if ok3 else 'FAIL'}")
    th = [mp.pi * k / 400 for k in range(1, 400)]
    rv = [rho(t) for t in th]
    signs = sum(1 for a, b in zip(rv, rv[1:]) if a * b < 0)
    print(f"rho on the cut: max |rho| {mp.nstr(max(abs(v) for v in rv), 3)}, {signs} sign changes (the series' continuation, not data)")
    if "--table" in sys.argv:
        print("| D | Delta (series) |")
        print("|---|---|")
        for D in (3.04375, 3.5, 4, 5, 6, 8, 10, 20, 30, 60, 67.77, 100, 1000):
            print(f"| {D} | {mp.nstr(delta(D), 15)} |")
    sys.exit(0 if ok1 and ok2 and ok3 else 1)


if __name__ == "__main__":
    main()
