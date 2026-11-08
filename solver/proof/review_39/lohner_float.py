#!/usr/bin/env python3
"""A double-precision control of the lemma 6b1(5) disc inputs (#39 review; NOT rigorous).

The regenerated Lohner boxes (dump_inputs.py) must contain the flow of every point of the arc's start box. Here, for
each given row, the midpoint of the box at the arc's start (disc 0, the ray's state at rho_j) is integrated again in
double precision along the arc |u| = rho_j (b_float.rhs_dual, the review's numpy transcription; DOP853) to the disc
centres. Reported per disc: |y_float - mid(box)| against rad(box) + tol. The comparison only discriminates where the
box is much wider than the double-precision error (rows 15-17 near the real axis). A first version integrated the start
ray too, from s0 = 405 e^{-3 i pi/4}: its double-precision error at rho_17 (1.8e-12) exceeds the ray box there
(1.2e-12), and the arc amplifies both alike, so that version could not discriminate.

    python3 lohner_float.py --ray=ray.json --inputs=<disc_inputs dir> [--rows=17,16,15] [--mutate=w]
        --mutate=w: w scaled by 1 + 1e-6: the wide boxes must then miss
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b_float as BF  # noqa: E402

E3 = np.exp(-3j * np.pi / 4)


def pmid(s):
    return float(s.strip("[]").split("+/-")[0])


def me_mid(p):
    """complex midpoint of an exact [[man, exp], [man, exp]] pair"""
    return complex(p[0][0][0] * 2.0 ** p[0][0][1], p[1][0][0] * 2.0 ** p[1][0][1])


def me_rad(p):
    return max(p[0][1][0] * 2.0 ** p[0][1][1], p[1][1][0] * 2.0 ** p[1][1][1])


def integrate(y0, w, path, t0, t1, ts):
    """dy/dt = F(u(t), y) u'(t) along u = path(t); values at the parameters ts"""
    def fun(t, y):
        u, du = path(t)
        F, _ = BF.rhs_dual(y, u, w)
        return F * du
    sol = solve_ivp(fun, (t0, t1), y0, method="DOP853", rtol=1e-13, atol=1e-15, t_eval=ts, dense_output=False)
    return sol.y


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    with open(opts["ray"]) as fh:
        ray = json.load(fh)
    nd = BF.ND
    w = pmid(ray["starts"]["0"][nd][0])
    if opts.get("mutate") == "w":
        w *= 1 + 1e-6                                       # MUTATION w
    rows = [int(v) for v in opts.get("rows", "17,16,15").split(",")]
    tol = 1e-9
    worst_in, nmiss, ndisc, ndiscr = 0.0, 0, 0, 0
    for j in rows:
        with open(os.path.join(opts["inputs"], f"row_{j:02d}.json")) as fh:
            inp = json.load(fh)
        rho = inp["rho"]
        phis = [-(3 * np.pi / 4) * (1 - t) for t in inp["t"]]
        path = lambda p, r=rho: (r * np.exp(1j * p), 1j * r * np.exp(1j * p))  # noqa: E731
        y0 = np.array([me_mid(p) for p in inp["discs"][0]["Y0"]])
        ya = integrate(y0, w, path, phis[0], phis[-1], phis)
        for k, d in enumerate(inp["discs"]):
            mids = np.array([me_mid(p) for p in d["Y0"]])
            rads = np.array([me_rad(p) for p in d["Y0"]])
            dev = np.abs(ya[:, k] - mids)
            ratio = (dev / (rads + tol)).max()
            disc_ = rads.max() > 100 * tol
            ndisc += 1
            ndiscr += disc_
            inside = ratio <= 1
            nmiss += not inside
            worst_in = max(worst_in, ratio)
            print(f"row {j} disc {k}: max |y_float - mid| / (rad + 1e-9) = {ratio:.3g}, box radius {rads.max():.2e}"
                  f"{' (discriminating)' if disc_ else ''}: {'inside' if inside else 'OUTSIDE'}", flush=True)
    mut = opts.get("mutate") == "w"
    print(f"{ndisc} discs, {ndiscr} with boxes wider than 100 x tol; outside: {nmiss}"
          + (f"   [MUTATION w: the discriminating boxes must miss]" if mut else "   (must be 0)"))
    if mut:
        return 0 if nmiss >= ndiscr > 0 else 3
    return 0 if nmiss == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
