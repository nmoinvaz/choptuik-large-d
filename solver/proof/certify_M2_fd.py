#!/usr/bin/env python3
"""Second derivatives of the D-jump function by central differences of the C^1 numerics (proof/A_M.md section 10).
NOT rigorous, and independent of the second-order (C^2) code: it calls only certify_M.numerics_point (the plain
256-bit Taylor integrator of certify_M_lohner.numerics_run, with the derivative columns of the first variational
equation) at points p^ + s delta_k T_k of the first-order chart. Used for three things:
  * the curvature Gamma of the second-order chart (any Gamma is allowed: the test is what is rigorous);
  * the scale estimates of the design (the size of Y D^2 gt);
  * the control D2 of the C^2 run (certify_M2.py d2check): the rigorous enclosure of D^2 gt over the box must contain
    this independent estimate of D^2 gt(0).

    python3 certify_M2_fd.py run M --centre=refine.json --use_out --k=K --s=S --delta=D --out=f.json
        one numerics point at p^ + S D T_K (K = 0..M; S = +1, -1, +2, -2): g and Dg in raw p-coordinates
    python3 certify_M2_fd.py combine M --centre=refine.json --use_out --glob=PREFIX [--out=f.json]
        the central differences D2[i][j][k] = d^2 gt_i / dx_j dx_k at 0 (first-order chart), from the +-delta points
        (second order) and the +-2 delta points (Richardson); the error estimate is |D2(delta) - D2(2 delta)| / 3
"""
import glob as globmod
import json
import os
import sys
import time

import numpy as np
from flint import arb, arb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_M as CM  # noqa: E402

DATA = os.path.join(HERE, "data")


def chart1(centre_file, use_out, M):
    """p^ (strings) and T (exact arb rows) of the first-order chart, from a refinement record (or, for a file whose
    name contains '_point_', a point run: certify_M.centre_and_tilt's --point)"""
    o = {"point": centre_file} if "_point_" in os.path.basename(centre_file) else {"centre": centre_file}
    if use_out:
        o["use_out"] = True
    pstr, Cm = CM.centre_and_tilt(o, M)
    return pstr, CM.T_matrix(Cm, M)


def run_point(M, centre_file, use_out, k, s, delta, out):
    ctx.prec = CM.KPREC
    pstr, T = chart1(centre_file, use_out, M)
    p = CM.exact_p(pstr)
    ctx.prec = CM.KPREC
    d = arb(delta) * s
    pp = [p[i] + d * T[i][k] for i in range(M + 1)]
    ps = [x.mid().str(60, radius=False) for x in pp]
    t0 = time.time()
    g, Dg, steps = CM.numerics_point(M, ps)
    res = {"M": M, "k": k, "s": s, "delta": delta, "p": ps, "p_hat": pstr, "g": [x.str(40) for x in g],
           "Dg": [[x.str(40) for x in r] for r in Dg], "steps": steps, "sec": time.time() - t0,
           "note": "numerics (certify_M.numerics_point), not rigorous"}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"fd point M = {M} k = {k} s = {s:+d} delta = {delta}: {sum(steps)} steps, {time.time() - t0:.0f} s -> {out}")


def combine(M, centre_file, use_out, prefix, out=None, verbose=True):
    """D2 (m x m x m, numpy float) and its error estimate from the saved points; D1 = Dgt(0) estimate"""
    m = M + 1
    ctx.prec = CM.KPREC
    _, T = chart1(centre_file, use_out, M)
    Tn = np.array([[float(T[i][j].mid()) for j in range(m)] for i in range(m)])
    # the differences are taken in arb at KPREC (Dgt has entries up to 1e14 and the inner steps are 1e-22: in double
    # precision the rounding of Dgt alone would be 1e19 in D2); only the results are converted to floats
    Tm = arb_mat(T)
    pts = {}
    for f in globmod.glob(prefix + "*.json"):
        d = json.load(open(f))
        Dg = arb_mat([[arb(arb(x).mid()) for x in r] for r in d["Dg"]])
        pts[(d["k"], d["s"])] = (arb(d["delta"]), Dg * Tm)

    def tonp(A):
        return np.array([[float(A[i, j].mid()) for j in range(m)] for i in range(m)])
    D2 = np.full((m, m, m), np.nan)
    err = np.full((m, m, m), np.nan)
    for k in range(m):
        if (k, 1) not in pts or (k, -1) not in pts:
            continue
        dl, Gp = pts[(k, 1)]
        _, Gm = pts[(k, -1)]
        c1 = (Gp - Gm) * (1 / (2 * dl))
        if (k, 2) in pts and (k, -2) in pts:
            _, Gp2 = pts[(k, 2)]
            _, Gm2 = pts[(k, -2)]
            c2 = (Gp2 - Gm2) * (1 / (4 * dl))
            D2[:, :, k] = tonp((4 * c1 - c2) * arb(1) / 3)      # Richardson: fourth order
            err[:, :, k] = np.abs(tonp(c1 - c2)) / 3            # the size of the second-order term (estimate)
        else:
            D2[:, :, k] = tonp(c1)
    res = {"M": M, "centre": centre_file, "D2": D2.tolist(), "err": err.tolist(), "T": Tn.tolist(),
           "points": sorted([list(k) for k in pts]), "note": "numerics (central differences), not rigorous"}
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
    if verbose:
        np.set_printoptions(precision=3, linewidth=160)
        for k in range(m):
            print(f"D2[:, :, {k}] (d/dx_{k} of Dgt, first-order chart):\n{D2[:, :, k]}")
            print(f"   error estimate (max) {np.nanmax(err[:, :, k]) if not np.all(np.isnan(err[:, :, k])) else float('nan'):.2e}")
    return D2, err


def structure(M, centre_file, fdfile, chart_file=None):
    """the scale estimates of the design (proof/A_M.md 10.4), numerics: Y D^2 gt at 0 (Y = Dgt(0)^-1, Dgt from the
    refinement's Dg times T), in the first-order chart, or in the second-order chart of chart_file, as the largest
    entry per (row block, pair block); and the box limits they imply if the enclosures were exact: an outer
    half-width h needs |Y D^2|_(o,oo) h < 1 (outer rows) and 2 |Y D^2|_(z,oz) h < 1 (inner rows); an inner width
    needs 2 |Y D^2|_(o,oH) w < 1 and |Y D^2|_(z,HH) w < 1."""
    m = M + 1
    fd = json.load(open(fdfile))
    D2 = np.array(fd["D2"])
    T = np.array(fd["T"])
    with open(centre_file) as fh:
        rec = json.load(fh)
    Dg = np.array([[float(arb(x).mid()) for x in r] for r in rec["history"][-1]["Dg"]])
    D1 = Dg @ T
    if chart_file:
        G = [[float(x) for x in r] for r in json.load(open(chart_file))["Gamma"]]
        for j in range(2):
            for k in range(2):
                for i in range(2, m):
                    D2[:, j, k] += D1[:, i] * G[i - 2][j + k]
    Y = np.linalg.inv(D1)
    YD2 = np.einsum("ia,ajk->ijk", Y, D2)
    blocks = {"o": [0, 1]}
    for nu in range(3, M + 1, 2):
        blocks[f"z{nu}"] = [2 + (nu - 3), 3 + (nu - 3)]
    names = list(blocks)
    print(f"M = {M}: max |Y D^2 gt| per (row block, pair block) at 0 ({'second' if chart_file else 'first'}-order chart)")
    print("   rows / pairs  " + "  ".join(f"{a + b:>9s}" for ai, a in enumerate(names) for b in names[ai:]))
    for r in names:
        vals = [np.abs(YD2[np.ix_(blocks[r], blocks[a], blocks[b])]).max()
                for ai, a in enumerate(names) for b in names[ai:]]
        print(f"   {r:13s} " + "  ".join(f"{v:9.2e}" for v in vals))
    oo = np.abs(YD2[np.ix_([0, 1], [0, 1], [0, 1])]).max()
    zoz = 0.0
    for z in names[1:]:
        for zz in names[1:]:
            zoz = max(zoz, np.abs(YD2[np.ix_(blocks[z], [0, 1], blocks[zz])]).max())
    print(f"   outer rows: h < 1/|Y D^2|_(o,oo) = {1 / oo:.2e}; inner rows: h < 1/(2 max |Y D^2|_(z,oz)) = "
          f"{(1 / (2 * zoz)) if zoz else float('inf'):.2e}")
    for zz in names[1:]:
        a = np.abs(YD2[np.ix_([0, 1], [0, 1], blocks[zz])]).max()
        b = max(np.abs(YD2[np.ix_(blocks[z], blocks[zz], blocks[zz])]).max() for z in names[1:])
        print(f"   inner width {zz}: w < 1/(2 |Y D^2|_(o,o{zz})) = {1 / (2 * a):.2e} and < 1/|Y D^2|_(z,{zz}{zz}) = "
              f"{1 / b:.2e}")
    return YD2


def opts_of(argv):
    return {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else True) for a in argv if a.startswith("--")}


if __name__ == "__main__":
    argv = sys.argv[1:]
    o = opts_of(argv)
    pos = [a for a in argv[1:] if not a.startswith("--")]
    M = int(pos[0])
    if argv[0] == "run":
        run_point(M, o["centre"], "use_out" in o, int(o["k"]), int(o["s"]), o["delta"], o["out"])
    elif argv[0] == "combine":
        combine(M, o["centre"], "use_out" in o, o["glob"], o.get("out"))
    elif argv[0] == "structure":
        structure(M, o["centre"], o["fd"], o.get("chart"))
