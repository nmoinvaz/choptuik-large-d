#!/usr/bin/env python3
"""The shift of Delta from holding the harmonics on the 1/n fit (chain_fit) instead of on T75's trend (chain), at
given D: each backbone, as export_delta_all.py traces it, is interpolated through the nine of its states nearest D
(Lagrange, mpmath), and the shift is chain_fit - chain (notes/QUAD_DOUBLE.md 13.29). The interpolation's spread is the
largest change from dropping the nearer or the farther end state of either nine, not a bound. --plant E adds E to
every chain_fit value (the control: every shift must move by E). Writes data/chain_fit_shift.txt.
    cd hp && python3 chain_fit_shift.py RAWDIR [--at=79.77,100.12,149.56,173.5,200.27] [--plant=0]
"""
import os
import sys

import mpmath as mp

import export_delta_all as E

HERE = os.path.dirname(os.path.abspath(__file__))


def lagrange(pts, x):
    """the polynomial through pts [(D, value)] at x"""
    out = mp.mpf(0)
    for i, (xi, yi) in enumerate(pts):
        w = mp.mpf(1)
        for j, (xj, _) in enumerate(pts):
            if j != i:
                w *= (x - xj) / (xi - xj)
        out += w * yi
    return out


def at(pts, x, n=9):
    """the interpolant through the n points nearest x, and its spread over dropping either end of them"""
    near = sorted(sorted(pts, key=lambda p: abs(p[0] - x))[:n])
    v = lagrange(near, x)
    return v, max(abs(lagrange(near[1:], x) - v), abs(lagrange(near[:-1], x) - v)), (near[0][0], near[-1][0])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if not args:
        sys.exit(__doc__)
    E.RAW = os.path.abspath(args[0])
    xs = [mp.mpf(v) for v in opts.get("at", "79.77,100.12,149.56,173.5,200.27").split(",")]
    plant = mp.mpf(opts.get("plant", "0"))
    states, by_dir = E.chain_states()
    held = {d: E.held_of(d) for d in by_dir}
    pts = {}
    for kind in ("chain_fit", "chain"):
        bb = E.backbone(states, by_dir, held, kind)
        pts[kind] = [(mp.mpf(states[r][0]["D"]), E.total(states[r][1]) + (plant if kind == "chain_fit" else 0))
                     for r in bb]
    lines = [f"chain_fit - chain, nine states of each backbone; plant {mp.nstr(plant, 2)}",
             "| D | chain_fit | chain | chain_fit - chain | spread | chain_fit states | chain states |",
             "|---|---|---|---|---|---|---|"]
    for x in xs:
        if not all(min(p[0] for p in pts[k]) <= x <= max(p[0] for p in pts[k]) for k in pts):
            lines.append(f"| {mp.nstr(x, 6)} | outside a backbone | | | | | |")
            continue
        (vf, sf, rf), (vc, sc, rc) = at(pts["chain_fit"], x), at(pts["chain"], x)
        lines.append(f"| {mp.nstr(x, 6)} | {mp.nstr(vf, 20)} | {mp.nstr(vc, 20)} | {mp.nstr(vf - vc, 5, min_fixed=1, max_fixed=0)}"
                     f" | {mp.nstr(max(sf, sc), 2, min_fixed=1, max_fixed=0)} | {float(rf[0]):.2f}-{float(rf[1]):.2f}"
                     f" | {float(rc[0]):.2f}-{float(rc[1]):.2f} |")
    print("\n".join(lines))
    if plant == 0:
        with open(os.path.join(HERE, "data", "chain_fit_shift.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
