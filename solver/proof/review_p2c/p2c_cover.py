#!/usr/bin/env python3
"""P2c review (#991): lemma 6b1's M = 1 covering, its geometry checked independently of lemma6b1_cover.py (not opened
before the numbers were committed). The definitions are A_certify.md 12.3's statements; the data are the stored runs
lemma6b1_runs/row_00.json .. row_11.json (rho_j, r_j, and per disc t_jk and its certificate flag).

Tiles: T_jk = (a_j, b_j) x (phi_jk - alpha_j, phi_jk + alpha_j) in polar coordinates (|u|, arg u), with
a_j = rho_j - (3/5) r_j, b_j = rho_j + (3/5) r_j, alpha_j = (3/5)(3 pi/4)/K_j, phi_jk = -(3 pi/4)(1 - t_jk), taken as
exact rationals and pi-multiples. G0 = {|u| > 257, -3 pi/4 - 0.4 < arg u < 0}. A1 = [25, 260] x [-3 pi/4, 0].
Every angle lies in (-pi, pi), so polar rectangles are rectangles in (|u|, arg u) and connectedness may be read there.

Checks (P2_remaining.md 3.1):
  G1  data: 12 rows, 178 discs, all flagged certified, t_j0 = 0, t_jK = 1, r_j < rho_j;
  G2  each closed tile lies in its open disc (the distance from the centre is largest at a corner);
  G3  the open tiles cover A1 (bands cover [25, 260], each row's angles cover [-3 pi/4, 0], strict overlaps);
  G4  tile-free: A1 cut into closed polar cells, each proven inside one open disc (bisection, cap 1e5 cells);
  G5  the gluing in A_certify.md 12.3's order: each meeting set M_m (T_m against G0 and every earlier tile, pieces kept
      unless proven empty) is connected through proven-nonempty overlaps, the agreement point q_m lies in T_m and in
      the partner piece, and the curve of agreement lies in both discs (half-arcs 2 rho sin(dphi/4) < r; ray segments
      |rho - |q|| < r);
  G6  every tile's partner comes earlier (so all are reached from (0, 0) on the start ray), and s0 is in G0.

    python3 p2c_cover.py [--runs=DIR] [--plant=drop|gap] [--log=FILE]

--plant=drop removes disc (5, 6); --plant=gap sets row 3's alpha to 0.49 dphi, so consecutive tiles there no longer
overlap. Each must fail (exit 1). Arithmetic: mpmath intervals at 200 bits for pi, sin and cos; exact rationals
otherwise. A comparison counts only when it is proven.
"""
import json
import os
import sys
import time
from fractions import Fraction as Fr

from mpmath import iv, mp

iv.prec = 200
mp.prec = 200
HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
RUNS = OPTS.get("runs", os.path.join(os.path.dirname(HERE), "lemma6b1_runs"))
PLANT = OPTS.get("plant")
PI = iv.pi
Q34 = 3 * PI / 4


def I(x):
    """an exact rational (or int) as an interval"""
    x = Fr(x)
    return iv.mpf(x.numerator) / iv.mpf(x.denominator)


def lo(x):
    return mp.mpf(iv.mpf(x)._mpi_[0])


def hi(x):
    return mp.mpf(iv.mpf(x)._mpi_[1])


def lt(x, y):
    """proven x < y"""
    return hi(x) < lo(y)


def load_rows():
    rows = []
    for j in range(12):
        with open(os.path.join(RUNS, f"row_{j:02d}.json")) as fh:
            d = json.load(fh)
        rho = Fr(repr(d["rho"]))                     # the decimal the run used
        discs = [(dd["k"], Fr(dd["t"]), bool(dd["ok"])) for dd in d["discs"]]
        rows.append(dict(j=d["j"], rho=rho, r=Fr(d["r"]), n=d["n"], discs=discs))
    return rows


def build(rows):
    """rows -> tiles in the gluing order, each with polar intervals and its disc"""
    tiles = []
    for R in rows:
        K = len(R["discs"]) - 1
        a, b = R["rho"] - Fr(3, 5) * R["r"], R["rho"] + Fr(3, 5) * R["r"]
        alpha = I(Fr(3, 5)) * Q34 / K
        R.update(K=K, a=a, b=b, alpha=alpha)
        if PLANT == "gap" and R["j"] == 3:
            dphi = Q34 * I(R["discs"][1][1] - R["discs"][0][1])
            R["alpha"] = alpha = I(Fr(49, 100)) * dphi
        for k, t, ok in R["discs"]:
            if PLANT == "drop" and (R["j"], k) == (5, 6):
                continue
            phi = -Q34 * I(1 - t)
            tiles.append(dict(j=R["j"], k=k, t=t, ok=ok, rho=R["rho"], r=R["r"], a=a, b=b, phi=phi, alpha=alpha,
                              rl=(I(a), I(b)), tl=(phi - alpha, phi + alpha)))
    return tiles


def g2_in_disc(T):
    """closed tile inside the open disc: worst corner distance^2 < r^2"""
    rho, r, ca = I(T["rho"]), I(T["r"]), iv.cos(T["alpha"])
    worst = max((hi(x * x + rho * rho - 2 * x * rho * ca) for x in (I(T["a"]), I(T["b"]))))
    return worst < lo(r * r), float(lo(r * r) - worst)


def g3_cover(rows, tiles):
    fails = []
    if not lt(I(260), I(rows[0]["b"])):
        fails.append("b_0 <= 260")
    if not lt(I(rows[-1]["a"]), I(25)):
        fails.append("a_11 >= 25")
    for j in range(1, len(rows)):
        if not lt(I(rows[j - 1]["a"]), I(rows[j]["b"])):
            fails.append(f"bands {j - 1}, {j} do not overlap")
    for R in rows:
        row = [T for T in tiles if T["j"] == R["j"]]
        if not lt(row[0]["tl"][0], -Q34):
            fails.append(f"row {R['j']}: the first tile does not reach below -3pi/4")
        if not lt(I(0), row[-1]["tl"][1]):
            fails.append(f"row {R['j']}: the last tile does not reach above 0")
        for A, B in zip(row[:-1], row[1:]):
            if not lt(B["tl"][0], A["tl"][1]):
                fails.append(f"row {R['j']}: tiles {A['k']}, {B['k']} do not overlap")
    return fails


def cell_in_disc(r1, r2, t1, t2, T):
    """closed polar cell [r1, r2] x [t1, t2] inside the open disc of T (interval proof)"""
    rho, r = I(T["rho"]), I(T["r"])
    phi = -Q34 * I(1 - T["t"])
    m = max(hi(abs(t1 - phi)), hi(abs(t2 - phi)))
    if m >= mp.pi:
        return False
    cm = iv.cos(iv.mpf(m))
    worst = max(hi(x * x + rho * rho - 2 * x * rho * cm) for x in (r1, r2))
    return worst < lo(r * r)


def g4_cells(tiles, cap=100000):
    """A1 cut into closed polar cells, each proven inside one open disc; cells bisected when none holds"""
    import math
    cen = [(float(T["rho"]) * math.cos(-2.356194490192345 * float(1 - T["t"])),
            float(T["rho"]) * math.sin(-2.356194490192345 * float(1 - T["t"])), T) for T in tiles]
    stack = []
    nr, nt = 94, 95
    for a in range(nr):
        for b in range(nt):
            stack.append((I(25) + I(Fr(235, nr)) * a, I(25) + I(Fr(235, nr)) * (a + 1),
                          -Q34 + Q34 * I(Fr(b, nt)), -Q34 + Q34 * I(Fr(b + 1, nt)), 0))
    done, failed, ncells = 0, [], 0
    while stack:
        r1, r2, t1, t2, depth = stack.pop()
        ncells += 1
        if ncells > cap:
            return done, ncells, ["cap reached"]
        rm, tm = float(mp.mpf(lo(r1) + hi(r2)) / 2), float(mp.mpf(lo(t1) + hi(t2)) / 2)
        x, y = rm * math.cos(tm), rm * math.sin(tm)
        order = sorted(cen, key=lambda c: ((c[0] - x) ** 2 + (c[1] - y) ** 2) / float(c[2]["r"]) ** 2)[:6]
        if any(cell_in_disc(r1, r2, t1, t2, c[2]) for c in order):
            done += 1
            continue
        if depth >= 12:
            failed.append((float(lo(r1)), float(hi(r2)), float(lo(t1)), float(hi(t2))))
            continue
        if hi(r2 - r1) / 260 > hi(t2 - t1):
            mid = (r1 + r2) / 2
            stack += [(r1, mid, t1, t2, depth + 1), (mid, r2, t1, t2, depth + 1)]
        else:
            mid = (t1 + t2) / 2
            stack += [(r1, r2, t1, mid, depth + 1), (r1, r2, mid, t2, depth + 1)]
    return done, ncells, [f"cell {f} in no disc" for f in failed[:5]] + (
        [f"... {len(failed)} cells in no disc"] if len(failed) > 5 else [])


def inter(R1, R2):
    """intersection of two open rectangles ((rl), (tl)): ('empty'|'nonempty'|'undecided', rect)"""
    rl = (iv.mpf([max(lo(R1[0][0]), lo(R2[0][0])), max(hi(R1[0][0]), hi(R2[0][0]))]),
          iv.mpf([min(lo(R1[0][1]), lo(R2[0][1])), min(hi(R1[0][1]), hi(R2[0][1]))]))
    tl = (iv.mpf([max(lo(R1[1][0]), lo(R2[1][0])), max(hi(R1[1][0]), hi(R2[1][0]))]),
          iv.mpf([min(lo(R1[1][1]), lo(R2[1][1])), min(hi(R1[1][1]), hi(R2[1][1]))]))
    if lo(rl[0]) >= hi(rl[1]) or lo(tl[0]) >= hi(tl[1]):
        return "empty", None
    if lt(rl[0], rl[1]) and lt(tl[0], tl[1]):
        return "nonempty", (rl, tl)
    return "undecided", (rl, tl)


def inside(q, R):
    """the point q = (|q|, arg q) (intervals) strictly inside the open rectangle R"""
    return lt(R[0][0], q[0]) and lt(q[0], R[0][1]) and lt(R[1][0], q[1]) and lt(q[1], R[1][1])


def g5_glue(rows, tiles):
    BIG = iv.mpf(10) ** 6
    G0 = ((I(257), BIG), (-Q34 - I(Fr(2, 5)), I(0)))
    idx = {(T["j"], T["k"]): i for i, T in enumerate(tiles)}
    fails, npieces, nlinks, nspan = [], 0, 0, 0
    for m, T in enumerate(tiles):
        Tm = (T["rl"], T["tl"])
        pieces = []
        st, rect = inter(Tm, G0)
        if st != "empty":
            pieces.append(("G0", st, rect))
        for l in range(m):
            S = tiles[l]
            st, rect = inter(Tm, (S["rl"], S["tl"]))
            if st != "empty":
                pieces.append(((S["j"], S["k"]), st, rect))
        npieces += len(pieces)
        name = (T["j"], T["k"])
        if not pieces:
            fails.append(f"tile {name}: empty meeting set")
            continue
        # connectivity through proven-nonempty overlaps
        n = len(pieces)
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for k2 in range(i + 1, n):
                if pieces[i][2] is None or pieces[k2][2] is None:
                    continue
                st, _ = inter(pieces[i][2], pieces[k2][2])
                if st == "nonempty":
                    adj[i].add(k2)
                    adj[k2].add(i)
                    nlinks += 1
        seen, todo = {0}, [0]
        while todo:
            for nb in adj[todo.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
        nspan += len(seen) - 1
        if len(seen) != n:
            fails.append(f"tile {name}: meeting set not shown connected ({len(seen)} of {n} pieces linked)")
        # the partner and the agreement point
        j, k = name
        R = rows[j]
        if (j, k) == (0, 0):
            partner, q = "G0", (I(T["rho"]), -Q34)
            curve = True
        elif k >= 1:
            prev = [S for S in tiles[:m] if S["j"] == j and S["k"] < k]
            if not prev or prev[-1]["k"] != k - 1:
                fails.append(f"tile {name}: its partner ({j}, {k - 1}) is missing")
                continue
            S = prev[-1]
            partner = (j, k - 1)
            q = (I(R["rho"]), (S["phi"] + T["phi"]) / 2)
            dphi = T["phi"] - S["phi"]
            curve = lt(2 * I(R["rho"]) * iv.sin(dphi / 4), I(R["r"]))
        else:
            if (j - 1, 0) not in idx:
                fails.append(f"tile {name}: its partner ({j - 1}, 0) is missing")
                continue
            partner = (j - 1, 0)
            Rp = rows[j - 1]
            rq = (I(Rp["a"]) + I(R["b"])) / 2
            q = (rq, -Q34)
            curve = lt(abs(I(Rp["rho"]) - rq), I(Rp["r"])) and lt(abs(rq - I(R["rho"])), I(R["r"]))
        pp = [p for p in pieces if p[0] == partner]
        if not pp:
            fails.append(f"tile {name}: no piece with its partner {partner}")
            continue
        if pp[0][1] != "nonempty" or not inside(q, pp[0][2]):
            fails.append(f"tile {name}: the agreement point is not proven inside T_m and its partner {partner}")
        if not curve:
            fails.append(f"tile {name}: the curve of agreement with {partner} is not inside both discs")
    return fails, npieces, nlinks, nspan


def main():
    t0 = time.time()
    rows = load_rows()
    tiles = build(rows)
    out, lines = {}, []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"P2c covering check (lemma 6b1, M = 1): runs {RUNS}, plant = {PLANT}")
    nd = sum(len(R["discs"]) for R in rows)
    g1 = [] if (len(rows) == 12 and nd == 178) else [f"{len(rows)} rows, {nd} discs"]
    for R in rows:
        ts = [t for _, t, _ in R["discs"]]
        if ts[0] != 0 or ts[-1] != 1 or any(b <= a for a, b in zip(ts[:-1], ts[1:])):
            g1.append(f"row {R['j']}: t_k not 0 .. 1 increasing")
        if not all(ok for _, _, ok in R["discs"]):
            g1.append(f"row {R['j']}: a disc not certified")
        if not R["r"] < R["rho"]:
            g1.append(f"row {R['j']}: r >= rho")
    log(f"G1 data: {len(rows)} rows, {nd} discs, {len(tiles)} tiles in the gluing: {'ok' if not g1 else g1}")
    g2 = []
    margin = None
    for T in tiles:
        ok, mg = g2_in_disc(T)
        if not ok:
            g2.append((T["j"], T["k"]))
        margin = mg if margin is None else min(margin, mg)
    log(f"G2 closed tiles in their open discs: {'ok' if not g2 else f'FAIL {g2[:5]}'} (smallest margin in r^2 - d^2: "
        f"{margin:.4g})")
    g3 = g3_cover(rows, tiles)
    log(f"G3 open tiles cover A1: {'ok' if not g3 else 'FAIL ' + '; '.join(g3[:6])}")
    done, ncells, g4 = g4_cells(tiles)
    log(f"G4 A1 in polar cells, each in one open disc: {'ok' if not g4 else 'FAIL ' + '; '.join(g4)} "
        f"({done} cells proven, {ncells} examined)")
    g5, npieces, nlinks, nspan = g5_glue(rows, tiles)
    log(f"G5 gluing: {'ok' if not g5 else 'FAIL ' + '; '.join(g5[:6]) + (f' (+{len(g5) - 6} more)' if len(g5) > 6 else '')}"
        f" ({npieces} pieces, {nlinks} proven links among pieces, {nspan} of them in spanning trees)")
    s0_in = lt(I(257), I(rows[0]["rho"]))
    g6 = [] if s0_in else ["s0 not in G0"]
    log(f"G6 every partner earlier, all reached from (0, 0), s0 in G0: {'ok' if not g6 else 'FAIL ' + '; '.join(g6)}")
    ok = not (g1 or g2 or g3 or g4 or g5 or g6)
    log(f"=> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
    out = dict(plant=PLANT, ok=ok, tiles=len(tiles), pieces=npieces, links=nlinks, span=nspan, cells=done,
               examined=ncells, g2_margin=margin, fails=dict(g1=g1, g2=[list(x) for x in g2], g3=g3, g4=g4, g5=g5,
                                                            g6=g6))
    tag = PLANT or "real"
    with open(os.path.join(HERE, f"p2c_cover_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    if OPTS.get("log"):
        with open(os.path.join(HERE, OPTS["log"]), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
