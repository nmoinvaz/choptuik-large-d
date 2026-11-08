#!/usr/bin/env python3
"""The second-zero test of the harmonic-1 Stokes multiplier S1 (notes/HARMONY.md item 1; comms 2026-11-07, T153 and
T155). The amplitude coefficients sigma_0..sigma_16 at seven frequency nodes omega/omega_inf = 1.06-1.14 (K 33, J 1000,
960 bits; raw T155-win2/ and T153-macm3/) are summed by Borel-Pade [m/(16-m)], m = 5..9 (zerolib). In each sum the
zero x1 of S1 near x = A^2 ~ 13 is tracked across the nodes.

Criterion, set in comms before the runs: Im x1 changes sign between two nodes in every ring-free sum. A sum is
ring-free when every Borel-Pade pole has |t| < 2 at every node, the frozen type rule of notes/PLANTED_SELECTION.md 6.
Controls, set at the same time, through the same code: a planted family whose x1 crosses the real axis at 1.095 must
be found crossing within 0.005 of it, and one whose crossing is moved to 1.15 must show no sign change in 1.06-1.14.
The plants are plant_x1.py's exponential-sum model with x1(wf) = x1c + v (wf - wfc), v the velocity measured on the
data, coefficients to k = 16 with 1e-12 relative noise.

    cd solver/harmony/stokes && python3 second_zero_nodes.py RAWDIR [--h=0.3]
    cd solver/harmony/stokes && python3 second_zero_nodes.py RAWDIR --k45 [--h=0.3]

--k45 (#82): sigma_0..sigma_22 at the three nodes 1.09, 1.10, 1.11 (K 45, J 1500, 1280 bits; raw I82-win2/), summed by
[m/(22-m)], m = 8..14, with the same criterion, and the same two controls as #82 set them before the runs: a planted
crossing at 1.095 found within 0.005 in every ring-free sum, and one at 1.15 showing no sign change in 1.09-1.11.

Prints the crossing per sum and exits 1 when the criterion or either control fails.
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

import zerolib as Z

mp.mp.dps = 50
NODES = [("1.06", "T155-win2"), ("1.08", "T153-macm3"), ("1.09", "T155-win2"), ("1.10", "T153-macm3"),
         ("1.11", "T155-win2"), ("1.12", "T153-macm3"), ("1.14", "T155-win2")]
SUMS = [(m, 16 - m) for m in range(5, 10)]
NCOEF = 17
FILE = "sigF_w{wf}_K33_J1000.json"
NODES45 = [("1.09", "I82-win2"), ("1.10", "I82-win2"), ("1.11", "I82-win2")]
SUMS45 = [(m, 22 - m) for m in range(8, 15)]
X1_GUESS = mp.mpc(13.3, 0)


def load(path):
    d = json.load(open(path))
    return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["sigma_str"]][:NCOEF]


def x1_of(sig, m, n, prev, h):
    """(x1, max |t|): the zero of the [m/n] sum nearest prev among those with 10 < Re x < 16"""
    ts, rs, c0 = Z.borel_pade(sig, m, n)
    zs, _ = Z.zeros_in_disc(Z.bp_numpy(ts, rs, c0), Z.bp_func(ts, rs, c0), R=17.0, h=h)
    cand = [z for z in zs if 10 < float(z.real) < 16 and abs(float(z.imag)) < 3]
    tmax = float(max(abs(t) for t in ts))
    if not cand:
        return None, tmax
    return min(cand, key=lambda z: abs(z - prev)), tmax


def track(sigs, h):
    """per sum: the x1 at each node, the type rule, and the crossings of Im x1 = 0 (linear between nodes)"""
    out = {}
    for m, n in SUMS:
        prev, xs, ok = X1_GUESS, [], True
        for (wf, _), sig in zip(NODES, sigs):
            x, tmax = x1_of(sig, m, n, prev, h)
            ok &= tmax < 2.0
            xs.append(x)
            if x is not None:
                prev = x
        cross = []
        for i in range(len(xs) - 1):
            a, b = xs[i], xs[i + 1]
            if a is None or b is None:
                continue
            ia, ib = float(a.imag), float(b.imag)
            if ia == 0 or ia * ib < 0:
                w0, w1 = float(NODES[i][0]), float(NODES[i + 1][0])
                cross.append(w0 + (w1 - w0) * ia / (ia - ib))
        out[(m, n)] = dict(xs=xs, ring_free=ok, cross=cross)
    return out


def plant(wfc, v, x1c, rng):
    """plant_x1.py's model at each node, with x1 moving linearly through the real axis at wfc"""
    x0 = mp.mpc("7.39408", "0")
    beta, pa = mp.mpc("4.16", "-1.60"), mp.mpc("-0.5", "0")
    P = [mp.mpc("0.705", "0.055"), mp.mpc("0.935", "0.463"), mp.mpc("1.279", "1.081")]
    g = lambda s: P[0] * (1 - s) * (1 - 2 * s) + 4 * P[1] * s * (1 - s) + P[2] * s * (2 * s - 1)  # noqa: E731
    dg = lambda s: P[0] * (4 * s - 3) + 4 * P[1] * (1 - 2 * s) + P[2] * (4 * s - 1)  # noqa: E731
    u, w = np.polynomial.legendre.leggauss(64)
    S = [(1 + mp.mpf(a)) / 2 for a in u]
    W = [mp.mpf(b) / 2 for b in w]
    T = [g(s) for s in S]
    R = [wt * (1 - s) ** (beta - 2) * s ** pa * dg(s) for s, wt in zip(S, W)]
    L0 = mp.fsum(R)
    lc = [mp.fsum(r * t ** k for t, r in zip(T, R)) / mp.factorial(k) / L0 for k in range(NCOEF)]
    sigs = []
    for wf, _ in NODES:
        x1 = x1c + v * (mp.mpf(wf) - wfc)
        c = lc
        for z in (x0, x1):
            c = [c[0]] + [c[k] - c[k - 1] / z for k in range(1, len(c))]
        sigs.append([ck * (1 + 1e-12 * complex(*rng.normal(size=2))) for ck in c])
    return sigs


def report(label, res):
    print(label)
    for (m, n), r in res.items():
        xs = "  ".join("—" if x is None else Z.cnum(x, 3) for x in r["xs"])
        cr = ", ".join(f"{c:.4f}" for c in r["cross"]) or "none"
        print(f"  [{m}/{n}] ring-free {r['ring_free']}: crossing at {cr}; x1: {xs}")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in argv if a.startswith("--"))
    if len(args) != 1:
        sys.exit(__doc__)
    raw, h = args[0], float(opts.get("h", 0.3))
    global NODES, SUMS, NCOEF, FILE
    if "k45" in opts:
        NODES, SUMS, NCOEF, FILE = NODES45, SUMS45, 23, "sigF_w{wf}_K45_J1500_p1280.json"
    lo, hi = NODES[0][0], NODES[-1][0]
    sigs = [load(os.path.join(raw, d, FILE.format(wf=wf))) for wf, d in NODES]
    real = track(sigs, h)
    report("data", real)
    free = [r for r in real.values() if r["ring_free"]]
    ok_data = bool(free) and all(len(r["cross"]) >= 1 for r in free)
    # the velocity of x1 on the data, from the ring-free sums' end nodes
    vs = [(r["xs"][-1] - r["xs"][0]) / (float(NODES[-1][0]) - float(NODES[0][0])) for r in free
          if r["xs"][0] is not None and r["xs"][-1] is not None]
    v = sum(vs) / len(vs) if vs else mp.mpc(8, -11)
    x1c = mp.mpc(13.45, 0)
    rng = np.random.default_rng(11)
    p1 = track(plant(mp.mpf("1.095"), v, x1c, rng), h)
    report(f"planted crossing at 1.095 (v = {Z.cnum(v, 3)}), must be found within 0.005", p1)
    free1 = [r for r in p1.values() if r["ring_free"]]
    ok_p1 = bool(free1) and all(len(r["cross"]) == 1 and abs(r["cross"][0] - 1.095) < 0.005 for r in free1)
    p2 = track(plant(mp.mpf("1.15"), v, x1c, rng), h)
    report(f"planted crossing at 1.15, must show no sign change in {lo}-{hi}", p2)
    free2 = [r for r in p2.values() if r["ring_free"]]
    ok_p2 = bool(free2) and all(len(r["cross"]) == 0 for r in free2)
    print(f"criterion on the data: {'pass' if ok_data else 'FAIL'} ({len(free)} ring-free sums); "
          f"control 1.095: {'pass' if ok_p1 else 'FAIL'}; control 1.15: {'pass' if ok_p2 else 'FAIL'}")
    return 0 if (ok_data and ok_p1 and ok_p2) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
