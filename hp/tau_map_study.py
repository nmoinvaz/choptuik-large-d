#!/usr/bin/env python3
"""How many tau modes a mapped tau grid would save at low D (notes/LOW_D_SPEEDUP.md, step 5): on a stored converged
state, the Fourier decay rate per harmonic of every field block (the centre data, the four fields at each node, the
horizon data), on the uniform grid and after one shared periodic map tau = g(s) that concentrates points on the pulse.

The grid is shared by every field at every node (the residual steps in x at fixed tau points), so one map serves all
blocks, and the block with the slowest decay sets the modes needed: the gain is that block's rate after the map over
its rate before. Two map families, both of period pi (the half-period antisymmetry, g(s + pi) = g(s) + pi, and two
pulses per period): sine, g(s) = s - alpha sin(2 (s - s0)) / 2, with g' = 1 - alpha cos(2 (s - s0)); and arctan
(Boyd, J. Comput. Phys. 98 (1992), the periodic tan map), g(s) = s0 + arctan(L tan(s - s0)) with g' at the pulse L.
The map's centre s0 is scanned over [0, pi) for each map, since the pulse sits at different tau in different fields
and nodes (printed for the p fields). The measure is the modes a block needs: the highest harmonic of its parity whose
coefficient exceeds eps times the block's largest, at levels eps well above the stored values' double-precision floor;
the mapped fields are the state's trigonometric interpolants evaluated on a grid --fine times finer (so the map's
compression does not alias) and transformed there.

    python3 tau_map_study.py RAWDIR [--state T100-win2/states/lowDhm_x0.000157-0.975_D3.0444_M384.npz] [--eps 1e-10,1e-12] [--out data/tau_map_study.json]
        [--ns0 24] [--Ls 0.5,0.3,0.2,0.1]

Minutes, one thread. Prints the table and writes the JSON.
"""
import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402


def blocks(zh, zl, M, K):
    """[(name, parity, values on the half period)] in the state's layout (driver.resample_z)"""
    names = ["centre p", "centre a"] + [f"node {i} {f}" for i in range(1, K + 1) for f in ("p", "m", "a", "c")] + ["horizon p", "horizon a"]
    parity = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    z = np.asarray(zh, float) + np.asarray(zl, float)
    return [(names[j], parity[j], z[j * M:(j + 1) * M]) for j in range(4 * (K + 1))]


def coeffs(par, f):
    full = np.r_[f, -f] if par == "odd" else np.r_[f, f]
    return np.fft.rfft(full) / len(full)


def k_needed(c, par, eps):
    """the highest harmonic of the block's parity with |c_k| above eps times the block's largest: the modes the block
    needs at the relative level eps (kept well above the double-precision floor of the stored values, about 1e-16)"""
    k = np.arange(len(c))
    sel = k % 2 == (1 if par == "odd" else 0)
    a = np.abs(c) * sel
    above = np.nonzero(a > eps * a.max())[0]
    return int(above.max()) if len(above) else 0


def interpolant(C, g):
    """the blocks' trigonometric interpolants (coefficients C, rows; one period) at the points g, all blocks at once"""
    k = np.arange(C.shape[1])
    w = np.where((k == 0) | (k == C.shape[1] - 1), 1.0, 2.0)
    E = np.exp(1j * np.outer(g, k))
    return (E @ (w[None, :] * C).T).real                      # (len(g), blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--state", default="T100-win2/states/lowDhm_x0.000157-0.975_D3.0444_M384.npz")
    ap.add_argument("--eps", default="1e-10,1e-12")
    ap.add_argument("--fine", type=int, default=8)
    ap.add_argument("--ns0", type=int, default=24, help="centres s0 scanned over [0, pi)")
    ap.add_argument("--Ls", default="0.5,0.3,0.2,0.1", help="arctan map parameters")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tau_map_study.json"))
    a = ap.parse_args()
    d = np.load(os.path.join(a.raw, a.state))
    meta = json.loads(str(d["meta"]))
    M, K = int(meta["M"]), len(meta["nodes"])             # K nodes, each with p, m, a, c
    zh, zl = d["zh"], d["zl"]
    assert len(zh) == 4 * M * (K + 1) + 1, (len(zh), M, K)
    B = blocks(zh, zl, M, K)
    names, pars = [b[0] for b in B], [b[1] for b in B]
    C = np.array([coeffs(par, f) for _, par, f in B])        # (blocks, M + 1)
    epss = [float(e) for e in a.eps.split(",")]
    N = 2 * M
    Nf = a.fine * N                                          # the mapped fields on a finer grid: no aliasing
    sf = 2 * np.pi * np.arange(Nf) / Nf
    def need(Cm, eps):
        return {n: k_needed(Cm[i], pars[i], eps) for i, n in enumerate(names)}
    base = {e: need(C, e) for e in epss}
    print(f"{a.state}: D {meta['D']}, M {M} (harmonics to {M}), {K} nodes, {len(B)} blocks")
    for e in epss:
        w = max(base[e], key=base[e].get)
        print(f"uniform grid, eps {e:.0e}: modes needed {min(base[e].values())}..{base[e][w]} (slowest: {w})")
    # pulse positions of the p fields mod pi (circular), across the nodes
    vals = interpolant(C, sf)
    pp = [(n, float(sf[np.argmax(np.abs(vals[:, i]))] % np.pi)) for i, n in enumerate(names) if n.endswith(" p")]
    print("pulse of p mod pi: " + ", ".join(f"{n.replace('node ', 'n').replace(' p', '')} {v:.2f}" for n, v in pp))
    rows = []
    for fam, params in (("sine", (0.5, 0.7, 0.9)), ("arctan", tuple(float(v) for v in a.Ls.split(",")))):
        for par_ in params:
            best = None
            for s0 in np.linspace(0, np.pi, a.ns0, endpoint=False):
                u = sf - s0
                g = sf - par_ * np.sin(2 * u) / 2 if fam == "sine" else s0 + np.arctan(par_ * np.tan(u)) + np.pi * np.round(u / np.pi)
                Q = interpolant(C, g)                        # (Nf, blocks)
                Cm = np.array([np.fft.rfft(Q[:, i]) / Nf for i in range(len(names))])
                worst = {e: max(need(Cm, e).values()) for e in epss}
                score = worst[epss[0]]
                if best is None or score < best[0]:
                    best = (score, s0, worst, need(Cm, epss[0]))
            score, s0, worst, per = best
            slowest = max(per, key=per.get)
            gains = {e: max(base[e].values()) / worst[e] for e in epss}
            rows.append(dict(family=fam, parameter=par_, s0=float(s0), worst=worst, slowest_block=slowest, gains=gains))
            print(f"  {fam:6s} {par_:4.2f} at s0 {s0:.2f}: modes needed " + ", ".join(f"{worst[e]} at {e:.0e} ({gains[e]:.2f}x)" for e in epss)
                  + f"; slowest {slowest}")
    bestrow = max(rows, key=lambda r: r["gains"][epss[0]])
    gm = bestrow["gains"][epss[0]]
    print(f"best: {bestrow['family']} {bestrow['parameter']} at s0 {bestrow['s0']:.2f}: {gm:.2f}x fewer modes for the slowest "
          f"block, about {gm ** 3:.1f}x per Newton call at the same accuracy (a call grows like M^3)")
    json.dump(dict(state=a.state, D=meta["D"], M=M, nodes=meta["nodes"], eps=epss, fine=a.fine,
                   uniform={str(e): base[e] for e in epss}, pulse_p=pp, maps=rows, best=bestrow),
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
