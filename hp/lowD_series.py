#!/usr/bin/env python3
"""Convergence of the centre and horizon series of a double-double state (notes/LOW_D.md): the series coefficients
(Solver.series, computed to KC centre / KS horizon terms) and, at several evaluation points, the size of the last three
terms max |c_j| x^j (the truncation error of the series there; the solver uses x = first node for the centre series and
u = 1 - last node for the horizon series), and the radius of convergence by the root test over the last third of the
coefficients, per harmonic band of the fields.

    python3 lowD_series.py STATE.npz [--kc=200] [--ks=300] [--xc=0.005,0.01,0.02] [--uh=0.2,0.3,0.4] [--threads=2]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    kc = int(opts.get("kc", 200))
    ks = int(opts.get("ks", 300))
    xc = [float(v) for v in opts.get("xc", "0.005,0.01,0.02").split(",")]
    uh = [float(v) for v in opts.get("uh", "0.2,0.3,0.4").split(",")]
    threads = int(opts.get("threads", 2))
    for f in args:
        zh, zl, meta = load(f)
        D, M, nodes = meta["D"], meta["M"], list(meta["nodes"])
        s = make_solver(D, M, nodes=nodes, kt=meta.get("kt", 36), kc=kc, ks=ks, threads=threads)
        print(f"{os.path.basename(f)}: D {D} M {M} nodes {nodes[0]}..{nodes[-1]} (state built with centre/horizon terms "
              f"{meta.get('kc')}/{meta.get('ks')})", flush=True)
        for which, name, xs in ((0, "centre", xc), (1, "horizon", uh)):
            h, l = s.series(zh, zl, which)
            c = (np.asarray(h) + np.asarray(l)).reshape(-1, 4, M)
            J = c.shape[0]
            # harmonic content of each coefficient: fields are on the half grid; odd/even parity per field
            amp = np.abs(c).max(axis=(1, 2))
            jj = np.arange(J)
            j0 = (2 * J) // 3
            sl = np.polyfit(jj[j0:], np.log(np.maximum(amp[j0:], 1e-300)), 1)[0]
            R = float(np.exp(-sl))
            bands = []
            for lo, hi in ((0, M // 4), (M // 4, M // 2), (M // 2, M)):
                spec = []
                for j in range(J):
                    m = 0.0
                    for fi in range(4):
                        g = c[j, fi]
                        full = np.r_[g, -g] if fi in (0, 1) else np.r_[g, g]
                        sp = np.abs(np.fft.rfft(full))[lo:hi + 1]
                        m = max(m, sp.max() / M)
                    spec.append(m)
                spec = np.array(spec)
                s2 = np.polyfit(jj[j0:], np.log(np.maximum(spec[j0:], 1e-300)), 1)[0]
                bands.append(f"k {lo}-{hi}: R {np.exp(-s2):.4f}")
            print(f"  {name} series, {J} terms: radius (root test, last third) {R:.4f}; " + ", ".join(bands), flush=True)
            for x in xs:
                t = amp * x ** jj
                for n in sorted({60, 100, 150, J - 1} & set(range(J))):
                    pass
                cuts = [n for n in (40, 60, 100, 150, 200, 300) if n < J]
                print(f"    at {'x' if which == 0 else 'u'} = {x}: last-3-term size with N terms: "
                      + ", ".join(f"N {n}: {t[n - 3:n].max():.1e}" for n in cuts) + f", N {J}: {t[-3:].max():.1e}", flush=True)


if __name__ == "__main__":
    main()
