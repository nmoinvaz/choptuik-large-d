#!/usr/bin/env python3
"""The f64 horizon series at high kc (notes/LOW_D.md): for an f64 solution, the regular horizon series (DSS.ssh_series,
Python reference) evaluated at u0 = 1 - x_K with Ks terms: the largest term max_j |c_j| u0^j against the sum (the
cancellation factor, times 1e-16 the f64 rounding floor of the sum), the size of the last terms (truncation), and the
change of the sum from Ks to a larger Ks.

    python3 lowD_horizon.py STATE.npy [--Ks=60,90,120] [--D=...]

STATE: solver/data/neB_kc{kc}_..._D{D}.npy (x_K = 0.6 nodes). Per harmonic band the numbers are maxima over the four
fields and over the tau grid.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spec_long import make_long  # noqa: E402

NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    f = args[0]
    kc = int(re.search(r"kc(\d+)", os.path.basename(f)).group(1))
    D = float(opts.get("D", re.findall(r"D([0-9.]+)\.npy", f)[0]))
    Kss = [int(v) for v in opts.get("Ks", "60,90,120").split(",")]
    z = np.load(f)
    sums = {}
    for Ks in Kss:
        S = make_long(kc, D - 2, nodes=NL, Ks=Ks)
        parts, Dl = S.split(z[None, :])
        pars = S.parities()
        mS = S.phys(parts[-2], pars[-2])
        aS = S.phys(parts[-1], pars[-1])
        omega = np.array([2 * np.pi / z[-1]])
        P, Mh, A, C = S.ssh_series(mS, aS, omega)
        u = S.u0
        terms = np.array([[np.abs(Q[j]).max() * u ** j for j in range(len(Q))] for Q in (P, Mh, A, C)])
        tot = np.stack([S.eval_series(Q, u) for Q in (P, Mh, A, C)])
        big = terms.max(axis=1)
        last = terms[:, -3:].max(axis=1)
        mag = np.abs(tot).reshape(4, -1).max(axis=1)
        sums[Ks] = tot
        print(f"D {D} kc {kc} Ks {Ks} u0 {u:.2f}: largest term / |sum| per field (p, m, a, c): "
              + " ".join(f"{b:.1e}/{s:.1e}" for b, s in zip(big, mag))
              + f"   last 3 terms {' '.join(f'{x:.1e}' for x in last)}   j of largest term "
              + " ".join(str(int(np.argmax(terms[i]))) for i in range(4)), flush=True)
        if Ks == max(Kss):
            # radius of convergence in u per harmonic band: root test on the coefficients over the last third
            bands = [(0, 15), (17, 31), (33, 47), (49, 63)]
            out = []
            for lo, hi in bands:
                if lo > kc:
                    continue
                cj = []
                for j in range(len(P)):
                    m = 0.0
                    for Q in (P, Mh, A, C):
                        spec = np.abs(np.fft.rfft(np.atleast_2d(Q[j])[0])) / S.N
                        m = max(m, spec[lo:hi + 1].max())
                    cj.append(m)
                cj = np.array(cj)
                j0, j1 = len(cj) * 2 // 3, len(cj) - 1
                ok = (cj[j0:j1 + 1] > 0)
                js = np.arange(j0, j1 + 1)[ok]
                slope = np.polyfit(js, np.log(cj[j0:j1 + 1][ok]), 1)[0] if ok.sum() > 3 else np.nan
                out.append(f"k {lo}-{hi}: R {np.exp(-slope):.3f}")
            print("   radius of convergence in u (root test, coefficients j >= 2Ks/3): " + ", ".join(out), flush=True)
    ks = sorted(sums)
    for a, b in zip(ks, ks[1:]):
        print(f"   sum(Ks {b}) - sum(Ks {a}): max {np.abs(sums[b] - sums[a]).max():.1e}")


if __name__ == "__main__":
    main()
