#!/usr/bin/env python3
"""Where across omega the zero of S_1 becomes real, order by order, from an omega scan (scan_omega.py output).

ln S_1 = sum_k tau_k x^k (x = eps^2, as in zero_selection.py). A simple zero of S_1 at real positive x0, nearer than any
other singularity, gives tau_k -> -1/(k x0^k): arg(tau_k) -> pi. For a complex x0, arg(tau_k) ~ pi - k arg(x0) turns with k.
So at each k the omega where arg(tau_k) = pi estimates the omega where the zero is real, and the sequence over k
converges to the selected omega. Printed: arg(tau_k)/pi per omega, the crossing omega for each k (linear
interpolation), the Aitken limit of the last three crossings, and the corrected ratio estimates
eps0 = sqrt(tau_k/tau_{k+1} * k/(k+1)) where they become real.

    python3 arg_crossing.py scan13b [KMAX]        KMAX: highest k used (default: all sigma terms present)
"""
import glob
import json
import os
import sys

import mpmath as mp

mp.mp.dps = 40


def taus(sig):
    """coefficients of ln(sum sigma_k x^k), sigma_0 = 1"""
    n = len(sig)
    L = [mp.mpc(0)] * n
    for k in range(1, n):              # k L_k = k s_k - sum_{m=1}^{k-1} m L_m s_{k-m}
        L[k] = sig[k] - sum(m * L[m] * sig[k - m] for m in range(1, k)) / k
    return L


def main():
    d = sys.argv[1]
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(fn))
        key = "sigma_str" if "sigma_str" in r else "sigma"
        sig = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r[key]]
        rows.append((float(r["wf"]), taus(sig)))
    rows.sort()
    n = min(len(t) for _, t in rows)
    if kmax:
        n = min(n, kmax + 1)
    print("wf      " + " ".join(f"arg(tau_{k})/pi" for k in range(2, n)))
    for wf, t in rows:
        print(f"{wf:.4f}  " + " ".join(f"{float(mp.arg(t[k]) / mp.pi) % 2:14.4f}" for k in range(2, n)))
    cross = []
    for k in range(2, n):
        prev = None
        for wf, t in rows:
            a = float(mp.arg(t[k]) / mp.pi) % 2
            if prev is not None and (prev[1] - 1) * (a - 1) < 0 and abs(a - prev[1]) < 0.5:
                w0 = prev[0] + (1 - prev[1]) * (wf - prev[0]) / (a - prev[1])
                cross.append((k, w0))
                print(f"arg(tau_{k}) = pi at wf = {w0:.4f}")
                break
            prev = (wf, a)
    if len(cross) >= 3:
        x0, x1, x2 = (c[1] for c in cross[-3:])
        den = (x2 - x1) - (x1 - x0)
        if den:
            print(f"Aitken limit of the last three crossings: wf = {x2 - (x2 - x1) ** 2 / den:.4f}")
    for k in range(2, n - 1):
        prev = None
        for wf, t in rows:
            e = complex(mp.sqrt(t[k] / t[k + 1] * k / (k + 1)))
            if prev is not None and prev[1].imag * e.imag < 0:
                f = prev[1].imag / (prev[1].imag - e.imag)
                print(f"ratio tau_{k}/tau_{k + 1}: eps0 real at wf = {prev[0] + f * (wf - prev[0]):.4f}, "
                      f"A0 = {prev[1].real + f * (e.real - prev[1].real):.3f}")
            prev = (wf, e)


if __name__ == "__main__":
    main()
