"""Amplitude of the leading-order large-D profile B(tau) = c q / x (q = (p - m)/2) near the centre, against n = D - 2,
from the converged NE solutions: does the critical solution approach flat space (B -> 0) as D grows, and how fast?

    python3 largeD_amp.py            (from /home/claude/dss, where the solutions live)
"""
import glob
import re

import numpy as np

from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L


def profile(D, z, nodes, kc=23):
    S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=60), z[-1])
    parts, _ = S.split(z[None, :])
    parts = [p[0] for p in parts]
    pars = S.parities()
    F = [S.phys(c[None, :], p)[0] for c, p in zip(parts, pars)]
    out = {}
    for i, x in enumerate(nodes):
        p, m, a, c = F[2 + 4 * i:6 + 4 * i]
        out[x] = (p, m, a, c)
    return out, float(z[-1])


def harm(f):
    F = np.fft.rfft(f) / len(f)
    return 2 * np.abs(F)


rows = []
files = sorted(glob.glob("neB_rk2_D*.npy")) + sorted(glob.glob("ne7_js_rk3_s111111118_D*.npy"))
for f in files:
    D = float(re.findall(r"D([0-9.]+)\.npy", f)[0])
    z = np.load(f)
    nodes = NODES_L + ([0.7] if "ne7" in f else [])
    try:
        pr, Dl = profile(D, z, nodes)
    except Exception as e:
        continue
    p, m, a, c = pr[0.02]
    B = c * 0.5 * (p - m) / 0.02
    pK, mK, aK, cK = pr[nodes[-1]]
    h = harm(B)
    rows.append((D, Dl, h[1], h[3], h[5], np.abs(B).max(), 1 - aK.min(), 1 - (a := pr[0.3][2]).min()))
rows.sort()
print(f"{'D':>7s} {'n':>6s} {'Delta':>9s} {'B_1':>8s} {'B_3/B_1':>8s} {'B_5/B_1':>8s} {'B_1*sqrt(n)':>11s} {'B_1*n':>8s} {'1-min a(0.3)':>12s}")
for D, Dl, b1, b3, b5, bm, ak, a3 in rows:
    n = D - 2
    print(f"{D:7.3f} {n:6.2f} {Dl:9.6f} {b1:8.5f} {b3 / b1:8.4f} {b5 / b1:8.4f} {b1 * n ** .5:11.4f} {b1 * n:8.4f} {a3:12.5f}")
