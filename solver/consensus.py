"""Delta(D) across setups: interpolate every setup to a common D grid (local cubic), report each, the median of the
trusted ones and their spread. Setups are (xK, kc, smooth)."""
import numpy as np, sys
from data_ne import load
SETUPS = [(0.6, 23, True), (0.6, 31, True), (0.7, 31, True), (0.7, 23, True), (0.6, 23, False), (0.7, 23, False),
          (0.6, 31, False), (0.7, 31, False)]
TRUST = [(0.6, 23, True), (0.6, 31, True), (0.7, 31, True)]


def interp(rows, D):
    Ds = np.array(sorted(k for k in rows if rows[k]["src"] == "ne"))
    if len(Ds) < 4 or D < Ds[0] - 1e-9 or D > Ds[-1] + 1e-9:
        return np.nan
    i = np.argsort(abs(Ds - D))[:4]
    y = np.array([rows[k]["Delta"] for k in Ds[i]])
    return np.polyval(np.polyfit(Ds[i] - D, y, 3), 0.0)


def table(grid):
    data = {s: load(xK=s[0], kc=s[1], smooth=s[2]) for s in SETUPS}
    out = []
    for D in grid:
        vals = {s: interp(data[s], D) for s in SETUPS}
        t = np.array([vals[s] for s in TRUST if np.isfinite(vals[s])])
        out.append((D, vals, np.median(t) if len(t) else np.nan, (t.max() - t.min()) if len(t) > 1 else np.nan, len(t)))
    return out


if __name__ == "__main__":
    grid = np.arange(23.2, 40.01, 0.5)
    hdr = " ".join(f"{s[0]}/{s[1]}/{'sj' if s[2] else '--'}".rjust(12) for s in SETUPS)
    print(f"{'D':>6} {'median':>14} {'spread':>8} n  | differences from median: {hdr}")
    for D, vals, med, spr, n in table(grid):
        if not np.isfinite(med):
            continue
        print(f"{D:6.2f} {med:14.10f} {spr:8.1e} {n} | " + " ".join(
            (f"{vals[s] - med:+12.1e}" if np.isfinite(vals[s]) else " " * 12) for s in SETUPS))
