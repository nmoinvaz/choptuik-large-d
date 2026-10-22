#!/usr/bin/env python3
"""Ecker, Ecker, Grumiller & Jechtl's published low-D data (github.com/EckerChristian/CriticalSpacetimeCrystal, not
included here) per Fourier resolution: Delta and gamma at 128 / 256 / 512 / 1024 modes for D <= 4, and the change
between resolutions (their self-convergence), next to our values in hp/data/lowD_dd.jsonl and solver/data/lowD_*.jsonl.

    git clone --depth 1 https://github.com/EckerChristian/CriticalSpacetimeCrystal ecc
    python3 lowD_ecker.py ecc [--dmax=4.0]
    python3 lowD_ecker.py ecc --fields     (their exported fields: amplitude and harmonic decay at the centre)

Reads only their data files (FortranData/dim*/<N>/Delta.out, gam.out; DeltaList.dat, GammaList.dat).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def first_float(path):
    try:
        return float(open(path).read().split()[0])
    except Exception:
        return None


def fields(ecc):
    """their exported fields (plots/fields_<D>_0.json): U(tau) at the smallest x, the period located by the best
    antiperiodicity of a spline through the samples, then |F_1| = sqrt(D - 2) c_1 / x (our F = sqrt(n) dp/dx at x = 0;
    U/x -> p1 in our normalisation: their D = 4 gives our |F_1| = 16.23), F_3/F_1 and the decay rate beta of the odd
    harmonics over k = 9..31; with the antiperiodicity error of the located period (0 for an exact period)."""
    from scipy.interpolate import CubicSpline
    for fn in sorted(glob.glob(os.path.join(ecc, "plots", "fields_*_0.json"))):
        D = float(os.path.basename(fn).split("_")[1])
        d = json.load(open(fn))
        xs = sorted([k for k in d if k.replace(".", "").isdigit()], key=float)
        x0 = xs[0]
        v = np.array(d[x0]["U"], float)
        cs = CubicSpline(np.arange(len(v)), v)
        best = None
        for P in np.linspace(len(v) - 8, len(v), 801):
            w = cs(np.arange(256) * P / 256)
            e = float(np.abs(w[:128] + w[128:]).max() / np.abs(w).max())
            if best is None or e < best[0]:
                best = (e, P)
        e, P = best
        w = cs(np.arange(256) * P / 256)
        c = np.abs(np.fft.rfft(w)) / 256 * 2
        k = np.arange(len(c))
        sel = (k % 2 == 1) & (k >= 9) & (k <= 31)
        beta = -np.polyfit(k[sel], np.log(c[sel]), 1)[0]
        F1 = np.sqrt(D - 2) * c[1] / float(x0)
        print(f"  D {D}: x {x0}, {len(v)} samples, period {P:.3f} samples (antiperiodicity error {e:.1e}): |F_1| {F1:.2f}, "
              f"F_3/F_1 {c[3] / c[1]:.3f}, beta(9..31) {beta:.3f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ecc = args[0] if args else "ecc"
    if "--fields" in sys.argv:
        print("Ecker et al.'s exported fields at the smallest x:")
        fields(ecc)
        return
    dmax = float(opts.get("dmax", 4.0))
    root = os.path.join(ecc, "complete_data", "FortranData")
    DL = {round(float(a), 6): float(b) for a, b in (l.split() for l in open(os.path.join(ecc, "complete_data", "DeltaList.dat")) if l.strip())}
    GL = {round(float(a), 6): float(b) for a, b in (l.split() for l in open(os.path.join(ecc, "complete_data", "GammaList.dat")) if l.strip())}
    rows = []
    for d in glob.glob(os.path.join(root, "dim*")):
        D = float(os.path.basename(d)[3:])
        if D > dmax + 1e-9:
            continue
        v, g = {}, {}
        for res in ("128", "256", "512", "1024"):
            v[res] = first_float(os.path.join(d, res, "Delta.out"))
            g[res] = first_float(os.path.join(d, res, "gam.out"))
        rows.append((D, v, g))
    rows.sort()
    print("Ecker et al. Fortran data: Delta by number of Fourier modes, and the change between resolutions")
    print("     D     Delta(128)      Delta(256)      Delta(512)      Delta(1024)   256-128   512-256  1024-512   DeltaList")
    for D, v, g in rows:
        f = lambda x: f"{x:.12f}" if x is not None else " " * 14
        dd = lambda a, b: f"{v[b] - v[a]:+.1e}" if v[a] is not None and v[b] is not None else "    -   "
        print(f"{D:8.5f}  {f(v['128'])}  {f(v['256'])}  {f(v['512'])}  {f(v['1024'])}  {dd('128', '256')}  {dd('256', '512')}  "
              f"{dd('512', '1024')}   {DL.get(round(D, 6), float('nan')):.12f}")
    print("\ngamma by number of Fourier modes (gam.out) and GammaList")
    for D, v, g in rows:
        f = lambda x: f"{x:.9f}" if x is not None else " " * 11
        print(f"{D:8.5f}  {f(g['128'])}  {f(g['256'])}  {f(g['512'])}  {f(g['1024'])}   {GL.get(round(D, 6), float('nan')):.9f}")
    # ours
    ours = []
    extra = [os.path.join(HERE, "..", "solver", "data", "scan_lowD3_kc47.jsonl")]
    for fn in [os.path.join(HERE, "data", "lowD_dd.jsonl"), os.path.join(HERE, "data", "lowD_gamma.jsonl")] + sorted(glob.glob(os.path.join(HERE, "..", "solver", "data", "lowD_f64*.jsonl"))) + extra:
        if os.path.exists(fn):
            for l in open(fn):
                if l.strip():
                    r = json.loads(l)
                    if r.get("converged", True):
                        ours.append((os.path.basename(fn), r))
    if ours:
        print("\nours against their lists (DeltaList / GammaList at the same D; off their grid, cubic through their 4 "
              "nearest points, marked ~)")
        dk, gk = np.array(sorted(DL)), np.array(sorted(GL))

        def interp(tab, keys, D):
            if round(D, 6) in tab:
                return tab[round(D, 6)], ""
            j = np.argsort(np.abs(keys - D))[:4]
            if np.abs(keys[j] - D).max() > 0.1:
                return None, ""
            return float(np.polyval(np.polyfit(keys[j] - D, [tab[k] for k in keys[j]], 3), 0.0)), "~"

        for fn, r in ours:
            D = r["D"]
            e, m1 = interp(DL, dk, D)
            eg, m2 = interp(GL, gk, D)
            Dl = r.get("Delta", 0) + r.get("Delta_lo", 0)
            s = f"  {fn:28s} D {D:.6f} M/kc {r.get('M', r.get('kc'))}:" + (f" Delta {Dl:.12f}" if "Delta" in r else "")
            if e is not None and "Delta" in r:
                s += f"  Ecker{m1} {e:.12f}  Ecker-ours {e - Dl:+.2e}"
            if r.get("gamma") and eg is not None:
                s += f"  gamma ours {r['gamma']:.9f} Ecker{m2} {eg:.9f} diff {eg - r['gamma']:+.1e}"
            print(s)


if __name__ == "__main__":
    main()
