#!/usr/bin/env python3
"""Ecker, Ecker, Grumiller & Jechtl's published low-D data (github.com/EckerChristian/CriticalSpacetimeCrystal, not
included here) per Fourier resolution: Delta and gamma at 128 / 256 / 512 / 1024 modes for D <= 4, and the change
between resolutions (their self-convergence), next to our values in hp/data/lowD_dd.jsonl and solver/data/lowD_*.jsonl.

    git clone --depth 1 https://github.com/EckerChristian/CriticalSpacetimeCrystal ecc
    python3 lowD_ecker.py ecc [--dmax=4.0]

Reads only their data files (FortranData/dim*/<N>/Delta.out, gam.out; DeltaList.dat, GammaList.dat).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def first_float(path):
    try:
        return float(open(path).read().split()[0])
    except Exception:
        return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ecc = args[0] if args else "ecc"
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
    for fn in [os.path.join(HERE, "data", "lowD_dd.jsonl")] + sorted(glob.glob(os.path.join(HERE, "..", "solver", "data", "lowD_*.jsonl"))):
        if os.path.exists(fn):
            for l in open(fn):
                if l.strip():
                    r = json.loads(l)
                    ours.append((os.path.basename(fn), r))
    if ours:
        print("\nours against their lists (DeltaList / GammaList at the same D, where they have it)")
        for fn, r in ours:
            D = r["D"]
            e = DL.get(round(D, 6))
            eg = GL.get(round(D, 6))
            Dl = r.get("Delta", 0) + r.get("Delta_lo", 0)
            s = f"  {fn:28s} D {D:.6f} M/kc {r.get('M', r.get('kc'))}: Delta {Dl:.12f}"
            if e is not None:
                s += f"  Ecker {e:.12f}  Ecker-ours {e - Dl:+.2e}"
            if r.get("gamma") and eg is not None:
                s += f"  gamma ours {r['gamma']:.9f} Ecker {eg:.9f} diff {eg - r['gamma']:+.1e}"
            print(s)


if __name__ == "__main__":
    main()
