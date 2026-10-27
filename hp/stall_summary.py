#!/usr/bin/env python3
"""The stall detector's results (stall_margin.py runs on the raw/T76 branch): the weakest direction's sigma against
the recorded max|dDelta/dF|, sigma_max, u kappa and the residual floor u sigma_max |z| at the converged states, power
laws in n = D - 2 (least squares in ln, the largest ln residual quoted), and T48's attempts by outcome.

    git worktree add ../raw-T76 raw/T76          (from the repository root, once)
    cd hp && python3 stall_summary.py ../../raw-T76
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U = 2.0 ** -104


def fit(n, y):
    """ln y = a + b ln n; returns b, a and the largest |ln residual|"""
    x, ly = np.log(n), np.log(y)
    b, a = np.polyfit(x, ly, 1)
    return b, a, float(np.abs(ly - (a + b * x)).max())


def main():
    if len(sys.argv) < 2 or not os.path.isfile(os.path.join(sys.argv[1], "kappa_states.jsonl")):
        sys.exit(__doc__)
    raw = sys.argv[1]
    sens = {round(float(r["D"]), 2): float(r["max_dDelta_dF"])
            for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))) if r["max_dDelta_dF"]}
    recs = [json.loads(x) for f in ("kappa_states.jsonl", "smoke_kappa.jsonl")
            for x in open(os.path.join(raw, f)) if x.strip()]
    recs = sorted({r["D"]: r for r in recs}.values(), key=lambda r: r["D"])
    print("converged states, full system (kfreeze 13):")
    print("        D   sigma_min harm  sigma*max|dD/dF|  sigma_max   u*kappa      |F|   |F|/(u*smax*|z|)")
    rows = []
    for r in recs:
        w = r["dirs"][0]
        zh = np.load(r["state"] if os.path.isabs(r["state"]) else os.path.join(HERE, r["state"]))["zh"]
        z = float(np.linalg.norm(zh))
        near = min(sens, key=lambda d: abs(d - r["D"]))
        prod = w["sigma"] * sens[near] if abs(near - r["D"]) < 0.05 else float("nan")
        rows.append((r["D"] - 2, w["sigma"], w["sigma_max"], w["u_kappa"]))
        print(f"  {r['D']:8.4f}  {w['sigma']:9.2e}  {w['harm']:3d}  {prod:14.2e}  {w['sigma_max']:9.2e}  "
              f"{w['u_kappa']:8.2e}  {r['F']:8.1e}  {r['F'] / (U * w['sigma_max'] * z):10.3f}")
    n = np.array([x[0] for x in rows])
    for name, col in (("sigma_min", 1), ("sigma_max", 2), ("u*kappa", 3)):
        b, a, res = fit(n, np.array([x[col] for x in rows]))
        extra = f"; = 1 at D = {math.exp(-a / b) + 2:.2f}" if col == 3 else ""
        extra += f"; {(59 / 58) ** b:.2f}x per unit of D at D 60" if col == 2 else ""
        print(f"  {name} ~ n^{b:.2f} over D {n.min() + 2:.2f}-{n.max() + 2:.2f}, largest |ln residual| {res:.2f}{extra}")
    pins = [json.loads(x) for x in open(os.path.join(raw, "kappa_pinned.jsonl")) if x.strip()]
    print("\nT73's last states (quad-double horizon):")
    for r in pins:
        w = r["dirs"][0]
        print(f"  D {r['D']:.4f} {'pin ' + str(r['pin']) if r['pin'] is not None else 'full system':11s}: weakest "
              f"harmonic {w['harm']}, sigma {w['sigma']:.2e}, sigma_max {w['sigma_max']:.2e}, u*kappa {w['u_kappa']:.2e}")
    seen, att = set(), []
    for x in open(os.path.join(raw, "seeds.jsonl")):
        r = json.loads(x)
        a = r["attempt"]
        key = (a["D_from"], a["D_to"], a["ok"], a["F_seed"])
        if key not in seen:
            seen.add(key)
            att.append((a["D_from"], a["ok"], a["F_seed"], r["F"], r["theta_max"], r["kappa_max"]))
    print("\nT48's attempts, seeds rebuilt (F_seed rebuilt / logged; logged F_seed, kappa and theta by outcome):")
    for lo, hi in ((62.0, 62.2), (64.0, 64.2), (67.60, 67.76)):
        w = [x for x in att if lo <= x[0] <= hi]
        rat = sorted(x[3] / x[2] for x in w)
        print(f"  D_from {lo}-{hi}: {sum(x[1] for x in w)} converged, {sum(not x[1] for x in w)} failed; "
              f"rebuilt / logged F_seed {rat[0]:.2f}-{rat[-1]:.2f}")
        for name, i in (("F_seed", 2), ("kappa", 5), ("theta", 4)):
            o = sorted(x[i] for x in w if x[1])
            f = sorted(x[i] for x in w if not x[1])
            print(f"    {name:6s} converged {o[0]:.2g}-{o[-1]:.2g}, failed {f[0]:.2g}-{f[-1]:.2g}")


if __name__ == "__main__":
    main()
