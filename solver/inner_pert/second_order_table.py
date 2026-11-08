"""The tables of notes/SECOND_ORDER.md section 4-5 from the JSON results of second_order_run.py (M = 1, 3) and
second_order_hier.py (M = 5) in the raw-data folder (I94-mac of choptuik-large-d-raw, given as the argument), against
the fitted coefficients of second_order_fits.py (the values below are its output, log second_order_fits.log).

    Delta = Delta* + Delta_1/n + Delta_2/n^2,  Delta_2 = 2 pi (omega_1^2/omega*^3 - omega_2/omega*^2);
    in eps_E = 1/(D - 1) (EEG): Delta_2^E = Delta_2 + Delta_1;  in 1/D: Delta_2^D = Delta_2 + 2 Delta_1.
The M -> infinity readings assume the next error ratio r of an alternating sequence (notes/FIRST_ORDER.md section 6):
limit = x_5 + (-r/(1 - r)) (x_3 - x_5); they are readings of three-term sequences, not error bars.

    python3 second_order_table.py /path/to/choptuik-large-d-raw/I94-mac
"""
import json
import os
import sys

import numpy as np

raw = sys.argv[1]
FIT = dict(D1=10.37528, D2=-49.1685, D2lo=-49.16867, D2hi=-49.16780, L1=3.0324, L2=-34.376, L2lo=-34.3788, L2hi=-34.3738)


def load(name):
    fn = os.path.join(raw, name)
    return json.load(open(fn)) if os.path.exists(fn) else None


rows = {1: [load("second_order_M1_xi0.2.json")], 3: [load("second_order_M3_xi0.2.json"), load("second_order_M3_xi0.175.json")],
        5: [load("second_order_hier_M5_xi0.2.json"), load("second_order_hier_M5_xi0.19.json"),
            load("second_order_hier_M5_xi0.2_rtol1e-13.json")]}
print("Delta_2 (1/n) by run:")
D2 = {}
for M, rs in rows.items():
    vals = [r["Delta2"] for r in rs if r]
    for r in rs:
        if r:
            print(f"  M = {M}  xi1 = {r['xi1']:g}  rtol = {r['rtol']:g}:  omega_2 = {r['omega2']:.3f}  Delta_1 = {r['Delta1']:.5f}  Delta_2 = {r['Delta2']:.4f}"
                  + (f"  (all gauge data 0: {r['Delta2_gauge0']:.4f})" if "Delta2_gauge0" in r else "")
                  + (f"  lambda_2 = {r['lam2']:.4f}" if "lam2" in r else "")
                  + (f"  (gauge 0: lambda_1 {r['lam1_gauge0']:.4f}, lambda_2 {r['lam2_gauge0']:.4f})" if "lam2_gauge0" in r else ""))
    D2[M] = (float(np.mean(vals)), (max(vals) - min(vals)) / 2)
print("\nDelta_2: mean over runs, +- half the spread of the runs (numerical error of the model value)")
for M, (m, h) in D2.items():
    print(f"  M = {M}: {m:.3f} +- {h:.3f}   difference from the fit {m - FIT['D2']:+.3f} ({(m - FIT['D2']) / abs(FIT['D2']) * 100:+.2f}%)")
print(f"  fit (spread over fit choices): {FIT['D2']} [{FIT['D2lo']}, {FIT['D2hi']}]")
e = {M: D2[M][0] - FIT["D2"] for M in D2}
print(f"  error ratios: M3/M1 {e[3] / e[1]:+.2f}, M5/M3 {e[5] / e[3]:+.2f}")
x3, x5 = D2[3][0], D2[5][0]
for r in (-0.05, -0.2):
    print(f"  M -> infinity reading with r = {r}: {x5 + (-r / (1 - r)) * (x3 - x5):.3f}")
print(f"  the fit lies {(FIT['D2'] - x5) / (x3 - x5) * 100:.0f}% of the way from M = 5 to M = 3")
D1 = rows[5][0]["Delta1"]
print(f"\nconventions (M = 5, mean of the runs): Delta_2 (1/n) {D2[5][0]:.3f};  in eps_E = 1/(D-1): {D2[5][0] + D1:.3f};  in 1/D: "
      f"{D2[5][0] + 2 * D1:.3f};  fit: {FIT['D2'] + FIT['D1']:.3f} (eps_E), {FIT['D2'] + 2 * FIT['D1']:.3f} (1/D)")
print("\nlambda_2 by M (light-cone gauge):")
L2 = {}
for M, rs in rows.items():
    vals = [r["lam2"] for r in rs if r and "lam2" in r]
    if vals:
        L2[M] = (float(np.mean(vals)), (max(vals) - min(vals)) / 2)
        print(f"  M = {M}: {L2[M][0]:.3f} +- {L2[M][1]:.3f} ({len(vals)} runs)   difference from the fit {L2[M][0] - FIT['L2']:+.3f}")
print(f"  fit (estimator range, its planted test failed at 0.02; section 5): {FIT['L2']} [{FIT['L2lo']}, {FIT['L2hi']}]")
if len(L2) >= 3:
    l3, l5 = L2[3][0], L2[5][0]
    print(f"  error ratios: M3/M1 {(l3 - FIT['L2']) / (L2[1][0] - FIT['L2']):+.3f}, M5/M3 {(l5 - FIT['L2']) / (l3 - FIT['L2']):+.3f}")
    for r in (-0.05, -0.2):
        print(f"  M -> infinity reading with r = {r}: {l5 + (-r / (1 - r)) * (l3 - l5):.3f}")
Dinf = 1.643893746
print(f"\nfit in omega: omega_1 = {-2 * np.pi * FIT['D1'] / Dinf ** 2:.4f}, "
      f"omega_2 = 2 pi (Delta_1^2/Delta_inf^3 - Delta_2/Delta_inf^2) = {2 * np.pi * (FIT['D1'] ** 2 / Dinf ** 3 - FIT['D2'] / Dinf ** 2):.3f}")
