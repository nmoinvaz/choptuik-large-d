#!/usr/bin/env python3
"""The numerical M-harmonic zero to full double precision (double precision, NOT rigorous): hier.py prints (A, w/w_inf)
at the end of the outer Newton step but (B_nu, C_nu) only at intermediate iterates, so this re-solves the higher levels
S_3 = .. = S_M = 0 at the printed (A, w/w_inf) with solver/inner_stokes/hier.Hier (unchanged), prints x to 12 digits,
and evaluates the de-mixed residual there (inner_np.residual_demixed at xi1 = 0.2, and hier's Schur-corrected r_1).

    python3 refine_zero_M.py --M=3 --A=2.726475886 --wf=1.015603095 --x=0.144419478,0.011087426 [--json=f.json]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
import hier  # noqa: E402
import inner_np as I  # noqa: E402


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    M = int(opts["M"])
    x = np.array([float(opts["A"]), float(opts["wf"])] + [float(v) for v in opts["x"].split(",")])
    t0 = time.time()
    h = hier.Hier(M, 0.2, top=3, verbose=1)
    h.schur = True
    x, r = h.solve_from(3, x)
    raw = h.res_raw(x)
    print(f"M = {M}: x = [" + ", ".join(f"{v:.12f}" for v in x) + "]")
    print(f"  hier residual (Schur-corrected): " + ", ".join(f"{v:+.3e}" for v in r))
    print(f"  hier residual (raw): " + ", ".join(f"{v:+.3e}" for v in raw))
    rd = I.residual_demixed(x, M, r1=0.2)
    print(f"  inner_np.residual_demixed at xi1 = 0.2: " + ", ".join(f"{v:+.3e}" for v in rd) + f"  [{time.time() - t0:.0f} s]")
    if "json" in opts:
        with open(opts["json"], "w") as fh:
            json.dump({"M": M, "x": [f"{v:.12f}" for v in x], "r_schur": list(map(float, r)), "r_raw": list(map(float, raw)),
                       "r_demixed_0.2": list(map(float, rd))}, fh, indent=1)


if __name__ == "__main__":
    main()
