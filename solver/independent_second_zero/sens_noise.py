"""Sensitivity of the real-data crossing to the coefficients' accuracy (issue #203, information only): the real sigma_n
each times (1 + rel (g1 + i g2)), g standard normal, through the same estimator.

    python3 sens_noise.py <raw clone>/I82-win2 REL [SEED]
"""
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lapl  # noqa: E402
import planted  # noqa: E402
import second_zero as sz  # noqa: E402

D, rel = sys.argv[1], float(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
mp.mp.dps = lapl.DPS
sigs = [lapl.load_sigma(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json"))[0] for w in sz.WS]
sigs = [planted.noisy(s, seed + 1000 * j, rel) for j, s in enumerate(sigs)]
mp.mp.dps = lapl.DPS
print(f"real sigma with relative noise {rel}, seed {seed}")
res = sz.run_three(sigs, log=lambda s: print(s, flush=True) if "rejected" not in s else None)
sz.fmt_summary(res)
