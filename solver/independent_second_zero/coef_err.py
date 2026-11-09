"""Relative accuracy of sigma_n from the fit plateaus of the data (issue #203, information only):
err(S_1^(2n+1)) / (|sigma_n| |S_1^(1)|), with err from fit_info["<2n+1>_1"][1] and |S_1^(1)| from the run's log line
"k 1: S_1^(1) = ...".  (The JSON's sigma_err is DBL_MAX for every n, since the k = 1 fit has err inf.)

    python3 coef_err.py <raw clone>/I82-win2
"""
import json
import os
import re
import sys

D = sys.argv[1]
for w in ["1.09", "1.10", "1.11"]:
    d = json.load(open(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json")))
    line = next(s for s in open(os.path.join(D, f"w{w}.log")) if s.startswith("k 1: S_1^(1)"))
    z = complex(re.search(r"= \(([^)]*)\)", line).group(1).replace(" ", ""))
    rel = []
    for n in range(1, len(d["sigma"])):
        err = d["fit_info"][f"{2 * n + 1}_1"][1]
        rel.append(err / (abs(complex(*d["sigma"][n])) * abs(z)))
    print(f"w {w}: |S_1^(1)| {abs(z):.3f}, sigma_err[0] {d['sigma_err'][0]:.3e}; relative fit error of sigma_1..22: "
          + " ".join(f"{r:.1e}" for r in rel) + f"; max {max(rel):.1e}")
