#!/usr/bin/env python3
"""Check of the horizon exponent mu (matter/CSS.md, section 3): off the solution (omega shifted by d), the jump across
the horizon behaves as J ~ c (x_h - x)^mu continued around x_h, so |J(R)| ~ R^{Re mu} for small circle radii R.

    python3 css_mucheck.py BG.jsonl D1,D2,...
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_lateral as cl

bg = {r["D"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
for D in [float(s) for s in sys.argv[2].split(",")]:
    r = bg[D]; n = D - 2
    xh = r["X"]
    Rs = np.array([0.02, 0.04, 0.08]) * xh
    d = 1e-5
    J = np.array([abs(cl.jump(n, r["kappa"], r["omega"] + d, r["f0"], R, xh=xh)[0] - cl.jump(n, r["kappa"], r["omega"], r["f0"], R, xh=xh)[0]) for R in Rs])
    slopes = np.diff(np.log(J)) / np.diff(np.log(Rs))
    print("D=%g: predicted Re mu = %.4f; measured d log|J| / d log R = %s" % (D, r["mu"][0], " ".join("%.4f" % s for s in slopes)), flush=True)
