#!/usr/bin/env python3
"""Uniqueness check (matter/CSS.md section 6): the number of zeros of det M(lambda) in a box of the right half plane,
by the argument principle, for backgrounds in BG.jsonl.

    python3 css_count.py BG.jsonl OUT.jsonl D:RE0:RE1:IM[:RFRAC] [...]

A growth rate is a zero of det M for every circle radius R = RFRAC x_h (default 0.85); the artefacts of the jump method
move with R.  So the number of growth rates in the box is at most the smallest count over the radii tried: when some R
gives exactly the known zeros (lambda = 1 and the relevant mode), there is no other growth rate in the box.

The phase of det M is tracked along the box with bisection until each step is below 0.5 rad and the two halves agree
with the whole.  Known zeros: lambda = 1 (time translation) and the relevant mode; +-i omega and 0 lie on Re = 0,
outside the box.  The method has spurious zeros near the resonance line Im lambda = kappa_s Im mu (where mu_pert(lambda) is real).
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_pert as cp

bg = {r["D"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
out = sys.argv[2]
for spec in sys.argv[3:]:
    parts = [float(v) for v in spec.split(":")]
    D, re0, re1, im1 = parts[:4]
    rf = parts[4] if len(parts) > 4 else 0.85
    r = bg[D]; n = D - 2; xh = r["X"]
    res_line = r["kappa_s"] * r["mu"][1]
    res_first = r["kappa_s"] * (r["mu"][0] + 1)
    w, nev = cp.winding(n, r["kappa"], r["omega"], r["f0"], xh, rf * xh, re0, re1, im1, K=12)
    rec = dict(D=D, box=[re0, re1, -im1, im1], R_over_xh=rf, winding=w, evaluations=nev,
               resonance_line_Im=res_line, first_resonance_Re=res_first)
    print("D=%g, R = %.2f x_h: zeros of det M in %.2f < Re lam < %.2f, |Im lam| < %.2f: %.4f  (%d evaluations; resonance line Im %.3f)"
          % (D, rf, re0, re1, im1, w, nev, res_line), flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
