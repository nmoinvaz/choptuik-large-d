#!/usr/bin/env python3
"""Uniqueness check (matter/CSS.md section 6): the number of zeros of det M(lambda) in a box of the right half plane,
by the argument principle, for backgrounds in BG.jsonl.

    python3 css_count.py BG.jsonl OUT.jsonl D:RE0:RE1:IM[:RFRAC] [...] [--K=N]
    python3 css_count.py selftest       the planted long-edge box (exit 1 if it fails or its control does not fire)

A growth rate is a zero of det M for every circle radius R = RFRAC x_h (default 0.85); the artefacts of the jump method
move with R.  So the number of growth rates in the box is at most the smallest count over the radii tried: when some R
gives exactly the known zeros (lambda = 1 and the relevant mode), there is no other growth rate in the box.

The phase of det M is tracked along the box (css_pert.winding_path) from one starting sample per 0.25 of edge, with
bisection until each step is below 0.5 rad and the two halves agree with the whole; --K=N starts every edge at N
samples instead (12 before #809), which on long edges can miss whole turns of the phase.  Known zeros: lambda = 1 (time translation) and the relevant mode; +-i omega and 0 lie on Re = 0,
outside the box.  The method has spurious zeros near the resonance line Im lambda = kappa_s Im mu (where mu_pert(lambda) is real).
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_pert as cp

if sys.argv[1] == "selftest":
    # Planted truth (#809): F = exp(pi lam^2) (lam - z1)(lam - z2)(lam - z3), three zeros inside the box
    # 0.2 < Re < 2, |Im| < 6. On the right edge (Re = 2) the phase 4 pi y turns by 4 pi per interval of a K = 12 start
    # (length 1), 2 pi per half, so the bisection test passes the intervals where the zeros add little phase: the four
    # with Im < -3 or Im > 5 lose 2 turns each, and K = 12 counts 3 - 8 = -5. One sample per 0.25 of edge leaves every
    # half-interval below pi, so the default must count 3. Pass: default = 3 to 1e-6 and K = 12 differs from 3 by at
    # least 1 (the control).
    zs = [1.0 + 0.5j, 1.5 - 2.0j, 0.7 + 4.0j]

    def F(z):
        return np.exp(np.pi * z * z) * np.prod([z - q for q in zs])
    box = [complex(0.2, -6), complex(2, -6), complex(2, 6), complex(0.2, 6)]
    w = cp.winding_path(F, box)
    w12 = cp.winding_path(F, box, K=12)
    ok, ctl = abs(w - 3) < 1e-6, abs(w12 - 3) >= 1
    print("selftest: default %.6f (expect 3), K = 12 %.6f (expect -5, the control)" % (w, w12))
    print("selftest %s, control %s" % ("PASS" if ok else "FAIL", "fires" if ctl else "DOES NOT FIRE"))
    sys.exit(0 if ok and ctl else 1)
K = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--K=")), None)
bg = {r["D"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
out = sys.argv[2]
for spec in [a for a in sys.argv[3:] if not a.startswith("--")]:
    parts = [float(v) for v in spec.split(":")]
    D, re0, re1, im1 = parts[:4]
    rf = parts[4] if len(parts) > 4 else 0.85
    r = bg[D]; n = D - 2; xh = r["X"]
    res_line = r["kappa_s"] * r["mu"][1]
    res_first = r["kappa_s"] * (r["mu"][0] + 1)
    w, nev = cp.winding(n, r["kappa"], r["omega"], r["f0"], xh, rf * xh, re0, re1, im1, K=K)
    rec = dict(D=D, box=[re0, re1, -im1, im1], R_over_xh=rf, winding=w, evaluations=nev, K=K,
               resonance_line_Im=res_line, first_resonance_Re=res_first)
    print("D=%g, R = %.2f x_h: zeros of det M in %.2f < Re lam < %.2f, |Im lam| < %.2f: %.4f  (%d evaluations; resonance line Im %.3f)"
          % (D, rf, re0, re1, im1, w, nev, res_line), flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
