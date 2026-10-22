#!/usr/bin/env python3
"""Growth rate lambda(D) of the relevant mode of the CSS solution, and gamma = 1/lambda (radius exponent).

    python3 css_gamma.py BG.jsonl OUT.jsonl [LAMBDA_AT_FIRST_D]

Reads the backgrounds written by css_lateral.py (cont) and finds the root of det M(lambda) (css_pert.py, complex-path
shooting) at each D, starting from a guess extrapolated quadratically in 1/n from the previous points; the root is
recomputed with a second circle radius (0.6 x_h instead of 0.85 x_h) as a check.
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_pert as cp
import css_lateral as cl

bg = [json.loads(l) for l in open(sys.argv[1])]
bg.sort(key=lambda r: r["D"])
out = sys.argv[2]
lam0 = float(sys.argv[3]) if len(sys.argv) > 3 else 3.786349906
done = set()
if os.path.exists(out):
    for l in open(out):
        done.add(json.loads(l)["D"])
hist = []
if os.path.exists(out):
    hist = sorted([(json.loads(l)["D"], json.loads(l)["lam"]) for l in open(out)])
for r in bg:
    D = r["D"]; n = D - 2; kap = r["kappa"]
    if D in done:
        continue
    if len(hist) >= 3:
        xs = np.array([1 / (h[0] - 2) for h in hist[-3:]]); ys = np.array([h[1] for h in hist[-3:]])
        guess = np.polyval(np.polyfit(xs, ys, 2), 1 / n)
    elif len(hist) == 2:
        (Da, la), (Db, lb) = hist
        guess = lb + (lb - la) * (1 / n - 1 / (Db - 2)) / (1 / (Db - 2) - 1 / (Da - 2))
    elif len(hist) == 1:
        guess = hist[0][1]
    else:
        guess = lam0
    xh = r["X"]
    lam = cp.find_root(n, kap, r["omega"], r["f0"], complex(guess, 0.0), xh, 0.85 * xh, tol=1e-11, maxit=40)
    lam2 = cp.find_root(n, kap, r["omega"], r["f0"], lam, xh, 0.6 * xh, tol=1e-11, maxit=40)
    rec = dict(D=D, kappa=kap, omega=r["omega"], lam=lam.real, lam_im=lam.imag, lam_R2=lam2.real,
               gamma=1 / lam.real, guess=float(np.real(guess)))
    hist.append((D, lam.real))
    print("D=%6.2f  lambda = %.12f %+.1ei   (R2: %.1e)   gamma = %.12f   n(lambda-2) = %.6f"
          % (D, lam.real, lam.imag, lam2.real - lam.real, 1 / lam.real, n * (lam.real - 2)), flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
