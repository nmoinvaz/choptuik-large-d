"""kappa fixed: follow the CSS background and its oscillatory mode in D with small steps (secant predictors).
    python hopf_D.py KAPPA D0 OMEGA0 F00 LAMRE LAMIM D1 STEP
Prints JSON lines; stops when the two circle radii disagree by more than 1e-6 (lost mode)."""
import sys, json, time
REPO = '/Users/nathan/Source/choptuik-large-d/solver/matter'
sys.path.insert(0, REPO)
import numpy as np
import css_pert as cp
import css_lateral as cl
kap, D, om, f0, lr, li, D1, st = [float(s) for s in sys.argv[1:9]]
lam = complex(lr, li)
H = []   # (D, om, f0, lam)
while D <= D1 + 1e-12:
    t = time.time()
    if len(H) >= 2:
        (Da, oa, fa, la), (Db, ob, fb, lb) = H[-2], H[-1]
        r = (D - Db) / (Db - Da)
        om, f0, lam = ob + r * (ob - oa), fb + r * (fb - fa), lb + r * (lb - la)
    n = D - 2
    xh = cl.horizon_estimate(n, kap, om, f0)
    p, aJ, xh = cl.newton(n, kap, om, f0, 0.85 * xh, verbose=False)
    om, f0 = p
    xh = cl.horizon_estimate(n, kap, om, f0)
    l85 = cp.find_root(n, kap, om, f0, lam, xh, 0.85 * xh)
    l60 = cp.find_root(n, kap, om, f0, l85, xh, 0.6 * xh)
    print(json.dumps(dict(D=D, kappa=kap, omega=om, f0=f0, absJ=aJ, lam85=[l85.real, l85.imag], lam60=[l60.real, l60.imag],
                          sec=round(time.time() - t, 1))), flush=True)
    if abs(l85 - l60) > 1e-6 or aJ > 1e-9:
        print(json.dumps(dict(stop='radius disagreement or background not converged', D=D)), flush=True)
        break
    H.append((D, om, f0, l85)); lam = l85
    D = round(D + st, 10)
