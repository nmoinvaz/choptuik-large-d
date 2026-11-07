"""Fixed D: continue the CSS and its oscillatory mode in kappa from kappa = 0 until Re lambda changes sign (the Hopf point
kappa_c of Hirschmann-Eardley), then refine kappa_c by secant on Re lambda(kappa).
    python hopf_kappa.py D OMEGA0 F00 LAMRE LAMIM DKAPPA KMAX
Prints JSON lines; the last line is the Hopf point."""
import sys, json, time
REPO = '/Users/nathan/Source/choptuik-large-d/solver/matter'
sys.path.insert(0, REPO)
import numpy as np
import css_pert as cp
import css_lateral as cl
D, om, f0, lr, li, dk, kmax = [float(s) for s in sys.argv[1:8]]
n = D - 2
lam = complex(lr, li)
H = []   # (kappa, om, f0, lam)

def solve(k, om, f0, lam):
    xh = cl.horizon_estimate(n, k, om, f0)
    p, aJ, xh = cl.newton(n, k, om, f0, 0.85 * xh, verbose=False)
    xh = cl.horizon_estimate(n, k, p[0], p[1])
    l85 = cp.find_root(n, k, p[0], p[1], lam, xh, 0.85 * xh)
    l60 = cp.find_root(n, k, p[0], p[1], l85, xh, 0.6 * xh)
    return p[0], p[1], aJ, l85, l60

def predict(k):
    if len(H) < 2:
        return H[-1][1:] if H else (om, f0, lam)
    (ka, oa, fa, la), (kb, ob, fb, lb) = H[-2], H[-1]
    r = (k - kb) / (kb - ka)
    return ob + r * (ob - oa), fb + r * (fb - fa), lb + r * (lb - la)

k = 0.0
while k <= kmax + 1e-12:
    t = time.time()
    o0, g0, l0 = predict(k)
    o, g, aJ, l85, l60 = solve(k, o0, g0, l0)
    print(json.dumps(dict(D=D, kappa=k, omega=o, f0=g, absJ=aJ, lam85=[l85.real, l85.imag], lam60=[l60.real, l60.imag],
                          sec=round(time.time() - t, 1))), flush=True)
    if abs(l85 - l60) > 1e-6 or aJ > 1e-9:
        print(json.dumps(dict(stop='lost', kappa=k))); sys.exit(1)
    H.append((k, o, g, l85))
    if len(H) >= 2 and H[-2][3].real > 0 >= l85.real:
        break
    k = round(k + dk, 10)
else:
    print(json.dumps(dict(stop='no crossing up to kmax'))); sys.exit(2)
# secant on Re lambda(kappa) between the last two points
(ka, oa, fa, la), (kb, ob, fb, lb) = H[-2], H[-1]
for it in range(8):
    kc = kb - lb.real * (kb - ka) / (lb.real - la.real)
    r = (kc - kb) / (kb - ka)
    o, g, aJ, l85, l60 = solve(kc, ob + r * (ob - oa), fb + r * (fb - fa), lb + r * (lb - la))
    (ka, oa, fa, la), (kb, ob, fb, lb) = (kb, ob, fb, lb), (kc, o, g, l85)
    if abs(l85.real) < 1e-10:
        break
print(json.dumps(dict(hopf=True, D=D, kappa_c=kc, omega=o, f0=g, lam85=[l85.real, l85.imag], lam60=[l60.real, l60.imag],
                      T_H=2 * np.pi / l85.imag, phase_over_2pi=o / l85.imag)), flush=True)
