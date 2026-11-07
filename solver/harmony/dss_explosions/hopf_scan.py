"""The kappa = 0 (free complex scalar) CSS solution's oscillatory unstable mode, followed in D, and the kappa path at D = 4.

Uses the repository's css_pert.py / css_lateral.py (read-only import). Prints JSON lines to stdout.
    python hopf_scan.py D_LIST          kappa = 0, backgrounds from solver/matter/data/css_bg_k0.jsonl
    python hopf_scan.py kappa K_LIST    D = 4, background continued in kappa from kappa = 0
    python hopf_scan.py control         lambda = 1 at the true background, and at omega + 1e-3 (must move)
"""
import sys, json, time
REPO = '/Users/nathan/Source/choptuik-large-d/solver/matter'
sys.path.insert(0, REPO)
import numpy as np
import css_pert as cp
import css_lateral as cl

bg = {}
for l in open(REPO + '/data/css_bg_k0.jsonl'):
    d = json.loads(l); bg[d['D']] = d

def root(D, kap, om, f0, lam0, Rf):
    n = D - 2
    xh = cl.horizon_estimate(n, kap, om, f0)
    return cp.find_root(n, kap, om, f0, lam0, xh, Rf * xh)

if sys.argv[1] == 'control':
    d = bg[4.0]
    for dom in (0.0, 1e-3):
        t = time.time()
        lam = root(4.0, 0.0, d['omega'] + dom, d['f0'], complex(0.98, 0.0), 0.85)
        print(json.dumps(dict(test='lambda=1', domega=dom, lam=[lam.real, lam.imag], err=abs(lam - 1), sec=time.time() - t)), flush=True)
    sys.exit()

if sys.argv[1] == 'kappa':
    ks = [float(s) for s in sys.argv[2].split(',')]
    d = bg[4.0]; om, f0 = d['omega'], d['f0']; lam = complex(0.2586698, 3.5022488); n = 2.0
    hist = []
    for k in ks:
        t = time.time()
        xh = cl.horizon_estimate(n, k, om, f0)
        p, aJ, xh = cl.newton(n, k, om, f0, 0.85 * xh, verbose=False)
        om, f0 = p
        if len(hist) >= 2:   # linear extrapolation of lambda in kappa
            (k1, l1), (k2, l2) = hist[-2], hist[-1]
            lam = l2 + (l2 - l1) * (k - k2) / (k2 - k1)
        la = root(4.0, k, om, f0, lam, 0.85)
        lb = root(4.0, k, om, f0, la, 0.6)
        hist.append((k, la)); lam = la
        print(json.dumps(dict(D=4.0, kappa=k, omega=om, f0=f0, absJ=aJ, lam85=[la.real, la.imag], lam60=[lb.real, lb.imag],
                              sec=time.time() - t)), flush=True)
    sys.exit()

Ds = [float(s) for s in sys.argv[1].split(',')]
lam = complex(0.2586698, 3.5022488)
hist = []
for D in Ds:
    d = bg[D]; t = time.time()
    if len(hist) >= 2:
        (D1, l1), (D2, l2) = hist[-2], hist[-1]
        lam = l2 + (l2 - l1) * (D - D2) / (D2 - D1)
    la = root(D, 0.0, d['omega'], d['f0'], lam, 0.85)
    lb = root(D, 0.0, d['omega'], d['f0'], la, 0.6)
    hist.append((D, la)); lam = la
    print(json.dumps(dict(D=D, kappa=0.0, omega=d['omega'], lam85=[la.real, la.imag], lam60=[lb.real, lb.imag],
                          sec=time.time() - t)), flush=True)
