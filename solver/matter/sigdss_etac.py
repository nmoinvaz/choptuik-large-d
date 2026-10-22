"""D = 4: fit Delta(eta) near the DSS-CSS transition to Lechner et al.'s form Delta = -a ln(eta - eta_c) + b
(data/sigdss_d4eta15.jsonl, kc 15), and compare a with 2/lambda_CSS (lambda_CSS = 5.1428 at eta = 0.17, SIGMA.md).
usage: python sigdss_etac.py [eta_max]"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
etamax = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
rs = [json.loads(l) for l in open(os.path.join(HERE, "data", "sigdss_d4eta15.jsonl"))]
rs = [r for r in rs if abs(r["D"] - 4) < 1e-9 and r["eps"] > 0]
eta = np.array([1 / r["eps"] for r in rs]); Dl = np.array([r["Delta"] for r in rs])
o = np.argsort(eta); eta, Dl = eta[o], Dl[o]
out = {}
for emax in (etamax, 0.21, 0.23):
    sel = eta <= emax + 1e-12
    if sel.sum() < 4:
        continue
    f = lambda p: -p[0] * np.log(eta[sel] - p[2]) + p[1] - Dl[sel]
    res = least_squares(f, [0.39, 0.0, 0.17], bounds=([0, -10, 0.1], [5, 10, eta[sel].min() - 1e-6]))
    a, b, ec = res.x
    rel = np.max(np.abs(res.fun) / Dl[sel])
    print(f"eta <= {emax}: {int(sel.sum())} points (eta {eta[sel].min():.4f}..{eta[sel].max():.4f}); a = {a:.4f} "
          f"(2/a = {2 / a:.3f}; 2/5.1428 = {2 / 5.1428:.4f}), b = {b:.4f}, eta_c = {ec:.5f}, max rel err {rel:.1e}")
    out[str(emax)] = dict(n=int(sel.sum()), a=a, b=b, eta_c=ec, lam_from_a=2 / a, max_rel_err=rel)
json.dump(dict(points=[[float(e), float(d)] for e, d in zip(eta, Dl)], fits=out),
          open(os.path.join(HERE, "data", "sigdss_etac_d4.json"), "w"), indent=1)
