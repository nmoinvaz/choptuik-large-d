"""Tables: the kappa = 0 CSS oscillatory mode in D, and the Hopf points kappa_c(D), against Choptuik's Delta(D)
(hp/data/delta_all.csv). T_H = 2 pi / Im lambda and phase = omega / Im lambda define the cycle born at a Hopf point,
F(tau + T_H) = e^{2 pi i phase} F(tau); Choptuik's real DSS has T = Delta/2, phase = 1/2."""
import json, csv, glob, sys
import numpy as np
O = sys.argv[1]
rows = list(csv.DictReader(open('/Users/nathan/Source/choptuik-large-d/hp/data/delta_all.csv')))
pts = sorted((float(r['D']), float(r['Delta_q0'])) for r in rows)
Dd = np.array([p[0] for p in pts]); Dl = np.array([p[1] for p in pts])
def chop(D):
    return float(np.interp(D, Dd, Dl))
print("kappa = 0, the oscillatory mode in D")
print("| D | omega (CSS) | Re lambda | Im lambda | T_H = 2pi/Im | Delta_C/2 | T_H/(Delta_C/2) | omega/Im | radius diff |")
print("|---|---|---|---|---|---|---|---|---|")
for l in open(O + '/osc_D_k0.jsonl'):
    d = json.loads(l)
    if 'stop' in d or abs(d['D'] - round(d['D'])) > 1e-9:
        continue
    lr, li = d['lam85']; dl = abs(complex(*d['lam85']) - complex(*d['lam60']))
    T = 2 * np.pi / li; h = chop(d['D']) / 2
    print("| %g | %.8f | %.8f | %.8f | %.6f | %.6f | %.5f | %.5f | %.0e |" % (d['D'], d['omega'], lr, li, T, h, T / h, d['omega'] / li, dl))
print()
print("Hopf points (Re lambda = 0) in kappa")
print("| D | kappa_c | omega_c | Im lambda | T_H | Delta_C/2 | T_H/(Delta_C/2) | phase = omega/Im |")
print("|---|---|---|---|---|---|---|---|")
for fn in sorted(glob.glob(O + '/hopf_k_D*.jsonl')):
    last = [json.loads(l) for l in open(fn)][-1]
    if not last.get('hopf'):
        print("| %s | not found: %s |" % (fn.split('_')[-1], last)); continue
    h = chop(last['D']) / 2
    print("| %g | %.6f | %.7f | %.7f | %.6f | %.6f | %.5f | %.5f |" % (last['D'], last['kappa_c'], last['omega'], last['lam85'][1],
                                                                       last['T_H'], h, last['T_H'] / h, last['phase_over_2pi']))
