#!/usr/bin/env python3
"""Locate the zeros of det M(lambda) in a box (argument principle: winding number and sum of the zeros), refine each
by secant iteration, and recompute it with a second circle radius: a growth rate does not move with R, an artefact of
the jump method does (matter/CSS.md section 6).

    python3 css_locate.py BG.jsonl OUT.jsonl D RE0 RE1 IM0 IM1
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_pert as cp

bg = {r["D"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
out = sys.argv[2]
D, re0, re1, im0, im1 = [float(v) for v in sys.argv[3:8]]
r = bg[D]; n = D - 2; xh = r["X"]; kap = r["kappa"]; om = r["omega"]; f0 = r["f0"]
R = 0.85 * xh
corners = [complex(re0, im0), complex(re1, im0), complex(re1, im1), complex(re0, im1), complex(re0, im0)]
vals = {}
def val(z):
    if z not in vals:
        vals[z] = cp.detM(n, kap, om, f0, z, xh, R)
    return vals[z]
W = 0.0; S = 0.0 + 0.0j
for a, c in zip(corners[:-1], corners[1:]):
    def seg(t0, t1, depth=0):
        za, zb = a + (c - a) * t0, a + (c - a) * t1
        d = np.angle(val(zb) / val(za))
        tm = 0.5 * (t0 + t1); zm = a + (c - a) * tm
        d1 = np.angle(val(zm) / val(za)); d2 = np.angle(val(zb) / val(zm))
        if (abs(d1 + d2 - d) > 1e-6 or abs(d) > 0.3) and depth < 30:
            w1, s1 = seg(t0, tm, depth + 1); w2, s2 = seg(tm, t1, depth + 1)
            return w1 + w2, s1 + s2
        # d log det = i dphase + d log|det|; lambda d log det on this step (midpoint rule)
        dl = np.log(abs(val(zb)) / abs(val(za))) + 1j * (d1 + d2)
        return d1 + d2, zm * dl
    for t0, t1 in zip(np.linspace(0, 1, 13)[:-1], np.linspace(0, 1, 13)[1:]):
        w, sm = seg(t0, t1)
        W += w; S += sm
wn = W / (2 * np.pi); ssum = S / (2j * np.pi)
print("D=%g box Re %.2f..%.2f Im %.2f..%.2f: winding %.3f, sum of zeros %.6f%+.6fi (%d evaluations)"
      % (D, re0, re1, im0, im1, wn, ssum.real, ssum.imag, len(vals)), flush=True)
rec = dict(D=D, box=[re0, re1, im0, im1], winding=wn, sum=[ssum.real, ssum.imag])
k = int(round(wn))
if k == 1:
    z1 = cp.find_root(n, kap, om, f0, ssum, xh, R, tol=1e-10, maxit=40)
    z2 = cp.find_root(n, kap, om, f0, z1, xh, 0.6 * xh, tol=1e-10, maxit=40)
    print("  zero: %.8f%+.8fi at R = 0.85 x_h;  %.8f%+.8fi at R = 0.6 x_h;  moves by %.1e" % (z1.real, z1.imag, z2.real, z2.imag, abs(z2 - z1)), flush=True)
    rec.update(zero_R085=[z1.real, z1.imag], zero_R06=[z2.real, z2.imag], moves=abs(z2 - z1),
               kappa_s=r["kappa_s"], mu=r["mu"])
with open(out, "a") as fo:
    fo.write(json.dumps(rec) + "\n")
