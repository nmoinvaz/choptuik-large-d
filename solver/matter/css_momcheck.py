#!/usr/bin/env python3
"""The linearized momentum constraint (not imposed) on the collocation eigenvectors (matter/CSS.md, section 1).

    python3 css_momcheck.py [KAPPA] [D1,D2,...]

For each D: background and spectrum by collocation (N = 48, A = 5), then for the relevant mode, lambda = 1 and the
+-i omega pair, the relative residual of  lambda dv = delta(v_tau)  (css_model.momentum_check), against the same for a
spurious (unconverged, high-frequency) eigenvector as a reference.
"""
import sys, os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_model as cm
import css_bg as cb

kap = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
Ds = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4.0, 5.0, 6.0]
om, X, f0 = (1.17695272200, 2.60909347510, 0.892555411872) if kap == 1 else (1.9154446, 5.0035380, np.sqrt(4 * np.pi) * 0.67217263 / 1.9154446)
g = cm.Grid(48, 5.0 if kap == 1 else 4.0)
Y = cb.guess_from_shooting(g, 2, kap, om, f0, X, wcut=0.999)
Y, res, it = cm.newton_bg(g, Y, 2, kap, tol=1e-10)
D = 4.0
for Dt in Ds:
    while D < Dt - 1e-9:
        D = min(D + 0.5, Dt)
        Y, res, it = cm.newton_bg(g, Y, D - 2, kap, tol=1e-10)
    n = D - 2
    lam, V = cm.spectrum(g, Y, n, kap)
    omD = Y[cm.NF * g.M]
    out = []
    order = np.argsort(-lam.real)
    for k in order[:8]:
        z = lam[k]
        if abs(z) < 1e-4:
            continue
        r = cm.momentum_check(g, Y, n, kap, z, V[:, k])
        out.append("%.6f%+.6fi: %.1e" % (z.real, z.imag, r))
    # a spurious high-frequency eigenvalue as a reference
    k = np.argmax(np.abs(lam.imag) * (lam.real > -3))
    out.append("[spurious %.2f%+.1fi: %.1e]" % (lam[k].real, lam[k].imag, cm.momentum_check(g, Y, n, kap, lam[k], V[:, k])))
    print("D=%g omega=%.8f  " % (D, omD) + ";  ".join(out), flush=True)
