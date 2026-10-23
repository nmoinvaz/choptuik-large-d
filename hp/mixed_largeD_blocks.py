#!/usr/bin/env python3
"""Which block of the f64 Jacobian spoils the mixed step at large D (notes/MIXED_PRECISION.md section 8)? At each state
(perturbed by 1e-12 relative, two seeds) the f64 Jacobian gets one block of 4M columns, or of 4M rows, replaced by the
exact (double-double hi) one, and the step error against the double-double step is printed.

    cd hp && python3 mixed_largeD_blocks.py 50,55,60
"""
import glob
import re
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from hpkernel.driver import load, make_solver
avail = {float(re.search(r"_D([0-9.]+)\.npz$", p).group(1)): p for p in glob.glob(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'data', 'states', 'up*_D*.npz'))}
for D0 in [float(x) for x in sys.argv[1].split(',')]:
    p = avail[min(avail, key=lambda d: abs(d - D0))]
    zh, zl, m = load(p)
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=2, kfreeze=13 if m["D"] >= 35 else None)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    n = s.size; M = m["M"]; bs = 4 * M
    nb = (n - 1) // bs
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        ph = np.asarray(zh) * (1 + 1e-12 * rng.standard_normal(n)); pl = np.asarray(zl).copy()
        Jh, _ = s.jacobian(ph, pl); Jh = np.asarray(Jh)
        J64 = np.asarray(s.jacobian_f64(ph, pl))
        dh, dl, Fh, Fl, *_ = s.newton(ph, pl)
        ddz = np.asarray(dh) + np.asarray(dl); b = -(np.asarray(Fh) + np.asarray(Fl)); nr = np.linalg.norm(ddz)
        c = lambda J: np.linalg.norm(sla.lu_solve(sla.lu_factor(J, check_finite=False), b, check_finite=False) - ddz) / nr
        out = [f"all f64 {c(J64):.1e}"]
        for k in range(nb + (1 if n > nb * bs else 0)):
            sl = slice(k * bs, min((k + 1) * bs, n))
            Jc = J64.copy(); Jc[:, sl] = Jh[:, sl]
            Jr = J64.copy(); Jr[sl, :] = Jh[sl, :]
            out.append(f"blk{k}: cols {c(Jc):.1e} rows {c(Jr):.1e}")
        print(f"D {m['D']:.2f} n {n} bs {bs} seed {seed}: " + "; ".join(out), flush=True)
