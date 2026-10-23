#!/usr/bin/env python3
"""Prototype of the correction loop for the mixed-precision step at large D (notes/MIXED_PRECISION.md section 8): GMRES
on J M^-1 (right preconditioning), M the f64 LU of the f64-built Jacobian, J x standing in for an exact double-double
Jacobian-vector product rounded to f64 (here the double-double Jacobian's hi part times x). Reports the products
needed and the step error against the double-double step, for three GMRES tolerances, two perturbations per state.

    cd hp && python3 mixed_largeD_gmres.py 20,40,45,50,55,60
"""
import glob
import os
import re
import sys

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import gmres, LinearOperator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import load, make_solver
H = os.path.dirname(os.path.abspath(__file__))
avail = {float(re.search(r"_D([0-9.]+)\.npz$", p).group(1)): p for p in glob.glob(H + '/data/states/up*_D*.npz')}
for D0 in [float(x) for x in sys.argv[1].split(',')]:
    p = avail[min(avail, key=lambda d: abs(d - D0))]
    zh, zl, m = load(p)
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=2, kfreeze=13 if m["D"] >= 35 else None)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    n = s.size
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        ph = np.asarray(zh) * (1 + 1e-12 * rng.standard_normal(n)); pl = np.asarray(zl).copy()
        Jh, Jl = s.jacobian(ph, pl); Jh = np.asarray(Jh)
        J64 = np.asarray(s.jacobian_f64(ph, pl))
        dh, dl, Fh, Fl, *_ = s.newton(ph, pl)
        ddz = np.asarray(dh) + np.asarray(dl); b = -(np.asarray(Fh) + np.asarray(Fl)); nr = np.linalg.norm(ddz)
        lu = sla.lu_factor(J64, check_finite=False)
        x0 = sla.lu_solve(lu, b, check_finite=False)
        res = []
        for tol in (1e-4, 1e-8, 1e-12):
            cnt = [0]
            def mv(y):  # right preconditioning: J M^-1 y (J: the exact Jacobian's hi part, as a dd JVP rounded to f64)
                cnt[0] += 1
                return Jh @ sla.lu_solve(lu, y, check_finite=False)
            A = LinearOperator((n, n), matvec=mv, dtype=float)
            y, info = gmres(A, b, rtol=tol, atol=0, restart=60, maxiter=5)
            x = sla.lu_solve(lu, y, check_finite=False)
            res.append(f"rtol {tol:.0e}: {cnt[0]} products, step error {np.linalg.norm(x - ddz) / nr:.1e}{'' if info == 0 else f' (info {info})'}")
        print(f"D {m['D']:.2f} seed {seed}: f64 alone {np.linalg.norm(x0 - ddz) / nr:.1e}; " + "; ".join(res), flush=True)
