#!/usr/bin/env python3
"""The corrector at a predicted state of the Taylor continuation (notes/QUAD_DOUBLE.md 13.16): from the saved
prediction (anm_step.py --save_pred) the pinned solver at D with the held harmonics from the base state, then the
residual's block norms, plain newton_solve (verbose), deflated_newton (the corrector T75's scan used: the step with
its component in the w weakest directions removed) and lm_solve, each from the same prediction.

    python3 anm_newton_diag.py PRED.npz [--threads 2] [--tol 1e-20] [--w 24]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_solve, deflated_newton, rnorm, block_norms  # noqa: E402
from pinning import PinnedSolver, band_coeffs, band_field  # noqa: E402


def main():
    args = list(sys.argv[1:])
    opt = dict(threads=2, tol=1e-20, w=24)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    d = np.load(args[0]); m = json.loads(str(d["meta"]))
    ph, pl, zh0, zl0 = d["zh"], d["zl"], d["zh0"], d["zl0"]
    D, M, nodes, kf = m["D"], m["M"], list(m["nodes"]), m["kfreeze"]
    s = make_solver(D, M, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=opt["threads"], kfreeze=kf)
    s.set_segs(s.plan(ph, pl, 1e-32, 0.05))
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    h_hi, h_lo = band_field(band_coeffs(zh0[:M], zl0[:M], M, ks_held), ks_held, M)
    sp = PinnedSolver(s, M, h_hi, h_lo)
    Fh, Fl = sp.residual(ph, pl)
    print(f"{os.path.basename(args[0])}: D {D}, pin {kf} (held {ks_held}), predicted |F|_2 {rnorm(Fh, Fl):.2e}; block max "
          + " ".join(f"{x:.1e}" for x in block_norms(s, np.asarray(Fh), np.asarray(Fl))), flush=True)
    w = opt["w"]
    for name, fn in (("newton_solve", lambda: newton_solve(sp, ph, pl, tol=opt["tol"], it_max=4, verbose=True)),
                     ("deflated_newton", lambda: deflated_newton(sp, ph, pl, tol=opt["tol"], it_max=12, w=w, verbose=True)),
                     ("deflated_newton junk_only", lambda: deflated_newton(sp, ph, pl, tol=opt["tol"], it_max=12, w=w, verbose=True, junk_only=True)),
                     ("deflated_newton junk_only 2w", lambda: deflated_newton(sp, ph, pl, tol=opt["tol"], it_max=12, w=2 * w, verbose=True, junk_only=True)),
                     ("deflated_newton junk_only delta_max 1e-7", lambda: deflated_newton(sp, ph, pl, tol=opt["tol"], it_max=12, w=w, verbose=True, junk_only=True, delta_max=1e-7))):
        print(f"--- {name}", flush=True)
        t0 = time.time()
        try:
            zh, zl, hist = fn()
            Fh, Fl = sp.residual(zh, zl)
            print(f"    {name}: |F|_2 {rnorm(Fh, Fl):.2e} after {len(hist)} records, {time.time() - t0:.0f} s; block max "
                  + " ".join(f"{x:.1e}" for x in block_norms(s, np.asarray(Fh), np.asarray(Fl)))
                  + f"; moved {float(np.max(np.abs((zh - ph) + (zl - pl)))):.2e}, Delta {float(zh[-1] + zl[-1]):.15f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"    {name}: {type(ex).__name__}: {ex}", flush=True)


if __name__ == "__main__":
    main()
