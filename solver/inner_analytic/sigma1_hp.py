#!/usr/bin/env python3
"""sigma_1 (and the linear multiplier S^(1)) to as many digits as the late-order fits give, at any real omega.

Calls the amplitude code (solver/inner_amp/amp_flint_par.py, AmpPar, --fit phys) with K = 3, the order at which
sigma_1 = S_1^(3)/S_1^(1) is complete, and overrides omega with an exact value (an integer or a decimal string), so that
integer-relation tests can use an omega without a transcendental factor. Nothing in the amplitude code is changed.

    python3 sigma1_hp.py W J [prec] [--mf LO:HI] [--wf]      W = omega (exact), or with --wf W = omega/omega_inf
                                                             (omega_inf = 2 pi/1.64389375, as in amp_flint.py)

prints one JSON line: omega, J, prec, sigma_1 and S^(1) as 70-digit strings, the fit's error estimates, and
S^(1) * Gamma(a) Gamma(b) / (-i), which is 1 for the exact linear multiplier S^(1) = -i/(Gamma(a) Gamma(b)),
a = (1 + i omega)/2, b = (2 + i omega)/2 (solver/proof/exact_sigma.py); its distance from 1 measures the fit.
"""
import json
import os
import sys
import time

import mpmath as mp
from flint import acb, arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inner_amp"))
from amp_flint_par import AmpPar  # noqa: E402


def main(argv):
    argv = list(argv)
    mf = (8, 20)
    if "--mf" in argv:
        i = argv.index("--mf")
        mf = tuple(int(x) for x in argv[i + 1].split(":"))
        del argv[i:i + 2]
    use_wf = "--wf" in argv
    if use_wf:
        argv.remove("--wf")
    W, J = argv[1], int(argv[2])
    prec = int(argv[3]) if len(argv) > 3 else 1280
    t0 = time.time()
    A = AmpPar(1.0, 3, J, prec=prec, procs=1, fit="phys", mf=mf)
    mp.mp.dps = 120
    if use_wf:
        wmp = mp.mpf(W) * 2 * mp.pi / mp.mpf("1.64389375")
    else:
        wmp = mp.mpf(W)
    A.wmp = wmp
    A.w = acb(arb(mp.nstr(wmp, 110)))
    sig, _ = A.run(verbose=False)
    mp.mp.dps = 80
    s1 = A.sig_acb[1]                     # sig_acb[0] is sigma_0 = 1
    # S^(1) is not stored by run(); refit it from the order-1 data
    S1, e1 = A.S_at(1, 1)
    w = A.w
    a, b = (1 + acb(0, 1) * w) / 2, (2 + acb(0, 1) * w) / 2
    ratio = S1 * a.gamma() * b.gamma() / acb(0, -1)
    rec = {"omega": mp.nstr(wmp, 40), "J": J, "prec": prec, "mf": list(mf),
           "sigma1": [s1.real.mid().str(70, radius=False), s1.imag.mid().str(70, radius=False)],
           "S1": [S1.real.mid().str(70, radius=False), S1.imag.mid().str(70, radius=False)],
           "S1_closed_ratio": [ratio.real.mid().str(30, radius=False), ratio.imag.mid().str(30, radius=False)],
           "sigma_err": A.sigma_err[1] if hasattr(A, "sigma_err") else None,
           "fit_info": {k: list(v) for k, v in A.fit_info.items()}, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv)
