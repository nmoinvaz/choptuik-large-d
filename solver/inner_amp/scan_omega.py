"""Parallel omega scan of the inner Stokes multiplier series (amp_flint.py), N worker processes.

For each omega = wf * omega_inf it computes sigma_0..sigma_{(K-1)/2} (S_1/S_1^(1) = sum sigma_k eps^(2k)) and the tuned
harmonics f_nu^(k), and writes one JSON per omega into OUTDIR (skipping omegas already done, so it can be resumed).

    python3 scan_omega.py WF_LO WF_HI N_OMEGA K J PROCS [OUTDIR] [PREC]
    e.g. python3 scan_omega.py 0.70 1.30 13 13 500 8 scan13

Runtime (python-flint 0.9, 320 bits; measured in the cloud container on a shared core, load ~6): K = 9, J = 500
about 17 min per omega (k = 3: 2.3 min, 5: 7 min, 7: 12 min); expect ~2x faster on a free core. Cost grows roughly
like K^3 J. Use J >= 500: the fitted constants converge only once the neighbour families j^(+-2 i omega) are resolved
(sigma_4 moves by 1.5e-4 between J = 160 and 500; sigma_1..3 agree to 1e-7). With 8 processes on 8 threads,
16 omegas at K = 13, J = 500 take roughly 1.5-3 h.
Then:  python3 zero_selection.py OUTDIR    (Pade zeros of S_1 in eps^2 and the omega where the zero is real).
"""
import os
import sys
import json
from multiprocessing import Pool


def one(args):
    wf, K, J, outdir, prec = args
    fn = os.path.join(outdir, f"sigF_w{wf:.5f}_K{K}_J{J}.json")
    if os.path.exists(fn):
        return fn
    os.nice(10) if hasattr(os, "nice") else None
    from amp_flint import Amp
    import mpmath as mp
    A = Amp(wf, K, J, prec=prec)
    sig, f = A.run(verbose=False)
    json.dump({"wf": wf, "K": K, "J": J, "prec": prec,
               "sigma_str": [[mp.nstr(z.real, 25), mp.nstr(z.imag, 25)] for z in sig],
               "sigma": [[float(z.real), float(z.imag)] for z in sig], "f": f}, open(fn, "w"))
    return fn


if __name__ == "__main__":
    lo, hi, n, K, J, procs = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    outdir = sys.argv[7] if len(sys.argv) > 7 else f"scan_K{K}_J{J}"
    prec = int(sys.argv[8]) if len(sys.argv) > 8 else 320
    os.makedirs(outdir, exist_ok=True)
    wfs = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]
    with Pool(procs) as p:
        for fn in p.imap_unordered(one, [(wf, K, J, outdir, prec) for wf in wfs]):
            print("done", fn, flush=True)
