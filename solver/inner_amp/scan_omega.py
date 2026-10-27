"""Parallel omega scan of the inner Stokes multiplier series (amp_flint_par.py), PROCS omegas at a time.

For each omega = wf * omega_inf it computes sigma_0..sigma_{(K-1)/2} (S_1/S_1^(1) = sum sigma_k eps^(2k)) and the tuned
harmonics f_nu^(k), and writes one JSON per omega into OUTDIR (skipping omegas already done, so it can be resumed).

    python3 scan_omega.py WF_LO WF_HI N_OMEGA K J PROCS [OUTDIR] [PREC] [--inner M] [--serial]
                          [--fit phys [--mf LO:HI] [--fit-step S] [--phys-prec BITS]]
    e.g. python3 scan_omega.py 0.70 1.30 13 13 500 8 scan13             8 omegas at a time, 1 process each
         python3 scan_omega.py 0.70 1.30 13 17 500 2 scan17 --inner 4  2 omegas at a time, 4 processes each

PROCS omegas run at once and each uses M processes (--inner, default 1), so PROCS x M processes in all. Prefer M = 1
while there are at least as many omegas left as cores (no communication at all); use M > 1 for a few omegas at large K,
where one omega would otherwise run for hours. --serial uses the original amp_flint.Amp (slow; for checks).

Runtime (cloud container, one process, shared core): K = 7, J = 300: 3.4 s (amp_flint: 90 s); K = 11, J = 500: 38 s;
K = 17, J = 500: 147 s; K = 13, J = 1000: ~5 min. The (j, mu) products are ~95% of it and grow like ~K^3.5 and ~J^2.5
(the rows grow like j!, which makes the ball products slower than linear in J).
Use J >= 500: the fitted constants converge only once the neighbour families j^(+-2 i omega) are resolved (sigma_4
moves by 1.5e-4 between J = 160 and 500; sigma_1..3 agree to 1e-7). From order 13 on, J = 500 is NOT enough: the nu = 3
tuning f_3^(13) is 1.1e-3 at J = 500, 7.7e-5 at 750, 1.3e-5 at 1000, and sigma_6 moves by 1e-2 then 2e-4; at J = 500
sigma_7, sigma_8 blow up (1e6, 1e11). For K >= 13 use J >= 1000 and compare two J.
Then:  python3 zero_selection.py OUTDIR    (Pade zeros of S_1 in eps^2 and the omega where the zero is real).
"""
import os
import sys
import json
import time
import multiprocessing as mpr


def one(args):
    wf, K, J, outdir, prec, inner, serial, fitkw = args
    fn = os.path.join(outdir, f"sigF_w{wf:.5f}_K{K}_J{J}.json")
    if os.path.exists(fn):
        return fn
    if hasattr(os, "nice"):
        os.nice(10)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mpmath as mp
    t0 = time.time()
    if serial:
        from amp_flint import Amp
        A = Amp(wf, K, J, prec=prec)
    else:
        from amp_flint_par import AmpPar
        A = AmpPar(wf, K, J, prec=prec, procs=inner, **fitkw)
    sig, f = A.run(verbose=False)
    tmp = fn + ".tmp"
    rec = {"wf": wf, "K": K, "J": J, "prec": prec,
           "sigma_str": [[mp.nstr(z.real, 25), mp.nstr(z.imag, 25)] for z in sig],
           "sigma": [[float(z.real), float(z.imag)] for z in sig], "f": f,
           "procs": 1 if serial else inner, "seconds": round(time.time() - t0, 1)}
    if fitkw:
        rec.update({"fit": A.fit_mode, "mf": list(A.mf), "fit_step": A.fit_step, "phys_prec": A.phys_prec,
                    "sigma_err": A.sigma_err, "fit_info": A.fit_info})
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, fn)
    return fn


def _run_one(args):
    print("done", one(args), flush=True)


def main(argv):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    inner, serial = 1, False
    if "--inner" in argv:
        i = argv.index("--inner")
        inner = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--serial" in argv:
        serial = True
        argv = [a for a in argv if a != "--serial"]
    fitkw = {}
    for flag, key, conv in (("--fit", "fit", str), ("--mf", "mf", lambda v: tuple(int(x) for x in v.split(":"))),
                            ("--fit-step", "fit_step", int), ("--phys-prec", "phys_prec", int)):
        if flag in argv:
            i = argv.index(flag)
            fitkw[key] = conv(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
    if fitkw.get("fit", "legacy") == "legacy":
        fitkw = {}
    lo, hi, n, K, J, procs = float(argv[1]), float(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), int(argv[6])
    outdir = argv[7] if len(argv) > 7 else f"scan_K{K}_J{J}"
    prec = int(argv[8]) if len(argv) > 8 else 320
    require_raw(os.path.join(outdir, "x"), allow=allow)     # raw/<task> worktree, or --allow_unraw
    os.makedirs(outdir, exist_ok=True)
    wfs = [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]
    jobs = [(wf, K, J, outdir, prec, inner, serial, fitkw) for wf in wfs]
    cx = mpr.get_context("spawn")
    if inner <= 1:
        with cx.Pool(procs) as p:
            for fn in p.imap_unordered(one, jobs):
                print("done", fn, flush=True)
    else:
        # pool workers are daemonic and may not start the inner workers, so run each omega in its own process
        running = []
        for job in jobs:
            while True:
                running = [p for p in running if p.is_alive()]
                if len(running) < procs:
                    break
                running[0].join(timeout=5)
            p = cx.Process(target=_run_one, args=(job,))
            p.start()
            running.append(p)
        for p in running:
            p.join()


if __name__ == "__main__":
    main(sys.argv)
