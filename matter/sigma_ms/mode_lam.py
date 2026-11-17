#!/usr/bin/env python3
"""Growth rates of the sigma-model's DSS solution on the multishoot kernel (#918): the real roots of det M(lambda) at
saved states, M the mode matrix (mode_check.py checks it).

    PYTHONPATH=<site> python3 matter/sigma_ms/mode_lam.py STATE.npz [STATE.npz ...] --out DIR --guess LAM
        [--threads 2] [--widths 1e-3,1e-2,5e-2] [--scan A,B,N] [--no_gauge] [--no_phys] [--xtol 1e-13] [--gate 31]
        [--prev D1:LAM1,D2:LAM2] [--gauge_window 1e-12]

For each state, in the order given:
  the gauge rate lambda_g = mean over tau of 1 - c_x(tau, 1) (exact for the mode that relabels the light cone) and
      the root of det M nearest it, which measures the mode matrix's error at this state (gauge_diff: the larger
      distance from lambda_g of the two rates that enclose the sign change);
  the root nearest the guess: the growth rate lambda of the unstable mode (gamma = 1/lambda). --guess serves the first
      state. Each later one takes the root of the state before it, or the line through the two before it in D.
  with --scan A,B,N the sign and ln|det M| at the N + 1 rates from A to B, and the number of sign changes.
A root is the sign change of det M, from the double-double LU, located by Brent's method (smdriver.det_root). A
Newton iteration on (mode, lambda) is not used: the Jacobian's near-null band (SIGMA_DSS.md section 6) is in M at
every lambda, and its small singular values do not change sign.

--gate L waits before each state until the machine's 1-minute load is below L. --prev gives the rates of states
done before, so that the first state's guess is their line in D as well (a chain continued in a later run).
--gauge_window W first takes det M at lambda_g - W and lambda_g + W only: opposite signs enclose the gauge root within
W of lambda_g with two determinants, and equal signs fall back to the search of --gauge_widths.

Appends one record per state to DIR/sm_lam.jsonl and logs to DIR/sm_lam_<tag>.log. Exit status 1 when a state's
residual exceeds --fmax (1e-26), or a root is not bracketed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smdriver as sd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--guess", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--widths", default="1e-3,1e-2,5e-2")
    ap.add_argument("--gauge_widths", default="1e-3,1e-2")
    ap.add_argument("--scan", default="")
    ap.add_argument("--no_gauge", action="store_true")
    ap.add_argument("--no_phys", action="store_true")
    ap.add_argument("--xtol", type=float, default=1e-13)
    ap.add_argument("--fmax", type=float, default=1e-26)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--prev", default="", help="D:lambda pairs of states done before, in the order of the chain")
    ap.add_argument("--gauge_window", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    widths = tuple(float(x) for x in a.widths.split(","))
    gwidths = tuple(float(x) for x in a.gauge_widths.split(","))
    # (D, lambda) of the states done
    prev = [tuple(float(v) for v in x.split(":")) for x in a.prev.split(",") if x]
    bad = False
    for path in a.states:
        z, meta = sd.load(path)
        tag = str(meta.get("tag", os.path.basename(path)))
        logf = open(os.path.join(a.out, f"sm_lam_{tag}.log"), "a")

        def log(msg):
            line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        while a.gate and os.getloadavg()[0] >= a.gate:
            log(f"waiting: load {os.getloadavg()[0]:.2f}")
            time.sleep(60)
        t0 = time.time()
        zh, zl = np.ascontiguousarray(z[0], float), np.ascontiguousarray(z[1], float)
        s = sd.solver_of(meta, a.threads)
        D = float(meta.get("D", 4.0))
        F = sd.fnorm(s.residual([zh, zl]))
        lg = float(sd.gauge_rate(zh, zl, meta))
        log(f"start {sd.now()}: {os.path.basename(path)}, eta {meta['eta']}, D {D}, M {meta['M']}, |F| {F:.2e}, "
            f"Delta {zh[-1]:.15f}, lambda_g {lg:.15f}, threads {a.threads}, load {os.getloadavg()[0]:.2f}")
        rec = dict(tag=tag, state=os.path.basename(path), eta=meta["eta"], D=D, M=meta["M"], K=len(meta["nodes"]),
                   kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], F=F, Delta_hi=float(zh[-1]), Delta_lo=float(zl[-1]),
                   lam_g=lg, threads=a.threads, xtol=a.xtol, when=sd.now())
        if F > a.fmax:
            log(f"  |F| above {a.fmax:g}: skipped")
            rec["skipped"] = "residual"
            bad = True
        else:
            cache = {}
            fn = sd.det_fn(s, zh, zl, log, cache)
            if not a.no_gauge:
                w = a.gauge_window
                if w and fn(lg - w)[0] != fn(lg + w)[0]:
                    root, br = lg, (lg - w, lg + w)
                else:
                    root, w = sd.det_root_near(fn, lg, gwidths, a.xtol)
                    br = None if root is None else sd.sign_bracket(cache, root)
                rec.update(gauge_root=root, gauge_sign_change=br, gauge_bracket=w,
                           gauge_diff=None if br is None else max(abs(br[0] - lg), abs(br[1] - lg)))
                if br is None:
                    log(f"  gauge: no sign change within lambda_g (1 -+ {w:g})")
                    bad = True
                else:
                    log(f"  gauge: det M changes sign between lambda_g {br[0] - lg:+.3e} and lambda_g "
                        f"{br[1] - lg:+.3e}")
            if not a.no_phys:
                if len(prev) >= 2 and prev[-1][0] != prev[-2][0]:
                    (D1, l1), (D2, l2) = prev[-2], prev[-1]
                    guess = l2 + (l2 - l1) * (D - D2) / (D2 - D1)
                elif prev:
                    guess = prev[-1][1]
                else:
                    guess = a.guess
                if not guess > 0:
                    raise SystemExit("--guess is needed for the first state")
                n0 = len(cache)
                root, w = sd.det_root_near(fn, guess, widths, a.xtol)
                rec.update(lam=root, guess=guess, bracket=w, evals=len(cache) - n0,
                           lam_sign_change=None if root is None else sd.sign_bracket(cache, root))
                if root is None:
                    log(f"  lambda: no sign change within {guess:.9f} (1 -+ {w:g})")
                    bad = True
                else:
                    prev.append((D, root))
                    log(f"  lambda {root:.15f} (gamma {1 / root:.15f}), guess {guess:.9f}, bracket {w:g}, "
                        f"{len(cache) - n0} determinants")
            if a.scan:
                lo, hi, npt = a.scan.split(",")
                grid = np.linspace(float(lo), float(hi), int(npt) + 1)
                sc = [[float(x), *fn(x)] for x in grid]
                ch = [(p[0], q[0]) for p, q in zip(sc, sc[1:]) if p[1] != q[1]]
                rec.update(scan=sc, scan_changes=ch)
                log(f"  scan {lo}..{hi}: {len(sc)} rates, {len(ch)} sign changes, in {ch}")
            rec["det"] = sorted((k, sg, la) for k, (sg, la) in cache.items())
        rec["t"] = round(time.time() - t0, 1)
        with open(os.path.join(a.out, "sm_lam.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"done {sd.now()} ({rec['t']} s)")
        logf.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
