#!/usr/bin/env python3
"""Rounding or the model: the chain's quad-double series at one base in the two arithmetic forms of the horizon series
(make_solver's horizon_alpha 0, the chain's, and 1: the same model, different rounding; notes/QUAD_DOUBLE.md 13.19 used
the same test on the double-double jets at D 82.16, where the high orders differed by their own size).

At the base (a four-part state of anm_run.py) the series z_1..z_N is built as anm_run.py --jet_q 1 builds it
(coefficients_q: the quad-double jet through the quad-double LU, the held harmonics on the trend of --trend relative to
this base) in both forms, on one Taylor plan. Printed and recorded per order k: |z_k| (form 0) and
d_k = |z_k(0) - z_k(1)| / |z_k(0)|, max norms over the unknowns.

Criteria (written before the first run, 2026-11-05 17:55 UTC):
- the orders k are the model's where d_k <= 1e-6, rounding where d_k >= 0.1, undetermined between;
- the control: the double-double series (anm_run.py coefficients, one refinement) in the two forms must give d_k >= 0.1
  at some order k <= 12 at the same base, as at 82.16. If it does not, the test cannot see rounding here and exits 1.

    python3 series_forms_check.py RAWDIR OUT.jsonl --state anm-mac/elimq_pilot/anm_pin3_D131.0559.npz [--trend T75]
        [--threads 4] [--N 24] [--kfreeze 3]

OUT.jsonl must lie in a task folder of the raw-data repository (hp/raw_out.py). Exit 1 when the control does not fire.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from anm_run import coefficients, coefficients_q  # noqa: E402
from hpkernel.driver import load_q, make_solver  # noqa: E402
from multidouble import dd_add, dd_norm  # noqa: E402
from pinning import band_coeffs, band_field  # noqa: E402
from raw_out import require_raw  # noqa: E402

mp.mp.dps = 50


def binom(sx, j):
    r = 1.0
    for t in range(j):
        r *= (sx - t) / (t + 1)
    return r


def main():
    args = list(sys.argv[1:])
    opt = dict(state="anm-mac/elimq_pilot/anm_pin3_D131.0559.npz", trend="T75", threads=4, N=24, kfreeze=3)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); v = args[i + 1]; del args[i:i + 2]
            opt[k] = type(opt[k])(v)
    raw, out = args[0], args[1]
    require_raw(out)
    zq, m = load_q(os.path.join(raw, opt["state"]))
    zq = tuple(np.ascontiguousarray(np.asarray(p)) for p in zq)
    zh, zl = zq[0].copy(), zq[1].copy()
    D, M, N, kf = mp.mpf(m["D_str"]), int(m["M"]), opt["N"], opt["kfreeze"]
    nodes, kt, kc, ks = list(m["nodes"]), m["kt"], m["kc"], m["ks"]
    K = len(nodes); rows = slice(K * 4 * M, K * 4 * M + M)
    # the held harmonics' trend relative to this base, as anm_run.py re-bases it at each new base
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    base = band_coeffs(zh[:M], zl[:M], M, ks_held)
    slopes = [[0.0, 0.0] for _ in ks_held]
    ns, vals = [], []
    for f in sorted(glob.glob(os.path.join(raw, opt["trend"], "pin*_D*.npz")))[::4]:
        dd = np.load(f); mm = json.loads(str(dd["meta"]))
        if int(mm["M"]) == M:
            ns.append(float(mm["D"]) - 2.0); vals.append(band_coeffs(dd["zh"][:M], dd["zl"][:M], M, ks_held))
    ns = np.array(ns)
    for i in range(len(ks_held)):
        for j in (0, 1):
            v = np.array([c[i][j] for c in vals])
            if len(v) >= 6 and (np.all(v > 0) or np.all(v < 0)):
                sl, c0 = np.polyfit(np.log(ns), np.log(np.abs(v)), 1)
                if float(np.max(np.abs(np.log(np.abs(v)) - (c0 + sl * np.log(ns))))) < 0.05:
                    slopes[i][j] = float(sl)
    n0 = float(D) - 2.0
    held_jet = lambda j: [(a * binom(sa, j) / n0 ** j, b * binom(sb, j) / n0 ** j)  # noqa: E731
                          for (a, b), (sa, sb) in zip(base, slopes)]
    held = (lambda j: band_field(held_jet(j), ks_held, M), rows) if any(any(s) for s in slopes) else None
    print(f"base {opt['state']}: D {mp.nstr(D, 12)}, M {M}, N {N}, trend {opt['trend']}, held {ks_held}", flush=True)
    segs = None
    series = {}
    for prec in ("qd", "dd"):
        for alpha in (0, 1):
            t0 = time.time()
            s = make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=opt["threads"], kfreeze=kf,
                            horizon_alpha=alpha)
            if segs is None:
                segs = s.plan(zh, zl, 1e-32, 0.05)
            s.set_segs(segs)
            if prec == "qd":
                s.newton_q(*zq)
                Zq, norms = coefficients_q(s, zq, N, held=held)
                Z = [(np.asarray(z[0]), np.asarray(z[1])) for z in Zq]
            else:
                s.newton(zh, zl)
                Zh, Zl, norms, _ = coefficients(s, zh, zl, N, held=held, refine_it=1)
                Z = [(np.asarray(a), np.asarray(b)) for a, b in zip(Zh, Zl)]
            series[(prec, alpha)] = (Z, norms)
            print(f"  {prec}, horizon_alpha {alpha}: {time.time() - t0:.0f} s, |z_k| k = 1..{N}: "
                  + " ".join(f"{v:.2e}" for v in norms[1:]), flush=True)
    rec = dict(state=opt["state"], D=float(D), D_str=m["D_str"], N=N, trend=opt["trend"])
    for prec in ("qd", "dd"):
        (Z0, n0s), (Z1, _) = series[(prec, 0)], series[(prec, 1)]
        d = []
        for k in range(1, N + 1):
            dh, dl = dd_add(Z0[k][0], Z0[k][1], -Z1[k][0], -Z1[k][1])
            d.append(float(dd_norm(np.asarray(dh), np.asarray(dl))) / n0s[k] if n0s[k] else float("nan"))
        rec[prec] = dict(norms=n0s, d=d)
        verdict = ["model" if v <= 1e-6 else "rounding" if v >= 0.1 else "?" for v in d]
        print(f"{prec}: d_k, k = 1..{N}: " + " ".join(f"{v:.1e}" for v in d), flush=True)
        print(f"{prec}: reading per order: " + " ".join(f"{k}:{r}" for k, r in enumerate(verdict, 1)), flush=True)
    fired = any(v >= 0.1 for v in rec["dd"]["d"][:12])
    rec["control_fired"] = fired
    print(f"control (double-double series, d_k >= 0.1 at some k <= 12): {'fired' if fired else 'DID NOT FIRE'}", flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    sys.exit(0 if fired else 1)


if __name__ == "__main__":
    main()
