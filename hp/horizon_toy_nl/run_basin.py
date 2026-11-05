#!/usr/bin/env python3
"""Planted truth, controls and Newton basins of the toy's formulations (toy_nl.py).

    python3 run_basin.py FORM DPS S1,S2,... [--tests A,B] [--ue 0.1]

FORM: series (F1, ours), inward (F2), reseed (F3: F1 after b_h is recomputed from the node data), elim (F3e: b_h a
function of the node data at every residual). For each s:
  planted: Newton from the planted solution rounded to DPS digits; err = max |node - node*| (b0, y0);
  controls (must fire, i.e. err > 100 x planted err): 'gamma' (the formulation's coupling ga x (1 + 1e-6)), 'half'
  (series and Taylor steps stopped at terms below 10^-(DPS/2));
  basin B: every unknown z_i -> z_i (1 + delta r_i), |r_i| in [0.5, 1], random sign, seeds 1-3; converged = Newton's
  node error below max(100 x planted err, 10^-(DPS/2)); the largest delta on a quarter-decade grid (bisection) from
  which all three seeds converge;
  basin A: only the mass-carrying unknown off by +delta (b_h for series/inward, b0 for reseed/elim).
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
import mpmath as mp  # noqa: E402
from mpmath import mpf  # noqa: E402

import toy_nl as tn  # noqa: E402


def build(form, T, ue):
    if form == "series":
        F = tn.res_series(T)
        z = [T.ref["bh"], T.ref["b0"], T.ref["y0"]]
        node = lambda z: (z[1], z[2])  # noqa: E731
        massidx = 0
        prep = None
    elif form == "reseed":
        F = tn.res_series(T)
        z = [T.ref["bh"], T.ref["b0"], T.ref["y0"]]
        node = lambda z: (z[1], z[2])  # noqa: E731
        massidx = 1

        def prep(z):
            bh = tn.reseed(T, z[1], z[2])
            return None if bh is None else [bh, z[1], z[2]]
    elif form == "inward":
        F = tn.res_inward(T, ue)
        z = [T.ref["b0"], T.ref["y0"], T.ref["bh"]]
        node = lambda z: (z[0], z[1])  # noqa: E731
        massidx = 2
        prep = None
    elif form.startswith("pade"):
        L = int(form[4:]) if len(form) > 4 else 24
        F = tn.res_pade(T, L)
        z = [T.ref["bh"], T.ref["b0"], T.ref["y0"]]
        node = lambda z: (z[1], z[2])  # noqa: E731
        massidx = 0
        prep = None
    elif form == "elim":
        F, st = tn.res_elim(T)
        z = [T.ref["b0"], T.ref["y0"]]
        node = lambda z: (z[0], z[1])  # noqa: E731
        massidx = 0

        def prep(z, st=st):
            st.clear()
            return z
    else:
        raise SystemExit("unknown form " + form)
    return F, z, node, massidx, prep


def solve(form, T, ue, z0):
    F, zs, node, mi, prep = build(form, T, ue)
    z0 = [mpf(x) for x in z0]
    if prep is not None:
        z0 = prep(z0)
        if z0 is None:
            return None, None, []
    z, nF, hist = tn.newton(F, z0, T.dps, T=T)
    b0, y0 = node(z)
    err = max(abs(b0 - T.ref["b0"]), abs(y0 - T.ref["y0"]))
    return err, nF, hist


def near(zs, dps):
    """the planted solution moved by 10^-(3 dps/4) in every unknown (seed 99): Newton must move to pass"""
    rnd = random.Random(99)
    d = mpf(10) ** (-(3 * dps) // 4)
    return [z * (1 + d * mpf(rnd.choice((-1, 1)) * rnd.uniform(0.5, 1.0))) for z in zs]


def main():
    form, dps, ss = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    tests = "B"
    ue = "0.1"
    if "--tests" in sys.argv:
        tests = sys.argv[sys.argv.index("--tests") + 1]
    if "--ue" in sys.argv:
        ue = sys.argv[sys.argv.index("--ue") + 1]
    tag = f"{form}{'_ue' + ue if form == 'inward' else ''}_{dps}{'_inexact' if os.environ.get('INEXACT_START') else ''}"
    out = open(f"basin_{tag}.jsonl", "a")
    mp.mp.dps = dps
    for s in ss:
        t0 = time.time()
        T = tn.Toy(s, dps)
        F, zs, node, mi, prep = build(form, T, ue)
        zs = [+x for x in zs]
        zc = near(zs, dps)
        e_pl, nF_pl, h_pl = solve(form, T, ue, zc)
        rec = dict(form=form, ue=float(ue) if form == "inward" else None, dps=dps, s=float(s), D=2 * float(s) + 4,
                   planted_err=float(e_pl) if e_pl is not None else None, planted_hist=h_pl)
        print(f"[{tag}] s {s}: planted err {float(e_pl) if e_pl is not None else None:.1e} |F| hist {h_pl}", flush=True)
        ctl = {}
        for mut in (() if "--noctl" in sys.argv else ("gamma", "half")):
            Tm = tn.Toy(s, dps, mutate=mut)
            e_m, nF_m, h_m = solve(form, Tm, ue, zc)
            # fires when the mutated problem does not converge (final |F| 1000 x the planted one's) or lands
            # 100 x farther from the planted solution
            fired = (e_m is None or e_pl is None or e_m > 100 * e_pl or not h_m or not h_pl
                     or h_m[-1] > h_pl[-1] + 3)
            ctl[mut] = dict(err=float(e_m) if e_m is not None else None, fired=bool(fired))
            print(f"[{tag}] s {s}: control {mut}: err {float(e_m) if e_m is not None else None} fired {fired}", flush=True)
        rec["controls"] = ctl
        thr = max(100 * (e_pl if e_pl is not None else 1), mpf(10) ** (-dps // 2))
        rec["thr"] = float(thr)

        def conv(z0):
            e, nF, h = solve(form, T, ue, z0)
            return e is not None and e < thr, (float(e) if e is not None else None), h

        prec_fail = e_pl is None or e_pl > mpf(10) ** (-dps / 4)
        rec["precision_limited"] = bool(prec_fail)
        for test in [t for t in tests.split(",") if t in ("A", "B")]:
            if prec_fail:
                rec[f"basin_{test}"] = "n/a: the planted test fails at this precision"
                rec[f"basin_{test}_log10"] = None
                print(f"[{tag}] s {s}: basin {test}: n/a (planted test fails at this precision)", flush=True)
                continue
            log = []

            def ok_at(m):
                d = mpf(10) ** (-mpf(m) / 2)
                res = True
                if test == "B":
                    for seed in ((1,) if "--seed1" in sys.argv else (1, 2, 3)):
                        rnd = random.Random(seed)
                        r = [rnd.choice((-1, 1)) * rnd.uniform(0.5, 1.0) for _ in zs]
                        z0 = [zs[i] * (1 + d * mpf(r[i])) for i in range(len(zs))]
                        c, e, h = conv(z0)
                        log.append(dict(m=m, seed=seed, conv=c, err=e, nit=len(h) - 1))
                        print(f"[{tag}] s {s}: delta 10^-{m / 2} seed {seed}: conv {c} err {e} nit {len(h) - 1}", flush=True)
                        if not c:
                            res = False
                            break
                else:
                    z0 = list(zs)
                    z0[mi] = zs[mi] * (1 + d)
                    c, e, h = conv(z0)
                    log.append(dict(m=m, conv=c, err=e, nit=len(h) - 1))
                    res = c
                return res

            hi, lo = 1, dps - 4
            if "--mrange" in sys.argv:
                hi, lo = (int(v) for v in sys.argv[sys.argv.index("--mrange") + 1].split(","))
            if ok_at(hi):
                basin = f">= 10^-{hi / 2}"
                bm = hi
            elif not ok_at(lo):
                basin = f"< 10^-{lo / 2}"
                bm = None
            else:
                while lo - hi > 1:
                    mid = (hi + lo) // 2
                    if ok_at(mid):
                        lo = mid
                    else:
                        hi = mid
                bm = lo
                basin = f"10^-{lo / 2}"
                # monotonicity check two decades below the edge
                if lo + 4 <= dps - 4 and not ok_at(lo + 4):
                    basin += " (non-monotone: fails 2 decades below)"
            rec[f"basin_{test}"] = basin
            rec[f"basin_{test}_log10"] = (-bm / 2) if bm is not None else None
            rec[f"log_{test}"] = log
            print(f"[{tag}] s {s}: basin {test}: {basin}  ({len(log)} solves)", flush=True)
        rec["seconds"] = time.time() - t0
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{tag}] s {s}: done in {rec['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
