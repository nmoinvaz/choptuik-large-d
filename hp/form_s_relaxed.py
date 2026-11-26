"""Form S's Moebius horizon series with relaxed products (Solver.set_horizon_relaxed, #1096) against its
term-by-term sums, at saved states.

    python3 form_s_relaxed.py check OUT STATE [STATE ...] [--parts 2,4,8] [--threads 4]
    python3 form_s_relaxed.py jaccheck OUT STATE [STATE ...] [--parts 2,4,8] [--threads 4] [--cols 0,15,31]
    python3 form_s_relaxed.py time OUT STATE [--parts 4,8] [--what horizon,residual,jac] [--threads 4] [--ks KS]

check: the horizon block (Solver.horizon_s: p, m, c at the last node, 3M values) in double-, quad- and
octo-double, relaxed against term by term, and each precision's own error, term by term against the octo-double
sums. Bars, fixed before the first run, relative to max |h|: |h_relaxed - h_naive| <= 1e-28 (double-double),
1e-58 (quad-double), 1e-110 (octo-double). Amendment (after the D 2000 run, where the quad-double term-by-term
sums themselves erred by 3.7e-57 against the octo-double ones): in double- and quad-double the bar is also met when the
relaxed sums are as accurate as the term-by-term ones, |h_relaxed - h_od| <= 2 |h_naive - h_od| (the precision's
floor). The control, relaxed with one square of the product c U left out (mutate 56, form_s.rs's FS_MUT_RELAX_DROP),
must miss both. One json line per state and precision into OUT/check.jsonl; exit 1 if a bar is missed or a control
passes.

time: wall seconds, term by term and relaxed, of the horizon block (horizon_s) in each precision of --parts, the full
residual (residual_s), and the double-double Jacobian's horizon columns (jacobian_sub_s on the horizon rows and the
m_h and Delta columns: the 33 column tasks that run the horizon series with dual lanes), one json line each into
OUT/time.jsonl. --ks overrides the state's horizon order (the estimate for a larger D: the series' cost depends on ks
alone).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from form_s import make_solver_s, opts  # noqa: E402

M = 32
BARS = {2: 1e-28, 4: 1e-58, 8: 1e-110}
FS_MUT_RELAX_DROP = 56


def solver(path, thr, ks=None):
    from octo_corrector import load_o
    z8, meta = load_o(path)
    with mp.workdps(60):
        Dt = mp.mpf(meta["D_str"])
    kw = dict(kt=int(meta["kt"]), kc=int(meta["kc"]), ks=int(ks or meta["ks"]), threads=thr, segs=meta["segs"],
              mob=float(meta["hmob"]))
    s = make_solver_s(Dt, M, list(meta["nodes"]), kfreeze=3, **kw)
    if not hasattr(s, "set_horizon_relaxed") or not hasattr(s, "horizon_s"):
        raise SystemExit("this hpkernel has no set_horizon_relaxed / horizon_s (#1096); rebuild hp from wip/large-d-fastmul")
    return s, z8, meta, kw["ks"]


def mpv(parts):
    """each value of a part list (2, 4 or 8 arrays) as an mpf, exactly"""
    a = [np.asarray(p, dtype=float) for p in parts]
    return [mp.fsum(mp.mpf(float(q[i])) for q in a) for i in range(len(a[0]))]


def maxdiff(a, b):
    return float(max(abs(x - y) for x, y in zip(a, b)))


def horizon(s, z8, p, relaxed, mutate=0):
    s.set_horizon_relaxed(relaxed)
    t0 = time.time()
    h = s.horizon_s([np.ascontiguousarray(z8[q]) for q in range(p)], mutate)
    return h, time.time() - t0


def check(out, paths, parts, thr):
    mp.mp.dps = 160
    ok = True
    with open(os.path.join(out, "check.jsonl"), "a") as fh:
        for path in paths:
            s, z8, meta, ks = solver(path, thr)
            ref = None
            res = {}
            for p in sorted(parts, reverse=True):
                hn, tn = horizon(s, z8, p, 0)
                hr, tr = horizon(s, z8, p, 1)
                hc, tc = horizon(s, z8, p, 1, FS_MUT_RELAX_DROP)
                vn, vr, vc = mpv(hn), mpv(hr), mpv(hc)
                if p == 8:
                    ref = vn
                scale = float(max(abs(v) for v in vn))
                d = maxdiff(vr, vn) / scale
                dc = maxdiff(vc, vn) / scale
                own = maxdiff(vn, ref) / scale if ref is not None and p < 8 else None
                own_r = maxdiff(vr, ref) / scale if ref is not None and p < 8 else None
                own_c = maxdiff(vc, ref) / scale if ref is not None and p < 8 else None
                # the bar, or (double- and quad-double) as accurate as the term-by-term sums against octo-double
                floor_ok = lambda v: own is not None and v <= 2.0 * own  # noqa: E731
                passed = d <= BARS[p] or floor_ok(own_r)
                fired = not (dc <= BARS[p]) and not floor_ok(own_c)
                ok &= passed and fired
                rec = dict(state=os.path.basename(path), D=float(meta["D"]), ks=ks, parts=p, max_h=scale,
                           relaxed_vs_naive=d, naive_vs_od=own, relaxed_vs_od=own_r, control_vs_naive=dc,
                           control_vs_od=own_c,
                           bar=BARS[p], passed=passed, control_fired=fired, s_naive=round(tn, 2),
                           s_relaxed=round(tr, 2), s_control=round(tc, 2),
                           utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                res[p] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"D {meta['D']:g} (ks {ks}) {p} parts: |relaxed - naive| {d:.2e} (bar {BARS[p]:.0e}: "
                      f"{'PASS' if passed else 'FAIL'}), naive vs od {own if own is None else f'{own:.2e}'}, "
                      f"relaxed vs od {own_r if own_r is None else f'{own_r:.2e}'}, control {dc:.2e} "
                      f"({'fired' if fired else 'DID NOT FIRE'}); {tn:.1f} s naive, {tr:.1f} s relaxed", flush=True)
    return ok


def jaccheck(out, paths, parts, thr, pick=None):
    """The Jacobian's horizon columns (jacobian_sub_s on the horizon rows and the m_h and Delta columns, the 33 column
    tasks that run the horizon series in Du, Qu or Ou lanes) in double-, quad- and octo-double, relaxed against term
    by term, with the bars of check (amendment included) relative to max |J| of those columns, and the control
    (mutate 56) that must miss them; pick (--cols): only those m_h points' columns and Delta's. One json line per state
    and precision into OUT/jaccheck.jsonl."""
    mp.mp.dps = 160
    ok = True
    with open(os.path.join(out, "jaccheck.jsonl"), "a") as fh:
        for path in paths:
            s, z8, meta, ks = solver(path, thr)
            L = s.size_s
            K = len(meta["nodes"])
            rows = list(range(K * 3 * M, (K + 1) * 3 * M))
            cols = list(range(2 * M + 3 * M * K, 2 * M + 3 * M * K + M)) + [L - 1]
            if pick:
                # --cols: those m_h points' columns and Delta's (each one column task)
                cols = [2 * M + 3 * M * K + j for j in pick] + [L - 1]
            ref = None
            for p in sorted(parts, reverse=True):
                zp = [np.ascontiguousarray(z8[q]) for q in range(p)]
                out_v = {}
                secs = {}
                for tag, mode, mut in (("naive", 0, 0), ("relaxed", 1, 0), ("control", 1, FS_MUT_RELAX_DROP)):
                    s.set_horizon_relaxed(mode)
                    t0 = time.time()
                    J = s.jacobian_sub_s(zp, rows, cols, mut)
                    secs[tag] = round(time.time() - t0, 2)
                    out_v[tag] = mpv([np.asarray(q).ravel() for q in J])
                vn, vr, vc = out_v["naive"], out_v["relaxed"], out_v["control"]
                if p == 8:
                    ref = vn
                scale = float(max(abs(v) for v in vn))
                d = maxdiff(vr, vn) / scale
                dc = maxdiff(vc, vn) / scale
                own = maxdiff(vn, ref) / scale if ref is not None and p < 8 else None
                own_r = maxdiff(vr, ref) / scale if ref is not None and p < 8 else None
                own_c = maxdiff(vc, ref) / scale if ref is not None and p < 8 else None
                floor_ok = lambda v: own is not None and v <= 2.0 * own  # noqa: E731
                passed = d <= BARS[p] or floor_ok(own_r)
                fired = not (dc <= BARS[p]) and not floor_ok(own_c)
                ok &= passed and fired
                rec = dict(state=os.path.basename(path), D=float(meta["D"]), ks=ks, parts=p, cols=len(cols), max_J=scale,
                           relaxed_vs_naive=d, naive_vs_od=own, relaxed_vs_od=own_r, control_vs_naive=dc,
                           control_vs_od=own_c, bar=BARS[p], passed=passed, control_fired=fired, s_naive=secs["naive"],
                           s_relaxed=secs["relaxed"], s_control=secs["control"], threads=thr,
                           utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"D {meta['D']:g} (ks {ks}) {p} parts, J's horizon columns: |relaxed - naive| {d:.2e} "
                      f"({'PASS' if passed else 'FAIL'}), naive vs od {own if own is None else f'{own:.2e}'}, relaxed vs od "
                      f"{own_r if own_r is None else f'{own_r:.2e}'}, control {dc:.2e} ({'fired' if fired else 'DID NOT FIRE'}); "
                      f"{secs['naive']:.1f} s naive, {secs['relaxed']:.1f} s relaxed", flush=True)
    return ok


def timing(out, path, parts, what, thr, ks):
    s, z8, meta, ks = solver(path, thr, ks)
    L = s.size_s
    K = len(meta["nodes"])
    rows = list(range(K * 3 * M, (K + 1) * 3 * M))
    cols = list(range(2 * M + 3 * M * K, 2 * M + 3 * M * K + M)) + [L - 1]
    with open(os.path.join(out, "time.jsonl"), "a") as fh:
        def rec(kind, p, mode, sec, extra=None):
            r = dict(state=os.path.basename(path), D=float(meta["D"]), ks=ks, kind=kind, parts=p,
                     relaxed=mode, seconds=round(sec, 2), threads=thr,
                     utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            if extra:
                r.update(extra)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"D {meta['D']:g} ks {ks} {kind} {p} parts relaxed {mode}: {sec:.1f} s", flush=True)
        for p in parts:
            zp = [np.ascontiguousarray(z8[q]) for q in range(p)]
            for mode in (1, 0):
                s.set_horizon_relaxed(mode)
                if "horizon" in what:
                    t0 = time.time()
                    s.horizon_s(zp)
                    rec("horizon", p, mode, time.time() - t0)
                if "residual" in what:
                    t0 = time.time()
                    s.residual_s(zp)
                    rec("residual", p, mode, time.time() - t0)
        if "jac" in what:
            zp = [np.ascontiguousarray(z8[q]) for q in range(2)]
            for mode in (1, 0):
                s.set_horizon_relaxed(mode)
                t0 = time.time()
                J = s.jacobian_sub_s(zp, rows, cols, 0)
                rec("jac_horizon", 2, mode, time.time() - t0, dict(cols=len(cols)))


def main(argv):
    cmd, out = argv[0], argv[1]
    pos, o = opts(argv[2:], dict(parts="2,4,8", threads="4", what="horizon,residual,jac", ks="", cols=""))
    os.makedirs(out, exist_ok=True)
    parts = [int(p) for p in o["parts"].split(",")]
    thr = int(o["threads"])
    import hpkernel
    print(f"hpkernel {hpkernel.__file__}", flush=True)
    if cmd == "check":
        ok = check(out, pos, parts, thr)
        print("check: " + ("PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 1
    if cmd == "jaccheck":
        ok = jaccheck(out, pos, parts, thr, [int(j) for j in o["cols"].split(",")] if o["cols"] else None)
        print("jaccheck: " + ("PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 1
    if cmd == "time":
        timing(out, pos[0], parts, o["what"].split(","), thr, int(o["ks"]) if o["ks"] else None)
        return 0
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
