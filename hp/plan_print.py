#!/usr/bin/env python3
"""Issue #396: hp's computed node and segment plan at a given M, printed to the bit, so that two machines' plans can be
diffed entry by entry, and each planned Taylor step tested against its truncation target.

    cd hp && python3 plan_print.py STATE.npz --M=48 [--kf=none|13] [--threads=N] [--json=OUT]

The state is resampled to M tau points and its nodes fitted as hp/modecount_common.prepare does (fit_nodes), in a
fresh cache directory, so nothing an earlier run saved on this machine is read. It is then re-converged by Newton (as
setup does, without the complex solver), and planned by Solver.plan(z, 1e-32, 0.05), the plan every computed layout
uses. Printed and stored: the nodes, ks, and every segment's x points as hex floats, the plan's sha256 over those
bytes, and the solution's |F|.

Soundness: each planned step is taken once and as two half steps from the node state at the segment's start
(Solver.integrate, double-double), and their relative difference is the step's truncation, to be compared with the
target 1e-32 (it meets it when every step is within 100 times it). The end of each segment is compared with the next
node's state (the layout check: it must agree to about |F|).

Control, every run: the plan is made again with the tolerance 64 ulps above 1e-32 (a step goes like tol^(1/k), so
that moves a step by about two ulps, the size of a libm pow difference), and its hash must differ from the plain
plan's (so the printout resolves the planner's last bits). Exit 1 when the layout check fails or the control
does not change the plan.
"""
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOL = 1e-32
HMAX = 0.05


def plan_hash(segs):
    return hashlib.sha256(np.array([x for sg in segs for x in sg], dtype="<f8").tobytes()).hexdigest()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if not args or "M" not in opts:
        sys.exit(__doc__)
    import modecount_common as mc
    from hpkernel.driver import make_solver, newton_solve
    path, M = args[0], int(opts["M"])
    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    threads = int(opts.get("threads", 1))
    t0 = time.time()
    mc.DATA = tempfile.mkdtemp(prefix="plan_print_")          # a fresh cache: no earlier state of this machine is read
    zh, zl, meta, kf, _ = mc.prepare(path, M, threads, kf, print, True, None)
    s = make_solver(meta["D"], M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 420),
                    ks=meta.get("ks", 100), threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, TOL, HMAX))
    F = mc.resid(s, zh, zl)
    if F > 1e-24:
        zh, zl, _ = newton_solve(s, zh, zl, tol=1e-26, it_max=12, verbose=False)
        F = mc.resid(s, zh, zl)
    tol = TOL
    segs = s.plan(zh, zl, tol, HMAX)
    # a step goes like tol^(1/k), k about kt = 36, so 64 ulps of tol move a step by about two ulps, the size of a libm
    # pow difference in the step itself
    h_ulp = plan_hash(s.plan(zh, zl, TOL * (1 + 2.0 ** -46), HMAX))
    s.set_segs(segs)
    F = mc.resid(s, zh, zl)
    h = plan_hash(segs)
    nodes = list(meta["nodes"])
    print(f"{os.path.basename(path)}: D {meta['D']}, M {M}, kfreeze {kf}, ks {meta.get('ks')}, |F| {F:.2e}, "
          f"plan tol {tol!r} ({time.time() - t0:.0f} s)")
    print(f"nodes {[float(x).hex() for x in nodes]}")
    for i, sg in enumerate(segs):
        print(f"segment {i}: {len(sg) - 1} steps, x {' '.join(float(x).hex() for x in sg)}")
    control_ok = h_ulp != h
    print(f"plan sha256 {h}; at tol 64 ulps up {h_ulp[:16]}...: control {'pass (differs)' if control_ok else 'FAIL (same)'}")
    # soundness: every step once and as two halves, from the node state at the segment's start
    K = len(nodes)
    blk = lambda i: (np.ascontiguousarray(zh[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)]),
                     np.ascontiguousarray(zl[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)]))
    worst, rows, layout_ok = 0.0, [], True
    for i, sg in enumerate(segs):
        yh, yl = blk(i)
        seg_worst = 0.0
        for a, b in zip(sg[:-1], sg[1:]):
            m = a + 0.5 * (b - a)
            fh, fl = s.integrate(zh, zl, yh, yl, [a, b])
            hh, hl = s.integrate(zh, zl, yh, yl, [a, m, b])
            fh, fl, hh, hl = (np.asarray(v) for v in (fh, fl, hh, hl))
            d = float(np.max(np.abs((fh - hh) + (fl - hl)))) / float(np.max(np.abs(hh)))
            seg_worst = max(seg_worst, d)
            yh, yl = np.ascontiguousarray(hh), np.ascontiguousarray(hl)
        worst = max(worst, seg_worst)
        line = dict(segment=i, steps=len(sg) - 1, worst_step=seg_worst)
        if i + 1 < K:
            nh, nl = blk(i + 1)
            gap = float(np.max(np.abs((yh - nh) + (yl - nl)))) / float(np.max(np.abs(nh)))
            line["gap_to_next_node"] = gap
            layout_ok = layout_ok and gap <= max(1e3 * F, 1e-20)
        rows.append(line)
        print(f"  segment {i}: largest step truncation {seg_worst:.1e}"
              + (f", end against node {i + 1} {line['gap_to_next_node']:.1e}" if "gap_to_next_node" in line else ""))
    meets = worst <= 100 * TOL
    print(f"truncation: largest {worst:.1e} against the target {TOL:g}: {'meets it' if meets else 'DOES NOT meet it'}"
          f" (within 100 times); layout check {'pass' if layout_ok else 'FAIL'}")
    rec = dict(state=path, D=meta["D"], M=M, kfreeze=kf, ks=meta.get("ks"), F=F, tol=tol, nodes=[x.hex() for x in map(float, nodes)],
               segs=[[float(x).hex() for x in sg] for sg in segs], sha256=h, segments=rows, worst_step=worst,
               meets_target=meets, layout_ok=layout_ok, sha256_tol_ulp=h_ulp, control_ok=control_ok)
    if opts.get("json"):
        with open(opts["json"], "w") as fh_:
            json.dump(rec, fh_, indent=1)
    return 0 if layout_ok and control_ok else 1


if __name__ == "__main__":
    sys.exit(main())
