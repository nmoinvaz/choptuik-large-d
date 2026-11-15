#!/usr/bin/env python3
"""Is Newton's Delta step at a converged state determined at octo-double precision? Its spread under perturbations of
one block at a time by a relative size (#827).

    cd hp && python3 step_spread.py RAW STATE [--threads 3] [--rel 1e-65] [--seeds 1,2,3,4] [--ref D] [--json OUT]
    cd hp && python3 step_spread.py --selftest [--mutate=bar]

The system is proj_shift.py's (octo_corrector.py planted's at --rel 0: pinned_o at the state, its plan and held values;
no projection). The plain pinned Newton step (PinnedO.newton_o) is taken at the unperturbed state, at the horizon data
(m_h and a_h) perturbed by rel with random signs, z_i -> z_i (1 + rel s_i), s_i = +-1, for each seed, and at each
other block (reseed.offsets' centre, nodes, Delta) perturbed alone by the same size with the first seed, the centre
being the issue's control. Printed per run: the perturbation's largest move, the step's Delta part and the smallest
pivot.

Checks (exit 1 on failure): with --ref, the unperturbed step's Delta part equals it exactly (as a double, the sum of its
parts). The verdict (not a check): "determined" if every seed's Delta step is within 1e-3 relative of the unperturbed
one, else "not determined", with the spread over seeds. --selftest runs the verdict's rule on planted steps; its
control --mutate=bar (the bar at 1e3 instead of 1e-3) must fail."""
import json
import os
import sys
import time

import numpy as np

BAR = 1e-3


def verdict(ref, steps, bar=BAR):
    """'determined' if every step is within bar relative of ref, else 'not determined'; and the largest relative
    deviation"""
    dev = max(abs(x / ref - 1) for x in steps)
    return ("determined" if dev <= bar else "not determined"), dev


def selftest(mutate):
    bar = 1e3 if mutate == "bar" else BAR
    ref = 3.3110848587422764e-8
    cases = [([ref, ref * (1 + 1e-4), ref * (1 - 9e-4), ref], "determined"),
             ([ref, ref * 332.5, ref, ref], "not determined"),
             ([ref * (1 + 2e-3)] * 4, "not determined"),
             ([-ref, ref, ref, ref], "not determined")]
    ok = True
    for steps, want in cases:
        got, dev = verdict(ref, steps, bar)
        ok &= got == want
        print(f"  steps / ref {', '.join(f'{x / ref:.4g}' for x in steps)}: {got}, largest deviation {dev:.3g} "
              f"(want {want}){'' if got == want else '  <- wrong'}")
    print(f"step_spread selftest{' --mutate=' + mutate if mutate else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv):
    if argv and argv[0] == "--selftest":
        return selftest(next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--mutate=")), ""))
    import mpmath as mp
    from hpkernel.driver import load_q
    from multidouble import od_axpy
    from octo_corrector import o8, o8_of_q, omax, pinned_o, plant_start
    from proj_shift import blocks
    from reseed import offsets
    mp.mp.dps = 150
    raw, path = argv[0], argv[1]
    opt = dict(threads="3", rel="1e-65", seeds="1,2,3,4", ref="", json="")
    for k in list(opt):
        if "--" + k in argv:
            opt[k] = argv[argv.index("--" + k) + 1]
    unknown = [a for a in argv[2:] if a.startswith("--") and a[2:] not in opt]
    if unknown:
        raise SystemExit(f"step_spread.py: unknown options {unknown}")
    rel, seeds = float(opt["rel"]), [int(x) for x in opt["seeds"].split(",")]
    t0 = time.time()
    z4, meta = load_q(os.path.join(raw, path))
    z4 = o8(z4)
    s, sp, sq, so = pinned_o(meta, z4, threads=int(opt["threads"]))
    M = int(meta["M"])
    K = (len(z4[0]) - 1) // (4 * M) - 1               # reseed.offsets: L = 4 M (K + 1) + 1
    if offsets(M, K)["L"] != len(z4[0]):
        raise SystemExit(f"step_spread.py: no layout at M {M} has the state's length {len(z4[0])}")

    class RS:           # plant_start reads the layout only
        off = offsets(M, K)
    RS.K = K
    zs, _ = plant_start(z4, RS, 0.0, 1, 0)
    z = o8_of_q(zs)                                   # planted --rel 0's start, as proj_shift takes it
    L = len(z[0])
    bl = {name: (lo, hi) for name, lo, hi in blocks(M, K)}
    print(f"step_spread {path}: D {mp.nstr(mp.mpf(meta['D_str']), 15)}, M {M}, K {K}, L {L}; rel {rel:g}, seeds "
          f"{seeds}; pinned max|F| (octo) {omax(so.residual_o(z)):.4e} ({time.time() - t0:.0f} s)", flush=True)
    zero8 = [np.zeros(L)] * 7

    def run(name, which, seed):
        """the plain Newton step at z with the blocks in which perturbed by rel (seed's signs); None: unperturbed"""
        dz = np.zeros(L)
        if which:
            rng = np.random.default_rng(seed)
            sg = rng.choice([-1.0, 1.0], L)
            for b in which:
                lo, hi = bl[b]
                dz[lo:hi] = rel * sg[lo:hi] * (np.asarray(z[0][lo:hi]) + np.asarray(z[1][lo:hi]))
        zp = o8(od_axpy(1.0, np.ascontiguousarray(dz), *zero8, *z)) if which else z
        t1 = time.time()
        st = so.newton_o(zp)
        d = o8(st[0])
        dD = float(sum((mp.mpf(float(p[-1])) for p in d), mp.mpf(0)))
        r = dict(run=name, blocks=list(which or []), seed=seed if which else None, move=float(np.abs(dz).max()),
                 dDelta=dD, dDelta_parts=[float(p[-1]) for p in d], dz=float(np.abs(d[0]).max()),
                 piv=[st[3], st[4]], F=omax(st[1:3]), s=round(time.time() - t1, 1))
        print(f"  {name:22s} move {r['move']:.3e}: dDelta {dD:+.10e}, max|dz| {r['dz']:.4e}, smallest pivot "
              f"{st[3]:.4e}, max|F| {r['F']:.4e} ({r['s']:.0f} s)", flush=True)
        return r

    runs = [run("unperturbed", None, None)]
    for sd in seeds:
        runs.append(run(f"m_h, a_h seed {sd}", ("m_h", "a_h"), sd))
    for b in [n for n in bl if n not in ("m_h", "a_h")]:
        runs.append(run(f"{b} seed {seeds[0]}" + (" (control)" if b == "centre" else ""), (b,), seeds[0]))
    ref0 = runs[0]["dDelta"]
    ok = True
    rec = dict(state=path, D_str=meta["D_str"], rel=rel, seeds=seeds, runs=runs)
    if opt["ref"]:
        good = ref0 == float(opt["ref"])
        ok &= good
        rec["ref"] = dict(value=float(opt["ref"]), ok=good)
        print(f"  check: the unperturbed dDelta {ref0!r} against --ref {float(opt['ref'])!r}: "
              f"{'equal' if good else 'NOT equal'}: {'pass' if good else 'FAIL'}", flush=True)
    hz = [r["dDelta"] for r in runs if r["blocks"] == ["m_h", "a_h"]]
    v, dev = verdict(ref0, hz)
    rec.update(verdict=v, deviation=dev, spread=[min(hz), max(hz)])
    print(f"  verdict: {v}; the horizon seeds' dDelta {min(hz):+.6e} .. {max(hz):+.6e}, ratio to the unperturbed "
          f"{min(h / ref0 for h in hz):.4g} .. {max(h / ref0 for h in hz):.4g}, largest relative deviation {dev:.3g} "
          f"(bar {BAR:g})", flush=True)
    for r in runs[1 + len(seeds):]:
        print(f"  {r['run']}: dDelta / unperturbed {r['dDelta'] / ref0:.6g}", flush=True)
    rec["passed"] = bool(ok)
    if opt["json"]:
        with open(opt["json"], "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
    print(f"step_spread: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
