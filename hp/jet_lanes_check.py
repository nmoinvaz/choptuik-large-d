"""The octo-double series' jets at the lane counts of Solver.residual_jet_o: the jets of a stored series (anm_run.py
--save_jet 1, eight parts) recomputed call by call as coefficients_o makes them (z_0..z_{k-1}, nout k + 1, k = 1..N),
once with the default lanes (the smallest of 8, 12, ..., 32 that holds nout) and once with the counts the kernel took
before (8 up to nout 8, 24 up to 24, then 32).

    python3 jet_lanes_check.py RAW SERIES.npz STATE.npz [--threads 6] [--mutate 0] [--out PREFIX]

Pass, fixed before the first run: every output order of every call is the same bits, all eight parts, with both lane
choices. Reported: each call's seconds with each choice, their sums and the ratio. The two choices alternate which goes
first from call to call, so both see the same load. Control: --mutate 1 moves z_{k-1}'s leading part by one unit in
the last place at its largest entry in the old choice's input, which must differ (exit 1). STATE gives the solver: the
chain's own system and stored plan (octo_corrector.truth_system). Exit 1 on failure."""
import json
import sys
import time

import numpy as np

LANES = [8, 12, 16, 20, 24, 28, 32]


def lanes_new(need):
    return next(n for n in LANES if n >= need)


def lanes_old(need):
    return 8 if need <= 8 else (24 if need <= 24 else 32)


def same_bits(a, b):
    """the number of output orders whose eight parts differ in any bit"""
    return sum(any(not np.array_equal(np.asarray(p).view(np.uint64), np.asarray(r).view(np.uint64)) for p, r in zip(fa, fb))
               for fa, fb in zip(a, b))


def main(argv):
    from jet_singularity import load_jet
    from octo_corrector import truth_system
    raw, series, state = argv[0], argv[1], argv[2]
    opt = dict(threads=6, mutate=0, out="")
    for k in list(opt):
        if "--" + k in argv:
            v = argv[argv.index("--" + k) + 1]
            opt[k] = type(opt[k])(v) if opt[k] != "" else v
    import os
    Zo, meta = load_jet(os.path.join(raw, series))
    if meta["parts"] != 8:
        raise SystemExit(f"{series}: {meta['parts']} parts, not eight (anm_run.py --jet_o 1)")
    _, smeta, (s, _, _, _), info = truth_system(raw, state, threads=opt["threads"])
    N = len(Zo) - 1
    print(f"jet_lanes_check {series}: N {N}, D {meta['D_str'][:16]}, series of {meta.get('state')}; solver from {state} "
          f"({info['plan']} plan), {opt['threads']} threads, mutate {opt['mutate']}", flush=True)
    recs, bad, t_new, t_old = [], 0, 0.0, 0.0
    for k in range(1, N + 1):
        zs = [list(z) for z in Zo[:k]]
        zs_old = zs
        if opt["mutate"]:
            last = [np.array(p) for p in zs[-1]]
            j = int(np.argmax(np.abs(last[0])))
            last[0][j] = np.nextafter(last[0][j], np.inf)
            zs_old = zs[:-1] + [last]
        nout = k + 1
        ln, lo = lanes_new(nout), lanes_old(nout)
        runs = {}
        for which in (("new", "old") if k % 2 else ("old", "new")):
            t0 = time.time()
            if which == "new":
                F = s.residual_jet_o(zs, nout)
            else:
                F = s.residual_jet_o(zs_old, nout, lanes=lo)
            runs[which] = (F, time.time() - t0)
        nd = same_bits(runs["new"][0], runs["old"][0])
        bad += nd > 0
        t_new += runs["new"][1]
        t_old += runs["old"][1]
        rec = dict(k=k, nout=nout, lanes_new=ln, lanes_old=lo, s_new=round(runs["new"][1], 3), s_old=round(runs["old"][1], 3),
                   orders_differ=nd)
        recs.append(rec)
        print(f"  k {k:2d} nout {nout:2d}: lanes {ln:2d} {runs['new'][1]:8.2f} s, lanes {lo:2d} {runs['old'][1]:8.2f} s, "
              f"{'same bits' if nd == 0 else f'{nd} orders DIFFER'}", flush=True)
    ok = bad == 0
    summ = dict(series=series, state=state, N=N, threads=opt["threads"], mutate=opt["mutate"], s_new=round(t_new, 2),
                s_old=round(t_old, 2), ratio=round(t_old / t_new, 3), calls_differ=bad, passed=ok)
    print(f"  sums: new lanes {t_new:.1f} s, old lanes {t_old:.1f} s, old / new {t_old / t_new:.3f}", flush=True)
    print(f"  {'PASS' if ok else 'FAIL'}: {N - bad} of {N} calls the same bits", flush=True)
    if opt["out"]:
        with open(opt["out"] + ".jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(dict(summary=summ)) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
