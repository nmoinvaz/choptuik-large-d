"""The octo-double exact Jacobian with several tangents per evaluation (Solver.jacobian_o(z, tangents=n): multishoot's
jacobian_tangents with multidouble's Oun<n>, n columns of a block from one evaluation) against one tangent (n = 1,
Solver.jacobian_o(z, tangents=1)), on a stored eight-part state in its chain's own system and stored plan
(octo_corrector.truth_system).

    python3 jac_tangents_check.py RAW STATE.npz [--threads 6] [--ns 2,4,8] [--out PREFIX]

Pass, fixed before the first run: for every n, every part of every entry of the Jacobian is the same bits as n = 1's,
and the Newton step with the fastest n (Solver.newton_o(z, tangents=n)) gives the one-tangent step's dz and pivots
bit for bit. Reported: each n's seconds and n = 1's over them. Control, which must differ: n = 4 with Ctx::mutate
JAC_MUT_TANGENT_ORDER (25), each evaluation's tangents read in reverse order. Exit 1 on failure."""
import json
import sys
import time

import numpy as np

JAC_MUT_TANGENT_ORDER = 25


def differ(A, B):
    """the number of entries, over the eight parts, whose bits differ"""
    return int(sum(np.count_nonzero(np.asarray(a).view(np.uint64) != np.asarray(b).view(np.uint64)) for a, b in zip(A, B)))


def main(argv):
    from octo_corrector import truth_system
    raw, state = argv[0], argv[1]
    opt = dict(threads=6, ns="2,4,8", out="")
    for k in list(opt):
        if "--" + k in argv:
            v = argv[argv.index("--" + k) + 1]
            opt[k] = type(opt[k])(v)
    ns = [int(x) for x in opt["ns"].split(",")]
    z, meta, (s, _, _, _), info = truth_system(raw, state, threads=opt["threads"])
    if len(z) != 8:
        raise SystemExit(f"{state}: {len(z)} parts, not eight")
    zo = [np.ascontiguousarray(np.asarray(p)) for p in z]
    print(f"jac_tangents_check {state}: D {meta.get('D_str', meta.get('D'))!s:.18}, {info['plan']} plan, n {ns}, "
          f"{opt['threads']} threads", flush=True)
    t0 = time.time()
    J1 = s.jacobian_o(list(zo), tangents=1)
    t1 = time.time() - t0
    print(f"  n 1: {t1:.1f} s", flush=True)
    secs, diffs = {1: t1}, {}
    for n in ns:
        t0 = time.time()
        Jn = s.jacobian_o(list(zo), tangents=n)
        secs[n] = time.time() - t0
        diffs[n] = differ(J1, Jn)
        print(f"  n {n}: {secs[n]:.1f} s ({t1 / secs[n]:.2f}x), entries that differ from n 1: {diffs[n]}", flush=True)
        del Jn
    best = min(ns, key=lambda n: secs[n])
    st1 = s.newton_o(list(zo), tangents=1)
    stb = s.newton_o(list(zo), tangents=best)
    step_differ = differ(st1[0], stb[0]) + int(st1[3] != stb[3]) + int(st1[4] != stb[4])
    print(f"  Newton step with n {best} against n 1: {step_differ} entries or pivots differ", flush=True)
    Jc = s.jacobian_o(list(zo), mutate=JAC_MUT_TANGENT_ORDER, tangents=4)
    control = differ(J1, Jc)
    print(f"  control (n 4, tangents reversed): {control} entries differ", flush=True)
    ok = all(d == 0 for d in diffs.values()) and step_differ == 0
    fired = control > 0
    summ = dict(state=state, threads=opt["threads"], seconds={str(k): round(v, 2) for k, v in secs.items()},
                ratio={str(n): round(t1 / secs[n], 3) for n in ns}, differ={str(k): v for k, v in diffs.items()},
                best=best, step_differ=step_differ, control=control, passed=ok, control_fired=fired)
    print(f"  {'PASS' if ok else 'FAIL'}: every n's Jacobian {'is' if ok else 'is NOT'} n 1's bit for bit; the fastest "
          f"is n {best} at {t1 / secs[best]:.2f}x; control {'fired' if fired else 'DID NOT FIRE'}", flush=True)
    if opt["out"]:
        with open(opt["out"] + ".json", "w") as f:
            json.dump(summ, f, indent=1)
    return 0 if ok and fired else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
