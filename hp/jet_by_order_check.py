"""The octo-double series formed one order at a time (Solver.series_o, anm_run.py --jet_by_order 1) against the series by
jets of growing length (Solver.residual_jet_o, coefficients_o's default), on a stored eight-part state in its chain's
own system and plan (octo_corrector.truth_system), through one octo-double LU at the state (Solver.newton_o).

    python3 jet_by_order_check.py RAW STATE.npz [--N 24] [--threads 6] [--Np 6] [--out PREFIX]

Pass, fixed before the first run: z_1..z_N the same bits in all eight parts by both forms; the same with a planted
change of every order's right-hand side (a hook that moves one entry of b by a fixed relative 2^-200, given to both
forms, which tests that series_o applies anm_run's held rows and perturbations where coefficients_o does), at order
Np. Reported: the seconds of each form and their ratio. Controls, each of which must differ from the jets' series: the
order-by-order series with one kept intermediate dropped (series_o's mutate model::JET_MUT_AT_KEEP), and with the
planted change given to the order-by-order form only. Exit 1 on failure."""
import json
import sys
import time

import numpy as np

JET_MUT_AT_KEEP = 24


def differ(A, B):
    """the orders k >= 1 at which the series A and B differ in any bit of any part"""
    out = []
    for k in range(1, max(len(A), len(B))):
        if k >= len(A) or k >= len(B):
            out.append(k)
            continue
        if any(not np.array_equal(np.asarray(p).view(np.uint64), np.asarray(q).view(np.uint64)) for p, q in zip(A[k], B[k])):
            out.append(k)
    return out


def planted(k, b):
    """b with its largest entry's leading part moved by 2^-40 of itself, a change every order's solve sees (a change
    below the ulp of a lower part rounds away and tests nothing)"""
    b = [np.array(p) for p in b]
    i = int(np.argmax(np.abs(b[0])))
    b[0][i] = b[0][i] + abs(b[0][i]) * 2.0 ** -40
    return b


def main(argv):
    from anm_run import coefficients_o
    from octo_corrector import truth_system
    raw, state = argv[0], argv[1]
    opt = dict(N=24, threads=6, Np=6, out="")
    for k in list(opt):
        if "--" + k in argv:
            v = argv[argv.index("--" + k) + 1]
            opt[k] = type(opt[k])(v) if opt[k] != "" else v
    z, meta, (s, _, _, _), info = truth_system(raw, state, threads=opt["threads"])
    if len(z) != 8:
        raise SystemExit(f"{state}: {len(z)} parts, not eight")
    zo = [np.ascontiguousarray(np.asarray(p)) for p in z]
    N, Np = opt["N"], opt["Np"]
    print(f"jet_by_order_check {state}: D {meta.get('D_str', meta.get('D'))!s:.18}, {info['plan']} plan, N {N}, Np {Np}, "
          f"{opt['threads']} threads", flush=True)
    t0 = time.time()
    s.newton_o(list(zo))
    t_lu = time.time() - t0
    print(f"  newton_o at the state: {t_lu:.1f} s", flush=True)
    t0 = time.time()
    A, _ = coefficients_o(s, zo, N)
    t_jet = time.time() - t0
    t0 = time.time()
    Bz, _ = coefficients_o(s, zo, N, by_order=True)
    t_ord = time.time() - t0
    d_main = differ(A, Bz)
    print(f"  N {N}: jets {t_jet:.1f} s, by order {t_ord:.1f} s, ratio {t_jet / t_ord:.2f}; orders that differ {d_main}",
          flush=True)
    Ap, _ = coefficients_o(s, zo, Np, perturb=planted)
    Bp, _ = coefficients_o(s, zo, Np, perturb=planted, by_order=True)
    d_hook = differ(Ap, Bp)
    print(f"  Np {Np} with the planted change in both: orders that differ {d_hook}", flush=True)
    # controls
    Cz = [A[0]] + [list(z8) for z8 in s.series_o(list(zo), Np, None, mutate=JET_MUT_AT_KEEP)]
    c_keep = differ(A[:Np + 1], Cz)
    Ch, _ = coefficients_o(s, zo, Np, perturb=planted, by_order=True)
    c_hook = differ(A[:Np + 1], Ch)
    print(f"  controls: a kept intermediate dropped, orders that differ {c_keep}; the planted change in the "
          f"order-by-order form only, {c_hook}", flush=True)
    ok = not d_main and not d_hook
    controls = bool(c_keep) and bool(c_hook)
    summ = dict(state=state, N=N, Np=Np, threads=opt["threads"], s_lu=round(t_lu, 2), s_jets=round(t_jet, 2),
                s_by_order=round(t_ord, 2), ratio=round(t_jet / t_ord, 3), differ=d_main, differ_hook=d_hook,
                control_keep=c_keep, control_hook=c_hook, passed=ok, controls_fired=controls)
    print(f"  {'PASS' if ok else 'FAIL'}: {N - len(d_main)} of {N} orders the same bits, {Np - len(d_hook)} of {Np} "
          f"with the hook; controls {'fired' if controls else 'DID NOT FIRE'}", flush=True)
    if opt["out"]:
        with open(opt["out"] + ".json", "w") as f:
            json.dump(summ, f, indent=1)
    return 0 if ok and controls else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
