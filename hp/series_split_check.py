"""The octo-double series one order at a time with the horizon series' middle terms formed ahead of each order
(Solver.series_o split=1, Ctx::series_split) against the default form (split=0), on a stored eight-part state in its
chain's own system and plan (octo_corrector.truth_system), through one octo-double LU at the state (Solver.newton_o).

    python3 series_split_check.py RAW STATE.npz [--N 24] [--threads 6] [--Np 6] [--mutate 0] [--out PREFIX]

Pass, fixed before the first run: z_1 the same bits in all eight parts by both forms (order 1 has no middle terms);
for k = 2..N, d_split(k) = |z_k^split - z_k| / |z_k| <= 1e3 max(d_oa(k), 2^-420), where d_oa(k) is the same
difference of the horizon's alpha form (Solver.set_horizon_alpha(1), split 0: the same model with other rounding,
ojet_check.py's d_oa) and |.| the largest entry. Reported: the seconds of each form and their ratio. Control, which
must fail by more than 1e3 (max_k d / threshold > 1e3): the split form without the middle terms' order l = 1
(series_o's mutate model::SPLIT_MUT_DROP_L1) to order Np. --mutate 31 puts that control in the split form's place for
the whole check, which must then fail. Exit 1 on failure, 2 for a build without series_o's split."""
import json
import sys
import time

import numpy as np

SPLIT_MIDDLE = 1
SPLIT_MUT_DROP_L1 = 31


def series(s, zo, n, split=0, mutate=0):
    """z_0..z_n (eight-part lists) by Solver.series_o"""
    return [zo] + [[np.asarray(p) for p in z8] for z8 in s.series_o(list(zo), n, None, mutate=mutate, split=split)]


def rel(A, B):
    """d(k) = max |A_k - B_k| / max |B_k|, the difference in octo-double (multidouble's od_axpy), k = 0..len - 1"""
    from multidouble import od_axpy
    out = []
    for a, b in zip(A, B):
        d = od_axpy(-1.0, *(np.ascontiguousarray(p) for p in a), *(np.ascontiguousarray(p) for p in b))
        out.append(float(np.max(np.abs(d[0]))) / max(float(np.max(np.abs(b[0]))), 1e-300))
    return out


def same_bits(a, b):
    return all(np.array_equal(np.asarray(p).view(np.uint64), np.asarray(q).view(np.uint64)) for p, q in zip(a, b))


def main(argv):
    from octo_corrector import truth_system
    raw, state = argv[0], argv[1]
    opt = dict(N=24, threads=6, Np=6, mutate=0, out="")
    for k in list(opt):
        if "--" + k in argv:
            v = argv[argv.index("--" + k) + 1]
            opt[k] = type(opt[k])(v) if opt[k] != "" else v
    z, meta, (s, _, _, _), info = truth_system(raw, state, threads=opt["threads"])
    if len(z) != 8:
        raise SystemExit(f"{state}: {len(z)} parts, not eight")
    zo = [np.ascontiguousarray(np.asarray(p)) for p in z]
    N, Np, mut = opt["N"], opt["Np"], opt["mutate"]
    try:
        s.series_o(list(zo), 0, None, split=0)
    except TypeError:
        print("  this hpkernel has no series_o split (built before the split form)", flush=True)
        return 2
    except ValueError:
        pass
    print(f"series_split_check {state}: D {meta.get('D_str', meta.get('D'))!s:.18}, ks {meta.get('ks')}, {info['plan']} "
          f"plan, N {N}, Np {Np}, {opt['threads']} threads" + (f", mutate {mut} in the split form" if mut else ""),
          flush=True)
    t0 = time.time()
    s.newton_o(list(zo))
    print(f"  newton_o at the state: {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    Z = series(s, zo, N)
    t_ref = time.time() - t0
    t0 = time.time()
    Zs = series(s, zo, N, split=SPLIT_MIDDLE, mutate=mut)
    t_split = time.time() - t0
    print(f"  N {N}: default {t_ref:.1f} s, split {t_split:.1f} s, ratio {t_ref / t_split:.2f}", flush=True)
    # set_horizon_alpha drops the octo-double LU, so each form's LU is formed again at the state (outside the timings)
    s.set_horizon_alpha(1)
    s.newton_o(list(zo))
    t0 = time.time()
    Za = series(s, zo, N)
    t_alpha = time.time() - t0
    s.set_horizon_alpha(0)
    s.newton_o(list(zo))
    Zc = series(s, zo, Np, split=SPLIT_MIDDLE, mutate=SPLIT_MUT_DROP_L1)
    d_s, d_a, d_c = rel(Zs, Z), rel(Za, Z), rel(Zc, Z[:Np + 1])
    thr = [1e3 * max(d, 2.0 ** -420) for d in d_a]
    print(" k   |z_k|      d_split    d_oa       d_split/thr  d_ctrl", flush=True)
    for k in range(1, N + 1):
        zk = float(np.max(np.abs(Z[k][0])))
        c = f"  {d_c[k]:.2e}" if k <= Np else ""
        print(f"{k:2d}   {zk:.2e}   {d_s[k]:.2e}   {d_a[k]:.2e}   {d_s[k] / thr[k]:.2e}   {c}", flush=True)
    order1 = same_bits(Zs[1], Z[1])
    worst = max(d_s[k] / thr[k] for k in range(2, N + 1)) if N >= 2 else 0.0
    ctrl = max(d_c[k] / thr[k] for k in range(2, Np + 1)) if Np >= 2 else 0.0
    ok = order1 and worst <= 1.0
    fired = ctrl > 1e3
    print(f"  {'PASS' if ok else 'FAIL'}: z_1 {'the same bits' if order1 else 'DIFFERS'}, worst d_split / (1e3 d_oa) "
          f"{worst:.2e}; control (l = 1 dropped) {ctrl:.2e} times the threshold, {'fired' if fired else 'DID NOT FIRE'}",
          flush=True)
    if opt["out"]:
        summ = dict(state=state, N=N, Np=Np, threads=opt["threads"], mutate=mut, s_default=round(t_ref, 2),
                    s_split=round(t_split, 2), s_alpha=round(t_alpha, 2), ratio=round(t_ref / t_split, 3),
                    d_split=d_s[1:], d_oa=d_a[1:], d_ctrl=d_c[1:], order1_same=order1, worst=worst, control=ctrl,
                    passed=ok, control_fired=fired)
        with open(opt["out"] + ".json", "w") as f:
            json.dump(summ, f, indent=1)
    return 0 if ok and fired else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
