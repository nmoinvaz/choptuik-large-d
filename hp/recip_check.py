"""Check of form S's lane-precision reciprocals (Ctx::exact_recip, #1098) on a saved full state.

With double-double reciprocals (1/k in the series, 1/x0 and 1/(k+1) in the Taylor steps, the centre series' 1/k), the
octo-double residual at a fixed state moved by 1.7e-36 when the Taylor steps were halved (D 300, the last segment),
which the full system's near-null top harmonics turned into about 5e-20 in Delta; with lane-precision ones a change
of step plan moves the residual only by the steps' truncation. The check, on the state's own system (kfreeze 13 unless
--kfreeze): the residual's largest change when every step is halved, PASS when at most --bar (1e-44), and the control
(HP_EXACT_RECIP=0, the double-double reciprocals) must exceed --bar_ctl (1e-40). Second: the residual itself with the
two kinds of reciprocals, to show the size of the change at this state.

    python3 hp/recip_check.py STATE [--kfreeze 13] [--bar 1e-44] [--bar_ctl 1e-40] [--threads 2]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp  # noqa: E402

import form_s  # noqa: E402
from octo_corrector import load_o  # noqa: E402


def halve(g):
    return [g[0]] + [v for a, b in zip(g[:-1], g[1:]) for v in (0.5 * (a + b), b)]


def resid(z8, meta, D, kf, segs, thr):
    kw = dict(kt=int(meta.get("kt", 36)), kc=int(meta["kc"]), ks=int(meta["ks"]), threads=thr, segs=segs,
              mob=float(meta.get("hmob", 0.0) or 0.0))
    s = form_s.make_solver_s(D, int(meta["M"]), list(meta["nodes"]), kfreeze=kf, **kw)
    F = s.residual_s([np.ascontiguousarray(z8[q]) for q in range(8)])
    return np.asarray(F[0]) + np.asarray(F[1])


def main(argv):
    path = argv[0]
    opt = dict(zip(argv[1::2], argv[2::2]))
    kf, bar, bar_ctl, thr = (int(opt.get("--kfreeze", "13")), float(opt.get("--bar", "1e-44")),
                             float(opt.get("--bar_ctl", "1e-40")), int(opt.get("--threads", "2")))
    z8, meta = load_o(path)
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    segs2 = [halve(g) for g in meta["segs"]]
    out = {}
    for mode in ("1", "0"):
        os.environ["HP_EXACT_RECIP"] = mode
        Fb = resid(z8, meta, D, kf, meta["segs"], thr)
        Fh = resid(z8, meta, D, kf, segs2, thr)
        out[mode] = (float(np.abs(Fb).max()), float(np.abs(Fh - Fb).max()), Fb)
        print(f"D {float(D):g} (mob {float(meta.get('hmob', 0) or 0):g}, kfreeze {kf}) exact_recip {mode}: max |F| "
              f"{out[mode][0]:.3e}; halving every step changes F by {out[mode][1]:.3e}", flush=True)
    dF = float(np.abs(out["1"][2] - out["0"][2]).max())
    ok = out["1"][1] <= bar and out["0"][1] > bar_ctl
    print(f"  the two kinds of reciprocals differ at this state by {dF:.3e}; step-plan change {out['1'][1]:.2e} "
          f"(bar {bar:g}) against the control's {out['0'][1]:.2e} (must exceed {bar_ctl:g}): "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
