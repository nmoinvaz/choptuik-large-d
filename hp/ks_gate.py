"""The Moebius horizon's truncation at a saved form-S state (#1098): the octo-double residual's largest change when the
horizon series runs to ks + step instead of ks, for ks from the state's own upwards in steps until the change is at most
--bar; prints each ks's change and, last, the first ks that meets the bar (or none). In octo-double the change resolves
far below the quad-double residual's floor (the jumps' ks+200 |dF| sat at 1.2e-45 to 1.8e-45 at D 1291 whatever ks was).

    python3 hp/ks_gate.py STATE [--bar 1e-64] [--step 200] [--tries 6] [--kfreeze 13] [--threads 4]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp  # noqa: E402

import form_s  # noqa: E402
from octo_corrector import load_o  # noqa: E402


def opts(argv):
    o = dict(bar="1e-64", step="200", tries="6", kfreeze="13", threads="4")
    pos = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            o[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    return pos, o


def resid(z8, meta, D, ks, kf, thr):
    s = form_s.make_solver_s(D, int(meta["M"]), list(meta["nodes"]), kt=int(meta.get("kt", 36)), kc=int(meta["kc"]),
                             ks=ks, threads=thr, kfreeze=kf, segs=meta["segs"], mob=float(meta.get("hmob", 0.0) or 0.0))
    return np.array([np.asarray(f) for f in s.residual_s([np.ascontiguousarray(z8[q]) for q in range(8)])])


def main():
    pos, o = opts(sys.argv[1:])
    z8, meta = load_o(pos[0])
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    step, bar, kf, thr = int(o["step"]), float(o["bar"]), int(o["kfreeze"]), int(o["threads"])
    ks = int(meta["ks"])
    mp.dps = 160
    F0 = resid(z8, meta, D, ks, kf, thr)
    for _ in range(int(o["tries"])):
        F1 = resid(z8, meta, D, ks + step, kf, thr)
        d = max(abs(mp.fsum(mp.mpf(float(F1[q][i])) for q in range(8)) - mp.fsum(mp.mpf(float(F0[q][i])) for q in range(8)))
                for i in range(F0.shape[1]))
        print(f"ks {ks} -> {ks + step}: octo-double residual change {mp.nstr(d, 3)} (bar {bar:g})", flush=True)
        if d <= bar:
            print(f"gate ks {ks}", flush=True)
            return 0
        ks, F0 = ks + step, F1
    print("gate none", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
