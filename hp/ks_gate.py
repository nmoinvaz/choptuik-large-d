"""The Moebius horizon's truncation at a saved form-S state (#1098): the octo-double residual's largest change when the
horizon series runs to ks + step instead of ks, for ks from the state's own upwards in steps until the change is at most
--bar; prints each ks's change and, last, the first ks that meets the bar (or none). In octo-double the change resolves
far below the quad-double residual's floor (the jumps' ks+200 |dF| sat at 1.2e-45 to 1.8e-45 at D 1291 whatever ks was).

--down 1: when the state's ks meets the bar, ks is stepped down instead, while the change from ks - step to ks stays
at most --bar (the same measure, the change at ks - step's own ks + step), down to --ks_min; the gate is the last ks
that meets it. At D 4000 (a 9.995) the start ks 6017 read 2.0e-89 against a bar of 1e-70. The control (--down 1):
Delta moved by --ctl_dmh (1e-40) at the gated ks must change the residual by more than --bar, so a
measure that does not resolve the bar cannot pass.

    python3 hp/ks_gate.py STATE [--bar 1e-64] [--step 200] [--tries 6] [--kfreeze 13] [--threads 4]
        [--down 1 --ks_min 400 --ctl_dmh 1e-40] [--ks0 KS: start there instead of at the state's ks]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp  # noqa: E402

import form_s  # noqa: E402
from octo_corrector import load_o  # noqa: E402


def opts(argv):
    o = dict(bar="1e-64", step="200", tries="6", kfreeze="13", threads="4", down="0", ks_min="400", ctl_dmh="1e-40",
             ks0="")
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


def change(F1, F0):
    """the largest change between two octo-double residuals, their eight parts summed at the caller's precision"""
    def s(F, i):
        return mp.fsum(mp.mpf(float(F[q][i])) for q in range(8))
    return max(abs(s(F1, i) - s(F0, i)) for i in range(F0.shape[1]))


def down(z8, meta, D, ks, F_ks, step, bar, kf, thr, ks_min, dmh):
    """steps ks down from a ks that meets the bar while the change from ks - step to ks stays at most bar; returns the
    last ks that meets it, after the m_h control at that ks"""
    while ks - step >= ks_min:
        F_lo = resid(z8, meta, D, ks - step, kf, thr)
        d = change(F_ks, F_lo)
        print(f"down: ks {ks - step} -> {ks}: octo-double residual change {mp.nstr(d, 3)} (bar {bar:g})", flush=True)
        if d > bar:
            break
        ks, F_ks = ks - step, F_lo
    # the control: Delta (the state's last unknown) moved by dmh at the gated ks must move the residual past bar. dmh
    # goes into the third part: added to the second (about 1e-17, its unit in the last place about 1e-33) a 1e-40 is
    # lost to rounding, which is why the first two controls (m_h, then Delta, both in the second part) read 0.0
    zc = [np.array(z8[q], dtype=float) for q in range(8)]
    zc[2][-1] += dmh
    dc = change(resid(zc, meta, D, ks, kf, thr), F_ks)
    ok = dc > bar
    print(f"control: Delta + {dmh:g} at ks {ks} changes the residual by {mp.nstr(dc, 3)} (must exceed {bar:g}): "
          f"{'fires' if ok else 'FAILS'}", flush=True)
    return ks if ok else None


def main():
    pos, o = opts(sys.argv[1:])
    z8, meta = load_o(pos[0])
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    step, bar, kf, thr = int(o["step"]), float(o["bar"]), int(o["kfreeze"]), int(o["threads"])
    ks = int(o["ks0"]) if o["ks0"] else int(meta["ks"])
    mp.dps = 160
    F0 = resid(z8, meta, D, ks, kf, thr)
    for i in range(int(o["tries"])):
        F1 = resid(z8, meta, D, ks + step, kf, thr)
        d = change(F1, F0)
        print(f"ks {ks} -> {ks + step}: octo-double residual change {mp.nstr(d, 3)} (bar {bar:g})", flush=True)
        if d <= bar:
            if i == 0 and o["down"] == "1":
                kd = down(z8, meta, D, ks, F0, step, bar, kf, thr, int(o["ks_min"]), float(o["ctl_dmh"]))
                if kd is None:
                    print("gate none (the control failed)", flush=True)
                    return 1
                ks = kd
            print(f"gate ks {ks}", flush=True)
            return 0
        ks, F0 = ks + step, F1
    print("gate none", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
