#!/usr/bin/env python3
"""How far ElimO's projection moves a converged state, per block, and the Delta step of Newton's first move from the
state and from its projection (#823).

    cd hp && python3 proj_shift.py RAW STATE [--threads 3] [--switch 0] [--reseed_o 3] [--reseed_start pred]
        [--json OUT] [--ref_proj D] [--ref_plain D] [--mutate block]

The system is octo_corrector.py planted's at --rel 0 (pinned_o at the state, Reseeder with the given re-seeder
options, ElimO at --switch). Printed per block of the layout (centre, node i, m_h, a_h, Delta; reseed.offsets):
max|z|, max|z_proj - z| and the max of three steps in the block: the plain Newton step at the state (PinnedO), the
step at the projected point, and ElimO's step from the state (the projection's displacement plus that step, what
newton_od takes at switch 0). Then the pinned max|F| at both points and each step's Delta part.

Checks (exit 1 on failure): the displacement is zero outside m_h and a_h, the only data ElimO writes back
(octo_corrector.with_horizon); with --ref_proj and --ref_plain, ElimO's and the plain step's Delta parts agree with
those recorded values to 1e-6 relative (#816's +1.100878e-5 at switch 0, I823-mac's dz0_truth +3.311085e-8). Control:
--mutate block shifts the m_h block's start by one entry, so that the displacement's first m_h entry counts as
outside, and the first check must fail."""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np


def blocks(M, K, mutate=""):
    """(name, lo, hi) of reseed.offsets' layout; --mutate block moves m_h's start one entry later"""
    from reseed import offsets
    o = offsets(M, K)
    out = [("centre", 0, o["node"](0))] + [(f"node {i}", o["node"](i), o["node"](i + 1)) for i in range(K)]
    mh = o["mh"] + (1 if mutate == "block" else 0)
    out[-1] = (out[-1][0], out[-1][1], mh)
    return out + [("m_h", mh, o["ah"]), ("a_h", o["ah"], o["L"] - 1), ("Delta", o["L"] - 1, o["L"])]


def main(argv):
    from hpkernel.driver import load_q
    from freeze_round import mpv
    from multidouble import od_axpy
    from octo_corrector import ElimO, o8, o8_of_q, omax, pinned_o, plant_start
    from reseed import Reseeder
    mp.mp.dps = 150
    raw, path = argv[0], argv[1]
    opt = dict(threads="3", switch="0", reseed_o="3", reseed_start="pred", json="", ref_proj="", ref_plain="",
               mutate="")
    for k in list(opt):
        if "--" + k in argv:
            opt[k] = argv[argv.index("--" + k) + 1]
    unknown = [a for a in argv[2:] if a.startswith("--") and a[2:] not in opt]
    if unknown:
        raise SystemExit(f"proj_shift.py: unknown options {unknown}")
    t0 = time.time()
    z4, meta = load_q(os.path.join(raw, path))
    z4 = o8(z4)
    D = mp.mpf(meta["D_str"])
    s, sp, sq, so = pinned_o(meta, z4, threads=int(opt["threads"]))
    rs = Reseeder(meta, D, threads=int(opt["threads"]), qd=True, od=int(opt["reseed_o"]), qd_start=opt["reseed_start"])
    zs, _ = plant_start(z4, rs, 0.0, 1, 0)
    z = o8_of_q(zs)                                   # planted --rel 0's start, as _run_od receives it
    el = ElimO(so, rs, switch=float(opt["switch"]))
    zp = el.project(z)
    L, M, K = len(z[0]), rs.M, rs.K
    if L != 4 * M * (K + 1) + 1:
        raise SystemExit(f"proj_shift.py: L {L} is not the layout's 4 M (K + 1) + 1 at M {M}, K {K}")
    disp = mpv(o8(od_axpy(-1.0, *z, *zp)), range(L))
    zv = mpv(z, range(L))
    rec = dict(state=path, D_str=meta["D_str"], switch=float(opt["switch"]), reseed_o=int(opt["reseed_o"]),
               reseed_start=opt["reseed_start"], counts=dict(el.counts), mutate=opt["mutate"], blocks=[])
    print(f"proj_shift {path}: D {mp.nstr(D, 15)}, M {M}, K {K}, L {L}; ElimO switch {opt['switch']}, reseed_o "
          f"{opt['reseed_o']}, reseed_start {opt['reseed_start']}; projection {dict(el.counts)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    F_z, F_p = omax(so.residual_o(z)), omax(so.residual_o(zp))
    steps, dv = {}, {}
    for name, fn in (("plain at the state", lambda: so.newton_o(z)),
                     ("plain at the projection", lambda: so.newton_o(zp)),
                     ("ElimO's from the state", lambda: el.newton_o(z))):
        t1 = time.time()
        st = fn()
        dv[name] = d = mpv(o8(st[0]), range(L))
        steps[name] = dict(dDelta=float(d[-1]), dz=float(max(abs(x) for x in d)), piv=[st[3], st[4]],
                           s=round(time.time() - t1, 1))
        print(f"  {name}: dDelta {mp.nstr(d[-1], 8)}, max|dz| {mp.nstr(max(abs(x) for x in d), 6)}, pivots "
              f"{st[3]:.3e}..{st[4]:.3e} ({time.time() - t1:.0f} s)", flush=True)
    print("  per block, max|.|: the state z, the projection's move z_proj - z, and the three steps (plain at z, plain "
          "at z_proj, ElimO's from z)", flush=True)
    outside = mp.mpf(0)
    for name, lo, hi in blocks(M, K, opt["mutate"]):
        row = dict(block=name, lo=lo, hi=hi, z=max(abs(x) for x in zv[lo:hi]), disp=max(abs(x) for x in disp[lo:hi]))
        for key, sn in (("plain_z", "plain at the state"), ("plain_p", "plain at the projection"),
                        ("elim", "ElimO's from the state")):
            row[key] = max(abs(x) for x in dv[sn][lo:hi])
        cols = ("z", "disp", "plain_z", "plain_p", "elim")
        print(f"  {name:8s} " + " ".join(f"{mp.nstr(row[q], 5):>12s}" for q in cols), flush=True)
        if name not in ("m_h", "a_h"):
            outside = max(outside, row["disp"])
        rec["blocks"].append({q: (float(v) if isinstance(v, mp.mpf) else v) for q, v in row.items()})
    rec.update(F_state=F_z, F_proj=F_p, steps=steps, counts_end=dict(el.counts))
    print(f"  pinned max|F| (octo): state {F_z:.4e}, projection {F_p:.4e}", flush=True)
    ok = outside == 0
    print(f"  check: the displacement outside m_h and a_h {mp.nstr(outside, 3)} (must be 0): "
          f"{'pass' if ok else 'FAIL'}", flush=True)
    for key, name in (("ref_proj", "ElimO's from the state"), ("ref_plain", "plain at the state")):
        if opt[key]:
            ref = float(opt[key])
            rel = abs(steps[name]["dDelta"] / ref - 1)
            good = rel < 1e-6
            ok &= good
            rec[key] = dict(ref=ref, rel=rel, ok=good)
            print(f"  check: {name} dDelta against {ref:+.6e}: relative {rel:.2e} (bound 1e-6): "
                  f"{'pass' if good else 'FAIL'}", flush=True)
    rec["passed"] = bool(ok)
    if opt["json"]:
        with open(opt["json"], "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
    print(f"proj_shift: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
