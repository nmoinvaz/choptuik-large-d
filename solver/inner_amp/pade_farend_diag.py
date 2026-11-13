#!/usr/bin/env python3
"""Post-freeze diagnostics of #547 (labeled so in notes/PADE_FAR_END.md): why the frozen DA2 estimate of the real far
end lies outside the planted range.

For each member of pade_farend.ENSEMBLE, every zero of Q2 with |t| > 1.65 and |t| < 6 (the frozen window ended at 3),
for the real series (scan45J1250 at the three nodes, scan45J1500 at wf 1.00), the real series with one order dropped
(a_0..a_21 only), and the plants D1, D2 and D4 for comparison. Zeros with |t| <= 1.65 (the arc's interior, #548) are not
printed. A cluster is a set of zeros, from distinct members, within 0.05 of their median.

    python3 pade_farend_diag.py OUTDIR [--allow_unraw]
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_farend as F                                                # noqa: E402

pg = F.pg
RMIN, RMAX = 1.65, 6.0


def zeros(sig, n_coef=F.N_COEF):
    mp.mp.dps = F.DPS_DA
    sig = [mp.mpc(z) for z in sig[:n_coef]]
    out = []
    for degs, lp in F.members(2):
        if n_coef < F.N_COEF:
            lp -= F.N_COEF - n_coef
            if lp < 0:
                continue
        ts = F.da_zeros(sig, degs, lp, n_coef)
        out.append({"degs": degs, "lp": lp,
                    "t": sorted((complex(t) for t in ts or [] if RMIN < abs(t) < RMAX), key=abs)})
    return out


def clusters(mem, rad=0.05, need=6):
    """greedy: each zero as a seed; the zeros of distinct members within rad of the seed; keep those with >= need
    members, by size"""
    pts = [(i, t) for i, m in enumerate(mem) for t in m["t"]]
    found = []
    for _, t0 in pts:
        near = {}
        for i, t in pts:
            if abs(t - t0) < rad and (i not in near or abs(t - t0) < abs(near[i] - t0)):
                near[i] = t
        if len(near) >= need:
            v = np.array(list(near.values()))
            c = complex(np.median(v.real), np.median(v.imag))
            if all(abs(c - f["center"]) > rad for f in found):
                found.append({"center": c, "abs": abs(c), "members": len(near),
                              "mad": float(np.median(np.abs(v - c)))})
    return sorted(found, key=lambda f: -f["members"])


def report(lab, mem):
    print(f"== {lab}")
    for m in mem:
        print(f"   {m['degs']} lp {m['lp']}: " + "; ".join(f"{t:.3f} ({abs(t):.3f})" for t in m["t"]))
    cl = clusters(mem)
    for c in cl:
        print(f"   cluster {c['center']:.4f} |t| {c['abs']:.4f}: {c['members']} of {len(mem)} members, mad {c['mad']:.4f}")
    sys.stdout.flush()
    return {"members": mem, "clusters": cl}


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    outdir = argv[0]
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    rec = {"post_freeze": True, "window": [RMIN, RMAX]}
    mp.mp.dps = F.DPS_DA
    real = pg.load_scan(pg.REAL["J1250"])
    for wf in pg.NODES:
        rec[f"real J1250 wf {wf}"] = report(f"real J1250 wf {wf}", zeros(real[wf]))
    mp.mp.dps = F.DPS_DA
    rec["real J1500 wf 1.0"] = report("real J1500 wf 1.0", zeros(pg.load_scan(pg.REAL["J1500"])[1.0]))
    rec["real J1250 wf 1.0, a_0..a_21"] = report("real J1250 wf 1.0, a_0..a_21", zeros(real[1.0], F.N_COEF - 1))
    for name in ("D1 chord2", "D2 arcjump", "D4 chord2 R1.85"):
        kind, wf, R, dex, stretch, _ = F.PLANTS[name]
        pl = F.FarPlant(wf, kind, real[wf], R, dex, stretch)
        rec[name] = report(f"plant {name}, P1 {complex(pl.P1):.4f} |P1| {float(abs(pl.P1)):.4f}", zeros(pl.sigma()))
    json.dump(pg.jsonable(rec), open(os.path.join(outdir, "diag_post_freeze.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
