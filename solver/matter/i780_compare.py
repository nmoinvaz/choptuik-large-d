#!/usr/bin/env python3
"""#780: complex_imag.py's det M against complex_imag_dd.py's at the points of the double counts (D = 10, 20.04).

    python3 i780_compare.py STATE.npz DOUBLE_EVLOG --out=DD_EVLOG [--dd=0] [--rtol=1e-11] [--tol=1e-14] [--limit=N]
                          [--stride=S]

DOUBLE_EVLOG is a count's evaluation log of complex_imag.py (I95-mac/imag/evals_D*.jsonl); its unmutated points are
evaluated in their order by complex_imag_dd.DDTestField and appended to DD_EVLOG as complex_imag "eval" records
(resumable), with the double value and the differences; --stride=S takes every S-th complex point
(Im lam != 0). --dd=0 keeps every propagator in complex_imag.py's double
code (and its rtol, 1e-11), so the light-cone block (double-double, horizon_dd) and the LU are what differ; --dd=all
puts the node segments in double-double too (Taylor, --tol). complex_imag_dd.py count ... --evlog=DD_EVLOG then
reuses these values, so its windings are over the same points. Exits 1 when a phase differs by more than 1e-3 rad.
"""
import json
import math
import os
import sys
import time

import complex_imag as ci
import complex_imag_dd as cd

LIMIT_PHASE = 1e-3


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    o = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
    state, dlog, out = pos[0], pos[1], o["out"]
    bg = ci.Background(state, ks_series=100)
    ndd = None if o.get("dd", "0") == "all" else int(o.get("dd", "0"))
    tf = cd.DDTestField(bg, None, ndd=ndd, tol=float(o.get("tol", "1e-14")), K=32, rtol=float(o.get("rtol", "1e-11")))
    pts = []
    for line in open(dlog):
        r = json.loads(line)
        if r.get("kind") == "eval" and r.get("mutate") is None:
            pts.append((complex(*r["lam"]), complex(*r["lndet"])))
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done[tuple(r["lam"])] = r
    if "stride" in o:
        pts = [p for p in pts if p[0].imag != 0][::int(o["stride"])]
    limit = int(o.get("limit", len(pts)))
    first_bad, worst = None, (0.0, None)
    for i, (lam, vd) in enumerate(pts[:limit]):
        key = (lam.real, lam.imag)
        if key in done:
            r = done[key]
        else:
            t = time.time()
            v = tf.lndet(lam)
            r = dict(kind="eval", lam=[lam.real, lam.imag], lndet=[v.real, v.imag], mutate=None,
                     s=round(time.time() - t, 1),
                     double=[vd.real, vd.imag], dphase=math.remainder(v.imag - vd.imag, 2 * math.pi),
                     dln=v.real - vd.real, ndd=tf.ndd, horizon_tail=tf.last_tail)
            with open(out, "a") as fo:
                fo.write(json.dumps(r) + "\n")
            print("%3d lam %+.5f %+.5fi  ln|det| %.6f  dphase %+.2e  dln %+.2e  (%.1f s)" % (
                i, lam.real, lam.imag, v.real, r["dphase"], r["dln"], r["s"]), flush=True)
        if abs(r["dphase"]) > worst[0]:
            worst = (abs(r["dphase"]), r["lam"])
        if first_bad is None and abs(r["dphase"]) > LIMIT_PHASE:
            first_bad = (i, r["lam"], r["dphase"])
    print("%s: %d points, ndd %d; largest |dphase| %.2e rad at %s; first beyond %.0e rad: %s" % (
        os.path.basename(state), min(limit, len(pts)), tf.ndd, worst[0], worst[1], LIMIT_PHASE,
        "none" if first_bad is None else "point %d at %s, %+.3e rad" % first_bad))
    sys.exit(0 if first_bad is None else 1)


if __name__ == "__main__":
    main()
