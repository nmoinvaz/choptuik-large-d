#!/usr/bin/env python3
"""Locate the zeros of g found by a count, from the count's own samples (no new determinants).

    python3 modecount_moment.py TAG STATE.npz [BOX]          BOX: B1c (default) or U; prints winding and sum of zeros

For the upper half of a box symmetric about the real axis (B1c: R1, H1, V2 and the real axis from a2 to aR) or for
the strip U, the argument principle's first moment gives the sum of the zeros of g inside:

    sum z_j = W lam_0 - (1 / 2 pi i) * closed integral of ln g d lam,

with ln g continued along the contour from lam_0 and W the winding (integration by parts of the integral of
lam g'/g); the integral uses the trapezoid rule on the samples. With one zero inside this is its location, to the
accuracy the sample spacing allows (about 0.01-0.05 for the spacings used).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, dmat_odd  # noqa: E402

DATA = os.path.join(HERE, "data")


def main():
    tag, statefile = sys.argv[1], sys.argv[2]
    box = sys.argv[3] if len(sys.argv) > 3 else "B1c"
    r = [x for x in (json.loads(l) for l in open(os.path.join(DATA, "modecount_results.jsonl"))) if x["tag"] == tag][-1]
    zh, zl, m = load(statefile)
    M, K, D = m["M"], len(m["nodes"]), m["D"]
    n = D - 2
    ah = (zh + zl)[3 * M + 4 * M * K:4 * M + 4 * M * K]
    c1 = (n - 1) * (1 - ah ** 2)
    om, y2 = r["omega"], r["y2"]
    dh, dl = dmat_odd(M)
    Dm = (dh + dl) * om
    poles = []
    for kk in range(m.get("ks", 100) + 1):
        poles += [-complex(e) for e in np.linalg.eigvals(Dm + np.diag((c1 + 1) * (kk + 1) - n / 2))]
    Q = np.array([p for p in poles if abs(p.imag) < y2 + 3 * om])
    Kz = np.array([complex(*z) for z in r["expected_zeros"]])
    pc = r["trend"]
    ev = {}
    for line in open(os.path.join(DATA, f"modecount_{tag}_evals.jsonl")):
        e = json.loads(line)
        ev[(e["re"], e["im"])] = e

    def lng(lam):
        e = ev[min(ev, key=lambda k: abs(complex(*k) - lam))]
        assert abs(complex(e["re"], e["im"]) - lam) < 1e-9
        return complex(e["lnabs"], e["arg"]) + complex(np.sum(np.log(lam - Q)) - np.sum(np.log(lam - Kz))) - complex(np.polyval(pc, lam))

    def pts_of(name, rev=False):
        e = r["edges"][name]
        z0, z1 = complex(*e["z0"]), complex(*e["z1"])
        p = [z0 + s[0] * (z1 - z0) for s in e["samples"]]
        return p[::-1] if rev else p

    def join(spec):
        out = []
        for name, rev in spec:
            p = pts_of(name, rev)
            if out and abs(out[-1] - p[0]) < 1e-12:
                p = p[1:]
            out += p
        return out

    if box == "B1c":
        pts = join([("R1", False), ("H1", False), ("V2", True)])
        pts += sorted((complex(x, 0) for x, _, _ in r["real_axis"] if r["a2"] < x <= r["aR"] + 1e-12), key=lambda z: z.real)
    elif box == "U":
        pts = join([("H3", True), ("H2", True), ("H1", True), ("R2", False), ("T", False), ("L2", True)])
    else:
        raise SystemExit("box: B1c or U")
    assert abs(pts[-1] - pts[0]) < 1e-9, (pts[0], pts[-1])
    vals = [lng(p) for p in pts]
    un = [vals[0]]
    for v in vals[1:]:
        un.append(complex(v.real, un[-1].imag + math.remainder(v.imag - un[-1].imag, 2 * math.pi)))
    W = (un[-1].imag - un[0].imag) / (2 * math.pi)
    integral = sum(0.5 * (un[i] + un[i + 1]) * (pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
    s = W * pts[0] - integral / (2j * math.pi)
    print(f"{tag} {box}{' (upper half)' if box == 'B1c' else ''}: {len(pts)} samples, winding of g {W:+.4f}, "
          f"sum of the zeros of g inside {s.real:.4f} {s.imag:+.4f}i")


if __name__ == "__main__":
    main()
