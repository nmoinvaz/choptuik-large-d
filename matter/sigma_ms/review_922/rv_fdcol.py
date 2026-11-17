#!/usr/bin/env python3
"""Review of smdriver.newton_band (#922), K3b: where the finite-difference Jacobian of rv_newton.py and
Solver.jacobian differ, is it the finite difference's truncation? For chosen columns the central difference of
Solver.residual is taken at h = 2^-33, 2^-32 and 2^-31 and extrapolated (Richardson, (4 D(h) - D(2h)) / 3). A
truncation error falls by 4 per halving of h and leaves the extrapolated column at Solver.jacobian's; an error of
Solver.jacobian stays where it is.

    PYTHONPATH=<site> python3 rv_fdcol.py --state FILE.npz --cols 158,286 [--random 40] [--out FILE.json]
      Criterion, written before the run: the extrapolated columns equal Solver.jacobian's to 1e-18 in every entry.
      Exit 0 if so.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import smdriver as sd  # noqa: E402
from multidouble import dd_add  # noqa: E402


def column(s, zh, zl, j, h):
    e, zero = np.zeros(len(zh)), np.zeros(len(zh))
    e[j] = h
    ph, pl = dd_add(zh, zl, e, zero)
    mh, ml = dd_add(zh, zl, -e, zero)
    Fp, Fm = s.residual([ph, pl]), s.residual([mh, ml])
    dh, dl = dd_add(np.asarray(Fp[0]), np.asarray(Fp[1]), -np.asarray(Fm[0]), -np.asarray(Fm[1]))
    return np.asarray(dh) / (2 * h), np.asarray(dl) / (2 * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--cols", default="")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    z, meta = sd.load(a.state)
    zh, zl = np.asarray(z[0], float), np.asarray(z[1], float)
    eta = None if meta["eta"] == "inf" else float(meta["eta"])
    s = sd.make_solver(eta, meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                       threads=a.threads, D=meta["D"])
    s.set_segs(meta["segs"])
    Jh, Jl = (np.asarray(x, float) for x in s.jacobian(zh, zl))
    cols = [int(c) for c in a.cols.split(",") if c]
    cols += [int(c) for c in np.random.default_rng(922).choice(len(zh), a.random, replace=False)]
    rows, worst = [], 0.0
    for j in cols:
        d = {}
        for p in (33, 32, 31):
            ch, cl = column(s, zh, zl, j, 2.0 ** -p)
            d[p] = (ch, cl)
        e33 = (d[33][0] - Jh[:, j]) + (d[33][1] - Jl[:, j])
        e32 = (d[32][0] - Jh[:, j]) + (d[32][1] - Jl[:, j])
        e31 = (d[31][0] - Jh[:, j]) + (d[31][1] - Jl[:, j])
        rich = (4 * e33 - e32) / 3
        i = int(np.argmax(np.abs(e33)))
        row = dict(col=j, row=i, e33=float(np.abs(e33).max()), e32=float(np.abs(e32).max()),
                   e31=float(np.abs(e31).max()), ratio=float(abs(e32[i] / e33[i])) if e33[i] else float("nan"),
                   richardson=float(np.abs(rich).max()), colmax=float(np.abs(Jh[:, j]).max()))
        rows.append(row)
        worst = max(worst, row["richardson"])
        print(f"  column {j:5d}: max |FD - J| at h = 2^-33, 2^-32, 2^-31: {row['e33']:.2e}, {row['e32']:.2e}, "
              f"{row['e31']:.2e} (ratio {row['ratio']:.2f} at row {i}); extrapolated {row['richardson']:.2e}; largest "
              f"entry of the column {row['colmax']:.2e}", flush=True)
    ok = worst <= 1e-18
    print(f"K3b ({len(cols)} columns): the extrapolated columns differ from Solver.jacobian's by at most {worst:.2e}: "
          f"{'PASS' if ok else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(state=os.path.basename(a.state), rows=rows, worst=worst, ok=ok), fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
