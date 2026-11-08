#!/usr/bin/env python3
"""Export the inputs of lemma 6a's final step that are not stored (#39 review): Lambda, the weights and the matrix L
of the flat estimate, from the original code (lemma6_sector_M.lemma_S_M and flat_estimate_M, called as
lemma6_jump_M.main calls them: M = 5, rp = 0.08, k = 20, over x^ +- 1e-6, Z = 0.067^2). Balls are written exactly as
(mantissa, exponent) pairs. This is the original machinery; check_6a.py re-evaluates the step from L and B onward.

    python3 export_6a_flat.py --out=f.json [--rp=0.08] [--k=20] [--xia=0.067] [--hbox=1e-6]
"""
import json
import os
import sys
import time
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lemma6_jump_M  # noqa: E402,F401  (installs lemma6_common as the run did)
import lemma6_sector_M as L6SM  # noqa: E402
import fixed_point_M as FPM  # noqa: E402
from flint import ctx  # noqa: E402


def me(x):
    m, r = x.mid().man_exp(), x.rad().man_exp()
    return [[int(m[0]), int(m[1])], [int(r[0]), int(r[1])]]


def main():
    opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
    M, rp, k = 5, opts.get("rp", "0.08"), int(opts.get("k", "20"))
    xia, hbox = opts.get("xia", "0.067"), opts.get("hbox", "1e-6")
    Z = str(Decimal(xia) ** 2)
    ctx.prec = 320
    t0 = time.time()
    S = L6SM.lemma_S_M(M, rp, k, hbox, verbose=False)
    R = L6SM.flat_estimate_M(S, Z, diag=False, verbose=False)
    out = {"M": M, "rp": rp, "k": k, "xia": xia, "Z": Z, "hbox": hbox, "x_hat": list(FPM.ZEROS[M]),
           "W0_str": FPM.W0_STR, "ok_S": bool(S["ok"]), "lip_S": float(S["lip"].upper()), "ok_R": bool(R["ok"]),
           "lam": me(R["lam"]), "lam_str": R["lam"].str(12), "wts": [float(v) for v in R["wts"]],
           "comps": [list(c) for c in R["comps"]], "L": [[me(v) for v in row] for row in R["L"]],
           "R_S": me(S["R"]), "sec": time.time() - t0}
    with open(opts["out"], "w") as fh:
        json.dump(out, fh)
    print(f"lemma S ok {S['ok']} (contraction {out['lip_S']:.4f}); flat estimate ok {R['ok']}, Lambda {out['lam_str']}; "
          f"{len(out['wts'])} weights -> {opts['out']} [{out['sec']:.0f} s]")


if __name__ == "__main__":
    main()
