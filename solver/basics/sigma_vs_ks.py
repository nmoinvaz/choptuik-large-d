"""SSH Frobenius exponent of the stored chain states against their horizon series order ks (#1086).

For each stored state (hp layout: p1, c0, K node states, m_h, a_h, Delta; physical values on the half tau grid) this
prints D, ks, the real part of the leading singular exponent of p at the self-similarity horizon,

    sigma_R = (n/2 - 1 + <g>) / (1 - <g>),   g = (n - 1)(a_h^2 - 1) = c_x(1),   n = D - 2,

and Delta, and the difference of Delta from a reference chain's cubic spline in D (with the spline's leave-one-out
error). The exponent is that of the
homogeneous solution u^sigma q(theta), u = 1 - x, of x(x - c) p_x = -omega x p_th + G p - c p + (n/2) c (p + m) at
x = 1 (notes/LARGE_D_BASICS.md section 2).

Usage: python3 sigma_vs_ks.py STATES_DIR [--ref=SUBSTRING] [--mutate=g]
  STATES_DIR: a folder holding anm_pin3_D*.npz (searched recursively), e.g. a git archive of raw-data folders.
  --ref: substring of the paths of the reference chain (default ks293).
  --mutate=g: control; sigma_R from <g> = 0 (the flat-space exponent n/2 - 1), which must move the crossing.
"""
import glob
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline


def read(f):
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    y = z["zh"].astype(float)
    M, K, D = meta["M"], len(meta["nodes"]), meta["D"]
    return meta, y, M, K, D


def sigma_r(D, ah, mutate):
    n = D - 2
    g = 0.0 if mutate == "g" else float(((n - 1) * (ah ** 2 - 1)).mean())
    return (n / 2 - 1 + g) / (1 - g), g


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ref_key = opts.get("ref", "ks293")
    mutate = opts.get("mutate", "")
    rows = []
    for f in sorted(glob.glob(args[0] + "/**/anm_pin3_D*.npz", recursive=True)):
        meta, y, M, K, D = read(f)
        ah = y[3 * M + 4 * M * K: 4 * M + 4 * M * K]
        s, g = sigma_r(D, ah, mutate)
        rows.append(dict(D=D, path=f[len(args[0]):].split("/anm")[0], ks=meta["ks"], sig=s, g=g, Delta=y[-1]))
    rows.sort(key=lambda r: r["D"])
    ref = [r for r in rows if ref_key in r["path"]]
    rD, iu = np.unique(np.array([r["D"] for r in ref]), return_index=True)
    rX = np.array([r["Delta"] for r in ref])[iu]
    spline = CubicSpline(rD, rX) if rD.size > 3 else None
    if spline is not None:
        # leave-one-out interpolation error of the reference, the floor of the comparison
        loo = [abs(CubicSpline(np.delete(rD, i), np.delete(rX, i))(rD[i]) - rX[i]) for i in range(2, rD.size - 2)]
        print("# reference %s: %d points, D %.4f..%.4f, leave-one-out spline error median %.1e max %.1e"
              % (ref_key, rD.size, rD.min(), rD.max(), np.median(loo), max(loo)))
    print("# D ks sigma_R sigma_R-ks <g> Delta Delta-Delta_ref(interp) path")
    for r in rows:
        if ref_key in r["path"]:
            dd = 0.0
        elif spline is not None and rD.min() <= r["D"] <= rD.max():
            dd = r["Delta"] - float(spline(r["D"]))
        else:
            dd = float("nan")
        print("%.4f %d %.3f %+.3f %.5f %.12f %+.3e %s" % (r["D"], r["ks"], r["sig"], r["sig"] - r["ks"], r["g"],
                                                         r["Delta"], dd, r["path"]))


if __name__ == "__main__":
    main()
