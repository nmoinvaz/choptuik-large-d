"""The chain's stored Taylor plans against the fast modes' rates (#1086, notes/LARGE_D_BASICS.md section 4).

Each Taylor step of order kt from x to x + h maps a mode e^{n S(x)} by the truncated exponential
T_kt(z) = sum_{k<=kt} z^k/k!, z = n lam h, in place of e^z. For the two fast modes of the linearized equations
(notes/LARGE_D_BASICS.md 3.3) at leading order in 1/n and with c = 1,
  scalar mode F:  n lam_F = n c^2 / (x (x^2 - c^2))      (decays outward, F ~ x^-n (1 - x^2)^(n/2))
  mass mode A:    n lam_A = -(n - 1) / x                   (decays outward, A ~ x^-(n-1))
this prints per state the largest |z| of a step, the largest relative error |T_kt(z) e^-z - 1| over the steps, and
whether |T_kt(z)| exceeds 1 (the discrete step then amplifies a decaying mode).

Usage: python3 plan_stability.py STATE.npz [STATE.npz ...] [--kt=K] [--mutate=h2]
  --mutate=h2: control; doubles every step, which must raise the error by about 2^(kt+1).
"""
import json
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 50


def taylor_ratio(z, kt):
    z = mp.mpf(z)
    t = mp.mpf(1)
    s = mp.mpf(1)
    for k in range(1, kt + 1):
        t *= z / k
        s += t
    return s, s * mp.e ** (-z) - 1


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    mutate = opts.get("mutate", "")
    print("# D kt steps  max|z_F| max_err_F max|T(z_F)|  max|z_A| max_err_A   (file)")
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        segs = meta.get("segs")
        if not segs:
            print("%s: no stored plan" % f)
            continue
        kt = int(opts.get("kt", meta.get("kt", 36)))
        n = meta["D"] - 2
        zF, eF, TF, zA, eA = 0.0, 0.0, 0.0, 0.0, 0.0
        steps = 0
        for sg in segs:
            sg = np.array(sg, dtype=float)
            for x0, x1 in zip(sg[:-1], sg[1:]):
                h = (x1 - x0) * (2 if mutate == "h2" else 1)
                steps += 1
                # integrate the rate over the step at its left end (the plan's steps are short)
                lf = n / (x0 * (x0 ** 2 - 1)) * h
                la = -(n - 1) / x0 * h
                Tf, ef = taylor_ratio(lf, kt)
                Ta, ea = taylor_ratio(la, kt)
                zF, eF, TF = max(zF, abs(lf)), max(eF, float(abs(ef))), max(TF, float(abs(Tf)))
                zA, eA = max(zA, abs(la)), max(eA, float(abs(ea)))
        print("%.4f %d %d  %.2f %.2e %.3f  %.2f %.2e   (%s)" % (meta["D"], kt, steps, zF, eF, TF, zA, eA, f))


if __name__ == "__main__":
    main()
