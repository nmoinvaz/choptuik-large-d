#!/usr/bin/env python3
"""Zeros of the test field's det M_psi(lambda) (hp/modecount_cx, notes/GHOST_ADMIXTURE.md) along a horizontal line
Im lambda = s omega, located and refined: for decaying modes left of the counts' boxes, and for comparison with
published growth rates.

    python3 modecount_test_axis.py STATE.npz --lo=-0.6 --hi=-0.02 [--h=0.02] [--s=0] [--threads=N] [--M=N]
                                   [--out=hp/data/modecount_test_axis.jsonl]

g(lambda) = det M_psi(lambda) prod (lambda - p_j), p_j the candidate poles (hp/modecount_common.horizon_poles) within
omega/2 of the line, is sampled on the grid lo, lo + h, ..., hi of Re lambda. Where its phase changes by more than
pi/2 between neighbours (a zero of det M_psi on or near the line: on the real axis det M_psi is real and the phase
jumps by pi), the zero is refined by the secant method on g in the complex plane (mpmath.findroot), and the zero,
the phase steps and |g| along the line are recorded.
"""
import cmath
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modecount_common import setup, horizon_poles  # noqa: E402
from modecount_wind import parse  # noqa: E402


def main():
    args, opts = parse(sys.argv[1:])
    st = setup(args[0], M=int(opts["M"]) if "M" in opts else None, threads=int(opts.get("threads", 1)), log=print)
    lo, hi, h, s = float(opts["lo"]), float(opts["hi"]), float(opts.get("h", 0.02)), float(opts.get("s", 0))
    out = opts.get("out", os.path.join(HERE, "data", "modecount_test_axis.jsonl"))
    im = s * st.omega
    Q = [p for _, p in horizon_poles(st) if abs(p.imag - im) < 0.5 * st.omega and lo - 2 < p.real < hi + 2]
    print(f"D {st.D}: M {st.M}, |F| {st.F:.1e}, omega {st.omega:.6f}; line Im = {s} omega = {im:.6f}, Re {lo} .. {hi}; "
          f"poles multiplied out: {[(round(p.real, 4), round(p.imag, 4)) for p in Q]}", flush=True)

    def lng(lam):
        la, arg, pmin, _ = st.c.test_det(st.zh, st.zl, float(lam.real), float(lam.imag))
        v = complex(la, arg) + sum(cmath.log(lam - p) for p in Q)
        return v, pmin

    xs = np.arange(lo, hi + 0.5 * h, h)
    vals = []
    t0 = time.time()
    for x in xs:
        v, pmin = lng(complex(x, im))
        vals.append(v)
        print(f"  {x:+.4f}: ln|g| {v.real:12.4f}, arg g {math.remainder(v.imag, 2 * math.pi):+.4f}, min |u_kk| {pmin:.1e}",
              flush=True)
    steps = [math.remainder(b.imag - a.imag, 2 * math.pi) for a, b in zip(vals, vals[1:])]
    zeros = []
    ref = max(v.real for v in vals)
    for i, d in enumerate(steps):
        if abs(d) > math.pi / 2:
            z0 = complex(0.5 * (xs[i] + xs[i + 1]), im)

            def f(lam):
                v, _ = lng(complex(lam))
                return mp.mpc(cmath.exp(v - ref))

            try:
                z = complex(mp.findroot(f, (mp.mpc(z0), mp.mpc(z0 + 0.25 * h + 1e-3j)), solver="secant", tol=1e-24,
                                        maxsteps=40, verify=False))
                zeros.append(dict(start=[z0.real, z0.imag], re=z.real, im=z.imag, im_over_omega=z.imag / st.omega,
                                  ln_g_at=lng(z)[0].real))
                print(f"  zero near {z0.real:+.4f}: {z.real:+.10f} {z.imag:+.10f}i (Im / omega {z.imag / st.omega:+.6f})",
                      flush=True)
            except Exception as exc:          # noqa: BLE001  (secant failure: record and go on)
                zeros.append(dict(start=[z0.real, z0.imag], error=type(exc).__name__))
                print(f"  zero near {z0.real:+.4f}: refinement failed ({type(exc).__name__})", flush=True)
    rec = dict(state=os.path.basename(args[0]), D=st.D, M=st.M, F=st.F, omega=st.omega, line_im=im, s=s, lo=lo, hi=hi,
               h=h, poles=[[p.real, p.imag] for p in Q], zeros=zeros,
               samples=[[float(x), v.real, math.remainder(v.imag, 2 * math.pi)] for x, v in zip(xs, vals)],
               sec=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
