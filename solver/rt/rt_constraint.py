"""The constraint Omega_5^+ (EEG Eq. (5e), RT's Omega_2#) on rt_run.py states, in arb (issue #163).

    python3 solver/rt/rt_constraint.py STATE.npz [STATE.npz ...] [--prec 192] [--mutate=scale]

The square system solves (1 + P) Omega_1^+ and Omega_{2,3,4}^+ only; RT (arXiv:1203.3766, Sec. 2.3) show that the
constraint then holds for their solution. Two roots of one grid that differ along a weak direction may differ in it.
Printed: max over the grid of |Omega_5^+|, of |Omega_5^+ / xi| away from xi = 0, and its harmonic-0 part, for each
state, and for a second state of the same grid and D its difference from the first. Omega_5^+ = xi (D^+ omega_2^+ + mu omega_2^+) + 2 mu (omega_2^+ - omega_3^+) + xi (2 omega_3^+ omega_2^+
- (omega_2^+)^2 - (omega_4^+)^2), with omega_i^+ = -P omega_i in the parity-extended unknowns. Control: --mutate=scale
evaluates it with the coefficient 2 mu scaled by 1 + 1e-6, which must raise it on a converged state by at least 1e-9
(printed as a check; exit 1 when it does not).
"""
import os
import sys
import argparse
import numpy as np
import flint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_mp import RTmp
import rt_mpsolve as mps

arb = flint.arb


def omega5(rm, z, c2=1):
    w1, w2, w3, w4, mu = rm.unpack_m(z)
    P = lambda f: f[:, rm.rev]
    X = rm.xm[None, :]
    one_p_X = np.array([1 + v for v in rm.xm], dtype=object)[None, :]
    p2, p3, p4 = -P(w2), -P(w3), -P(w4)
    Dp = rm.Dper.dot(p2) + mu * one_p_X * p2.dot(rm.DxT)
    return X * (Dp + mu * p2) + 2 * c2 * mu * (p2 - p3) + X * (2 * p3 * p2 - p2 * p2 - p4 * p4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--mutate", default=None, choices=["scale"])
    a = ap.parse_args()
    flint.ctx.prec = a.prec
    ok = True
    first = {}
    for path in a.states:
        d = np.load(path)
        D, Nt, N = float(d["D"]), int(d["Nt"]), int(d["N"])
        rm = RTmp(D, Nt, N, prec=a.prec)
        z = mps.from_dd(d["zh"], d["zl"])
        O5 = np.vectorize(lambda v: float(v.mid()))(omega5(rm, z))
        far = np.abs(rm.xi) > 0.05
        h0 = O5.mean(axis=0)
        msg = (f"{path.rsplit('/', 1)[-1]}: D {D} {Nt}x{N}: max|Omega_5^+| {np.abs(O5).max():.2e}, "
               f"max|Omega_5^+/xi| (|xi| > 0.05) {np.abs(O5[:, far] / rm.xi[far]).max():.2e}, harmonic 0 "
               f"{np.abs(h0).max():.2e} (at xi = +1 {h0[0]:+.2e}, -1 {h0[-1]:+.2e})")
        key = (D, Nt, N)
        if key in first:
            e = mps.amax(mps.mids(z - first[key][1]))
            dO = np.abs(O5 - first[key][0])
            msg += (f"; against the first state of this grid: max|z - z_1| {e:.1e}, max|Omega_5 - Omega_5(z_1)| "
                    f"{dO.max():.1e}, harmonic 0 {np.abs(dO.mean(axis=0)).max():.1e}")
        else:
            first[key] = (O5, z)
        if a.mutate == "scale":
            Om = np.vectorize(lambda v: float(v.mid()))(omega5(rm, z, arb(1) + arb("1e-6")))
            rise = np.abs(Om).max() - np.abs(O5).max()
            msg += f"; mutated: {np.abs(Om).max():.2e} (rise {rise:.1e} >= 1e-9: {rise >= 1e-9})"
            ok &= rise >= 1e-9
        print(msg, flush=True)
    print(f"exit {0 if ok else 1}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
