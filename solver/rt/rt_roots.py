"""Two converged roots of one RT grid at one D (rt_run.py states solved from different starts): their difference, its
content by field, harmonic and xi, the two Deltas, and the bound it gives on the Jacobian along that difference.

    python3 solver/rt/rt_roots.py A.npz B.npz [--prec 192] [--mutate=same]

For the quadratic residual F(z2) - F(z1) = J(zm) e exactly, with zm = (z1 + z2) / 2 and e = z2 - z1, so
max|J(zm) e| <= max|F(z1)| + max|F(z2)|, and the ratio of that sum to max|e| bounds how weakly J(zm) acts along e
(in the max norm). The residuals and Deltas are evaluated in arb at --prec bits from the states' double-double parts.
Planted check first: a state against itself plus a planted e of size 1e-20 along mu must give max|e| = 1e-20 to 1e-30
and a Delta difference of 4 pi 1e-20; exits 1 otherwise. --mutate=same compares A with itself, so the check of a
non-zero difference must then fail (exit 1).
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


def load(path):
    d = np.load(path)
    return mps.from_dd(d["zh"], d["zl"] if "zl" in d else np.zeros_like(d["zh"])), float(d["D"]), int(d["Nt"]), int(d["N"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--mutate", default=None, choices=["same"])
    a = ap.parse_args()
    flint.ctx.prec = a.prec                     # before from_dd, which would round hi + lo at the default 53 bits
    za, D, Nt, N = load(a.a)
    zb, Db, Ntb, Nb = load(a.b if a.mutate != "same" else a.a)
    if (D, Nt, N) != (Db, Ntb, Nb):
        print(f"the states differ in D or grid: {(D, Nt, N)} and {(Db, Ntb, Nb)}; exit 1")
        sys.exit(1)
    rm = RTmp(D, Nt, N, prec=a.prec)
    # planted: z and z + 1e-20 along mu
    zp = za.copy()
    zp[-1] = zp[-1] + arb("1e-20")
    ep = mps.amax(mps.mids(zp - za))
    dp = float((rm.delta_m(zp)[0] - rm.delta_m(za)[0]).mid())
    ok_p = abs(ep - 1e-20) < 1e-30 and abs(dp - 4 * np.pi * 1e-20) < 1e-30
    print(f"planted e = 1e-20 along mu: max|e| {ep:.3e}, Delta difference {dp:.4e} (4 pi 1e-20 = {4 * np.pi * 1e-20:.4e}):"
          f" {'ok' if ok_p else 'FAIL'}")
    e = mps.mids(zb - za)
    me = mps.amax(e)
    Fa, Fb = mps.amax(rm.residual_m(za)), mps.amax(rm.residual_m(zb))
    da, db = rm.delta_m(za)[0], rm.delta_m(zb)[0]
    print(f"D {D} grid {Nt}x{N}: max|F(z1)| {Fa:.2e}, max|F(z2)| {Fb:.2e}, max|z2 - z1| {me:.3e}, "
          f"Delta_1 {da.mid().str(22, radius=False)}, Delta_2 {db.mid().str(22, radius=False)}, "
          f"Delta_2 - Delta_1 {float((db - da).mid()):+.3e}")
    if me > 0:
        print(f"  max|J(zm) e| / max|e| <= (max|F(z1)| + max|F(z2)|) / max|e| = {(Fa + Fb) / me:.1e}")
    ef = mps.to_f64(e)
    w1, w2, w3, w4, mu = rm.unpack(ef)
    print(f"  mu: {float(mu[0, 0, 0]):+.3e}")
    for nm, f in (("omega_1", w1[0]), ("omega_2", w2[0]), ("omega_3", w3[0]), ("omega_4", w4[0])):
        amp = np.abs(np.fft.fft(f, axis=0) / Nt).max(axis=1)
        col = np.abs(f).max(axis=0)
        print(f"  {nm}: max {np.abs(f).max():.2e}; harmonics 0-3 {', '.join(f'{v:.1e}' for v in amp[:4])}; at xi = 1 "
              f"{col[0]:.1e}, xi near 0 {col[len(col) // 2]:.1e}, xi = -1 {col[-1]:.1e}")
    ok = ok_p and me > 0
    print(f"exit {0 if ok else 1}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
