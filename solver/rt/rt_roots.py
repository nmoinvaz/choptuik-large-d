"""Two converged roots of one RT grid at one D (rt_run.py states solved from different starts): their difference, its
content by field, harmonic and xi, the two Deltas, and the bound it gives on the Jacobian along that difference.

    python3 solver/rt/rt_roots.py A.npz B.npz [--prec 192] [--mutate=same|mid]

For the quadratic residual F(z2) - F(z1) = J(zm) e exactly, with zm = (z1 + z2) / 2 and e = z2 - z1, so
max|J(zm) e| <= max|F(z1)| + max|F(z2)|, and the ratio of that sum to max|e| bounds how weakly J(zm) acts along e
(in the max norm). The residuals and Deltas are evaluated in arb at --prec bits from the states' double-double parts.
Planted check first: a state against itself plus a planted e of size 1e-20 along mu must give max|e| = 1e-20 to 1e-30
and a Delta difference of 4 pi 1e-20; exits 1 otherwise. --mutate=same compares A with itself, so the check of a
non-zero difference must then fail (exit 1).

Issue #164 adds the pair test. F(zm) = (F(z1) + F(z2)) / 2 - B(e, e) / 4 exactly, so B(e, e) = 2 (F(z1) + F(z2) -
2 F(zm)), and J(z1) e = F(z2) - F(z1) - B(e, e), J(z2) e = F(z2) - F(z1) + B(e, e). When max|F(zm)| is far above both
roots' residuals, the residual rises between them: they are two distinct roots of the discrete system, and J at either
acts along e with max|B(e, e)| / max|e| (up to the residuals). Otherwise the residual floor of the stored states (about
1e-30) hides B(e, e). Its planted check: the pair z, z (1 + 1e-12) must give B(e, e) = 1e-24 B(z, z) to 1e-6, with
B(z, z) = (F(2 z) - 2 F(z) + F(0)) / 2 by another route. --mutate=mid takes zm = z1 + 0.45 e, and the planted check must
then fail (exit 1).

Issue #163: the residual is that of the discretization the states were solved with, their `horizon` K (rt_run.py
--horizon; 0 when the state has no such field), and two states with different K are refused.
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


def horizon_of(path):
    """The horizon rows K the state was solved with (issue #163; 0 for states without the field)."""
    d = np.load(path)
    return int(d["horizon"]) if "horizon" in d else 0


def pair_B(rm, z1, z2, s):
    """F(z1), F(z2), F(z1 + s e) and B(e, e) = 2 (F(z1) + F(z2) - 2 F(z1 + s e)) (exact for s = 1/2)."""
    F1, F2 = rm.residual_m(z1), rm.residual_m(z2)
    Fm = rm.residual_m(z1 + s * (z2 - z1))
    return F1, F2, Fm, 2 * (F1 + F2 - 2 * Fm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--mutate", default=None, choices=["same", "mid"])
    a = ap.parse_args()
    flint.ctx.prec = a.prec                     # before from_dd, which would round hi + lo at the default 53 bits
    za, D, Nt, N = load(a.a)
    zb, Db, Ntb, Nb = load(a.b if a.mutate != "same" else a.a)
    if (D, Nt, N) != (Db, Ntb, Nb):
        print(f"the states differ in D or grid: {(D, Nt, N)} and {(Db, Ntb, Nb)}; exit 1")
        sys.exit(1)
    K = horizon_of(a.a)
    if horizon_of(a.b if a.mutate != "same" else a.a) != K:
        print(f"the states were solved with different horizon rows; exit 1")
        sys.exit(1)
    print(f"horizon rows K = {K}")
    rm = RTmp(D, Nt, N, prec=a.prec, horizon=K)
    # planted: z and z + 1e-20 along mu
    zp = za.copy()
    zp[-1] = zp[-1] + arb("1e-20")
    ep = mps.amax(mps.mids(zp - za))
    dp = float((rm.delta_m(zp)[0] - rm.delta_m(za)[0]).mid())
    ok_p = abs(ep - 1e-20) < 1e-30 and abs(dp - 4 * np.pi * 1e-20) < 1e-30
    print(f"planted e = 1e-20 along mu: max|e| {ep:.3e}, Delta difference {dp:.4e} (4 pi 1e-20 = {4 * np.pi * 1e-20:.4e}):"
          f" {'ok' if ok_p else 'FAIL'}")
    # planted pair with a known B(e, e): z and z (1 + 1e-12), B(e, e) = 1e-24 B(z, z)
    s_mid = arb("0.45") if a.mutate == "mid" else arb("0.5")
    zero = np.array([arb(0)] * len(za), dtype=object)
    Bzz = (rm.residual_m(2 * za) - 2 * rm.residual_m(za) + rm.residual_m(zero)) / 2
    Bp = pair_B(rm, za, za + za * arb("1e-12"), s_mid)[3]
    want = Bzz * arb("1e-24")
    errp = mps.amax(Bp - want) / mps.amax(want)
    ok_pair = errp < 1e-6
    print(f"planted pair z, z (1 + 1e-12): max|B(e, e)| {mps.amax(Bp):.3e}, against 1e-24 B(z, z) {mps.amax(want):.3e}, "
          f"relative error {errp:.1e}: {'ok' if ok_pair else 'FAIL'}")
    e = mps.mids(zb - za)
    me = mps.amax(e)
    Fa, Fb = mps.amax(rm.residual_m(za)), mps.amax(rm.residual_m(zb))
    da, db = rm.delta_m(za)[0], rm.delta_m(zb)[0]
    print(f"D {D} grid {Nt}x{N}: max|F(z1)| {Fa:.2e}, max|F(z2)| {Fb:.2e}, max|z2 - z1| {me:.3e}, "
          f"Delta_1 {da.mid().str(22, radius=False)}, Delta_2 {db.mid().str(22, radius=False)}, "
          f"Delta_2 - Delta_1 {float((db - da).mid()):+.3e}")
    if me > 0:
        print(f"  max|J(zm) e| / max|e| <= (max|F(z1)| + max|F(z2)|) / max|e| = {(Fa + Fb) / me:.1e}")
    if me > 0:
        F1, F2, Fm, Bee = pair_B(rm, za, zb, s_mid)
        mFm, mB = mps.amax(Fm), mps.amax(Bee)
        pair = mFm > 100 * max(Fa, Fb)
        print(f"  pair: max|F(zm)| {mFm:.2e}, max|B(e, e)| {mB:.2e}, max|B(e, e)| / max|e| {mB / me:.1e} (J at either "
              f"root along e, up to the residuals): "
              f"{'the residual rises between them, two distinct roots' if pair else 'hidden by the residual floor'}")
    ef = mps.to_f64(e)
    w1, w2, w3, w4, mu = rm.unpack(ef)
    print(f"  mu: {float(mu[0, 0, 0]):+.3e}")
    for nm, f in (("omega_1", w1[0]), ("omega_2", w2[0]), ("omega_3", w3[0]), ("omega_4", w4[0])):
        amp = np.abs(np.fft.fft(f, axis=0) / Nt).max(axis=1)
        col = np.abs(f).max(axis=0)
        print(f"  {nm}: max {np.abs(f).max():.2e}; harmonics 0-3 {', '.join(f'{v:.1e}' for v in amp[:4])}; at xi = 1 "
              f"{col[0]:.1e}, xi near 0 {col[len(col) // 2]:.1e}, xi = -1 {col[-1]:.1e}")
    ok = ok_p and ok_pair and me > 0
    print(f"exit {0 if ok else 1}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
