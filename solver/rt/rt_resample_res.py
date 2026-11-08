"""The residual of a grid-N root on finer grids N' after an exact resampling in xi (arb), by block and by xi: how far
the discrete root is from a root of the continuum equations (issue #163).

    python3 solver/rt/rt_resample_res.py STATE.npz 141,181 [--prec 192] [--horizon K] [--mutate=perturb]

The state's fields are degree-N polynomials in xi; on a finer Lobatto grid their values follow by barycentric
interpolation, computed here in arb from the double-double state (rt_model.resample_to works in f64 and leaves a
rounding residual of about 1e-13 at xi = +1). The finer grid's collocated residual (with K = 0, the equations
themselves) then samples the continuum residual of those polynomials: on points between the grid's, F(z) is what the
grid did not impose. Planted check: the state's own grid (N' = N) must give its stored residual (max|F| below 1e-25),
and --mutate=perturb adds 1e-20 to one interior value of omega_2 before resampling, which must raise max|F| on N' > N
above 1e-18 (exit 1 otherwise).
"""
import os
import sys
import argparse
import numpy as np
import flint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_mp import RTmp, interp_matrix
import rt_mpsolve as mps

arb = flint.arb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("grids")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--mutate", default=None, choices=["perturb"])
    a = ap.parse_args()
    flint.ctx.prec = a.prec
    d = np.load(a.state)
    D, Nt, N = float(d["D"]), int(d["Nt"]), int(d["N"])
    src = RTmp(D, Nt, N, prec=a.prec)
    z = mps.from_dd(d["zh"], d["zl"])
    w1, w2, w3, w4, mu = src.unpack_m(z)
    if a.mutate == "perturb":
        w2 = w2.copy()
        w2[3, N // 3] = w2[3, N // 3] + arb("1e-20")
    ok = True
    for Np in [int(v) for v in a.grids.split(",")]:
        tgt = RTmp(D, Nt, Np, prec=a.prec, horizon=a.horizon)
        M = interp_matrix(list(src.xm), list(tgt.xm), N)
        f = [g.dot(M.T) for g in (w1, w2, w3, w4)]
        Z = tgt.pack(f[0], f[1], f[2], f[3], mu)
        F = mps.to_f64(tgt.residual_m(Z))
        h = len(tgt.pos)
        blocks = [("E1", F[:Nt * h].reshape(Nt, h), tgt.xi[tgt.pos])]
        o = Nt * h
        for nm in ("O2", "O3", "O4"):
            blocks.append((nm, F[o:o + tgt.nf].reshape(Nt, Np + 1), tgt.xi))
            o += tgt.nf
        mx = np.abs(F).max()
        print(f"N' {Np} (K {a.horizon}): max|F| {mx:.2e}, phase row {F[-1]:+.1e}")
        for nm, B, xs in blocks:
            col = np.abs(B).max(axis=0)
            j = int(np.argmax(col))
            print(f"   {nm}: max {col.max():.2e} at xi {xs[j]:+.4f}; by xi " +
                  " ".join(f"{xs[i]:+.2f}:{col[i]:.0e}" for i in range(0, len(xs), max(1, len(xs) // 10))))
        if Np == N and a.mutate is None:
            ok &= mx < 1e-25
        if Np > N and a.mutate == "perturb":
            ok &= mx > 1e-18
    print(f"exit {0 if ok else 1}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
