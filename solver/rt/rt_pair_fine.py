"""Two roots of one RT grid on finer grids (issue #163): J along their difference, and the difference's and the
background's content by harmonic and Chebyshev mode.

    python3 solver/rt/rt_pair_fine.py A.npz B.npz GRIDS [--prec 192] [--mutate=f64]

GRIDS is a list of N' (e.g. 121,141,181). Both states are resampled in xi exactly (rt_mp.resample_xi_m, arb), and on
each grid the script prints max|F(z1)|, max|F(z2)| and max|J(z1) e| / max|e| with e = z2 - z1, J(z1) e = F(z2) - F(z1)
- B(e, e) and B(e, e) = 2 (F(z1) + F(z2) - 2 F(zm)) exact for the quadratic residual. A finer grid samples the
continuum residual of the states' polynomial fields between the old points, so a value that does not change with N'
belongs to the equations, not to the grid. Then for e and for z1: the largest amplitude over xi of each t harmonic of
each field, and the Chebyshev coefficients (arb) of harmonic 0 of omega_2, omega_3 and omega_4 (cos t/2 projection).
The residual is that of the states' own horizon rows (their `horizon`, 0 when absent).
Control: --mutate=f64 resamples with rt_model.resample_to in f64 instead; max|F(z1)| on the finer grids must then exceed
1e-20, against the double-double floor of the exact resampling (exit 1 when it does not).
"""
import os
import sys
import argparse
import numpy as np
import flint
import mpmath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT
from rt_mp import RTmp, resample_xi_m
import rt_mpsolve as mps

arb = flint.arb


def cheb_coeffs(vals, N):
    """Chebyshev coefficients of the Lobatto values (arb), by the discrete cosine sum at 200 bits."""
    with mpmath.workprec(200):
        v = [mpmath.mpf(x.mid().str(60, radius=False)) for x in vals]
        out = []
        for m in range(N + 1):
            s = mpmath.fsum((v[j] * (mpmath.mpf(1) / 2 if j in (0, N) else 1)) * mpmath.cos(mpmath.pi * j * m / N)
                            for j in range(N + 1))
            out.append(float(s * 2 / N * (mpmath.mpf(1) / 2 if m in (0, N) else 1)))
    return np.array(out)


def content(rt, rm, z, label):
    w1, w2, w3, w4, mu = rm.unpack_m(z)
    f64 = lambda g: np.vectorize(lambda x: float(x.mid()))(g)
    print(f"  {label}: mu {float(mu.mid()):+.3e}")
    for nm, g, anti in (("w1", w1, False), ("w2", w2, False), ("w3", w3, False), ("w4", w4, True)):
        f = f64(g)
        h = f * (np.exp(-0.5j * rt.t)[:, None] if anti else 1)
        H = np.abs(np.fft.fft(h, axis=0) / rt.Nt).max(axis=1)
        print(f"    {nm}: harmonics 0..{rt.Nt // 2} max over xi " + " ".join(f"{x:.0e}" for x in H[:rt.Nt // 2 + 1]))
    N = rt.N
    for nm, g, anti in (("w2", w2, False), ("w3", w3, False), ("w4", w4, True)):
        wt = [arb(1) / rt.Nt] * rt.Nt if not anti else [(rm.tm[a] / 2).cos() / rt.Nt for a in range(rt.Nt)]
        h0 = [sum((g[a, j] * wt[a] for a in range(rt.Nt)), arb(0)) for j in range(N + 1)]
        c = cheb_coeffs(h0, N)
        print(f"    {nm} harmonic {'1/2 (cos)' if anti else 0}: Chebyshev |c_m| m = 0, 10, ...: " +
              " ".join(f"{abs(x):.0e}" for x in c[::10]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("grids")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--mutate", default=None, choices=["f64"])
    a = ap.parse_args()
    flint.ctx.prec = a.prec
    da, db = np.load(a.a), np.load(a.b)
    D, Nt, N = float(da["D"]), int(da["Nt"]), int(da["N"])
    assert (float(db["D"]), int(db["Nt"]), int(db["N"])) == (D, Nt, N), "the states differ in D or grid"
    K = int(da["horizon"]) if "horizon" in da else 0
    print(f"{a.a.rsplit('/', 1)[-1]} and {a.b.rsplit('/', 1)[-1]}: D {D} {Nt}x{N}, horizon rows K {K}, mutate {a.mutate}")
    ok = True
    for Np in [int(v) for v in a.grids.split(",")]:
        rm = RTmp(D, Nt, Np, prec=a.prec, horizon=K)
        if a.mutate == "f64" and Np != N:
            src, tgt = RT(D, Nt, N), RT(D, Nt, Np)
            z1 = mps.from_dd(src.resample_to(da["zh"], tgt), src.resample_to(da["zl"], tgt))
            z2 = mps.from_dd(src.resample_to(db["zh"], tgt), src.resample_to(db["zl"], tgt))
        elif Np == N:
            z1, z2 = mps.from_dd(da["zh"], da["zl"]), mps.from_dd(db["zh"], db["zl"])
        else:
            z1 = resample_xi_m(da["zh"], da["zl"], D, Nt, N, Np, prec=a.prec)
            z2 = resample_xi_m(db["zh"], db["zl"], D, Nt, N, Np, prec=a.prec)
        e = mps.mids(z2 - z1)
        F1, F2 = rm.residual_m(z1), rm.residual_m(z2)
        Fm = rm.residual_m(z1 + e / 2)
        Bee = mps.mids(2 * (F1 + F2 - 2 * Fm))
        J1e = mps.mids(F2 - F1 - Bee)
        me = mps.amax(e)
        m1 = mps.amax(F1)
        print(f"  N' {Np}: max|F(z1)| {m1:.2e}, max|F(z2)| {mps.amax(F2):.2e}, max|e| {me:.3e}, max|B(e, e)| "
              f"{mps.amax(Bee):.2e}, max|J(z1) e| / max|e| {mps.amax(J1e) / me:.3e}", flush=True)
        if Np != N:
            ok &= (m1 > 1e-20) if a.mutate == "f64" else True
        if Np == N:
            rt = RT(D, Nt, N, horizon=K)
            content(rt, rm, e, "e = z2 - z1")
            content(rt, rm, z1, "z1 (background)")
    print(f"exit {0 if ok else 1}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
