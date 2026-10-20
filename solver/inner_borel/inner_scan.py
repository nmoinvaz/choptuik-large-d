"""Scan the fast-type Stokes data S_k(A, omega) of the Galerkin-truncated inner problem (F = A sin theta, harmonics
|mu| <= Mmax) for zeros. Uses inner_series.series and the Darboux fit of inner_stokes.py with the transport exponents.

    python3 inner_scan.py omega A0 w_lo w_hi n_w [Mmax J]     # scan omega/omega_inf at fixed A
    python3 inner_scan.py A     wf  A_lo A_hi n_A [Mmax J]    # scan A at fixed omega = wf * omega_inf
"""
import sys
import mpmath as mp
from inner_series import series
from inner_stokes import transport_modes, fit, to_mp, W0


def stokes(A, w, Mmax=1, J=200, nc=8, mus=(1, -1)):
    Fh = {1: -0.5j * A, -1: 0.5j * A}
    d, _, _ = series(Fh, w, J, M=Mmax + 1, prec=900)
    nus, E, V = transport_modes(Fh, w, Mmax)
    gam = [mp.mpf(1) / 2 - E[k] / 2 for k in range(len(nus))]
    order = sorted(range(len(nus)), key=lambda k: (mp.im(gam[k]), mp.re(gam[k])))
    out = []
    for mu in mus:
        dm = [to_mp(d[j][mu + Mmax + 1]) for j in range(J + 1)]
        S, _ = fit(dm, [gam[k] for k in order], J - 70, J, nc)
        out.append([S[i] for i, k in enumerate(order)])   # S_k v_k,mu: independent of the eigenvector normalization
    return [gam[k] for k in order], out


if __name__ == "__main__":
    what = sys.argv[1]; fixed = mp.mpf(sys.argv[2]); lo, hi, nn = mp.mpf(sys.argv[3]), mp.mpf(sys.argv[4]), int(sys.argv[5])
    Mmax = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    J = int(sys.argv[7]) if len(sys.argv) > 7 else 200
    for i in range(nn):
        v = lo + (hi - lo) * i / max(1, nn - 1)
        A, w = (fixed, v * W0) if what == "omega" else (v, fixed * W0)
        gam, S = stokes(A, w, Mmax, J)
        cells = "  ".join(f"g {mp.nstr(g, 5):>20s} S(mu=1) {mp.nstr(S[0][k], 6):>26s} S(mu=-1) {mp.nstr(S[1][k], 6):>26s}"
                          for k, g in enumerate(gam))
        print(f"A {mp.nstr(A, 5):>6s} w/w_inf {mp.nstr(w / W0, 5):>6s}: {cells}", flush=True)
