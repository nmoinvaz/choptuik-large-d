"""inner_Mconv.py with the late orders of each mode normalized by its exact linear (2F0) form, as agent C suggests:
basis_{k,m}(j) = 2^j Gamma(j + a_k - delta/2 - m) Gamma(j + b_k - delta/2) / Gamma(j + 1), with s_k = gamma_k - 1/2 + delta,
a_k = (1 + s_k)/2, b_k = (2 + s_k)/2, delta = <F^2>. These carry the (nu w)^2/j corrections exactly, so high modes are
represented by a few terms even where (nu w)^2 > j.

    python3 inner_Mconv2.py "{1: -1.36j, -1: 1.36j, 3: ..., -3: ...}" wf "3 5 7 9" [J] [nc]
"""
import sys
import mpmath as mp
from inner_series import series
from inner_stokes import transport_modes, to_mp, W0


def lsq(dm, gam, delta, j0, j1, nc, dps=160):
    old = mp.mp.dps; mp.mp.dps = dps
    cols = [(k, m) for k in range(len(gam)) for m in range(nc + 1)]
    ab = []
    for g in gam:
        s = g - mp.mpf(1) / 2 + delta
        ab.append(((1 + s) / 2 - delta / 2, (2 + s) / 2 - delta / 2))
    ref = max(mp.re(g) for g in gam)
    rows, rhs = [], []
    for j in range(j0, j1 + 1):
        sc = 1 / mp.gamma(j + ref)
        rows.append([mp.gamma(j + ab[k][0] - m) * mp.gamma(j + ab[k][1]) / mp.gamma(j + 1) * sc for k, m in cols])
        rhs.append(dm[j] / mp.mpf(2) ** j * sc)
    Am = mp.matrix(rows); b = mp.matrix(rhs)
    cn = [mp.sqrt(mp.fsum(abs(Am[i, c]) ** 2 for i in range(Am.rows))) for c in range(Am.cols)]
    for c in range(Am.cols):
        for i in range(Am.rows):
            Am[i, c] /= cn[c]
    y, res = mp.qr_solve(Am, b)
    x = [y[c] / cn[c] for c in range(Am.cols)]
    mp.mp.dps = old
    return [x[cols.index((k, 0))] for k in range(len(gam))], res / mp.norm(b)


if __name__ == "__main__":
    Fh = {int(k): complex(v) for k, v in eval(sys.argv[1]).items()}
    wf = mp.mpf(sys.argv[2]); Ms = [int(v) for v in sys.argv[3].split()]
    J = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    nc = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    w = W0 * wf
    delta = mp.fsum(abs(mp.mpc(v)) ** 2 for v in Fh.values())          # <F^2> = sum |F_mu|^2
    print(f"F {Fh}, omega/omega_inf {wf}, <F^2> {mp.nstr(delta, 8)}", flush=True)
    for M in Ms:
        d, _, _ = series(Fh, w, J, M=M + 1, prec=1200)
        nus, E, V = transport_modes(Fh, w, M)
        gam = [mp.mpf(1) / 2 - E[k] / 2 for k in range(len(nus))]
        lab = [nus[max(range(len(nus)), key=lambda i: abs(V[i, k]))] for k in range(len(nus))]
        j0 = J - 2 * len(nus) * (nc + 1) - 20
        for mu in (1, 3):
            if mu > M:
                continue
            dm = [to_mp(d[j][mu + M + 1]) for j in range(J + 1)]
            X, rel = lsq(dm, gam, delta, j0, J, nc)
            row = "  ".join(f"nu {lab[k]:+d}: {mp.nstr(X[k], 8)}" for k in sorted(range(len(lab)), key=lambda k: lab[k]) if abs(lab[k]) <= 3)
            print(f"M {M:2d} harmonic {mu} (fit j {j0}..{J}, residual {mp.nstr(rel, 2)}): {row}", flush=True)
