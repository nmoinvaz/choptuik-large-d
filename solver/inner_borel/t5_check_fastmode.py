"""Check the fast-mode u-series against the correction terms of a free Darboux fit (M = 1, A = 2.72, omega_inf)."""
import mpmath as mp
import inner_fastmode as FM
from inner_stokes import fit, to_mp
mp.mp.dps = 120
J = 260
d, modes, Mst, odd = FM.run(2.72, 1, J, mmax=12)
gam = [mp.mpc(mp.mpf(g.real.mid().str(100, radius=False)), mp.mpf(g.imag.mid().str(100, radius=False))) for _, g, _ in modes]
mu = 1
dm = [to_mp(d[j][mu + Mst]) for j in range(J + 1)]
import sys; nc = int(sys.argv[1]) if len(sys.argv) > 1 else 14
mp.mp.dps = 150
cols = [(k, m) for k in range(len(gam)) for m in range(nc + 1)]
rows, rhs = [], []
for j in range(J - 120, J + 1):
    sc = 1 / mp.gamma(j + mp.re(gam[0]))
    rows.append([mp.gamma(j + gam[k] - m) * sc for k, m in cols]); rhs.append(dm[j] / mp.mpf(2) ** j * sc)
Am = mp.matrix(rows); b = mp.matrix(rhs)
cn = [mp.sqrt(mp.fsum(abs(Am[i, c]) ** 2 for i in range(Am.rows))) for c in range(Am.cols)]
for c in range(Am.cols):
    for i in range(Am.rows):
        Am[i, c] /= cn[c]
y, res = mp.qr_solve(Am, b)
x = [y[c] / cn[c] for c in range(Am.cols)]
print("free fit relative residual", mp.nstr(res / mp.norm(b), 3))
for k, (kap, g, a) in enumerate(modes):
    a0 = to_mp(a[0][mu + Mst])
    print(f"mode {k}: gamma {mp.nstr(gam[k], 8)}")
    for m in range(0, 7):
        pred = to_mp(a[m][mu + Mst]) / a0 * mp.mpf(2) ** (-m)
        got = x[k * (nc + 1) + m] / x[k * (nc + 1)]
        print(f"   m {m}: fast-series a_m/a_0 2^-m {mp.nstr(pred, 12):>34s}   free-fit {mp.nstr(got, 12):>34s}   rel diff {mp.nstr(abs(got / pred - 1), 2)}")
