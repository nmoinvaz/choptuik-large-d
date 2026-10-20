"""Poisson summation of the exact pole sum: y_k = sum_l S_l, S_l = 2x(-1)^(k-1) int_0^inf r(J) (2J+1)^(k-1)
e^{i pi (2l+1) J} dJ with r(J) = Res_c(J)/(-1)^J continued to real J. Each S_l carries singulant Phi + i pi (l + 1/2)."""
import mpmath as mp
mp.mp.dps = 30
w_ = 2 * mp.pi / mp.mpf("1.64389375"); s = 1j * w_
a, b = (1 + s) / 2, (2 + s) / 2
def r(J, z):
    return (mp.gamma(a + J + 1) * mp.gamma(b + J + 1) / (mp.gamma(a) * mp.gamma(b) * mp.gamma(J + 1) * mp.gamma(J + 2))
            * z ** (J + 1) * mp.hyp2f1(a + J + 1, b + J + 1, J + 2, z))
for xs in ("0.5", "0.6"):
    x = mp.mpf(xs); z = x * x
    for k in (30, 60):
        exact = 2 * x * (-1) ** (k - 1) * mp.fsum(r(J, z) * (-1) ** J * (2 * J + 1) ** (k - 1) for J in range(700))
        S = {}
        for l in (-2, -1, 0, 1):
            f = lambda J: r(J, z) * (2 * J + 1) ** (k - 1) * mp.expj(mp.pi * (2 * l + 1) * J)
            S[l] = 2 * x * (-1) ** (k - 1) * mp.quad(f, mp.linspace(0, 400, 401))
        tot = sum(S.values())
        print(f"x {xs} k {k}: exact {mp.nstr(exact, 10)}  sum_l S_l {mp.nstr(tot, 10)}  rel {mp.nstr(abs(tot/exact-1), 2)}  "
              + "  ".join(f"|S_{l}/exact| {mp.nstr(abs(S[l]/exact), 3)}" for l in S), flush=True)
