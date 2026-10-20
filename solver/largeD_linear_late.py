"""Late orders of the 1/n expansion of the exact leading-order solution (Proposition 2 of the README):

    y = x 2F1((1+s)/2, (2+s)/2; (n+1)/2; x^2) = sum_j T_j x^(2j+1) (2 eps)^j / prod_{i<j} (1 + (2i+1) eps),
    T_j = ((1+s)/2)_j ((2+s)/2)_j / j!,   eps = 1/n,   s = i omega (fundamental harmonic).

y_k(x), the coefficient of eps^k, is a polynomial in x. If the series is Gevrey-1 of Dingle type,
y_k ~ K Q(x) Gamma(k - g) / chi(x)^(k - g), then two consecutive ratios r_k = y_{k+1}/y_k give
chi = 1 / (r_{k+1} - r_k) and g = k - chi r_k. The prediction (README, "the light-cone coefficient is a Stokes
multiplier"): chi = Phi(x) - i pi/2 with Phi = ln x - ln(1 - x^2)/2, and g = -(s + 1/2).

    python3 largeD_linear_late.py [K]
"""
import sys
import mpmath as mp

mp.mp.dps = 90
K = int(sys.argv[1]) if len(sys.argv) > 1 else 122
w = 2 * mp.pi / mp.mpf("1.64389375")
s = 1j * w
a, b = (1 + s) / 2, (2 + s) / 2
e = [[mp.mpf(1)] + [mp.mpf(0)] * K]          # e[j][m]: eps^m coefficient of prod_{i<j} 1/(1 + (2i+1) eps)
for j in range(1, K + 1):
    prev, q = e[-1], 2 * j - 1
    cur, acc = [], mp.mpf(0)
    for m in range(K + 1):
        acc = prev[m] - q * acc if m else prev[0]
        cur.append(acc)
    e.append(cur)
T = [mp.mpf(1)]
for j in range(1, K + 1):
    T.append(T[-1] * (a + j - 1) * (b + j - 1) / j)


def yk(k, x):
    return sum(T[j] * 2 ** j * x ** (2 * j + 1) * e[j][k - j] for j in range(k + 1))


if __name__ == "__main__":
    print(f"s = {mp.nstr(s, 8)}, predicted g = -(s + 1/2) = {mp.nstr(-(s + 0.5), 6)}")
    for xs in ("0.5", "0.6", "0.70710678118654752", "0.8", "0.9"):
        x = mp.mpf(xs)
        Phi = mp.log(x) - mp.log(1 - x * x) / 2
        ys = {k: yk(k, x) for k in range(K - 22, K + 1)}
        for k in (K - 22, K - 12, K - 2):
            r0, r1 = ys[k + 1] / ys[k], ys[k + 2] / ys[k + 1]
            chi = 1 / (r1 - r0)
            g = k - chi * r0
            print(f"x {mp.nstr(x, 5):>7s}  k {k}: chi {mp.nstr(chi, 7):>28s}  predicted {mp.nstr(Phi - 1j * mp.pi / 2, 7):>28s}"
                  f"  |diff| {mp.nstr(abs(chi - Phi + 1j * mp.pi / 2), 2):>8s}   g {mp.nstr(g, 5)}", flush=True)
