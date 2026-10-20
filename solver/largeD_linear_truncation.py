"""does the regular solution contain the fast mode on 0 < x < 1? Compare the exact x 2F1 with the 1/n series
truncated near its smallest term; a fast-mode content C*(singular Frobenius branch) would leave a floor of that size."""
import os
import sys
import mpmath as mp
sys.argv = ["x", "122"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "largeD_linear_late.py")).read().split('if __name__')[0])
for xs in ("0.8", "0.9"):
    x = mp.mpf(xs); z = x * x
    ys = [yk(k, x) for k in range(0, 100)]
    for n in (30, 40, 60):
        eps = mp.mpf(1) / n
        c = (mp.mpf(n) + 1) / 2
        exact = x * mp.hyp2f1(a, b, c, z)
        part = mp.mpf(0); best = None
        for k in range(len(ys)):
            part += ys[k] * eps ** k
            d = abs(exact - part)
            if best is None or d < best[0]:
                best = (d, k)
        lam = c - a - b
        Cl = mp.gamma(c) * mp.gamma(a + b - c) / (mp.gamma(a) * mp.gamma(b))
        sing = abs(Cl * x * (1 - z) ** lam * mp.hyp2f1(c - a, c - b, lam + 1, 1 - z))
        chi = mp.log(x) - mp.log(1 - z) / 2 - 1j * mp.pi / 2
        print(f"x {xs} n {n}: min |exact - truncated| {mp.nstr(best[0], 3)} at K={best[1]}   |C x sing. branch| {mp.nstr(sing, 3)}"
              f"   e^(-n|chi|) {mp.nstr(mp.exp(-n * abs(chi)), 3)}   e^(-n Re chi) {mp.nstr(mp.exp(-n * chi.real), 3)}", flush=True)
