"""Quantitative Dingle check for the linear 1/n series y = x 2F1((1+s)/2,(2+s)/2;(n+1)/2;x^2):
y_k(x) ~ K Q(x) Gamma(k-g) chi^(g-k),  chi = Phi - i pi/2,  g = -(s+1/2),  Q = x^2 (1-x^2)^(-1-s),
with the inner (x ~ sqrt n, 2F0) prediction K = e^{-i pi (1+s)} 2^{-s-1/2} / (Gamma(a) Gamma(b)).
Measured K_k(x) at several k, Richardson-extrapolated in 1/k."""
import mpmath as mp
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "largeD_linear_late.py")).read().split('if __name__')[0]
exec(src)
g = -(s + mp.mpf(1) / 2)
Kpred = mp.expj(-mp.pi * (1 + s)) * mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b))
print("K predicted:", mp.nstr(Kpred, 10))
for xs in ("0.6", "0.70710678118654752", "0.8", "0.9", "0.95"):
    x = mp.mpf(xs)
    chi = mp.log(x) - mp.log(1 - x * x) / 2 - 1j * mp.pi / 2
    Q = x ** 2 * (1 - x * x) ** (-1 - s)
    ks = list(range(K - 30, K + 1, 5))
    Kk = [yk(k, x) * chi ** (k - g) / (mp.gamma(k - g) * Q) for k in ks]
    # Richardson: fit Kk = K0 + c1/k + c2/k^2 + c3/k^3 on the last 4 points
    pts = list(zip(ks, Kk))[-4:]
    Mx = mp.matrix([[1, 1 / mp.mpf(k), 1 / mp.mpf(k) ** 2, 1 / mp.mpf(k) ** 3] for k, _ in pts])
    K0 = mp.lu_solve(Mx, mp.matrix([v for _, v in pts]))[0]
    print(f"x {mp.nstr(x, 5):>7s}: K_k at k={ks[0]}..{ks[-1]}: " + ", ".join(mp.nstr(v / Kpred, 5) for v in Kk) +
          f"   (all / K_pred);  Richardson K0/K_pred = {mp.nstr(K0 / Kpred, 8)}", flush=True)
