"""Consistency of the Euler-integral constant h_0 with the Dingle constant K = e^{-i pi(1+s)} 2^{-s-1/2}/(Gamma(a)Gamma(b))
(README, Proposition 4): ratio of the two leading-order predictions at large k, and Richardson of y_k/pred - 1 at x = 0.9."""
import mpmath as mp
exec(open("euler_late.py").read().split("for xs in")[0])
for xs in ("0.5", "0.8", "0.9"):
    x = mp.mpf(xs); z = x * x
    Phi = mp.log(x) - mp.log(1 - z) / 2
    t0 = Phi - 1j * mp.pi / 2
    h0, _ = h_const(t0, z)
    Kd = mp.expj(-mp.pi * (1 + s)) * mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b))
    Q = x ** 2 * (1 - z) ** (-1 - s)
    k = mp.mpf(10) ** 6
    mine = x * 2 ** (1 - b) / mp.gamma(b) * h0 / mp.gamma(a) * mp.exp(mp.loggamma(k + a) + mp.loggamma(k + b) - mp.loggamma(k + 1)) * t0 ** (-k)
    dingle = Kd * Q * mp.exp(mp.loggamma(k + s + mp.mpf(1) / 2)) * t0 ** (-(k + s + mp.mpf(1) / 2))
    print(f"x {xs}: (Euler-integral constant)/(README Dingle constant) at k = 1e6: {mp.nstr(mine / dingle, 10)}", flush=True)
x = mp.mpf("0.9"); z = x * x; Phi = mp.log(x) - mp.log(1 - z) / 2
taus = {l: Phi - 1j * mp.pi * (l + mp.mpf(1) / 2) for l in (0, -1)}
hs = {l: h_const(t, z)[0] for l, t in taus.items()}
ks = [100, 106, 112, 118, 122]
d = []
for k in ks:
    pred = sum(x * 2 ** (1 - b) / mp.gamma(b) * mp.gamma(k + b) * hs[l] * mp.gamma(k + a) / (mp.gamma(a) * mp.gamma(k + 1)) * taus[l] ** (-k) for l in (0, -1))
    d.append(yk(k, x) / pred - 1)
Mx = mp.matrix([[1, 1 / mp.mpf(k), 1 / mp.mpf(k) ** 2, 1 / mp.mpf(k) ** 3, 1 / mp.mpf(k) ** 4] for k in ks])
c = mp.lu_solve(Mx, mp.matrix(d))
print("x 0.9: y_k/pred - 1 = c0 + c1/k + ...: c0 =", mp.nstr(c[0], 4), " c1 =", mp.nstr(c[1], 5))
