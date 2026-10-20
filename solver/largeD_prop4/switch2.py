"""Dominance switch near the light cone, at high precision (dps 400): two-ratio chi estimates of the exact linear
late orders at x = 0.999, 0.9999 (|Phi - i pi/2| > pi) against x = 0.99 (< pi)."""
import mpmath as mp
mp.mp.dps = 400
K = 124
w = 2 * mp.pi / mp.mpf("1.64389375"); s = 1j * w; a, b = (1 + s) / 2, (2 + s) / 2
e = [[mp.mpf(1)] + [mp.mpf(0)] * K]
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
yk = lambda k, x: sum(T[j] * 2 ** j * x ** (2 * j + 1) * e[j][k - j] for j in range(k + 1))
for xs in ("0.99", "0.999", "0.9999"):
    x = mp.mpf(xs); Phi = mp.log(x) - mp.log(1 - x * x) / 2
    ys = {k: yk(k, x) for k in range(98, K + 1)}
    for k in (100, 110, 120):
        r0, r1 = ys[k + 1] / ys[k], ys[k + 2] / ys[k + 1]
        chi = 1 / (r1 - r0); g = k - chi * r0
        print(f"x {xs}: Phi - i pi/2 = {mp.nstr(Phi - 1j*mp.pi/2, 5)} (|.| {mp.nstr(abs(Phi - 1j*mp.pi/2), 4)}); k {k}: chi {mp.nstr(chi, 6)}  g {mp.nstr(g, 4)}", flush=True)
