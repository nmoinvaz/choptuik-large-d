"""C(n,s) = -2 pi i K' (n+1)^(s+1/2) e^{-i pi lambda} (1 + O(s^2/n)),  K' = 2^(-s-1/2)/(Gamma(a)Gamma(b)) (branch-free inner constant)."""
import mpmath as mp
mp.mp.dps = 40
w = 2 * mp.pi / mp.mpf("1.64389375")
for mu in (1, 3):
    s = 1j * mu * w; a, b = (1 + s) / 2, (2 + s) / 2
    Kp = mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b))
    for n in ("18", "58", "58.5", "200", "1000.3", "10000", "100000"):
        n = mp.mpf(n); kap = (n + 1) / 2; lam = n / 2 - 1 - s
        C = mp.gamma(kap) * mp.gamma(s + 1 - n / 2) / (mp.gamma(a) * mp.gamma(b))
        R = C / (2j * mp.pi * Kp * (n + 1) ** (s + mp.mpf(1) / 2) * mp.expj(-mp.pi * lam))
        print(f"mu {mu} n {mp.nstr(n,7):>8s}: R = {mp.nstr(R, 8):>28s}  n(R+1) = {mp.nstr(n * (R + 1), 5)}")
