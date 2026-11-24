"""Why a free wave cannot echo, and how strongly the SSH rejects each harmonic (#1086, notes/LARGE_D_BASICS.md 3.4).

The flat-space self-similar wave phi = e^{lambda tau} f(x), regular at the centre, is
f = 2F1(a, b; c; z), a = lambda/2, b = (lambda + 1)/2, c = (n + 1)/2, z = x^2 (derive.py check 5). At the SSH z = 1,

  f = A 2F1(a, b; a+b-c+1; 1-z) + B (1-z)^(n/2 - lambda) 2F1(c-a, c-b; c-a-b+1; 1-z),
  A = Gamma(c) Gamma(c-a-b) / (Gamma(c-a) Gamma(c-b)),   B = Gamma(c) Gamma(a+b-c) / (Gamma(a) Gamma(b)),

and the second (singular) part is the fast mode F ~ (sqrt(1 - x^2)/x)^n. B = 0 only for lambda = 0, -1, -2, ...,
so no harmonic lambda = i j omega is regular at both ends. This prints, for harmonics j (lambda = i j omega) and n,
  |B/A|            the singular part's weight relative to the regular part
  R(x)             |singular part| / |regular part| at x (default x* = 1/sqrt 2, where F is O(1))
and fits ln R against j and ln n.

Usage: python3 free_field.py [--omega=3.73] [--x=0.70710678] [--mutate=A]
  --mutate=A: control; swaps A and B in the ratio, which must change the j-slope's sign.
"""
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 120


def series(a, b, c, z):
    """2F1 by its power series (|z| < 1), summed until the terms fall below 10^-110 of the largest, at a working
    precision of 120 + n digits since the connection series cancel by about n/4 digits. mpmath's hyp2f1 loses
    accuracy for the connection formula's parameters at n >= 240 (third parameter near -n/2)."""
    s, t, k, big = mp.mpf(0), mp.mpc(1), 0, mp.mpf(1)
    while True:
        s += t
        big = max(big, abs(t))
        t *= (a + k) * (b + k) / ((c + k) * (k + 1)) * z
        k += 1
        if abs(t) < mp.mpf(10) ** -110 * big and k > 4 * abs(c) + 50:  # past the second rise at k ~ 2|c|
            return s


def parts(n, lam, x):
    with mp.workdps(120 + n):
        return _parts(n, lam, x)


def _parts(n, lam, x):
    a, b, c = lam / 2, (lam + 1) / 2, mp.mpf(n + 1) / 2
    z = mp.mpf(x) ** 2
    A = mp.gamma(c) * mp.gamma(c - a - b) / (mp.gamma(c - a) * mp.gamma(c - b))
    B = mp.gamma(c) * mp.gamma(a + b - c) / (mp.gamma(a) * mp.gamma(b))
    reg = A * series(a, b, a + b - c + 1, 1 - z)
    sing = B * (1 - z) ** (c - a - b) * series(c - a, c - b, c - a - b + 1, 1 - z)
    whole = series(a, b, c, z)
    return A, B, reg, sing, whole


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    om = mp.mpf(opts.get("omega", "3.73"))
    x = mp.mpf(opts.get("x", "0.70710678118654752"))
    mutate = opts.get("mutate", "")
    js = [1, 3, 5, 7, 9, 11, 13]
    ns = [60, 120, 240, 480, 960]
    print("# omega %s x %s; |B/A| and R = |sing|/|reg| at x; connection check |reg + sing - f|/|f|" % (om, x))
    lnR = np.zeros((len(js), len(ns)))
    for i, j in enumerate(js):
        row = []
        for k, n in enumerate(ns):
            A, B, reg, sing, whole = parts(n, mp.mpc(0, j * om), x)
            if mutate == "A":
                A, B, reg, sing = B, A, sing, reg
            chk = abs(reg + sing - whole) / abs(whole)
            R = abs(sing) / abs(reg)
            lnR[i, k] = float(mp.log(R))
            row.append("n%d |B/A| %.2e R %.2e (chk %.0e)" % (n, abs(B / A), R, chk))
        print("j %2d  " % j + "  ".join(row))
    for k, n in enumerate(ns):
        sl = np.polyfit(js, lnR[:, k], 1)[0]
        print("n %d: d ln R / dj = %.3f  (-pi omega/2 = %.3f)" % (n, sl, -float(mp.pi * om / 2)))
    for i, j in enumerate(js):
        sl = np.polyfit(np.log(ns), lnR[i, :], 1)[0]
        print("j %d: d ln R / d ln n = %.3f" % (j, sl))
    # |sin(pi (i j omega - n/2))|^2 = sinh^2(pi j omega) + sin^2(pi n/2): the relative period-2 modulation in n
    print("relative modulation of |B/A| in n, 2 exp(-2 pi j omega): " +
          " ".join("j%d %.2e" % (j, float(2 * mp.e ** (-2 * mp.pi * j * om))) for j in (1, 3)))


if __name__ == "__main__":
    main()
