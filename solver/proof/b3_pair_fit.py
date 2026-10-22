"""Location of a dominant conjugate pair of Borel singularities of a real series (proof/B3_borel.md, Section 5).

For a real series sum c_k eps^k whose Borel coefficients d_k = c_k/(k-1)! are dominated by a conjugate pair a, conj(a),
d_k ~ K k^(g-1) a^(-k) + c.c., the d_k satisfy d_{k+1} = s_k d_k - p_k d_{k-1} with s_k -> 2 Re(1/a), p_k -> 1/|a|^2
and O(1/k) corrections. s_k, p_k are solved from four consecutive d's, a_k is the root of t^2 - s_k t + p_k (as 1/t),
and a_k is extrapolated in 1/k (Richardson, orders 1 and 2). Planted-truth mode: --planted=re,im (a pair at re +- i im
with g = 2.3 + 0.7i plus a weaker pair at twice the modulus) goes through the same code.

    python3 b3_pair_fit.py file.json g0|xstar|g1        (or --planted=0,3.14159265)
"""
import json
import sys

import mpmath as mp

mp.mp.dps = 80
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
POS = [a for a in sys.argv[1:] if not a.startswith("--")]


def series():
    if "planted" in ARGS:
        re, im = (mp.mpf(v) for v in ARGS["planted"].split(","))
        a = mp.mpc(re, im)
        g = mp.mpc("2.3", "0.7")
        b2 = 2 * a * mp.expjpi(mp.mpf("0.13"))
        K = int(ARGS.get("K", 120))
        return [2 * mp.re(mp.gamma(k + g) / a ** (k + g) + mp.mpf("0.3") * mp.gamma(k + 1) / b2 ** (k + 1)) for k in range(1, K + 1)]
    d = json.load(open(POS[0]))
    return [mp.mpf(re) for re, im, r in d[POS[1]]]


def main():
    c = series()
    d = [c[k] / mp.factorial(k) for k in range(len(c))]   # c[k] is order k+1: zeta^k / k!
    rows = []
    for k in range(len(d) - 40, len(d) - 2):
        A = mp.matrix([[d[k], -d[k - 1]], [d[k + 1], -d[k]]])
        rhs = mp.matrix([d[k + 1], d[k + 2]])
        try:
            s, p = mp.lu_solve(A, rhs)
        except ZeroDivisionError:
            continue
        t = (s + mp.sqrt(s * s - 4 * p)) / 2
        rows.append((k, 1 / t))
    # Richardson in 1/k on the averaged a_k (average over a window removes the beat oscillation)
    def avg(i, w=4):
        return sum(r[1] for r in rows[i - w:i + 1]) / (w + 1)
    print(" k     a_k (raw)                      |a_k|     arg/pi")
    for k, a in rows[-8:]:
        print(f" {k:3d}  {mp.nstr(a, 12):30s}  {mp.nstr(abs(a), 8)}  {mp.nstr(mp.arg(a) / mp.pi, 6)}")
    n = len(rows)
    k1, k2 = rows[n - 1][0], rows[n - 11][0]
    a1, a2 = avg(n - 1), avg(n - 11)
    rich = (k1 * a1 - k2 * a2) / (k1 - k2)
    print(f"window-averaged a at k = {k2}, {k1}: {mp.nstr(a2, 10)}, {mp.nstr(a1, 10)}")
    print(f"Richardson (1/k) extrapolation: a = {mp.nstr(rich, 10)}   |a| {mp.nstr(abs(rich), 8)}  arg/pi "
          f"{mp.nstr(mp.arg(rich) / mp.pi, 6)}")
    if "planted" in ARGS:
        re, im = (mp.mpf(v) for v in ARGS["planted"].split(","))
        err = abs(rich - mp.mpc(re, abs(im))) if mp.im(rich) > 0 else abs(rich - mp.mpc(re, -abs(im)))
        print(f"planted error: {mp.nstr(err, 3)}")


main()
