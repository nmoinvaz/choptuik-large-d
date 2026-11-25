"""How well do Chebyshev polynomials of degree N contain the horizon branch (1 - xi)^sigma on [-1, 1]? (#1089)

A model for the question of #1086 section 6.1: does a fixed Chebyshev degree N (80 in arXiv:2611.07007) impose
analyticity at the horizon ever more weakly once sigma passes N? The exact Chebyshev coefficients are

    a_k = 2^(1 - sigma) (-1)^k Gamma(2 sigma + 1) / (Gamma(sigma + k + 1) Gamma(sigma - k + 1)),  k >= 1 (a_0 half),

and E(sigma, N) = sum_{k > N} |a_k| / sum_k |a_k| is the l1 share (RT's norm) the degree-N space misses. A small E means
the branch is nearly a polynomial of degree N, so the discrete system has a near-null vector made of it; E near 1 means
the polynomial space excludes it. The real near-null vector is a centre-regular solution whose horizon component has
its own weight (notes/LARGE_D_BASICS.md 3.4), so this is a model of one factor, not of the Jacobian.

Usage: python3 cheb_branch_tail.py [--N=80] [--mutate=formula]
  The control compares a_k from the formula with a quadrature of (2/pi) int_0^pi (1 - cos t)^sigma cos(k t) dt at
  sigma 7.3 and 41.7; they must agree to 1e-25 of the integrand's maximum 2^sigma. --mutate=formula drops the (-1)^k, which must fail it (exit 1).
"""
import sys

import mpmath as mp

mp.mp.dps = 60


def coef(sigma, k, mutate=False):
    s = mp.mpf(sigma)
    lg = (1 - s) * mp.log(2) + mp.loggamma(2 * s + 1) - mp.loggamma(s + k + 1)
    a = mp.exp(lg) * mp.rgamma(s - k + 1)
    if not mutate:
        a *= (-1) ** k
    return a / 2 if k == 0 else a


def tail_share(sigma, N, kmax):
    head = mp.fsum(abs(coef(sigma, k)) for k in range(N + 1))
    tail = mp.fsum(abs(coef(sigma, k)) for k in range(N + 1, kmax))
    return tail / (head + tail)


def main():
    N = 80
    for a in sys.argv[1:]:
        if a.startswith('--N='):
            N = int(a[4:])
    mutate = '--mutate=formula' in sys.argv
    ok = True
    for s in ('7.3', '41.7'):
        for k in (3, 10, 60):
            s_ = mp.mpf(s)
            q = 2 / mp.pi * mp.quad(lambda t: (1 - mp.cos(t)) ** s_ * mp.cos(k * t), [0, mp.pi / 2, mp.pi])
            f = coef(s_, k, mutate)
            rel = abs(f - q) / 2 ** s_  # error relative to the integrand's maximum, 2^sigma
            print(f"control sigma {s} k {k}: formula {mp.nstr(f, 12)}, quadrature {mp.nstr(q, 12)}, rel {mp.nstr(rel, 3)}")
            ok = ok and rel < 1e-25
    print(f"control {'passed' if ok else 'FAILED'}" + (" (mutated)" if mutate else ""))
    if mutate or not ok:
        sys.exit(1 if not ok else 0)
    print(f"\nN = {N}: share of (1 - xi)^sigma beyond degree N (l1), kmax = 8N")
    print(f"{'sigma':>7} {'E(sigma, N)':>14}")
    for s in (10.5, 20.5, 40.5, 60.5, 70.5, 80.5, 90.5, 100.5, 120.5, 160.5, 200.5, 300.5, 500.5, 1000.5):
        print(f"{s:7.1f} {mp.nstr(tail_share(s, N, 8 * N), 4):>14}")


if __name__ == '__main__':
    main()
